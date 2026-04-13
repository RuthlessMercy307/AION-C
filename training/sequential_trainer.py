"""
training/sequential_trainer.py — Motor-Sequential Training para MoSE.

Técnica única de la arquitectura MoSE: entrenar el modelo por FASES,
donde en cada fase sólo una parte del pipeline tiene requires_grad=True
y el resto está congelado. Esto reduce dramáticamente el costo del
backward pass y del optimizer step, permitiendo entrenar localmente
sin cargar 1.1B params al optimizer al mismo tiempo.

Fases canónicas:

    Phase 1 — Backbone pretraining
        Trainable: encoder + decoder + unifier
        Frozen:    motors + orchestrator (orchestrator es trivial, 0.7M)
        Objetivo:  el backbone aprende representaciones generales

    Phase 2 — Motor specialization (una vez por cada motor)
        Trainable: un solo motor
        Frozen:    encoder + decoder + unifier + orchestrator + otros 4 motores
        Objetivo:  el motor aprende su dominio específico

    Phase 3 — Orchestrator routing
        Trainable: orchestrator
        Frozen:    todo lo demás
        Objetivo:  el router aprende qué motor activar por dominio

    Phase 4 — Cross-motor harmonization (LoRA)
        Trainable: solo LoRA adapters sobre las proyecciones del unifier
        Frozen:    todos los base weights
        Objetivo:  adapters que ajustan las interacciones entre motores

Ventajas sobre full training:
    - Backward pass corre solo sobre weights trainable → mucho más rápido
    - Optimizer state ocupa O(trainable_params), no O(total_params)
    - Memoria pico cae a ~2-3 GB por fase en lugar de 25+ GB
    - Cada fase es un checkpoint natural — se puede pausar y reanudar
    - Compatible con Fase F (adapters se añaden en Phase 4)

Limitaciones:
    - Forward pass sigue atravesando todo el pipeline (no hay ahorro ahí)
    - La calidad final depende de que el orden de fases sea sensato
    - El backbone pretrained no tiene "señal de routing" (se arregla en Phase 3)

Referencia académica (propuesta): "Module-Sequential Training for Mixture
of Specialized Engines", en preparación.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.chunked_trainer import resolve_device, device_label


# ════════════════════════════════════════════════════════════════════════════
# Phase constants
# ════════════════════════════════════════════════════════════════════════════

PHASE_1_BACKBONE     = "phase_1_backbone"
PHASE_2_MOTOR        = "phase_2_motor"
PHASE_3_ORCHESTRATOR = "phase_3_orchestrator"
PHASE_4_ADAPTERS     = "phase_4_adapters"


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SequentialConfig:
    device: str = "cpu"
    lr_phase_1: float = 3e-4
    lr_phase_2: float = 1e-4
    lr_phase_3: float = 5e-4
    lr_phase_4: float = 2e-4
    grad_clip: float = 0.5
    log_every: int = 10
    # Routing loss weight for Phase 2 (supervised to the target motor)
    phase_2_routing_weight: float = 0.3
    # Phase 1 includes motors in the forward pass (they're frozen but present)
    phase_1_motors_frozen_random: bool = True

    # ── Memory-saving flags for Phase 1 on small RAM ──────────────────
    # AdamW uses 8 bytes/param of optimizer state (m + v in fp32).
    # SGD with momentum uses 4 bytes/param (only the velocity buffer).
    # For a 953M backbone: AdamW ≈ 7.6 GB state, SGD-M ≈ 3.8 GB state.
    # Use "sgd" for Phase 1 on 16 GB RAM to avoid swap; use "adamw" on
    # bigger RAM for faster convergence.
    phase_1_optimizer: str = "adamw"   # "adamw" | "sgd"
    phase_1_momentum: float = 0.9       # used when optimizer is sgd


# ════════════════════════════════════════════════════════════════════════════
# Freezing helpers
# ════════════════════════════════════════════════════════════════════════════

def freeze_all(pipeline: nn.Module) -> int:
    """Set requires_grad=False on every parameter. Returns count frozen."""
    n = 0
    for p in pipeline.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            n += 1
    return n


def unfreeze_module(module: nn.Module) -> int:
    """Set requires_grad=True on every parameter of a module."""
    n = 0
    for p in module.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)
            n += 1
    return n


def unfreeze_by_path(pipeline: nn.Module, path: str) -> int:
    """Unfreeze a module identified by a dotted path like 'motors.cora'."""
    mod = pipeline
    for part in path.split("."):
        mod = getattr(mod, part)
    return unfreeze_module(mod)


def count_trainable(pipeline: nn.Module) -> Tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = 0
    total = 0
    for p in pipeline.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def build_optimizer(
    pipeline: nn.Module,
    lr: float,
    weight_decay: float = 1e-2,
    kind: str = "adamw",
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Build the optimizer on only the currently trainable params.

    Args:
        kind: "adamw" (default) or "sgd"
              AdamW: 8 bytes/param state, better convergence, more memory.
              SGD-M: 4 bytes/param state, needs good LR schedule, less memory.

    For 16 GB RAM training Phase 1 of a 953M backbone, prefer "sgd" to
    avoid swap thrashing from the AdamW moment buffers.
    """
    trainable_params = [p for p in pipeline.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("build_optimizer called with no trainable params")
    kind = kind.lower()
    if kind == "adamw":
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    if kind == "sgd":
        return torch.optim.SGD(
            trainable_params, lr=lr, momentum=momentum,
            weight_decay=weight_decay, nesterov=True,
        )
    raise ValueError(f"unknown optimizer kind: {kind}")


# ════════════════════════════════════════════════════════════════════════════
# Phase result
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseResult:
    name: str
    trainable_params: int
    total_params: int
    trainable_pct: float
    n_steps: int
    total_seconds: float
    sps: float
    seconds_per_step: float
    mean_loss: float
    final_loss: float
    peak_rss_gb: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ════════════════════════════════════════════════════════════════════════════
# SequentialTrainer
# ════════════════════════════════════════════════════════════════════════════

class SequentialTrainer:
    """Driver de Motor-Sequential Training.

    Uso:
        trainer = SequentialTrainer(pipeline, SequentialConfig(device="cpu"))

        # Fase 1
        result_1 = trainer.run_phase_1_backbone(data_fn, n_steps=2000)

        # Fase 2 por motor
        for motor_name in ["cora", "forge_c", "axiom", "muse", "empathy"]:
            result = trainer.run_phase_2_motor(motor_name, data_fn, n_steps=1500)

        # Fase 3
        trainer.run_phase_3_orchestrator(data_fn, n_steps=500)

        # Fase 4
        trainer.run_phase_4_adapters(data_fn, n_steps=500)

    data_fn es una función `() -> (token_ids, target_motor_idx)` que
    devuelve un batch para el step actual. Puede ser un generador que
    recorre el dataset, un mock para benchmarks, etc.
    """

    def __init__(
        self,
        pipeline: nn.Module,
        config: Optional[SequentialConfig] = None,
        monitoring: Optional[Any] = None,
    ) -> None:
        self.config = config or SequentialConfig()
        self.pipeline = pipeline
        self.device = resolve_device(self.config.device)
        self.device_label = device_label(self.device)
        self.pipeline.to(self.device)
        self.phase_history: List[PhaseResult] = []
        self.monitoring = monitoring  # MonitoringContext or None
        # Flag set by control file "pause" / "stop"
        self._pause_requested = False
        self._stop_requested = False

    # ── Forced-motor orchestrator wrapper ───────────────────────────────
    def _install_forced_motor(self, motor_name: str) -> Callable:
        """Monkey-patches the orchestrator to always activate `motor_name`.

        Returns a callable that restores the original behavior.
        """
        orch = self.pipeline.orchestrator
        original_forward = orch.forward
        motor_idx = list(self.pipeline.motors.keys()).index(motor_name)

        from orchestrator.model import MotorActivation, OrchestratorOutput

        def forced_forward(concepts, query_text=None):
            # Still compute the real logits so that routing loss can flow.
            real_out = original_forward(concepts, query_text)
            activation = MotorActivation(
                motor_name=motor_name,
                score=1.0,
                n_iterations=3,
                rank=1,
                motor_idx=motor_idx,
            )
            return OrchestratorOutput(
                activations=[activation],
                scores=real_out.scores,
                logits=real_out.logits,
                routing_mode="forced_phase_2",
                n_active=1,
            )

        orch.forward = forced_forward

        def restore():
            orch.forward = original_forward

        return restore

    # ── Common training loop ────────────────────────────────────────────
    def _train_loop(
        self,
        phase_name: str,
        n_steps: int,
        optimizer: torch.optim.Optimizer,
        data_fn: Callable[[], Tuple[torch.Tensor, int]],
        extra_loss_fn: Optional[Callable[[Any, int], torch.Tensor]] = None,
        skip_lm_loss: bool = False,
        log_every: Optional[int] = None,
    ) -> PhaseResult:
        """Training loop común a todas las fases.

        Args:
            phase_name: etiqueta del phase result
            n_steps: cantidad de steps a correr
            optimizer: ya construido con los params trainables correctos
            data_fn: función que produce (token_ids, target_motor_idx)
            extra_loss_fn: función opcional que añade una loss adicional
                           (usado en Phase 3 para routing loss)
        """
        log_every = log_every or self.config.log_every
        trainable, total = count_trainable(self.pipeline)
        print(f"  trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        losses: List[float] = []
        peak_rss_gb = 0.0
        try:
            import psutil
            proc = psutil.Process()
        except Exception:
            proc = None

        self.pipeline.train()
        t_start = time.perf_counter()
        mon = self.monitoring
        poll_every = mon.poll_every if mon is not None else 10
        mon_log_every = mon.log_every if mon is not None else 25

        for step in range(1, n_steps + 1):
            # ── Control file poll (pause/stop/adjust_lr) ──────────────
            if mon is not None and step % poll_every == 0:
                action = mon.poll_control()
                if action is not None:
                    act = action.get("action", "")
                    author = action.get("author", "?")
                    mon.write_note(f"action={act} from {author}", author="training")
                    print(f"    [control] received '{act}' from {author}")
                    if act == "pause":
                        self._pause_requested = True
                        print(f"    [control] PAUSE requested, breaking loop")
                        break
                    elif act == "stop":
                        self._stop_requested = True
                        print(f"    [control] STOP requested, breaking loop")
                        break
                    elif act == "adjust_lr":
                        try:
                            new_lr = float(action.get("value", 0))
                            if new_lr > 0:
                                for pg in optimizer.param_groups:
                                    pg["lr"] = new_lr
                                print(f"    [control] LR adjusted to {new_lr}")
                                mon.write_note(f"lr adjusted to {new_lr}", author="training")
                        except (TypeError, ValueError):
                            pass
                    elif act == "note":
                        mon.write_note(f"note: {action.get('message', '')}", author=author)

            token_ids, target_motor_idx = data_fn()
            token_ids = token_ids.to(self.device)

            optimizer.zero_grad()
            out = self.pipeline(token_ids)
            logits = out.logits if hasattr(out, "logits") else out

            if skip_lm_loss:
                loss = None
            else:
                loss = F.cross_entropy(
                    logits[0, :-1],
                    token_ids[0, 1:],
                    ignore_index=0,
                )
            if extra_loss_fn is not None:
                extra = extra_loss_fn(out, target_motor_idx)
                if extra is not None:
                    loss = extra if loss is None else loss + extra
            if loss is None:
                loss = torch.zeros((), requires_grad=True, device=token_ids.device)

            loss.backward()
            nn.utils.clip_grad_norm_(
                (p for p in self.pipeline.parameters() if p.requires_grad),
                self.config.grad_clip,
            )
            optimizer.step()

            losses.append(float(loss.item()))

            # ── Monitoring: log_every writes to metrics.jsonl ─────────
            should_log_to_console = (step % log_every == 0 or step == 1)
            should_log_to_monitoring = (mon is not None and
                                         (step % mon_log_every == 0 or step == 1))

            if should_log_to_console or should_log_to_monitoring:
                elapsed = time.perf_counter() - t_start
                sps_so_far = step / max(elapsed, 1e-6)
                rss = proc.memory_info().rss / (1024 ** 3) if proc else None
                if rss is not None:
                    peak_rss_gb = max(peak_rss_gb, rss)

                if should_log_to_console:
                    rss_str = f"rss={rss:.2f}GB" if rss is not None else ""
                    print(f"    step {step}/{n_steps}: loss={loss.item():.3f} "
                          f"sps={sps_so_far:.3f} {rss_str}")

                if should_log_to_monitoring:
                    current_lr = optimizer.param_groups[0]["lr"]
                    eta_min = (n_steps - step) / max(sps_so_far, 1e-6) / 60
                    mon.log_step(
                        step=step,
                        phase=phase_name,
                        loss=float(loss.item()),
                        sps=sps_so_far,
                        lr=current_lr,
                        elapsed_sec=elapsed,
                        eta_min=eta_min,
                    )

        t_end = time.perf_counter()
        total_sec = t_end - t_start
        sps = n_steps / max(total_sec, 1e-6)

        result = PhaseResult(
            name=phase_name,
            trainable_params=trainable,
            total_params=total,
            trainable_pct=round(100 * trainable / total, 2),
            n_steps=n_steps,
            total_seconds=round(total_sec, 2),
            sps=round(sps, 4),
            seconds_per_step=round(total_sec / max(n_steps, 1), 3),
            mean_loss=round(sum(losses) / max(len(losses), 1), 4),
            final_loss=round(losses[-1], 4) if losses else 0.0,
            peak_rss_gb=round(peak_rss_gb, 2) if peak_rss_gb else None,
        )
        self.phase_history.append(result)
        return result

    # ── Phase 1: backbone ────────────────────────────────────────────────
    def run_phase_1_backbone(
        self,
        data_fn: Callable[[], Tuple[torch.Tensor, int]],
        n_steps: int,
    ) -> PhaseResult:
        print()
        print(f"[{PHASE_1_BACKBONE}] training encoder + decoder + unifier")
        print(f"  frozen: motors + orchestrator")

        freeze_all(self.pipeline)
        unfreeze_module(self.pipeline.encoder)
        unfreeze_module(self.pipeline.decoder)
        unfreeze_module(self.pipeline.unifier)

        print(f"  optimizer: {self.config.phase_1_optimizer}")
        optimizer = build_optimizer(
            self.pipeline,
            lr=self.config.lr_phase_1,
            kind=self.config.phase_1_optimizer,
            momentum=self.config.phase_1_momentum,
        )
        return self._train_loop(
            phase_name=PHASE_1_BACKBONE,
            n_steps=n_steps,
            optimizer=optimizer,
            data_fn=data_fn,
        )

    # ── Phase 2: per-motor ───────────────────────────────────────────────
    def run_phase_2_motor(
        self,
        motor_name: str,
        data_fn: Callable[[], Tuple[torch.Tensor, int]],
        n_steps: int,
    ) -> PhaseResult:
        print()
        print(f"[{PHASE_2_MOTOR}:{motor_name}] training only this motor")
        print(f"  frozen: backbone + other motors")
        print(f"  orchestrator temporarily FORCED to always pick {motor_name}")

        freeze_all(self.pipeline)
        unfreeze_by_path(self.pipeline, f"motors.{motor_name}")

        optimizer = build_optimizer(self.pipeline, self.config.lr_phase_2)

        # Install the forced-motor patch so the motor's params participate
        # in every forward pass. Without this, the orchestrator might pick
        # different motors per step and the target motor's grads are empty.
        restore = self._install_forced_motor(motor_name)

        try:
            return self._train_loop(
                phase_name=f"{PHASE_2_MOTOR}:{motor_name}",
                n_steps=n_steps,
                optimizer=optimizer,
                data_fn=data_fn,
                extra_loss_fn=None,
            )
        finally:
            restore()

    # ── Phase 3: orchestrator ────────────────────────────────────────────
    def run_phase_3_orchestrator(
        self,
        data_fn: Callable[[], Tuple[torch.Tensor, int]],
        n_steps: int,
    ) -> PhaseResult:
        print()
        print(f"[{PHASE_3_ORCHESTRATOR}] training routing only")
        print(f"  frozen: encoder + decoder + unifier + all motors")

        freeze_all(self.pipeline)
        unfreeze_module(self.pipeline.orchestrator)

        optimizer = build_optimizer(self.pipeline, self.config.lr_phase_3)

        def _routing_loss(out, target_motor_idx):
            if not hasattr(out, "orchestrator") or out.orchestrator is None:
                return None
            try:
                logits = out.orchestrator.logits
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                target = torch.tensor([target_motor_idx], dtype=torch.long, device=logits.device)
                return F.cross_entropy(logits, target)
            except Exception:
                return None

        return self._train_loop(
            phase_name=PHASE_3_ORCHESTRATOR,
            n_steps=n_steps,
            optimizer=optimizer,
            data_fn=data_fn,
            extra_loss_fn=_routing_loss,
            skip_lm_loss=True,  # LM loss has no grad path to orchestrator
        )

    # ── Phase 4: LoRA adapters ───────────────────────────────────────────
    def run_phase_4_adapters(
        self,
        data_fn: Callable[[], Tuple[torch.Tensor, int]],
        n_steps: int,
        rank: int = 8,
    ) -> PhaseResult:
        print()
        print(f"[{PHASE_4_ADAPTERS}] training LoRA adapters only")
        print(f"  frozen: all base weights")

        from growth.adapters import (
            LoRAConfig, build_adapter_pack, attach_adapter_pack,
            auto_target_paths,
        )

        # Freeze everything
        freeze_all(self.pipeline)

        # Attach LoRA packs on each motor's top projections
        packs = []
        for motor_name, motor in self.pipeline.motors.items():
            targets = auto_target_paths(motor, max_targets=4)
            if not targets:
                continue
            pack = build_adapter_pack(
                motor, targets, LoRAConfig(rank=rank, alpha=16),
                concept_name="harmonization",
                motor_name=motor_name,
            )
            attach_adapter_pack(motor, pack)
            packs.append(pack)

        # Adapters are trainable by default (their params have requires_grad=True
        # from the LoRALinear init, since only base is frozen).
        # Move adapter params to device
        for pack in packs:
            pack.to(self.device)

        # Build optimizer on currently trainable params (= only adapter params)
        optimizer = build_optimizer(self.pipeline, self.config.lr_phase_4)

        result = self._train_loop(
            phase_name=PHASE_4_ADAPTERS,
            n_steps=n_steps,
            optimizer=optimizer,
            data_fn=data_fn,
        )
        result.notes = f"{len(packs)} LoRA packs attached"
        return result

    # ── Checkpoint helpers ───────────────────────────────────────────────
    def save_checkpoint(self, path: Path, phase: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.pipeline.state_dict(),
            "phase": phase,
            "history": [r.to_dict() for r in self.phase_history],
        }, str(path))
        print(f"  checkpoint saved: {path} (after {phase})")

    def load_checkpoint(self, path: Path) -> str:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        self.pipeline.load_state_dict(ck["model_state"], strict=False)
        self.pipeline.to(self.device)
        return ck.get("phase", "?")

    # ── Introspection ────────────────────────────────────────────────────
    def total_params(self) -> int:
        return sum(p.numel() for p in self.pipeline.parameters())
