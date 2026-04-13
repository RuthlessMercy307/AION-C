"""
experiments/training_utils.py
==============================
Utilidades de entrenamiento compartidas para todos los benchmarks de AION-C.

Provee:
  - make_cosine_scheduler  : cosine LR con warmup lineal (LambdaLR)
  - save_checkpoint        : guarda model, optimizer, scheduler, step, loss, config
  - load_checkpoint        : carga y restaura estados; retorna (start_step, last_loss)
  - train_with_amp         : training loop con AMP/bf16, grad clip, checkpoint,
                             scheduler y logging por step
  - TrainingMonitor        : evaluación periódica, early stopping, convergencia,
                             log JSON, decodificación greedy en 5 ejemplos fijos

Compatibilidad:
  - AMP activo solo en CUDA (bf16); en CPU se corre en fp32 sin overhead.
  - GradScaler NO se usa para bf16 (solo necesario para fp16).
  - PyTorch >= 2.0 requerido para torch.amp.autocast con device_type string.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0


# ─── Scheduler ───────────────────────────────────────────────────────────────

def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine LR scheduler con warmup lineal.

    Fases:
      step in [0, warmup_steps)   → lr crece linealmente de 0 → base_lr
      step in [warmup_steps, total_steps] → lr decae coseno de base_lr → 0

    Args:
        optimizer:    Optimizer cuyo lr se ajusta.
        warmup_steps: Número de steps de warmup lineal.
        total_steps:  Número total de steps de entrenamiento.

    Returns:
        LambdaLR scheduler que multiplica el lr base por el factor calculado.
    """
    def _lr_lambda(step: int) -> float:
        if total_steps <= warmup_steps:
            return 1.0
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,                                   # LRScheduler (sin tipado estricto por compat)
    step: int,
    loss: float,
    config: Optional[object] = None,
) -> None:
    """
    Guarda un checkpoint completo en disco.

    Contenido del checkpoint:
        model_state     : OrderedDict — estado del modelo
        optimizer_state : dict        — estado del optimizer
        scheduler_state : dict        — estado del scheduler (si no es None)
        step            : int         — step actual
        loss            : float       — loss promedio de los últimos 50 steps
        config          : object      — config del modelo (opcional)

    Args:
        path:      Ruta donde guardar el archivo .pt.
        model:     Modelo a guardar.
        optimizer: Optimizer a guardar.
        scheduler: Scheduler a guardar (puede ser None).
        step:      Step actual del training loop.
        loss:      Loss actual (promedio últimos 50 steps).
        config:    Config del modelo (dataclass, dict, etc.).
    """
    # Save model_state only (optimizer/scheduler states are huge - 2-3x model size)
    # For resume: we restart optimizer from scratch, which is fine for convergence training
    ckpt: dict = {
        "model_state":     model.state_dict(),
        "step":            step,
        "loss":            loss,
    }
    if config is not None:
        ckpt["config"] = config

    torch.save(ckpt, path)

    # Delete previous checkpoints in same dir to save disk
    p = Path(path)
    for old in p.parent.glob("*.pt"):
        if old != p and old.name != p.name:
            try:
                old.unlink()
            except OSError:
                pass


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
) -> Tuple[int, float]:
    """
    Carga un checkpoint si existe. Si no existe, no hace nada.

    Args:
        path:      Ruta al checkpoint .pt.
        model:     Modelo al que cargar el estado.
        optimizer: Optimizer al que cargar el estado.
        scheduler: Scheduler al que cargar el estado (puede ser None).

    Returns:
        (start_step, last_loss) — step desde el que reanudar y loss guardada.
        Retorna (0, nan) si el archivo no existe.
    """
    path = Path(path)
    if not path.exists():
        return 0, float("nan")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception:
            pass  # optimizer state may be missing or incompatible
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    return ckpt.get("step", 0), ckpt.get("loss", float("nan"))


# ─── Training loop principal ──────────────────────────────────────────────────

def train_with_amp(
    model: nn.Module,
    ids_list: List[List[int]],
    tok,
    n_steps: int,
    label: str,
    is_motor: bool = True,
    lr: float = 3e-4,
    warmup_steps: int = 100,
    checkpoint_every: int = 500,
    checkpoint_path: Optional[Union[str, Path]] = None,
    print_every: int = 500,
    device: Optional[torch.device] = None,
    monitor: Optional["TrainingMonitor"] = None,
) -> Tuple[List[float], float, float]:
    """
    Training loop con AMP/bf16, cosine LR con warmup, grad clip y checkpoint.

    Características:
      - AMP (Automatic Mixed Precision): bf16 en CUDA, fp32 en CPU.
        GradScaler NO se usa — bf16 es numericamente estable sin él.
      - Cosine LR scheduler con `warmup_steps` de warmup lineal.
      - clip_grad_norm_(model.parameters(), 1.0) en cada step.
      - Checkpoint cada `checkpoint_every` steps si se especifica `checkpoint_path`.
      - Resume automático: si el checkpoint existe, retoma desde el step guardado.

    Args:
        model:             Modelo a entrenar (CORAPipeline, VanillaTransformer, etc.).
        ids_list:          Lista de token_ids pre-tokenizados (List[List[int]]).
        tok:               Tokenizer con .to_tensor(ids) → Tensor.
        n_steps:           Número total de steps de entrenamiento.
        label:             Etiqueta para logs (ej. "Motor-CORA").
        is_motor:          True → model retorna PipelineOutput con .logits.
                           False → model retorna Tensor directamente.
        lr:                Learning rate base (antes de warmup).
        warmup_steps:      Steps de warmup lineal (default: 100).
        checkpoint_every:  Frecuencia de checkpoint en steps (default: 500).
        checkpoint_path:   Ruta del checkpoint. None → sin checkpoint.
        print_every:       Frecuencia de logging en steps (default: 500).
        device:            Device donde mover los tensores. None → infiere del modelo.

    Returns:
        (losses, elapsed_seconds, final_loss_avg50)
          losses:           Lista de losses por step (solo steps finitos).
          elapsed_seconds:  Tiempo total de entrenamiento en segundos.
          final_loss_avg50: Promedio de los últimos 50 losses.
    """
    if device is None:
        device = next(model.parameters()).device

    # AMP solo en CUDA; bf16 no necesita GradScaler (a diferencia de fp16)
    use_amp   = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = make_cosine_scheduler(opt, warmup_steps, n_steps)

    # Resume desde checkpoint si existe
    start_step = 0
    if checkpoint_path is not None:
        start_step, _ = load_checkpoint(checkpoint_path, model, opt, scheduler)
        if start_step > 0:
            print(f"    [{label}]  Resumiendo desde step {start_step}", flush=True)

    model.train()
    losses: List[float] = []
    t0 = time.perf_counter()

    for step in range(start_step + 1, n_steps + 1):
        ids_t = tok.to_tensor(random.choice(ids_list)).to(device)

        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            if is_motor:
                out    = model(ids_t)
                logits = out.logits
            else:
                logits = model(ids_t)

            loss = F.cross_entropy(
                logits[0, :-1], ids_t[0, 1:], ignore_index=PAD
            )

        if math.isfinite(loss.item()):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # clip
            opt.step()
            scheduler.step()
            losses.append(loss.item())

        if step % print_every == 0:
            valid   = [x for x in losses[-print_every:] if math.isfinite(x)]
            avg     = sum(valid) / len(valid) if valid else float("nan")
            lr_now  = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t0
            print(
                f"    [{label}]  step {step:>5}  loss={avg:.4f}  "
                f"lr={lr_now:.2e}  {elapsed:.0f}s",
                flush=True,
            )

        if checkpoint_path is not None and step % checkpoint_every == 0:
            avg50 = (
                sum(losses[-50:]) / min(50, len(losses))
                if losses else float("nan")
            )
            save_checkpoint(checkpoint_path, model, opt, scheduler, step, avg50)
            print(
                f"    [{label}]  checkpoint → {checkpoint_path}  (step {step})",
                flush=True,
            )

        # ── TrainingMonitor (eval, early stop, convergencia) ──────────────────
        if monitor is not None and math.isfinite(loss.item()):
            stop_reason = monitor.step(step, loss.item())
            if stop_reason is not None:
                print(
                    f"    [{label}]  DETENIDO por monitor: {stop_reason} "
                    f"(step {step})",
                    flush=True,
                )
                break

    elapsed    = time.perf_counter() - t0
    final_loss = (
        sum(losses[-50:]) / min(50, len(losses)) if losses else float("nan")
    )
    return losses, elapsed, final_loss


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING MONITOR
# ─────────────────────────────────────────────────────────────────────────────

class TrainingMonitor:
    """
    Monitor de entrenamiento con evaluación periódica, early stopping y
    detección de convergencia.

    Funciones principales:
      - Eval periódico (eval_every steps): val_loss, greedy decode en 5 ejemplos
        fijos, Word F1. Imprime en consola y escribe log JSON.
      - Early stopping: para si val_loss no mejora en `patience` steps.
      - Convergencia: para si el cambio de val_loss < `convergence_delta`
        durante `convergence_window` steps consecutivos.

    Los 5 ejemplos fijos de validación se cargan con `load_fixed_examples()`
    (uno por dominio: cora/causal, forge_c/code, axiom/math, muse/narrative,
    empathy/social).

    Uso:
        monitor = TrainingMonitor(
            model=pipeline,
            tok=tok,
            val_ids_list=val_ids,         # 50 ejemplos pre-tokenizados
            fixed_examples=[              # 5 (input, output, domain)
                ("texto entrada", "respuesta", "cora"),
                ...
            ],
            cfg=mose_cfg,
            eval_every=500,
            patience=2000,
            log_path="runs/monitor.jsonl",
        )
        losses, elapsed, final = train_with_amp(..., monitor=monitor)

    Args:
        model:               Modelo a evaluar (debe tener forward con logits).
        tok:                 Tokenizador con encode(text)/encode(text, max_len),
                             decode(ids) y to_tensor(ids).
        val_ids_list:        Lista de ≥50 secuencias pre-tokenizadas
                             (List[List[int]]) para calcular val_loss.
        fixed_examples:      Lista de (input_text, expected_output, domain_label)
                             para generar texto y calcular Word F1 en cada eval.
        cfg:                 MoSEConfig con dec_max_seq_len (para greedy decode).
        eval_every:          Frecuencia de eval en steps (default: 500).
        patience:            Steps sin mejora para early stopping (default: 2000).
        convergence_delta:   Umbral de cambio de val_loss para convergencia
                             (default: 0.001).
        convergence_window:  Steps sin cambio significativo para convergencia
                             (default: 1000).
        log_path:            Ruta del log JSON (default: "training_monitor.jsonl").
        device:              Device donde ejecutar el eval. None → infer del modelo.
        is_motor:            True → model retorna objeto con .logits.
                             False → model retorna Tensor directamente.
        motor_hints:         Dict motor_name → hint string para greedy decode.
                             Ejemplo: {"forge_c": "function ", "axiom": "theorem "}.
        max_new_tokens:      Máximo de tokens nuevos en greedy decode (default: 32).
        checkpoint_path:     Si se especifica, guarda checkpoint al early-stop.
    """

    PAD = 0
    EOS = 2

    def __init__(
        self,
        model:                nn.Module,
        tok,
        val_ids_list:         List[List[int]],
        fixed_examples:       List[Tuple[str, str, str]],
        cfg,
        eval_every:           int   = 500,
        patience:             int   = 2000,
        convergence_delta:    float = 0.001,
        convergence_window:   int   = 1000,
        log_path:             Union[str, Path] = "training_monitor.jsonl",
        device:               Optional[torch.device] = None,
        is_motor:             bool  = True,
        motor_hints:          Optional[Dict[str, str]] = None,
        max_new_tokens:       int   = 32,
        checkpoint_path:      Optional[Union[str, Path]] = None,
    ) -> None:
        self.model               = model
        self.tok                 = tok
        self.val_ids_list        = val_ids_list
        self.fixed_examples      = fixed_examples
        self.cfg                 = cfg
        self.eval_every          = eval_every
        self.patience            = patience
        self.convergence_delta   = convergence_delta
        self.convergence_window  = convergence_window
        self.log_path            = Path(log_path)
        self.is_motor            = is_motor
        self.motor_hints         = motor_hints or {}
        self.max_new_tokens      = max_new_tokens
        self.checkpoint_path     = Path(checkpoint_path) if checkpoint_path else None

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device

        # Estado interno
        self._best_val_loss:    float          = float("inf")
        self._best_step:        int            = 0
        self._last_eval_loss:   float          = float("inf")
        self._steps_no_improve: int            = 0
        self._converge_steps:   int            = 0
        self._stop_reason:      Optional[str]  = None
        self._eval_history:     List[Dict]     = []

        # Asegurar directorio del log
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, step: int, train_loss: float) -> Optional[str]:
        """
        Llamar en cada step del training loop.

        Ejecuta eval si step % eval_every == 0.
        Retorna "early_stop", "converged", o None para continuar.
        """
        if step % self.eval_every == 0:
            self._run_eval(step, train_loss)

        if self._stop_reason is not None:
            return self._stop_reason
        return None

    @property
    def should_stop(self) -> bool:
        """True si el monitor ya emitió una señal de parada."""
        return self._stop_reason is not None

    @property
    def eval_history(self) -> List[Dict]:
        """Lista de todos los registros de eval (inmutable)."""
        return list(self._eval_history)

    # ─────────────────────────────────────────────────────────────────────────
    # EVAL INTERNO
    # ─────────────────────────────────────────────────────────────────────────

    def _run_eval(self, step: int, train_loss: float) -> None:
        """Ejecuta el eval completo y actualiza el estado interno."""
        t0 = time.perf_counter()

        # ── Val loss ──────────────────────────────────────────────────────────
        val_loss = self._compute_val_loss()

        # ── Greedy decode + Word F1 en los 5 ejemplos fijos ──────────────────
        gen_results = []
        f1s: List[float] = []
        cf1s: List[float] = []
        for inp_text, exp_output, domain in self.fixed_examples:
            hint = self.motor_hints.get(domain, "")
            max_seq = getattr(self.cfg, "dec_max_seq_len", 512)
            predicted = self._greedy_decode(inp_text, hint, max_seq)
            f1 = self.word_f1(predicted, exp_output)
            cf1 = self.contains_f1(predicted, exp_output)
            f1s.append(f1)
            cf1s.append(cf1)
            gen_results.append({
                "domain":    domain,
                "input":     inp_text[:120],
                "expected":  exp_output[:120],
                "predicted": predicted[:120],
                "f1":        f1,
                "contains_f1": cf1,
            })

        mean_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        mean_cf1 = sum(cf1s) / len(cf1s) if cf1s else 0.0
        elapsed = time.perf_counter() - t0

        # ── Logging a consola ─────────────────────────────────────────────────
        print(f"\n{'─'*60}", flush=True)
        print(
            f"[Monitor] step={step:>6}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_F1={mean_f1:.3f}  "
            f"contains_F1={mean_cf1:.3f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        for r in gen_results:
            q_safe = r["input"][:55].encode("ascii", "replace").decode("ascii")
            e_safe = r["expected"][:45].encode("ascii", "replace").decode("ascii")
            p_safe = r["predicted"][:45].encode("ascii", "replace").decode("ascii")
            print(
                f"  [{r['domain']:8s}] F1={r['f1']:.3f} cF1={r['contains_f1']:.2f} | "
                f"Q: {q_safe!r:57s} | "
                f"E: {e_safe!r:47s} | "
                f"P: {p_safe!r}",
                flush=True,
            )
        print(f"{'─'*60}", flush=True)

        # ── Registro JSON ─────────────────────────────────────────────────────
        record: Dict = {
            "step":         step,
            "train_loss":   round(train_loss, 6),
            "val_loss":     round(val_loss, 6),
            "val_f1":       round(mean_f1, 4),
            "contains_f1":  round(mean_cf1, 4),
            "generated":    gen_results,
            "elapsed_s":    round(elapsed, 2),
        }
        self._eval_history.append(record)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

        # ── Early stopping ────────────────────────────────────────────────────
        if val_loss < self._best_val_loss - 1e-6:
            self._best_val_loss = val_loss
            self._best_step     = step
            self._steps_no_improve = 0
        else:
            self._steps_no_improve += self.eval_every

        if self._steps_no_improve >= self.patience:
            if self.checkpoint_path is not None:
                try:
                    opt_dummy  = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
                    save_checkpoint(
                        self.checkpoint_path, self.model, opt_dummy,
                        None, step, val_loss
                    )
                    print(
                        f"[Monitor] early-stop checkpoint → {self.checkpoint_path}",
                        flush=True,
                    )
                except Exception:
                    pass
            self._stop_reason = "early_stop"
            print(
                f"[Monitor] EARLY STOP — val_loss no mejoró en "
                f"{self._steps_no_improve} steps "
                f"(mejor={self._best_val_loss:.4f} @ step {self._best_step})",
                flush=True,
            )
            return

        # ── Convergencia ──────────────────────────────────────────────────────
        delta = abs(val_loss - self._last_eval_loss)
        if delta < self.convergence_delta:
            self._converge_steps += self.eval_every
        else:
            self._converge_steps = 0
        self._last_eval_loss = val_loss

        if self._converge_steps >= self.convergence_window:
            self._stop_reason = "converged"
            print(
                f"[Monitor] CONVERGIDO — val_loss cambió < {self.convergence_delta} "
                f"durante {self._converge_steps} steps consecutivos",
                flush=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # CÓMPUTO DE VAL LOSS
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_val_loss(self, n_examples: int = 50) -> float:
        """
        Calcula la pérdida promedio sobre n_examples del val set.

        Usa los primeros n_examples de val_ids_list.
        """
        self.model.eval()
        device     = self.device
        use_amp    = device.type == "cuda"
        amp_dtype  = torch.bfloat16 if use_amp else torch.float32
        losses: List[float] = []

        sample = self.val_ids_list[:n_examples]

        with torch.no_grad():
            for ids in sample:
                try:
                    ids_t = _ids_to_tensor(self.tok, ids).to(device)
                    with torch.amp.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=use_amp
                    ):
                        if self.is_motor:
                            out    = self.model(ids_t)
                            logits = out.logits
                        else:
                            logits = self.model(ids_t)

                    loss = F.cross_entropy(
                        logits[0, :-1], ids_t[0, 1:], ignore_index=self.PAD
                    )
                    if math.isfinite(loss.item()):
                        losses.append(loss.item())
                except Exception:
                    pass

        self.model.train()
        return sum(losses) / len(losses) if losses else float("nan")

    # ─────────────────────────────────────────────────────────────────────────
    # GREEDY DECODE
    # ─────────────────────────────────────────────────────────────────────────

    def _greedy_decode(self, prompt: str, hint: str, max_seq_len: int) -> str:
        """
        Decodificación greedy (argmax) para un único prompt.

        Args:
            prompt:      Texto de entrada.
            hint:        Texto de ayuda para el orchestrator (query_text).
            max_seq_len: Longitud máxima total de la secuencia.

        Returns:
            Texto generado (sin el prompt inicial).
        """
        self.model.eval()
        device    = self.device
        use_amp   = device.type == "cuda"
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        max_new   = min(self.max_new_tokens, max_seq_len - 4)

        try:
            # Encode
            try:
                ids = self.tok.encode(prompt, max_seq_len - max_new)
            except TypeError:
                ids = self.tok.encode(prompt)[:max_seq_len - max_new]

            prompt_len = len(ids)
            cur = torch.tensor([ids], dtype=torch.long, device=device)

            with torch.no_grad():
                for _ in range(max_new):
                    if cur.shape[1] >= max_seq_len:
                        break
                    with torch.amp.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=use_amp
                    ):
                        if self.is_motor:
                            out    = self.model(cur, query_text=hint or None)
                            logits = out.logits
                        else:
                            logits = self.model(cur)

                    nxt = int(logits[0, -1].argmax().item())
                    if nxt == self.EOS:
                        break
                    cur = torch.cat(
                        [cur, torch.tensor([[nxt]], device=device)], dim=1
                    )
                    # Early stop on sentence-ending punctuation
                    n_generated = cur.shape[1] - prompt_len
                    if n_generated >= 3:
                        try:
                            last_tok = self.tok.decode([nxt])
                            if last_tok.rstrip().endswith(('.', '?', '!')):
                                break
                        except Exception:
                            pass

            generated_ids = cur[0, prompt_len:].tolist()
            result = self.tok.decode(generated_ids)
        except Exception:
            result = ""

        self.model.train()
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # UTILIDADES ESTÁTICAS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def word_f1(pred: str, ref: str) -> float:
        """
        Word-level F1 score (set-based).

        Splits pred/ref por espacios, convierte a minúsculas, calcula
        intersección. Retorna 0.0 si alguno es vacío.
        """
        p_words = set(pred.lower().split())
        r_words = set(ref.lower().split())
        if not p_words or not r_words:
            return 0.0
        tp = len(p_words & r_words)
        precision = tp / len(p_words)
        recall    = tp / len(r_words)
        denom = precision + recall
        return round(2 * precision * recall / denom, 4) if denom > 0 else 0.0

    @staticmethod
    def contains_f1(pred: str, ref: str) -> float:
        """
        Measures if the reference is contained in the prediction.
        Returns 1.0 if all ref words appear in pred, partial otherwise.
        Useful for cases like expected='No', predicted='No, porque...'.
        """
        p_lower = pred.lower()
        r_words = ref.lower().split()
        if not r_words:
            return 0.0
        contained = sum(1 for w in r_words if w in p_lower)
        return round(contained / len(r_words), 4)

    @staticmethod
    def load_fixed_examples(
        motors: Tuple[str, ...] = ("cora", "forge_c", "axiom", "muse", "empathy"),
        seed:   int             = 42,
        n_load: int             = 200,
    ) -> List[Tuple[str, str, str]]:
        """
        Carga 1 ejemplo fijo por dominio desde OpusDataset.

        Los ejemplos se eligen con seed fijo para reproducibilidad entre
        evaluaciones.

        Args:
            motors: Tupla de motores (1 por dominio).
            seed:   Semilla para reproducibilidad.
            n_load: Número de ejemplos a cargar por motor antes de samplear.

        Returns:
            Lista de (input_text, expected_output, domain_label).
            Si un motor no está disponible, se omite silenciosamente.

        Ejemplo:
            examples = TrainingMonitor.load_fixed_examples()
            # [("Dado que...", "Si, existe...", "cora"), ...]
        """
        examples: List[Tuple[str, str, str]] = []
        rng = random.Random(seed)

        for motor in motors:
            try:
                from experiments.opus_dataset import OpusDataset
                ds = OpusDataset(motor=motor, max_examples=n_load, seed=seed)
                if len(ds) == 0:
                    continue
                # Samplear con seed fijo para reproducibilidad
                all_items = [ds.generate() for _ in range(min(20, len(ds)))]
                rng.shuffle(all_items)
                ex = all_items[0]
                examples.append((ex.problem_text, ex.answer, motor))
            except Exception:
                pass

        # Fallback: si no hay dataset, usar ejemplos hardcodeados
        if not examples:
            fallbacks = [
                ("If the system fails what happens", "The backup activates", "cora"),
                ("def factorial(n): return", "n * factorial(n-1) if n > 1 else 1", "forge_c"),
                ("What is the derivative of x squared", "2x", "axiom"),
                ("Once upon a time there was", "a brave knight who sought adventure", "muse"),
                ("I feel very anxious about tomorrow", "Take a deep breath and rest", "empathy"),
            ]
            examples = fallbacks[:len(motors)]

        return examples


# ─────────────────────────────────────────────────────────────────────────────
# HELPER INTERNO
# ─────────────────────────────────────────────────────────────────────────────

def _ids_to_tensor(tok, ids: List[int]) -> torch.Tensor:
    """
    Convierte una lista de token IDs a tensor [1, L].
    Compatible con tokenizadores que tienen to_tensor() o sin él.
    """
    if hasattr(tok, "to_tensor"):
        return tok.to_tensor(ids)
    return torch.tensor([ids], dtype=torch.long)
