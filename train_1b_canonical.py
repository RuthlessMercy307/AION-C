#!/usr/bin/env python3
"""
train_1b_canonical.py — Fase E: training del 1.1B con dataset canónico 70K
=============================================================================

Implementa los criterios del MEGA-PROMPT (Parte 1 + Fase E):

  - fp16 mixed precision (autocast + GradScaler)
  - Routing supervisado (cross-entropy contra dominio del record)
  - Load balancing loss (penaliza overuse de motores)
  - LM loss (cross-entropy autoregressive con ignore_index=0)
  - Cosine LR con warmup=300
  - Eval cada 200 steps con generation_quality_score (50 prompts canónicos)
  - Save best por gen_quality.combined (NO val_loss)
  - Early stopping patience=500 steps sin mejora
  - BrainVersionManager guardando como brain/v* con metadata
  - Resume capability (start_step + optimizer state opcional)
  - WeightedRandomSampler 50/50 para SKILL/MEM balance (Fase B opción c)
  - EOS token append (Bug fix Fase C)

Configs disponibles:
  --config tiny  → 5.5M params (para dry-run y debug local)
  --config 1b    → 1.1B params (producción Vast)

Uso típico:
  Local dry-run:
    python train_1b_canonical.py --config tiny --steps 10 --dry-run

  Vast.ai (RTX 4090):
    python train_1b_canonical.py --config 1b --steps 15000

  Resume tras interrupción:
    python train_1b_canonical.py --config 1b --resume checkpoints/aion_1b_canonical.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Force UTF-8 stdout on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

from synth.canonical_dataloader import (
    load_canonical_records, encode_record, weighted_sampler_indices,
    domain_to_motor_idx, MOTOR_NAMES, EOS_TOKEN_ID, quick_stats,
)
from evaluation.eval_prompts import EVAL_PROMPTS
from evaluation.metrics import generation_quality_score, GenerationQualityResult
from brain.version_manager import BrainVersionManager

# ── Fase F: integración cognitiva desde step 0 ─────────────────────────
from sparse import (
    SparseConfig, SparseLinear, SparsityTracker,
    attach_sparse_gates, sparsity_loss,
)
from growth import (
    LoRAConfig, auto_target_paths, build_adapter_pack,
    attach_adapter_pack, detach_adapter_pack,
)


def attach_fase_f_to_pipeline(
    pipeline,
    sparsity_target: float = 0.5,
    gate_hidden: int = 8,
    max_targets_per_motor: int = 6,
) -> Tuple[Dict[str, Dict[str, "SparseLinear"]], "SparsityTracker"]:
    """Adjunta gates de activación esparsa (Parte 27) a los 5 motores.

    Retorna:
        per_motor_sparse: dict motor_name → {path → SparseLinear}
        tracker: SparsityTracker sobre el pipeline completo

    Warm-start: el bias del gate fc2 se inicializa de forma que
    sigmoid(bias) ≈ sparsity_target, para que la activación media del
    primer forward sea la deseada y el modelo aprenda A SER ROBUSTO a la
    sparsity desde el primer step (no la aprende en post-entrenamiento).
    """
    cfg = SparseConfig(
        target_density=sparsity_target,
        mode="continuous",
        gate_hidden=gate_hidden,
    )
    per_motor: Dict[str, Dict[str, SparseLinear]] = {}
    for motor_name, motor in pipeline.motors.items():
        targets = auto_target_paths(motor, max_targets=max_targets_per_motor)
        if not targets:
            per_motor[motor_name] = {}
            continue
        per_motor[motor_name] = attach_sparse_gates(motor, targets, cfg)
    tracker = SparsityTracker(pipeline)
    return per_motor, tracker


def verify_adapter_scaffolding(pipeline, device: torch.device) -> Dict[str, Any]:
    """Smoke test: al final del training, un adapter se puede attach+detach
    sobre CADA motor y los pesos base quedan intactos.

    Es la lección del 'identity skill OOD': verificamos en CI que la
    arquitectura cognitiva (adapters + sparse) es invariante bajo
    attach/detach después del training completo.
    """
    results: Dict[str, Any] = {}
    for motor_name, motor in pipeline.motors.items():
        try:
            targets = auto_target_paths(motor, max_targets=3)
            if not targets:
                results[motor_name] = {"ok": True, "note": "no linear targets"}
                continue
            # Snapshot de pesos base
            snaps = {}
            for t in targets:
                mod = motor
                for a in t.split("."):
                    mod = getattr(mod, a)
                # El target está envuelto por SparseLinear → tomamos el .base
                base_linear = mod.base if isinstance(mod, SparseLinear) else mod
                snaps[t] = base_linear.weight.detach().clone()
            # Attach/detach un adapter dummy
            pack = build_adapter_pack(
                motor, targets, LoRAConfig(rank=2, alpha=4),
                "scaffold_test", motor_name,
            )
            attach_adapter_pack(motor, pack)
            for t in targets:
                with torch.no_grad():
                    pack.get(t).lora_B.normal_(0, 0.1)
            detach_adapter_pack(motor, pack)
            # Verify bit-a-bit
            all_equal = True
            for t in targets:
                mod = motor
                for a in t.split("."):
                    mod = getattr(mod, a)
                base_linear = mod.base if isinstance(mod, SparseLinear) else mod
                if not torch.allclose(base_linear.weight, snaps[t], atol=1e-7):
                    all_equal = False
                    break
            results[motor_name] = {"ok": all_equal, "n_targets": len(targets)}
        except Exception as exc:
            results[motor_name] = {"ok": False, "error": str(exc)}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline configs
# ─────────────────────────────────────────────────────────────────────────────


def build_pipeline(config: str, vocab_size: int):
    from router.pipeline import MoSEPipeline, MoSEConfig

    if config == "tiny":
        cfg = MoSEConfig(
            hidden_dim=64, vocab_size=vocab_size,
            enc_n_layers=2, enc_state_dim=4, enc_expand=2, enc_d_conv=4, enc_ffn_mult=2,
            orch_mlp_hidden=32, orch_max_motors=3, orch_min_confidence=0.3,
            motor_max_nodes=8, motor_n_heads=4, motor_threshold=0.01, unif_n_heads=4,
            dec_n_layers=2, dec_n_heads=4, dec_max_seq_len=128,
            dec_state_dim=4, dec_expand=2, dec_d_conv=4, dec_ffn_mult=2,
        )
    elif config == "1b":
        # Config del 1.1B con context 1024 (dec_max_seq_len bumped from 512)
        cfg = MoSEConfig(
            hidden_dim=1024, vocab_size=vocab_size,
            enc_n_layers=12, enc_state_dim=16, enc_expand=2, enc_d_conv=4, enc_ffn_mult=4,
            orch_mlp_hidden=512, orch_max_motors=3, orch_min_confidence=0.3,
            motor_max_nodes=8, motor_n_heads=8, motor_threshold=0.01, unif_n_heads=8,
            dec_n_layers=16, dec_n_heads=8, dec_max_seq_len=1024,
            dec_state_dim=16, dec_expand=2, dec_d_conv=4, dec_ffn_mult=4,
        )
    else:
        raise ValueError(f"unknown config: {config}")

    return MoSEPipeline(cfg), cfg


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler: cosine decay con warmup
# ─────────────────────────────────────────────────────────────────────────────


def make_cosine_warmup_scheduler(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.01):
    """
    LambdaLR que hace warmup lineal hasta `warmup_steps`, luego cosine decay
    hasta `max_steps`. min_lr_ratio es el mínimo en relación al LR base.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Generation helper for eval
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def greedy_generate(
    pipeline,
    tok,
    prompt: str,
    device: torch.device,
    max_new: int = 40,
    use_amp: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Greedy decode con prompt canónico minimal `[USER: ...]\\n[AION:`.
    Devuelve (texto_generado, motor_top).
    """
    pipeline.eval()
    canonical = f"[USER: {prompt}]\n[AION:"
    try:
        ids = tok.encode(canonical, 96)
    except TypeError:
        ids = tok.encode(canonical)[:96]
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    plen = len(ids)

    motor: Optional[str] = None
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else _nullctx()

    with autocast_ctx:
        out = pipeline(cur)
    if hasattr(out, "active_motors") and out.active_motors:
        motor = out.active_motors[0]

    for _ in range(max_new):
        with autocast_ctx:
            out = pipeline(cur)
        nxt = int(out.logits[0, -1].float().argmax().item())
        if nxt in (0, EOS_TOKEN_ID):
            break
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        if cur.shape[1] >= 160:
            break

    text = tok.decode(cur[0, plen:].tolist()) if cur.shape[1] > plen else ""
    return text, motor


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *args): return False


def evaluate_generation_quality(pipeline, tok, device, use_amp: bool, max_prompts: Optional[int] = None) -> GenerationQualityResult:
    """Corre los 50 eval prompts canónicos y devuelve el GenerationQualityResult."""
    prompts = list(EVAL_PROMPTS)
    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    def gen_fn(query: str) -> Tuple[str, Optional[str]]:
        return greedy_generate(pipeline, tok, query, device, max_new=40, use_amp=use_amp)

    return generation_quality_score(prompts, gen_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────


def train(
    dataset_path:    Path,
    config:          str,
    n_steps:         int,
    lr:              float,
    grad_accum:      int,
    routing_w:       float,
    balance_w:       float,
    clip:            float,
    warmup:          int,
    eval_every:      int,
    patience:        int,
    save_dir:        Path,
    brain_dir:       Path,
    resume:          Optional[Path],
    log_every:       int,
    seed:            int,
    use_amp:         bool,
    dry_run:         bool,
    max_eval_prompts: Optional[int],
    fase_f:          bool = True,
    sparsity_w:      float = 0.1,
    sparsity_target: float = 0.5,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("=" * 68)
    log(f"FASE E — Training {config} canonical (FROM SCRATCH)")
    log("=" * 68)
    log(f"Device: {device}  AMP: {use_amp}")
    log(f"Steps: {n_steps}  grad_accum: {grad_accum}  lr: {lr}  warmup: {warmup}")
    log(f"Routing_w: {routing_w}  Balance_w: {balance_w}  Clip: {clip}")
    log(f"Fase F: {fase_f}  Sparsity_w: {sparsity_w}  Target density: {sparsity_target}")
    log(f"Eval every: {eval_every}  Patience: {patience}")
    log(f"Save dir: {save_dir}")
    log(f"Brain dir: {brain_dir}")
    log(f"Dry run: {dry_run}")
    log("")

    # ── Dataset ────────────────────────────────────────────────────────
    log(f"[1/5] Loading dataset {dataset_path.name}...")
    records = load_canonical_records(dataset_path)
    stats = quick_stats(records)
    log(f"  Records: {stats.total:,}  with_skill_or_mem={stats.n_with:,}  without={stats.n_without:,}")
    log(f"  by_domain: {stats.by_domain}")
    log(f"  by_type:   {stats.by_type}")

    # ── Tokenizer ──────────────────────────────────────────────────────
    log(f"[2/5] Loading tokenizer...")
    from experiments.train_production import build_tokenizer
    tok = build_tokenizer(32_000)
    log(f"  Vocab: {tok.vocab_size}")

    # ── Pipeline ───────────────────────────────────────────────────────
    log(f"[3/5] Building pipeline ({config})...")
    pipeline, cfg = build_pipeline(config, tok.vocab_size)
    n_params_base = sum(p.numel() for p in pipeline.parameters())
    log(f"  Base params: {n_params_base:,} ({n_params_base/1e6:.1f}M = {n_params_base/1e9:.2f}B)")

    # ── Fase F: attach sparse gates desde step 0 (Parte 27) ────────────
    sparsity_tracker = None
    per_motor_sparse: Dict[str, Dict[str, SparseLinear]] = {}
    if fase_f:
        log(f"  Attaching Fase F sparse gates (target density={sparsity_target})...")
        per_motor_sparse, sparsity_tracker = attach_fase_f_to_pipeline(
            pipeline,
            sparsity_target=sparsity_target,
            gate_hidden=8,
            max_targets_per_motor=6,
        )
        total_gates = sum(len(v) for v in per_motor_sparse.values())
        log(f"  Attached {total_gates} sparse gates across {len(per_motor_sparse)} motors")
        n_params_with_gates = sum(p.numel() for p in pipeline.parameters())
        gate_overhead = n_params_with_gates - n_params_base
        log(
            f"  After gates: {n_params_with_gates:,} params "
            f"(+{gate_overhead:,} gate params, {100*gate_overhead/max(n_params_base,1):.2f}% overhead)"
        )
    n_params = sum(p.numel() for p in pipeline.parameters())

    start_step = 0
    if resume is not None and resume.exists():
        log(f"  Resuming from {resume}")
        ck = torch.load(str(resume), map_location="cpu", weights_only=False)
        pipeline.load_state_dict(ck["model_state"], strict=False)
        start_step = int(ck.get("step", 0))
        log(f"  Resumed at step {start_step}")

    pipeline.to(device)
    if device.type == "cuda":
        log(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Sampler (50/50 SKILL/MEM balance) ──────────────────────────────
    log(f"[4/5] Sampling step indices (WeightedRandomSampler 50/50)...")
    n_samples = (n_steps - start_step) * grad_accum
    indices = weighted_sampler_indices(records, n_steps=n_samples, target_ratio=0.5, seed=seed)
    n_with = sum(1 for i in indices if records[i].has_skill or records[i].has_mem)
    log(f"  Samples to draw: {len(indices):,}  with_skill_or_mem: {n_with} ({100*n_with/max(1,len(indices)):.1f}%)")

    # ── Optimizer + scheduler + scaler ─────────────────────────────────
    log(f"[5/5] Optimizer + scheduler + scaler...")
    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = make_cosine_warmup_scheduler(optimizer, warmup, n_steps)
    scaler: Optional[torch.amp.GradScaler] = None
    if use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # State for routing/balance logging
    activation_ema = (torch.ones(5) / 5.0).to(device)
    routing_correct = 0
    routing_total = 0
    motor_counts: Dict[str, int] = {m: 0 for m in MOTOR_NAMES}

    # Best checkpoint tracking (by generation_quality.combined)
    best_combined = -1.0
    best_step = 0
    no_improve_steps = 0
    eval_history: List[Dict[str, Any]] = []

    save_dir.mkdir(parents=True, exist_ok=True)
    brain_dir.mkdir(parents=True, exist_ok=True)
    bvm = BrainVersionManager(brain_dir)
    ckpt_path = save_dir / f"aion_{config}_canonical.pt"
    metrics_path = save_dir / f"aion_{config}_canonical.metrics.json"

    log("")
    log(f"Training starts. step {start_step+1} → {n_steps}")
    log("")

    sample_pos = 0

    for step in range(start_step + 1, n_steps + 1):
        pipeline.train()
        optimizer.zero_grad()
        accum_lm = 0.0
        accum_route = 0.0
        accum_balance = 0.0
        accum_sp = 0.0

        for _ in range(grad_accum):
            if sample_pos >= len(indices):
                # Should not happen but safety
                idx = random.randrange(len(records))
            else:
                idx = indices[sample_pos]
            sample_pos += 1
            record = records[idx]
            ids = encode_record(tok, record, max_len=cfg.dec_max_seq_len)
            if len(ids) < 4:
                continue
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)

            try:
                if use_amp and device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        out = pipeline(ids_t)
                        lm = F.cross_entropy(
                            out.logits[0, :-1], ids_t[0, 1:], ignore_index=0
                        ) / grad_accum

                        # Routing supervision
                        concepts = pipeline.encoder(ids_t)
                        pooled = concepts.mean(1).mean(0, keepdim=True)
                        orch_log = pipeline.orchestrator.classifier(pooled)
                        tgt = domain_to_motor_idx(record.domain)
                        rl = F.cross_entropy(
                            orch_log, torch.tensor([tgt], dtype=torch.long, device=device)
                        ) / grad_accum

                        # Balance loss (penaliza overuse via EMA)
                        probs = F.softmax(orch_log.squeeze(0), dim=-1)
                        bl = 5.0 * (activation_ema * probs).sum() / grad_accum

                        sp = torch.zeros((), device=device)
                        if fase_f:
                            sp_val = sparsity_loss(pipeline, target=sparsity_target)
                            sp = sp_val.to(device) / grad_accum
                        loss = lm + routing_w * rl + balance_w * bl + sparsity_w * sp
                else:
                    out = pipeline(ids_t)
                    lm = F.cross_entropy(
                        out.logits[0, :-1], ids_t[0, 1:], ignore_index=0
                    ) / grad_accum
                    concepts = pipeline.encoder(ids_t)
                    pooled = concepts.mean(1).mean(0, keepdim=True)
                    orch_log = pipeline.orchestrator.classifier(pooled)
                    tgt = domain_to_motor_idx(record.domain)
                    rl = F.cross_entropy(
                        orch_log, torch.tensor([tgt], dtype=torch.long, device=device)
                    ) / grad_accum
                    probs = F.softmax(orch_log.squeeze(0), dim=-1)
                    bl = 5.0 * (activation_ema * probs).sum() / grad_accum
                    sp = torch.zeros((), device=device)
                    if fase_f:
                        sp_val = sparsity_loss(pipeline, target=sparsity_target)
                        sp = sp_val.to(device) / grad_accum
                    loss = lm + routing_w * rl + balance_w * bl + sparsity_w * sp

                if not math.isfinite(loss.item()):
                    continue

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_lm += lm.item() * grad_accum
                accum_route += rl.item() * grad_accum
                accum_balance += bl.item() * grad_accum
                if fase_f:
                    accum_sp += float(sp.item()) * grad_accum

                activation_ema = 0.99 * activation_ema + 0.01 * probs.detach()
                if hasattr(out, "active_motors"):
                    for m in out.active_motors:
                        if m in motor_counts:
                            motor_counts[m] += 1
                expected_motor = MOTOR_NAMES[tgt]
                if hasattr(out, "active_motors") and expected_motor in out.active_motors:
                    routing_correct += 1
                routing_total += 1
            except Exception as exc:
                log(f"  step {step} micro {sample_pos} ERROR: {exc}")
                continue

        # Gradient clipping + step
        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(pipeline.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(pipeline.parameters(), clip)
            optimizer.step()
        scheduler.step()

        # Periodic logging
        if step % log_every == 0:
            elapsed = time.perf_counter() - t0
            sps = (step - start_step) / max(elapsed, 1e-6)
            eta_min = (n_steps - step) / max(sps, 1e-6) / 60
            lr_now = scheduler.get_last_lr()[0]
            racc = 100 * routing_correct / max(1, routing_total)
            vram = ""
            if device.type == "cuda":
                vram = f"  VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB"
            sp_str = ""
            if fase_f and sparsity_tracker is not None:
                report = sparsity_tracker.collect()
                density = report.get("avg_density", 0.0)
                sp_str = f"  sp={accum_sp/grad_accum:.3f}  dens={density:.2f}"
            log(
                f"  step {step:>5}/{n_steps}  "
                f"lm={accum_lm/grad_accum:.3f}  "
                f"route={accum_route/grad_accum:.3f}  "
                f"bal={accum_balance/grad_accum:.3f}  "
                f"acc={racc:.0f}%{sp_str}  "
                f"lr={lr_now:.1e}  "
                f"{sps:.2f}sps  ETA {eta_min:.0f}m{vram}"
            )

        # Eval cada eval_every steps
        if step % eval_every == 0 or step == n_steps:
            log(f"  Evaluating at step {step} on canonical eval prompts...")
            eval_t0 = time.perf_counter()
            result = evaluate_generation_quality(
                pipeline, tok, device, use_amp, max_prompts=max_eval_prompts,
            )
            eval_secs = time.perf_counter() - eval_t0
            entry = {
                "step":             step,
                "lm_loss":          accum_lm / grad_accum,
                "routing_acc":      100 * routing_correct / max(1, routing_total),
                "gen_exact_match":  round(result.exact_match, 4),
                "gen_bleu":         round(result.bleu, 4),
                "gen_routing":      round(result.routing_accuracy, 4),
                "gen_combined":     round(result.combined, 4),
                "eval_seconds":     round(eval_secs, 2),
                "lr":               scheduler.get_last_lr()[0],
            }
            eval_history.append(entry)

            log(
                f"  EVAL step {step}: exact={result.exact_match:.3f} "
                f"bleu={result.bleu:.3f} "
                f"routing={result.routing_accuracy:.3f} "
                f"COMBINED={result.combined:.4f} "
                f"({eval_secs:.1f}s)"
            )
            log(f"  per_domain: {result.per_domain}")

            # Best checkpoint by combined score
            if result.combined > best_combined + 1e-6:
                best_combined = result.combined
                best_step = step
                no_improve_steps = 0
                payload = {
                    "model_state":    pipeline.state_dict(),
                    "config_name":    f"aion_{config}_canonical",
                    "step":           step,
                    "n_params":       n_params,
                    "best_combined":  best_combined,
                    "result":         result.to_dict(),
                    "history":        eval_history,
                }
                torch.save(payload, str(ckpt_path))
                log(f"  ★ NEW BEST: combined={best_combined:.4f} → saved {ckpt_path.name}")
            else:
                no_improve_steps += eval_every
                log(f"  no improvement (best={best_combined:.4f} @ step {best_step}, no_improve={no_improve_steps})")

            # Save metrics history JSON
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({
                    "best_combined": best_combined,
                    "best_step":     best_step,
                    "n_params":      n_params,
                    "config":        config,
                    "history":       eval_history,
                }, f, ensure_ascii=False, indent=2)

            # Early stopping
            if no_improve_steps >= patience:
                log(f"  EARLY STOP at step {step} (patience {patience} exceeded)")
                break

        # Dry run breakout
        if dry_run and step >= n_steps:
            log(f"  DRY RUN: completed {step} steps successfully")
            break

    # ── Final save: BrainVersionManager ────────────────────────────────
    elapsed = time.perf_counter() - t0
    log("")
    log(f"DONE in {elapsed/60:.1f} min")
    log(f"Best combined: {best_combined:.4f} @ step {best_step}")

    # ── Fase F: verificación final de scaffolding (adapters+sparse) ────
    scaffold_report: Dict[str, Any] = {}
    if fase_f:
        log("")
        log("Fase F scaffolding verification (attach/detach smoke test)...")
        scaffold_report = verify_adapter_scaffolding(pipeline, device)
        all_ok = all(v.get("ok") for v in scaffold_report.values())
        for motor_name, res in scaffold_report.items():
            status = "OK" if res.get("ok") else "FAIL"
            detail = ""
            if "n_targets" in res:
                detail = f" ({res['n_targets']} targets)"
            if "error" in res:
                detail = f" error={res['error']}"
            log(f"  {motor_name}: {status}{detail}")
        log(f"  scaffolding_all_ok = {all_ok}")
        if sparsity_tracker is not None:
            final_density = sparsity_tracker.collect()
            log(f"  final sparsity report: avg_density={final_density.get('avg_density', 0.0):.3f}")

    if best_combined > 0:
        try:
            version = bvm.save_version(
                state_dict=pipeline.state_dict(),
                notes=f"aion_{config}_canonical from scratch — best combined={best_combined:.4f}",
                metrics={
                    "combined":         best_combined,
                    "best_step":        float(best_step),
                    "n_params":         float(n_params),
                    "elapsed_minutes":  elapsed / 60,
                },
                metadata={
                    "config":         config,
                    "n_steps":        n_steps,
                    "history_len":    len(eval_history),
                    "from_scratch":   True,
                    "dataset":        dataset_path.name,
                    "motor_counts":   motor_counts,
                },
            )
            log(f"BrainVersionManager: saved as {version.id} (parent={version.parent_id})")
        except Exception as exc:
            log(f"BrainVersionManager save FAILED: {exc}")

    return {
        "best_combined":    best_combined,
        "best_step":        best_step,
        "n_steps":          n_steps,
        "elapsed":          elapsed,
        "history":          eval_history,
        "n_params":         n_params,
        "fase_f":           fase_f,
        "scaffold_report":  scaffold_report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Train AION-C 1.1B (or tiny) on the canonical 70K dataset")
    ap.add_argument("--config",     choices=("tiny", "1b"), default="1b")
    ap.add_argument("--dataset",    type=Path, default=ROOT / "datasets" / "dataset_canonical_86k.jsonl")
    ap.add_argument("--save-dir",   type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--brain-dir",  type=Path, default=ROOT / "brain")
    ap.add_argument("--resume",     type=Path, default=None)
    ap.add_argument("--steps",      type=int,   default=15000)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int,   default=16)
    ap.add_argument("--routing-w",  type=float, default=1.0)
    ap.add_argument("--balance-w",  type=float, default=0.5)
    ap.add_argument("--clip",       type=float, default=0.5)
    ap.add_argument("--warmup",     type=int,   default=300)
    ap.add_argument("--eval-every", type=int,   default=200)
    ap.add_argument("--patience",   type=int,   default=500)
    ap.add_argument("--log-every",  type=int,   default=50)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--no-amp",     action="store_true", help="Disable fp16 mixed precision")
    ap.add_argument("--dry-run",    action="store_true", help="Limit to N steps without saving brain version")
    ap.add_argument("--max-eval-prompts", type=int, default=None,
                    help="Limit eval to first N prompts (useful for dry-run)")
    # Fase F integration
    ap.add_argument("--no-fase-f", action="store_true",
                    help="Disable Fase F sparse gates + scaffolding verification")
    ap.add_argument("--sparsity-w", type=float, default=0.1,
                    help="Weight of sparsity_loss in combined loss (Parte 27)")
    ap.add_argument("--sparsity-target", type=float, default=0.5,
                    help="Target density for sparse gates (0.5=permissive, 0.15=aggressive)")
    args = ap.parse_args()

    use_amp = not args.no_amp and torch.cuda.is_available()

    train(
        dataset_path     = args.dataset,
        config           = args.config,
        n_steps          = args.steps,
        lr               = args.lr,
        grad_accum       = args.grad_accum,
        routing_w        = args.routing_w,
        balance_w        = args.balance_w,
        clip             = args.clip,
        warmup           = args.warmup,
        eval_every       = args.eval_every,
        patience         = args.patience,
        save_dir         = args.save_dir,
        brain_dir        = args.brain_dir,
        resume           = args.resume,
        log_every        = args.log_every,
        seed             = args.seed,
        use_amp          = use_amp,
        dry_run          = args.dry_run,
        max_eval_prompts = args.max_eval_prompts,
        fase_f           = not args.no_fase_f,
        sparsity_w       = args.sparsity_w,
        sparsity_target  = args.sparsity_target,
    )


if __name__ == "__main__":
    main()
