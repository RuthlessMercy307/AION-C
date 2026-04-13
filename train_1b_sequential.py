"""
train_1b_sequential.py — Motor-Sequential Training para AION-C 1.1B.

Técnica única de la arquitectura MoSE que reduce dramáticamente el costo
del training local al entrenar componentes por fases, cada una con sólo
una fracción del modelo trainable.

Fases:
    1. Backbone (encoder + decoder + unifier)        — 953M trainable
    2. Motor specialization, una vez por cada motor  —  ~30M trainable cada
    3. Orchestrator routing                           — 0.7M trainable
    4. LoRA cross-motor harmonization                 — ~300K trainable

Uso:
    # Plan A: todo local, full Phase 1 (~5 horas en Ryzen 3600 + 64 GB)
    python train_1b_sequential.py

    # Plan B: Phase 1 reducida (~3.5 horas)
    python train_1b_sequential.py --phase-1-steps 300

    # Plan C: empezar desde un checkpoint de Phase 1 hecho en Vast.ai
    python train_1b_sequential.py --skip-phase-1 \\
        --resume-checkpoint checkpoints/aion_1b_after_phase_1.pt

    # Correr solo una fase
    python train_1b_sequential.py --only phase_2_motor:cora

    # Sanity check rápido con tiny
    python train_1b_sequential.py --config tiny --steps-scale 0.1

El training es PAUSABLE: entre fases se guarda un checkpoint. Para
reanudar desde la última fase terminada:

    python train_1b_sequential.py --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import torch

from synth.canonical_dataloader import (
    load_canonical_records, encode_record,
    domain_to_motor_idx, MOTOR_NAMES, EOS_TOKEN_ID,
)
from experiments.benchmark_local import build_pipeline
from training.sequential_trainer import (
    SequentialConfig, SequentialTrainer, PhaseResult,
)


# ════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# Default plan
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_PLAN = [
    ("phase_1_backbone",         1500),   # reduced from 2000 for 16 GB compatibility
    ("phase_2_motor:cora",       1500),
    ("phase_2_motor:forge_c",    1500),
    ("phase_2_motor:axiom",      1500),
    ("phase_2_motor:muse",       1000),
    ("phase_2_motor:empathy",    1000),
    ("phase_3_orchestrator",      500),
    ("phase_4_adapters",          500),
]


# ════════════════════════════════════════════════════════════════════════════
# Data source
# ════════════════════════════════════════════════════════════════════════════

class CanonicalDataSource:
    """Provee batches del dataset canonical filtrados por dominio."""

    def __init__(self, dataset_path: Path, tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        log(f"Loading dataset from {dataset_path.name}...")
        self.records = load_canonical_records(dataset_path)
        log(f"  loaded {len(self.records):,} records")

        # Split by domain for Phase 2 per-motor filtering
        self._by_domain: Dict[str, List[int]] = {}
        for i, rec in enumerate(self.records):
            self._by_domain.setdefault(rec.domain, []).append(i)
        sizes = {k: len(v) for k, v in self._by_domain.items()}
        log(f"  by domain: {sizes}")

    def _encode(self, rec) -> torch.Tensor:
        ids = encode_record(self.tokenizer, rec, max_len=self.max_len)
        # Hard truncate at self.max_len (encode_record may ignore its max_len)
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        # Ensure at least 4 tokens
        if len(ids) < 4:
            ids = ids + [0] * (4 - len(ids))
        return torch.tensor([ids], dtype=torch.long)

    def all_data_fn(self):
        """Yields records from the entire dataset, uniform sampling."""
        import random as _random
        rng = _random.Random(42)

        def _fn():
            idx = rng.randrange(len(self.records))
            rec = self.records[idx]
            motor_idx = domain_to_motor_idx(rec.domain)
            return self._encode(rec), motor_idx
        return _fn

    def domain_data_fn(self, domain: str):
        """Yields only records from a specific domain (for Phase 2)."""
        import random as _random
        rng = _random.Random(hash(domain))
        indices = self._by_domain.get(domain, [])
        if not indices:
            log(f"  WARN: no records for domain {domain}, falling back to all")
            return self.all_data_fn()

        def _fn():
            rec_idx = indices[rng.randrange(len(indices))]
            rec = self.records[rec_idx]
            motor_idx = domain_to_motor_idx(rec.domain)
            return self._encode(rec), motor_idx
        return _fn

    def balanced_motor_data_fn(self):
        """Round-robin across the 5 real motor domains.

        Used in Phase 3 (orchestrator) to guarantee each class is seen
        equally often, avoiding collapse-to-majority behavior.
        Excludes 'general' and 'metacognitive' which map to motor 0 by default
        and would corrupt the training signal with class imbalance.
        """
        import random as _random
        motor_domains = ["cora", "forge_c", "muse", "axiom", "empathy"]
        rngs = {d: _random.Random(hash(d) & 0xFFFFFFFF) for d in motor_domains}
        indices_by_domain = {d: self._by_domain.get(d, []) for d in motor_domains}
        assert all(indices_by_domain[d] for d in motor_domains), "missing domain records"
        counter = {"i": 0}

        def _fn():
            domain = motor_domains[counter["i"] % len(motor_domains)]
            counter["i"] += 1
            idxs = indices_by_domain[domain]
            rec = self.records[idxs[rngs[domain].randrange(len(idxs))]]
            motor_idx = domain_to_motor_idx(rec.domain)
            return self._encode(rec), motor_idx
        return _fn


# ════════════════════════════════════════════════════════════════════════════
# Motor name mapping
# ════════════════════════════════════════════════════════════════════════════

MOTOR_TO_DOMAIN = {
    "cora": "cora",
    "forge_c": "forge_c",
    "axiom": "axiom",
    "muse": "muse",
    "empathy": "empathy",
}


# ════════════════════════════════════════════════════════════════════════════
# Main training pipeline
# ════════════════════════════════════════════════════════════════════════════

def train_sequential(
    dataset_path: Path,
    config: str,
    save_dir: Path,
    plan: List[Tuple[str, int]],
    resume_checkpoint: Optional[Path] = None,
    only: Optional[str] = None,
    skip_phases: Optional[List[str]] = None,
    steps_scale: float = 1.0,
    phase_1_optimizer: str = "adamw",
    phase_1_lr: Optional[float] = None,
    monitoring: bool = False,
    log_dir: Optional[Path] = None,
    max_len: int = 128,
    device: str = "cpu",
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    save_dir.mkdir(parents=True, exist_ok=True)
    skip_phases = skip_phases or []

    log("=" * 72)
    log(f"Motor-Sequential Training — config={config}")
    log("=" * 72)

    # Tokenizer
    log("Loading tokenizer...")
    from experiments.train_production import build_tokenizer
    tok = build_tokenizer(32_000)
    log(f"  vocab: {tok.vocab_size}")

    # Pipeline
    log(f"Building {config} pipeline...")
    pipeline, cfg = build_pipeline(config, vocab_size=tok.vocab_size)
    n_params = sum(p.numel() for p in pipeline.parameters())
    log(f"  params: {n_params:,} ({n_params/1e9:.2f}B)")

    # Trainer
    default_lr = 1e-2 if phase_1_optimizer == "sgd" else 3e-4
    seq_cfg = SequentialConfig(
        device=device,
        log_every=25,
        phase_1_optimizer=phase_1_optimizer,
        lr_phase_1=phase_1_lr if phase_1_lr is not None else default_lr,
    )
    log(f"  device: {device}")
    log(f"  phase 1 optimizer: {seq_cfg.phase_1_optimizer}  lr: {seq_cfg.lr_phase_1}")

    # Monitoring context
    mon_ctx = None
    if monitoring:
        from training.monitoring import create_monitoring_context
        mon_ctx = create_monitoring_context(
            log_dir=log_dir,
            enable_watchdog=True,
            log_every=25,
            poll_every=10,
        )
        log(f"  monitoring enabled: {mon_ctx.log_dir}")
        log(f"    metrics:  {mon_ctx.log_dir / 'metrics.jsonl'}")
        log(f"    control:  {mon_ctx.log_dir / 'control.json'}")
        log(f"    watchdog: active, interval 60s")

    trainer = SequentialTrainer(pipeline, seq_cfg, monitoring=mon_ctx)

    # Resume from checkpoint if requested
    resume_from_phase = None
    if resume_checkpoint and resume_checkpoint.exists():
        log(f"Resuming from {resume_checkpoint}")
        resume_from_phase = trainer.load_checkpoint(resume_checkpoint)
        log(f"  resumed after phase: {resume_from_phase}")

    # Data source — use the lower of max_len and model's dec_max_seq_len
    effective_max_len = min(max_len, cfg.dec_max_seq_len)
    log(f"  max_len (token truncation): {effective_max_len}")
    data_source = CanonicalDataSource(dataset_path, tok, max_len=effective_max_len)

    # Run phases
    results: List[PhaseResult] = []
    phase_reached = False if resume_from_phase else True

    for phase_name, n_steps in plan:
        # Resume logic: skip phases until we reach the one AFTER resume
        if not phase_reached:
            if phase_name == resume_from_phase:
                phase_reached = True
            continue

        if only and phase_name != only:
            continue
        if phase_name in skip_phases:
            log(f"[{phase_name}] SKIPPED by flag")
            continue

        # Scale steps
        effective_steps = max(1, int(n_steps * steps_scale))

        # Select data source for this phase
        if phase_name.startswith("phase_2_motor:"):
            motor_name = phase_name.split(":", 1)[1]
            domain = MOTOR_TO_DOMAIN.get(motor_name, motor_name)
            data_fn = data_source.domain_data_fn(domain)
        elif phase_name == "phase_3_orchestrator":
            data_fn = data_source.balanced_motor_data_fn()
        else:
            data_fn = data_source.all_data_fn()

        # Dispatch to the correct phase method
        log("")
        try:
            if phase_name == "phase_1_backbone":
                result = trainer.run_phase_1_backbone(data_fn, n_steps=effective_steps)
            elif phase_name.startswith("phase_2_motor:"):
                motor_name = phase_name.split(":", 1)[1]
                result = trainer.run_phase_2_motor(motor_name, data_fn, n_steps=effective_steps)
            elif phase_name == "phase_3_orchestrator":
                result = trainer.run_phase_3_orchestrator(data_fn, n_steps=effective_steps)
            elif phase_name == "phase_4_adapters":
                result = trainer.run_phase_4_adapters(data_fn, n_steps=effective_steps)
            else:
                log(f"  UNKNOWN phase: {phase_name}")
                continue
            results.append(result)
        except Exception as exc:
            log(f"  PHASE FAILED: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            break

        # Checkpoint after each phase (pausable)
        ckpt_path = save_dir / f"aion_{config}_sequential.pt"
        trainer.save_checkpoint(ckpt_path, phase=phase_name)
        gc.collect()

        # Check if pause/stop was requested by control file during this phase
        if trainer._pause_requested:
            log(f"PAUSED after {phase_name}. Checkpoint saved at {ckpt_path}.")
            log(f"Resume with: python train_1b_sequential.py --config {config} --resume")
            break
        if trainer._stop_requested:
            log(f"STOPPED after {phase_name}. Checkpoint saved.")
            break

    # Final summary
    total_elapsed = time.perf_counter() - t0
    log("")
    log("=" * 72)
    log(f"DONE in {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h)")
    log("=" * 72)
    for r in results:
        log(f"  {r.name:<30} {r.sps:.3f} sps  loss {r.final_loss:.3f}  "
            f"{r.total_seconds/60:.1f} min")

    summary = {
        "config": config,
        "total_params": n_params,
        "total_elapsed_sec": total_elapsed,
        "total_elapsed_min": round(total_elapsed / 60, 2),
        "results": [r.to_dict() for r in results],
    }
    summary_path = save_dir / f"aion_{config}_sequential.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Summary: {summary_path}")

    # Cleanup monitoring
    if mon_ctx is not None:
        mon_ctx.write_note(f"training complete ({total_elapsed/60:.1f} min)", author="training")
        mon_ctx.close()
        log(f"Monitoring log dir: {mon_ctx.log_dir}")

    return summary


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=("tiny", "small_300m", "1b"), default="1b")
    p.add_argument("--dataset", type=Path,
                   default=ROOT / "datasets" / "dataset_canonical_86k.jsonl")
    p.add_argument("--save-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the last checkpoint in save_dir")
    p.add_argument("--resume-checkpoint", type=Path, default=None,
                   help="Resume from a specific checkpoint file")
    p.add_argument("--skip-phase-1", action="store_true",
                   help="Skip backbone pretraining (Phase 1). Useful if "
                        "resuming from a cloud-trained backbone.")
    p.add_argument("--only", type=str, default=None,
                   help="Run only this phase, e.g. phase_2_motor:cora")
    p.add_argument("--phase-1-steps", type=int, default=None,
                   help="Override phase 1 step count (default 2000)")
    p.add_argument("--phase-2-steps", type=int, default=None,
                   help="Override phase 2 step count (default 1500 for cora/forge/axiom, 1000 for muse/empathy)")
    p.add_argument("--steps-scale", type=float, default=1.0,
                   help="Multiply all step counts by this factor")
    p.add_argument("--phase-1-optimizer", choices=("adamw", "sgd"), default="adamw",
                   help="Optimizer for Phase 1. Use 'sgd' on 16 GB RAM to "
                        "avoid swap from AdamW moment buffers (saves ~4 GB).")
    p.add_argument("--phase-1-lr", type=float, default=None,
                   help="Phase 1 learning rate (default 3e-4 for adamw, 1e-2 for sgd)")
    p.add_argument("--monitoring", choices=("on", "off"), default="off",
                   help="Enable structured logging + control file + watchdog thread")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Custom log directory (default: training/logs/sequential_<ts>)")
    p.add_argument("--max-len", type=int, default=128,
                   help="Hard truncation length for token sequences. Lower = "
                        "faster per step. Dataset median is ~130 tokens, so 128 "
                        "captures most content without slowing each step.")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "dml"), default="auto",
                   help="Device to train on. 'auto' prefers cuda > dml > cpu.")
    args = p.parse_args()

    # Build plan
    plan = []
    for phase_name, default_steps in DEFAULT_PLAN:
        if phase_name == "phase_1_backbone" and args.phase_1_steps is not None:
            plan.append((phase_name, args.phase_1_steps))
        elif phase_name.startswith("phase_2_motor:") and args.phase_2_steps is not None:
            plan.append((phase_name, args.phase_2_steps))
        else:
            plan.append((phase_name, default_steps))

    skip = ["phase_1_backbone"] if args.skip_phase_1 else []

    resume_ckpt = args.resume_checkpoint
    if args.resume and resume_ckpt is None:
        resume_ckpt = args.save_dir / f"aion_{args.config}_sequential.pt"
        if not resume_ckpt.exists():
            resume_ckpt = None

    train_sequential(
        dataset_path=args.dataset,
        config=args.config,
        save_dir=args.save_dir,
        plan=plan,
        resume_checkpoint=resume_ckpt,
        only=args.only,
        skip_phases=skip,
        steps_scale=args.steps_scale,
        phase_1_optimizer=args.phase_1_optimizer,
        phase_1_lr=args.phase_1_lr,
        monitoring=(args.monitoring == "on"),
        log_dir=args.log_dir,
        max_len=args.max_len,
        device=args.device,
    )


if __name__ == "__main__":
    main()
