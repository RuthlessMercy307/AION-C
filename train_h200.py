#!/usr/bin/env python3
"""
train_h200.py -- Script maestro para entrenamiento completo en H200
====================================================================

UN SOLO COMANDO:
    python train_h200.py

Entrena hasta CONVERGENCIA, sin limites de tiempo.
Cada fase termina cuando:
  - val_loss se estabiliza (convergence_delta < threshold por N steps)
  - val_loss deja de mejorar (patience agotado)
  - val_F1 supera el threshold de calidad

NO hay limites de steps ni de tiempo. Si necesita 6 horas, entrena 6 horas.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch
import torch.nn as nn


# ============================================================================
# CONFIGURATION -- Sin limites de steps, solo calidad
# ============================================================================

@dataclass
class H200Config:
    """Training config. Steps are set to 999999 (effectively unlimited).
    Training stops ONLY by convergence, early stopping, or F1 threshold."""
    name: str
    mose_config_name: str

    # Learning rates per phase
    ph0_lr: float;  ph1_lr: float;  ph2_lr: float
    ph3_orch_lr: float;  ph3_e2e_lr: float;  ph4_lr: float

    # Warmup steps per phase
    ph0_warmup: int;  ph1_warmup: int;  ph2_warmup: int
    ph3_warmup: int

    # Eval frequency (how often to check quality)
    eval_every: int

    # Convergence criteria (ONLY way training stops)
    patience: int          # steps without val_loss improvement -> early stop
    conv_delta: float      # threshold for "loss is stable"
    conv_window: int       # steps of stable loss -> converged
    f1_threshold: float    # if val_F1 exceeds this, phase is done (0 = disabled)

    # Batching: real GPU-parallel batch + optional gradient accumulation
    # effective_batch = train_batch_size * grad_accum_steps
    train_batch_size: int
    grad_accum_steps: int

    # Data
    n_train: int;  n_val: int;  max_seq: int

    # Checkpoint frequency (in steps, for crash recovery)
    ckpt_every: int


# Configs: steps son 999999 (se para por calidad, no por steps)
_UNLIMITED = 999_999

CONFIGS = {
    "tiny": H200Config(
        name="tiny", mose_config_name="tiny",
        ph0_lr=3e-4, ph1_lr=2e-4, ph2_lr=3e-4,
        ph3_orch_lr=1e-3, ph3_e2e_lr=5e-5, ph4_lr=1e-4,
        ph0_warmup=50, ph1_warmup=50, ph2_warmup=20, ph3_warmup=20,
        eval_every=100,
        patience=500, conv_delta=0.001, conv_window=300, f1_threshold=0.0,
        train_batch_size=1, grad_accum_steps=1,
        n_train=1000, n_val=50, max_seq=128, ckpt_every=300,
    ),
    "medium": H200Config(
        name="medium", mose_config_name="medium",
        ph0_lr=3e-4, ph1_lr=2e-4, ph2_lr=2e-4,
        ph3_orch_lr=5e-4, ph3_e2e_lr=2e-5, ph4_lr=8e-5,
        ph0_warmup=500, ph1_warmup=500, ph2_warmup=200, ph3_warmup=200,
        eval_every=500,
        patience=4000, conv_delta=0.0005, conv_window=2000, f1_threshold=0.0,
        train_batch_size=4, grad_accum_steps=1,
        n_train=10000, n_val=200, max_seq=512, ckpt_every=1000,
    ),
    "production": H200Config(
        name="production", mose_config_name="production",
        ph0_lr=3e-4, ph1_lr=2e-4, ph2_lr=1e-4,
        ph3_orch_lr=3e-4, ph3_e2e_lr=1e-5, ph4_lr=5e-5,
        ph0_warmup=500, ph1_warmup=500, ph2_warmup=200, ph3_warmup=200,
        eval_every=200,
        patience=2000, conv_delta=0.0003, conv_window=1000, f1_threshold=0.0,
        train_batch_size=4, grad_accum_steps=1,
        n_train=20000, n_val=300, max_seq=128, ckpt_every=500,
    ),
}


def to_hparams(cfg: H200Config, phase_lr: float, warmup: int):
    """Convert to _TrainHparams for a specific phase. Steps = UNLIMITED."""
    from experiments.train_production import _TrainHparams
    return _TrainHparams(
        # All phases get unlimited steps -- monitor decides when to stop
        ph0_steps=_UNLIMITED, ph0_lr=phase_lr, ph0_warmup=warmup,
        ph0_eval=cfg.eval_every, ph0_patience=cfg.patience, ph0_conv_win=cfg.conv_window,
        ph1_steps=_UNLIMITED, ph1_lr=phase_lr, ph1_warmup=warmup,
        ph1_eval=cfg.eval_every, ph1_patience=cfg.patience, ph1_conv_win=cfg.conv_window,
        ph2_steps=_UNLIMITED, ph2_lr=phase_lr, ph2_warmup=warmup,
        ph2_eval=cfg.eval_every, ph2_patience=cfg.patience, ph2_conv_win=cfg.conv_window,
        ph3_orch_steps=_UNLIMITED, ph3_orch_lr=cfg.ph3_orch_lr,
        ph3_e2e_steps=_UNLIMITED, ph3_e2e_lr=cfg.ph3_e2e_lr,
        ph3_warmup=cfg.ph3_warmup, ph3_eval=cfg.eval_every,
        ph3_patience=cfg.patience, ph3_conv_win=cfg.conv_window,
        ph4_steps=_UNLIMITED, ph4_lr=cfg.ph4_lr, ph4_eval=cfg.eval_every,
        conv_delta=cfg.conv_delta,
        train_batch_size=cfg.train_batch_size,
        grad_accum_steps=cfg.grad_accum_steps,
        n_train=cfg.n_train, n_val=cfg.n_val, max_seq=cfg.max_seq,
        ckpt_every=cfg.ckpt_every,
    )


# ============================================================================
# TEE LOGGER -- stdout + file simultaneously
# ============================================================================

class TeeLogger:
    """Writes to both stdout and a log file simultaneously."""

    def __init__(self, log_path: Path):
        self._stdout = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(str(log_path), "w", encoding="utf-8", buffering=1)
        self._path = log_path

    def write(self, msg: str):
        self._stdout.write(msg)
        try:
            self._file.write(msg)
        except (ValueError, OSError):
            pass  # file closed or disk full

    def flush(self):
        self._stdout.flush()
        try:
            self._file.flush()
        except (ValueError, OSError):
            pass

    def close(self):
        try:
            self._file.close()
        except (ValueError, OSError):
            pass

    @property
    def encoding(self):
        return getattr(self._stdout, "encoding", "utf-8")

    def isatty(self):
        return False

    def fileno(self):
        return self._stdout.fileno()


_tee: Optional[TeeLogger] = None


def install_tee_logger(output_dir: Path) -> TeeLogger:
    """Install TeeLogger: all print() output goes to stdout + file."""
    global _tee
    log_path = output_dir / "training_log.txt"
    _tee = TeeLogger(log_path)
    sys.stdout = _tee
    return _tee


# ============================================================================
# METRICS LOGGER -- JSONL with per-eval records
# ============================================================================

_metrics_path: Optional[Path] = None


def init_metrics_log(output_dir: Path):
    """Initialize the central metrics JSONL file."""
    global _metrics_path
    _metrics_path = output_dir / "training_metrics.jsonl"
    # Truncate if exists (new training run)
    with open(str(_metrics_path), "w", encoding="utf-8") as f:
        pass


def flush_phase_metrics(
    config_name: str,
    phase_name: str,
    monitor_log_path: Path,
    phase_elapsed_s: float,
):
    """Read the monitor's JSONL log and append to central metrics with context."""
    if _metrics_path is None:
        return
    if not monitor_log_path.exists():
        return

    records = []
    try:
        with open(str(monitor_log_path), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception:
        return

    with open(str(_metrics_path), "a", encoding="utf-8") as f:
        for rec in records:
            enriched = {
                "config": config_name,
                "phase": phase_name,
                "step": rec.get("step"),
                "train_loss": rec.get("train_loss"),
                "val_loss": rec.get("val_loss"),
                "val_f1": rec.get("val_f1"),
                "elapsed_s": rec.get("elapsed_s"),
                "phase_elapsed_s": round(phase_elapsed_s, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generated": rec.get("generated", []),
            }
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    log(f"  Metrics: {len(records)} eval records -> {_metrics_path.name}")


# ============================================================================
# HELPERS
# ============================================================================

def log(msg: str, level: str = "INFO"):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "vram_gb": props.total_memory / 1e9,
        "bf16": torch.cuda.is_bf16_supported(),
    }


def vram_report() -> str:
    if not torch.cuda.is_available():
        return "CPU mode"
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    return f"VRAM: {alloc:.2f} GB allocated, {peak:.2f} GB peak"


def phase_banner(
    phase: str,
    description: str,
    trainable_params: int,
    total_params: int,
    n_train_examples: int,
    convergence_info: str,
    lr: float,
    eval_every: int,
    batch_size: int = 1,
    grad_accum: int = 1,
):
    """Print a clear banner at the start of each phase."""
    eff = batch_size * grad_accum
    log("=" * 70)
    log(f"  {phase}: {description}")
    log(f"  Trainable params : {trainable_params:>12,} / {total_params:,} total")
    log(f"  Training data    : {n_train_examples:>12,} examples")
    log(f"  Learning rate    : {lr:.1e}")
    log(f"  Batch            : {batch_size} real x {grad_accum} accum = {eff} effective")
    log(f"  Eval every       : {eval_every} steps")
    log(f"  Stop criteria    : {convergence_info}")
    log(f"  NO step limit -- trains until convergence")
    log("=" * 70)


# ============================================================================
# DATASET SETUP
# ============================================================================

def find_opus_datasets() -> Optional[Path]:
    candidates = [
        _ROOT / "datasets" / "opus",
        _ROOT.parent / "DataSet-Generator-Claude-Opus" / "mose_distillation_datasets" / "datasets",
    ]
    for c in candidates:
        if c.exists() and (c / "mose_cora.jsonl").exists():
            return c
    return None


def ensure_instruction_dataset():
    it_path = _ROOT / "datasets" / "instruction_tuning.jsonl"
    if it_path.exists():
        n = sum(1 for _ in open(str(it_path), encoding="utf-8"))
        log(f"Instruction tuning dataset: {n:,} examples")
        return
    log("Generating instruction tuning dataset...")
    from synth.instruction_gen import InstructionGenerator, write_jsonl
    gen = InstructionGenerator(seed=42)
    examples = gen.generate_all()
    write_jsonl(examples, it_path)
    log(f"Generated {len(examples):,} instruction tuning examples")


# ============================================================================
# TRAIN ONE CONFIG -- Quality-driven, no time limits
# ============================================================================

def train_one_config(
    cfg: H200Config,
    device: torch.device,
    output_dir: Path,
    ds_root: Optional[Path],
) -> Dict[str, Any]:
    """Train one model through all 5 phases until convergence."""
    from experiments.train_production import (
        build_pipeline_and_tok, load_all_datasets,
        run_phase0, run_phase1, run_phase2, run_phase3, run_phase4,
        save_checkpoint, _encode, count_trainable, freeze_all_except,
    )

    run_dir = output_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    conv_info = (
        f"patience={cfg.patience}, conv_delta={cfg.conv_delta}, "
        f"conv_window={cfg.conv_window}"
    )

    log("")
    log("#" * 70)
    log(f"#  TRAINING: {cfg.name.upper()} -- until convergence")
    log(f"#  Output: {run_dir}")
    log("#" * 70)

    t_start = time.perf_counter()

    # Build pipeline
    log(f"Building pipeline ({cfg.mose_config_name})...")
    pipeline, tok, mose_cfg = build_pipeline_and_tok(cfg.mose_config_name, device)
    total_params = sum(p.numel() for p in pipeline.parameters())
    log(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    log(vram_report())

    # Load datasets
    log("Loading datasets...")
    datasets = load_all_datasets(
        max_examples=cfg.n_train + cfg.n_val + 100,
        eval_size=cfg.n_val,
        dataset_root=ds_root,
    )
    domains = list(datasets.keys())
    n_data = sum(len(ds[0]) for ds in datasets.values()) if datasets else 0
    log(f"Domains: {domains or ['synthetic fallback']}, {n_data} total examples")

    results = {"config": cfg.name, "params": total_params, "phases": {}}

    # ── Phase 0: Learn to Speak ──
    hp0 = to_hparams(cfg, cfg.ph0_lr, cfg.ph0_warmup)
    phase_banner("PHASE 0", "Learn to Speak (encoder + decoder only)",
                 count_trainable(pipeline), total_params,
                 cfg.n_train, conv_info, cfg.ph0_lr, cfg.eval_every, cfg.train_batch_size, cfg.grad_accum_steps)
    r0 = run_phase0(pipeline, mose_cfg, tok, datasets, hp0, device, run_dir)
    results["phases"]["phase0"] = _phase_result_dict(r0)
    flush_phase_metrics(cfg.name, "phase0", run_dir / "phase0_monitor.jsonl", r0.elapsed_s)
    log(f"Phase 0 done: {r0.summary_line()}")
    log(vram_report())

    # ── Phase 1: Full Backbone ──
    hp1 = to_hparams(cfg, cfg.ph1_lr, cfg.ph1_warmup)
    phase_banner("PHASE 1", "Full Backbone (all modules end-to-end)",
                 total_params, total_params,
                 cfg.n_train, conv_info, cfg.ph1_lr, cfg.eval_every, cfg.train_batch_size, cfg.grad_accum_steps)
    r1 = run_phase1(pipeline, mose_cfg, tok, datasets, hp1, device, run_dir)
    results["phases"]["phase1"] = _phase_result_dict(r1)
    flush_phase_metrics(cfg.name, "phase1", run_dir / "phase1_monitor.jsonl", r1.elapsed_s)
    log(f"Phase 1 done: {r1.summary_line()}")
    log(vram_report())

    # ── Phase 2: Motor Specialization ──
    hp2 = to_hparams(cfg, cfg.ph2_lr, cfg.ph2_warmup)
    motor_names = list(datasets.keys()) if datasets else ["cora"]
    phase_banner("PHASE 2", f"Motor Specialization ({len(motor_names)} motors)",
                 0, total_params,
                 cfg.n_train, conv_info, cfg.ph2_lr, cfg.eval_every, cfg.train_batch_size, cfg.grad_accum_steps)
    r2_list = run_phase2(pipeline, mose_cfg, tok, datasets, hp2, device, run_dir)
    results["phases"]["phase2"] = [
        {"motor": r.extra.get("motor", "?"), "steps": r.steps_run,
         "loss": r.final_loss, "time": r.elapsed_s, "stop": r.stop_reason}
        for r in r2_list
    ]
    for r in r2_list:
        motor = r.extra.get("motor", "unknown")
        flush_phase_metrics(cfg.name, f"phase2_{motor}",
                           run_dir / f"phase2_{motor}_monitor.jsonl", r.elapsed_s)
        log(f"Phase 2 [{motor}]: {r.summary_line()}")
    log(vram_report())

    # ── Phase 3: E2E + Orchestrator ──
    hp3 = to_hparams(cfg, cfg.ph3_e2e_lr, cfg.ph3_warmup)
    phase_banner("PHASE 3", "Orchestrator routing + E2E fine-tune",
                 total_params, total_params,
                 cfg.n_train, conv_info, cfg.ph3_e2e_lr, cfg.eval_every, cfg.train_batch_size, cfg.grad_accum_steps)
    r3 = run_phase3(pipeline, mose_cfg, tok, datasets, hp3, device, run_dir)
    results["phases"]["phase3"] = _phase_result_dict(r3)
    flush_phase_metrics(cfg.name, "phase3", run_dir / "phase3_e2e_monitor.jsonl", r3.elapsed_s)
    log(f"Phase 3 done: {r3.summary_line()}")
    log(vram_report())

    # ── Phase 4: Instruction Tuning (LoRA) ──
    # IT data is longer (median 134 tokens) — use seq=512, batch=1
    hp4 = to_hparams(cfg, cfg.ph4_lr, cfg.ph3_warmup)
    hp4.max_seq = 512
    hp4.train_batch_size = 1
    hp4.grad_accum_steps = 1
    phase_banner("PHASE 4", "Instruction Tuning with LoRA rank=16 (seq=512, bs=1)",
                 0, total_params,
                 28500, conv_info, cfg.ph4_lr, cfg.eval_every, 1, 1)
    r4 = run_phase4(pipeline, mose_cfg, tok, hp4, device, run_dir, interactive=False)
    results["phases"]["phase4"] = _phase_result_dict(r4)
    log(f"Phase 4 done: {r4.summary_line()}")
    log(vram_report())

    elapsed = time.perf_counter() - t_start
    results["total_time_s"] = elapsed
    results["total_time_min"] = elapsed / 60

    # Save config_name in checkpoints for run_local.py
    for ckpt_name in ["phase4_instruction.pt", "phase3_final.pt"]:
        ckpt_path = run_dir / ckpt_name
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            ckpt["config_name"] = cfg.name
            torch.save(ckpt, str(ckpt_path))

    # Generate sample responses
    log("Generating sample responses...")
    results["samples"] = generate_samples(pipeline, tok, device, cfg.max_seq)

    # Cleanup
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"Config {cfg.name} completed in {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
    return results


def _phase_result_dict(r) -> Dict:
    return {
        "steps": r.steps_run, "loss": r.final_loss,
        "val_loss": r.best_val_loss, "f1": r.best_val_f1,
        "time_s": r.elapsed_s, "time_min": r.elapsed_s / 60,
        "stop_reason": r.stop_reason,
    }


# ============================================================================
# EVAL QUESTIONS
# ============================================================================

EVAL_QUESTIONS = [
    ("Quien eres?", "identity"),
    ("What architecture do you use?", "identity"),
    ("Escribe una funcion en Python que calcule el factorial", "code"),
    ("Find the bug: def add(a,b): return a-b", "code"),
    ("Resuelve: 2x + 5 = 17", "math"),
    ("Calculate 347 + 892", "math"),
    ("Explica por que el cielo es azul", "reasoning"),
    ("What would happen if gravity disappeared?", "reasoning"),
    ("Como le digo a mi jefe que no estoy de acuerdo?", "social"),
    ("How do I give negative feedback to a colleague?", "social"),
]


def generate_samples(pipeline, tok, device, max_seq: int) -> List[Dict]:
    from experiments.train_production import _encode
    pipeline.eval()
    samples = []
    for q, category in EVAL_QUESTIONS:
        ids = _encode(tok, q, max_seq)
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = pipeline(ids_t)
        pred_ids = out.logits[0].argmax(dim=-1).tolist()
        try:
            resp = tok.decode(pred_ids)
        except Exception:
            resp = str(pred_ids[:20])
        samples.append({"question": q, "category": category, "response": resp[:300]})
        log(f"  [{category:>10}] Q: {q[:40]:<40}  A: {resp[:60]}")
    return samples


# ============================================================================
# QUANTIZATION
# ============================================================================

def quantize_best_model(output_dir: Path) -> Optional[str]:
    """Quantize the best available model to INT4."""
    from inference.quantize import quantize_state_dict

    # Find best checkpoint
    for config_name in ["production", "medium", "tiny"]:
        cfg_dir = output_dir / config_name
        for ckpt_name in ["phase4_instruction.pt", "phase3_final.pt"]:
            p = cfg_dir / ckpt_name
            if p.exists():
                log(f"Quantizing {config_name} model to INT4...")
                ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
                sd = ckpt.get("model_state", ckpt)
                quantized, stats = quantize_state_dict(sd, group_size=128)
                q_ratio = stats["quantized"] / max(1, stats["total_params"])
                log(f"Quantized {stats['quantized']:,} params ({q_ratio:.0%})")
                out_path = cfg_dir / f"{config_name}_int4.pt"
                torch.save({"quantized_state": quantized, "config_name": config_name}, str(out_path))
                log(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
                return str(out_path)
    log("No checkpoint found for quantization", "WARN")
    return None


# ============================================================================
# COMPARISON TABLE
# ============================================================================

def print_comparison(all_results: List[Dict], output_dir: Path):
    log("")
    log("=" * 85)
    log("  FINAL COMPARISON TABLE")
    log("=" * 85)

    header = (f"{'Config':<12} {'Params':>10} "
              f"{'Ph0':>8} {'Ph1':>8} {'Ph3':>8} {'Ph4':>8} "
              f"{'Steps':>8} {'Time':>8} {'Stop':>12}")
    log(header)
    log("-" * 85)

    for r in all_results:
        if r.get("error"):
            log(f"{r['config']:<12} FAILED: {r['error'][:50]}")
            continue
        ph = r["phases"]
        ph0_loss = ph.get("phase0", {}).get("loss", float("nan"))
        ph1_loss = ph.get("phase1", {}).get("loss", float("nan"))
        ph3_loss = ph.get("phase3", {}).get("loss", float("nan"))
        ph4_loss = ph.get("phase4", {}).get("loss", float("nan"))
        total_steps = sum(
            p.get("steps", 0) for k, p in ph.items()
            if isinstance(p, dict)
        )
        # Phase 2 is a list
        if isinstance(ph.get("phase2"), list):
            total_steps += sum(m.get("steps", 0) for m in ph["phase2"])
        time_str = f"{r['total_time_min']:.0f}m"
        stop = ph.get("phase4", {}).get("stop_reason", "?")
        log(f"{r['config']:<12} {r['params']/1e6:>8.1f}M "
            f"{ph0_loss:>8.3f} {ph1_loss:>8.3f} {ph3_loss:>8.3f} {ph4_loss:>8.3f} "
            f"{total_steps:>8,} {time_str:>8} {stop:>12}")

    log("")
    log("Sample responses from best model:")
    log("-" * 85)
    best = all_results[-1] if all_results else None
    if best and "samples" in best:
        for s in best["samples"]:
            log(f"  [{s['category']:>10}] Q: {s['question'][:50]}")
            log(f"             A: {s['response'][:120]}")
    log("=" * 85)

    summary_path = output_dir / "training_results.json"
    with open(str(summary_path), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    log(f"Results saved: {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_global = time.perf_counter()

    # Create output dir early so TeeLogger can write
    output_dir = _ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Install TeeLogger: all output goes to stdout + output/training_log.txt
    tee = install_tee_logger(output_dir)

    # Initialize central metrics log
    init_metrics_log(output_dir)

    print("=" * 70)
    print("  AION-C H200 Training Script")
    print("  Trains until CONVERGENCE -- no time limits")
    print("  Budget: up to $30 (~6h H200)")
    print(f"  Log: {output_dir / 'training_log.txt'}")
    print(f"  Metrics: {output_dir / 'training_metrics.jsonl'}")
    print("=" * 70)
    print()

    # GPU detection
    gi = gpu_info()
    if gi["available"]:
        log(f"GPU: {gi['name']}")
        log(f"VRAM: {gi['vram_gb']:.1f} GB")
        log(f"BF16: {gi['bf16']}")
        device = torch.device("cuda")

        if gi["vram_gb"] < 6:
            log("VRAM < 6 GB. Only tiny will run.", "WARN")
            configs_to_run = ["tiny"]
        elif gi["vram_gb"] < 20:
            log("VRAM < 20 GB. Medium + tiny will run.", "WARN")
            configs_to_run = ["medium", "tiny"]
        else:
            log("VRAM >= 20 GB. All 3 configs: production first.")
            configs_to_run = ["production", "medium", "tiny"]
    else:
        log("No GPU. Running tiny on CPU.", "WARN")
        device = torch.device("cpu")
        configs_to_run = ["tiny"]

    log(f"Output: {output_dir}")

    ds_root = find_opus_datasets()
    log(f"Opus datasets: {ds_root or 'NOT FOUND (synthetic fallback)'}")

    ensure_instruction_dataset()

    # Print training plan
    log("")
    log("TRAINING PLAN (quality-driven, no step limits):")
    for name in configs_to_run:
        cfg = CONFIGS[name]
        log(f"  {name:>12}: patience={cfg.patience}, "
            f"conv_delta={cfg.conv_delta}, conv_window={cfg.conv_window}, "
            f"eval_every={cfg.eval_every}")
    log("")

    # Train each config
    all_results = []
    for config_name in configs_to_run:
        cfg = CONFIGS[config_name]
        try:
            results = train_one_config(cfg, device, output_dir, ds_root)
            all_results.append(results)
        except Exception as e:
            log(f"Config {config_name} FAILED: {e}", "ERROR")
            traceback.print_exc()
            all_results.append({
                "config": config_name, "params": 0,
                "phases": {}, "error": str(e),
                "total_time_s": 0, "total_time_min": 0,
            })

    # Quantize best model
    try:
        quantize_best_model(output_dir)
    except Exception as e:
        log(f"Quantization failed: {e}", "WARN")

    # Final comparison
    print_comparison(all_results, output_dir)

    total = time.perf_counter() - t_global
    log(f"\nTotal time: {total/60:.1f} min ({total/3600:.2f} hours)")
    log(f"Output: {output_dir}")

    print()
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print()
    print("  Outputs:")
    print(f"    Log:     {output_dir / 'training_log.txt'}")
    print(f"    Metrics: {output_dir / 'training_metrics.jsonl'}")
    print(f"    Results: {output_dir / 'training_results.json'}")
    print()
    print("  To chat with AION-C:")
    print("    python -m inference.run_local")
    print()
    print("  To chat with quantized model:")
    print("    python -m inference.run_local --quantized \\")
    print(f"      --checkpoint {output_dir}/production/production_int4.pt")
    print("=" * 70)

    # Restore stdout and close log file
    if _tee is not None:
        sys.stdout = _tee._stdout
        _tee.close()
        # Print final message to real stdout (already restored)
        print(f"\nFull log saved: {output_dir / 'training_log.txt'}")


if __name__ == "__main__":
    main()
