"""
experiments/benchmark_sequential.py — Benchmark Motor-Sequential Training.

Corre 50 steps por cada fase para medir throughput real.

Fases medidas:
    1. Phase 1 backbone   (encoder + decoder + unifier trainable)
    2. Phase 2 motor cora  (solo motor cora trainable)
    3. Phase 2 motor forge_c (solo forge_c trainable)
    4. Phase 3 orchestrator (solo router)
    5. Phase 4 adapters   (solo LoRA adapters)

Reporta sps por fase + comparación con baseline full training.

Uso:
    python -m experiments.benchmark_sequential --config 1b --steps 50
    python -m experiments.benchmark_sequential --config small_300m --steps 30
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from experiments.benchmark_local import build_pipeline, make_fake_batch
from experiments.hw_monitor import HWMonitor
from training.sequential_trainer import (
    SequentialConfig,
    SequentialTrainer,
    PhaseResult,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ════════════════════════════════════════════════════════════════════════════
# Fake dataset
# ════════════════════════════════════════════════════════════════════════════

def make_data_fn(vocab_size: int, seq_len: int, batch: int = 1):
    """Returns a function that yields (token_ids, motor_idx) deterministically."""
    import random as _random
    rng = _random.Random(42)
    def _fn():
        ids = make_fake_batch(vocab_size, seq_len, batch=batch)
        motor_idx = rng.randint(0, 4)
        return ids, motor_idx
    return _fn


# ════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    model_config: str = "1b",
    n_steps: int = 50,
    vocab_size: int = 32000,
    seq_len: int = 64,
    phases_to_run: List[str] = None,
) -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mon = HWMonitor()
    mon.warmup()
    snap_before = mon.snapshot()

    print("=" * 78)
    print(f"MOTOR-SEQUENTIAL BENCHMARK — {model_config} config, {n_steps} steps/phase")
    print("=" * 78)
    print(f"Hardware: {mon.summary(snap_before)}")
    print()

    # Build pipeline once
    print(f"Building {model_config} pipeline...")
    pipeline, cfg = build_pipeline(model_config, vocab_size=vocab_size)
    total_params = sum(p.numel() for p in pipeline.parameters())
    print(f"  total params: {total_params:,} ({total_params/1e6:.1f}M)")
    print()

    trainer = SequentialTrainer(
        pipeline,
        SequentialConfig(device="cpu", log_every=max(n_steps // 5, 1)),
    )

    data_fn = make_data_fn(vocab_size, seq_len)

    results: List[PhaseResult] = []
    phases_requested = phases_to_run or ["p1", "p2_cora", "p2_forge_c", "p3", "p4"]

    try:
        if "p1" in phases_requested:
            r = trainer.run_phase_1_backbone(data_fn, n_steps=n_steps)
            results.append(r)
            gc.collect()

        if "p2_cora" in phases_requested:
            r = trainer.run_phase_2_motor("cora", data_fn, n_steps=n_steps)
            results.append(r)
            gc.collect()

        if "p2_forge_c" in phases_requested:
            r = trainer.run_phase_2_motor("forge_c", data_fn, n_steps=n_steps)
            results.append(r)
            gc.collect()

        if "p3" in phases_requested:
            r = trainer.run_phase_3_orchestrator(data_fn, n_steps=n_steps)
            results.append(r)
            gc.collect()

        if "p4" in phases_requested:
            r = trainer.run_phase_4_adapters(data_fn, n_steps=n_steps)
            results.append(r)
            gc.collect()
    except Exception as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()

    snap_after = mon.snapshot()

    # Build summary
    summary = {
        "model_config": model_config,
        "total_params": total_params,
        "n_steps_per_phase": n_steps,
        "seq_len": seq_len,
        "hw_before": snap_before.to_dict(),
        "hw_after": snap_after.to_dict(),
        "phases": [r.to_dict() for r in results],
        "host": mon.summary(snap_before),
    }

    out_path = RESULTS_DIR / "benchmark_sequential.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


# ════════════════════════════════════════════════════════════════════════════
# Summary table
# ════════════════════════════════════════════════════════════════════════════

def print_summary_table(summary: Dict[str, Any]) -> None:
    print()
    print("=" * 88)
    print(f"RESULTS — {summary['model_config']} ({summary['total_params']/1e6:.1f}M params)")
    print("=" * 88)
    print()
    header = f"{'phase':<30} {'trainable':>16} {'pct':>7} {'sps':>8} {'s/step':>8} {'peak RSS':>10}"
    print(header)
    print("-" * len(header))
    for p in summary["phases"]:
        name = p["name"][:30]
        tr = f"{p['trainable_params']/1e6:.1f}M"
        pct = f"{p['trainable_pct']:.1f}%"
        sps = f"{p['sps']:.4f}"
        sps_sec = f"{p['seconds_per_step']:.2f}"
        rss = f"{p['peak_rss_gb']:.1f}GB" if p.get('peak_rss_gb') else "n/a"
        print(f"{name:<30} {tr:>16} {pct:>7} {sps:>8} {sps_sec:>8} {rss:>10}")

    print()
    print("-" * len(header))

    # Per-phase extrapolations
    phase_steps = {
        "phase_1_backbone":                2000,
        "phase_2_motor:cora":              1500,
        "phase_2_motor:forge_c":           1500,
        "phase_2_motor:axiom":             1500,
        "phase_2_motor:muse":              1000,
        "phase_2_motor:empathy":           1000,
        "phase_3_orchestrator":             500,
        "phase_4_adapters":                 500,
    }
    # Use per-phase measured sps; for phases we didn't measure, reuse cora's sps
    sps_by_name: Dict[str, float] = {p["name"]: p["sps"] for p in summary["phases"]}
    default_motor_sps = None
    for p in summary["phases"]:
        if p["name"].startswith("phase_2_motor:") and p["sps"] > 0:
            default_motor_sps = p["sps"]
            break

    print("EXTRAPOLATED TIME FOR FULL SEQUENTIAL TRAINING:")
    print()
    total_sec = 0.0
    for name, n in phase_steps.items():
        sps = sps_by_name.get(name)
        if sps is None:
            if name.startswith("phase_2_motor:") and default_motor_sps:
                sps = default_motor_sps
            else:
                continue
        if sps <= 0:
            continue
        sec = n / sps
        total_sec += sec
        print(f"  {name:<30} {n:>6} steps @ {sps:.3f} sps = {sec/60:.1f} min")
    print("-" * 60)
    print(f"  {'TOTAL':<30} {'':>6}          {total_sec/60:.1f} min ({total_sec/3600:.2f} h)")
    print()

    # Decision
    ok_motor_results = [p for p in summary["phases"]
                        if p["name"].startswith("phase_2_motor:")]
    if ok_motor_results:
        min_motor_sps = min(p["sps"] for p in ok_motor_results)
        print("-" * len(header))
        if min_motor_sps >= 0.3:
            print(f"  OK: motor phase sps {min_motor_sps:.3f} >= 0.3 target -> LOCAL training viable")
        else:
            print(f"  WARN: motor phase sps {min_motor_sps:.3f} < 0.3 target -> consider Vast.ai")


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=("tiny", "small_300m", "1b"), default="1b")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--phases", type=str, default=None,
                   help="Comma-separated phases: p1,p2_cora,p2_forge_c,p3,p4")
    args = p.parse_args()

    phases = args.phases.split(",") if args.phases else None
    summary = run_benchmark(
        model_config=args.config,
        n_steps=args.steps,
        seq_len=args.seq_len,
        phases_to_run=phases,
    )
    print_summary_table(summary)
    print()
    print(f"Full report: {RESULTS_DIR / 'benchmark_sequential.json'}")


if __name__ == "__main__":
    main()
