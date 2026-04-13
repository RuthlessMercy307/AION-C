"""
experiments/benchmark_local.py — Benchmark de throughput real en hardware local.

Corre 3 configuraciones sobre el mismo modelo (default 1.1B, ajustable con
--config) por un número corto de steps (default 50) y reporta throughput,
uso de memoria, y extrapolación a N steps completos.

Configuraciones:
    A: "gpu_chunk_5"   — device GPU (cuda/dml), chunk_size=5
    B: "gpu_chunk_10"  — device GPU, chunk_size=10
    C: "cpu_pure"      — device CPU, sin offload (baseline)

Uso:
    python -m experiments.benchmark_local               # 1b, 50 steps each
    python -m experiments.benchmark_local --config tiny # sanity check
    python -m experiments.benchmark_local --config 1b --steps 30

Genera:
    - Output en consola con tabla comparativa
    - experiments/results/benchmark_local.json con datos crudos
    - Extrapolación a 5000 / 15000 steps

No arranca training real; es sólo medición.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.hw_monitor import HWMonitor
from training.chunked_trainer import (
    ChunkedTrainer,
    ChunkedTrainerConfig,
    resolve_device,
    device_label,
)


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"


# ════════════════════════════════════════════════════════════════════════════
# Pipeline builder
# ════════════════════════════════════════════════════════════════════════════

def build_pipeline(config: str, vocab_size: int):
    """Construye el pipeline en CPU. Soporta tiny / small_300m / 1b."""
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
    elif config == "small_300m":
        cfg = MoSEConfig(
            hidden_dim=512, vocab_size=vocab_size,
            enc_n_layers=6, enc_state_dim=8, enc_expand=2, enc_d_conv=4, enc_ffn_mult=4,
            orch_mlp_hidden=256, orch_max_motors=3, orch_min_confidence=0.3,
            motor_max_nodes=8, motor_n_heads=8, motor_threshold=0.01, unif_n_heads=8,
            dec_n_layers=8, dec_n_heads=8, dec_max_seq_len=256,
            dec_state_dim=8, dec_expand=2, dec_d_conv=4, dec_ffn_mult=4,
        )
    elif config == "1b":
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
    pipeline = MoSEPipeline(cfg)
    return pipeline, cfg


# ════════════════════════════════════════════════════════════════════════════
# Benchmark result models
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ConfigResult:
    name: str
    device_spec: str
    device_label: str
    chunk_size: Optional[int]
    ok: bool
    error: Optional[str] = None
    n_steps_completed: int = 0
    total_seconds: float = 0.0
    sps: float = 0.0                   # steps per second
    seconds_per_step: float = 0.0
    mean_loss: float = 0.0
    hw_before: Dict[str, Any] = field(default_factory=dict)
    hw_after: Dict[str, Any] = field(default_factory=dict)
    hw_peak: Dict[str, Any] = field(default_factory=dict)
    # Extrapolation
    est_5000_steps_min: float = 0.0
    est_15000_steps_min: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    model_config: str
    model_params: int
    n_steps: int
    timestamp: float
    host_summary: str
    results: List[ConfigResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ════════════════════════════════════════════════════════════════════════════
# Fake tokenizer for the benchmark (no dependency on disk)
# ════════════════════════════════════════════════════════════════════════════

def make_fake_batch(vocab_size: int, seq_len: int, batch: int = 1) -> torch.Tensor:
    """Generates a random token_ids tensor for benchmark purposes.

    Avoids the overhead of loading the real tokenizer and dataset. The
    measured throughput is the PIPELINE's compute cost, not the
    dataloader's.
    """
    g = torch.Generator().manual_seed(42)
    return torch.randint(1, vocab_size - 1, (batch, seq_len), generator=g, dtype=torch.long)


# ════════════════════════════════════════════════════════════════════════════
# One-config runner
# ════════════════════════════════════════════════════════════════════════════

def run_one_config(
    name: str,
    device_spec: str,
    chunk_size: Optional[int],
    model_config: str,
    vocab_size: int,
    seq_len: int,
    n_steps: int,
    verbose: bool = True,
) -> ConfigResult:
    """Ejecuta una configuración del benchmark y devuelve el ConfigResult."""
    print()
    print("=" * 70)
    print(f"[{name}] device={device_spec} chunk_size={chunk_size}")
    print("=" * 70)

    result = ConfigResult(
        name=name,
        device_spec=device_spec,
        device_label="?",
        chunk_size=chunk_size,
        ok=False,
    )
    mon = HWMonitor()
    mon.warmup()

    # Snapshot BEFORE
    time.sleep(0.3)
    snap_before = mon.snapshot()
    result.hw_before = snap_before.to_dict()
    print(f"  HW before: {mon.summary(snap_before)}")

    peak_rss = snap_before.proc_rss_gb or 0.0
    peak_vram = snap_before.gpu_vram_used_gb or 0.0

    try:
        # Build pipeline
        pipeline, cfg = build_pipeline(model_config, vocab_size=vocab_size)
        n_params = sum(p.numel() for p in pipeline.parameters())
        print(f"  model params: {n_params:,} ({n_params/1e6:.1f}M)")

        # Build trainer — this moves the pipeline to the target device.
        # The optimizer MUST be created AFTER this so its param_groups
        # reference the device-resident tensors, not stale CPU ones.
        trainer_cfg = ChunkedTrainerConfig(
            device=device_spec,
            chunk_size=chunk_size or 10,
            offload_every_step=False,
            amp=False,
        )
        trainer = ChunkedTrainer(pipeline, trainer_cfg)
        result.device_label = trainer.device_label
        print(f"  device: {trainer.device_label}")
        print(f"  chunks: {len(trainer.chunks)} (informational split)")

        # Optimizer: AFTER the pipeline is on device, so param_groups
        # reference the correct tensors.
        optimizer = torch.optim.AdamW(pipeline.parameters(), lr=1e-4)

        # Fake data batch
        batch = make_fake_batch(vocab_size, seq_len)

        # Warmup 1 step (not counted)
        print(f"  warmup step...")
        try:
            trainer.train_step(batch, optimizer)
        except Exception as exc:
            raise RuntimeError(f"warmup step failed: {exc}") from exc

        # Real measurement
        print(f"  running {n_steps} measurement steps...")
        losses: List[float] = []
        t_start = time.perf_counter()
        for step in range(1, n_steps + 1):
            info = trainer.train_step(batch, optimizer)
            losses.append(info["loss"])
            # Peak tracking via occasional snapshot
            if step % max(n_steps // 10, 1) == 0 or step == 1:
                snap = mon.snapshot()
                peak_rss = max(peak_rss, snap.proc_rss_gb or 0.0)
                if snap.gpu_vram_used_gb is not None:
                    peak_vram = max(peak_vram, snap.gpu_vram_used_gb)
                if verbose:
                    sps_so_far = step / max(time.perf_counter() - t_start, 1e-6)
                    print(f"    step {step}/{n_steps}: loss={info['loss']:.3f} "
                          f"sps={sps_so_far:.3f} "
                          f"rss={snap.proc_rss_gb or 0:.2f}GB")
        t_end = time.perf_counter()

        total = t_end - t_start
        result.n_steps_completed = n_steps
        result.total_seconds = round(total, 2)
        result.sps = round(n_steps / max(total, 1e-6), 4)
        result.seconds_per_step = round(total / max(n_steps, 1), 3)
        result.mean_loss = round(sum(losses) / max(len(losses), 1), 3)
        result.ok = True

        # Extrapolations
        if result.sps > 0:
            sec_per_step = 1.0 / result.sps
            result.est_5000_steps_min = round(5000 * sec_per_step / 60, 1)
            result.est_15000_steps_min = round(15000 * sec_per_step / 60, 1)

        # Cleanup
        del trainer, pipeline, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"  FAILED: {result.error}")
            traceback.print_exc()

    # Snapshot AFTER
    snap_after = mon.snapshot()
    result.hw_after = snap_after.to_dict()
    result.hw_peak = {
        "proc_rss_gb": round(peak_rss, 2),
        "vram_used_gb": round(peak_vram, 2) if peak_vram > 0 else None,
    }
    print(f"  HW after: {mon.summary(snap_after)}")
    print(f"  HW peak: rss={result.hw_peak['proc_rss_gb']}GB "
          f"vram={result.hw_peak['vram_used_gb']}")
    if result.ok:
        print(f"  RESULT: sps={result.sps} · est_5000={result.est_5000_steps_min}min "
              f"· est_15000={result.est_15000_steps_min}min")
    return result


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    model_config: str = "1b",
    n_steps: int = 50,
    vocab_size: int = 32000,
    seq_len: int = 64,
    configs_to_run: Optional[List[str]] = None,
) -> BenchmarkReport:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mon = HWMonitor()
    mon.warmup()
    time.sleep(0.3)
    host_snap = mon.snapshot()
    host_summary = mon.summary(host_snap)

    # Pipeline size estimate
    pipeline_tmp, _ = build_pipeline(model_config, vocab_size=vocab_size)
    model_params = sum(p.numel() for p in pipeline_tmp.parameters())
    del pipeline_tmp
    gc.collect()

    report = BenchmarkReport(
        model_config=model_config,
        model_params=model_params,
        n_steps=n_steps,
        timestamp=time.time(),
        host_summary=host_summary,
    )

    # Define the 3 configs (in order)
    all_configs = [
        ("A_gpu_chunk_5",  "auto",    5),
        ("B_gpu_chunk_10", "auto",    10),
        ("C_cpu_pure",     "cpu",     None),
    ]

    selected = configs_to_run or [c[0] for c in all_configs]
    for name, device, chunk in all_configs:
        if name not in selected:
            continue
        result = run_one_config(
            name=name,
            device_spec=device,
            chunk_size=chunk,
            model_config=model_config,
            vocab_size=vocab_size,
            seq_len=seq_len,
            n_steps=n_steps,
        )
        report.results.append(result)
        # Save after each config so we have partial results if a later one crashes
        _save_report(report)

    return report


def _save_report(report: BenchmarkReport) -> None:
    path = RESULTS_DIR / "benchmark_local.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8")


def print_summary_table(report: BenchmarkReport) -> None:
    print()
    print("=" * 78)
    print(f"BENCHMARK SUMMARY — {report.model_config} ({report.model_params/1e6:.1f}M params, {report.n_steps} steps)")
    print("=" * 78)
    print()
    header = f"{'config':<18} {'device':<20} {'sps':>8} {'s/step':>8} {'peak RSS':>10} {'peak VRAM':>10} {'5Ksteps':>10}"
    print(header)
    print("-" * len(header))
    for r in report.results:
        if not r.ok:
            status = f"FAIL: {r.error[:30] if r.error else '?'}"
            print(f"{r.name:<18} {r.device_label:<20} {status}")
            continue
        peak_rss = f"{r.hw_peak.get('proc_rss_gb', 0) or 0:.1f}GB"
        peak_vram = f"{r.hw_peak.get('vram_used_gb') or 0:.1f}GB" if r.hw_peak.get('vram_used_gb') else "n/a"
        est5k = f"{r.est_5000_steps_min:.0f}min"
        print(f"{r.name:<18} {r.device_label:<20} {r.sps:>8.4f} {r.seconds_per_step:>8.3f} {peak_rss:>10} {peak_vram:>10} {est5k:>10}")

    # Recommendation
    print()
    print("-" * len(header))
    ok_results = [r for r in report.results if r.ok and r.sps > 0]
    if not ok_results:
        print("  NO CONFIG SUCCESS. Check errors above.")
        return

    best = max(ok_results, key=lambda r: r.sps)
    print(f"  Best: {best.name} @ {best.sps:.4f} sps -> 5000 steps = {best.est_5000_steps_min:.0f} min")
    if best.sps < 0.05:
        print(f"  WARNING: best sps {best.sps:.4f} < 0.05 — local training is not viable.")
        print(f"  Consider Vast.ai ($1-4 for the full run).")


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=("tiny", "small_300m", "1b"), default="1b")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated config names to run (A_gpu_chunk_5,B_gpu_chunk_10,C_cpu_pure)")
    args = p.parse_args()

    only = args.only.split(",") if args.only else None
    report = run_benchmark(
        model_config=args.config,
        n_steps=args.steps,
        vocab_size=args.vocab,
        seq_len=args.seq_len,
        configs_to_run=only,
    )
    print_summary_table(report)
    path = RESULTS_DIR / "benchmark_local.json"
    print()
    print(f"Full report: {path}")


if __name__ == "__main__":
    main()
