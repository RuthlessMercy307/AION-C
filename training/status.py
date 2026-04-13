"""
training/status.py — Resumen humano del training en curso.

Lee metrics.jsonl del log más reciente y produce un reporte de ~10
líneas que contesta: qué fase corre, cuánto va, loss actual, ETA,
RAM/temp, y alertas pendientes del watchdog.

Uso:
    python training/status.py

    # En modo "quick" para copy-paste a un chat con Claude
    python training/status.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.monitoring.logger import MetricsLogger, find_latest_log_dir
from training.health_check import check_health


# Plan by default (matches DEFAULT_PLAN en train_1b_sequential.py)
DEFAULT_STEPS = {
    "phase_1_backbone":        1500,   # updated default
    "phase_2_motor:cora":      1500,
    "phase_2_motor:forge_c":   1500,
    "phase_2_motor:axiom":     1500,
    "phase_2_motor:muse":      1000,
    "phase_2_motor:empathy":   1000,
    "phase_3_orchestrator":     500,
    "phase_4_adapters":         500,
}


def fmt_duration(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.1f}h"


def read_control(log_dir: Path) -> Optional[Dict[str, Any]]:
    path = log_dir / "control.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_events_tail(log_dir: Path, n: int = 5) -> List[str]:
    path = log_dir / "events.log"
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        return lines[-n:]
    except OSError:
        return []


def build_report(log_dir: Path) -> str:
    metrics_path = log_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return f"(no metrics.jsonl in {log_dir})"

    logger = MetricsLogger(metrics_path)
    entries = logger.read_all()
    if not entries:
        return f"(metrics.jsonl is empty in {log_dir.name})"

    latest = entries[-1]
    step = latest.get("step", 0)
    phase = latest.get("phase", "?")
    loss = latest.get("loss", 0.0)
    sps = latest.get("sps", 0.0)
    ram = latest.get("ram_gb")
    cpu_temp = latest.get("cpu_temp")
    proc_rss = latest.get("proc_rss_gb")

    # ETA for this phase
    phase_target_steps = DEFAULT_STEPS.get(phase)
    eta_str = "?"
    phase_pct = "?"
    if phase_target_steps and sps > 0:
        remaining = max(0, phase_target_steps - step)
        eta_sec = remaining / sps
        eta_str = fmt_duration(eta_sec)
        phase_pct = f"{100 * step / phase_target_steps:.0f}%"

    # Health check
    health = check_health(log_dir)

    # Control file state
    control = read_control(log_dir)
    pending_action = None
    if control and control.get("action") and control.get("action") != "none":
        pending_action = control

    # Log dir summary
    first = entries[0]
    start_ts = first.get("timestamp", 0)
    elapsed_sec = time.time() - start_ts if start_ts else 0
    total_entries = len(entries)

    # Format report
    lines = []
    lines.append(f"-- TRAINING STATUS --")
    lines.append(f"  Log: {log_dir.name}")
    lines.append(f"  Phase: {phase} | step {step}/{phase_target_steps or '?'} ({phase_pct})")
    lines.append(f"  Loss: {loss:.4f} | sps: {sps:.3f} | ETA this phase: {eta_str}")
    ram_str = f"{ram:.1f}GB" if ram is not None else "?"
    rss_str = f"{proc_rss:.1f}GB" if proc_rss is not None else "?"
    temp_str = f"{cpu_temp:.0f}C" if cpu_temp is not None else "n/a"
    lines.append(f"  RAM: {ram_str} (proc {rss_str}) | CPU temp: {temp_str}")
    lines.append(f"  Elapsed: {fmt_duration(elapsed_sec)} | entries: {total_entries}")
    lines.append(f"  Health: {health.status}")
    if health.reasons:
        for r in health.reasons[:3]:
            lines.append(f"    ! {r}")
    if pending_action:
        lines.append(f"  PENDING ACTION: {pending_action.get('action')} "
                     f"(by {pending_action.get('author', '?')})")
        if pending_action.get("message"):
            lines.append(f"    msg: {pending_action['message']}")
    tail = read_events_tail(log_dir, n=3)
    if tail:
        lines.append(f"  Recent events:")
        for e in tail:
            lines.append(f"    {e}")
    return "\n".join(lines)


def build_quick_report(log_dir: Path) -> str:
    """One-liner style para pegar en chat."""
    metrics_path = log_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return "(no metrics yet)"

    logger = MetricsLogger(metrics_path)
    entries = logger.read_all()
    if not entries:
        return "(no entries)"
    latest = entries[-1]
    health = check_health(log_dir)

    step = latest.get("step", 0)
    phase = latest.get("phase", "?")
    loss = latest.get("loss", 0.0)
    sps = latest.get("sps", 0.0)
    ram = latest.get("ram_gb", 0)
    phase_total = DEFAULT_STEPS.get(phase, "?")
    return (f"[{health.status}] {phase} step {step}/{phase_total} "
            f"loss={loss:.3f} sps={sps:.3f} ram={ram:.1f}GB "
            f"({len(entries)} entries)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--logs-base", type=Path,
                   default=ROOT / "training" / "logs")
    p.add_argument("--quick", action="store_true",
                   help="One-line status for copy-paste")
    args = p.parse_args()

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = find_latest_log_dir(args.logs_base)
        if log_dir is None:
            print(f"No training logs found under {args.logs_base}")
            return

    if args.quick:
        print(build_quick_report(log_dir))
    else:
        print(build_report(log_dir))


if __name__ == "__main__":
    main()
