"""
training/health_check.py — Lee metrics.jsonl y reporta status.

Retorna JSON con:
    {
      "status": "HEALTHY" | "WARNING" | "CRITICAL",
      "reasons": ["...", "..."],
      "snapshot": { ultimo entry del log },
      "trend": { loss_trend, sps_trend, ram_trend }
    }

Reglas:
    CRITICAL si:
        - loss sube en 3 checkpoints seguidos
        - RAM > 14 GB
        - CPU temp > 85°C
        - sps cae > 50% vs baseline
        - routing_acc < 70%

    WARNING si:
        - loss estanca por 200 steps (8 checkpoints a 25 steps/check)
        - RAM > 12 GB
        - temp > 80°C
        - sps cae 20-50%

    HEALTHY si ninguna de las anteriores.

Uso:
    # Status del log más reciente bajo training/logs/
    python training/health_check.py

    # Status de un log específico
    python training/health_check.py --log-dir training/logs/sequential_20260411_100000

    # Solo el status code (para scripts): exit 0=HEALTHY 1=WARNING 2=CRITICAL
    python training/health_check.py --quiet
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.monitoring.logger import MetricsLogger, find_latest_log_dir


# ════════════════════════════════════════════════════════════════════════════
# Thresholds
# ════════════════════════════════════════════════════════════════════════════

class Thresholds:
    # CRITICAL
    RAM_CRITICAL_GB        = 14.0
    CPU_TEMP_CRITICAL_C    = 85.0
    SPS_DROP_CRITICAL_PCT  = 50.0   # % drop vs baseline
    ROUTING_ACC_CRITICAL   = 70.0   # %
    LOSS_RISES_CRITICAL    = 3      # rises in a row
    # WARNING
    RAM_WARNING_GB         = 12.0
    CPU_TEMP_WARNING_C     = 80.0
    SPS_DROP_WARNING_PCT   = 20.0
    LOSS_PLATEAU_STEPS     = 200    # steps with no improvement


# ════════════════════════════════════════════════════════════════════════════
# Status model
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class HealthReport:
    status: str  # HEALTHY | WARNING | CRITICAL
    reasons: List[str] = field(default_factory=list)
    snapshot: Dict[str, Any] = field(default_factory=dict)
    trend: Dict[str, Any] = field(default_factory=dict)
    log_dir: str = ""
    n_entries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def exit_code(self) -> int:
        return {"HEALTHY": 0, "WARNING": 1, "CRITICAL": 2}.get(self.status, 0)


# ════════════════════════════════════════════════════════════════════════════
# Main checker
# ════════════════════════════════════════════════════════════════════════════

def check_health(log_dir: Path) -> HealthReport:
    metrics_path = log_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return HealthReport(
            status="HEALTHY",
            reasons=["no metrics.jsonl yet — training hasn't started or is initializing"],
            log_dir=str(log_dir),
            n_entries=0,
        )

    logger = MetricsLogger(metrics_path)
    entries = logger.read_all()
    n = len(entries)

    if n == 0:
        return HealthReport(
            status="HEALTHY",
            reasons=["no metric entries yet"],
            log_dir=str(log_dir),
            n_entries=0,
        )

    latest = entries[-1]
    reasons_critical: List[str] = []
    reasons_warning: List[str] = []

    # ── RAM ─────────────────────────────────────────────────────────────
    ram_gb = latest.get("ram_gb")
    if ram_gb is not None:
        if ram_gb >= Thresholds.RAM_CRITICAL_GB:
            reasons_critical.append(f"RAM {ram_gb:.1f}GB >= {Thresholds.RAM_CRITICAL_GB}GB")
        elif ram_gb >= Thresholds.RAM_WARNING_GB:
            reasons_warning.append(f"RAM {ram_gb:.1f}GB >= {Thresholds.RAM_WARNING_GB}GB")

    # ── CPU temp ────────────────────────────────────────────────────────
    cpu_temp = latest.get("cpu_temp")
    if cpu_temp is not None:
        if cpu_temp >= Thresholds.CPU_TEMP_CRITICAL_C:
            reasons_critical.append(f"CPU temp {cpu_temp:.0f}C >= {Thresholds.CPU_TEMP_CRITICAL_C}C")
        elif cpu_temp >= Thresholds.CPU_TEMP_WARNING_C:
            reasons_warning.append(f"CPU temp {cpu_temp:.0f}C >= {Thresholds.CPU_TEMP_WARNING_C}C")

    # ── Routing accuracy ────────────────────────────────────────────────
    routing = latest.get("routing_acc")
    if routing is not None and routing < Thresholds.ROUTING_ACC_CRITICAL:
        reasons_critical.append(f"routing_acc {routing:.1f}% < {Thresholds.ROUTING_ACC_CRITICAL}%")

    # ── Loss trend: rising 3 times in a row? ────────────────────────────
    if n >= 4:
        recent_losses = [e.get("loss") for e in entries[-4:]
                         if e.get("loss") is not None]
        if len(recent_losses) >= 4:
            rising = all(recent_losses[i] < recent_losses[i + 1]
                         for i in range(3))
            if rising:
                reasons_critical.append(
                    f"loss rising 3x in a row: {recent_losses}")

    # ── Loss plateau ────────────────────────────────────────────────────
    # Plateau = no improvement in the last 200 steps (~8 checkpoints at 25 steps each)
    if n >= 10:
        window = entries[-10:]
        window_losses = [e.get("loss") for e in window if e.get("loss") is not None]
        if len(window_losses) >= 10:
            first = window_losses[0]
            last = window_losses[-1]
            if abs(first - last) < 0.01 and first > 1.0:
                first_step = window[0].get("step", 0)
                last_step = window[-1].get("step", 0)
                if last_step - first_step >= Thresholds.LOSS_PLATEAU_STEPS:
                    reasons_warning.append(
                        f"loss plateau: {first:.3f}->{last:.3f} over "
                        f"{last_step - first_step} steps")

    # ── SPS drop ────────────────────────────────────────────────────────
    # Compare current sps to baseline (mean of first 3 entries of this phase)
    current_phase = latest.get("phase", "")
    same_phase = [e for e in entries if e.get("phase") == current_phase]
    if len(same_phase) >= 5:
        baseline_sps_values = [e.get("sps") for e in same_phase[:3]
                               if e.get("sps") is not None and e.get("sps") > 0]
        current_sps = latest.get("sps")
        if baseline_sps_values and current_sps is not None and current_sps > 0:
            baseline = sum(baseline_sps_values) / len(baseline_sps_values)
            drop_pct = 100 * (baseline - current_sps) / baseline if baseline > 0 else 0
            if drop_pct >= Thresholds.SPS_DROP_CRITICAL_PCT:
                reasons_critical.append(
                    f"sps dropped {drop_pct:.0f}% ({baseline:.3f}->{current_sps:.3f})")
            elif drop_pct >= Thresholds.SPS_DROP_WARNING_PCT:
                reasons_warning.append(
                    f"sps dropped {drop_pct:.0f}% ({baseline:.3f}->{current_sps:.3f})")

    # ── Final status ────────────────────────────────────────────────────
    if reasons_critical:
        status = "CRITICAL"
    elif reasons_warning:
        status = "WARNING"
    else:
        status = "HEALTHY"

    # ── Trend dict ──────────────────────────────────────────────────────
    trend = {}
    if n >= 3:
        trend["loss_start"]   = entries[0].get("loss")
        trend["loss_latest"]  = latest.get("loss")
        trend["sps_latest"]   = latest.get("sps")
        trend["ram_latest"]   = latest.get("ram_gb")
        trend["step_latest"]  = latest.get("step")
        trend["phase_latest"] = latest.get("phase")

    return HealthReport(
        status=status,
        reasons=reasons_critical + reasons_warning,
        snapshot=latest,
        trend=trend,
        log_dir=str(log_dir),
        n_entries=n,
    )


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Specific log directory to check")
    p.add_argument("--logs-base", type=Path,
                   default=ROOT / "training" / "logs",
                   help="Base directory of all logs")
    p.add_argument("--quiet", action="store_true",
                   help="Only print status and exit with status code")
    p.add_argument("--json", action="store_true",
                   help="Print full report as JSON")
    args = p.parse_args()

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = find_latest_log_dir(args.logs_base)
        if log_dir is None:
            print("No training logs found under", args.logs_base)
            sys.exit(0)

    report = check_health(log_dir)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    elif args.quiet:
        print(report.status)
    else:
        print(f"Status: {report.status}")
        print(f"Log dir: {report.log_dir}")
        print(f"Entries: {report.n_entries}")
        if report.trend:
            print(f"Latest step: {report.trend.get('step_latest')} "
                  f"({report.trend.get('phase_latest')})")
            print(f"Latest loss: {report.trend.get('loss_latest')}")
            print(f"Latest sps:  {report.trend.get('sps_latest')}")
            print(f"Latest RAM:  {report.trend.get('ram_latest')} GB")
        if report.reasons:
            print()
            print("Reasons:")
            for r in report.reasons:
                print(f"  - {r}")

    sys.exit(report.exit_code())


if __name__ == "__main__":
    main()
