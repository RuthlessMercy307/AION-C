"""
run_all.py — Ejecuta los 5 experimentos de Parte 21 en orden y reporta.

Uso:
    python -m experiments.fase_f.run_all

Genera experiments/fase_f/results/*.json + un run_all_summary.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from experiments.fase_f import (
    exp1_sequential_learning,
    exp2_cross_domain,
    exp3_stress,
    exp4_compositional,
    exp5_self_evaluation,
)
from experiments.fase_f.common import ExperimentReport, RESULTS_DIR, write_report


EXPERIMENTS = [
    exp1_sequential_learning.run,
    exp2_cross_domain.run,
    exp3_stress.run,
    exp4_compositional.run,
    exp5_self_evaluation.run,
]


def run_all() -> List[ExperimentReport]:
    reports: List[ExperimentReport] = []
    for fn in EXPERIMENTS:
        report = fn()
        write_report(report)
        reports.append(report)
        print(f"{report.experiment_id}: passed={report.passed} — {report.summary}")
    return reports


def summarize(reports: List[ExperimentReport]) -> dict:
    return {
        "started_at": min(r.started_at for r in reports),
        "ended_at": max(r.ended_at for r in reports),
        "total_ms": sum(r.duration_ms for r in reports),
        "n_experiments": len(reports),
        "n_passed": sum(1 for r in reports if r.passed),
        "all_passed": all(r.passed for r in reports),
        "reports": [r.to_dict() for r in reports],
    }


def main() -> None:
    reports = run_all()
    summary = summarize(reports)
    out = RESULTS_DIR / "run_all_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"{summary['n_passed']}/{summary['n_experiments']} experiments passed")
    print(f"summary: {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
