"""
tests/test_experiments_fase_f.py — Parte 21 (5 experimentos de validación).

Corre cada experimento en tamaño reducido + el runner completo, y verifica
que todos producen un ExperimentReport válido y passed=True.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.fase_f import (
    exp1_sequential_learning,
    exp2_cross_domain,
    exp3_stress,
    exp4_compositional,
    exp5_self_evaluation,
)
from experiments.fase_f.common import ExperimentReport, write_report
from experiments.fase_f.run_all import run_all, summarize, EXPERIMENTS


# ════════════════════════════════════════════════════════════════════════════
# Experimentos individuales (reducidos para velocidad)
# ════════════════════════════════════════════════════════════════════════════

class TestIndividualExperiments:
    def test_exp1_sequential_learning(self):
        report = exp1_sequential_learning.run(n_concepts=20, check_every=5)
        assert isinstance(report, ExperimentReport)
        assert report.passed is True
        assert report.metrics["min_exam_pass_rate"] == 1.0
        assert report.metrics["checkpoints"] == 4

    def test_exp2_cross_domain(self):
        report = exp2_cross_domain.run()
        assert report.passed is True
        assert report.metrics["min_exam_pass_rate"] == 1.0
        assert len(report.details["per_motor_pass_rate"]) == 5

    def test_exp3_stress_small(self):
        report = exp3_stress.run(n=200)
        assert report.passed is True
        assert report.metrics["all_six_phases"] is True
        assert report.metrics["prune_stats"]["total"] == 200
        # Debe haber algún cluster (python, rust, hola son recurrentes)
        assert report.metrics["compress_clusters"] >= 1

    def test_exp4_compositional(self):
        report = exp4_compositional.run()
        assert report.passed is True
        assert report.metrics["accuracy"] >= 0.75

    def test_exp5_self_evaluation(self):
        report = exp5_self_evaluation.run()
        assert report.passed is True
        assert report.metrics["accuracy"] >= 0.75


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

class TestRunner:
    def test_run_all_registers_five(self):
        assert len(EXPERIMENTS) == 5

    def test_run_all_summary_shape(self):
        # Overriding n para exp1/exp3 sería ideal, pero run_all usa defaults.
        # En su lugar, construimos un mini-runner con las mismas funciones
        # pero con argumentos reducidos.
        reports = []
        reports.append(exp1_sequential_learning.run(n_concepts=10, check_every=5))
        reports.append(exp2_cross_domain.run())
        reports.append(exp3_stress.run(n=100))
        reports.append(exp4_compositional.run())
        reports.append(exp5_self_evaluation.run())
        summary = summarize(reports)
        assert summary["n_experiments"] == 5
        assert summary["n_passed"] == 5
        assert summary["all_passed"] is True
        assert summary["total_ms"] > 0


# ════════════════════════════════════════════════════════════════════════════
# Write report side-effect
# ════════════════════════════════════════════════════════════════════════════

class TestWriteReport:
    def test_write_creates_json(self, tmp_path: Path):
        import json
        r = ExperimentReport(
            experiment_id="test_dummy",
            name="dummy",
            started_at=0.0,
            ended_at=1.0,
            passed=True,
            summary="ok",
        )
        out = write_report(r, results_dir=tmp_path)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["experiment_id"] == "test_dummy"
        assert data["duration_ms"] == 1000.0
