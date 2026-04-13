"""
experiments/fase_f/ — Experimentos de validación (Parte 21 del MEGA-PROMPT).

5 experimentos ejecutables que validan el bloque cognitivo (Partes 22-27)
sobre el tiny model local ANTES de escalar a Fase E.

    exp1 — Sequential learning test
    exp2 — Cross-domain test
    exp3 — Stress test (1000 iteraciones)
    exp4 — Compositional test
    exp5 — Self-evaluation test

Cada script genera un reporte JSON en experiments/fase_f/results/.
Un runner `run_all.py` los ejecuta todos en orden y produce un resumen.
"""

from experiments.fase_f.common import (
    ExperimentReport,
    FakeMotor,
    make_exam,
    write_report,
)

__all__ = [
    "ExperimentReport",
    "FakeMotor",
    "make_exam",
    "write_report",
]
