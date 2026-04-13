"""
exp5_self_evaluation.py — Parte 21.5 (self-evaluation test).

Pregunta científica:
    ¿El RewardEstimator distingue entre una buena y una mala respuesta usando
    únicamente señales? ¿La confianza intrínseca (entropía + simbólico) aporta
    cuando no hay señales explícitas?

Procedimiento:
    1. Definir escenarios con señales claramente positivas y negativas.
    2. Para cada escenario, calcular el reward y verificar que cae en la
       zona esperada (< 0.4 para malas, > 0.6 para buenas).
    3. Reportar confusion-matrix y accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

from experiments.fase_f.common import ExperimentReport, write_report
from reward import (
    ExplicitSignal,
    ImplicitSignals,
    IntrinsicSignals,
    RewardConfig,
    RewardEstimator,
    RewardSignals,
)


@dataclass
class Scenario:
    name: str
    signals: RewardSignals
    expected: str   # "good" | "bad" | "neutral"


def _scenarios() -> List[Scenario]:
    return [
        Scenario(
            "explicit_thumbs_up",
            RewardSignals(explicit=ExplicitSignal.UP),
            "good",
        ),
        Scenario(
            "explicit_thumbs_down",
            RewardSignals(explicit=ExplicitSignal.DOWN),
            "bad",
        ),
        Scenario(
            "explicit_correction",
            RewardSignals(explicit=ExplicitSignal.CORRECTION),
            "bad",
        ),
        Scenario(
            "implicit_no_correction_continue",
            RewardSignals(
                explicit=ExplicitSignal.NONE,
                implicit=ImplicitSignals(no_correction_continue=True, code_copied=True),
            ),
            "good",
        ),
        Scenario(
            "implicit_re_asked",
            RewardSignals(
                explicit=ExplicitSignal.NONE,
                implicit=ImplicitSignals(re_asked_similar=True),
            ),
            "bad",
        ),
        Scenario(
            "intrinsic_symbolic_inconsistent",
            RewardSignals(
                explicit=ExplicitSignal.NONE,
                intrinsic=IntrinsicSignals(
                    token_entropy_mean=1.5,
                    symbolic_consistent=False,
                    unifier_agreement=0.3,
                ),
            ),
            "bad",
        ),
        Scenario(
            "intrinsic_strong_agreement",
            RewardSignals(
                explicit=ExplicitSignal.NONE,
                intrinsic=IntrinsicSignals(
                    token_entropy_mean=0.1,
                    symbolic_consistent=True,
                    unifier_agreement=1.0,
                ),
                implicit=ImplicitSignals(no_correction_continue=True),
            ),
            "good",
        ),
        Scenario(
            "pure_neutral",
            RewardSignals(),
            "neutral",
        ),
    ]


def _classify(mean: float) -> str:
    if mean > 0.58:
        return "good"
    if mean < 0.42:
        return "bad"
    return "neutral"


def run() -> ExperimentReport:
    start = time.time()
    estimator = RewardEstimator(RewardConfig())
    results: List[Dict[str, Any]] = []
    confusion = {
        "good": {"good": 0, "bad": 0, "neutral": 0},
        "bad":  {"good": 0, "bad": 0, "neutral": 0},
        "neutral": {"good": 0, "bad": 0, "neutral": 0},
    }
    for s in _scenarios():
        est = estimator.compute(s.signals)
        got = _classify(est.mean)
        results.append({
            "scenario": s.name,
            "expected": s.expected,
            "got": got,
            "mean": est.mean,
            "std": est.std,
            "conservative": est.conservative,
        })
        confusion[s.expected][got] += 1

    n_correct = sum(confusion[k][k] for k in confusion)
    total = len(results)
    accuracy = n_correct / total if total else 0.0
    passed = accuracy >= 0.75

    end = time.time()
    return ExperimentReport(
        experiment_id="exp5_self_evaluation",
        name="Self-evaluation test",
        started_at=start,
        ended_at=end,
        passed=passed,
        summary=f"{n_correct}/{total} scenarios classified correctly (acc={accuracy:.2f})",
        metrics={
            "total_cases": total,
            "correct": n_correct,
            "accuracy": accuracy,
        },
        details={
            "confusion_matrix": confusion,
            "scenarios": results,
        },
    )


def main() -> None:
    report = run()
    write_report(report)
    print(f"{report.experiment_id}: passed={report.passed} — {report.summary}")


if __name__ == "__main__":  # pragma: no cover
    main()
