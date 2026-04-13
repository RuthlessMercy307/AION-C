"""
exp4_compositional.py — Parte 21.4 (compositional test).

Pregunta científica:
    ¿El TrajectoryPlanner descompone correctamente queries compuestas en
    secuencias de motores coherentes? ¿El CompositeOrchestrator ejecuta en
    el orden correcto y pasa outputs previos a los pasos dependientes?

Procedimiento:
    1. Un conjunto de queries con secuencia esperada (ground truth manual).
    2. Planear cada query y comparar con la esperada.
    3. Ejecutar cada una con un stub generate_fn que identifica pasos y
       valida que los prompts contienen los outputs previos.
    4. Reportar accuracy por caso y global.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from composition import (
    CompositeOrchestrator,
    TrajectoryPlanner,
)
from experiments.fase_f.common import ExperimentReport, write_report


CASES: List[Dict[str, Any]] = [
    {
        "query": "escribe una función python que sume dos números",
        "expected": ["forge_c"],
    },
    {
        "query": "me siento triste hoy",
        "expected": ["empathy"],
    },
    {
        "query": "calcula el 15% de 240",
        "expected": ["axiom"],
    },
    {
        "query": "por qué llueve en invierno",
        "expected": ["cora"],
    },
    {
        "query": "escribe un poema sobre el mar",
        "expected": ["muse"],
    },
    {
        "query": "explica este código como cuento",
        "expected": ["forge_c", "muse"],
    },
    {
        "query": "cuéntame este teorema como poema",
        "expected_last": "muse",  # último step debe ser muse
    },
    {
        "query": "explica python y rust, ¿por qué son diferentes?",
        "expected_last": "cora",  # unifier al final
    },
]


def _check(case: Dict[str, Any], planner: TrajectoryPlanner) -> Dict[str, Any]:
    traj = planner.plan(case["query"])
    seq = traj.motor_sequence()
    ok = True
    reason = ""
    if "expected" in case:
        if seq != case["expected"]:
            ok = False
            reason = f"expected {case['expected']}, got {seq}"
    if "expected_last" in case:
        if seq[-1] != case["expected_last"]:
            ok = False
            reason = f"expected last={case['expected_last']}, got {seq}"

    # Ejecutar para verificar que no crashea
    def gen(motor, prompt, max_tokens):
        return f"[{motor}:{case['query'][:10]}]"
    result = CompositeOrchestrator(gen).execute(traj)

    # Si hay múltiples pasos con depends_on, verificar prompt del último
    exec_ok = True
    if len(result.step_results) > 1:
        last = result.step_results[-1]
        for dep_idx in traj.steps[-1].depends_on:
            dep_out = result.step_results[dep_idx].output
            if dep_out not in last.prompt:
                exec_ok = False
                reason = f"last step prompt missing dep {dep_idx}"

    return {
        "query": case["query"],
        "expected": case.get("expected") or case.get("expected_last"),
        "got": seq,
        "ok": ok and exec_ok,
        "reason": reason,
    }


def run() -> ExperimentReport:
    start = time.time()
    planner = TrajectoryPlanner()
    results = [_check(c, planner) for c in CASES]
    end = time.time()

    n_ok = sum(1 for r in results if r["ok"])
    total = len(results)
    accuracy = n_ok / total if total else 0.0
    passed = accuracy >= 0.75

    return ExperimentReport(
        experiment_id="exp4_compositional",
        name="Compositional test",
        started_at=start,
        ended_at=end,
        passed=passed,
        summary=f"{n_ok}/{total} queries planned correctly (acc={accuracy:.2f})",
        metrics={
            "total_cases": total,
            "correct": n_ok,
            "accuracy": accuracy,
        },
        details={"cases": results},
    )


def main() -> None:
    report = run()
    write_report(report)
    print(f"{report.experiment_id}: passed={report.passed} — {report.summary}")


if __name__ == "__main__":  # pragma: no cover
    main()
