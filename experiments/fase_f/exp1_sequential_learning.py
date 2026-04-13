"""
exp1_sequential_learning.py — Parte 21.1 (sequential learning test).

Pregunta científica:
    Después de aprender N conceptos secuencialmente (uno por uno, cada uno
    creando un adapter), ¿los 10 exámenes originales siguen pasando 10/10?

Procedimiento:
    1. Crear FakeMotor determinista.
    2. Snapshot de los 10 exámenes originales (output de referencia).
    3. Para i en [1, N]:
          - Crear adapter LoRA, attach, mutar pesos (simula fine-tune),
            guardar al registry, DETACH.
          - Cada 10 aprendizajes, medir exam_pass_rate.
    4. Verificar que pass_rate == 1.0 siempre después del detach
       (garantía dura de Parte 22.1).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import List

import torch

from experiments.fase_f.common import (
    ExperimentReport,
    FakeMotor,
    MOTOR_TARGET_PATHS,
    exam_outputs,
    exam_pass_rate,
    make_exam,
    write_report,
)
from growth import (
    AdapterRegistry,
    LoRAConfig,
    attach_adapter_pack,
    build_adapter_pack,
    detach_adapter_pack,
)


def run(n_concepts: int = 100, check_every: int = 10, seed: int = 42) -> ExperimentReport:
    start = time.time()
    motor = FakeMotor(seed=seed)
    exam = make_exam()
    reference = exam_outputs(motor, exam)

    history: List[dict] = []
    min_pass_rate = 1.0

    with tempfile.TemporaryDirectory() as tmp:
        reg = AdapterRegistry(Path(tmp))
        for i in range(1, n_concepts + 1):
            name = f"concept_{i:04d}"
            pack = build_adapter_pack(
                motor, MOTOR_TARGET_PATHS, LoRAConfig(rank=4), name, "forge_c"
            )
            attach_adapter_pack(motor, pack)
            for path in MOTOR_TARGET_PATHS:
                with torch.no_grad():
                    pack.get(path).lora_B.normal_(0, 0.25)
            reg.save(pack)
            detach_adapter_pack(motor, pack)

            if i % check_every == 0:
                pr = exam_pass_rate(motor, exam, reference)
                history.append({"concepts_learned": i, "exam_pass_rate": pr})
                min_pass_rate = min(min_pass_rate, pr)

    end = time.time()
    passed = min_pass_rate == 1.0
    return ExperimentReport(
        experiment_id="exp1_sequential_learning",
        name="Sequential learning test",
        started_at=start,
        ended_at=end,
        passed=passed,
        summary=(
            f"learned {n_concepts} concepts, min exam_pass_rate={min_pass_rate:.4f}"
        ),
        metrics={
            "n_concepts": n_concepts,
            "min_exam_pass_rate": min_pass_rate,
            "final_exam_pass_rate": history[-1]["exam_pass_rate"] if history else 1.0,
            "checkpoints": len(history),
        },
        details={"history": history},
    )


def main() -> None:
    report = run()
    write_report(report)
    print(f"{report.experiment_id}: passed={report.passed} — {report.summary}")


if __name__ == "__main__":  # pragma: no cover
    main()
