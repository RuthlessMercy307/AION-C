"""
exp2_cross_domain.py — Parte 21.2 (cross-domain interference test).

Pregunta científica:
    Si aprendemos adapters de MÚLTIPLES motores (forge_c, axiom, cora, muse,
    empathy), ¿cada motor mantiene intacto su exam aislado? ¿Los adapters
    de un motor interfieren con otro?

Procedimiento:
    1. Crear un FakeMotor por cada dominio simulado (5 motores).
    2. Snapshot del exam de cada motor (10 inputs por motor).
    3. Para cada motor, crear 3 adapters y guardarlos.
    4. Medir exam_pass_rate de cada motor después de que TODOS los adapters
       están guardados pero NINGUNO attached (producción normal).
    5. Verificar interferencia == 0.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Dict, List

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

MOTORS = ("cora", "forge_c", "axiom", "muse", "empathy")
ADAPTERS_PER_MOTOR = 3


def run(seed: int = 123) -> ExperimentReport:
    start = time.time()
    # 1 motor físico por cada dominio lógico
    motors: Dict[str, FakeMotor] = {
        name: FakeMotor(seed=seed + idx)
        for idx, name in enumerate(MOTORS)
    }
    exams = {name: make_exam(seed=seed + 100 + i) for i, name in enumerate(MOTORS)}
    references = {
        name: exam_outputs(motors[name], exams[name]) for name in MOTORS
    }

    with tempfile.TemporaryDirectory() as tmp:
        reg = AdapterRegistry(Path(tmp))
        # Crear ADAPTERS_PER_MOTOR adapters en cada motor
        for name in MOTORS:
            motor = motors[name]
            for j in range(ADAPTERS_PER_MOTOR):
                concept = f"{name}_skill_{j}"
                pack = build_adapter_pack(
                    motor, MOTOR_TARGET_PATHS, LoRAConfig(rank=4), concept, name
                )
                attach_adapter_pack(motor, pack)
                for path in MOTOR_TARGET_PATHS:
                    with torch.no_grad():
                        pack.get(path).lora_B.normal_(0, 0.3)
                reg.save(pack)
                detach_adapter_pack(motor, pack)

    # Medir pass rate post-everything
    per_motor_pass: Dict[str, float] = {}
    for name in MOTORS:
        per_motor_pass[name] = exam_pass_rate(
            motors[name], exams[name], references[name]
        )

    min_pass = min(per_motor_pass.values())
    mean_pass = sum(per_motor_pass.values()) / len(per_motor_pass)
    passed = min_pass == 1.0

    end = time.time()
    return ExperimentReport(
        experiment_id="exp2_cross_domain",
        name="Cross-domain interference test",
        started_at=start,
        ended_at=end,
        passed=passed,
        summary=(
            f"5 motors × {ADAPTERS_PER_MOTOR} adapters; "
            f"min_pass={min_pass:.4f}, mean_pass={mean_pass:.4f}"
        ),
        metrics={
            "n_motors": len(MOTORS),
            "adapters_per_motor": ADAPTERS_PER_MOTOR,
            "min_exam_pass_rate": min_pass,
            "mean_exam_pass_rate": mean_pass,
        },
        details={"per_motor_pass_rate": per_motor_pass},
    )


def main() -> None:
    report = run()
    write_report(report)
    print(f"{report.experiment_id}: passed={report.passed} — {report.summary}")


if __name__ == "__main__":  # pragma: no cover
    main()
