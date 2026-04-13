"""
training/ — Infraestructura de entrenamiento de AION-C
=======================================================

Componentes:
  anti_forgetting.py — Las 5 capas de defensa contra catastrophic forgetting
                       (Parte 9.3 del MEGA-PROMPT)
"""

from .anti_forgetting import (
    MotorIsolation,
    WeightImportanceTracker,
    ExamItem, ExamResult, ExamRunner,
    RollbackManager, should_rollback,
    SelectiveReplay, compute_weight_delta,
)

__all__ = [
    "MotorIsolation",
    "WeightImportanceTracker",
    "ExamItem",
    "ExamResult",
    "ExamRunner",
    "RollbackManager",
    "should_rollback",
    "SelectiveReplay",
    "compute_weight_delta",
]
