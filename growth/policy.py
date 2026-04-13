"""
growth/policy.py — Reglas de decisión de crecimiento (Parte 22.4).

Dado un concepto nuevo X, decide si hay que:
    NO_GROWTH       — el motor ya lo maneja (acc ≥ 0.70)
    ADAPTER         — adapter low-rank (0.30 ≤ acc < 0.70)
    EXPAND_MOTOR    — expandir el motor existente (acc < 0.30, dominio afín)
    SUB_MOTOR       — crear sub-motor hijo (acc < 0.30, dominio distinto)

Los umbrales son los del MEGA-PROMPT parte 22.4. Todo tunable por config.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GrowthDecision(str, Enum):
    NO_GROWTH = "no_growth"
    ADAPTER = "adapter"
    EXPAND_MOTOR = "expand_motor"
    SUB_MOTOR = "sub_motor"


@dataclass
class GrowthPolicy:
    """Parámetros del policy de crecimiento.

    adapter_lower:  accuracy mínima para considerar adapter (si no, expand/sub).
    adapter_upper:  accuracy máxima — arriba no se crece.
    max_adapters_per_motor: límite duro para VRAM. Más allá fuerza pruning antes.
    """
    adapter_lower: float = 0.30
    adapter_upper: float = 0.70
    max_adapters_per_motor: int = 8


def decide_growth(
    baseline_accuracy: float,
    policy: Optional[GrowthPolicy] = None,
    domain_distinct: bool = False,
    current_adapters_in_motor: int = 0,
) -> GrowthDecision:
    """Aplica las reglas de 22.4.

    Args:
        baseline_accuracy: accuracy del motor candidato en un mini-test del
                           concepto. Rango [0, 1].
        policy:            override de umbrales.
        domain_distinct:   True si el dominio nuevo es estructuralmente
                           distinto al del motor (caller lo decide por heurística
                           o por el planner). Determina EXPAND vs SUB_MOTOR.
        current_adapters_in_motor: cuántos adapters tiene ya el motor. Si se
                           alcanza el tope, la decisión sube un escalón porque
                           no cabe otro adapter sin hacer pruning.

    Returns:
        GrowthDecision
    """
    p = policy or GrowthPolicy()

    if not (0.0 <= baseline_accuracy <= 1.0):
        raise ValueError(f"baseline_accuracy must be in [0,1], got {baseline_accuracy}")

    if baseline_accuracy >= p.adapter_upper:
        return GrowthDecision.NO_GROWTH

    if baseline_accuracy >= p.adapter_lower:
        # Rango natural de adapter, a menos que el motor ya esté lleno.
        if current_adapters_in_motor >= p.max_adapters_per_motor:
            # Subir un escalón: expansión o sub-motor según domain_distinct.
            return GrowthDecision.SUB_MOTOR if domain_distinct else GrowthDecision.EXPAND_MOTOR
        return GrowthDecision.ADAPTER

    # acc < adapter_lower → hay que ir más allá de un adapter
    return GrowthDecision.SUB_MOTOR if domain_distinct else GrowthDecision.EXPAND_MOTOR
