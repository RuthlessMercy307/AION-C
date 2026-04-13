"""
AION-C synth — Generadores de datos sintéticos para entrenamiento.
"""

from .causal_graph_gen import (
    AnswerType,
    CausalExample,
    CausalGraphGenerator,
    VerificationResult,
    verify_example,
)

__all__ = [
    "AnswerType",
    "CausalExample",
    "CausalGraphGenerator",
    "VerificationResult",
    "verify_example",
]
