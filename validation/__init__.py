"""
AION-C validation — AionCValidator (VAL).

Verificador encoder-only que evalua coherencia de la respuesta con el CausalGraph.
Realiza 4 checks: faithfulness, consistency, completeness, hallucination.
"""

from .model import (
    AionCValidator,
    ValidatorConfig,
    ValidationIssue,
    ValidationResult,
)

__all__ = [
    "AionCValidator",
    "ValidatorConfig",
    "ValidationIssue",
    "ValidationResult",
]
