"""
AION-C budget — BudgetManager.

Clasifica queries en niveles de complejidad y asigna presupuesto de iteraciones
al CRE. Previene desperdicio de compute en queries triviales.
"""

from .manager import (
    HEURISTIC_THRESHOLDS,
    BudgetLevel,
    BudgetManager,
    BudgetOutput,
    QueryComplexityClassifier,
)

__all__ = [
    "HEURISTIC_THRESHOLDS",
    "BudgetLevel",
    "BudgetManager",
    "BudgetOutput",
    "QueryComplexityClassifier",
]
