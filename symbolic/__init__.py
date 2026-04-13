"""
symbolic/ — Neuro-symbolic completo (Parte 20 del MEGA-PROMPT)
================================================================

Reglas formales por motor que verifican / extienden el grafo construido
por la red neuronal. Si hay conflicto entre neural y simbólico → simbólico
gana (las reglas lógicas son verdad, los patrones estadísticos no siempre).

Componentes:
  SymbolicGraph              — grafo mínimo (nodos + aristas) sin torch
  SymbolicNode / SymbolicEdge
  SymbolicRule               — base ABC para reglas
  RuleResult                 — modificaciones + conflictos + notas

  axiom_rules:    TransitivityRule, ContradictionRule, SubstitutionRule, ArithmeticRule
  forge_c_rules:  TypeCheckRule, NullCheckRule, LoopDetectionRule, DeadCodeRule
  cora_rules:     CausalTransitivityRule, CausalContradictionRule, CounterfactualRule

  SymbolicEngine             — apply_all con conflict resolution
  HybridResult               — grafo final + reglas aplicadas + conflictos
  build_engine_for_motor()   — factory que devuelve un engine por motor
"""

from .graph import SymbolicGraph, SymbolicNode, SymbolicEdge
from .rules import SymbolicRule, RuleResult
from .axiom_rules import (
    TransitivityRule, ContradictionRule, SubstitutionRule, ArithmeticRule,
    AXIOM_RULES,
)
from .forge_c_rules import (
    TypeCheckRule, NullCheckRule, LoopDetectionRule, DeadCodeRule,
    FORGE_C_RULES,
)
from .cora_rules import (
    CausalTransitivityRule, CausalContradictionRule, CounterfactualRule,
    CORA_RULES,
)
from .engine import SymbolicEngine, HybridResult, build_engine_for_motor

__all__ = [
    "SymbolicGraph", "SymbolicNode", "SymbolicEdge",
    "SymbolicRule", "RuleResult",
    "TransitivityRule", "ContradictionRule", "SubstitutionRule", "ArithmeticRule",
    "AXIOM_RULES",
    "TypeCheckRule", "NullCheckRule", "LoopDetectionRule", "DeadCodeRule",
    "FORGE_C_RULES",
    "CausalTransitivityRule", "CausalContradictionRule", "CounterfactualRule",
    "CORA_RULES",
    "SymbolicEngine", "HybridResult", "build_engine_for_motor",
]
