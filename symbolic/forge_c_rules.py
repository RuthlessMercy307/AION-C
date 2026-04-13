"""
symbolic/forge_c_rules.py — Reglas formales para FORGE-C (Parte 20.1)
========================================================================

  - TypeCheckRule:    si una función espera int y recibe string → conflicto
  - NullCheckRule:    si una variable nullable se usa sin check → warning
  - LoopDetectionRule: ciclos en el grafo de CALLS → recursión infinita
  - DeadCodeRule:     nodos function/block sin incoming "calls" → código muerto

Convenciones:
  Nodos: type="function" | "variable" | "block" | "literal"
         props: {"return_type": "int", "param_types": [...], "nullable": True}
  Aristas:
         "calls"      function → function (llamada)
         "passes"     value → param (paso de argumento) con props={"position": 0}
         "uses"       block → variable (referencia/uso)
         "checks"     block → variable (check explícito de null)
"""

from __future__ import annotations

from typing import List, Optional

from .graph import SymbolicGraph, SymbolicNode, SymbolicEdge
from .rules import SymbolicRule, RuleResult


class TypeCheckRule(SymbolicRule):
    """
    Para cada arista 'passes' de un valor a un parámetro, verifica que el
    tipo del valor coincida con el tipo esperado del parámetro.
    """

    name  = "forge_c.type_check"
    motor = "forge_c"

    def applies_to(self, graph: SymbolicGraph) -> bool:
        return any(e.relation == "passes" for e in graph.edges)

    def apply(self, graph: SymbolicGraph) -> RuleResult:
        result = RuleResult()
        for e in graph.edges_with_relation("passes"):
            value_node = graph.find_node(e.source)
            param_node = graph.find_node(e.target)
            if value_node is None or param_node is None:
                continue
            value_type = value_node.props.get("type") or value_node.type
            expected = param_node.props.get("expected_type")
            if expected is None or value_type is None:
                continue
            if value_type != expected:
                msg = (
                    f"type mismatch: {e.source} ({value_type}) "
                    f"passed to {e.target} (expects {expected})"
                )
                if msg not in result.conflicts:
                    result.conflicts.append(msg)
                    result.modified = True
        return result


class NullCheckRule(SymbolicRule):
    """
    Si una variable tiene props={"nullable": True} y aparece como target
    de un edge 'uses' SIN un edge 'checks' del mismo block, marca warning.
    """

    name  = "forge_c.null_check"
    motor = "forge_c"

    def apply(self, graph: SymbolicGraph) -> RuleResult:
        result = RuleResult()
        nullable_vars = {
            n.id for n in graph.nodes
            if n.type == "variable" and n.props.get("nullable")
        }
        for e in graph.edges_with_relation("uses"):
            if e.target not in nullable_vars:
                continue
            # ¿Hay un check del mismo block?
            checked = any(
                ce.source == e.source and ce.target == e.target
                for ce in graph.edges_with_relation("checks")
            )
            if not checked:
                msg = (
                    f"nullable variable '{e.target}' used in '{e.source}' "
                    f"without null check"
                )
                if msg not in result.notes:
                    result.notes.append(msg)
                    result.modified = True
        return result


class LoopDetectionRule(SymbolicRule):
    """Cualquier ciclo en relación 'calls' es recursión infinita potencial."""

    name  = "forge_c.loop_detection"
    motor = "forge_c"

    def applies_to(self, graph: SymbolicGraph) -> bool:
        return any(e.relation == "calls" for e in graph.edges)

    def apply(self, graph: SymbolicGraph) -> RuleResult:
        result = RuleResult()
        if graph.has_cycle(relation="calls"):
            result.conflicts.append("loop detected in 'calls' graph (potential infinite recursion)")
            result.modified = True
        return result


class DeadCodeRule(SymbolicRule):
    """Funciones/bloques sin incoming 'calls' (excepto 'main') = dead code."""

    name  = "forge_c.dead_code"
    motor = "forge_c"

    ROOT_NAMES = {"main", "__main__", "entry"}

    def apply(self, graph: SymbolicGraph) -> RuleResult:
        result = RuleResult()
        for n in graph.nodes:
            if n.type not in ("function", "block"):
                continue
            label = (n.label or n.id).strip().lower()
            if label in self.ROOT_NAMES or n.id in self.ROOT_NAMES:
                continue
            if not graph.edges_to(n.id, relation="calls"):
                msg = f"dead code: {n.type} '{n.id}' has no incoming calls"
                if msg not in result.notes:
                    result.notes.append(msg)
                    result.modified = True
        return result


FORGE_C_RULES = [
    TypeCheckRule(),
    NullCheckRule(),
    LoopDetectionRule(),
    DeadCodeRule(),
]


__all__ = [
    "TypeCheckRule",
    "NullCheckRule",
    "LoopDetectionRule",
    "DeadCodeRule",
    "FORGE_C_RULES",
]
