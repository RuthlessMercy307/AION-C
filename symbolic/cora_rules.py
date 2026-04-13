"""
symbolic/cora_rules.py — Reglas formales para CORA (Parte 20.1)
==================================================================

  - CausalTransitivityRule:  A causes B, B causes C → A causes C
  - CausalContradictionRule: A causes B y A prevents B → conflicto
  - CounterfactualRule:      dada una intervención (remove A), ¿sigue
                             existiendo un path a un target?

Convenciones de aristas:
  "causes"      A → B  (A causa B)
  "prevents"    A → B  (A previene B)
  "enables"     A → B  (A habilita B)

CounterfactualRule no muta el grafo: produce una `note` con el resultado.
"""

from __future__ import annotations

from typing import List, Optional

from .graph import SymbolicGraph, SymbolicNode, SymbolicEdge
from .rules import SymbolicRule, RuleResult


class CausalTransitivityRule(SymbolicRule):
    """A causes B, B causes C → A causes C."""

    name  = "cora.transitivity"
    motor = "cora"

    def applies_to(self, graph: SymbolicGraph) -> bool:
        return len(graph.edges_with_relation("causes")) >= 2

    def apply(self, graph: SymbolicGraph) -> RuleResult:
        result = RuleResult()
        causes_edges = graph.edges_with_relation("causes")
        existing = {(e.source, e.target) for e in causes_edges}
        for e1 in causes_edges:
            for e2 in causes_edges:
                if e1.target == e2.source and e1.source != e2.target:
                    pair = (e1.source, e2.target)
                    if pair not in existing:
                        new_edge = SymbolicEdge(
                            source=e1.source, target=e2.target,
                            relation="causes",
                            props={"derived_by": "transitivity",
                                   "via": e1.target},
                        )
                        graph.add_edge(new_edge)
                        result.added_edges.append(new_edge)
                        result.modified = True
                        existing.add(pair)
                        result.notes.append(
                            f"causal: {e1.source} → {e1.target} → {e2.target}"
                        )
        return result


class CausalContradictionRule(SymbolicRule):
    """Si A causes B y A prevents B coexisten → conflicto."""

    name  = "cora.contradiction"
    motor = "cora"

    def apply(self, graph: SymbolicGraph) -> RuleResult:
        result = RuleResult()
        for e in graph.edges_with_relation("causes"):
            if graph.has_edge(e.source, e.target, relation="prevents"):
                msg = (
                    f"causal contradiction: {e.source} both causes "
                    f"and prevents {e.target}"
                )
                if msg not in result.conflicts:
                    result.conflicts.append(msg)
                    result.modified = True
        return result


class CounterfactualRule(SymbolicRule):
    """
    Counterfactual: dada una intervención (eliminar un nodo `intervention`),
    ¿sigue existiendo un path causal hacia `target`?

    Se invoca explícitamente con .check(graph, intervention, target) — el
    .apply() del Rule base es no-op a menos que graph.props tenga la
    intervención precargada en su metadata.

    Esta regla NO modifica el grafo; produce solo notas.
    """

    name  = "cora.counterfactual"
    motor = "cora"

    def apply(self, graph: SymbolicGraph) -> RuleResult:
        result = RuleResult()
        # Soporta intervenciones declaradas en props del grafo a través de un
        # nodo especial "_intervention" cuyas props llevan {"remove": "x", "target": "y"}
        intervention_node = graph.find_node("_intervention")
        if intervention_node is None:
            return result
        remove = intervention_node.props.get("remove")
        target = intervention_node.props.get("target")
        if not remove or not target:
            return result
        outcome = self.check(graph, remove, target)
        result.notes.append(
            f"counterfactual: remove '{remove}' → "
            f"path to '{target}' {'still exists' if outcome else 'broken'}"
        )
        return result

    def check(self, graph: SymbolicGraph, intervention: str, target: str) -> bool:
        """
        Devuelve True si, eliminando `intervention`, sigue habiendo un
        path causal (relation='causes') hacia `target`.
        """
        clone = graph.copy()
        clone.remove_node(intervention)
        # ¿Algún nodo restante alcanza target via causes?
        for n in clone.nodes:
            if n.id == target:
                continue
            if clone.has_path(n.id, target, relation="causes"):
                return True
        return False


CORA_RULES = [
    CausalTransitivityRule(),
    CausalContradictionRule(),
    CounterfactualRule(),
]


__all__ = [
    "CausalTransitivityRule",
    "CausalContradictionRule",
    "CounterfactualRule",
    "CORA_RULES",
]
