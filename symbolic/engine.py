"""
symbolic/engine.py — Engine + ejecución híbrida (Parte 20.2)
==============================================================

Para cada query:
  1. Red neuronal (CRE) construye el grafo inicial
  2. Reglas simbólicas verifican y extienden el grafo
  3. Si hay conflicto entre neural y simbólico → simbólico gana
  4. El grafo verificado pasa al decoder

Esto es genuinamente neuro-symbolic: la red neuronal PROPONE,
las reglas simbólicas VERIFICAN.

API:
    engine = SymbolicEngine(rules=AXIOM_RULES)
    result = engine.apply_all(graph, max_iters=5)
    # result.graph             — grafo final (después de aplicar reglas)
    # result.applied_rules     — reglas que dispararon
    # result.conflicts         — conflictos resueltos a favor del símbolo
    # result.notes             — info de inferencias derivadas
    # result.removed_edges     — aristas neurales descartadas por conflicto
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .graph import SymbolicEdge, SymbolicGraph, SymbolicNode
from .rules import RuleResult, SymbolicRule
from .axiom_rules import AXIOM_RULES
from .forge_c_rules import FORGE_C_RULES
from .cora_rules import CORA_RULES


@dataclass
class HybridResult:
    """Resultado de la ejecución híbrida neuro-symbolic."""
    graph:           SymbolicGraph
    applied_rules:   List[str] = field(default_factory=list)
    conflicts:       List[str] = field(default_factory=list)
    notes:           List[str] = field(default_factory=list)
    added_nodes:     List[SymbolicNode] = field(default_factory=list)
    added_edges:     List[SymbolicEdge] = field(default_factory=list)
    removed_edges:   List[SymbolicEdge] = field(default_factory=list)
    iterations:      int = 0

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)


class SymbolicEngine:
    """
    Motor que aplica una colección de reglas hasta punto fijo
    (max_iters por defecto).

    Conflict resolution: cuando una regla reporta un conflicto sobre una
    arista que existe en el grafo, esa arista se ELIMINA (símbolo gana
    sobre la propuesta neural). Las aristas eliminadas quedan registradas
    en `result.removed_edges` para auditoría.
    """

    def __init__(self, rules: List[SymbolicRule]) -> None:
        self.rules = list(rules)

    def apply_all(self, graph: SymbolicGraph, max_iters: int = 5) -> HybridResult:
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        result = HybridResult(graph=graph)

        for it in range(1, max_iters + 1):
            any_modified = False
            for rule in self.rules:
                if not rule.applies_to(graph):
                    continue
                rr = rule.apply(graph)
                if rr.modified or rr.conflicts:
                    if rule.name not in result.applied_rules:
                        result.applied_rules.append(rule.name)
                if rr.modified:
                    any_modified = True
                result.added_nodes.extend(rr.added_nodes)
                result.added_edges.extend(rr.added_edges)
                result.notes.extend(rr.notes)
                # Resolver conflictos: símbolo gana
                for conflict_msg in rr.conflicts:
                    if conflict_msg not in result.conflicts:
                        result.conflicts.append(conflict_msg)
                    self._resolve_conflict(graph, conflict_msg, result)
            result.iterations = it
            if not any_modified:
                break

        return result

    def _resolve_conflict(
        self,
        graph: SymbolicGraph,
        conflict_msg: str,
        result: HybridResult,
    ) -> None:
        """
        Conflict resolution policy: para conflictos del tipo
        "X both causes and prevents Y" o "X both implies and negates Y"
        eliminamos la arista PREVENTS / NEGATES, conservando CAUSES / IMPLIES.

        Ese es el patrón "neural propuso ambas cosas pero la lógica gana":
        en estos pares, la afirmación positiva suele ser la que la red neural
        consideró más probable, y el conflicto se resuelve quitando la
        oposición. Otras políticas pueden inyectarse subclaseando.
        """
        msg = conflict_msg.lower()
        if "both causes and prevents" in msg:
            for e in list(graph.edges_with_relation("prevents")):
                # Solo eliminamos si la contraparte 'causes' existe
                if graph.has_edge(e.source, e.target, relation="causes"):
                    graph.remove_edge(e.source, e.target, relation="prevents")
                    result.removed_edges.append(e)
        elif "both implies and negates" in msg:
            for e in list(graph.edges_with_relation("negates")):
                if graph.has_edge(e.source, e.target, relation="implies"):
                    graph.remove_edge(e.source, e.target, relation="negates")
                    result.removed_edges.append(e)


def build_engine_for_motor(motor: str) -> SymbolicEngine:
    """Factory: devuelve un engine configurado con las reglas del motor."""
    rules_by_motor = {
        "axiom":   AXIOM_RULES,
        "forge_c": FORGE_C_RULES,
        "cora":    CORA_RULES,
    }
    rules = rules_by_motor.get(motor, [])
    return SymbolicEngine(rules=rules)


__all__ = [
    "HybridResult",
    "SymbolicEngine",
    "build_engine_for_motor",
]
