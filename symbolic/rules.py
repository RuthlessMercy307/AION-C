"""
symbolic/rules.py — Base de reglas simbólicas
===============================================

Cada regla:
  - tiene un name y un motor (axiom/forge_c/cora/...)
  - applies_to(graph) → bool : decide si vale la pena correrla
  - apply(graph)      → RuleResult : modifica el grafo (in-place) y reporta

RuleResult acumula:
  - modified: bool
  - added_nodes / added_edges / removed_edges
  - conflicts (lista de strings) — la red neuronal proponía X pero el símbolo
    encontró que X es inconsistente
  - notes (info benigna)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .graph import SymbolicGraph, SymbolicNode, SymbolicEdge


@dataclass
class RuleResult:
    modified:      bool = False
    added_nodes:   List[SymbolicNode] = field(default_factory=list)
    added_edges:   List[SymbolicEdge] = field(default_factory=list)
    removed_edges: List[SymbolicEdge] = field(default_factory=list)
    conflicts:    List[str] = field(default_factory=list)
    notes:        List[str] = field(default_factory=list)

    def merge(self, other: "RuleResult") -> None:
        if other.modified:
            self.modified = True
        self.added_nodes.extend(other.added_nodes)
        self.added_edges.extend(other.added_edges)
        self.removed_edges.extend(other.removed_edges)
        self.conflicts.extend(other.conflicts)
        self.notes.extend(other.notes)


class SymbolicRule:
    """ABC para reglas simbólicas."""

    name:  str = ""
    motor: str = ""

    def applies_to(self, graph: SymbolicGraph) -> bool:
        return True

    def apply(self, graph: SymbolicGraph) -> RuleResult:
        raise NotImplementedError


__all__ = ["RuleResult", "SymbolicRule"]
