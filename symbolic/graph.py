"""
symbolic/graph.py — Grafo simbólico mínimo para neuro-symbolic
================================================================

Estructura ligera de grafo (sin torch ni dependencias del crystallizer)
para que las reglas simbólicas operen sobre algo testeable y serializable.

Un adapter futuro convertirá CausalGraph (de core/graph.py) a SymbolicGraph
y viceversa, pero las reglas trabajan sobre esta representación pura.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set


@dataclass
class SymbolicNode:
    id:    str
    label: str = ""
    type:  str = "concept"
    props: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":    self.id,
            "label": self.label,
            "type":  self.type,
            "props": dict(self.props),
        }


@dataclass
class SymbolicEdge:
    source:   str
    target:   str
    relation: str
    props:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source":   self.source,
            "target":   self.target,
            "relation": self.relation,
            "props":    dict(self.props),
        }


class SymbolicGraph:
    """
    Grafo dirigido con nodos tipados y aristas con relación nombrada.

    No usa torch. Todas las operaciones son determinísticas.
    """

    def __init__(
        self,
        nodes: Optional[List[SymbolicNode]] = None,
        edges: Optional[List[SymbolicEdge]] = None,
    ) -> None:
        self.nodes: List[SymbolicNode] = list(nodes or [])
        self.edges: List[SymbolicEdge] = list(edges or [])
        self._node_by_id: Dict[str, SymbolicNode] = {n.id: n for n in self.nodes}

    # ── nodos ──────────────────────────────────────────────────────────

    def add_node(self, node: SymbolicNode) -> None:
        if node.id in self._node_by_id:
            return
        self.nodes.append(node)
        self._node_by_id[node.id] = node

    def find_node(self, node_id: str) -> Optional[SymbolicNode]:
        return self._node_by_id.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._node_by_id

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._node_by_id:
            return False
        self.nodes = [n for n in self.nodes if n.id != node_id]
        del self._node_by_id[node_id]
        # remove all edges touching it
        self.edges = [
            e for e in self.edges
            if e.source != node_id and e.target != node_id
        ]
        return True

    # ── aristas ────────────────────────────────────────────────────────

    def add_edge(self, edge: SymbolicEdge) -> None:
        # Crea nodos vacíos si no existen
        if edge.source not in self._node_by_id:
            self.add_node(SymbolicNode(id=edge.source))
        if edge.target not in self._node_by_id:
            self.add_node(SymbolicNode(id=edge.target))
        # Evita duplicar exactamente la misma arista
        for e in self.edges:
            if (e.source == edge.source and e.target == edge.target
                    and e.relation == edge.relation):
                return
        self.edges.append(edge)

    def has_edge(self, source: str, target: str, relation: Optional[str] = None) -> bool:
        for e in self.edges:
            if e.source == source and e.target == target:
                if relation is None or e.relation == relation:
                    return True
        return False

    def edges_from(self, node_id: str, relation: Optional[str] = None) -> List[SymbolicEdge]:
        return [
            e for e in self.edges
            if e.source == node_id and (relation is None or e.relation == relation)
        ]

    def edges_to(self, node_id: str, relation: Optional[str] = None) -> List[SymbolicEdge]:
        return [
            e for e in self.edges
            if e.target == node_id and (relation is None or e.relation == relation)
        ]

    def edges_with_relation(self, relation: str) -> List[SymbolicEdge]:
        return [e for e in self.edges if e.relation == relation]

    def remove_edge(self, source: str, target: str, relation: Optional[str] = None) -> int:
        before = len(self.edges)
        self.edges = [
            e for e in self.edges
            if not (e.source == source and e.target == target
                    and (relation is None or e.relation == relation))
        ]
        return before - len(self.edges)

    # ── conectividad ───────────────────────────────────────────────────

    def has_path(self, source: str, target: str, relation: Optional[str] = None) -> bool:
        if source == target:
            return True
        if source not in self._node_by_id or target not in self._node_by_id:
            return False
        seen: Set[str] = set()
        stack = [source]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for e in self.edges_from(cur, relation):
                if e.target == target:
                    return True
                if e.target not in seen:
                    stack.append(e.target)
        return False

    def has_cycle(self, relation: Optional[str] = None) -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n.id: WHITE for n in self.nodes}

        def dfs(node_id: str) -> bool:
            color[node_id] = GRAY
            for e in self.edges_from(node_id, relation):
                t = e.target
                if t not in color:
                    color[t] = WHITE
                if color[t] == GRAY:
                    return True
                if color[t] == WHITE and dfs(t):
                    return True
            color[node_id] = BLACK
            return False

        for n in self.nodes:
            if color.get(n.id, WHITE) == WHITE:
                if dfs(n.id):
                    return True
        return False

    # ── utilidades ─────────────────────────────────────────────────────

    def copy(self) -> "SymbolicGraph":
        return SymbolicGraph(
            nodes=[SymbolicNode(n.id, n.label, n.type, dict(n.props)) for n in self.nodes],
            edges=[SymbolicEdge(e.source, e.target, e.relation, dict(e.props)) for e in self.edges],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def __len__(self) -> int:
        return len(self.nodes)


__all__ = ["SymbolicNode", "SymbolicEdge", "SymbolicGraph"]
