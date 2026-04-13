"""
core/graph.py — AION-C Causal Energy Network
=============================================

Contrato central de datos entre todos los módulos de CORA.
CausalNode, CausalEdge, CausalGraph, CAUSAL_RELATIONS.

Paso 1 del plan: las dataclasses que todos los módulos importan.
No depende de PyTorch — son estructuras de datos puras.
PyTorch se introduce en los módulos que las consumen (CEC, GC, MP, etc.).
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Generator, Iterator, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# TIPOS DE NODO
# ─────────────────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    """
    Tipos de nodo en el grafo causal.

    Cada tipo tiene implicaciones semánticas para el message passing:
    - HYPOTHESIS: nodo no verificado → mayor incertidumbre en la energía
    - QUESTION:   nodo que el sistema debe resolver → penaliza completeness
    - FACT:       nodo verificado externamente → ancla el grafo (grounded)
    """
    ENTITY     = "entity"      # Objeto o agente concreto (persona, lugar, cosa)
    EVENT      = "event"       # Algo que ocurre en el tiempo (acción puntual)
    STATE      = "state"       # Condición persistente (temperatura alta, mercado bajista)
    ACTION     = "action"      # Operación ejecutable (comprar, lanzar, activar)
    HYPOTHESIS = "hypothesis"  # Suposición sin confirmar — necesita evidencia
    FACT       = "fact"        # Verdad verificada externamente — ancla del grafo
    QUESTION   = "question"    # Pregunta abierta — aumenta energía de completeness


NODE_TYPES: List[str] = [t.value for t in NodeType]


# ─────────────────────────────────────────────────────────────────────────────
# RELACIONES CAUSALES — EL VOCABULARIO COMPLETO DE AION-C
# ─────────────────────────────────────────────────────────────────────────────

class CausalRelation(str, Enum):
    """
    Vocabulario completo de relaciones causales para AION-C.

    Convención: A --relation--> B (todas son dirigidas).
    Las relaciones simétricas (CONTRADICTS, EQUIVALENT, CORRELATES)
    se almacenan como dos aristas opuestas en el grafo.

    Agrupadas semánticamente:

    CAUSALES DIRECTAS
      CAUSES       A produce B con mecanismo directo (A→B, fuerte)
      ENABLES      A hace posible que B ocurra (condición necesaria)
      PREVENTS     A impide que B ocurra (A→¬B)
      LEADS_TO     A→…→B cadena indirecta, varios pasos

    LÓGICAS / EPISTÉMICAS
      IMPLIES      A→B por deducción lógica (si A entonces B)
      FOLLOWS_FROM B es consecuencia de A (perspectiva desde B)
      CONTRADICTS  A y B no pueden ser verdad juntos (mutual exclusion)
      EQUIVALENT   A↔B equivalencia bidireccional

    EVIDENCIALES
      SUPPORTS     A es evidencia a favor de B (aumenta confianza en B)
      WEAKENS      A reduce la probabilidad o fuerza de B
      REQUIRES     A necesita que B exista/ocurra primero (dependencia)

    TEMPORALES
      PRECEDES     A ocurre antes de B (temporal, sin causalidad garantizada)

    ESTRUCTURALES
      PART_OF      A es componente de B (composición)
      INSTANCE_OF  A es caso particular de B (taxonomía)
      CORRELATES   A y B co-ocurren sin causalidad establecida
      ANALOGOUS_TO A es análogo a B en otro dominio o nivel de abstracción
    """

    # Causales directas
    CAUSES       = "causes"
    ENABLES      = "enables"
    PREVENTS     = "prevents"
    LEADS_TO     = "leads_to"

    # Lógicas / epistémicas
    IMPLIES      = "implies"
    FOLLOWS_FROM = "follows_from"
    CONTRADICTS  = "contradicts"
    EQUIVALENT   = "equivalent"

    # Evidenciales
    SUPPORTS     = "supports"
    WEAKENS      = "weakens"
    REQUIRES     = "requires"

    # Temporal
    PRECEDES     = "precedes"

    # Estructurales
    PART_OF      = "part_of"
    INSTANCE_OF  = "instance_of"
    CORRELATES   = "correlates"
    ANALOGOUS_TO = "analogous_to"


# Lista ordenada — el índice numérico es estable y lo usa BilinearEdgeDetector
CAUSAL_RELATIONS: List[str] = [r.value for r in CausalRelation]
CAUSAL_RELATIONS_LIST = CAUSAL_RELATIONS  # alias explícito del plan

# ── Agrupaciones semánticas ── usadas por EnergyFunction y TypedMessagePassing

INHIBITORY_RELATIONS: Set[str] = {
    CausalRelation.PREVENTS,
    CausalRelation.CONTRADICTS,
    CausalRelation.WEAKENS,
}

POSITIVE_RELATIONS: Set[str] = {
    CausalRelation.CAUSES,
    CausalRelation.ENABLES,
    CausalRelation.LEADS_TO,
    CausalRelation.SUPPORTS,
    CausalRelation.IMPLIES,
    CausalRelation.FOLLOWS_FROM,
    CausalRelation.EQUIVALENT,
}

TEMPORAL_RELATIONS: Set[str] = {
    CausalRelation.PRECEDES,
    CausalRelation.LEADS_TO,
}

SYMMETRIC_RELATIONS: Set[str] = {
    CausalRelation.CONTRADICTS,
    CausalRelation.EQUIVALENT,
    CausalRelation.CORRELATES,
    CausalRelation.ANALOGOUS_TO,
}

STRUCTURAL_RELATIONS: Set[str] = {
    CausalRelation.PART_OF,
    CausalRelation.INSTANCE_OF,
    CausalRelation.CORRELATES,
    CausalRelation.ANALOGOUS_TO,
}

# Pares que generan contradicción cuando ambos existen entre el mismo par A→B
CONTRADICTION_PAIRS: List[Tuple[CausalRelation, CausalRelation]] = [
    (CausalRelation.CAUSES,   CausalRelation.PREVENTS),
    (CausalRelation.ENABLES,  CausalRelation.PREVENTS),
    (CausalRelation.SUPPORTS, CausalRelation.WEAKENS),
    (CausalRelation.IMPLIES,  CausalRelation.CONTRADICTS),
    (CausalRelation.CAUSES,   CausalRelation.WEAKENS),
]


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL NODE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CausalNode:
    """
    Nodo del grafo causal. Representa un concepto atómico de razonamiento.

    El `vector` es el embedding conceptual producido por el SequenceEncoder.
    Es None hasta que el GraphConstructor lo puebla durante el forward pass.

    `confidence` mide qué tan seguro está el GC de que este nodo
    pertenece al grafo (distinto de la certeza sobre el contenido del nodo).

    `grounded` es True si el nodo tiene respaldo externo verificado
    (hecho conocido, memoria recuperada). Los nodos no grounded
    contribuyen más a la energía de grounding del CEC.
    """

    node_id: str
    label: str
    node_type: NodeType = NodeType.ENTITY
    confidence: float = 1.0
    grounded: bool = False
    vector: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        if isinstance(self.node_type, str):
            self.node_type = NodeType(self.node_type)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CausalNode) and self.node_id == other.node_id

    def __repr__(self) -> str:
        return (
            f"CausalNode(id={self.node_id!r}, label={self.label!r}, "
            f"type={self.node_type.value}, conf={self.confidence:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL EDGE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CausalEdge:
    """
    Arista dirigida en el grafo causal.

        source_id --[relation]--> target_id

    `strength`:   magnitud de la relación (0=débil, 1=determinista)
    `confidence`: certeza del GraphConstructor sobre la existencia de la arista

    Los índices `source_idx` / `target_idx` son asignados por CausalGraph
    cuando la arista se agrega al grafo. Son los índices enteros de los nodos
    en el tensor de features que usa el CEC (TypedMessagePassing los necesita).
    """

    source_id: str
    target_id: str
    relation: CausalRelation
    strength: float = 1.0
    confidence: float = 1.0
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: Dict = field(default_factory=dict)

    # Asignados por CausalGraph.add_edge — no pasar en construcción directa
    source_idx: int = field(default=-1, compare=False, repr=False)
    target_idx: int = field(default=-1, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"strength must be in [0, 1], got {self.strength!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        if isinstance(self.relation, str):
            self.relation = CausalRelation(self.relation)
        if self.source_id == self.target_id:
            raise ValueError(
                f"Self-loops not allowed: source_id == target_id == {self.source_id!r}"
            )

    def __hash__(self) -> int:
        return hash(self.edge_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CausalEdge) and self.edge_id == other.edge_id

    def __repr__(self) -> str:
        return (
            f"CausalEdge({self.source_id!r} --[{self.relation.value}]--> "
            f"{self.target_id!r}, str={self.strength:.2f}, conf={self.confidence:.2f})"
        )

    # ── Propiedades semánticas ───────────────────────────────────────────────

    @property
    def is_inhibitory(self) -> bool:
        """True si la relación inhibe o bloquea al nodo destino."""
        return self.relation.value in INHIBITORY_RELATIONS

    @property
    def is_positive(self) -> bool:
        """True si la relación refuerza o activa al nodo destino."""
        return self.relation.value in POSITIVE_RELATIONS

    @property
    def is_symmetric(self) -> bool:
        """True si la relación es semánticamente simétrica (A↔B)."""
        return self.relation.value in SYMMETRIC_RELATIONS

    @property
    def is_temporal(self) -> bool:
        """True si la relación implica ordenamiento temporal."""
        return self.relation.value in TEMPORAL_RELATIONS

    @property
    def is_structural(self) -> bool:
        """True si la relación es composicional o taxonómica."""
        return self.relation.value in STRUCTURAL_RELATIONS


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class CausalGraph:
    """
    Grafo causal dirigido. Estructura de datos central de AION-C.

    Contrato:
      - Los node_id son únicos.
      - Los edge_id son únicos.
      - No se permiten auto-bucles (source_id == target_id).
      - Los índices enteros (source_idx, target_idx) de cada arista
        son asignados automáticamente al agregar la arista y reflejan
        el orden de inserción de los nodos. Se recalculan si se eliminan nodos.

    Módulos consumidores:
      - GraphConstructor  → construye el grafo desde concept vectors
      - CausalEnergyCore  → itera MP sobre el grafo
      - TypedMessagePassing → usa source_idx / target_idx para indexar tensores
      - EnergyFunction    → usa adjacency, node_types, grounded_mask
      - SequenceDecoder   → usa node features para cross-attention
    """

    def __init__(self, graph_id: Optional[str] = None, root_question: str = "") -> None:
        self.graph_id: str = graph_id or str(uuid.uuid4())[:8]
        self.root_question: str = root_question  # Pregunta que el CEC debe resolver

        # Almacenamiento principal
        self._nodes: Dict[str, CausalNode] = {}
        self._edges: Dict[str, CausalEdge] = {}

        # Índices para acceso eficiente
        self._out_edges: Dict[str, List[str]] = defaultdict(list)  # node_id → [edge_id]
        self._in_edges:  Dict[str, List[str]] = defaultdict(list)  # node_id → [edge_id]

        # Orden de inserción de nodos (estable → índices para tensores)
        self._node_order: List[str] = []

    # ── Propiedades ──────────────────────────────────────────────────────────

    @property
    def nodes(self) -> List[CausalNode]:
        """Lista de nodos en orden de inserción."""
        return [self._nodes[nid] for nid in self._node_order]

    @property
    def edges(self) -> List[CausalEdge]:
        """Lista de aristas en orden de inserción."""
        return list(self._edges.values())

    @property
    def node_index(self) -> Dict[str, int]:
        """Mapeo node_id → índice entero (para indexar tensores en el CEC)."""
        return {nid: i for i, nid in enumerate(self._node_order)}

    @property
    def grounded_mask(self) -> List[bool]:
        """Máscara booleana: True si el nodo i está grounded. Uso: EnergyFunction."""
        return [self._nodes[nid].grounded for nid in self._node_order]

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return (
            f"CausalGraph(id={self.graph_id!r}, "
            f"nodes={len(self._nodes)}, edges={len(self._edges)})"
        )

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    # ── Acceso ───────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> CausalNode:
        """Devuelve el nodo. Lanza KeyError si no existe."""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} not in graph")
        return self._nodes[node_id]

    def get_edge(self, edge_id: str) -> CausalEdge:
        """Devuelve la arista. Lanza KeyError si no existe."""
        if edge_id not in self._edges:
            raise KeyError(f"Edge {edge_id!r} not in graph")
        return self._edges[edge_id]

    def get_node_index(self, node_id: str) -> int:
        """Índice entero del nodo (para tensores). Lanza KeyError si no existe."""
        try:
            return self._node_order.index(node_id)
        except ValueError:
            raise KeyError(f"Node {node_id!r} not in graph")

    def successors(self, node_id: str) -> List[CausalNode]:
        """Nodos a los que apuntan las aristas salientes de node_id."""
        return [
            self._nodes[self._edges[eid].target_id]
            for eid in self._out_edges.get(node_id, [])
        ]

    def predecessors(self, node_id: str) -> List[CausalNode]:
        """Nodos que apuntan hacia node_id."""
        return [
            self._nodes[self._edges[eid].source_id]
            for eid in self._in_edges.get(node_id, [])
        ]

    def out_edges(self, node_id: str) -> List[CausalEdge]:
        """Aristas salientes de node_id."""
        return [self._edges[eid] for eid in self._out_edges.get(node_id, [])]

    def in_edges(self, node_id: str) -> List[CausalEdge]:
        """Aristas entrantes a node_id."""
        return [self._edges[eid] for eid in self._in_edges.get(node_id, [])]

    def edges_between(self, source_id: str, target_id: str) -> List[CausalEdge]:
        """Todas las aristas directas de source_id a target_id."""
        return [
            self._edges[eid]
            for eid in self._out_edges.get(source_id, [])
            if self._edges[eid].target_id == target_id
        ]

    # ── Mutación ─────────────────────────────────────────────────────────────

    def add_node(self, node: CausalNode) -> "CausalGraph":
        """
        Agrega un nodo al grafo.

        Si el node_id ya existe, no hace nada (idempotente).
        Devuelve self para encadenamiento: graph.add_node(a).add_node(b)
        """
        if node.node_id in self._nodes:
            return self
        self._nodes[node.node_id] = node
        self._node_order.append(node.node_id)
        return self

    def add_edge(self, edge: CausalEdge) -> "CausalGraph":
        """
        Agrega una arista al grafo.

        Precondición: source_id y target_id deben existir como nodos.
        Asigna source_idx y target_idx basándose en el orden de nodos.
        Devuelve self para encadenamiento.
        """
        if edge.source_id not in self._nodes:
            raise ValueError(
                f"Source node {edge.source_id!r} not in graph. Add it first."
            )
        if edge.target_id not in self._nodes:
            raise ValueError(
                f"Target node {edge.target_id!r} not in graph. Add it first."
            )

        idx = self.node_index
        edge.source_idx = idx[edge.source_id]
        edge.target_idx = idx[edge.target_id]

        self._edges[edge.edge_id] = edge
        self._out_edges[edge.source_id].append(edge.edge_id)
        self._in_edges[edge.target_id].append(edge.edge_id)
        return self

    def remove_node(self, node_id: str) -> "CausalGraph":
        """
        Elimina un nodo y todas sus aristas conectadas.

        Recalcula source_idx / target_idx de todas las aristas restantes
        para reflejar el nuevo orden de nodos.
        Devuelve self.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} not in graph")

        # Recolectar aristas a eliminar
        edge_ids_to_remove: Set[str] = set(
            self._out_edges.get(node_id, []) +
            self._in_edges.get(node_id, [])
        )

        # Limpiar índices secundarios del nodo
        for eid in edge_ids_to_remove:
            edge = self._edges[eid]
            if edge.source_id != node_id:
                self._out_edges[edge.source_id] = [
                    e for e in self._out_edges[edge.source_id] if e != eid
                ]
            if edge.target_id != node_id:
                self._in_edges[edge.target_id] = [
                    e for e in self._in_edges[edge.target_id] if e != eid
                ]
            del self._edges[eid]

        del self._nodes[node_id]
        self._node_order.remove(node_id)
        self._out_edges.pop(node_id, None)
        self._in_edges.pop(node_id, None)

        # Recalcular índices enteros en las aristas restantes
        new_idx = self.node_index
        for edge in self._edges.values():
            edge.source_idx = new_idx[edge.source_id]
            edge.target_idx = new_idx[edge.target_id]

        return self

    def remove_edge(self, edge_id: str) -> "CausalGraph":
        """Elimina una arista por su edge_id. Devuelve self."""
        if edge_id not in self._edges:
            raise KeyError(f"Edge {edge_id!r} not in graph")
        edge = self._edges[edge_id]
        self._out_edges[edge.source_id].remove(edge_id)
        self._in_edges[edge.target_id].remove(edge_id)
        del self._edges[edge_id]
        return self

    # ── Análisis de grafo ────────────────────────────────────────────────────

    def detect_cycles(self) -> List[List[str]]:
        """
        Detecta todos los ciclos en el grafo dirigido.

        Excluye relaciones simétricas (CONTRADICTS, EQUIVALENT, CORRELATES,
        ANALOGOUS_TO) porque en esas la doble arista A→B + B→A no es un ciclo
        lógico — es la representación correcta de la simetría.

        Algoritmo: DFS con marcado de estado (WHITE/GRAY/BLACK).
        Devuelve una lista de ciclos, cada ciclo como lista de node_ids.

        Complejidad: O(V + E)
        """
        # Construir adjacency excluyendo relaciones simétricas
        adj: Dict[str, List[str]] = defaultdict(list)
        for edge in self._edges.values():
            if edge.relation.value not in SYMMETRIC_RELATIONS:
                adj[edge.source_id].append(edge.target_id)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {nid: WHITE for nid in self._nodes}
        cycles: List[List[str]] = []
        path: List[str] = []

        def dfs(node: str) -> None:
            color[node] = GRAY
            path.append(node)
            for neighbor in adj[node]:
                if color[neighbor] == GRAY:
                    # Ciclo encontrado — extraer desde el punto de inicio
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                elif color[neighbor] == WHITE:
                    dfs(neighbor)
            path.pop()
            color[node] = BLACK

        for node_id in self._node_order:
            if color[node_id] == WHITE:
                dfs(node_id)

        return cycles

    def find_contradictions(self) -> List[Tuple[CausalEdge, CausalEdge]]:
        """
        Detecta pares de aristas que se contradicen entre sí.

        Dos tipos de contradicción:
        1. DIRECTA: misma fuente, mismo destino, relaciones opuestas.
           Ejemplo: A --CAUSES--> B y A --PREVENTS--> B simultáneamente.

        2. SIMÉTRICA: relación asimétrica que contradice otra en sentido opuesto.
           Ejemplo: A --CAUSES--> B y B --PREVENTS--> A
           (si A causa B, que B impida A es circular y contradictorio).

        Devuelve lista de tuplas (edge1, edge2) donde el par es contradictorio.
        Cada par aparece una sola vez (no se duplica en orden inverso).
        """
        contradictions: List[Tuple[CausalEdge, CausalEdge]] = []
        seen: Set[Tuple[str, str]] = set()

        edge_list = list(self._edges.values())

        for i, e1 in enumerate(edge_list):
            for e2 in edge_list[i + 1:]:

                # Tipo 1 — misma fuente y destino, relaciones opuestas
                if e1.source_id == e2.source_id and e1.target_id == e2.target_id:
                    pair_key = (e1.edge_id, e2.edge_id)
                    if pair_key not in seen:
                        for rel_a, rel_b in CONTRADICTION_PAIRS:
                            if (
                                (e1.relation == rel_a and e2.relation == rel_b) or
                                (e1.relation == rel_b and e2.relation == rel_a)
                            ):
                                contradictions.append((e1, e2))
                                seen.add(pair_key)
                                break

                # Tipo 2 — relaciones opuestas en sentido inverso
                elif e1.source_id == e2.target_id and e1.target_id == e2.source_id:
                    pair_key = tuple(sorted([e1.edge_id, e2.edge_id]))
                    if pair_key not in seen:
                        for rel_a, rel_b in CONTRADICTION_PAIRS:
                            if (
                                (e1.relation == rel_a and e2.relation == rel_b) or
                                (e1.relation == rel_b and e2.relation == rel_a)
                            ):
                                contradictions.append((e1, e2))
                                seen.add(pair_key)
                                break

        return contradictions

    def has_path(self, source_id: str, target_id: str) -> bool:
        """
        BFS para determinar si existe un camino dirigido de source a target.
        Útil para detectar dependencias transitivas en el CEC.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return False
        if source_id == target_id:
            return False
        visited: Set[str] = set()
        queue: deque[str] = deque([source_id])
        while queue:
            current = queue.popleft()
            if current == target_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            for succ in self.successors(current):
                if succ.node_id not in visited:
                    queue.append(succ.node_id)
        return False

    # ── Representaciones para el CEC ─────────────────────────────────────────

    def to_adjacency(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Adjacency dict para uso en EnergyFunction.

        Formato: {source_id: {target_id: [relation_value, ...]}}
        Múltiples aristas entre el mismo par → lista de relaciones.
        """
        adj: Dict[str, Dict[str, List[str]]] = {}
        for nid in self._node_order:
            adj[nid] = defaultdict(list)
        for edge in self._edges.values():
            adj[edge.source_id][edge.target_id].append(edge.relation.value)
        return {k: dict(v) for k, v in adj.items()}

    def summary(self) -> Dict:
        """
        Resumen del grafo para logging, MetaReasoner y debugging.
        """
        type_counts: Dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            type_counts[node.node_type.value] += 1

        relation_counts: Dict[str, int] = defaultdict(int)
        for edge in self._edges.values():
            relation_counts[edge.relation.value] += 1

        cycles = self.detect_cycles()
        contradictions = self.find_contradictions()

        avg_conf_nodes = (
            sum(n.confidence for n in self._nodes.values()) / len(self._nodes)
            if self._nodes else 0.0
        )
        avg_conf_edges = (
            sum(e.confidence for e in self._edges.values()) / len(self._edges)
            if self._edges else 0.0
        )

        return {
            "graph_id":           self.graph_id,
            "root_question":      self.root_question,
            "n_nodes":            len(self._nodes),
            "n_edges":            len(self._edges),
            "node_types":         dict(type_counts),
            "relation_types":     dict(relation_counts),
            "n_cycles":           len(cycles),
            "n_contradictions":   len(contradictions),
            "grounded_nodes":     sum(1 for n in self._nodes.values() if n.grounded),
            "avg_node_confidence": round(avg_conf_nodes, 4),
            "avg_edge_confidence": round(avg_conf_edges, 4),
            "has_questions":      any(
                n.node_type == NodeType.QUESTION for n in self._nodes.values()
            ),
        }

    # ── Iteración ─────────────────────────────────────────────────────────────

    def iter_nodes(self) -> Iterator[CausalNode]:
        return iter(self.nodes)

    def iter_edges(self) -> Iterator[CausalEdge]:
        return iter(self.edges)
