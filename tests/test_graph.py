"""
tests/test_graph.py — Tests exhaustivos para core/graph.py
===========================================================

Cubre:
  - CausalNode: construcción, validación, hash, eq
  - CausalEdge: construcción, validación, propiedades semánticas
  - CAUSAL_RELATIONS: completitud, indexado, agrupaciones semánticas
  - CausalGraph: add/remove nodos y aristas, índices enteros
  - CausalGraph.detect_cycles: acíclico, un ciclo, múltiples ciclos,
    relaciones simétricas (no falsos positivos)
  - CausalGraph.find_contradictions: directa, inversa, sin contradicciones
  - CausalGraph.has_path: BFS
  - CausalGraph.to_adjacency, summary
  - Casos de borde: grafos vacíos, un nodo, aristas múltiples entre mismo par

Ejecutar:
  cd IAS/AION-C
  python -m pytest tests/test_graph.py -v
"""

import sys
import os

# Hacer importable el paquete desde la raíz IAS/AION-C
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.graph import (
    CausalEdge,
    CausalGraph,
    CausalNode,
    CausalRelation,
    NodeType,
    CAUSAL_RELATIONS,
    CAUSAL_RELATIONS_LIST,
    CONTRADICTION_PAIRS,
    INHIBITORY_RELATIONS,
    NODE_TYPES,
    POSITIVE_RELATIONS,
    STRUCTURAL_RELATIONS,
    SYMMETRIC_RELATIONS,
    TEMPORAL_RELATIONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def make_node(nid: str, label: str = None, node_type=NodeType.ENTITY, **kw) -> CausalNode:
    return CausalNode(node_id=nid, label=label or nid, node_type=node_type, **kw)


def make_edge(src: str, tgt: str, rel=CausalRelation.CAUSES, **kw) -> CausalEdge:
    return CausalEdge(source_id=src, target_id=tgt, relation=rel, **kw)


def triangle_graph() -> CausalGraph:
    """A → B → C → A (ciclo)"""
    g = CausalGraph(graph_id="triangle")
    g.add_node(make_node("A"))
    g.add_node(make_node("B"))
    g.add_node(make_node("C"))
    g.add_edge(make_edge("A", "B"))
    g.add_edge(make_edge("B", "C"))
    g.add_edge(make_edge("C", "A"))
    return g


def linear_graph() -> CausalGraph:
    """A → B → C (sin ciclos)"""
    g = CausalGraph(graph_id="linear")
    g.add_node(make_node("A"))
    g.add_node(make_node("B"))
    g.add_node(make_node("C"))
    g.add_edge(make_edge("A", "B"))
    g.add_edge(make_edge("B", "C"))
    return g


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — CAUSAL_RELATIONS Y CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalRelations:

    def test_causal_relations_is_list_of_strings(self):
        assert isinstance(CAUSAL_RELATIONS, list)
        assert all(isinstance(r, str) for r in CAUSAL_RELATIONS)

    def test_causal_relations_list_alias_identical(self):
        assert CAUSAL_RELATIONS is CAUSAL_RELATIONS_LIST

    def test_causal_relations_no_duplicates(self):
        assert len(CAUSAL_RELATIONS) == len(set(CAUSAL_RELATIONS))

    def test_causal_relations_all_enum_values_present(self):
        enum_values = {r.value for r in CausalRelation}
        assert enum_values == set(CAUSAL_RELATIONS)

    def test_required_relations_present(self):
        required = {
            "causes", "enables", "prevents", "implies",
            "contradicts", "leads_to", "supports", "weakens",
            "requires", "precedes", "part_of", "instance_of",
            "correlates", "analogous_to", "follows_from", "equivalent",
        }
        assert required.issubset(set(CAUSAL_RELATIONS)), (
            f"Faltan: {required - set(CAUSAL_RELATIONS)}"
        )

    def test_causal_relations_index_stable(self):
        """El índice numérico de cada relación es estable — BilinearEdgeDetector depende de esto."""
        causes_idx = CAUSAL_RELATIONS.index("causes")
        contradicts_idx = CAUSAL_RELATIONS.index("contradicts")
        assert causes_idx != contradicts_idx
        assert CAUSAL_RELATIONS[causes_idx] == "causes"
        assert CAUSAL_RELATIONS[contradicts_idx] == "contradicts"

    def test_inhibitory_relations_are_subset(self):
        assert INHIBITORY_RELATIONS.issubset(set(CAUSAL_RELATIONS))
        assert "prevents" in INHIBITORY_RELATIONS
        assert "contradicts" in INHIBITORY_RELATIONS
        assert "weakens" in INHIBITORY_RELATIONS

    def test_positive_relations_are_subset(self):
        assert POSITIVE_RELATIONS.issubset(set(CAUSAL_RELATIONS))
        assert "causes" in POSITIVE_RELATIONS
        assert "enables" in POSITIVE_RELATIONS

    def test_symmetric_relations_are_subset(self):
        assert SYMMETRIC_RELATIONS.issubset(set(CAUSAL_RELATIONS))

    def test_inhibitory_and_positive_are_disjoint(self):
        assert not INHIBITORY_RELATIONS.intersection(POSITIVE_RELATIONS)

    def test_node_types_completeness(self):
        assert "entity" in NODE_TYPES
        assert "event" in NODE_TYPES
        assert "state" in NODE_TYPES
        assert "action" in NODE_TYPES
        assert "hypothesis" in NODE_TYPES
        assert "fact" in NODE_TYPES
        assert "question" in NODE_TYPES

    def test_contradiction_pairs_are_tuples_of_relations(self):
        for rel_a, rel_b in CONTRADICTION_PAIRS:
            assert isinstance(rel_a, CausalRelation)
            assert isinstance(rel_b, CausalRelation)
            assert rel_a != rel_b


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — CAUSAL NODE
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalNode:

    def test_basic_construction(self):
        node = CausalNode(node_id="n1", label="Inflación")
        assert node.node_id == "n1"
        assert node.label == "Inflación"
        assert node.node_type == NodeType.ENTITY
        assert node.confidence == 1.0
        assert node.grounded is False
        assert node.vector is None
        assert node.metadata == {}

    def test_all_fields(self):
        node = CausalNode(
            node_id="n2",
            label="Crisis",
            node_type=NodeType.EVENT,
            confidence=0.85,
            grounded=True,
            vector=[0.1, 0.2, 0.3],
            metadata={"source": "text"},
        )
        assert node.node_type == NodeType.EVENT
        assert node.confidence == 0.85
        assert node.grounded is True
        assert node.vector == [0.1, 0.2, 0.3]
        assert node.metadata["source"] == "text"

    def test_node_type_from_string(self):
        node = CausalNode(node_id="n3", label="X", node_type="hypothesis")
        assert node.node_type == NodeType.HYPOTHESIS

    def test_confidence_boundary_valid(self):
        CausalNode(node_id="a", label="A", confidence=0.0)
        CausalNode(node_id="b", label="B", confidence=1.0)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            CausalNode(node_id="x", label="X", confidence=-0.01)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            CausalNode(node_id="x", label="X", confidence=1.001)

    def test_hash_by_node_id(self):
        n1 = CausalNode(node_id="n1", label="A")
        n2 = CausalNode(node_id="n1", label="B")  # mismo id, distinto label
        assert hash(n1) == hash(n2)

    def test_equality_by_node_id(self):
        n1 = CausalNode(node_id="n1", label="A")
        n2 = CausalNode(node_id="n1", label="diferente")
        n3 = CausalNode(node_id="n2", label="A")
        assert n1 == n2
        assert n1 != n3

    def test_node_in_set(self):
        n1 = CausalNode(node_id="n1", label="A")
        n2 = CausalNode(node_id="n1", label="B")
        s = {n1}
        assert n2 in s

    def test_repr_contains_key_fields(self):
        node = CausalNode(node_id="n1", label="test", confidence=0.75)
        r = repr(node)
        assert "n1" in r
        assert "test" in r
        assert "0.75" in r


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — CAUSAL EDGE
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalEdge:

    def test_basic_construction(self):
        edge = CausalEdge(
            source_id="A", target_id="B", relation=CausalRelation.CAUSES
        )
        assert edge.source_id == "A"
        assert edge.target_id == "B"
        assert edge.relation == CausalRelation.CAUSES
        assert edge.strength == 1.0
        assert edge.confidence == 1.0
        assert edge.source_idx == -1  # No asignado hasta agregar al grafo
        assert edge.target_idx == -1

    def test_relation_from_string(self):
        edge = CausalEdge(source_id="A", target_id="B", relation="prevents")
        assert edge.relation == CausalRelation.PREVENTS

    def test_self_loop_raises(self):
        with pytest.raises(ValueError, match="Self-loops"):
            CausalEdge(source_id="A", target_id="A", relation=CausalRelation.CAUSES)

    def test_strength_boundary_valid(self):
        CausalEdge(source_id="A", target_id="B", relation=CausalRelation.CAUSES, strength=0.0)
        CausalEdge(source_id="A", target_id="B", relation=CausalRelation.CAUSES, strength=1.0)

    def test_strength_out_of_range_raises(self):
        with pytest.raises(ValueError, match="strength"):
            CausalEdge(source_id="A", target_id="B", relation=CausalRelation.CAUSES, strength=1.5)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            CausalEdge(source_id="A", target_id="B", relation=CausalRelation.CAUSES, confidence=-0.1)

    def test_edge_id_autogenerated(self):
        e1 = CausalEdge(source_id="A", target_id="B", relation=CausalRelation.CAUSES)
        e2 = CausalEdge(source_id="A", target_id="B", relation=CausalRelation.CAUSES)
        assert e1.edge_id != e2.edge_id

    def test_hash_and_equality_by_edge_id(self):
        e1 = CausalEdge(source_id="A", target_id="B", relation=CausalRelation.CAUSES)
        e2 = CausalEdge(source_id="A", target_id="B", relation=CausalRelation.CAUSES)
        assert e1 != e2  # Distintos edge_id
        e3 = CausalEdge(source_id="A", target_id="B", relation=CausalRelation.CAUSES, edge_id=e1.edge_id)
        assert e1 == e3

    def test_is_inhibitory(self):
        e_prevents = make_edge("A", "B", CausalRelation.PREVENTS)
        e_contradicts = make_edge("A", "B", CausalRelation.CONTRADICTS)
        e_weakens = make_edge("A", "B", CausalRelation.WEAKENS)
        e_causes = make_edge("A", "B", CausalRelation.CAUSES)

        assert e_prevents.is_inhibitory is True
        assert e_contradicts.is_inhibitory is True
        assert e_weakens.is_inhibitory is True
        assert e_causes.is_inhibitory is False

    def test_is_positive(self):
        e_causes = make_edge("A", "B", CausalRelation.CAUSES)
        e_enables = make_edge("A", "B", CausalRelation.ENABLES)
        e_prevents = make_edge("A", "B", CausalRelation.PREVENTS)

        assert e_causes.is_positive is True
        assert e_enables.is_positive is True
        assert e_prevents.is_positive is False

    def test_is_symmetric(self):
        e_contradicts = make_edge("A", "B", CausalRelation.CONTRADICTS)
        e_correlates = make_edge("A", "B", CausalRelation.CORRELATES)
        e_causes = make_edge("A", "B", CausalRelation.CAUSES)

        assert e_contradicts.is_symmetric is True
        assert e_correlates.is_symmetric is True
        assert e_causes.is_symmetric is False

    def test_is_temporal(self):
        e_precedes = make_edge("A", "B", CausalRelation.PRECEDES)
        e_leads_to = make_edge("A", "B", CausalRelation.LEADS_TO)
        e_causes = make_edge("A", "B", CausalRelation.CAUSES)

        assert e_precedes.is_temporal is True
        assert e_leads_to.is_temporal is True
        assert e_causes.is_temporal is False

    def test_is_structural(self):
        e_part_of = make_edge("A", "B", CausalRelation.PART_OF)
        e_causes = make_edge("A", "B", CausalRelation.CAUSES)

        assert e_part_of.is_structural is True
        assert e_causes.is_structural is False

    def test_repr(self):
        edge = CausalEdge(source_id="A", target_id="B", relation=CausalRelation.CAUSES)
        r = repr(edge)
        assert "A" in r
        assert "B" in r
        assert "causes" in r


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — CAUSAL GRAPH — CONSTRUCCIÓN Y ACCESO
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalGraphConstruction:

    def test_empty_graph(self):
        g = CausalGraph(graph_id="g0")
        assert len(g) == 0
        assert g.nodes == []
        assert g.edges == []

    def test_add_node_returns_self(self):
        g = CausalGraph()
        n = make_node("A")
        result = g.add_node(n)
        assert result is g

    def test_add_node_idempotent(self):
        g = CausalGraph()
        n = make_node("A")
        g.add_node(n).add_node(n).add_node(n)
        assert len(g) == 1

    def test_add_node_chainable(self):
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B")).add_node(make_node("C"))
        assert len(g) == 3

    def test_add_edge_returns_self(self):
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        e = make_edge("A", "B")
        result = g.add_edge(e)
        assert result is g

    def test_add_edge_missing_source_raises(self):
        g = CausalGraph()
        g.add_node(make_node("B"))
        with pytest.raises(ValueError, match="Source node"):
            g.add_edge(make_edge("A", "B"))

    def test_add_edge_missing_target_raises(self):
        g = CausalGraph()
        g.add_node(make_node("A"))
        with pytest.raises(ValueError, match="Target node"):
            g.add_edge(make_edge("A", "B"))

    def test_node_indices_assigned_on_add_edge(self):
        g = CausalGraph()
        g.add_node(make_node("A"))
        g.add_node(make_node("B"))
        e = make_edge("A", "B")
        assert e.source_idx == -1
        g.add_edge(e)
        assert e.source_idx == 0
        assert e.target_idx == 1

    def test_node_index_order_of_insertion(self):
        g = CausalGraph()
        for nid in ["X", "Y", "Z"]:
            g.add_node(make_node(nid))
        idx = g.node_index
        assert idx["X"] == 0
        assert idx["Y"] == 1
        assert idx["Z"] == 2

    def test_get_node(self):
        g = CausalGraph()
        n = make_node("A")
        g.add_node(n)
        assert g.get_node("A") is n

    def test_get_node_missing_raises(self):
        g = CausalGraph()
        with pytest.raises(KeyError):
            g.get_node("nonexistent")

    def test_get_edge(self):
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        e = make_edge("A", "B")
        g.add_edge(e)
        assert g.get_edge(e.edge_id) is e

    def test_get_edge_missing_raises(self):
        g = CausalGraph()
        with pytest.raises(KeyError):
            g.get_edge("nonexistent")

    def test_contains(self):
        g = CausalGraph()
        g.add_node(make_node("A"))
        assert "A" in g
        assert "B" not in g

    def test_nodes_in_insertion_order(self):
        g = CausalGraph()
        for nid in ["C", "A", "B"]:
            g.add_node(make_node(nid))
        assert [n.node_id for n in g.nodes] == ["C", "A", "B"]

    def test_repr(self):
        g = CausalGraph(graph_id="test")
        g.add_node(make_node("A"))
        r = repr(g)
        assert "test" in r
        assert "1" in r

    def test_multiple_edges_same_pair(self):
        """Se permiten múltiples aristas entre el mismo par (con distinta relación)."""
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        e1 = make_edge("A", "B", CausalRelation.CAUSES)
        e2 = make_edge("A", "B", CausalRelation.SUPPORTS)
        g.add_edge(e1).add_edge(e2)
        assert len(g.edges) == 2

    def test_edges_between(self):
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        e1 = make_edge("A", "B", CausalRelation.CAUSES)
        e2 = make_edge("A", "B", CausalRelation.SUPPORTS)
        g.add_edge(e1).add_edge(e2)
        result = g.edges_between("A", "B")
        assert len(result) == 2
        relations = {e.relation for e in result}
        assert CausalRelation.CAUSES in relations
        assert CausalRelation.SUPPORTS in relations


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — REMOVE NODE Y REMOVE EDGE
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalGraphRemoval:

    def test_remove_node(self):
        g = linear_graph()
        g.remove_node("B")
        assert "B" not in g
        assert len(g) == 2

    def test_remove_node_cascades_edges(self):
        """Eliminar un nodo elimina todas sus aristas conectadas."""
        g = linear_graph()  # A→B→C
        assert len(g.edges) == 2
        g.remove_node("B")
        assert len(g.edges) == 0  # Ambas aristas A→B y B→C se eliminan

    def test_remove_node_recalculates_indices(self):
        g = CausalGraph()
        for nid in ["A", "B", "C"]:
            g.add_node(make_node(nid))
        e = make_edge("A", "C")
        g.add_edge(e)
        assert e.source_idx == 0
        assert e.target_idx == 2

        g.remove_node("B")  # B era índice 1 — C pasa a ser índice 1
        assert e.source_idx == 0
        assert e.target_idx == 1

    def test_remove_node_missing_raises(self):
        g = CausalGraph()
        with pytest.raises(KeyError):
            g.remove_node("nonexistent")

    def test_remove_edge(self):
        g = linear_graph()  # A→B→C
        edge_ids = [e.edge_id for e in g.edges]
        g.remove_edge(edge_ids[0])
        assert len(g.edges) == 1

    def test_remove_edge_missing_raises(self):
        g = CausalGraph()
        with pytest.raises(KeyError):
            g.remove_edge("nonexistent")

    def test_remove_edge_updates_out_edges(self):
        g = linear_graph()  # A→B→C
        e_ab = g.out_edges("A")[0]
        g.remove_edge(e_ab.edge_id)
        assert g.out_edges("A") == []
        assert len(g.in_edges("B")) == 0

    def test_remove_middle_node_leaves_ends(self):
        g = linear_graph()  # A→B→C
        g.remove_node("B")
        assert "A" in g
        assert "C" in g
        assert g.successors("A") == []
        assert g.predecessors("C") == []


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — SUCCESSORS / PREDECESSORS
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalGraphNeighbors:

    def test_successors(self):
        g = linear_graph()  # A→B→C
        succ_a = {n.node_id for n in g.successors("A")}
        assert succ_a == {"B"}

    def test_predecessors(self):
        g = linear_graph()  # A→B→C
        pred_c = {n.node_id for n in g.predecessors("C")}
        assert pred_c == {"B"}

    def test_isolated_node_no_neighbors(self):
        g = CausalGraph()
        g.add_node(make_node("X"))
        assert g.successors("X") == []
        assert g.predecessors("X") == []

    def test_multiple_successors(self):
        g = CausalGraph()
        for nid in ["A", "B", "C", "D"]:
            g.add_node(make_node(nid))
        g.add_edge(make_edge("A", "B"))
        g.add_edge(make_edge("A", "C"))
        g.add_edge(make_edge("A", "D"))
        succ = {n.node_id for n in g.successors("A")}
        assert succ == {"B", "C", "D"}


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — DETECT CYCLES
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectCycles:

    def test_empty_graph_no_cycles(self):
        g = CausalGraph()
        assert g.detect_cycles() == []

    def test_single_node_no_cycles(self):
        g = CausalGraph()
        g.add_node(make_node("A"))
        assert g.detect_cycles() == []

    def test_linear_graph_no_cycles(self):
        g = linear_graph()  # A→B→C
        assert g.detect_cycles() == []

    def test_triangle_has_one_cycle(self):
        g = triangle_graph()  # A→B→C→A
        cycles = g.detect_cycles()
        assert len(cycles) >= 1
        # El ciclo debe contener los 3 nodos
        all_in_cycle = set()
        for cycle in cycles:
            all_in_cycle.update(cycle)
        assert "A" in all_in_cycle
        assert "B" in all_in_cycle
        assert "C" in all_in_cycle

    def test_two_node_cycle(self):
        """A→B y B→A forman un ciclo con relaciones asíncronas."""
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        g.add_edge(make_edge("A", "B", CausalRelation.CAUSES))
        g.add_edge(make_edge("B", "A", CausalRelation.CAUSES))
        cycles = g.detect_cycles()
        assert len(cycles) >= 1

    def test_symmetric_relation_not_a_cycle(self):
        """
        A --CONTRADICTS--> B + B --CONTRADICTS--> A
        es la representación correcta de una relación simétrica,
        NO es un ciclo de razonamiento.
        """
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        g.add_edge(make_edge("A", "B", CausalRelation.CONTRADICTS))
        g.add_edge(make_edge("B", "A", CausalRelation.CONTRADICTS))
        cycles = g.detect_cycles()
        assert cycles == [], (
            "CONTRADICTS simétrico no debe reportarse como ciclo"
        )

    def test_symmetric_equivalent_not_a_cycle(self):
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        g.add_edge(make_edge("A", "B", CausalRelation.EQUIVALENT))
        g.add_edge(make_edge("B", "A", CausalRelation.EQUIVALENT))
        assert g.detect_cycles() == []

    def test_dag_with_diamond_no_cycle(self):
        """
        Diamante: A→B, A→C, B→D, C→D. No hay ciclos.
        """
        g = CausalGraph()
        for nid in ["A", "B", "C", "D"]:
            g.add_node(make_node(nid))
        g.add_edge(make_edge("A", "B"))
        g.add_edge(make_edge("A", "C"))
        g.add_edge(make_edge("B", "D"))
        g.add_edge(make_edge("C", "D"))
        assert g.detect_cycles() == []

    def test_multiple_disconnected_cycles(self):
        """Dos ciclos independientes en el mismo grafo."""
        g = CausalGraph()
        # Ciclo 1: X→Y→X
        g.add_node(make_node("X")).add_node(make_node("Y"))
        g.add_edge(make_edge("X", "Y")).add_edge(make_edge("Y", "X"))
        # Ciclo 2: P→Q→R→P
        g.add_node(make_node("P")).add_node(make_node("Q")).add_node(make_node("R"))
        g.add_edge(make_edge("P", "Q")).add_edge(make_edge("Q", "R")).add_edge(make_edge("R", "P"))

        cycles = g.detect_cycles()
        assert len(cycles) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — FIND CONTRADICTIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestFindContradictions:

    def test_empty_graph_no_contradictions(self):
        assert CausalGraph().find_contradictions() == []

    def test_no_contradictions(self):
        g = linear_graph()  # A→B→C, todas CAUSES
        assert g.find_contradictions() == []

    def test_direct_contradiction_causes_prevents(self):
        """A CAUSES B y A PREVENTS B es una contradicción directa."""
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        g.add_edge(make_edge("A", "B", CausalRelation.CAUSES))
        g.add_edge(make_edge("A", "B", CausalRelation.PREVENTS))
        contradictions = g.find_contradictions()
        assert len(contradictions) == 1

    def test_direct_contradiction_supports_weakens(self):
        """A SUPPORTS B y A WEAKENS B es una contradicción."""
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        g.add_edge(make_edge("A", "B", CausalRelation.SUPPORTS))
        g.add_edge(make_edge("A", "B", CausalRelation.WEAKENS))
        contradictions = g.find_contradictions()
        assert len(contradictions) == 1

    def test_direct_contradiction_implies_contradicts(self):
        """A IMPLIES B y A CONTRADICTS B es imposible."""
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        g.add_edge(make_edge("A", "B", CausalRelation.IMPLIES))
        g.add_edge(make_edge("A", "B", CausalRelation.CONTRADICTS))
        contradictions = g.find_contradictions()
        assert len(contradictions) == 1

    def test_inverse_contradiction(self):
        """A CAUSES B y B PREVENTS A es contradictorio en sentido inverso."""
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        g.add_edge(make_edge("A", "B", CausalRelation.CAUSES))
        g.add_edge(make_edge("B", "A", CausalRelation.PREVENTS))
        contradictions = g.find_contradictions()
        assert len(contradictions) == 1

    def test_contradiction_no_duplicates(self):
        """Cada par contradictorio aparece exactamente una vez."""
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        g.add_edge(make_edge("A", "B", CausalRelation.CAUSES))
        g.add_edge(make_edge("A", "B", CausalRelation.PREVENTS))
        contradictions = g.find_contradictions()
        assert len(contradictions) == 1

    def test_compatible_parallel_edges_not_contradiction(self):
        """A CAUSES B y A SUPPORTS B son compatibles."""
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        g.add_edge(make_edge("A", "B", CausalRelation.CAUSES))
        g.add_edge(make_edge("A", "B", CausalRelation.SUPPORTS))
        assert g.find_contradictions() == []

    def test_contradiction_returns_the_edges(self):
        """Los objetos retornados son las aristas originales."""
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        e1 = make_edge("A", "B", CausalRelation.CAUSES)
        e2 = make_edge("A", "B", CausalRelation.PREVENTS)
        g.add_edge(e1).add_edge(e2)
        contradictions = g.find_contradictions()
        pair = contradictions[0]
        edge_ids = {pair[0].edge_id, pair[1].edge_id}
        assert e1.edge_id in edge_ids
        assert e2.edge_id in edge_ids

    def test_multiple_contradictions(self):
        """Varios pares contradictorios en el mismo grafo."""
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B")).add_node(make_node("C"))
        # Par 1: A↔B
        g.add_edge(make_edge("A", "B", CausalRelation.CAUSES))
        g.add_edge(make_edge("A", "B", CausalRelation.PREVENTS))
        # Par 2: A↔C
        g.add_edge(make_edge("A", "C", CausalRelation.SUPPORTS))
        g.add_edge(make_edge("A", "C", CausalRelation.WEAKENS))
        contradictions = g.find_contradictions()
        assert len(contradictions) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — HAS PATH
# ─────────────────────────────────────────────────────────────────────────────

class TestHasPath:

    def test_direct_edge(self):
        g = linear_graph()  # A→B→C
        assert g.has_path("A", "B") is True

    def test_indirect_path(self):
        g = linear_graph()  # A→B→C
        assert g.has_path("A", "C") is True

    def test_no_path(self):
        g = linear_graph()  # A→B→C (dirigido)
        assert g.has_path("C", "A") is False

    def test_same_node(self):
        g = linear_graph()
        # Un nodo no tiene camino a sí mismo en un grafo acíclico
        assert g.has_path("A", "A") is False

    def test_cycle_has_path(self):
        g = triangle_graph()  # A→B→C→A
        assert g.has_path("A", "C") is True
        assert g.has_path("C", "A") is True

    def test_missing_node(self):
        g = linear_graph()
        assert g.has_path("A", "Z") is False
        assert g.has_path("Z", "A") is False


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — TO ADJACENCY Y SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphRepresentations:

    def test_to_adjacency_structure(self):
        g = linear_graph()  # A→B→C
        adj = g.to_adjacency()
        assert "A" in adj
        assert "B" in adj["A"]
        assert "causes" in adj["A"]["B"]

    def test_to_adjacency_empty(self):
        g = CausalGraph()
        adj = g.to_adjacency()
        assert adj == {}

    def test_to_adjacency_multiple_relations(self):
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        g.add_edge(make_edge("A", "B", CausalRelation.CAUSES))
        g.add_edge(make_edge("A", "B", CausalRelation.SUPPORTS))
        adj = g.to_adjacency()
        relations = adj["A"]["B"]
        assert "causes" in relations
        assert "supports" in relations

    def test_summary_basic_fields(self):
        g = linear_graph()
        s = g.summary()
        assert s["n_nodes"] == 3
        assert s["n_edges"] == 2
        assert s["n_cycles"] == 0
        assert s["n_contradictions"] == 0

    def test_summary_with_cycle(self):
        g = triangle_graph()
        s = g.summary()
        assert s["n_cycles"] >= 1

    def test_summary_with_contradiction(self):
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        g.add_edge(make_edge("A", "B", CausalRelation.CAUSES))
        g.add_edge(make_edge("A", "B", CausalRelation.PREVENTS))
        s = g.summary()
        assert s["n_contradictions"] == 1

    def test_summary_grounded_count(self):
        g = CausalGraph()
        g.add_node(CausalNode("A", "A", grounded=True))
        g.add_node(CausalNode("B", "B", grounded=False))
        g.add_node(CausalNode("C", "C", grounded=True))
        s = g.summary()
        assert s["grounded_nodes"] == 2

    def test_summary_node_types(self):
        g = CausalGraph()
        g.add_node(make_node("A", node_type=NodeType.ENTITY))
        g.add_node(make_node("B", node_type=NodeType.HYPOTHESIS))
        s = g.summary()
        assert s["node_types"]["entity"] == 1
        assert s["node_types"]["hypothesis"] == 1

    def test_summary_has_questions(self):
        g = CausalGraph(root_question="¿qué causa la inflación?")
        g.add_node(make_node("Q", node_type=NodeType.QUESTION))
        s = g.summary()
        assert s["has_questions"] is True

    def test_summary_avg_confidence(self):
        g = CausalGraph()
        g.add_node(CausalNode("A", "A", confidence=0.8))
        g.add_node(CausalNode("B", "B", confidence=0.4))
        s = g.summary()
        assert abs(s["avg_node_confidence"] - 0.6) < 1e-4

    def test_grounded_mask_order(self):
        g = CausalGraph()
        g.add_node(CausalNode("A", "A", grounded=True))
        g.add_node(CausalNode("B", "B", grounded=False))
        g.add_node(CausalNode("C", "C", grounded=True))
        mask = g.grounded_mask
        assert mask == [True, False, True]


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — INTEGRACIÓN (WORKFLOW COMPLETO)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_causal_chain_workflow(self):
        """
        Simula el flujo GraphConstructor → CausalGraph → CEC.
        Construye: inflación → sube_tipos → baja_crédito → baja_inversión
        Verifica que el grafo es acíclico y sin contradicciones.
        """
        g = CausalGraph(graph_id="economia", root_question="¿qué pasa si sube la inflación?")

        nodes = [
            CausalNode("inflacion",     "Inflación",            NodeType.STATE,     confidence=0.9, grounded=True),
            CausalNode("sube_tipos",    "Subida de tipos",      NodeType.ACTION,    confidence=0.85),
            CausalNode("baja_credito",  "Caída del crédito",    NodeType.STATE,     confidence=0.8),
            CausalNode("baja_inversion","Caída de inversión",   NodeType.STATE,     confidence=0.75),
            CausalNode("respuesta",     "Contracción económica",NodeType.QUESTION,  confidence=0.5),
        ]
        for n in nodes:
            g.add_node(n)

        edges = [
            CausalEdge("inflacion",    "sube_tipos",     CausalRelation.CAUSES,   strength=0.9, confidence=0.85),
            CausalEdge("sube_tipos",   "baja_credito",   CausalRelation.LEADS_TO, strength=0.8, confidence=0.8),
            CausalEdge("baja_credito", "baja_inversion", CausalRelation.CAUSES,   strength=0.75, confidence=0.7),
            CausalEdge("inflacion",    "baja_inversion", CausalRelation.LEADS_TO, strength=0.6, confidence=0.65),
            CausalEdge("baja_inversion","respuesta",     CausalRelation.IMPLIES,  strength=0.7, confidence=0.6),
        ]
        for e in edges:
            g.add_edge(e)

        assert len(g) == 5
        assert len(g.edges) == 5
        assert g.detect_cycles() == []
        assert g.find_contradictions() == []
        assert g.has_path("inflacion", "respuesta")

        s = g.summary()
        assert s["grounded_nodes"] == 1
        assert s["has_questions"] is True
        assert s["n_cycles"] == 0
        assert s["n_contradictions"] == 0

    def test_tensor_indices_consistent_after_removal(self):
        """
        Los índices source_idx/target_idx deben ser válidos para indexar
        un tensor de [n_nodes, dim] tras eliminar un nodo.
        """
        g = CausalGraph()
        for nid in ["A", "B", "C", "D"]:
            g.add_node(make_node(nid))
        e_ac = make_edge("A", "C")
        e_bd = make_edge("B", "D")
        g.add_edge(e_ac).add_edge(e_bd)

        # Eliminar B — C pasa de índice 2 a 1, D de 3 a 2
        g.remove_node("B")

        n_nodes = len(g)  # 3 nodos: A(0), C(1), D(2)
        assert e_ac.source_idx < n_nodes
        assert e_ac.target_idx < n_nodes
        # e_bd fue eliminada junto con B
        remaining_edge_ids = {e.edge_id for e in g.edges}
        assert e_ac.edge_id in remaining_edge_ids
        assert e_bd.edge_id not in remaining_edge_ids

    def test_causal_relations_indexable_for_bilinear_detector(self):
        """
        Verifica que CAUSAL_RELATIONS puede usarse como vocabulario
        estable para el BilinearEdgeDetector (n_relation_types = len(CAUSAL_RELATIONS)).
        """
        n_types = len(CAUSAL_RELATIONS)
        assert n_types > 0
        # Simular acceso por índice como hace BilinearEdgeDetector
        for i, rel_name in enumerate(CAUSAL_RELATIONS):
            assert CAUSAL_RELATIONS[i] == rel_name
            assert CausalRelation(rel_name)  # el nombre es válido en el Enum

    def test_typed_message_passing_lookup(self):
        """
        Verifica que cada relación en CAUSAL_RELATIONS puede usarse
        como clave en un dict (como hace TypedMessagePassing con message_fns).
        """
        message_fns = {rel: f"fn_{rel}" for rel in CAUSAL_RELATIONS}
        g = CausalGraph()
        g.add_node(make_node("A")).add_node(make_node("B"))
        e = make_edge("A", "B", CausalRelation.CAUSES)
        g.add_edge(e)

        # TypedMessagePassing busca: self.message_fns[edge.relation.value]
        assert e.relation.value in message_fns
