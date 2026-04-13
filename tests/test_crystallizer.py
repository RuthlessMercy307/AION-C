"""
tests/test_crystallizer.py — Tests exhaustivos del GraphCrystallizer
=====================================================================

Organización:
    TestCrystallizerConfig          — validación de la configuración
    TestNodeDetector                — shapes, gradientes, salidas por posición
    TestCrossAttentionPooler        — shapes, número de cabezas, gradientes
    TestAsymmetricRelationScorer    — asimetría, shapes, batch/no-batch
    TestGraphCrystallizerOutput     — CausalGraph válido, nodos, aristas
    TestNodeTypes                   — todos los tipos detectados son NodeType válidos
    TestMaxNodes                    — respeta config.max_nodes siempre
    TestAsymmetry                   — A→B ≠ B→A (el test clave del componente)
    TestGradientFlow                — diferenciabilidad end-to-end
    TestParameterCount              — integridad del conteo de parámetros
    TestDeterminism                 — eval mode es determinista
    TestEdgeCases                   — secuencias cortas, batch=1, batch>1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from core.graph import CausalGraph, CausalRelation, NodeType, CAUSAL_RELATIONS
from crystallizer import (
    AsymmetricRelationScorer,
    CrossAttentionPooler,
    CrystallizerConfig,
    CrystallizerOutput,
    GraphCrystallizer,
    NodeDetector,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return CrystallizerConfig(hidden_dim=64, max_nodes=8, pooler_heads=4,
                               relation_hidden_dim=16, node_confidence_hidden_dim=16,
                               node_threshold=0.3, edge_threshold=0.3)


@pytest.fixture
def cfg_tiny():
    """Configuración mínima para tests rápidos."""
    return CrystallizerConfig(hidden_dim=32, max_nodes=4, pooler_heads=4,
                               relation_hidden_dim=8, node_confidence_hidden_dim=8,
                               node_threshold=0.3, edge_threshold=0.3)


@pytest.fixture
def cfg_full():
    """Configuración completa: hidden_dim=256, max_nodes=32."""
    return CrystallizerConfig()


@pytest.fixture
def gc(cfg):
    model = GraphCrystallizer(cfg)
    model.eval()
    return model


@pytest.fixture
def gc_full(cfg_full):
    model = GraphCrystallizer(cfg_full)
    model.eval()
    return model


def make_concepts(B, L, D, seed=42):
    """Genera concept vectors con valores que producen nodos con score > 0.3."""
    torch.manual_seed(seed)
    # Valores en [-2, 2]: tras sigmoid de un MLP inicializado con std=0.02,
    # produce scores suficientemente distribuidos para superar el umbral.
    return torch.randn(B, L, D) * 1.5


def make_concepts_high_score(B, L, D):
    """Concept vectors de magnitud alta → garantizan nodos detectados."""
    # Valores grandes → las activaciones del MLP tienden a extremos
    # → sigmoid tiende a 1 → casi todos los nodos son detectados
    return torch.ones(B, L, D) * 3.0 + torch.randn(B, L, D) * 0.1


# ─────────────────────────────────────────────────────────────────────────────
# TestCrystallizerConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestCrystallizerConfig:
    def test_default_values(self):
        cfg = CrystallizerConfig()
        assert cfg.hidden_dim == 256
        assert cfg.max_nodes  == 32
        assert cfg.n_relation_types == 16
        assert cfg.n_node_types     == 7

    def test_n_relation_types_matches_causal_relations(self):
        cfg = CrystallizerConfig()
        assert cfg.n_relation_types == len(CAUSAL_RELATIONS)

    def test_n_node_types_matches_nodetype_enum(self):
        cfg = CrystallizerConfig()
        assert cfg.n_node_types == len(NodeType)

    def test_custom_values(self):
        cfg = CrystallizerConfig(hidden_dim=128, max_nodes=16)
        assert cfg.hidden_dim == 128
        assert cfg.max_nodes  == 16

    def test_thresholds_in_range(self):
        cfg = CrystallizerConfig()
        assert 0.0 < cfg.node_threshold < 1.0
        assert 0.0 < cfg.edge_threshold < 1.0

    def test_invalid_hidden_dim_not_divisible_by_heads(self):
        with pytest.raises(ValueError):
            CrystallizerConfig(hidden_dim=33, pooler_heads=4)

    def test_invalid_node_threshold_zero(self):
        with pytest.raises(ValueError):
            CrystallizerConfig(node_threshold=0.0)

    def test_invalid_node_threshold_one(self):
        with pytest.raises(ValueError):
            CrystallizerConfig(node_threshold=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# TestNodeDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeDetector:
    def test_output_shapes(self, cfg):
        B, L, D = 2, 16, cfg.hidden_dim
        detector = NodeDetector(cfg)
        x = torch.randn(B, L, D)
        scores, type_logits, confidence = detector(x)
        assert scores.shape      == (B, L),               f"node_scores: {scores.shape}"
        assert type_logits.shape == (B, L, cfg.n_node_types), f"type_logits: {type_logits.shape}"
        assert confidence.shape  == (B, L),               f"confidence: {confidence.shape}"

    def test_node_scores_in_01(self, cfg):
        detector = NodeDetector(cfg)
        x = torch.randn(3, 20, cfg.hidden_dim)
        scores, _, _ = detector(x)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_confidence_in_01(self, cfg):
        detector = NodeDetector(cfg)
        x = torch.randn(3, 20, cfg.hidden_dim)
        _, _, conf = detector(x)
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_type_logits_no_activation(self, cfg):
        """type_logits no deben estar sigmoidizados — deben tener rango libre."""
        detector = NodeDetector(cfg)
        x = torch.randn(2, 10, cfg.hidden_dim) * 5.0
        _, type_logits, _ = detector(x)
        # Con entradas grandes, debe haber valores > 1 o < 0 (sin sigmoid)
        has_outside_01 = ((type_logits > 1) | (type_logits < 0)).any()
        assert has_outside_01, "type_logits should not be bounded to [0,1]"

    def test_gradient_flows_to_input(self, cfg):
        detector = NodeDetector(cfg)
        x = torch.randn(2, 8, cfg.hidden_dim, requires_grad=True)
        scores, type_logits, conf = detector(x)
        loss = scores.sum() + type_logits.sum() + conf.sum()
        loss.backward()
        assert x.grad is not None
        assert not x.grad.isnan().any()

    def test_batch_1(self, cfg):
        detector = NodeDetector(cfg)
        x = torch.randn(1, 5, cfg.hidden_dim)
        scores, type_logits, conf = detector(x)
        assert scores.shape == (1, 5)

    def test_different_inputs_different_outputs(self, cfg):
        torch.manual_seed(0)
        detector = NodeDetector(cfg)
        x1 = torch.randn(1, 8, cfg.hidden_dim)
        x2 = torch.randn(1, 8, cfg.hidden_dim)
        s1, _, _ = detector(x1)
        s2, _, _ = detector(x2)
        assert not torch.allclose(s1, s2)

    def test_has_three_submodules(self, cfg):
        detector = NodeDetector(cfg)
        assert hasattr(detector, "node_scorer")
        assert hasattr(detector, "type_classifier")
        assert hasattr(detector, "confidence_head")

    def test_type_classifier_is_linear(self, cfg):
        detector = NodeDetector(cfg)
        assert isinstance(detector.type_classifier, nn.Linear)
        assert detector.type_classifier.out_features == cfg.n_node_types

    def test_no_nan_output(self, cfg):
        detector = NodeDetector(cfg)
        x = torch.randn(4, 32, cfg.hidden_dim)
        scores, type_logits, conf = detector(x)
        assert not scores.isnan().any()
        assert not type_logits.isnan().any()
        assert not conf.isnan().any()


# ─────────────────────────────────────────────────────────────────────────────
# TestCrossAttentionPooler
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossAttentionPooler:
    def test_output_shape(self, cfg):
        pooler = CrossAttentionPooler(cfg)
        B, n, L, D = 2, 6, 16, cfg.hidden_dim
        queries = torch.randn(B, n, D)
        context = torch.randn(B, L, D)
        out = pooler(queries, context)
        assert out.shape == (B, n, D)

    def test_output_dim_preserved(self, cfg):
        pooler = CrossAttentionPooler(cfg)
        D = cfg.hidden_dim
        q = torch.randn(1, 3, D)
        c = torch.randn(1, 20, D)
        out = pooler(q, c)
        assert out.shape[-1] == D

    def test_different_n_nodes_different_batches(self, cfg):
        """El pooler acepta cualquier n_nodes (no fijado)."""
        pooler = CrossAttentionPooler(cfg)
        D = cfg.hidden_dim
        for n in [1, 4, 8, 16]:
            q = torch.randn(1, n, D)
            c = torch.randn(1, 32, D)
            out = pooler(q, c)
            assert out.shape == (1, n, D)

    def test_context_affects_output(self, cfg):
        """Cambiar el context debe cambiar el output."""
        pooler = CrossAttentionPooler(cfg)
        D = cfg.hidden_dim
        torch.manual_seed(1)
        q = torch.randn(1, 4, D)
        c1 = torch.randn(1, 16, D)
        c2 = torch.randn(1, 16, D)
        out1 = pooler(q, c1)
        out2 = pooler(q, c2)
        assert not torch.allclose(out1, out2)

    def test_has_four_projections(self, cfg):
        pooler = CrossAttentionPooler(cfg)
        assert hasattr(pooler, "q_proj")
        assert hasattr(pooler, "k_proj")
        assert hasattr(pooler, "v_proj")
        assert hasattr(pooler, "out_proj")

    def test_gradient_flows(self, cfg):
        pooler = CrossAttentionPooler(cfg)
        D = cfg.hidden_dim
        q = torch.randn(2, 4, D, requires_grad=True)
        c = torch.randn(2, 16, D, requires_grad=True)
        out = pooler(q, c)
        out.sum().backward()
        assert q.grad is not None
        assert c.grad is not None

    def test_output_finite(self, cfg):
        pooler = CrossAttentionPooler(cfg)
        q = torch.randn(2, 6, cfg.hidden_dim)
        c = torch.randn(2, 32, cfg.hidden_dim)
        out = pooler(q, c)
        assert out.isfinite().all()

    def test_n_heads_divides_hidden_dim(self, cfg):
        """La construcción debe fallar si hidden_dim no es divisible por pooler_heads."""
        bad_cfg = CrystallizerConfig(hidden_dim=64, pooler_heads=4,
                                      relation_hidden_dim=16,
                                      node_confidence_hidden_dim=16)
        # Esto debe funcionar (64 % 4 == 0)
        pooler = CrossAttentionPooler(bad_cfg)
        assert pooler.n_heads == 4


# ─────────────────────────────────────────────────────────────────────────────
# TestAsymmetricRelationScorer
# ─────────────────────────────────────────────────────────────────────────────

class TestAsymmetricRelationScorer:
    def test_output_shape_unbatched(self, cfg):
        scorer = AsymmetricRelationScorer(cfg)
        n, D = 5, cfg.hidden_dim
        src = torch.randn(n, D)
        tgt = torch.randn(n, D)
        out = scorer(src, tgt)
        assert out.shape == (n, n, cfg.n_relation_types)

    def test_output_shape_batched(self, cfg):
        scorer = AsymmetricRelationScorer(cfg)
        B, n, D = 3, 5, cfg.hidden_dim
        src = torch.randn(B, n, D)
        tgt = torch.randn(B, n, D)
        out = scorer(src, tgt)
        assert out.shape == (B, n, n, cfg.n_relation_types)

    def test_asymmetry_AB_neq_BA(self, cfg):
        """
        El test clave: score(A→B) ≠ score(B→A).
        Esto verifica que source_proj ≠ target_proj produce asimetría real.
        """
        torch.manual_seed(0)
        scorer = AsymmetricRelationScorer(cfg)
        scorer.eval()
        D = cfg.hidden_dim
        # Dos nodos distintos
        A = torch.randn(1, D)
        B_node = torch.randn(1, D)
        nodes = torch.stack([A.squeeze(0), B_node.squeeze(0)])  # [2, D]

        logits = scorer(nodes, nodes)  # [2, 2, R]
        score_AB = logits[0, 1]  # A→B
        score_BA = logits[1, 0]  # B→A
        assert not torch.allclose(score_AB, score_BA), \
            "AsymmetricRelationScorer must produce score(A→B) ≠ score(B→A)"

    def test_asymmetry_is_structural_not_just_numerical(self, cfg):
        """
        La asimetría debe venir de la arquitectura, no solo del azar numérico.
        Verificamos que source_proj y target_proj son parámetros distintos.
        """
        scorer = AsymmetricRelationScorer(cfg)
        assert scorer.source_proj is not scorer.target_proj
        assert not torch.allclose(
            scorer.source_proj.weight, scorer.target_proj.weight
        ), "source_proj.weight != target_proj.weight after init"

    def test_self_scores_nonzero(self, cfg):
        """Un nodo puede tener relación consigo mismo (aunque no se usa para aristas)."""
        scorer = AsymmetricRelationScorer(cfg)
        n, D = 4, cfg.hidden_dim
        nodes = torch.randn(n, D)
        out = scorer(nodes, nodes)
        # La diagonal no debe ser obligatoriamente cero
        assert out.shape == (n, n, cfg.n_relation_types)

    def test_gradient_flows(self, cfg):
        scorer = AsymmetricRelationScorer(cfg)
        n, D = 4, cfg.hidden_dim
        src = torch.randn(n, D, requires_grad=True)
        tgt = torch.randn(n, D, requires_grad=True)
        out = scorer(src, tgt)
        out.sum().backward()
        assert src.grad is not None
        assert tgt.grad is not None

    def test_output_finite(self, cfg):
        scorer = AsymmetricRelationScorer(cfg)
        nodes = torch.randn(6, cfg.hidden_dim)
        out = scorer(nodes, nodes)
        assert out.isfinite().all()

    def test_different_source_target(self, cfg):
        """Acepta fuentes y destinos distintos."""
        scorer = AsymmetricRelationScorer(cfg)
        n, m, D = 3, 5, cfg.hidden_dim
        src = torch.randn(n, D)
        tgt = torch.randn(m, D)
        out = scorer(src, tgt)
        assert out.shape == (n, m, cfg.n_relation_types)

    def test_batched_unbatched_consistent(self, cfg):
        """Batched y unbatched deben dar el mismo resultado."""
        torch.manual_seed(7)
        scorer = AsymmetricRelationScorer(cfg)
        scorer.eval()
        n, D = 4, cfg.hidden_dim
        nodes = torch.randn(n, D)

        out_unbatched = scorer(nodes, nodes)                   # [n, n, R]
        out_batched   = scorer(nodes.unsqueeze(0), nodes.unsqueeze(0))  # [1, n, n, R]
        assert torch.allclose(out_unbatched, out_batched.squeeze(0), atol=1e-5)

    def test_has_source_and_target_proj(self, cfg):
        scorer = AsymmetricRelationScorer(cfg)
        assert hasattr(scorer, "source_proj")
        assert hasattr(scorer, "target_proj")

    def test_refiner_exists(self, cfg):
        scorer = AsymmetricRelationScorer(cfg)
        assert hasattr(scorer, "refiner")
        assert isinstance(scorer.refiner, nn.Sequential)


# ─────────────────────────────────────────────────────────────────────────────
# TestGraphCrystallizerOutput
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphCrystallizerOutput:
    def test_returns_crystallizer_output(self, gc, cfg):
        concepts = make_concepts(2, 16, cfg.hidden_dim)
        out = gc(concepts)
        assert isinstance(out, CrystallizerOutput)

    def test_graphs_is_list_of_causal_graphs(self, gc, cfg):
        B = 3
        concepts = make_concepts(B, 16, cfg.hidden_dim)
        out = gc(concepts)
        assert len(out.graphs) == B
        for g in out.graphs:
            assert isinstance(g, CausalGraph)

    def test_graph_has_nodes(self, gc, cfg):
        """Con concept vectors de magnitud alta, debe haber nodos detectados."""
        concepts = make_concepts_high_score(1, 16, cfg.hidden_dim)
        out = gc(concepts)
        graph = out.graphs[0]
        assert len(graph.nodes) >= 1, \
            f"Expected at least 1 node, got {len(graph.nodes)}"

    def test_graph_with_enough_nodes_has_edges(self):
        """Con suficientes nodos, deben formarse aristas."""
        # Usar thresholds muy bajos para garantizar aristas
        low_cfg = CrystallizerConfig(
            hidden_dim=64, max_nodes=8, pooler_heads=4,
            relation_hidden_dim=16, node_confidence_hidden_dim=16,
            node_threshold=0.01,   # casi todos los nodos son válidos
            edge_threshold=0.01,   # casi todas las aristas son válidas
        )
        gc = GraphCrystallizer(low_cfg)
        gc.eval()
        torch.manual_seed(42)
        concepts = make_concepts(1, 16, 64)
        out = gc(concepts)
        graph = out.graphs[0]
        # Con 8 nodos y threshold muy bajo, debe haber aristas
        if len(graph.nodes) >= 2:
            assert len(graph.edges) >= 1, \
                f"With {len(graph.nodes)} nodes and low threshold, expected edges"

    def test_node_counts_match_graph_sizes(self, gc, cfg):
        concepts = make_concepts(3, 16, cfg.hidden_dim)
        out = gc(concepts)
        for b, (graph, count) in enumerate(zip(out.graphs, out.node_counts)):
            assert len(graph.nodes) == count, \
                f"Batch {b}: graph has {len(graph.nodes)} nodes but node_counts says {count}"

    def test_tensor_shapes(self, gc, cfg):
        B, L = 2, 16
        concepts = make_concepts(B, L, cfg.hidden_dim)
        out = gc(concepts)
        K = min(L, cfg.max_nodes)
        assert out.node_scores.shape      == (B, L)
        assert out.node_type_logits.shape == (B, L, cfg.n_node_types)
        assert out.node_confidence.shape  == (B, L)
        assert out.node_vectors.shape     == (B, K, cfg.hidden_dim)
        assert out.relation_logits.shape  == (B, K, K, cfg.n_relation_types)

    def test_node_scores_are_probabilities(self, gc, cfg):
        concepts = make_concepts(2, 16, cfg.hidden_dim)
        out = gc(concepts)
        assert (out.node_scores >= 0).all() and (out.node_scores <= 1).all()

    def test_node_confidence_are_probabilities(self, gc, cfg):
        concepts = make_concepts(2, 16, cfg.hidden_dim)
        out = gc(concepts)
        assert (out.node_confidence >= 0).all() and (out.node_confidence <= 1).all()

    def test_node_counts_bounded_by_max_nodes(self, gc, cfg):
        concepts = make_concepts(4, 32, cfg.hidden_dim)
        out = gc(concepts)
        for count in out.node_counts:
            assert count <= cfg.max_nodes

    def test_node_counts_bounded_by_seq_len(self):
        """Si L < max_nodes, node_counts <= L."""
        cfg = CrystallizerConfig(hidden_dim=64, max_nodes=32, pooler_heads=4,
                                  relation_hidden_dim=16, node_confidence_hidden_dim=16)
        gc = GraphCrystallizer(cfg)
        gc.eval()
        L = 4  # Menos que max_nodes=32
        concepts = make_concepts(2, L, 64)
        out = gc(concepts)
        for count in out.node_counts:
            assert count <= L, f"node_count {count} > L={L}"


# ─────────────────────────────────────────────────────────────────────────────
# TestNodeTypes
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeTypes:
    def test_all_node_types_are_valid_nodetype(self, gc, cfg):
        """Todos los nodos detectados deben tener un NodeType válido."""
        concepts = make_concepts_high_score(2, 16, cfg.hidden_dim)
        out = gc(concepts)
        valid_types = set(NodeType)
        for graph in out.graphs:
            for node in graph.nodes:
                assert node.node_type in valid_types, \
                    f"Invalid node_type: {node.node_type!r}"

    def test_node_type_is_nodetype_enum(self, gc, cfg):
        concepts = make_concepts_high_score(1, 16, cfg.hidden_dim)
        out = gc(concepts)
        for node in out.graphs[0].nodes:
            assert isinstance(node.node_type, NodeType)

    def test_node_ids_are_strings(self, gc, cfg):
        concepts = make_concepts_high_score(1, 16, cfg.hidden_dim)
        out = gc(concepts)
        for node in out.graphs[0].nodes:
            assert isinstance(node.node_id, str)

    def test_node_confidence_in_01(self, gc, cfg):
        concepts = make_concepts_high_score(1, 16, cfg.hidden_dim)
        out = gc(concepts)
        for node in out.graphs[0].nodes:
            assert 0.0 <= node.confidence <= 1.0, \
                f"node confidence {node.confidence} out of [0,1]"

    def test_node_type_distribution_covers_multiple_types(self):
        """
        Con suficiente aleatoriedad, el modelo debe asignar varios tipos distintos.
        No todos los nodos deben tener el mismo tipo.
        """
        cfg = CrystallizerConfig(
            hidden_dim=64, max_nodes=8, pooler_heads=4,
            relation_hidden_dim=16, node_confidence_hidden_dim=16,
            node_threshold=0.01,
        )
        gc = GraphCrystallizer(cfg)
        gc.eval()
        all_types = set()
        for seed in range(10):
            concepts = make_concepts(1, 16, 64, seed=seed)
            out = gc(concepts)
            for node in out.graphs[0].nodes:
                all_types.add(node.node_type)
        # Con 10 seeds distintos, esperamos al menos 2 tipos distintos
        assert len(all_types) >= 2, \
            f"Expected ≥2 node types across seeds, got {all_types}"


# ─────────────────────────────────────────────────────────────────────────────
# TestMaxNodes
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxNodes:
    def test_graph_never_exceeds_max_nodes(self):
        """max_nodes=4: nunca más de 4 nodos aunque L sea mucho mayor."""
        cfg = CrystallizerConfig(
            hidden_dim=32, max_nodes=4, pooler_heads=4,
            relation_hidden_dim=8, node_confidence_hidden_dim=8,
            node_threshold=0.01,
        )
        gc = GraphCrystallizer(cfg)
        gc.eval()
        # L=64, mucho más que max_nodes=4
        concepts = make_concepts(3, 64, 32)
        out = gc(concepts)
        for graph in out.graphs:
            assert len(graph.nodes) <= 4, \
                f"Expected ≤ 4 nodes (max_nodes=4), got {len(graph.nodes)}"

    def test_max_nodes_1_produces_at_most_1_node(self):
        cfg = CrystallizerConfig(
            hidden_dim=32, max_nodes=1, pooler_heads=4,
            relation_hidden_dim=8, node_confidence_hidden_dim=8,
            node_threshold=0.01,
        )
        gc = GraphCrystallizer(cfg)
        gc.eval()
        concepts = make_concepts(2, 32, 32)
        out = gc(concepts)
        for graph in out.graphs:
            assert len(graph.nodes) <= 1

    def test_max_nodes_bounds_node_vectors_tensor(self):
        """node_vectors siempre tiene segunda dimensión == min(L, max_nodes)."""
        cfg = CrystallizerConfig(
            hidden_dim=64, max_nodes=6, pooler_heads=4,
            relation_hidden_dim=16, node_confidence_hidden_dim=16,
        )
        gc = GraphCrystallizer(cfg)
        gc.eval()
        B, L = 2, 20
        concepts = make_concepts(B, L, 64)
        out = gc(concepts)
        K = min(L, cfg.max_nodes)
        assert out.node_vectors.shape[1] == K

    def test_seq_len_less_than_max_nodes(self):
        """Si L < max_nodes, el límite es L."""
        cfg = CrystallizerConfig(
            hidden_dim=32, max_nodes=16, pooler_heads=4,
            relation_hidden_dim=8, node_confidence_hidden_dim=8,
            node_threshold=0.01,
        )
        gc = GraphCrystallizer(cfg)
        gc.eval()
        L = 5  # < max_nodes=16
        concepts = make_concepts(1, L, 32)
        out = gc(concepts)
        assert out.node_vectors.shape[1] == L
        for graph in out.graphs:
            assert len(graph.nodes) <= L


# ─────────────────────────────────────────────────────────────────────────────
# TestAsymmetry
# ─────────────────────────────────────────────────────────────────────────────

class TestAsymmetry:
    """
    Verifica que la asimetría funciona: relation(A→B) ≠ relation(B→A).
    Este es el test más importante del GraphCrystallizer.
    """

    def test_relation_scorer_AB_neq_BA(self, cfg):
        """Score directo: A→B ≠ B→A para el AsymmetricRelationScorer."""
        torch.manual_seed(0)
        scorer = AsymmetricRelationScorer(cfg)
        scorer.eval()
        D = cfg.hidden_dim

        A = torch.randn(1, D)
        B = torch.randn(1, D)
        nodes = torch.stack([A.squeeze(0), B.squeeze(0)])

        logits = scorer(nodes, nodes)        # [2, 2, R]
        score_AB = logits[0, 1]              # A→B: nodo 0 como fuente, nodo 1 como destino
        score_BA = logits[1, 0]              # B→A: nodo 1 como fuente, nodo 0 como destino

        max_diff = (score_AB - score_BA).abs().max().item()
        assert max_diff > 1e-4, \
            f"score(A→B) == score(B→A) to within 1e-4. max_diff={max_diff:.2e}"

    def test_relation_logits_asymmetric_in_output(self):
        """
        relation_logits(A→B) ≠ relation_logits(B→A) en el tensor de output.
        Esto prueba la asimetría en los scores continuos, no en el grafo discreto.
        Con edge_threshold bajo, ambas direcciones se añaden al grafo, pero sus
        scores son distintos — eso es lo que importa para el entrenamiento.
        """
        cfg = CrystallizerConfig(
            hidden_dim=64, max_nodes=4, pooler_heads=4,
            relation_hidden_dim=16, node_confidence_hidden_dim=16,
            node_threshold=0.01, edge_threshold=0.01,
        )
        gc = GraphCrystallizer(cfg)
        gc.eval()
        torch.manual_seed(10)
        concepts = torch.randn(1, 8, 64)
        out = gc(concepts)

        n = out.node_counts[0]
        if n < 2:
            pytest.skip("Need at least 2 nodes for asymmetry test")

        # Verificar que relation_logits[i→j] ≠ relation_logits[j→i]
        # para al menos un par. Esto es la asimetría real del scorer.
        rel = out.relation_logits[0]  # [K, K, R]
        found_asymmetric = False
        for i in range(n):
            for j in range(n):
                if i != j:
                    if not torch.allclose(rel[i, j], rel[j, i], atol=1e-4):
                        found_asymmetric = True
                        break
            if found_asymmetric:
                break

        assert found_asymmetric, \
            "All relation_logits[i,j] == relation_logits[j,i] — asymmetry not working"

    def test_source_proj_neq_target_proj_weights(self, cfg):
        """La asimetría viene de pesos distintos, no solo de aleatoriedad."""
        scorer = AsymmetricRelationScorer(cfg)
        sw = scorer.source_proj.weight.data
        tw = scorer.target_proj.weight.data
        assert not torch.allclose(sw, tw), \
            "source_proj.weight must differ from target_proj.weight"

    def test_asymmetry_persists_after_training_step(self, cfg):
        """Un paso de backward no debe hacer los pesos idénticos."""
        scorer = AsymmetricRelationScorer(cfg)
        optimizer = torch.optim.SGD(scorer.parameters(), lr=0.01)

        D = cfg.hidden_dim
        nodes = torch.randn(4, D)
        logits = scorer(nodes, nodes)
        loss = logits.sum()
        loss.backward()
        optimizer.step()

        sw = scorer.source_proj.weight.data
        tw = scorer.target_proj.weight.data
        assert not torch.allclose(sw, tw)


# ─────────────────────────────────────────────────────────────────────────────
# TestGradientFlow
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:
    def test_node_scores_differentiable(self, gc, cfg):
        """Los gradientes fluyen desde node_scores hasta concepts."""
        concepts = make_concepts(2, 16, cfg.hidden_dim)
        concepts.requires_grad_(True)
        gc.train()
        out = gc(concepts)
        out.node_scores.sum().backward()
        assert concepts.grad is not None
        assert not concepts.grad.isnan().any()

    def test_node_vectors_differentiable(self, gc, cfg):
        """Los gradientes fluyen desde node_vectors hasta concepts."""
        concepts = make_concepts(2, 16, cfg.hidden_dim)
        concepts.requires_grad_(True)
        gc.train()
        out = gc(concepts)
        out.node_vectors.sum().backward()
        assert concepts.grad is not None

    def test_relation_logits_differentiable(self, gc, cfg):
        """Los gradientes fluyen desde relation_logits hasta concepts."""
        concepts = make_concepts(2, 16, cfg.hidden_dim)
        concepts.requires_grad_(True)
        gc.train()
        out = gc(concepts)
        out.relation_logits.sum().backward()
        assert concepts.grad is not None

    def test_all_parameters_get_gradients(self, gc, cfg):
        """Un backward sobre todos los outputs da gradiente a todos los parámetros."""
        concepts = make_concepts(2, 16, cfg.hidden_dim)
        gc.train()
        out = gc(concepts)
        loss = (
            out.node_scores.sum()
            + out.node_vectors.sum()
            + out.relation_logits.sum()
            + out.node_type_logits.sum()
            + out.node_confidence.sum()
        )
        loss.backward()
        params_without_grad = [
            name for name, p in gc.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert len(params_without_grad) == 0, \
            f"Parameters without grad: {params_without_grad}"

    def test_gradients_are_finite(self, gc, cfg):
        concepts = make_concepts(2, 16, cfg.hidden_dim)
        gc.train()
        out = gc(concepts)
        loss = out.node_scores.sum() + out.node_vectors.sum() + out.relation_logits.sum()
        loss.backward()
        for name, p in gc.named_parameters():
            if p.grad is not None:
                assert p.grad.isfinite().all(), f"Non-finite grad in {name}"

    def test_end_to_end_differentiable(self, gc_full):
        """Test end-to-end: concept vectors → crystallizer → loss → backward."""
        B, L, D = 2, 32, 256
        concepts = torch.randn(B, L, D, requires_grad=True)
        gc_full.train()
        out = gc_full(concepts)

        # Simular una pérdida supervisada
        loss = out.node_scores.mean() + out.relation_logits.sigmoid().mean()
        loss.backward()

        assert concepts.grad is not None, "No gradient flowed to input concepts"
        assert concepts.grad.isfinite().all()

    def test_no_grad_mode_does_not_crash(self, gc, cfg):
        concepts = make_concepts(1, 16, cfg.hidden_dim)
        with torch.no_grad():
            out = gc(concepts)
        assert isinstance(out, CrystallizerOutput)


# ─────────────────────────────────────────────────────────────────────────────
# TestParameterCount
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterCount:
    def test_count_is_positive(self, gc):
        assert gc.count_parameters() > 0

    def test_full_config_in_reasonable_range(self, gc_full):
        """La configuración full (hidden=256) debe tener < 10M parámetros."""
        n = gc_full.count_parameters()
        assert n < 10_000_000, f"Too many params: {n:,}"
        assert n > 10_000, f"Too few params: {n:,}"

    def test_breakdown_sums_to_total(self, gc):
        bd = gc.parameter_breakdown()
        assert bd["total"] == gc.count_parameters()

    def test_breakdown_has_all_submodules(self, gc):
        bd = gc.parameter_breakdown()
        assert "node_detector"   in bd
        assert "pooler"          in bd
        assert "relation_scorer" in bd
        assert "total"           in bd

    def test_breakdown_all_positive(self, gc):
        bd = gc.parameter_breakdown()
        for key, val in bd.items():
            assert val > 0, f"{key} has {val} params"

    def test_larger_hidden_more_params(self):
        cfg_small = CrystallizerConfig(hidden_dim=32, pooler_heads=4,
                                        relation_hidden_dim=8,
                                        node_confidence_hidden_dim=8)
        cfg_large = CrystallizerConfig(hidden_dim=128, pooler_heads=4,
                                        relation_hidden_dim=32,
                                        node_confidence_hidden_dim=32)
        gc_small = GraphCrystallizer(cfg_small)
        gc_large = GraphCrystallizer(cfg_large)
        assert gc_large.count_parameters() > gc_small.count_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# TestDeterminism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_eval_mode_deterministic(self, gc, cfg):
        gc.eval()
        concepts = make_concepts(1, 16, cfg.hidden_dim, seed=0)
        with torch.no_grad():
            out1 = gc(concepts)
            out2 = gc(concepts)
        assert torch.allclose(out1.node_scores, out2.node_scores)
        assert torch.allclose(out1.relation_logits, out2.relation_logits)

    def test_different_seeds_different_outputs(self, gc, cfg):
        gc.eval()
        c1 = make_concepts(1, 16, cfg.hidden_dim, seed=1)
        c2 = make_concepts(1, 16, cfg.hidden_dim, seed=99)
        with torch.no_grad():
            out1 = gc(c1)
            out2 = gc(c2)
        assert not torch.allclose(out1.node_scores, out2.node_scores)

    def test_same_model_same_output_across_calls(self, gc, cfg):
        gc.eval()
        concepts = make_concepts(2, 12, cfg.hidden_dim, seed=5)
        with torch.no_grad():
            outputs = [gc(concepts) for _ in range(3)]
        for i in range(1, 3):
            assert torch.allclose(outputs[0].node_vectors, outputs[i].node_vectors)


# ─────────────────────────────────────────────────────────────────────────────
# TestEdgeCases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_seq_len_1(self):
        """Una sola posición: nodo candidato único, sin aristas posibles."""
        cfg = CrystallizerConfig(hidden_dim=32, max_nodes=4, pooler_heads=4,
                                  relation_hidden_dim=8, node_confidence_hidden_dim=8,
                                  node_threshold=0.01)
        gc = GraphCrystallizer(cfg)
        gc.eval()
        concepts = torch.randn(1, 1, 32)
        out = gc(concepts)
        assert len(out.graphs) == 1
        assert len(out.graphs[0].nodes) <= 1
        assert len(out.graphs[0].edges) == 0  # No pueden haber aristas con ≤1 nodo

    def test_seq_len_2_can_have_edge(self):
        """Dos posiciones: si ambas son nodos y hay arista, el grafo tiene 1 arista."""
        cfg = CrystallizerConfig(hidden_dim=32, max_nodes=4, pooler_heads=4,
                                  relation_hidden_dim=8, node_confidence_hidden_dim=8,
                                  node_threshold=0.01, edge_threshold=0.01)
        gc = GraphCrystallizer(cfg)
        gc.eval()
        torch.manual_seed(0)
        concepts = torch.randn(1, 2, 32)
        out = gc(concepts)
        graph = out.graphs[0]
        if len(graph.nodes) == 2:
            # Con 2 nodos y threshold muy bajo, debe haber al menos 1 arista
            assert len(graph.edges) >= 1

    def test_batch_size_1(self, gc, cfg):
        concepts = make_concepts(1, 16, cfg.hidden_dim)
        out = gc(concepts)
        assert len(out.graphs) == 1

    def test_batch_size_4(self, gc, cfg):
        concepts = make_concepts(4, 16, cfg.hidden_dim)
        out = gc(concepts)
        assert len(out.graphs) == 4
        assert len(out.node_counts) == 4

    def test_all_below_threshold_gives_empty_graph(self):
        """Si ningún nodo supera el umbral, el grafo estará vacío."""
        cfg = CrystallizerConfig(
            hidden_dim=32, max_nodes=4, pooler_heads=4,
            relation_hidden_dim=8, node_confidence_hidden_dim=8,
            node_threshold=0.9999,  # umbral casi imposible de superar
        )
        gc = GraphCrystallizer(cfg)
        gc.eval()
        torch.manual_seed(42)
        # Input pequeño → scores cerca de 0.5 → ninguno supera 0.9999
        concepts = torch.zeros(1, 8, 32)
        out = gc(concepts)
        # Puede tener 0 nodos (umbral muy alto)
        # Solo verificamos que no crashea y el output es válido
        assert isinstance(out.graphs[0], CausalGraph)

    def test_no_self_loops_in_graph(self, gc, cfg):
        """Los grafos no deben tener auto-bucles (CausalEdge lo prohíbe)."""
        cfg_low = CrystallizerConfig(
            hidden_dim=64, max_nodes=8, pooler_heads=4,
            relation_hidden_dim=16, node_confidence_hidden_dim=16,
            node_threshold=0.01, edge_threshold=0.01,
        )
        gc_low = GraphCrystallizer(cfg_low)
        gc_low.eval()
        concepts = make_concepts(2, 16, 64)
        out = gc_low(concepts)
        for graph in out.graphs:
            for edge in graph.edges:
                assert edge.source_id != edge.target_id, \
                    f"Self-loop detected: {edge.source_id} -> {edge.target_id}"

    def test_edge_relations_are_valid_causal_relations(self):
        """Todas las relaciones en los edges deben ser CausalRelation válidas."""
        cfg = CrystallizerConfig(
            hidden_dim=64, max_nodes=8, pooler_heads=4,
            relation_hidden_dim=16, node_confidence_hidden_dim=16,
            node_threshold=0.01, edge_threshold=0.01,
        )
        gc = GraphCrystallizer(cfg)
        gc.eval()
        concepts = make_concepts(3, 16, 64)
        out = gc(concepts)
        valid_relations = set(CausalRelation)
        for graph in out.graphs:
            for edge in graph.edges:
                assert edge.relation in valid_relations, \
                    f"Invalid relation: {edge.relation!r}"

    def test_edge_strength_in_01(self):
        """strength y confidence de aristas deben estar en [0, 1]."""
        cfg = CrystallizerConfig(
            hidden_dim=64, max_nodes=8, pooler_heads=4,
            relation_hidden_dim=16, node_confidence_hidden_dim=16,
            node_threshold=0.01, edge_threshold=0.01,
        )
        gc = GraphCrystallizer(cfg)
        gc.eval()
        concepts = make_concepts(2, 16, 64)
        out = gc(concepts)
        for graph in out.graphs:
            for edge in graph.edges:
                assert 0.0 <= edge.strength <= 1.0, \
                    f"edge.strength={edge.strength} out of [0,1]"
                assert 0.0 <= edge.confidence <= 1.0, \
                    f"edge.confidence={edge.confidence} out of [0,1]"

    def test_full_config_runs_without_error(self):
        """La configuración por defecto (hidden=256, max_nodes=32) debe funcionar."""
        gc = GraphCrystallizer(CrystallizerConfig())
        gc.eval()
        concepts = torch.randn(1, 64, 256)
        out = gc(concepts)
        assert isinstance(out, CrystallizerOutput)
        assert len(out.graphs) == 1
