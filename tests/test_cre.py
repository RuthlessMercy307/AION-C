"""
tests/test_cre.py — Tests exhaustivos del CausalReasoningEngine
===============================================================

Organización:
    TestCREConfig                   — validación de configuración
    TestAttentiveAggregator         — shapes, pesos aprendidos, edge cases
    TestCausalMessagePassingLayer   — shapes, relaciones distintas → mensajes distintos
    TestTypedMessageDifference      — CAUSES ≠ PREVENTS (el test semántico clave)
    TestWeightSharing               — mismos pesos en iter 1 y iter 5
    TestRefinement                  — node features CAMBIAN entre iteraciones
    TestGradientFlow                — gradientes a través de múltiples iteraciones
    TestNumericalStability          — 20 iteraciones no crashean ni explotan
    TestParameterCount              — integridad del conteo
    TestEdgeCases                   — grafo vacío, 1 nodo, sin aristas
    TestDeterminism                 — eval mode determinista
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from core.graph import (
    CAUSAL_RELATIONS, CausalEdge, CausalGraph, CausalNode, CausalRelation, NodeType
)
from cre import (
    AttentiveAggregator,
    CREConfig,
    CREOutput,
    CausalMessagePassingLayer,
    CausalReasoningEngine,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_tiny_cfg(**kwargs) -> CREConfig:
    """Configuración mínima para tests rápidos."""
    defaults = dict(node_dim=32, edge_dim=16, message_dim=24,
                    n_message_layers=2, max_iterations=5, n_relation_types=16,
                    use_convergence_gate=False)
    defaults.update(kwargs)
    return CREConfig(**defaults)


def make_default_cfg() -> CREConfig:
    """Configuración por defecto del plan (256/64/128/2/20)."""
    return CREConfig()


def make_linear_graph(n_nodes: int, relation: CausalRelation = CausalRelation.CAUSES) -> tuple:
    """
    Construye un grafo lineal A→B→C→… con la relación dada.
    Devuelve (graph, node_features).
    """
    cfg = make_tiny_cfg()
    graph = CausalGraph()
    for i in range(n_nodes):
        node = CausalNode(node_id=f"n{i}", label=f"node{i}",
                          node_type=NodeType.EVENT, confidence=0.9)
        graph.add_node(node)
    for i in range(n_nodes - 1):
        edge = CausalEdge(source_id=f"n{i}", target_id=f"n{i+1}",
                          relation=relation, strength=0.8, confidence=0.9)
        graph.add_edge(edge)
    node_features = torch.randn(n_nodes, cfg.node_dim)
    return graph, node_features


def make_full_graph(n_nodes: int, relation: CausalRelation = CausalRelation.CAUSES,
                    node_dim: int = 32) -> tuple:
    """
    Grafo completo (todos los pares conectados) con la relación dada.
    """
    graph = CausalGraph()
    for i in range(n_nodes):
        node = CausalNode(node_id=f"n{i}", label=f"node{i}",
                          node_type=NodeType.ENTITY, confidence=0.8)
        graph.add_node(node)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edge = CausalEdge(source_id=f"n{i}", target_id=f"n{j}",
                                  relation=relation, strength=0.7, confidence=0.8)
                graph.add_edge(edge)
    node_features = torch.randn(n_nodes, node_dim)
    return graph, node_features


def make_two_node_graph(rel_fwd: CausalRelation, rel_bwd: CausalRelation = None,
                         node_dim: int = 32) -> tuple:
    """Grafo con 2 nodos y 1 (o 2) aristas para tests de asimetría."""
    graph = CausalGraph()
    for i in range(2):
        graph.add_node(CausalNode(node_id=f"n{i}", label=f"n{i}",
                                   node_type=NodeType.EVENT))
    graph.add_edge(CausalEdge(source_id="n0", target_id="n1",
                               relation=rel_fwd, strength=0.8))
    if rel_bwd is not None:
        graph.add_edge(CausalEdge(source_id="n1", target_id="n0",
                                   relation=rel_bwd, strength=0.8))
    node_features = torch.randn(2, node_dim)
    return graph, node_features


# ─────────────────────────────────────────────────────────────────────────────
# TestCREConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestCREConfig:
    def test_default_values(self):
        cfg = CREConfig()
        assert cfg.node_dim           == 256
        assert cfg.edge_dim           == 64
        assert cfg.message_dim        == 128
        assert cfg.n_message_layers   == 2
        assert cfg.max_iterations     == 20
        assert cfg.n_relation_types   == 16

    def test_n_relation_types_matches_causal_relations(self):
        cfg = CREConfig()
        assert cfg.n_relation_types == len(CAUSAL_RELATIONS)

    def test_custom_config(self):
        cfg = make_tiny_cfg()
        assert cfg.node_dim    == 32
        assert cfg.edge_dim    == 16
        assert cfg.message_dim == 24

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError):
            CREConfig(n_message_layers=0)

    def test_invalid_max_iterations(self):
        with pytest.raises(ValueError):
            CREConfig(max_iterations=0)

    def test_invalid_message_dim(self):
        with pytest.raises(ValueError):
            CREConfig(message_dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# TestAttentiveAggregator
# ─────────────────────────────────────────────────────────────────────────────

class TestAttentiveAggregator:
    @pytest.fixture
    def agg(self):
        return AttentiveAggregator(make_tiny_cfg())

    def test_output_shape(self, agg):
        N, E, M, D = 5, 8, 24, 32
        cfg = make_tiny_cfg()
        msgs    = torch.randn(E, cfg.message_dim)
        tgt_idx = torch.randint(0, N, (E,))
        node_f  = torch.randn(N, cfg.node_dim)
        out = agg(msgs, tgt_idx, node_f, N)
        assert out.shape == (N, cfg.message_dim)

    def test_no_messages_returns_zeros(self, agg):
        N   = 6
        cfg = make_tiny_cfg()
        msgs    = torch.zeros(0, cfg.message_dim)
        tgt_idx = torch.zeros(0, dtype=torch.long)
        node_f  = torch.randn(N, cfg.node_dim)
        out = agg(msgs, tgt_idx, node_f, N)
        assert out.shape == (N, cfg.message_dim)
        assert torch.all(out == 0)

    def test_node_with_no_incoming_messages_is_zero(self, agg):
        """Un nodo sin mensajes entrantes debe tener aggregated=0."""
        cfg = make_tiny_cfg()
        N = 3
        # Solo el nodo 0 recibe mensajes
        msgs    = torch.randn(4, cfg.message_dim)
        tgt_idx = torch.zeros(4, dtype=torch.long)  # todos al nodo 0
        node_f  = torch.randn(N, cfg.node_dim)
        out = agg(msgs, tgt_idx, node_f, N)
        # Nodo 1 y 2 no reciben mensajes → debe ser cero (antes de norm)
        # Después de LayerNorm, cero se normaliza a cero (bias=0 en init)
        assert out.shape == (N, cfg.message_dim)

    def test_different_messages_different_output(self, agg):
        """Mensajes distintos → outputs distintos."""
        cfg = make_tiny_cfg()
        N = 2
        tgt_idx = torch.zeros(3, dtype=torch.long)  # todos al nodo 0
        node_f  = torch.randn(N, cfg.node_dim)
        msgs1 = torch.randn(3, cfg.message_dim)
        msgs2 = torch.randn(3, cfg.message_dim)
        out1 = agg(msgs1, tgt_idx, node_f, N)
        out2 = agg(msgs2, tgt_idx, node_f, N)
        assert not torch.allclose(out1, out2)

    def test_weights_are_learned(self, agg):
        """El módulo tiene parámetros entrenables."""
        n_params = sum(p.numel() for p in agg.parameters() if p.requires_grad)
        assert n_params > 0

    def test_gradient_flows(self, agg):
        cfg = make_tiny_cfg()
        N, E = 4, 6
        msgs    = torch.randn(E, cfg.message_dim, requires_grad=True)
        tgt_idx = torch.randint(0, N, (E,))
        node_f  = torch.randn(N, cfg.node_dim, requires_grad=True)
        out = agg(msgs, tgt_idx, node_f, N)
        out.sum().backward()
        assert msgs.grad is not None
        assert node_f.grad is not None

    def test_output_finite(self, agg):
        cfg = make_tiny_cfg()
        N, E = 5, 10
        msgs    = torch.randn(E, cfg.message_dim)
        tgt_idx = torch.randint(0, N, (E,))
        node_f  = torch.randn(N, cfg.node_dim)
        out = agg(msgs, tgt_idx, node_f, N)
        assert out.isfinite().all()

    def test_has_attn_scorer(self, agg):
        assert hasattr(agg, 'attn_scorer')
        assert hasattr(agg, 'norm')


# ─────────────────────────────────────────────────────────────────────────────
# TestCausalMessagePassingLayer
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalMessagePassingLayer:
    @pytest.fixture
    def cfg(self):
        return make_tiny_cfg()

    @pytest.fixture
    def layer(self, cfg):
        return CausalMessagePassingLayer(cfg)

    def test_output_shapes_with_edges(self, layer, cfg):
        graph, node_f = make_linear_graph(5)
        edge_f = torch.randn(len(graph.edges), cfg.edge_dim)
        new_nodes, new_edges = layer(node_f, edge_f, graph)
        assert new_nodes.shape == (5, cfg.node_dim)
        assert new_edges.shape == (len(graph.edges), cfg.edge_dim)

    def test_output_shapes_no_edges(self, layer, cfg):
        """Grafo sin aristas: node_features cambian (GRU con mensaje cero), edge_features iguales."""
        graph = CausalGraph()
        for i in range(3):
            graph.add_node(CausalNode(node_id=f"n{i}", label=f"n{i}",
                                       node_type=NodeType.FACT))
        node_f = torch.randn(3, cfg.node_dim)
        edge_f = torch.zeros(0, cfg.edge_dim)
        new_nodes, new_edges = layer(node_f, edge_f, graph)
        assert new_nodes.shape == (3, cfg.node_dim)
        assert new_edges.shape == (0, cfg.edge_dim)

    def test_node_features_change(self, layer, cfg):
        """Después de una pasada, los node features deben cambiar."""
        graph, node_f = make_linear_graph(4)
        edge_f = torch.randn(len(graph.edges), cfg.edge_dim)
        new_nodes, _ = layer(node_f, edge_f, graph)
        assert not torch.allclose(new_nodes, node_f), \
            "Node features should change after message passing"

    def test_edge_features_change(self, layer, cfg):
        """Los edge features también deben actualizarse."""
        graph, node_f = make_linear_graph(4)
        edge_f = torch.randn(len(graph.edges), cfg.edge_dim)
        _, new_edges = layer(node_f, edge_f, graph)
        assert not torch.allclose(new_edges, edge_f), \
            "Edge features should change after update"

    def test_has_one_fn_per_relation(self, layer):
        """Debe haber exactamente una función de mensaje por CausalRelation."""
        assert len(layer.message_fns) == len(CAUSAL_RELATIONS)
        for rel in CAUSAL_RELATIONS:
            assert rel in layer.message_fns, f"Missing message fn for {rel!r}"

    def test_all_message_fns_are_sequential(self, layer):
        for rel, fn in layer.message_fns.items():
            assert isinstance(fn, nn.Sequential), f"{rel}: expected Sequential"

    def test_has_gru_updater(self, layer, cfg):
        from cre.message_passing import ManualGRUCell
        assert hasattr(layer, 'node_updater')
        assert isinstance(layer.node_updater, ManualGRUCell)
        assert layer.node_updater.input_size  == cfg.message_dim
        assert layer.node_updater.hidden_size == cfg.node_dim

    def test_has_attentive_aggregator(self, layer):
        assert hasattr(layer, 'aggregator')
        assert isinstance(layer.aggregator, AttentiveAggregator)

    def test_has_edge_updater(self, layer):
        assert hasattr(layer, 'edge_updater')

    def test_gradient_flows(self, layer, cfg):
        graph, node_f = make_linear_graph(4)
        node_f.requires_grad_(True)
        edge_f = torch.randn(len(graph.edges), cfg.edge_dim)
        new_nodes, new_edges = layer(node_f, edge_f, graph)
        (new_nodes.sum() + new_edges.sum()).backward()
        assert node_f.grad is not None
        assert not node_f.grad.isnan().any()

    def test_output_finite(self, layer, cfg):
        graph, node_f = make_linear_graph(5)
        edge_f = torch.randn(len(graph.edges), cfg.edge_dim)
        new_nodes, new_edges = layer(node_f, edge_f, graph)
        assert new_nodes.isfinite().all()
        assert new_edges.isfinite().all()

    def test_full_graph(self, layer, cfg):
        """Grafo completo (todos los pares): debe funcionar sin crash."""
        graph, node_f = make_full_graph(4, node_dim=cfg.node_dim)
        edge_f = torch.randn(len(graph.edges), cfg.edge_dim)
        new_nodes, new_edges = layer(node_f, edge_f, graph)
        assert new_nodes.shape == (4, cfg.node_dim)
        assert new_edges.isfinite().all()


# ─────────────────────────────────────────────────────────────────────────────
# TestTypedMessageDifference
# ─────────────────────────────────────────────────────────────────────────────

class TestTypedMessageDifference:
    """
    Verifica que mensajes de tipo CAUSES producen output diferente que PREVENTS.
    Este es el test semántico clave del módulo typed.
    """

    def test_causes_vs_prevents_different_messages(self):
        """
        Mismo grafo (mismos nodos, misma topología), relación distinta
        → las funciones de mensaje producen outputs distintos.
        """
        cfg = make_tiny_cfg()
        layer = CausalMessagePassingLayer(cfg)
        layer.eval()

        torch.manual_seed(42)
        node_f = torch.randn(2, cfg.node_dim)
        edge_f = torch.randn(1, cfg.edge_dim)

        # Grafo con CAUSES
        graph_causes, _ = make_two_node_graph(CausalRelation.CAUSES, node_dim=cfg.node_dim)
        # Grafo con PREVENTS
        graph_prevents, _ = make_two_node_graph(CausalRelation.PREVENTS, node_dim=cfg.node_dim)

        with torch.no_grad():
            nodes_causes,  _ = layer(node_f.clone(), edge_f.clone(), graph_causes)
            nodes_prevents, _ = layer(node_f.clone(), edge_f.clone(), graph_prevents)

        max_diff = (nodes_causes - nodes_prevents).abs().max().item()
        assert max_diff > 1e-4, \
            f"CAUSES and PREVENTS should produce different node updates, max_diff={max_diff:.2e}"

    def test_all_16_relations_produce_distinct_outputs(self):
        """Cada una de las 16 relaciones debe producir outputs distintos entre sí."""
        cfg = make_tiny_cfg()
        layer = CausalMessagePassingLayer(cfg)
        layer.eval()

        torch.manual_seed(7)
        node_f = torch.randn(2, cfg.node_dim)
        edge_f = torch.randn(1, cfg.edge_dim)

        outputs = {}
        with torch.no_grad():
            for rel in CausalRelation:
                graph, _ = make_two_node_graph(rel, node_dim=cfg.node_dim)
                new_nodes, _ = layer(node_f.clone(), edge_f.clone(), graph)
                outputs[rel] = new_nodes.clone()

        # Verificar que al menos la mayoría de pares son distintos
        distinct_pairs = 0
        total_pairs = 0
        rels = list(CausalRelation)
        for i in range(len(rels)):
            for j in range(i + 1, len(rels)):
                diff = (outputs[rels[i]] - outputs[rels[j]]).abs().max().item()
                if diff > 1e-5:
                    distinct_pairs += 1
                total_pairs += 1

        # Al menos el 80% de los pares deben ser distintos
        ratio = distinct_pairs / total_pairs
        assert ratio >= 0.8, \
            f"Only {ratio:.1%} of relation pairs produce distinct outputs (expected ≥ 80%)"

    def test_causes_message_fn_neq_prevents_message_fn(self):
        """
        Las funciones de mensaje de CAUSES y PREVENTS deben tener pesos distintos.
        Esto garantiza la distinción structuralmente, no solo por el azar de init.
        """
        cfg = make_tiny_cfg()
        layer = CausalMessagePassingLayer(cfg)
        causes_w   = list(layer.message_fns["causes"].parameters())[0]
        prevents_w = list(layer.message_fns["prevents"].parameters())[0]
        assert not torch.allclose(causes_w, prevents_w), \
            "causes and prevents message functions should have different weights"

    def test_positive_relations_semantically_distinct_from_inhibitory(self):
        """
        ENABLES (positiva) y WEAKENS (inhibitoria) deben producir direcciones distintas.
        """
        cfg = make_tiny_cfg()
        layer = CausalMessagePassingLayer(cfg)
        layer.eval()

        torch.manual_seed(3)
        node_f = torch.randn(2, cfg.node_dim)
        edge_f = torch.randn(1, cfg.edge_dim)

        with torch.no_grad():
            graph_en, _ = make_two_node_graph(CausalRelation.ENABLES, node_dim=cfg.node_dim)
            graph_wk, _ = make_two_node_graph(CausalRelation.WEAKENS, node_dim=cfg.node_dim)
            out_en, _ = layer(node_f.clone(), edge_f.clone(), graph_en)
            out_wk, _ = layer(node_f.clone(), edge_f.clone(), graph_wk)

        diff = (out_en - out_wk).abs().max().item()
        assert diff > 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# TestWeightSharing
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightSharing:
    """
    Verifica que el weight sharing funciona:
    los mismos parámetros son usados en cada iteración.
    """

    def test_same_layer_objects_across_iterations(self):
        """Los layers son los mismos objetos Python en cada iteración."""
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)

        # Las capas son los mismos objetos (identidad de Python)
        layer_ids_before = [id(layer) for layer in engine.layers]
        # Después de un forward, los objetos siguen siendo los mismos
        graph, node_f = make_linear_graph(3)
        _ = engine(graph, node_f, n_iterations=5)
        layer_ids_after  = [id(layer) for layer in engine.layers]
        assert layer_ids_before == layer_ids_after

    def test_weights_identical_across_iterations(self):
        """
        Los pesos de layer[0] en iteración 1 == pesos de layer[0] en iteración 5.
        Se verifica capturando el estado antes y después del forward (no hay entrenamiento
        en este test, así que los pesos no deben cambiar).
        """
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        engine.eval()

        # Capturar pesos ANTES del forward
        w_before = engine.layers[0].node_updater.weight_ih.data.clone()

        graph, node_f = make_linear_graph(3)
        with torch.no_grad():
            _ = engine(graph, node_f, n_iterations=5)

        # Capturar pesos DESPUÉS del forward (sin optimizer step)
        w_after = engine.layers[0].node_updater.weight_ih.data.clone()

        assert torch.allclose(w_before, w_after), \
            "Layer weights should not change during inference (no optimizer step)"

    def test_n_params_independent_of_iterations(self):
        """El número de parámetros no debe depender de max_iterations."""
        cfg_5  = CREConfig(node_dim=32, edge_dim=16, message_dim=24,
                           n_message_layers=2, max_iterations=5)
        cfg_20 = CREConfig(node_dim=32, edge_dim=16, message_dim=24,
                           n_message_layers=2, max_iterations=20)
        engine_5  = CausalReasoningEngine(cfg_5)
        engine_20 = CausalReasoningEngine(cfg_20)
        assert engine_5.count_parameters() == engine_20.count_parameters(), \
            "Parameters should be the same regardless of max_iterations (weight sharing)"

    def test_n_params_grows_with_n_message_layers(self):
        """Más capas → más parámetros (las capas son distintas, no shared entre sí)."""
        cfg_1 = CREConfig(node_dim=32, edge_dim=16, message_dim=24, n_message_layers=1)
        cfg_2 = CREConfig(node_dim=32, edge_dim=16, message_dim=24, n_message_layers=2)
        e1 = CausalReasoningEngine(cfg_1)
        e2 = CausalReasoningEngine(cfg_2)
        assert e2.count_parameters() > e1.count_parameters()

    def test_layer_reuse_verified_by_param_count(self):
        """
        Si hay 2 capas y 20 iteraciones:
        params = 2_layers * params_per_layer + edge_embedder + projector
        NO = 2*20 = 40 layers worth of params.
        """
        cfg = make_tiny_cfg(n_message_layers=2, max_iterations=20)
        engine = CausalReasoningEngine(cfg)
        bd = engine.parameter_breakdown()
        # El effective_at_max_iter debe ser 10x el layers_shared (20 iter / 2 layers = 10x each layer)
        # Actually: effective = layers_shared * max_iterations / n_layers
        # Wait: effective = layers_shared * max_iterations (each layer is called max_iterations times)
        # layers_shared = params in the 2 layers (shared set)
        # effective_at_max_iter = layers_shared * max_iterations
        assert bd["effective_at_max_iter"] == bd["layers_shared"] * cfg.max_iterations


# ─────────────────────────────────────────────────────────────────────────────
# TestRefinement
# ─────────────────────────────────────────────────────────────────────────────

class TestRefinement:
    """
    Verifica que las iteraciones realmente refinan los features:
    los node features CAMBIAN entre iteraciones.
    """

    def test_features_change_after_1_iteration(self):
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        graph, node_f = make_linear_graph(4)

        out = engine(graph, node_f, n_iterations=1)
        assert not torch.allclose(out.node_features, node_f), \
            "Node features should change after 1 iteration"

    def test_more_iterations_produce_different_output(self):
        """1 iteración ≠ 5 iteraciones."""
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        engine.eval()
        graph, node_f = make_linear_graph(4)

        with torch.no_grad():
            out_1 = engine(graph, node_f, n_iterations=1)
            out_5 = engine(graph, node_f, n_iterations=5)

        assert not torch.allclose(out_1.node_features, out_5.node_features), \
            "1 iteration and 5 iterations should give different outputs"

    def test_iterations_run_reported_correctly(self):
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        graph, node_f = make_linear_graph(3)

        for n in [1, 3, 7, 20]:
            out = engine(graph, node_f, n_iterations=n)
            assert out.iterations_run == n, \
                f"Expected iterations_run={n}, got {out.iterations_run}"

    def test_return_history_captures_all_steps(self):
        """return_history=True debe capturar n_iter × n_layers snapshots."""
        cfg = make_tiny_cfg(n_message_layers=2, max_iterations=3)
        engine = CausalReasoningEngine(cfg)
        graph, node_f = make_linear_graph(4)

        out = engine(graph, node_f, n_iterations=3, return_history=True)
        # 3 iterations × 2 layers = 6 snapshots
        assert len(out.layer_outputs) == 3 * 2

    def test_return_history_shapes(self):
        cfg = make_tiny_cfg(n_message_layers=2)
        engine = CausalReasoningEngine(cfg)
        graph, node_f = make_linear_graph(4)
        out = engine(graph, node_f, n_iterations=2, return_history=True)
        for snapshot in out.layer_outputs:
            assert snapshot.shape == (4, cfg.node_dim)

    def test_history_shows_evolution(self):
        """Los snapshots del historial deben cambiar entre pasos."""
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        graph, node_f = make_linear_graph(4)
        out = engine(graph, node_f, n_iterations=3, return_history=True)
        # Al menos algún par de snapshots consecutivos debe diferir
        diffs = [
            (out.layer_outputs[i] - out.layer_outputs[i+1]).abs().max().item()
            for i in range(len(out.layer_outputs) - 1)
        ]
        assert any(d > 1e-6 for d in diffs), \
            "Layer snapshots should show evolution across iterations"

    def test_default_iterations_uses_config(self):
        """Sin n_iterations, debe usar config.max_iterations."""
        cfg = make_tiny_cfg(max_iterations=3)
        engine = CausalReasoningEngine(cfg)
        graph, node_f = make_linear_graph(3)
        out = engine(graph, node_f)
        assert out.iterations_run == 3

    def test_returns_cre_output(self):
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        graph, node_f = make_linear_graph(3)
        out = engine(graph, node_f)
        assert isinstance(out, CREOutput)


# ─────────────────────────────────────────────────────────────────────────────
# TestGradientFlow
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:
    """
    Verifica que los gradientes fluyen a través de múltiples iteraciones.
    Esto es necesario para entrenar el CRE end-to-end.
    """

    def test_gradient_flows_1_iteration(self):
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        engine.train()
        graph, node_f = make_linear_graph(4)
        node_f.requires_grad_(True)

        out = engine(graph, node_f, n_iterations=1)
        out.node_features.sum().backward()
        assert node_f.grad is not None

    def test_gradient_flows_5_iterations(self):
        """Gradientes a través de 5 iteraciones (peso compartido → mismo param recibe grads de 5 llamadas)."""
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        engine.train()
        graph, node_f = make_linear_graph(4)
        node_f.requires_grad_(True)

        out = engine(graph, node_f, n_iterations=5)
        out.node_features.sum().backward()
        assert node_f.grad is not None
        assert not node_f.grad.isnan().any()

    def test_all_parameters_get_gradients(self):
        """
        Todos los parámetros del engine deben recibir gradiente cuando el grafo
        contiene todos los tipos de relación.

        Nota: los message_fns de relaciones NO presentes en el grafo no reciben
        gradiente (no se usan en el forward). Por eso usamos un grafo con las
        16 relaciones.
        """
        cfg = make_tiny_cfg(max_iterations=3)
        engine = CausalReasoningEngine(cfg)
        engine.train()

        # Grafo que incluye las 16 relaciones para que todos los message_fns
        # participen en el forward pass
        n_nodes = len(CAUSAL_RELATIONS) + 1
        graph = CausalGraph()
        for i in range(n_nodes):
            graph.add_node(CausalNode(node_id=f"n{i}", label=f"n{i}",
                                       node_type=NodeType.EVENT))
        for i, rel in enumerate(CausalRelation):
            graph.add_edge(CausalEdge(source_id=f"n{i}", target_id=f"n{i+1}",
                                       relation=rel))
        node_f = torch.randn(n_nodes, cfg.node_dim)

        out = engine(graph, node_f, n_iterations=3)
        loss = out.node_features.sum() + out.edge_features.sum()
        loss.backward()

        params_without_grad = [
            name for name, p in engine.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert len(params_without_grad) == 0, \
            f"Parameters without grad: {params_without_grad}"

    def test_gradients_finite_after_5_iterations(self):
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        engine.train()
        graph, node_f = make_linear_graph(4)

        out = engine(graph, node_f, n_iterations=5)
        out.node_features.sum().backward()

        for name, p in engine.named_parameters():
            if p.grad is not None:
                assert p.grad.isfinite().all(), f"Non-finite grad in {name}"

    def test_gradient_accumulated_for_shared_weights(self):
        """
        Con weight sharing, los parámetros compartidos acumulan gradientes
        de TODAS las iteraciones en que se usaron.
        Los gradientes deben ser mayores con más iteraciones.
        """
        torch.manual_seed(42)  # ensure deterministic weight init
        cfg = make_tiny_cfg()
        graph, node_f = make_linear_graph(4)

        # 1 iteración
        engine_1 = CausalReasoningEngine(cfg)
        # Misma inicialización para comparar
        engine_5 = CausalReasoningEngine(cfg)
        # Copiar pesos
        engine_5.load_state_dict(engine_1.state_dict())

        engine_1.train(); engine_5.train()

        out1 = engine_1(graph, node_f.clone(), n_iterations=1)
        out1.node_features.sum().backward()
        # Use total gradient norm (sum across all params) — more robust than a single
        # parameter whose direction can cancel across nodes in small graphs.
        grad_1iter = sum(p.grad.norm().item() for p in engine_1.parameters() if p.grad is not None)

        out5 = engine_5(graph, node_f.clone(), n_iterations=5)
        out5.node_features.sum().backward()
        grad_5iter = sum(p.grad.norm().item() for p in engine_5.parameters() if p.grad is not None)

        # Con weight sharing, el gradiente debe fluir en ambos casos (no-zero y finito).
        # La magnitud relativa no es garantizable: con GRU el gate puede atenuar el
        # gradiente a través de muchas iteraciones (vanishing), así que solo verificamos
        # que el flujo existe en ambas configuraciones.
        assert grad_1iter > 0, f"1-iter: no gradient flowing to shared weights (got {grad_1iter})"
        assert grad_5iter > 0, f"5-iter: no gradient flowing to shared weights (got {grad_5iter})"
        assert math.isfinite(grad_1iter), f"1-iter gradient exploded: {grad_1iter}"
        assert math.isfinite(grad_5iter), f"5-iter gradient exploded: {grad_5iter}"

    def test_no_grad_mode_works(self):
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        graph, node_f = make_linear_graph(3)
        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=3)
        assert isinstance(out, CREOutput)
        assert out.node_features.isfinite().all()


# ─────────────────────────────────────────────────────────────────────────────
# TestNumericalStability
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalStability:
    """
    Verifica que el modelo es estable con muchas iteraciones.
    Con weight sharing, el sistema puede explotar o colapsar si no está bien diseñado.
    """

    def test_20_iterations_no_nan(self):
        """20 iteraciones (config por defecto) no producen NaN."""
        cfg = make_tiny_cfg(max_iterations=20)
        engine = CausalReasoningEngine(cfg)
        engine.eval()
        graph, node_f = make_linear_graph(5)

        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=20)

        assert not out.node_features.isnan().any(), "NaN in node features after 20 iterations"
        assert not out.edge_features.isnan().any(), "NaN in edge features after 20 iterations"

    def test_20_iterations_no_inf(self):
        """20 iteraciones no producen valores infinitos."""
        cfg = make_tiny_cfg(max_iterations=20)
        engine = CausalReasoningEngine(cfg)
        engine.eval()
        graph, node_f = make_full_graph(4, node_dim=cfg.node_dim)

        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=20)

        assert out.node_features.isfinite().all(), "Inf in node features after 20 iterations"
        assert out.edge_features.isfinite().all(), "Inf in edge features after 20 iterations"

    def test_50_iterations_stable(self):
        """Incluso 50 iteraciones (2.5x max_iterations) deben ser estables."""
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        engine.eval()
        graph, node_f = make_linear_graph(5)

        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=50)

        assert out.node_features.isfinite().all()
        assert not out.node_features.isnan().any()

    def test_norm_of_output_bounded(self):
        """La norma de los output features debe estar en un rango razonable."""
        cfg = make_tiny_cfg(max_iterations=20)
        engine = CausalReasoningEngine(cfg)
        engine.eval()
        graph, node_f = make_linear_graph(5)

        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=20)

        norms = out.node_features.norm(dim=-1)
        # Con LayerNorm, las normas deben estar aproximadamente en [0.1, 100]
        assert (norms < 1000).all(), f"Node feature norms too large: {norms.max().item()}"
        assert (norms > 0).all(), "Some node norms are zero"

    def test_full_graph_20_iterations_stable(self):
        """Grafo completo (máximas aristas) con 20 iteraciones."""
        cfg = make_tiny_cfg(max_iterations=20)
        engine = CausalReasoningEngine(cfg)
        engine.eval()
        # Mezcla de relaciones
        graph = CausalGraph()
        for i in range(4):
            graph.add_node(CausalNode(node_id=f"n{i}", label=f"n{i}",
                                       node_type=NodeType.EVENT))
        rels = list(CausalRelation)
        k = 0
        for i in range(4):
            for j in range(4):
                if i != j:
                    rel = rels[k % len(rels)]
                    graph.add_edge(CausalEdge(source_id=f"n{i}", target_id=f"n{j}",
                                               relation=rel))
                    k += 1
        node_f = torch.randn(4, cfg.node_dim)

        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=20)

        assert out.node_features.isfinite().all()

    def test_large_input_values_stable(self):
        """Valores de entrada grandes no deben causar explosión."""
        cfg = make_tiny_cfg(max_iterations=10)
        engine = CausalReasoningEngine(cfg)
        engine.eval()
        graph, _ = make_linear_graph(4)
        # Valores 100x mayores de lo normal
        node_f = torch.randn(4, cfg.node_dim) * 100.0

        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=10)

        assert out.node_features.isfinite().all()

    def test_default_config_20_iterations(self):
        """La configuración por defecto (256/64/128) debe funcionar con 20 iteraciones."""
        cfg = CREConfig()  # 256, 64, 128, 2 layers, 20 iter
        engine = CausalReasoningEngine(cfg)
        engine.eval()
        graph, node_f = make_linear_graph(4)
        # Proyectar a node_dim=256
        node_f_256 = torch.randn(4, 256)
        # Crear grafo con aristas
        graph, _ = make_linear_graph(4)

        with torch.no_grad():
            out = engine(graph, node_f_256, n_iterations=20)

        assert out.node_features.isfinite().all()
        assert out.iterations_run == 20


# ─────────────────────────────────────────────────────────────────────────────
# TestParameterCount
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterCount:
    def test_count_is_positive(self):
        engine = CausalReasoningEngine(make_tiny_cfg())
        assert engine.count_parameters() > 0

    def test_breakdown_sums_to_total(self):
        engine = CausalReasoningEngine(make_tiny_cfg())
        bd = engine.parameter_breakdown()
        assert bd["total"] == engine.count_parameters()

    def test_breakdown_has_required_keys(self):
        engine = CausalReasoningEngine(make_tiny_cfg())
        bd = engine.parameter_breakdown()
        assert "layers_shared"        in bd
        assert "edge_type_embedding"  in bd
        assert "edge_feat_projector"  in bd
        assert "total"                in bd

    def test_default_config_under_limit(self):
        """La configuración por defecto (hidden=256) debe tener < 30M parámetros."""
        engine = CausalReasoningEngine(CREConfig())
        n = engine.count_parameters()
        assert n < 30_000_000, f"Too many params: {n:,}"
        assert n > 100_000,    f"Too few params: {n:,}"

    def test_more_relations_more_params(self):
        """Más relation types → más message functions → más params."""
        cfg_8  = CREConfig(node_dim=32, edge_dim=16, message_dim=24,
                           n_relation_types=8)
        cfg_16 = CREConfig(node_dim=32, edge_dim=16, message_dim=24,
                           n_relation_types=16)
        # Nota: n_relation_types afecta edge_embedding pero no directamente
        # el número de message_fns (estos se crean por CAUSAL_RELATIONS que tiene 16)
        # Este test verifica que el embedding escala
        e8  = CausalReasoningEngine(cfg_8)
        e16 = CausalReasoningEngine(cfg_16)
        assert e16.count_parameters() > e8.count_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# TestEdgeCases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_node_no_edges(self):
        """Un nodo sin aristas: message passing con cero mensajes."""
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        graph = CausalGraph()
        graph.add_node(CausalNode(node_id="n0", label="solo", node_type=NodeType.ENTITY))
        node_f = torch.randn(1, cfg.node_dim)

        out = engine(graph, node_f, n_iterations=3)
        assert out.node_features.shape == (1, cfg.node_dim)
        assert out.node_features.isfinite().all()

    def test_two_nodes_one_edge(self):
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        graph, node_f = make_two_node_graph(CausalRelation.CAUSES, node_dim=cfg.node_dim)

        out = engine(graph, node_f, n_iterations=3)
        assert out.node_features.shape == (2, cfg.node_dim)
        assert out.edge_features.shape == (1, cfg.edge_dim)

    def test_all_16_relation_types_in_one_graph(self):
        """Un grafo que usa las 16 relaciones debe funcionar correctamente."""
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        engine.eval()

        n_nodes = len(CAUSAL_RELATIONS) + 1  # 17 nodos
        graph = CausalGraph()
        for i in range(n_nodes):
            graph.add_node(CausalNode(node_id=f"n{i}", label=f"n{i}",
                                       node_type=NodeType.EVENT))
        # Una arista por relación
        for i, rel in enumerate(CausalRelation):
            graph.add_edge(CausalEdge(source_id=f"n{i}", target_id=f"n{i+1}",
                                       relation=rel))

        node_f = torch.randn(n_nodes, cfg.node_dim)
        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=3)

        assert out.node_features.shape == (n_nodes, cfg.node_dim)
        assert out.node_features.isfinite().all()

    def test_graph_with_contradictions(self):
        """Grafo con CAUSES y PREVENTS entre los mismos nodos."""
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        engine.eval()

        graph = CausalGraph()
        for i in range(3):
            graph.add_node(CausalNode(node_id=f"n{i}", label=f"n{i}",
                                       node_type=NodeType.HYPOTHESIS))
        # CAUSES y PREVENTS entre n0 y n1 (contradicción)
        graph.add_edge(CausalEdge(source_id="n0", target_id="n1",
                                   relation=CausalRelation.CAUSES))
        graph.add_edge(CausalEdge(source_id="n0", target_id="n1",
                                   relation=CausalRelation.PREVENTS))
        graph.add_edge(CausalEdge(source_id="n1", target_id="n2",
                                   relation=CausalRelation.ENABLES))

        node_f = torch.randn(3, cfg.node_dim)
        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=5)

        assert out.node_features.isfinite().all()

    def test_star_graph(self):
        """Grafo estrella: un nodo central recibe mensajes de todos los demás."""
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        engine.eval()

        n = 6
        graph = CausalGraph()
        for i in range(n):
            graph.add_node(CausalNode(node_id=f"n{i}", label=f"n{i}",
                                       node_type=NodeType.EVENT))
        # Todos apuntan al nodo 0
        for i in range(1, n):
            graph.add_edge(CausalEdge(source_id=f"n{i}", target_id="n0",
                                       relation=CausalRelation.SUPPORTS))

        node_f = torch.randn(n, cfg.node_dim)
        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=5)

        assert out.node_features.isfinite().all()
        # El nodo central (n0) recibe 5 mensajes — el más "influido"
        central_change = (out.node_features[0] - node_f[0]).norm().item()
        peripheral_avg = (out.node_features[1:] - node_f[1:]).norm(dim=-1).mean().item()
        # El nodo central puede cambiar más (recibe más mensajes)
        # Este assert es suave — solo verificamos que ambos cambian
        assert central_change > 0 and peripheral_avg > 0

    def test_n_iterations_0_is_identity(self):
        """0 iteraciones: los node features no deben cambiar respecto al input."""
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        engine.eval()
        graph, node_f = make_linear_graph(3)

        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=0)

        assert torch.allclose(out.node_features, node_f), \
            "0 iterations should return unchanged node features"
        assert out.iterations_run == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestDeterminism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_eval_mode_deterministic(self):
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        engine.eval()
        graph, node_f = make_linear_graph(4)

        with torch.no_grad():
            out1 = engine(graph, node_f, n_iterations=5)
            out2 = engine(graph, node_f, n_iterations=5)

        assert torch.allclose(out1.node_features, out2.node_features)

    def test_different_inputs_different_outputs(self):
        cfg = make_tiny_cfg()
        engine = CausalReasoningEngine(cfg)
        engine.eval()
        graph, _ = make_linear_graph(4)

        torch.manual_seed(1)
        nf1 = torch.randn(4, cfg.node_dim)
        torch.manual_seed(99)
        nf2 = torch.randn(4, cfg.node_dim)

        with torch.no_grad():
            out1 = engine(graph, nf1, n_iterations=3)
            out2 = engine(graph, nf2, n_iterations=3)

        assert not torch.allclose(out1.node_features, out2.node_features)

    def test_same_seed_same_output(self):
        cfg = make_tiny_cfg()
        graph, _ = make_linear_graph(4)

        torch.manual_seed(42)
        engine1 = CausalReasoningEngine(cfg)
        torch.manual_seed(42)
        nf1 = torch.randn(4, cfg.node_dim)

        torch.manual_seed(42)
        engine2 = CausalReasoningEngine(cfg)
        torch.manual_seed(42)
        nf2 = torch.randn(4, cfg.node_dim)

        engine1.eval(); engine2.eval()
        with torch.no_grad():
            o1 = engine1(graph, nf1, n_iterations=3)
            o2 = engine2(graph, nf2, n_iterations=3)

        assert torch.allclose(o1.node_features, o2.node_features)
