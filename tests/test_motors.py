"""
tests/test_motors.py — Tests para BaseMotor y CORAMotor
========================================================

Verifica que:
    1. BaseMotor es realmente abstracta (no se puede instanciar directamente)
    2. CORAMotor implementa correctamente la interfaz BaseMotor
    3. CORAMotor.define_node_types() y define_relations() devuelven los valores correctos
    4. CORAMotor.build_graph() produce CrystallizerOutput con shapes correctos
    5. CORAMotor.reason() produce CREOutput con shapes correctos
    6. CORAMotor.get_graph_repr() produce tensor [k_nodes, D] en todos los casos
    7. CORAMotor es un nn.Module entrenables (tiene parámetros)
    8. Integración end-to-end: build_graph → reason → get_graph_repr
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from core.graph import CAUSAL_RELATIONS, NODE_TYPES, CausalGraph, CausalNode, CausalEdge, NodeType, CausalRelation
from motors.base_motor import BaseMotor
from motors.cora.motor import CORAMotor, CORAMotorConfig
from crystallizer.config import CrystallizerConfig
from cre.config import CREConfig


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_tiny_config() -> CORAMotorConfig:
    """Config mínima para tests rápidos."""
    cryst = CrystallizerConfig(hidden_dim=64, max_nodes=8, pooler_heads=4)
    cre   = CREConfig(node_dim=64, edge_dim=16, message_dim=32,
                      n_message_layers=1, max_iterations=3)
    return CORAMotorConfig(crystallizer=cryst, cre=cre)


def make_motor() -> CORAMotor:
    return CORAMotor(make_tiny_config())


def make_small_graph(n_nodes: int = 3, n_edges: int = 2) -> CausalGraph:
    graph = CausalGraph()
    for i in range(n_nodes):
        graph.add_node(CausalNode(
            node_id=f"n{i}", label=f"node{i}",
            node_type=NodeType.EVENT, confidence=0.9,
        ))
    for i in range(min(n_edges, n_nodes - 1)):
        graph.add_edge(CausalEdge(
            source_id=f"n{i}", target_id=f"n{i+1}",
            relation=CausalRelation.CAUSES, strength=0.8, confidence=0.9,
        ))
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# 1. BASE MOTOR ABSTRACT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseMotorAbstract:
    """BaseMotor no se puede instanciar directamente."""

    def test_cannot_instantiate_base_motor_directly(self):
        with pytest.raises(TypeError):
            BaseMotor()

    def test_incomplete_subclass_cannot_instantiate(self):
        """Una subclase que no implementa todos los métodos abstractos falla."""
        class IncompleteMotor(BaseMotor):
            def define_node_types(self): return []
            def define_relations(self): return []
            # build_graph, reason, get_graph_repr no implementados

        with pytest.raises(TypeError):
            IncompleteMotor()

    def test_complete_subclass_can_instantiate(self):
        """Una subclase que implementa todos los métodos se puede instanciar."""
        class MinimalMotor(BaseMotor):
            def define_node_types(self): return ["entity"]
            def define_relations(self): return ["causes"]
            def build_graph(self, concepts): return None
            def reason(self, graph, node_features, n_iterations=3): return None
            def get_graph_repr(self, cre_output, k_nodes): return None

        m = MinimalMotor()
        assert isinstance(m, BaseMotor)
        assert isinstance(m, nn.Module)

    def test_base_motor_is_nn_module(self):
        """BaseMotor hereda de nn.Module."""
        assert issubclass(BaseMotor, nn.Module)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CORA MOTOR IMPLEMENTS BASE MOTOR
# ─────────────────────────────────────────────────────────────────────────────

class TestCORAMotorInterface:
    """CORAMotor implementa todos los métodos de BaseMotor."""

    def test_cora_motor_is_base_motor(self):
        motor = make_motor()
        assert isinstance(motor, BaseMotor)

    def test_cora_motor_is_nn_module(self):
        motor = make_motor()
        assert isinstance(motor, nn.Module)

    def test_has_define_node_types(self):
        motor = make_motor()
        assert callable(getattr(motor, "define_node_types", None))

    def test_has_define_relations(self):
        motor = make_motor()
        assert callable(getattr(motor, "define_relations", None))

    def test_has_build_graph(self):
        motor = make_motor()
        assert callable(getattr(motor, "build_graph", None))

    def test_has_reason(self):
        motor = make_motor()
        assert callable(getattr(motor, "reason", None))

    def test_has_get_graph_repr(self):
        motor = make_motor()
        assert callable(getattr(motor, "get_graph_repr", None))

    def test_has_trainable_parameters(self):
        motor = make_motor()
        params = list(motor.parameters())
        assert len(params) > 0, "CORAMotor must have trainable parameters"

    def test_has_crystallizer_and_cre_submodules(self):
        motor = make_motor()
        assert hasattr(motor, "crystallizer")
        assert hasattr(motor, "cre")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DEFINE_NODE_TYPES AND DEFINE_RELATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestCORAMotorIntrospection:
    """define_node_types y define_relations devuelven los valores correctos."""

    def test_node_types_match_core(self):
        motor = make_motor()
        result = motor.define_node_types()
        assert isinstance(result, list)
        assert set(result) == set(NODE_TYPES)

    def test_node_types_count(self):
        motor = make_motor()
        assert len(motor.define_node_types()) == 7

    def test_relations_match_core(self):
        motor = make_motor()
        result = motor.define_relations()
        assert isinstance(result, list)
        assert set(result) == set(CAUSAL_RELATIONS)

    def test_relations_count(self):
        motor = make_motor()
        assert len(motor.define_relations()) == 16

    def test_node_types_are_strings(self):
        motor = make_motor()
        for nt in motor.define_node_types():
            assert isinstance(nt, str)

    def test_relations_are_strings(self):
        motor = make_motor()
        for rel in motor.define_relations():
            assert isinstance(rel, str)


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD_GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class TestCORABuildGraph:
    """build_graph produce CrystallizerOutput con shapes correctos."""

    def test_returns_graphs_list(self):
        motor = make_motor()
        motor.eval()
        concepts = torch.randn(2, 16, 64)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        assert hasattr(out, "graphs")
        assert len(out.graphs) == 2

    def test_graphs_are_causal_graphs(self):
        motor = make_motor()
        motor.eval()
        concepts = torch.randn(2, 16, 64)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        for g in out.graphs:
            assert isinstance(g, CausalGraph)

    def test_node_vectors_shape(self):
        motor = make_motor()
        motor.eval()
        B, L, D = 3, 12, 64
        concepts = torch.randn(B, L, D)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        # node_vectors: [B, K, D] where K = min(L, max_nodes)
        K = min(L, motor.config.crystallizer.max_nodes)
        assert out.node_vectors.shape == (B, K, D)

    def test_node_counts_length(self):
        motor = make_motor()
        motor.eval()
        B = 4
        concepts = torch.randn(B, 10, 64)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        assert len(out.node_counts) == B

    def test_node_counts_within_bounds(self):
        motor = make_motor()
        motor.eval()
        concepts = torch.randn(2, 10, 64)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        K = min(10, motor.config.crystallizer.max_nodes)
        for count in out.node_counts:
            assert 0 <= count <= K

    def test_single_batch(self):
        motor = make_motor()
        motor.eval()
        concepts = torch.randn(1, 8, 64)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        assert len(out.graphs) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. REASON
# ─────────────────────────────────────────────────────────────────────────────

class TestCORAMotorReason:
    """reason() produce CREOutput con shapes correctos."""

    def test_returns_cre_output(self):
        motor = make_motor()
        motor.eval()
        graph = make_small_graph(3, 2)
        feats = torch.randn(3, 64)
        with torch.no_grad():
            out = motor.reason(graph, feats, n_iterations=2)
        assert hasattr(out, "node_features")
        assert hasattr(out, "edge_features")
        assert hasattr(out, "iterations_run")

    def test_node_features_shape(self):
        motor = make_motor()
        motor.eval()
        N = 5
        graph = make_small_graph(N, 3)
        feats = torch.randn(N, 64)
        with torch.no_grad():
            out = motor.reason(graph, feats, n_iterations=2)
        assert out.node_features.shape == (N, 64)

    def test_edge_features_shape(self):
        motor = make_motor()
        motor.eval()
        N, E = 4, 3
        graph = make_small_graph(N, E)
        feats = torch.randn(N, 64)
        with torch.no_grad():
            out = motor.reason(graph, feats, n_iterations=2)
        assert out.edge_features.shape == (E, motor.config.cre.edge_dim)

    def test_iterations_run(self):
        motor = make_motor()
        motor.eval()
        graph = make_small_graph(3, 2)
        feats = torch.randn(3, 64)
        n_iters = 3
        with torch.no_grad():
            out = motor.reason(graph, feats, n_iterations=n_iters)
        assert out.iterations_run == n_iters

    def test_graph_without_edges(self):
        motor = make_motor()
        motor.eval()
        graph = make_small_graph(4, 0)
        feats = torch.randn(4, 64)
        with torch.no_grad():
            out = motor.reason(graph, feats, n_iterations=2)
        assert out.node_features.shape == (4, 64)
        assert out.edge_features.shape == (0, motor.config.cre.edge_dim)

    def test_reason_default_iterations(self):
        motor = make_motor()
        motor.eval()
        graph = make_small_graph(3, 2)
        feats = torch.randn(3, 64)
        with torch.no_grad():
            out = motor.reason(graph, feats)
        assert out.iterations_run == 3


# ─────────────────────────────────────────────────────────────────────────────
# 6. GET_GRAPH_REPR
# ─────────────────────────────────────────────────────────────────────────────

class TestCORAGetGraphRepr:
    """get_graph_repr produce [k_nodes, D] en todos los casos."""

    def _make_cre_output(self, n_nodes: int, dim: int = 64, n_edges: int = 0):
        from cre.engine import CREOutput
        return CREOutput(
            node_features=torch.randn(n_nodes, dim),
            edge_features=torch.zeros(n_edges, 16),
            iterations_run=2,
            layer_outputs=[],
        )

    def test_output_shape_exact(self):
        motor = make_motor()
        cre_out = self._make_cre_output(n_nodes=8)
        result = motor.get_graph_repr(cre_out, k_nodes=8)
        assert result.shape == (8, 64)

    def test_output_shape_more_nodes_than_k(self):
        """Cuando N > k_nodes, selecciona los k más relevantes."""
        motor = make_motor()
        cre_out = self._make_cre_output(n_nodes=20)
        result = motor.get_graph_repr(cre_out, k_nodes=8)
        assert result.shape == (8, 64)

    def test_output_shape_fewer_nodes_than_k(self):
        """Cuando N < k_nodes, rellena con ceros."""
        motor = make_motor()
        cre_out = self._make_cre_output(n_nodes=3)
        result = motor.get_graph_repr(cre_out, k_nodes=8)
        assert result.shape == (8, 64)

    def test_padding_is_zeros(self):
        """Los vectores de relleno son exactamente cero."""
        motor = make_motor()
        feats = torch.ones(2, 64)
        from cre.engine import CREOutput
        cre_out = CREOutput(
            node_features=feats,
            edge_features=torch.zeros(0, 16),
            iterations_run=1,
            layer_outputs=[],
        )
        result = motor.get_graph_repr(cre_out, k_nodes=5)
        # Últimos 3 deben ser cero
        assert torch.all(result[2:] == 0.0)
        # Primeros 2 deben ser uno
        assert torch.all(result[:2] == 1.0)

    def test_empty_graph(self):
        """Grafo vacío (0 nodos) → tensor de ceros."""
        motor = make_motor()
        cre_out = self._make_cre_output(n_nodes=0)
        result = motor.get_graph_repr(cre_out, k_nodes=8)
        assert result.shape == (8, 64)
        assert torch.all(result == 0.0)

    def test_k_nodes_1(self):
        motor = make_motor()
        cre_out = self._make_cre_output(n_nodes=5)
        result = motor.get_graph_repr(cre_out, k_nodes=1)
        assert result.shape == (1, 64)

    def test_selects_highest_norm_nodes(self):
        """Cuando N > k_nodes, los nodos seleccionados son los de mayor norma."""
        motor = make_motor()
        from cre.engine import CREOutput
        feats = torch.zeros(5, 64)
        feats[2] = torch.ones(64) * 10.0   # norma alta → debe ser seleccionado
        feats[4] = torch.ones(64) * 5.0    # norma media → debe ser seleccionado
        # feats[0,1,3] tienen norma 0 → no seleccionados si k_nodes=2
        cre_out = CREOutput(
            node_features=feats,
            edge_features=torch.zeros(0, 16),
            iterations_run=1,
            layer_outputs=[],
        )
        result = motor.get_graph_repr(cre_out, k_nodes=2)
        assert result.shape == (2, 64)
        # Ambos nodos seleccionados deben tener norma alta (no cero)
        assert result.norm(dim=-1).min() > 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. END-TO-END INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestCORAMotorEndToEnd:
    """Pipeline completo: build_graph → reason → get_graph_repr."""

    def test_full_pipeline_single_item(self):
        motor = make_motor()
        motor.eval()

        concepts = torch.randn(1, 10, 64)

        with torch.no_grad():
            cryst_out = motor.build_graph(concepts)

        g     = cryst_out.graphs[0]
        n     = cryst_out.node_counts[0]
        feats = cryst_out.node_vectors[0, :n]   # [n, D]

        with torch.no_grad():
            cre_out = motor.reason(g, feats, n_iterations=2)

        repr_ = motor.get_graph_repr(cre_out, k_nodes=4)
        assert repr_.shape == (4, 64)

    def test_full_pipeline_batch(self):
        motor = make_motor()
        motor.eval()

        B = 3
        concepts = torch.randn(B, 12, 64)

        with torch.no_grad():
            cryst_out = motor.build_graph(concepts)

        reprs = []
        for b in range(B):
            g     = cryst_out.graphs[b]
            n     = cryst_out.node_counts[b]
            feats = cryst_out.node_vectors[b, :n]
            with torch.no_grad():
                cre_out = motor.reason(g, feats, n_iterations=2)
            repr_ = motor.get_graph_repr(cre_out, k_nodes=4)
            reprs.append(repr_)

        assert len(reprs) == B
        for r in reprs:
            assert r.shape == (4, 64)

    def test_gradients_flow_through_pipeline(self):
        """Los gradientes fluyen desde get_graph_repr hasta build_graph."""
        motor = make_motor()

        concepts = torch.randn(1, 8, 64, requires_grad=True)
        cryst_out = motor.build_graph(concepts)

        # node_vectors es diferenciable
        g     = cryst_out.graphs[0]
        n     = max(cryst_out.node_counts[0], 1)
        feats = cryst_out.node_vectors[0, :n]

        cre_out = motor.reason(g, feats, n_iterations=1)
        repr_   = motor.get_graph_repr(cre_out, k_nodes=4)

        loss = repr_.sum()
        loss.backward()

        assert concepts.grad is not None, "Gradients must flow to concepts"

    def test_motor_in_train_mode(self):
        motor = make_motor()
        motor.train()
        concepts = torch.randn(1, 8, 64)
        cryst_out = motor.build_graph(concepts)
        assert len(cryst_out.graphs) == 1

    def test_config_dimension_mismatch_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, pooler_heads=4)
        cre   = CREConfig(node_dim=128)  # mismatch
        with pytest.raises(ValueError):
            CORAMotorConfig(crystallizer=cryst, cre=cre)
