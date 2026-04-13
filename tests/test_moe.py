"""
tests/test_moe.py — Tests para SparseMoE y ExpertGroup
=======================================================

Cubre:
    - ExpertGroup: shapes, independencia de expertos, gradientes
    - SparseMoE:   shapes, routing, load balance loss, residual, gradientes
    - Engine integration: use_moe=True en CausalReasoningEngine
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from cre.config import CREConfig
from cre.moe import ExpertGroup, MoEOutput, SparseMoE
from cre.engine import CREOutput, CausalReasoningEngine
from core.graph import CausalGraph, CausalNode, CausalEdge, NodeType, CausalRelation


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_config() -> CREConfig:
    return CREConfig(
        node_dim             = 64,
        edge_dim             = 16,
        message_dim          = 32,
        n_message_layers     = 1,
        max_iterations       = 3,
        use_moe              = True,
        moe_n_groups         = 2,
        moe_experts_per_group= 2,   # 4 expertos totales
        moe_active_experts   = 2,
        moe_expert_hidden_mult = 2,
        moe_load_balance_weight = 0.01,
    )


@pytest.fixture
def expert_group() -> ExpertGroup:
    return ExpertGroup(n_experts=4, input_dim=64, output_dim=64)


@pytest.fixture
def sparse_moe(tiny_config) -> SparseMoE:
    return SparseMoE(tiny_config)


@pytest.fixture
def small_graph() -> CausalGraph:
    """Grafo de 4 nodos y 3 aristas para tests del engine."""
    g = CausalGraph()
    for i in range(4):
        g.add_node(CausalNode(node_id=f"n{i}", label=f"node {i}", node_type=NodeType.HYPOTHESIS))
    for i in range(3):
        g.add_edge(CausalEdge(
            source_id=f"n{i}",
            target_id=f"n{i+1}",
            relation=CausalRelation.CAUSES,
            strength=0.8,
            confidence=0.9,
        ))
    return g


# ─────────────────────────────────────────────────────────────────────────────
# EXPERT GROUP
# ─────────────────────────────────────────────────────────────────────────────

class TestExpertGroup:
    def test_forward_expert_output_shape(self, expert_group):
        x = torch.randn(5, 64)
        out = expert_group.forward_expert(0, x)
        assert out.shape == (5, 64)

    def test_all_experts_valid_shapes(self, expert_group):
        x = torch.randn(3, 64)
        for i in range(expert_group.n_experts):
            out = expert_group.forward_expert(i, x)
            assert out.shape == (3, 64)

    def test_n_experts_property(self, expert_group):
        assert expert_group.n_experts == 4

    def test_experts_are_independent(self, expert_group):
        """Dos expertos distintos deben producir outputs distintos."""
        x = torch.randn(3, 64)
        out0 = expert_group.forward_expert(0, x)
        out1 = expert_group.forward_expert(1, x)
        # Con inicialización aleatoria, deben diferir
        assert not torch.allclose(out0, out1, atol=1e-4)

    def test_gradients_flow_through_expert(self, expert_group):
        x = torch.randn(3, 64, requires_grad=True)
        out = expert_group.forward_expert(2, x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_single_node_input(self, expert_group):
        """Debe funcionar con un solo nodo."""
        x = torch.randn(1, 64)
        out = expert_group.forward_expert(0, x)
        assert out.shape == (1, 64)

    def test_custom_output_dim(self):
        """input_dim != output_dim debe funcionar."""
        group = ExpertGroup(n_experts=3, input_dim=32, output_dim=64)
        x = torch.randn(5, 32)
        out = group.forward_expert(0, x)
        assert out.shape == (5, 64)


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE MOE — SHAPE Y ROUTING
# ─────────────────────────────────────────────────────────────────────────────

class TestSparseMoEShapes:
    def test_output_shape(self, sparse_moe, tiny_config):
        N = 7
        x = torch.randn(N, tiny_config.node_dim)
        result = sparse_moe(x)
        assert result.output.shape == (N, tiny_config.node_dim)

    def test_router_probs_shape(self, sparse_moe, tiny_config):
        N = 6
        x = torch.randn(N, tiny_config.node_dim)
        result = sparse_moe(x)
        n_experts = tiny_config.moe_n_groups * tiny_config.moe_experts_per_group
        assert result.router_probs.shape == (N, n_experts)

    def test_top_k_indices_shape(self, sparse_moe, tiny_config):
        N = 5
        x = torch.randn(N, tiny_config.node_dim)
        result = sparse_moe(x)
        assert result.top_k_indices.shape == (N, tiny_config.moe_active_experts)

    def test_load_balance_loss_is_scalar(self, sparse_moe, tiny_config):
        x = torch.randn(8, tiny_config.node_dim)
        result = sparse_moe(x)
        assert result.load_balance_loss.ndim == 0

    def test_output_type_is_moe_output(self, sparse_moe, tiny_config):
        x = torch.randn(4, tiny_config.node_dim)
        result = sparse_moe(x)
        assert isinstance(result, MoEOutput)


class TestSparseMoERouting:
    def test_router_probs_sum_to_one(self, sparse_moe, tiny_config):
        x = torch.randn(10, tiny_config.node_dim)
        result = sparse_moe(x)
        sums = result.router_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_top_k_indices_in_valid_range(self, sparse_moe, tiny_config):
        n_experts = tiny_config.moe_n_groups * tiny_config.moe_experts_per_group
        x = torch.randn(8, tiny_config.node_dim)
        result = sparse_moe(x)
        assert result.top_k_indices.min() >= 0
        assert result.top_k_indices.max() < n_experts

    def test_load_balance_loss_nonnegative(self, sparse_moe, tiny_config):
        x = torch.randn(10, tiny_config.node_dim)
        result = sparse_moe(x)
        assert result.load_balance_loss.item() >= 0.0

    def test_router_probs_detached(self, sparse_moe, tiny_config):
        """router_probs y top_k_indices no deben tener grad_fn."""
        x = torch.randn(5, tiny_config.node_dim)
        result = sparse_moe(x)
        assert result.router_probs.grad_fn is None
        assert result.top_k_indices.grad_fn is None

    def test_routing_diversity_under_random_input(self, sparse_moe, tiny_config):
        """Con suficientes nodos, más de un experto debe ser seleccionado."""
        torch.manual_seed(42)
        x = torch.randn(50, tiny_config.node_dim)
        result = sparse_moe(x)
        n_experts = tiny_config.moe_n_groups * tiny_config.moe_experts_per_group
        unique_experts = result.top_k_indices.unique().numel()
        assert unique_experts > 1, "El router debe seleccionar más de un experto"


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE MOE — GRADIENTES Y RESIDUAL
# ─────────────────────────────────────────────────────────────────────────────

class TestSparseMoEGradients:
    def test_gradients_flow_to_input(self, sparse_moe, tiny_config):
        x = torch.randn(5, tiny_config.node_dim, requires_grad=True)
        result = sparse_moe(x)
        result.output.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradients_flow_through_load_balance_loss(self, sparse_moe, tiny_config):
        x = torch.randn(8, tiny_config.node_dim, requires_grad=True)
        result = sparse_moe(x)
        # La lb_loss fluye gradiente a través de mean_probs (que depende de x)
        total_loss = result.output.sum() + result.load_balance_loss
        total_loss.backward()
        assert x.grad is not None

    def test_output_has_residual(self, tiny_config):
        """output = LayerNorm(x + moe_transform(x)) — no debe ser cero."""
        moe = SparseMoE(tiny_config)
        x = torch.randn(4, tiny_config.node_dim)
        result = moe(x)
        # El output debe ser diferente del input por el residual + transformación
        assert not torch.allclose(result.output, x, atol=1e-3)

    def test_parameters_have_gradients(self, sparse_moe, tiny_config):
        x = torch.randn(6, tiny_config.node_dim)
        result = sparse_moe(x)
        result.output.sum().backward()
        # Al menos algunos parámetros deben tener gradientes
        params_with_grad = [
            p for p in sparse_moe.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(params_with_grad) > 0


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE MOE — LOAD BALANCE LOSS
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadBalanceLoss:
    def test_loss_scales_with_weight(self, tiny_config):
        config_low  = CREConfig(**{**tiny_config.__dict__, "moe_load_balance_weight": 0.001})
        config_high = CREConfig(**{**tiny_config.__dict__, "moe_load_balance_weight": 0.1})
        # Mismo seed para mismo routing
        torch.manual_seed(0)
        x = torch.randn(10, tiny_config.node_dim)
        moe_low  = SparseMoE(config_low)
        moe_high = SparseMoE(config_high)
        # Copiar pesos para que el routing sea idéntico
        moe_high.load_state_dict(moe_low.state_dict())
        r_low  = moe_low(x)
        r_high = moe_high(x)
        # lb_loss_high debe ser ~100x lb_loss_low
        ratio = r_high.load_balance_loss.item() / (r_low.load_balance_loss.item() + 1e-12)
        assert abs(ratio - 100.0) < 5.0, f"Expected ratio ~100, got {ratio:.2f}"

    def test_loss_minimum_with_uniform_routing(self):
        """Con routing perfectamente uniforme, lb_loss = lb_weight × 1.0."""
        # No podemos forzar routing uniforme fácilmente, pero verificamos que el
        # loss está en el rango razonable [lb_weight, lb_weight × n_experts]
        config = CREConfig(
            node_dim=32, edge_dim=8, message_dim=16,
            use_moe=True, moe_n_groups=2, moe_experts_per_group=2,
            moe_active_experts=2, moe_load_balance_weight=0.01,
        )
        moe = SparseMoE(config)
        x = torch.randn(100, 32)
        result = moe(x)
        lb = result.load_balance_loss.item()
        n_experts = config.moe_n_groups * config.moe_experts_per_group
        assert 0.0 <= lb <= config.moe_load_balance_weight * n_experts * 1.5


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineWithMoE:
    def test_engine_instantiation_with_moe(self, tiny_config):
        engine = CausalReasoningEngine(tiny_config)
        assert engine.moe is not None
        assert engine.moe_gru is not None

    def test_engine_instantiation_without_moe(self):
        config = CREConfig(
            node_dim=64, edge_dim=16, message_dim=32,
            use_moe=False,
        )
        engine = CausalReasoningEngine(config)
        assert engine.moe is None
        assert engine.moe_gru is None

    def test_engine_forward_with_moe(self, tiny_config, small_graph):
        engine = CausalReasoningEngine(tiny_config)
        N = len(small_graph.nodes)
        x = torch.randn(N, tiny_config.node_dim)
        result = engine(small_graph, x)
        assert isinstance(result, CREOutput)
        assert result.node_features.shape == (N, tiny_config.node_dim)

    def test_engine_moe_returns_load_balance_loss(self, tiny_config, small_graph):
        engine = CausalReasoningEngine(tiny_config)
        N = len(small_graph.nodes)
        x = torch.randn(N, tiny_config.node_dim)
        result = engine(small_graph, x)
        assert result.load_balance_loss is not None
        assert result.load_balance_loss.ndim == 0

    def test_engine_no_moe_load_balance_loss_is_none(self, small_graph):
        config = CREConfig(
            node_dim=64, edge_dim=16, message_dim=32,
            n_message_layers=1, max_iterations=2, use_moe=False,
        )
        engine = CausalReasoningEngine(config)
        N = len(small_graph.nodes)
        x = torch.randn(N, config.node_dim)
        result = engine(small_graph, x)
        assert result.load_balance_loss is None

    def test_engine_gradients_flow_with_moe(self, tiny_config, small_graph):
        engine = CausalReasoningEngine(tiny_config)
        N = len(small_graph.nodes)
        x = torch.randn(N, tiny_config.node_dim, requires_grad=True)
        result = engine(small_graph, x)
        total_loss = result.node_features.sum()
        if result.load_balance_loss is not None:
            total_loss = total_loss + result.load_balance_loss
        total_loss.backward()
        assert x.grad is not None

    def test_engine_moe_iterations_accumulate_loss(self, tiny_config, small_graph):
        """Con más iteraciones, lb_loss debe ser mayor (acumulada por iter)."""
        engine = CausalReasoningEngine(tiny_config)
        N = len(small_graph.nodes)
        x = torch.randn(N, tiny_config.node_dim)

        result_1 = engine(small_graph, x, n_iterations=1)
        result_3 = engine(small_graph, x, n_iterations=3)

        lb_1 = result_1.load_balance_loss.item()
        lb_3 = result_3.load_balance_loss.item()
        # Con 3 iteraciones debería acumular ~3x más (puede variar por routing)
        assert lb_3 > lb_1, f"Expected lb_3 ({lb_3:.6f}) > lb_1 ({lb_1:.6f})"

    def test_count_parameters_includes_moe(self, tiny_config):
        config_no_moe = CREConfig(
            node_dim=64, edge_dim=16, message_dim=32,
            n_message_layers=1, max_iterations=3, use_moe=False,
        )
        engine_moe    = CausalReasoningEngine(tiny_config)
        engine_no_moe = CausalReasoningEngine(config_no_moe)
        assert engine_moe.count_parameters() > engine_no_moe.count_parameters()
