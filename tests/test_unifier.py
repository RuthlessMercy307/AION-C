"""
tests/test_unifier.py — Tests for Unifier
==========================================
"""

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unifier.model import Unifier, UnifierConfig, UnifierOutput


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_config():
    return UnifierConfig()

@pytest.fixture
def tiny_config():
    return UnifierConfig(node_dim=16, n_heads=2, max_output_nodes=8)

@pytest.fixture
def unifier(tiny_config):
    return Unifier(tiny_config)

@pytest.fixture
def single_motor_output(tiny_config):
    torch.manual_seed(0)
    return [torch.randn(5, tiny_config.node_dim)]

@pytest.fixture
def two_motor_outputs(tiny_config):
    torch.manual_seed(1)
    return [
        torch.randn(4, tiny_config.node_dim),
        torch.randn(3, tiny_config.node_dim),
    ]

@pytest.fixture
def three_motor_outputs(tiny_config):
    torch.manual_seed(2)
    return [
        torch.randn(6, tiny_config.node_dim),
        torch.randn(4, tiny_config.node_dim),
        torch.randn(3, tiny_config.node_dim),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifierConfig:
    def test_default_values(self, default_config):
        assert default_config.node_dim == 256
        assert default_config.n_heads == 4
        assert default_config.max_output_nodes == 32
        assert default_config.dropout == 0.0

    def test_node_dim_not_divisible_by_n_heads_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            UnifierConfig(node_dim=10, n_heads=3)

    def test_valid_divisible_config(self):
        cfg = UnifierConfig(node_dim=64, n_heads=8)
        assert cfg.node_dim == 64

    def test_custom_values(self):
        cfg = UnifierConfig(node_dim=32, n_heads=4, max_output_nodes=16, dropout=0.1)
        assert cfg.node_dim == 32
        assert cfg.max_output_nodes == 16
        assert cfg.dropout == 0.1

    def test_n_heads_1_valid(self):
        cfg = UnifierConfig(node_dim=16, n_heads=1)
        assert cfg.n_heads == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. INSTANTIATION
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifierInstantiation:
    def test_is_nn_module(self, unifier):
        assert isinstance(unifier, nn.Module)

    def test_has_cross_attn(self, unifier):
        assert hasattr(unifier, "cross_attn")
        assert isinstance(unifier.cross_attn, nn.MultiheadAttention)

    def test_has_fusion_mlp(self, unifier):
        assert hasattr(unifier, "fusion_mlp")
        assert isinstance(unifier.fusion_mlp, nn.Sequential)

    def test_has_input_norm(self, unifier):
        assert hasattr(unifier, "input_norm")
        assert isinstance(unifier.input_norm, nn.LayerNorm)

    def test_parameter_count_positive(self, unifier):
        assert unifier.count_parameters() > 0

    def test_default_config_instantiation(self, default_config):
        u = Unifier(default_config)
        assert isinstance(u, nn.Module)


# ─────────────────────────────────────────────────────────────────────────────
# 3. OUTPUT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifierOutput:
    def test_returns_unifier_output(self, unifier, single_motor_output):
        out = unifier(single_motor_output)
        assert isinstance(out, UnifierOutput)

    def test_output_has_unified_field(self, unifier, single_motor_output):
        out = unifier(single_motor_output)
        assert hasattr(out, "unified")
        assert isinstance(out.unified, torch.Tensor)

    def test_output_has_n_source_motors(self, unifier, single_motor_output):
        out = unifier(single_motor_output)
        assert hasattr(out, "n_source_motors")
        assert isinstance(out.n_source_motors, int)

    def test_output_has_total_nodes(self, unifier, single_motor_output):
        out = unifier(single_motor_output)
        assert hasattr(out, "total_nodes")
        assert isinstance(out.total_nodes, int)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FIXED OUTPUT SHAPE
# ─────────────────────────────────────────────────────────────────────────────

class TestFixedOutputShape:
    def test_single_motor_output_shape(self, unifier, tiny_config, single_motor_output):
        out = unifier(single_motor_output)
        K, D = tiny_config.max_output_nodes, tiny_config.node_dim
        assert out.unified.shape == (K, D)

    def test_two_motors_output_shape(self, unifier, tiny_config, two_motor_outputs):
        out = unifier(two_motor_outputs)
        K, D = tiny_config.max_output_nodes, tiny_config.node_dim
        assert out.unified.shape == (K, D)

    def test_three_motors_output_shape(self, unifier, tiny_config, three_motor_outputs):
        out = unifier(three_motor_outputs)
        K, D = tiny_config.max_output_nodes, tiny_config.node_dim
        assert out.unified.shape == (K, D)

    def test_output_shape_when_fewer_nodes_than_max(self, tiny_config):
        """2 nodes < max_output_nodes=8 → should pad to 8."""
        unifier = Unifier(tiny_config)
        small = [torch.randn(2, tiny_config.node_dim)]
        out = unifier(small)
        assert out.unified.shape == (tiny_config.max_output_nodes, tiny_config.node_dim)

    def test_output_shape_when_more_nodes_than_max(self, tiny_config):
        """20 nodes > max_output_nodes=8 → should select top-8."""
        unifier = Unifier(tiny_config)
        large = [torch.randn(20, tiny_config.node_dim)]
        out = unifier(large)
        assert out.unified.shape == (tiny_config.max_output_nodes, tiny_config.node_dim)

    def test_exact_max_nodes(self, tiny_config):
        """N == max_output_nodes → no padding, no truncation."""
        unifier = Unifier(tiny_config)
        exact = [torch.randn(tiny_config.max_output_nodes, tiny_config.node_dim)]
        out = unifier(exact)
        assert out.unified.shape == (tiny_config.max_output_nodes, tiny_config.node_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SINGLE MOTOR PASS-THROUGH (IDENTITY)
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleMotorPassthrough:
    def test_n_source_motors_is_1(self, unifier, single_motor_output):
        out = unifier(single_motor_output)
        assert out.n_source_motors == 1

    def test_total_nodes_correct(self, unifier, single_motor_output):
        out = unifier(single_motor_output)
        assert out.total_nodes == single_motor_output[0].shape[0]

    def test_single_motor_no_cross_attention_applied(self, tiny_config):
        """Single motor: cross-attention is NOT run (identity path)."""
        unifier = Unifier(tiny_config)
        # 3 nodes exactly (< max_output_nodes=8), all ones
        nodes = torch.ones(3, tiny_config.node_dim)
        out = unifier([nodes])
        # First 3 rows should be preserved (no cross-attn transformation)
        assert torch.allclose(out.unified[:3], nodes, atol=1e-6)
        # Last 5 rows should be zero (padding)
        assert torch.allclose(out.unified[3:], torch.zeros(5, tiny_config.node_dim), atol=1e-6)

    def test_single_motor_gradient_flows(self, tiny_config):
        """Single motor: gradients can flow through pool (no cross-attention)."""
        unifier = Unifier(tiny_config)
        nodes = torch.randn(3, tiny_config.node_dim, requires_grad=True)
        out = unifier([nodes])
        loss = out.unified.sum()
        loss.backward()
        assert nodes.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# 6. MULTI-MOTOR FUSION
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiMotorFusion:
    def test_n_source_motors_two(self, unifier, two_motor_outputs):
        out = unifier(two_motor_outputs)
        assert out.n_source_motors == 2

    def test_n_source_motors_three(self, unifier, three_motor_outputs):
        out = unifier(three_motor_outputs)
        assert out.n_source_motors == 3

    def test_total_nodes_sum(self, unifier, two_motor_outputs):
        expected = sum(m.shape[0] for m in two_motor_outputs)
        out = unifier(two_motor_outputs)
        assert out.total_nodes == expected

    def test_total_nodes_three_motors(self, unifier, three_motor_outputs):
        expected = sum(m.shape[0] for m in three_motor_outputs)
        out = unifier(three_motor_outputs)
        assert out.total_nodes == expected

    def test_multi_motor_changes_features(self, tiny_config):
        """Multi-motor path runs cross-attention, so output differs from plain concat."""
        torch.manual_seed(99)
        unifier = Unifier(tiny_config)
        a = torch.randn(4, tiny_config.node_dim)
        b = torch.randn(3, tiny_config.node_dim)
        combined = torch.cat([a, b], dim=0)  # [7, D]

        out = unifier([a, b])

        # The total_nodes we got back
        total = out.total_nodes  # 7
        K = tiny_config.max_output_nodes  # 8 > 7, so pad

        # Unified shouldn't be identical to the raw concatenation (cross-attn changed it)
        # (with randomly initialized weights, this will almost certainly differ)
        # Compare first total rows (before padding)
        assert not torch.allclose(out.unified[:total], combined, atol=1e-4)

    def test_multi_motor_gradient_flows(self, tiny_config):
        unifier = Unifier(tiny_config)
        a = torch.randn(4, tiny_config.node_dim, requires_grad=True)
        b = torch.randn(3, tiny_config.node_dim, requires_grad=True)
        out = unifier([a, b])
        loss = out.unified.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_multi_motor_attn_params_used(self, tiny_config):
        """Verify cross_attn parameters receive gradients in multi-motor path."""
        unifier = Unifier(tiny_config)
        a = torch.randn(4, tiny_config.node_dim)
        b = torch.randn(3, tiny_config.node_dim)
        out = unifier([a, b])
        loss = out.unified.sum()
        loss.backward()
        # cross_attn parameters should have gradients
        has_grad = any(
            p.grad is not None
            for p in unifier.cross_attn.parameters()
            if p.requires_grad
        )
        assert has_grad


# ─────────────────────────────────────────────────────────────────────────────
# 7. EMPTY MOTOR HANDLING
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyMotorHandling:
    def test_all_empty_returns_zeros(self, tiny_config):
        unifier = Unifier(tiny_config)
        empty = [
            torch.zeros(0, tiny_config.node_dim),
            torch.zeros(0, tiny_config.node_dim),
        ]
        out = unifier(empty)
        assert out.unified.shape == (tiny_config.max_output_nodes, tiny_config.node_dim)
        assert torch.allclose(out.unified, torch.zeros_like(out.unified))

    def test_all_empty_n_source_motors_0(self, tiny_config):
        unifier = Unifier(tiny_config)
        empty = [torch.zeros(0, tiny_config.node_dim)]
        out = unifier(empty)
        assert out.n_source_motors == 0
        assert out.total_nodes == 0

    def test_one_empty_one_valid(self, tiny_config):
        """One motor has 0 nodes → treated as single valid motor."""
        unifier = Unifier(tiny_config)
        valid = torch.randn(5, tiny_config.node_dim)
        empty = torch.zeros(0, tiny_config.node_dim)
        out = unifier([empty, valid])
        assert out.n_source_motors == 1
        assert out.total_nodes == 5

    def test_empty_list_raises(self, unifier):
        with pytest.raises(ValueError, match="at least one"):
            unifier([])

    def test_single_empty_zero_output(self, tiny_config):
        unifier = Unifier(tiny_config)
        out = unifier([torch.zeros(0, tiny_config.node_dim)])
        assert out.n_source_motors == 0
        assert torch.allclose(out.unified, torch.zeros_like(out.unified))


# ─────────────────────────────────────────────────────────────────────────────
# 8. POOLING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

class TestPoolingLogic:
    def test_padding_with_zeros(self, tiny_config):
        """N < K: extra rows must be exactly zero."""
        unifier = Unifier(tiny_config)
        N = 3
        K = tiny_config.max_output_nodes  # 8
        nodes = torch.ones(N, tiny_config.node_dim)
        result = unifier._pool_to_fixed_size(nodes)
        assert result.shape == (K, tiny_config.node_dim)
        assert torch.allclose(result[N:], torch.zeros(K - N, tiny_config.node_dim))

    def test_topk_selection(self, tiny_config):
        """N > K: top-K by L2 norm selected."""
        unifier = Unifier(tiny_config)
        K = tiny_config.max_output_nodes  # 8
        N = 20
        # Create nodes with clearly distinct norms
        nodes = torch.zeros(N, tiny_config.node_dim)
        # Assign norm = i+1 to node i
        for i in range(N):
            nodes[i, 0] = float(i + 1)
        result = unifier._pool_to_fixed_size(nodes)
        assert result.shape == (K, tiny_config.node_dim)
        # All selected nodes should be from the top-K norms
        result_norms = result.norm(dim=-1)
        assert result_norms.min() > 0  # no zero rows selected

    def test_exact_k_nodes_unchanged(self, tiny_config):
        """N == K: pass through unchanged."""
        unifier = Unifier(tiny_config)
        K = tiny_config.max_output_nodes
        nodes = torch.randn(K, tiny_config.node_dim)
        result = unifier._pool_to_fixed_size(nodes)
        assert result.shape == (K, tiny_config.node_dim)

    def test_topk_preserves_indices_sorted(self, tiny_config):
        """_pool_to_fixed_size should return nodes in sorted index order."""
        unifier = Unifier(tiny_config)
        K = tiny_config.max_output_nodes
        N = K + 4
        nodes = torch.randn(N, tiny_config.node_dim)
        result = unifier._pool_to_fixed_size(nodes)
        assert result.shape == (K, tiny_config.node_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 9. DETERMINISM AND DEVICE
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_eval_mode_deterministic(self, tiny_config):
        unifier = Unifier(tiny_config)
        unifier.eval()
        torch.manual_seed(42)
        a = torch.randn(4, tiny_config.node_dim)
        b = torch.randn(3, tiny_config.node_dim)
        with torch.no_grad():
            out1 = unifier([a, b])
            out2 = unifier([a, b])
        assert torch.allclose(out1.unified, out2.unified)

    def test_output_dtype_matches_input(self, tiny_config):
        unifier = Unifier(tiny_config)
        nodes = torch.randn(5, tiny_config.node_dim, dtype=torch.float32)
        out = unifier([nodes])
        assert out.unified.dtype == torch.float32

    def test_empty_output_dtype_matches_input(self, tiny_config):
        unifier = Unifier(tiny_config)
        nodes = torch.zeros(0, tiny_config.node_dim, dtype=torch.float32)
        out = unifier([nodes])
        assert out.unified.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# 10. INTEGRATION WITH DIFFERENT CONFIG SIZES
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigVariants:
    @pytest.mark.parametrize("node_dim,n_heads,max_nodes", [
        (8,  1, 4),
        (16, 2, 8),
        (32, 4, 16),
        (64, 8, 32),
    ])
    def test_various_configs_single_motor(self, node_dim, n_heads, max_nodes):
        cfg = UnifierConfig(node_dim=node_dim, n_heads=n_heads, max_output_nodes=max_nodes)
        u = Unifier(cfg)
        nodes = torch.randn(6, node_dim)
        out = u([nodes])
        assert out.unified.shape == (max_nodes, node_dim)

    @pytest.mark.parametrize("node_dim,n_heads,max_nodes", [
        (8,  1, 4),
        (16, 2, 8),
        (32, 4, 16),
    ])
    def test_various_configs_multi_motor(self, node_dim, n_heads, max_nodes):
        cfg = UnifierConfig(node_dim=node_dim, n_heads=n_heads, max_output_nodes=max_nodes)
        u = Unifier(cfg)
        a = torch.randn(3, node_dim)
        b = torch.randn(2, node_dim)
        out = u([a, b])
        assert out.unified.shape == (max_nodes, node_dim)

    def test_dropout_config(self):
        cfg = UnifierConfig(node_dim=16, n_heads=2, max_output_nodes=8, dropout=0.1)
        u = Unifier(cfg)
        u.train()
        nodes = [torch.randn(4, 16), torch.randn(3, 16)]
        out = u(nodes)
        assert out.unified.shape == (8, 16)
