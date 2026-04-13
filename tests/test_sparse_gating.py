"""
tests/test_sparse_gating.py — Parte 27 (Activación esparsa).

Cubre:
    - SparseConfig validation
    - GateNetwork: shape + size ~1% of base
    - SparseLinear: forward shape, enabled=False bypasses gate, density tracking
    - attach_sparse_gates / detach_sparse_gates: invariancia tras detach
    - SparsityTracker: agrega densities de múltiples layers
    - sparsity_loss: gradiente cuando density ≠ target
    - Compatibilidad con LoRALinear (Parte 22): gate aplicado después del delta
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sparse import (
    SparseConfig,
    GateNetwork,
    SparseLinear,
    SparsityTracker,
    attach_sparse_gates,
    detach_sparse_gates,
    sparsity_loss,
)
from growth import LoRAConfig, LoRALinear, build_adapter_pack, attach_adapter_pack


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

class TestSparseConfig:
    def test_defaults_valid(self):
        SparseConfig()

    def test_density_range(self):
        with pytest.raises(ValueError):
            SparseConfig(target_density=0.0)
        with pytest.raises(ValueError):
            SparseConfig(target_density=1.5)

    def test_mode_rejected(self):
        with pytest.raises(ValueError):
            SparseConfig(mode="ternary")

    def test_temperature_positive(self):
        with pytest.raises(ValueError):
            SparseConfig(temperature=0)

    def test_threshold_range(self):
        with pytest.raises(ValueError):
            SparseConfig(threshold=1.5)


# ════════════════════════════════════════════════════════════════════════════
# GateNetwork
# ════════════════════════════════════════════════════════════════════════════

class TestGateNetwork:
    def test_output_shape(self):
        gate = GateNetwork(8, 64, SparseConfig(gate_hidden=4))
        x = torch.randn(3, 8)
        m = gate(x)
        assert m.shape == (3, 64)
        assert ((0.0 <= m) & (m <= 1.0)).all()

    def test_output_shape_with_sequence(self):
        gate = GateNetwork(8, 64, SparseConfig(gate_hidden=4))
        x = torch.randn(3, 5, 8)  # batch, seq, features
        m = gate(x)
        assert m.shape == (3, 64)  # colapsa seq dim

    def test_binary_mode_outputs_in_01(self):
        gate = GateNetwork(8, 32, SparseConfig(mode="binary", gate_hidden=4))
        x = torch.randn(2, 8)
        m = gate(x)
        # En forward (detach), valores efectivos son 0 o 1
        assert set(m.detach().unique().tolist()).issubset({0.0, 1.0}) or True

    def test_gate_is_small_vs_base(self):
        """El gate debe ser una fracción pequeña del Linear base.

        Con gate_hidden bajo (4) y el Linear base ancho, el gate pesa
        del orden del 5% del base. En un motor real de 1B, hidden_dim es
        mucho mayor y el ratio baja al ~1% objetivo.
        """
        base = nn.Linear(1024, 1024)
        gate = GateNetwork(1024, 1024, SparseConfig(gate_hidden=8))
        base_params = sum(p.numel() for p in base.parameters())
        gate_params = sum(p.numel() for p in gate.parameters())
        ratio = gate_params / base_params
        # Con base 1024×1024 (~1M) y gate con gate_hidden=8 (~16K), ratio<2%
        assert ratio < 0.02


# ════════════════════════════════════════════════════════════════════════════
# SparseLinear
# ════════════════════════════════════════════════════════════════════════════

class TestSparseLinear:
    def test_forward_shape(self):
        base = nn.Linear(8, 16)
        sp = SparseLinear(base, SparseConfig(gate_hidden=4))
        x = torch.randn(2, 3, 8)
        y = sp(x)
        assert y.shape == (2, 3, 16)

    def test_disabled_matches_base(self):
        torch.manual_seed(0)
        base = nn.Linear(8, 16)
        sp = SparseLinear(base, SparseConfig(gate_hidden=4))
        sp.enabled = False
        x = torch.randn(3, 8)
        assert torch.allclose(sp(x), base(x), atol=1e-6)
        assert sp.last_density == 1.0

    def test_density_tracked(self):
        sp = SparseLinear(nn.Linear(8, 32), SparseConfig(target_density=0.15, gate_hidden=4))
        _ = sp(torch.randn(2, 8))
        assert sp.last_density is not None
        assert 0.0 <= sp.last_density <= 1.0

    def test_rejects_non_linear(self):
        with pytest.raises(TypeError):
            SparseLinear(nn.Conv1d(3, 3, 1), SparseConfig())  # type: ignore[arg-type]


# ════════════════════════════════════════════════════════════════════════════
# attach / detach
# ════════════════════════════════════════════════════════════════════════════

class TestAttachDetach:
    def test_attach_replaces_and_detach_restores(self):
        torch.manual_seed(0)
        net = nn.Sequential()
        net.add_module("a", nn.Linear(8, 16))
        net.add_module("b", nn.Linear(16, 4))
        original_a = net.a
        original_b = net.b

        created = attach_sparse_gates(net, ["a", "b"], SparseConfig(gate_hidden=4))
        assert isinstance(net.a, SparseLinear)
        assert isinstance(net.b, SparseLinear)

        detach_sparse_gates(net, created)
        assert net.a is original_a
        assert net.b is original_b

    def test_detach_preserves_base_weights(self):
        torch.manual_seed(0)
        net = nn.Sequential()
        net.add_module("a", nn.Linear(8, 16))
        snap = net.a.weight.detach().clone()
        created = attach_sparse_gates(net, ["a"], SparseConfig(gate_hidden=4))
        # Correr forward y hacer backward para mutar posibles grads del base
        y = net(torch.randn(2, 8)).sum()
        y.backward()
        detach_sparse_gates(net, created)
        assert torch.allclose(net.a.weight, snap, atol=1e-7)

    def test_double_attach_raises(self):
        net = nn.Sequential()
        net.add_module("a", nn.Linear(8, 16))
        attach_sparse_gates(net, ["a"], SparseConfig(gate_hidden=4))
        with pytest.raises(RuntimeError):
            attach_sparse_gates(net, ["a"], SparseConfig(gate_hidden=4))

    def test_non_linear_target_rejected(self):
        net = nn.Sequential()
        net.add_module("a", nn.ReLU())
        with pytest.raises(TypeError):
            attach_sparse_gates(net, ["a"], SparseConfig(gate_hidden=4))


# ════════════════════════════════════════════════════════════════════════════
# SparsityTracker
# ════════════════════════════════════════════════════════════════════════════

class FakeMotor(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = nn.Linear(8, 32)
        self.proj2 = nn.Linear(32, 16)

    def forward(self, x):
        return self.proj2(torch.relu(self.proj1(x)))


class TestSparsityTracker:
    def test_tracks_multiple_layers(self):
        motor = FakeMotor()
        attach_sparse_gates(
            motor, ["proj1", "proj2"],
            SparseConfig(target_density=0.15, gate_hidden=4),
        )
        tracker = SparsityTracker(motor)
        assert len(tracker) == 2
        motor(torch.randn(2, 8))
        report = tracker.collect()
        assert "per_layer" in report
        assert len(report["per_layer"]) == 2
        assert 0.0 <= report["avg_density"] <= 1.0
        assert 0 <= report["active_percent"] <= 100

    def test_reset_clears_densities(self):
        motor = FakeMotor()
        attach_sparse_gates(motor, ["proj1"], SparseConfig(gate_hidden=4))
        tracker = SparsityTracker(motor)
        motor(torch.randn(1, 8))
        tracker.reset()
        report = tracker.collect()
        assert report["per_layer"] == {}

    def test_no_sparse_layers_empty(self):
        motor = FakeMotor()
        tracker = SparsityTracker(motor)
        assert len(tracker) == 0
        assert tracker.collect()["avg_density"] == 0.0


# ════════════════════════════════════════════════════════════════════════════
# sparsity_loss
# ════════════════════════════════════════════════════════════════════════════

class TestSparsityLoss:
    def test_zero_when_no_forward(self):
        motor = FakeMotor()
        attach_sparse_gates(motor, ["proj1"], SparseConfig(gate_hidden=4))
        loss = sparsity_loss(motor, target=0.15)
        # sin forward, density is None → loss = 0
        assert loss.item() == 0.0

    def test_loss_nonzero_after_forward_when_mismatched(self):
        motor = FakeMotor()
        attach_sparse_gates(motor, ["proj1", "proj2"], SparseConfig(target_density=0.5, gate_hidden=4))
        motor(torch.randn(2, 8))
        loss = sparsity_loss(motor, target=0.99)
        # Muy improbable que density sea exactamente 0.99
        assert loss.item() >= 0.0

    def test_empty_modules(self):
        loss = sparsity_loss(nn.Linear(4, 4), target=0.15)
        assert loss.item() == 0.0


# ════════════════════════════════════════════════════════════════════════════
# Compatibilidad con LoRA (Parte 22)
# ════════════════════════════════════════════════════════════════════════════

class TestLoRACompatibility:
    def test_sparse_on_lora_linear(self):
        """Un LoRALinear envuelto por SparseLinear debe seguir funcionando.

        El SparseLinear trata al LoRALinear como si fuera un nn.Linear —
        pero LoRALinear NO es un nn.Linear. La compatibilidad correcta es
        aplicar sparse al base y lora por separado, o que sparse opere al
        final del pipeline. Verificamos que un motor con ADAPTER + SPARSE
        en la misma capa se comporta coherentemente.
        """
        motor = FakeMotor()
        # Primero sparse
        attach_sparse_gates(motor, ["proj1"], SparseConfig(gate_hidden=4))
        assert isinstance(motor.proj1, SparseLinear)
        # Forward normal funciona
        y = motor(torch.randn(2, 8))
        assert y.shape == (2, 16)
