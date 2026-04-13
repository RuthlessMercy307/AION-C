"""Tests for Phase 4 (LoRA) and interactive eval."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn


class TestLoRALinear:
    def test_forward_shape(self):
        from experiments.train_production import _LoRALinear
        original = nn.Linear(64, 32)
        lora = _LoRALinear(original, rank=4, alpha=4.0)
        x = torch.randn(2, 64)
        out = lora(x)
        assert out.shape == (2, 32)

    def test_original_frozen(self):
        from experiments.train_production import _LoRALinear
        original = nn.Linear(64, 32)
        lora = _LoRALinear(original, rank=4)
        assert not lora.original.weight.requires_grad
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_weight_property(self):
        from experiments.train_production import _LoRALinear
        original = nn.Linear(64, 32)
        lora = _LoRALinear(original, rank=4)
        # .weight and .bias must be accessible (PyTorch MultiheadAttention needs this)
        assert lora.weight is original.weight
        assert lora.bias is original.bias
        assert lora.weight.shape == (32, 64)
        assert lora.in_features == 64
        assert lora.out_features == 32

    def test_weight_property_no_bias(self):
        from experiments.train_production import _LoRALinear
        original = nn.Linear(64, 32, bias=False)
        lora = _LoRALinear(original, rank=4)
        assert lora.weight is original.weight
        assert lora.bias is None

    def test_lora_adds_to_original(self):
        from experiments.train_production import _LoRALinear
        original = nn.Linear(64, 32, bias=False)
        lora = _LoRALinear(original, rank=4, alpha=4.0)
        x = torch.randn(1, 64)
        # When B is zero, output should equal original
        with torch.no_grad():
            lora.lora_B.zero_()
        base = original(x)
        lora_out = lora(x)
        assert torch.allclose(base, lora_out, atol=1e-6)

    def test_gradient_flows(self):
        from experiments.train_production import _LoRALinear
        original = nn.Linear(32, 16)
        lora = _LoRALinear(original, rank=4)
        x = torch.randn(1, 32)
        out = lora(x)
        out.sum().backward()
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert original.weight.grad is None  # frozen


class TestApplyLoRA:
    def test_applies_to_linears(self):
        from experiments.train_production import apply_lora, _LoRALinear
        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        count = apply_lora(model, rank=4)
        assert count == 2
        assert isinstance(model[0], _LoRALinear)
        assert isinstance(model[2], _LoRALinear)

    def test_nested_modules(self):
        from experiments.train_production import apply_lora
        model = nn.ModuleDict({
            "block": nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 8)),
        })
        count = apply_lora(model, rank=4)
        assert count == 2

    def test_forward_after_lora(self):
        from experiments.train_production import apply_lora
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        apply_lora(model, rank=4)
        x = torch.randn(2, 32)
        out = model(x)
        assert out.shape == (2, 8)
