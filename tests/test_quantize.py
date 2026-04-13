"""Tests for inference/quantize.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from inference.quantize import (
    quantize_tensor_int4,
    dequantize_tensor_int4,
    quantize_state_dict,
    dequantize_state_dict,
)


class TestTensorQuantization:
    def test_roundtrip_preserves_shape(self):
        t = torch.randn(64, 128)
        qdata = quantize_tensor_int4(t, group_size=32)
        restored = dequantize_tensor_int4(qdata)
        assert restored.shape == t.shape

    def test_small_tensor(self):
        t = torch.randn(4, 8)
        qdata = quantize_tensor_int4(t, group_size=8)
        restored = dequantize_tensor_int4(qdata)
        assert restored.shape == t.shape

    def test_quantization_error_is_bounded(self):
        t = torch.randn(64, 64)
        qdata = quantize_tensor_int4(t, group_size=32)
        restored = dequantize_tensor_int4(qdata)
        error = (t - restored).abs().mean().item()
        # INT4 has 16 levels, so error should be reasonable
        assert error < 1.0, f"Quantization error too large: {error}"

    def test_constant_tensor(self):
        t = torch.ones(32, 32)
        qdata = quantize_tensor_int4(t, group_size=32)
        restored = dequantize_tensor_int4(qdata)
        assert restored.shape == t.shape

    def test_packed_format(self):
        t = torch.randn(32, 64)
        qdata = quantize_tensor_int4(t, group_size=32)
        # Packed should be half the size of the quantized values
        n_elements = 32 * 64
        # Account for padding
        expected_packed_approx = n_elements // 2
        assert qdata["packed"].numel() <= expected_packed_approx + 100


class TestStateDictQuantization:
    def test_roundtrip(self):
        sd = {
            "weight1": torch.randn(64, 128),
            "weight2": torch.randn(32, 64),
            "bias": torch.randn(64),  # 1D, should stay float
        }
        quantized, stats = quantize_state_dict(sd, group_size=32)
        restored = dequantize_state_dict(quantized)

        assert set(restored.keys()) == set(sd.keys())
        for key in sd:
            assert restored[key].shape == sd[key].shape

    def test_1d_tensors_stay_float(self):
        sd = {"bias": torch.randn(64)}
        quantized, stats = quantize_state_dict(sd, group_size=32)
        # bias should be kept as-is (1D)
        assert stats["kept_float"] == 64

    def test_stats(self):
        sd = {
            "w": torch.randn(64, 128),
            "b": torch.randn(64),
        }
        _, stats = quantize_state_dict(sd, group_size=32)
        assert stats["total_params"] == 64 * 128 + 64
        assert stats["quantized"] == 64 * 128
        assert stats["kept_float"] == 64

    def test_with_real_model(self):
        model = torch.nn.Linear(32, 16)
        sd = model.state_dict()
        quantized, stats = quantize_state_dict(sd, group_size=32)
        restored = dequantize_state_dict(quantized)

        model2 = torch.nn.Linear(32, 16)
        model2.load_state_dict(restored)
        # Should not crash
        x = torch.randn(1, 32)
        out = model2(x)
        assert out.shape == (1, 16)
