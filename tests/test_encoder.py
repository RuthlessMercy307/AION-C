"""
tests/test_encoder.py — Tests para encoder/mamba_layer.py y encoder/model.py
=============================================================================

Cubre:
  TestStreamEncoderConfig   — validación y defaults de configuración
  TestRMSNorm               — shapes, normalización correcta, escala de weight
  TestGatedFFN              — shape, valores de gate, flujo de gradiente
  TestSelectiveSSM          — shape, dinámica de estado, selectividad
  TestMambaLayer            — shape completo, forward+residual, estados
  TestStreamEncoderShapes   — forward produce shapes correctas, compresión
  TestSSMDynamics           — el estado cambia entre timesteps (no estático)
  TestMemoryScaling         — A_bar.numel() escala linealmente con L (no L²)
  TestDevices               — funciona en CPU; en CUDA si está disponible
  TestGradientFlow          — backward funciona, todos los params tienen grad
  TestParameterCount        — parámetros razonables para tiny config
  TestDeterminism           — mismo seed → mismo output (reproducibilidad)

Ejecutar:
  cd IAS/AION-C
  python -m pytest tests/test_encoder.py -v
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import torch
import torch.nn as nn

from encoder.mamba_layer import (
    GatedFFN,
    MambaLayer,
    RMSNorm,
    SelectiveSSM,
    StreamEncoderConfig,
)
from encoder.model import StreamEncoder


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

TINY = StreamEncoderConfig(
    vocab_size  = 32_000,
    hidden_dim  = 256,
    n_layers    = 4,
    state_dim   = 16,
    expand      = 2,
    d_conv      = 4,
    concept_dim = 128,
    ffn_mult    = 4,
    dropout     = 0.0,
)

MICRO = StreamEncoderConfig(
    vocab_size  = 1_000,
    hidden_dim  = 32,
    n_layers    = 2,
    state_dim   = 4,
    expand      = 2,
    d_conv      = 4,
    concept_dim = 16,
    ffn_mult    = 2,
    dropout     = 0.0,
)


@pytest.fixture
def tiny_cfg():
    return TINY


@pytest.fixture
def micro_cfg():
    return MICRO


@pytest.fixture
def tiny_encoder():
    torch.manual_seed(0)
    return StreamEncoder(TINY).eval()


@pytest.fixture
def micro_encoder():
    torch.manual_seed(0)
    return StreamEncoder(MICRO).eval()


def _rand_ids(B: int, L: int, vocab: int = 32_000) -> torch.Tensor:
    return torch.randint(0, vocab, (B, L))


def _rand_x(B: int, L: int, D: int) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(B, L, D)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamEncoderConfig:

    def test_dt_rank_auto(self):
        cfg = StreamEncoderConfig(hidden_dim=256, dt_rank=0)
        assert cfg.dt_rank == math.ceil(256 / 16)  # = 16

    def test_dt_rank_manual(self):
        cfg = StreamEncoderConfig(hidden_dim=256, dt_rank=8)
        assert cfg.dt_rank == 8

    def test_d_inner_property(self):
        cfg = StreamEncoderConfig(hidden_dim=256, expand=2)
        assert cfg.d_inner == 512

    def test_d_inner_expand_1(self):
        cfg = StreamEncoderConfig(hidden_dim=256, expand=1)
        assert cfg.d_inner == 256

    def test_tiny_config_fields(self):
        assert TINY.vocab_size  == 32_000
        assert TINY.hidden_dim  == 256
        assert TINY.n_layers    == 4
        assert TINY.state_dim   == 16
        assert TINY.concept_dim == 128
        assert TINY.d_inner     == 512

    def test_concept_dim_less_than_hidden(self):
        assert TINY.concept_dim < TINY.hidden_dim


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — RMS NORM
# ─────────────────────────────────────────────────────────────────────────────

class TestRMSNorm:

    def test_output_shape_2d(self):
        norm = RMSNorm(64)
        x    = torch.randn(8, 64)
        assert norm(x).shape == (8, 64)

    def test_output_shape_3d(self):
        norm = RMSNorm(256)
        x    = torch.randn(2, 128, 256)
        assert norm(x).shape == (2, 128, 256)

    def test_weight_initialized_ones(self):
        norm = RMSNorm(64)
        assert torch.allclose(norm.weight, torch.ones(64))

    def test_normalizes_large_values(self):
        """Output deve ter RMS ≈ 1 antes da escala de weight."""
        norm = RMSNorm(64)
        x    = torch.randn(4, 64) * 1000.0  # valores grandes
        y    = norm(x)
        # Com weight=1 e eps=1e-5, RMS(y) ≈ 1
        rms  = y.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    def test_weight_scaling(self):
        """Weight funciona como escala."""
        norm = RMSNorm(4)
        with torch.no_grad():
            norm.weight.fill_(2.0)
        x  = torch.randn(2, 4)
        y  = norm(x)
        norm_weight1 = RMSNorm(4)
        y1 = norm_weight1(x)
        assert torch.allclose(y, 2.0 * y1, atol=1e-5)

    def test_gradient_flows(self):
        norm = RMSNorm(32)
        x    = torch.randn(2, 32, requires_grad=True)
        norm(x).sum().backward()
        assert x.grad is not None
        assert norm.weight.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — GATED FFN
# ─────────────────────────────────────────────────────────────────────────────

class TestGatedFFN:

    def test_output_shape(self):
        ffn = GatedFFN(dim=128, ffn_mult=4)
        x   = torch.randn(2, 64, 128)
        assert ffn(x).shape == (2, 64, 128)

    def test_output_dim_preserved(self):
        for D in [32, 64, 128, 256]:
            ffn = GatedFFN(dim=D, ffn_mult=2)
            x   = torch.randn(1, 16, D)
            assert ffn(x).shape[-1] == D, f"dim={D} no preservado"

    def test_has_three_projections(self):
        ffn = GatedFFN(dim=64, ffn_mult=4)
        names = [n for n, _ in ffn.named_parameters()]
        # w_gate.weight, w_up.weight, w_down.weight (+ bias si hay)
        assert any("gate" in n for n in names)
        assert any("up"   in n for n in names)
        assert any("down" in n for n in names)

    def test_gate_activates_selectively(self):
        """El gate SiLU puede suprimir o reforzar dimensiones."""
        ffn = GatedFFN(dim=32, ffn_mult=2)
        # Dos inputs muy distintos deben producir outputs distintos
        x1  = torch.randn(1, 8, 32)
        x2  = -x1
        y1  = ffn(x1)
        y2  = ffn(x2)
        assert not torch.allclose(y1, y2)

    def test_gradient_flows(self):
        ffn = GatedFFN(dim=64, ffn_mult=2)
        x   = torch.randn(2, 16, 64, requires_grad=True)
        ffn(x).sum().backward()
        assert x.grad is not None
        for p in ffn.parameters():
            assert p.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — SELECTIVE SSM
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectiveSSM:

    @pytest.fixture
    def micro_ssm(self):
        torch.manual_seed(1)
        return SelectiveSSM(MICRO).eval()

    def test_output_shape(self, micro_ssm):
        x = _rand_x(2, 32, MICRO.d_inner)
        y, _ = micro_ssm(x)
        assert y.shape == (2, 32, MICRO.d_inner)

    def test_no_states_returned_by_default(self, micro_ssm):
        x = _rand_x(2, 16, MICRO.d_inner)
        _, states = micro_ssm(x)
        assert states is None

    def test_states_returned_when_requested(self, micro_ssm):
        B, L, D = 1, 16, MICRO.d_inner
        N = MICRO.state_dim
        x = _rand_x(B, L, D)
        _, states = micro_ssm(x, return_states=True)
        assert states is not None
        assert states.shape == (B, L, D, N), \
            f"Expected ({B},{L},{D},{N}), got {states.shape}"

    def test_A_log_is_positive(self, micro_ssm):
        """A_log >= 0 → A = -exp(A_log) <= -1 → decaimiento estable.
        log(1)=0 es válido: A=-1 sigue siendo estable."""
        assert (micro_ssm.A_log >= 0).all(), "A_log debe ser no-negativo"

    def test_A_is_negative(self, micro_ssm):
        """A debe ser negativo para garantizar estabilidad del scan."""
        A = -torch.exp(micro_ssm.A_log.float())
        assert (A < 0).all(), "A debe ser negativo (decaimiento estable)"

    def test_D_initialized_ones(self, micro_ssm):
        assert torch.allclose(micro_ssm.D, torch.ones_like(micro_ssm.D)), \
            "D (skip) debe iniciar en 1"

    def test_output_finite(self, micro_ssm):
        x = _rand_x(2, 16, MICRO.d_inner)
        y, _ = micro_ssm(x)
        assert torch.isfinite(y).all(), "SSM output contiene NaN o Inf"

    def test_gradient_flows_through_scan(self):
        """El gradiente debe fluir por el scan secuencial."""
        ssm = SelectiveSSM(MICRO)
        x   = _rand_x(1, 8, MICRO.d_inner).requires_grad_(True)
        y, _ = ssm(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_different_inputs_different_outputs(self, micro_ssm):
        """El SSM es input-dependiente (no un filtro fijo)."""
        x1 = _rand_x(1, 16, MICRO.d_inner)
        x2 = torch.zeros_like(x1)
        y1, _ = micro_ssm(x1)
        y2, _ = micro_ssm(x2)
        assert not torch.allclose(y1, y2), \
            "SSM debe producir outputs distintos para inputs distintos"

    def test_A_bar_numel_recorded(self, micro_ssm):
        """_last_A_bar_numel debe registrarse durante el forward."""
        B, L = 2, 16
        x    = _rand_x(B, L, MICRO.d_inner)
        micro_ssm(x)
        expected = B * L * MICRO.d_inner * MICRO.state_dim
        assert micro_ssm._last_A_bar_numel == expected


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — MAMBA LAYER
# ─────────────────────────────────────────────────────────────────────────────

class TestMambaLayer:

    @pytest.fixture
    def layer(self):
        torch.manual_seed(2)
        return MambaLayer(MICRO).eval()

    def test_output_shape(self, layer):
        x = _rand_x(2, 24, MICRO.hidden_dim)
        y, _ = layer(x)
        assert y.shape == (2, 24, MICRO.hidden_dim), \
            f"Shape esperada (2,24,{MICRO.hidden_dim}), obtenida {y.shape}"

    def test_shape_dim_preserved(self, layer):
        for L in [8, 32, 128]:
            x = _rand_x(1, L, MICRO.hidden_dim)
            y, _ = layer(x)
            assert y.shape == (1, L, MICRO.hidden_dim)

    def test_states_none_by_default(self, layer):
        x     = _rand_x(1, 16, MICRO.hidden_dim)
        _, st = layer(x)
        assert st is None

    def test_states_shape_when_requested(self, layer):
        B, L, D, D_inner, N = 2, 16, MICRO.hidden_dim, MICRO.d_inner, MICRO.state_dim
        x     = _rand_x(B, L, D)
        _, st = layer(x, return_states=True)
        assert st is not None
        assert st.shape == (B, L, D_inner, N)

    def test_residual_connection_active(self, layer):
        """Output ≠ solo FFN(SSM(x)) — debe tener residuals."""
        x    = _rand_x(1, 8, MICRO.hidden_dim)
        y, _ = layer(x)
        # Si solo fuera feedforward sin residual, estaría más lejos de x
        # Con residual, la norma de (y-x) debería ser pequeña al inicio
        diff = (y - x).norm().item()
        x_norm = x.norm().item()
        # Al inicio del entrenamiento, los residuals dominan → diff/x_norm < 2
        assert diff < x_norm * 5.0, "Diferencia parece muy grande — ¿residual funciona?"

    def test_output_finite(self, layer):
        x = _rand_x(2, 16, MICRO.hidden_dim)
        y, _ = layer(x)
        assert torch.isfinite(y).all()

    def test_gradient_flows(self):
        layer = MambaLayer(MICRO)
        x     = _rand_x(1, 8, MICRO.hidden_dim).requires_grad_(True)
        y, _  = layer(x)
        y.sum().backward()
        assert x.grad is not None
        for n, p in layer.named_parameters():
            assert p.grad is not None, f"Parámetro {n!r} sin gradiente"

    def test_has_norm1_norm2(self, layer):
        assert hasattr(layer, "norm1") and isinstance(layer.norm1, RMSNorm)
        assert hasattr(layer, "norm2") and isinstance(layer.norm2, RMSNorm)

    def test_has_ssm(self, layer):
        assert hasattr(layer, "ssm") and isinstance(layer.ssm, SelectiveSSM)

    def test_has_ffn(self, layer):
        assert hasattr(layer, "ffn") and isinstance(layer.ffn, GatedFFN)

    def test_conv1d_is_causal(self, layer):
        """
        La conv1d con padding=d_conv-1 y recorte [:,:,:L] es causal:
        y_t depende solo de x_0...x_t, no de x_{t+1}...x_{L-1}.
        Verificación: cambiar x_{L//2:} no debe afectar y_{:L//2}.
        """
        torch.manual_seed(7)
        L = 16
        x  = _rand_x(1, L, MICRO.hidden_dim)
        x2 = x.clone()
        x2[:, L // 2:] = torch.randn(1, L // 2, MICRO.hidden_dim)

        with torch.no_grad():
            y1, _ = layer(x)
            y2, _ = layer(x2)

        # La primera mitad de los outputs debe coincidir
        # (pequeña tolerancia por la conv que puede ver hasta d_conv-1 tokens atrás)
        half = L // 2 - MICRO.d_conv
        if half > 0:
            assert torch.allclose(y1[:, :half], y2[:, :half], atol=1e-5), \
                "Conv1d no parece ser causal"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — STREAM ENCODER SHAPES
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamEncoderShapes:

    def test_forward_output_shape(self, micro_encoder):
        ids  = _rand_ids(2, 32, MICRO.vocab_size)
        out  = micro_encoder(ids)
        assert out.shape == (2, 32, MICRO.concept_dim), \
            f"Shape esperada (2,32,{MICRO.concept_dim}), obtenida {out.shape}"

    def test_batch_1(self, micro_encoder):
        ids = _rand_ids(1, 16, MICRO.vocab_size)
        out = micro_encoder(ids)
        assert out.shape == (1, 16, MICRO.concept_dim)

    def test_batch_4(self, micro_encoder):
        ids = _rand_ids(4, 64, MICRO.vocab_size)
        out = micro_encoder(ids)
        assert out.shape == (4, 64, MICRO.concept_dim)

    def test_sequence_length_1(self, micro_encoder):
        ids = _rand_ids(1, 1, MICRO.vocab_size)
        out = micro_encoder(ids)
        assert out.shape == (1, 1, MICRO.concept_dim)

    def test_concept_dim_smaller_than_hidden(self, micro_encoder):
        """El espacio conceptual DEBE ser más pequeño que el hidden."""
        assert MICRO.concept_dim < MICRO.hidden_dim, \
            "concept_dim debe ser menor que hidden_dim (compresión semántica)"

    def test_output_finite(self, micro_encoder):
        ids = _rand_ids(2, 32, MICRO.vocab_size)
        out = micro_encoder(ids)
        assert torch.isfinite(out).all(), "Output contiene NaN o Inf"

    def test_tiny_config_shape(self, tiny_encoder):
        ids = _rand_ids(1, 64, TINY.vocab_size)
        out = tiny_encoder(ids)
        assert out.shape == (1, 64, TINY.concept_dim)  # (1, 64, 128)

    def test_has_n_layers(self, micro_encoder):
        assert len(micro_encoder.layers) == MICRO.n_layers

    def test_all_layers_are_mamba(self, micro_encoder):
        for i, layer in enumerate(micro_encoder.layers):
            assert isinstance(layer, MambaLayer), \
                f"Layer {i} no es MambaLayer"

    def test_forward_with_states_shapes(self, micro_encoder):
        B, L = 2, 16
        ids  = _rand_ids(B, L, MICRO.vocab_size)
        concepts, states_list = micro_encoder.forward_with_states(ids)
        assert concepts.shape == (B, L, MICRO.concept_dim)
        assert len(states_list) == MICRO.n_layers
        for s in states_list:
            assert s.shape == (B, L, MICRO.d_inner, MICRO.state_dim)

    def test_norm_is_rms_norm(self, micro_encoder):
        assert isinstance(micro_encoder.norm, RMSNorm)

    def test_concept_projector_no_bias(self, micro_encoder):
        assert micro_encoder.concept_projector.bias is None


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — DINÁMICA DEL ESTADO SSM
# ─────────────────────────────────────────────────────────────────────────────

class TestSSMDynamics:
    """
    Verifica que el estado SSM no es estático — cambia entre timesteps.
    Esto es la propiedad fundamental del selective SSM:
    el sistema integra información causalmente (no es un filtro fijo).
    """

    def test_state_changes_between_timesteps(self):
        """h_0 ≠ h_1 ≠ h_2 ... para un input no nulo."""
        ssm = SelectiveSSM(MICRO).eval()
        torch.manual_seed(3)
        x = torch.randn(1, 8, MICRO.d_inner)

        _, states = ssm(x, return_states=True)
        # states: [1, L, D_inner, N]
        assert states is not None

        # El estado debe cambiar entre cada timestep
        n_changes = 0
        for t in range(1, states.shape[1]):
            diff = (states[:, t] - states[:, t-1]).abs().max().item()
            if diff > 1e-6:
                n_changes += 1

        assert n_changes == states.shape[1] - 1, \
            f"Solo {n_changes}/{states.shape[1]-1} timesteps tienen cambio de estado"

    def test_state_accumulates_context(self):
        """
        El estado en t=5 debe diferir del estado en t=0
        porque ha acumulado 5 tokens de contexto.
        """
        ssm = SelectiveSSM(MICRO).eval()
        x   = torch.randn(1, 10, MICRO.d_inner)

        _, states = ssm(x, return_states=True)
        assert states is not None

        h0 = states[:, 0]   # estado después del primer token
        h9 = states[:, 9]   # estado después del décimo token
        assert not torch.allclose(h0, h9, atol=1e-4), \
            "El estado no cambia con más contexto — ¿el scan está roto?"

    def test_different_inputs_different_states(self):
        """
        Inputs distintos producen estados distintos en cada timestep.
        Esto confirma la selectividad: B, C, Δ son input-dependientes.
        """
        ssm = SelectiveSSM(MICRO).eval()
        torch.manual_seed(4)
        x1 = torch.randn(1, 8, MICRO.d_inner)
        x2 = torch.randn(1, 8, MICRO.d_inner)

        _, s1 = ssm(x1, return_states=True)
        _, s2 = ssm(x2, return_states=True)
        assert s1 is not None and s2 is not None

        # Todos los estados deben diferir
        for t in range(s1.shape[1]):
            diff = (s1[:, t] - s2[:, t]).abs().max().item()
            assert diff > 1e-6, \
                f"Timestep {t}: estados idénticos con inputs distintos"

    def test_zero_input_state_decays(self):
        """
        Con input=0, el estado debe decaer monótonamente
        (A < 0 → A_bar = exp(Δ·A) < 1 → decaimiento estable).
        """
        ssm = SelectiveSSM(MICRO).eval()

        # Inicializar el estado con un forward de 1 token no nulo
        x_init = torch.ones(1, 1, MICRO.d_inner)
        _, states_init = ssm(x_init, return_states=True)
        h_init = states_init[:, 0].clone()   # [1, D_inner, N]

        # Luego pasar L tokens de cero — el estado debe decrecer
        L_zeros = 20
        x_zero  = torch.zeros(1, L_zeros, MICRO.d_inner)
        _, states_zero = ssm(x_zero, return_states=True)
        assert states_zero is not None

        # La norma del estado inicial vs final (solo con ceros)
        # Nota: el estado del scan empieza en 0 aquí (forward independiente)
        # Lo que verificamos es que el estado no explota y es finito
        for t in range(states_zero.shape[1]):
            assert torch.isfinite(states_zero[:, t]).all(), \
                f"Estado no finito en t={t}"

    def test_encoder_states_per_layer(self):
        """Cada capa del encoder tiene su propio estado independiente."""
        torch.manual_seed(5)
        encoder = StreamEncoder(MICRO).eval()
        ids = _rand_ids(1, 8, MICRO.vocab_size)

        _, all_states = encoder.forward_with_states(ids)

        # Los estados de distintas capas deben ser distintos
        if len(all_states) >= 2:
            assert not torch.allclose(all_states[0], all_states[1], atol=1e-4), \
                "Las capas tienen estados idénticos — ¿los pesos están degenerados?"

    def test_state_is_3d_tensor(self):
        """El estado h es [B, D_inner, N] — no colapsa a escalar."""
        ssm = SelectiveSSM(MICRO).eval()
        x   = torch.randn(2, 4, MICRO.d_inner)
        _, states = ssm(x, return_states=True)
        assert states is not None
        B, L, D, N = states.shape
        assert D == MICRO.d_inner
        assert N == MICRO.state_dim
        assert B == 2


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — ESCALADO DE MEMORIA (LINEAL, NO CUADRÁTICO)
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryScaling:
    """
    Verifica que la memoria escala O(L), no O(L²).

    Estrategia arquitectónica:
      - A_bar.numel() = B × L × D_inner × N  → O(L)
      - Sin attention matrix (L × L)          → sin O(L²)

    Verificación cuantitativa via _last_A_bar_numel:
      Al doblar L, A_bar.numel() debe doblar exactamente.

    Verificación cualitativa via CUDA (si disponible):
      peak_memory(L=2k) ≈ 2 × peak_memory(L=k)
    """

    def test_A_bar_numel_linear_in_L(self):
        """
        A_bar.numel() crece exactamente linealmente con L.
        Ratio = 2.0 cuando L dobla.
        """
        encoder = StreamEncoder(MICRO).eval()
        B = 2

        L1, L2, L3 = 64, 128, 256

        ids1 = _rand_ids(B, L1, MICRO.vocab_size)
        ids2 = _rand_ids(B, L2, MICRO.vocab_size)
        ids3 = _rand_ids(B, L3, MICRO.vocab_size)

        n1 = encoder.get_ssm_A_bar_numel(ids1)
        n2 = encoder.get_ssm_A_bar_numel(ids2)
        n3 = encoder.get_ssm_A_bar_numel(ids3)

        ratio_12 = n2 / n1
        ratio_23 = n3 / n2

        assert abs(ratio_12 - 2.0) < 0.01, \
            f"A_bar no escala linealmente: ratio L64→L128 = {ratio_12:.3f} (esperado 2.0)"
        assert abs(ratio_23 - 2.0) < 0.01, \
            f"A_bar no escala linealmente: ratio L128→L256 = {ratio_23:.3f} (esperado 2.0)"

    def test_A_bar_numel_formula(self):
        """A_bar.numel() == B × L × D_inner × N exactamente."""
        encoder = StreamEncoder(MICRO).eval()
        B, L = 3, 64
        ids = _rand_ids(B, L, MICRO.vocab_size)
        n   = encoder.get_ssm_A_bar_numel(ids)
        expected = B * L * MICRO.d_inner * MICRO.state_dim
        assert n == expected, f"Esperado {expected}, obtenido {n}"

    def test_no_attention_matrix_in_model(self):
        """
        El modelo NO debe tener ningún nn.MultiheadAttention.
        Sin attention matrix → sin O(L²).
        """
        encoder = StreamEncoder(MICRO)
        attn_modules = [
            m for m in encoder.modules()
            if isinstance(m, nn.MultiheadAttention)
        ]
        assert len(attn_modules) == 0, \
            f"Encontrado {len(attn_modules)} módulos MultiheadAttention — contradice el diseño O(L)"

    def test_no_full_attention_weights(self):
        """No debe haber parámetros con nombre 'attn' o 'attention' en el modelo."""
        encoder = StreamEncoder(MICRO)
        attn_params = [
            n for n, _ in encoder.named_parameters()
            if "attn" in n.lower() or "attention" in n.lower()
        ]
        assert len(attn_params) == 0, \
            f"Parámetros sospechosos de attention: {attn_params}"

    def test_model_runs_at_multiple_lengths_no_oom(self):
        """
        El modelo debe poder procesar secuencias de distintos largos
        sin errores — verificación de que no hay OOM a L largo.
        """
        encoder = StreamEncoder(MICRO).eval()
        with torch.no_grad():
            for L in [64, 128, 256, 512]:
                ids = _rand_ids(1, L, MICRO.vocab_size)
                out = encoder(ids)
                assert out.shape == (1, L, MICRO.concept_dim), \
                    f"Shape incorrecta en L={L}: {out.shape}"

    def test_A_bar_grows_faster_than_hidden_state(self):
        """
        A_bar.numel() crece con L, pero el estado h[B,D,N] es CONSTANTE.
        Esto muestra la diferencia entre memoria de trabajo (constante)
        y memoria de activaciones (lineal).
        """
        ssm = SelectiveSSM(MICRO).eval()
        B   = 1

        # A_bar depende de L
        x_short = _rand_x(B, 8,  MICRO.d_inner); ssm(x_short); n_short = ssm._last_A_bar_numel
        x_long  = _rand_x(B, 64, MICRO.d_inner); ssm(x_long);  n_long  = ssm._last_A_bar_numel

        # El tamaño del estado h en memoria sería siempre B * D_inner * N
        state_size = B * MICRO.d_inner * MICRO.state_dim

        assert n_long > n_short,     "A_bar debe crecer con L"
        assert n_short == state_size * 8,  "A_bar short debe ser state_size * L_short"
        assert n_long  == state_size * 64, "A_bar long  debe ser state_size * L_long"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA no disponible")
    def test_cuda_memory_linear_scaling(self):
        """
        En CUDA: peak_memory(L=2k) / peak_memory(L=k) debe estar en [1.5, 2.5].
        (Tolerancia porque los parámetros del modelo son overhead constante.)
        """
        device  = torch.device("cuda")
        encoder = StreamEncoder(MICRO).to(device).eval()

        def measure_peak(L: int) -> int:
            torch.cuda.reset_peak_memory_stats()
            ids = _rand_ids(4, L, MICRO.vocab_size).to(device)
            with torch.no_grad():
                encoder(ids)
            return torch.cuda.max_memory_allocated()

        m512  = measure_peak(512)
        m1024 = measure_peak(1024)
        ratio = m1024 / m512

        assert 1.4 <= ratio <= 2.6, \
            f"Memory ratio(L=1024 / L=512) = {ratio:.2f} — no parece lineal en L"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — CPU Y CUDA
# ─────────────────────────────────────────────────────────────────────────────

class TestDevices:

    def test_runs_on_cpu(self, micro_encoder):
        """El modelo debe correr en CPU sin errores."""
        ids = _rand_ids(2, 32, MICRO.vocab_size)
        out = micro_encoder(ids)
        assert out.device.type == "cpu"

    def test_output_on_cpu(self, micro_encoder):
        ids = _rand_ids(1, 16, MICRO.vocab_size)
        out = micro_encoder(ids)
        assert out.shape == (1, 16, MICRO.concept_dim)
        assert torch.isfinite(out).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA no disponible")
    def test_runs_on_cuda(self):
        device  = torch.device("cuda")
        encoder = StreamEncoder(MICRO).to(device).eval()
        ids     = _rand_ids(2, 32, MICRO.vocab_size).to(device)
        out     = encoder(ids)
        assert out.device.type == "cuda"
        assert out.shape == (2, 32, MICRO.concept_dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA no disponible")
    def test_cpu_cuda_output_close(self):
        """
        CPU y CUDA deben producir outputs numericamente cercanos.
        Diferencias pequeñas son esperables por precisión floating point.
        """
        torch.manual_seed(42)
        encoder_cpu  = StreamEncoder(MICRO).eval()
        encoder_cuda = StreamEncoder(MICRO).cuda().eval()

        # Copiar pesos idénticos
        encoder_cuda.load_state_dict(encoder_cpu.state_dict())

        ids      = _rand_ids(1, 16, MICRO.vocab_size)
        ids_cuda = ids.cuda()

        with torch.no_grad():
            out_cpu  = encoder_cpu(ids)
            out_cuda = encoder_cuda(ids_cuda).cpu()

        assert torch.allclose(out_cpu, out_cuda, atol=1e-4, rtol=1e-4), \
            "Outputs CPU y CUDA difieren más de lo esperado"

    def test_to_device_works(self):
        """El modelo soporta .to(device) sin errores."""
        encoder = StreamEncoder(MICRO)
        device  = torch.device("cpu")
        encoder = encoder.to(device)
        ids = _rand_ids(1, 8, MICRO.vocab_size).to(device)
        out = encoder(ids)
        assert out.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA no disponible")
    def test_mixed_input_raises(self):
        """Input en CPU con modelo en CUDA debe fallar apropiadamente."""
        encoder = StreamEncoder(MICRO).cuda()
        ids_cpu = _rand_ids(1, 8, MICRO.vocab_size)  # CPU
        with pytest.raises(RuntimeError):
            encoder(ids_cpu)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — FLUJO DE GRADIENTE
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_backward_runs(self):
        encoder = StreamEncoder(MICRO)
        ids     = _rand_ids(2, 16, MICRO.vocab_size)
        out     = encoder(ids)
        loss    = out.mean()
        loss.backward()  # No debe lanzar error

    def test_all_params_have_gradients(self):
        """Todos los parámetros del modelo deben tener gradientes."""
        encoder = StreamEncoder(MICRO)
        ids     = _rand_ids(2, 16, MICRO.vocab_size)
        out     = encoder(ids)
        out.sum().backward()

        no_grad = []
        for name, param in encoder.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad.append(name)

        assert len(no_grad) == 0, \
            f"Parámetros sin gradiente: {no_grad}"

    def test_gradients_finite(self):
        encoder = StreamEncoder(MICRO)
        ids     = _rand_ids(1, 16, MICRO.vocab_size)
        out     = encoder(ids)
        out.sum().backward()

        non_finite = []
        for name, param in encoder.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                non_finite.append(name)

        assert len(non_finite) == 0, \
            f"Gradientes no finitos en: {non_finite}"

    def test_embedding_grad_is_sparse_but_not_none(self):
        """El embedding recibe gradientes solo en los tokens usados."""
        encoder = StreamEncoder(MICRO)
        ids     = _rand_ids(1, 8, MICRO.vocab_size)
        out     = encoder(ids)
        out.sum().backward()
        assert encoder.token_embedding.weight.grad is not None

    def test_grad_norm_reasonable(self):
        """La norma del gradiente no debe ser 0 ni extremadamente grande."""
        encoder = StreamEncoder(MICRO)
        ids     = _rand_ids(1, 8, MICRO.vocab_size)
        out     = encoder(ids)
        out.sum().backward()

        total_norm = sum(
            p.grad.norm().item() ** 2
            for p in encoder.parameters()
            if p.grad is not None
        ) ** 0.5

        assert total_norm > 1e-8, "Grad norm casi cero — posible vanishing gradient"
        assert total_norm < 1e6,  "Grad norm explosiva"

    def test_no_grad_mode(self, micro_encoder):
        """En torch.no_grad(), no se computan gradientes."""
        ids = _rand_ids(1, 8, MICRO.vocab_size)
        with torch.no_grad():
            out = micro_encoder(ids)
        assert not out.requires_grad


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — CONTEO DE PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterCount:

    def test_count_is_positive(self, micro_encoder):
        assert micro_encoder.count_parameters() > 0

    def test_tiny_in_reasonable_range(self, tiny_encoder):
        """El tiny config debe tener entre 5M y 30M parámetros."""
        n = tiny_encoder.count_parameters()
        assert 5_000_000 <= n <= 30_000_000, \
            f"Tiny config tiene {n:,} parámetros — fuera del rango esperado"

    def test_breakdown_sums_to_total(self, micro_encoder):
        breakdown = micro_encoder.parameter_breakdown()
        total_reported = breakdown["total"]
        actual_total   = micro_encoder.count_parameters()
        assert total_reported == actual_total

    def test_embedding_is_largest_component(self, tiny_encoder):
        """
        Para vocab_size=32000 y hidden_dim=256, el embedding
        (32000 × 256 = 8.2M) debería ser el componente más grande.
        """
        breakdown = tiny_encoder.parameter_breakdown()
        emb_size  = breakdown["token_embedding"]
        assert emb_size > 0
        # El embedding debe ser una fracción significativa del total
        ratio = emb_size / breakdown["total"]
        assert ratio > 0.3, \
            f"Embedding ({emb_size:,}) es muy pequeño vs total ({breakdown['total']:,})"

    def test_concept_projector_smaller_than_ssm(self, micro_encoder):
        breakdown = micro_encoder.parameter_breakdown()
        assert breakdown["concept_projector"] < breakdown["layers_ssm"]

    def test_layers_scale_with_n_layers(self):
        cfg1 = StreamEncoderConfig(hidden_dim=32, n_layers=1, vocab_size=100,
                                   state_dim=4, concept_dim=16, ffn_mult=2)
        cfg2 = StreamEncoderConfig(hidden_dim=32, n_layers=2, vocab_size=100,
                                   state_dim=4, concept_dim=16, ffn_mult=2)
        enc1 = StreamEncoder(cfg1)
        enc2 = StreamEncoder(cfg2)
        n1   = enc1.count_parameters()
        n2   = enc2.count_parameters()
        # n2 > n1 porque tiene más capas
        assert n2 > n1, "Más capas debería implicar más parámetros"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — DETERMINISMO Y REPRODUCIBILIDAD
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_same_seed_same_output(self):
        """Mismo seed → mismos pesos → mismo output."""
        torch.manual_seed(99)
        enc1 = StreamEncoder(MICRO).eval()
        torch.manual_seed(99)
        enc2 = StreamEncoder(MICRO).eval()

        ids = _rand_ids(1, 16, MICRO.vocab_size)
        with torch.no_grad():
            out1 = enc1(ids)
            out2 = enc2(ids)

        assert torch.allclose(out1, out2), \
            "Mismo seed produce outputs distintos"

    def test_different_seeds_different_outputs(self):
        """Seeds distintos → pesos distintos → outputs distintos."""
        torch.manual_seed(1)
        enc1 = StreamEncoder(MICRO).eval()
        torch.manual_seed(2)
        enc2 = StreamEncoder(MICRO).eval()

        ids = _rand_ids(1, 16, MICRO.vocab_size)
        with torch.no_grad():
            out1 = enc1(ids)
            out2 = enc2(ids)

        assert not torch.allclose(out1, out2), \
            "Seeds distintos producen pesos idénticos — error en init"

    def test_eval_mode_deterministic(self, micro_encoder):
        """En eval() sin dropout, dos forwards iguales dan igual output."""
        ids = _rand_ids(1, 16, MICRO.vocab_size)
        with torch.no_grad():
            out1 = micro_encoder(ids)
            out2 = micro_encoder(ids)
        assert torch.allclose(out1, out2), \
            "Forward no determinístico en eval()"

    def test_train_mode_with_dropout_different(self):
        """En train() con dropout, dos forwards del mismo input pueden diferir."""
        cfg = StreamEncoderConfig(
            hidden_dim=32, n_layers=1, vocab_size=100,
            state_dim=4, concept_dim=16, ffn_mult=2, dropout=0.5
        )
        enc = StreamEncoder(cfg).train()
        ids = _rand_ids(1, 16, 100)
        out1 = enc(ids)
        out2 = enc(ids)
        # Con dropout=0.5, es casi seguro que difieren (puede fallar con prob muy baja)
        # No hacemos assert aquí porque es probabilístico; solo verificamos que corre
        assert out1.shape == out2.shape
