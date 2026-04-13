"""
tests/test_config_3_5b.py — Tests para router/config_3_5b.py
=============================================================

Cubre:
  1. Factories: tiny / medium / production retornan MoSEScaleConfig
  2. count_params() tiny == count_params_real() (ratio 1.000)
  3. count_params() production ≈ 3.5 B ±10%
  4. estimate_vram_bf16() production < 130 GB (inferencia y training)
  5. Jerarquía tiny < medium < production en parámetros
  6. Fórmulas individuales: mamba_layer, cre, crystallizer, encoder, decoder
  7. estimate_vram_bf16(training=True) > estimate_vram_bf16(training=False)
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from router.config_3_5b import (
    MoSEScaleConfig,
    _mamba_layer_params,
    _cre_params,
    _crystallizer_params,
    _encoder_params,
    _decoder_params,
    _orchestrator_params,
    _unifier_params,
    _hybrid_decoder_layer_params,
    _cre_message_layer_params,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Factory methods
# ─────────────────────────────────────────────────────────────────────────────

class TestFactories:

    def test_tiny_returns_instance(self):
        cfg = MoSEScaleConfig.tiny()
        assert isinstance(cfg, MoSEScaleConfig)

    def test_medium_returns_instance(self):
        cfg = MoSEScaleConfig.medium()
        assert isinstance(cfg, MoSEScaleConfig)

    def test_production_returns_instance(self):
        cfg = MoSEScaleConfig.production()
        assert isinstance(cfg, MoSEScaleConfig)

    def test_tiny_has_correct_vocab(self):
        assert MoSEScaleConfig.tiny().vocab_size == 512

    def test_production_has_correct_vocab(self):
        assert MoSEScaleConfig.production().vocab_size == 32_000

    def test_tiny_enc_dim(self):
        assert MoSEScaleConfig.tiny().enc_dim == 64

    def test_medium_enc_dim(self):
        assert MoSEScaleConfig.medium().enc_dim == 768

    def test_production_enc_dim(self):
        assert MoSEScaleConfig.production().enc_dim == 1024

    def test_production_dec_dim(self):
        assert MoSEScaleConfig.production().dec_dim == 1536

    def test_tiny_state_dim(self):
        cfg = MoSEScaleConfig.tiny()
        assert cfg.enc_state_dim == 4
        assert cfg.dec_state_dim == 4


# ─────────────────────────────────────────────────────────────────────────────
# 2. count_params_real vs count_params (tiny)
# ─────────────────────────────────────────────────────────────────────────────

class TestTinyRealVsFormula:

    def test_tiny_formula_equals_real(self):
        """Fórmula == real para tiny (ratio exactamente 1.000)."""
        cfg     = MoSEScaleConfig.tiny()
        formula = cfg.count_params()
        real    = cfg.count_params_real()
        assert formula == real, (
            f"Formula={formula:,} real={real:,} diff={abs(formula-real)}"
        )

    def test_tiny_real_is_positive(self):
        real = MoSEScaleConfig.tiny().count_params_real()
        assert real > 0

    def test_tiny_params_under_2m(self):
        """Tiny es suficientemente pequeño para tests rápidos."""
        assert MoSEScaleConfig.tiny().count_params() < 2_000_000


# ─────────────────────────────────────────────────────────────────────────────
# 3. production ≈ 3.5 B ±10%
# ─────────────────────────────────────────────────────────────────────────────

class TestProductionScale:

    _TARGET = 3.5e9
    _TOL    = 0.10   # ±10%

    def test_production_in_range(self):
        p = MoSEScaleConfig.production().count_params()
        lo = self._TARGET * (1 - self._TOL)
        hi = self._TARGET * (1 + self._TOL)
        assert lo <= p <= hi, (
            f"Production params={p/1e9:.3f}B not in [{lo/1e9:.2f}B, {hi/1e9:.2f}B]"
        )

    def test_production_above_3b(self):
        """Mínimo absoluto de calidad."""
        p = MoSEScaleConfig.production().count_params()
        assert p > 3.0e9, f"Production params {p/1e9:.2f}B < 3.0B"

    def test_production_below_4b(self):
        """No excesivamente grande."""
        p = MoSEScaleConfig.production().count_params()
        assert p < 4.0e9, f"Production params {p/1e9:.2f}B > 4.0B"


# ─────────────────────────────────────────────────────────────────────────────
# 4. estimate_vram_bf16
# ─────────────────────────────────────────────────────────────────────────────

class TestVRAMEstimate:

    def test_inference_under_130gb(self):
        vram = MoSEScaleConfig.production().estimate_vram_bf16(training=False)
        assert vram < 130.0, f"VRAM inference={vram:.1f}GB >= 130GB"

    def test_training_under_130gb(self):
        vram = MoSEScaleConfig.production().estimate_vram_bf16(training=True)
        assert vram < 130.0, f"VRAM training={vram:.1f}GB >= 130GB"

    def test_training_gt_inference(self):
        cfg  = MoSEScaleConfig.production()
        assert cfg.estimate_vram_bf16(training=True) > cfg.estimate_vram_bf16(training=False)

    def test_inference_positive(self):
        assert MoSEScaleConfig.production().estimate_vram_bf16() > 0

    def test_tiny_inference_very_small(self):
        """Tiny debería necesitar < 1 GB."""
        vram = MoSEScaleConfig.tiny().estimate_vram_bf16()
        assert vram < 1.0, f"Tiny VRAM={vram:.3f}GB"

    def test_medium_inference_reasonable(self):
        """Medium ~410M params × 2 bytes × 2 overhead ≈ 1.6 GB."""
        vram = MoSEScaleConfig.medium().estimate_vram_bf16()
        assert 0.5 < vram < 10.0, f"Medium VRAM={vram:.2f}GB out of range"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Jerarquía tiny < medium < production
# ─────────────────────────────────────────────────────────────────────────────

class TestParamHierarchy:

    def test_tiny_lt_medium(self):
        t = MoSEScaleConfig.tiny().count_params()
        m = MoSEScaleConfig.medium().count_params()
        assert t < m, f"tiny={t:,} not < medium={m:,}"

    def test_medium_lt_production(self):
        m = MoSEScaleConfig.medium().count_params()
        p = MoSEScaleConfig.production().count_params()
        assert m < p, f"medium={m:,} not < production={p:,}"

    def test_tiny_lt_production(self):
        t = MoSEScaleConfig.tiny().count_params()
        p = MoSEScaleConfig.production().count_params()
        assert t < p


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fórmulas individuales de parámetros
# ─────────────────────────────────────────────────────────────────────────────

class TestMambaLayerFormula:

    def test_tiny_mamba_layer(self):
        """D=64, state_dim=4, ffn_mult=2 → 52736 (verificado contra real)."""
        params = _mamba_layer_params(D=64, state_dim=4, expand=2, d_conv=4, ffn_mult=2)
        assert params == 52_736

    def test_medium_mamba_layer_positive(self):
        params = _mamba_layer_params(D=768, state_dim=16, expand=2, d_conv=4, ffn_mult=4)
        assert params > 0

    def test_production_enc_mamba_layer(self):
        """D=1024, state_dim=16, ffn_mult=4 → 19251200."""
        params = _mamba_layer_params(D=1024, state_dim=16, expand=2, d_conv=4, ffn_mult=4)
        assert params == 19_251_200

    def test_scales_superlinearly_with_d(self):
        """Al doblar D, los params crecen >2x (GatedFFN domina con D²)."""
        p1 = _mamba_layer_params(D=64,  state_dim=16)
        p2 = _mamba_layer_params(D=128, state_dim=16)
        assert p2 > 2 * p1

    def test_ffn_mult_increases_params(self):
        p2 = _mamba_layer_params(D=128, state_dim=16, ffn_mult=2)
        p4 = _mamba_layer_params(D=128, state_dim=16, ffn_mult=4)
        assert p4 > p2


class TestCREFormula:

    def test_cre_tiny_cora_no_gate(self):
        """CRE(D=64, R=16, n_msg=1, gate=False): solo layers + emb + proj."""
        p = _cre_params(D=64, n_relations=16, n_message_layers=1, use_convergence_gate=False)
        assert p > 0

    def test_cre_with_gate_gt_without(self):
        """Con ConvergenceGate (WeaknessDetector) hay más params."""
        p_no  = _cre_params(D=256, n_relations=16, use_convergence_gate=False)
        p_yes = _cre_params(D=256, n_relations=16, use_convergence_gate=True)
        assert p_yes > p_no

    def test_cre_more_msg_layers_more_params(self):
        p1 = _cre_params(D=128, n_relations=16, n_message_layers=1)
        p2 = _cre_params(D=128, n_relations=16, n_message_layers=2)
        assert p2 > p1

    def test_cre_tiny_cora_with_gate_known(self):
        """
        D=64, R=16, n_msg=1, gate=True → debe coincidir con motor real
        (motor_cora - crystallizer_tiny_cora).
        """
        from router.pipeline import MoSEConfig, MoSEPipeline
        cfg   = MoSEConfig.tiny()
        model = MoSEPipeline(cfg)

        def _count(m):
            seen = set()
            n = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    if p.requires_grad:
                        n += p.numel()
            return n

        cre_real = _count(model.motors["cora"].cre)
        cre_formula = _cre_params(D=64, n_relations=16, n_message_layers=1,
                                   use_convergence_gate=True)
        assert cre_formula == cre_real, (
            f"CRE formula={cre_formula} real={cre_real}"
        )


class TestCrystallizerFormula:

    def test_crystallizer_tiny_cora_known(self):
        """
        D=64, T=7, R=16 → debe coincidir con crystallizer del motor real.
        """
        from router.pipeline import MoSEConfig, MoSEPipeline
        cfg   = MoSEConfig.tiny()
        model = MoSEPipeline(cfg)

        def _count(m):
            seen = set()
            n = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    if p.requires_grad:
                        n += p.numel()
            return n

        cryst_real    = _count(model.motors["cora"].crystallizer)
        cryst_formula = _crystallizer_params(D=64, n_node_types=7, n_relation_types=16)
        assert cryst_formula == cryst_real, (
            f"Crystallizer formula={cryst_formula} real={cryst_real}"
        )

    def test_crystallizer_scales_with_D(self):
        p64  = _crystallizer_params(64,  7, 16)
        p128 = _crystallizer_params(128, 7, 16)
        assert p128 > p64

    def test_crystallizer_scales_with_relations(self):
        p10 = _crystallizer_params(128, 7, 10)
        p16 = _crystallizer_params(128, 7, 16)
        assert p16 > p10


class TestEncoderFormula:

    def test_encoder_tiny_known(self):
        """
        D=64, L=2, V=512, state_dim=4, ffn_mult=2 → 142400 (verificado).
        """
        p = _encoder_params(D=64, n_layers=2, vocab_size=512,
                            state_dim=4, expand=2, d_conv=4, ffn_mult=2)
        assert p == 142_400

    def test_encoder_more_layers_more_params(self):
        p2 = _encoder_params(D=128, n_layers=2, vocab_size=1000, state_dim=16)
        p4 = _encoder_params(D=128, n_layers=4, vocab_size=1000, state_dim=16)
        assert p4 > p2


class TestDecoderFormula:

    def test_decoder_tiny_known(self):
        """
        D=64, L=2, V=512, state_dim=4, ffn_mult=2, n_heads=4,
        max_seq_len=128, max_graph_nodes=8 → 270922 (verificado).
        """
        p = _decoder_params(
            D=64, n_layers=2, vocab_size=512, state_dim=4,
            expand=2, d_conv=4, ffn_mult=2, n_heads=4,
            max_seq_len=128, max_graph_nodes=8, node_dim=64,
        )
        assert p == 270_922

    def test_decoder_more_layers_more_params(self):
        p2 = _decoder_params(D=128, n_layers=2, vocab_size=1000, state_dim=16,
                              n_heads=4, max_seq_len=128, max_graph_nodes=8)
        p4 = _decoder_params(D=128, n_layers=4, vocab_size=1000, state_dim=16,
                              n_heads=4, max_seq_len=128, max_graph_nodes=8)
        assert p4 > p2

    def test_decoder_node_dim_mismatch_adds_graph_proj(self):
        """Si node_dim != D, hay un graph_proj extra."""
        p_same = _decoder_params(D=256, n_layers=2, vocab_size=1000, state_dim=16,
                                  n_heads=4, node_dim=256)
        p_diff = _decoder_params(D=256, n_layers=2, vocab_size=1000, state_dim=16,
                                  n_heads=4, node_dim=128)
        assert p_diff > p_same


class TestOrchestratorFormula:

    def test_tiny_known(self):
        """D=64, H=32, N=5 → 2757 (verificado)."""
        p = _orchestrator_params(D=64, mlp_hidden=32, n_motors=5)
        assert p == 2_757

    def test_larger_hidden_more_params(self):
        p1 = _orchestrator_params(D=256, mlp_hidden=128)
        p2 = _orchestrator_params(D=256, mlp_hidden=512)
        assert p2 > p1


class TestUnifierFormula:

    def test_tiny_known(self):
        """D=64, n_heads=4 → 33600 (verificado)."""
        p = _unifier_params(D=64, n_heads=4)
        assert p == 33_600

    def test_scales_superlinearly_with_D(self):
        """Al doblar D, los params crecen >2x (dominan términos D²)."""
        p1 = _unifier_params(D=64)
        p2 = _unifier_params(D=128)
        assert p2 > 2 * p1


# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary / metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestSummary:

    def test_summary_returns_string(self):
        s = MoSEScaleConfig.production().summary()
        assert isinstance(s, str)
        assert "3" in s   # debería mostrar ~3.xB o 3xxx.xM

    def test_summary_tiny(self):
        s = MoSEScaleConfig.tiny().summary()
        assert isinstance(s, str)
        assert len(s) > 10

    def test_count_params_deterministic(self):
        """Múltiples llamadas retornan el mismo valor."""
        cfg = MoSEScaleConfig.production()
        assert cfg.count_params() == cfg.count_params()

    def test_vram_proportional_to_params(self):
        """VRAM inference ≈ 2× pesos bf16."""
        cfg    = MoSEScaleConfig.production()
        params = cfg.count_params()
        vram_gb = cfg.estimate_vram_bf16(training=False)
        expected_gb = params * 2 * 2 / 1e9
        assert abs(vram_gb - expected_gb) < 1e-6
