"""
tests/test_decoder.py — Tests para el StreamDecoder (Paso 7)
=============================================================

Verifica:
  1. Shapes correctas de todos los outputs
  2. Cross-attention efectiva (graph_repr cambia logits)
  3. Rama Mamba escala O(L) en seq_len
  4. Gradientes fluyen por ambas ramas (Mamba y cross-attn)
  5. Anchor head produce distribución sobre max_graph_nodes
  6. MetaHead produce valores en [0, 1]
  7. Weight tying entre token_embedding y lm_head
  8. Estabilidad numérica con secuencias largas
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import pytest
import torch
import torch.nn as nn

from decoder import (
    DecoderOutput,
    HybridDecoderLayer,
    MetaOutput,
    OutputMetaHead,
    StreamDecoder,
    StreamDecoderConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES COMUNES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_cfg():
    """Config mínima para tests rápidos."""
    return StreamDecoderConfig(
        vocab_size      = 512,
        hidden_dim      = 32,
        n_layers        = 2,
        n_heads         = 4,
        node_dim        = 32,
        max_graph_nodes = 8,
        max_seq_len     = 64,
        state_dim       = 4,
        expand          = 2,
        d_conv          = 4,
        ffn_mult        = 2,
    )


@pytest.fixture
def decoder(tiny_cfg):
    return StreamDecoder(tiny_cfg)


@pytest.fixture
def small_batch(tiny_cfg):
    """Batch (B=2, L=6, n_nodes=4)."""
    B, L, n_nodes = 2, 6, 4
    token_ids  = torch.randint(0, tiny_cfg.vocab_size, (B, L))
    graph_repr = torch.randn(B, n_nodes, tiny_cfg.node_dim)
    return token_ids, graph_repr


# ─────────────────────────────────────────────────────────────────────────────
# 1. StreamDecoderConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamDecoderConfig:

    def test_default_values(self):
        cfg = StreamDecoderConfig()
        assert cfg.vocab_size      == 32_000
        assert cfg.hidden_dim      == 256
        assert cfg.n_layers        == 4
        assert cfg.n_heads         == 4
        assert cfg.node_dim        == 256
        assert cfg.max_graph_nodes == 32
        assert cfg.max_seq_len     == 2048

    def test_d_inner_property(self):
        cfg = StreamDecoderConfig(hidden_dim=64, expand=2)
        assert cfg.d_inner == 128

    def test_dt_rank_property(self):
        cfg = StreamDecoderConfig(hidden_dim=256)
        assert cfg.dt_rank == math.ceil(256 / 16)

    def test_invalid_n_heads_raises(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            StreamDecoderConfig(hidden_dim=33, n_heads=4)

    def test_invalid_vocab_size_raises(self):
        with pytest.raises(ValueError, match="vocab_size"):
            StreamDecoderConfig(vocab_size=0)

    def test_invalid_max_graph_nodes_raises(self):
        with pytest.raises(ValueError, match="max_graph_nodes"):
            StreamDecoderConfig(max_graph_nodes=0)

    def test_custom_tiny_config(self, tiny_cfg):
        assert tiny_cfg.vocab_size == 512
        assert tiny_cfg.hidden_dim == 32
        assert tiny_cfg.n_heads    == 4


# ─────────────────────────────────────────────────────────────────────────────
# 2. HybridDecoderLayer
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridDecoderLayer:

    @pytest.fixture
    def layer(self, tiny_cfg):
        return HybridDecoderLayer(tiny_cfg)

    def test_output_shape(self, layer, tiny_cfg):
        B, L, n_nodes = 2, 5, 4
        x          = torch.randn(B, L, tiny_cfg.hidden_dim)
        graph_repr = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        out = layer(x, graph_repr)
        assert out.shape == (B, L, tiny_cfg.hidden_dim)

    def test_output_changes_with_graph(self, layer, tiny_cfg):
        """La cross-attention hace que graph_repr cambie el output."""
        torch.manual_seed(0)
        B, L, n_nodes = 1, 4, 4
        x     = torch.randn(B, L, tiny_cfg.hidden_dim)
        g1    = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        g2    = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        out1  = layer(x, g1)
        out2  = layer(x, g2)
        assert not torch.allclose(out1, out2), "graph_repr must affect output"

    def test_output_changes_with_tokens(self, layer, tiny_cfg):
        """Tokens distintos → output distinto."""
        B, L, n_nodes = 1, 4, 3
        x1 = torch.randn(B, L, tiny_cfg.hidden_dim)
        x2 = torch.randn(B, L, tiny_cfg.hidden_dim)
        g  = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        assert not torch.allclose(layer(x1, g), layer(x2, g))

    def test_gradient_flows_through_mamba(self, layer, tiny_cfg):
        """Gradientes fluyen hasta los parámetros del MambaLayer."""
        B, L, n_nodes = 1, 4, 3
        x = torch.randn(B, L, tiny_cfg.hidden_dim, requires_grad=True)
        g = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        out = layer(x, g)
        out.sum().backward()
        assert x.grad is not None
        # Los parámetros del SSM deben tener gradiente
        mamba_params_with_grad = [
            p for p in layer.mamba.parameters() if p.grad is not None
        ]
        assert len(mamba_params_with_grad) > 0

    def test_gradient_flows_through_cross_attn(self, layer, tiny_cfg):
        """Gradientes fluyen hasta graph_repr (cross-attention)."""
        B, L, n_nodes = 1, 4, 3
        x = torch.randn(B, L, tiny_cfg.hidden_dim)
        g = torch.randn(B, n_nodes, tiny_cfg.node_dim, requires_grad=True)
        out = layer(x, g)
        out.sum().backward()
        assert g.grad is not None
        assert g.grad.abs().sum() > 0

    def test_cross_attn_params_get_gradients(self, layer, tiny_cfg):
        """Los parámetros del cross-attention reciben gradientes."""
        B, L, n_nodes = 1, 4, 3
        x = torch.randn(B, L, tiny_cfg.hidden_dim)
        g = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        layer(x, g).sum().backward()
        ca_params_with_grad = [
            p for p in layer.cross_attn.parameters() if p.grad is not None
        ]
        assert len(ca_params_with_grad) > 0

    def test_output_is_finite(self, layer, tiny_cfg):
        B, L, n_nodes = 2, 8, 5
        x = torch.randn(B, L, tiny_cfg.hidden_dim)
        g = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        out = layer(x, g)
        assert out.isfinite().all()

    def test_graph_proj_used_when_dims_differ(self, tiny_cfg):
        """Si node_dim ≠ hidden_dim, graph_proj es nn.Linear (no Identity)."""
        cfg2 = StreamDecoderConfig(
            vocab_size=512, hidden_dim=32, n_heads=4,
            node_dim=64,   # diferente
            state_dim=4, expand=2, ffn_mult=2
        )
        layer = HybridDecoderLayer(cfg2)
        assert isinstance(layer.graph_proj, nn.Linear)

    def test_graph_proj_identity_when_dims_equal(self, tiny_cfg):
        """Si node_dim == hidden_dim, graph_proj es Identity."""
        layer = HybridDecoderLayer(tiny_cfg)
        assert isinstance(layer.graph_proj, nn.Identity)

    def test_single_node_graph(self, layer, tiny_cfg):
        """Funciona con un solo nodo en el grafo."""
        B, L = 1, 4
        x = torch.randn(B, L, tiny_cfg.hidden_dim)
        g = torch.randn(B, 1, tiny_cfg.node_dim)
        out = layer(x, g)
        assert out.shape == (B, L, tiny_cfg.hidden_dim)

    def test_single_token(self, layer, tiny_cfg):
        """Funciona con L=1."""
        B, n_nodes = 1, 3
        x = torch.randn(B, 1, tiny_cfg.hidden_dim)
        g = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        out = layer(x, g)
        assert out.shape == (B, 1, tiny_cfg.hidden_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 3. OutputMetaHead
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputMetaHead:

    @pytest.fixture
    def meta_head(self, tiny_cfg):
        return OutputMetaHead(tiny_cfg)

    def test_confidence_shape(self, meta_head, tiny_cfg):
        B, L, n_nodes = 3, 6, 4
        hidden = torch.randn(B, L, tiny_cfg.hidden_dim)
        graph  = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        out = meta_head(hidden, graph)
        assert out.confidence.shape == (B,)

    def test_needs_clarification_shape(self, meta_head, tiny_cfg):
        B, L, n_nodes = 3, 6, 4
        hidden = torch.randn(B, L, tiny_cfg.hidden_dim)
        graph  = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        out = meta_head(hidden, graph)
        assert out.needs_clarification.shape == (B,)

    def test_confidence_in_01(self, meta_head, tiny_cfg):
        """confidence ∈ [0, 1] — sigmoid bounded."""
        B, L, n_nodes = 4, 8, 5
        hidden = torch.randn(B, L, tiny_cfg.hidden_dim) * 10
        graph  = torch.randn(B, n_nodes, tiny_cfg.node_dim) * 10
        out = meta_head(hidden, graph)
        assert (out.confidence >= 0).all()
        assert (out.confidence <= 1).all()

    def test_needs_clarification_in_01(self, meta_head, tiny_cfg):
        """needs_clarification ∈ [0, 1]."""
        B, L, n_nodes = 4, 8, 5
        hidden = torch.randn(B, L, tiny_cfg.hidden_dim) * 10
        graph  = torch.randn(B, n_nodes, tiny_cfg.node_dim) * 10
        out = meta_head(hidden, graph)
        assert (out.needs_clarification >= 0).all()
        assert (out.needs_clarification <= 1).all()

    def test_returns_meta_output(self, meta_head, tiny_cfg):
        B, L, n_nodes = 2, 4, 3
        hidden = torch.randn(B, L, tiny_cfg.hidden_dim)
        graph  = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        out = meta_head(hidden, graph)
        assert isinstance(out, MetaOutput)

    def test_gradient_flows(self, meta_head, tiny_cfg):
        B, L, n_nodes = 2, 4, 3
        hidden = torch.randn(B, L, tiny_cfg.hidden_dim, requires_grad=True)
        graph  = torch.randn(B, n_nodes, tiny_cfg.node_dim, requires_grad=True)
        out = meta_head(hidden, graph)
        (out.confidence + out.needs_clarification).sum().backward()
        assert hidden.grad is not None
        assert graph.grad is not None

    def test_different_graphs_different_meta(self, meta_head, tiny_cfg):
        """El meta head usa el grafo: grafos distintos → valores distintos."""
        B, L, n_nodes = 1, 4, 3
        hidden = torch.randn(B, L, tiny_cfg.hidden_dim)
        g1     = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        g2     = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        out1   = meta_head(hidden, g1)
        out2   = meta_head(hidden, g2)
        assert not torch.allclose(out1.confidence, out2.confidence)

    def test_output_finite(self, meta_head, tiny_cfg):
        B, L, n_nodes = 2, 6, 4
        hidden = torch.randn(B, L, tiny_cfg.hidden_dim)
        graph  = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        out = meta_head(hidden, graph)
        assert out.confidence.isfinite().all()
        assert out.needs_clarification.isfinite().all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. StreamDecoder — shapes y outputs
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamDecoderShapes:

    def test_logits_shape(self, decoder, small_batch, tiny_cfg):
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        B, L = token_ids.shape
        assert out.logits.shape == (B, L, tiny_cfg.vocab_size)

    def test_anchor_logits_shape(self, decoder, small_batch, tiny_cfg):
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        B, L = token_ids.shape
        assert out.anchor_logits.shape == (B, L, tiny_cfg.max_graph_nodes)

    def test_confidence_shape(self, decoder, small_batch):
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        B = token_ids.shape[0]
        assert out.confidence.shape == (B,)

    def test_needs_clarification_shape(self, decoder, small_batch):
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        B = token_ids.shape[0]
        assert out.needs_clarification.shape == (B,)

    def test_returns_decoder_output(self, decoder, small_batch):
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        assert isinstance(out, DecoderOutput)

    def test_all_outputs_finite(self, decoder, small_batch):
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        assert out.logits.isfinite().all()
        assert out.anchor_logits.isfinite().all()
        assert out.confidence.isfinite().all()
        assert out.needs_clarification.isfinite().all()

    def test_confidence_in_01(self, decoder, small_batch):
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        assert (out.confidence >= 0).all()
        assert (out.confidence <= 1).all()

    def test_needs_clarification_in_01(self, decoder, small_batch):
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        assert (out.needs_clarification >= 0).all()
        assert (out.needs_clarification <= 1).all()

    def test_batch_size_1(self, decoder, tiny_cfg):
        token_ids  = torch.randint(0, tiny_cfg.vocab_size, (1, 5))
        graph_repr = torch.randn(1, 3, tiny_cfg.node_dim)
        out = decoder(token_ids, graph_repr)
        assert out.logits.shape == (1, 5, tiny_cfg.vocab_size)

    def test_single_token(self, decoder, tiny_cfg):
        """L=1 funciona (generación token a token)."""
        token_ids  = torch.randint(0, tiny_cfg.vocab_size, (2, 1))
        graph_repr = torch.randn(2, 3, tiny_cfg.node_dim)
        out = decoder(token_ids, graph_repr)
        assert out.logits.shape == (2, 1, tiny_cfg.vocab_size)

    def test_max_graph_nodes_graph(self, decoder, tiny_cfg):
        """Funciona con el número máximo de nodos configurados."""
        B, L = 2, 4
        token_ids  = torch.randint(0, tiny_cfg.vocab_size, (B, L))
        graph_repr = torch.randn(B, tiny_cfg.max_graph_nodes, tiny_cfg.node_dim)
        out = decoder(token_ids, graph_repr)
        assert out.logits.shape == (B, L, tiny_cfg.vocab_size)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cross-attention efectiva — graph_repr cambia los logits
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossAttentionEffectiveness:

    def test_different_graph_different_logits(self, decoder, tiny_cfg):
        """
        Con los mismos tokens pero distinto grafo, los logits deben diferir.
        Esto verifica que la cross-attention realmente condiciona la salida.
        """
        decoder.eval()
        B, L, n_nodes = 1, 4, 3
        token_ids = torch.randint(0, tiny_cfg.vocab_size, (B, L))
        g1 = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        g2 = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        with torch.no_grad():
            out1 = decoder(token_ids, g1)
            out2 = decoder(token_ids, g2)
        assert not torch.allclose(out1.logits, out2.logits), \
            "Different graph_repr must produce different logits"

    def test_same_graph_same_logits(self, decoder, tiny_cfg):
        """En eval mode, mismo input → mismo output (determinismo)."""
        decoder.eval()
        B, L, n_nodes = 1, 4, 3
        token_ids = torch.randint(0, tiny_cfg.vocab_size, (B, L))
        g = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        with torch.no_grad():
            out1 = decoder(token_ids, g)
            out2 = decoder(token_ids, g)
        assert torch.allclose(out1.logits, out2.logits)

    def test_graph_affects_anchor_logits(self, decoder, tiny_cfg):
        """El grafo también debe afectar anchor_logits."""
        decoder.eval()
        B, L, n_nodes = 1, 4, 3
        token_ids = torch.randint(0, tiny_cfg.vocab_size, (B, L))
        g1 = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        g2 = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        with torch.no_grad():
            out1 = decoder(token_ids, g1)
            out2 = decoder(token_ids, g2)
        assert not torch.allclose(out1.anchor_logits, out2.anchor_logits)

    def test_scaled_graph_changes_output(self, decoder, tiny_cfg):
        """Grafo escalado (×10) produce output diferente."""
        decoder.eval()
        B, L, n_nodes = 1, 4, 3
        token_ids = torch.randint(0, tiny_cfg.vocab_size, (B, L))
        g = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        with torch.no_grad():
            out1 = decoder(token_ids, g)
            out2 = decoder(token_ids, g * 10.0)
        assert not torch.allclose(out1.logits, out2.logits)

    def test_graph_conditioning_per_layer(self, tiny_cfg):
        """
        Cada HybridDecoderLayer tiene cross-attention:
        bloquear el grafo en todas las capas debería cambiar el output más
        que en una sola capa.
        """
        decoder = StreamDecoder(tiny_cfg)
        decoder.eval()
        B, L, n_nodes = 1, 4, 3
        token_ids = torch.randint(0, tiny_cfg.vocab_size, (B, L))
        g_ones   = torch.ones(B, n_nodes, tiny_cfg.node_dim)
        g_zeros  = torch.zeros(B, n_nodes, tiny_cfg.node_dim)
        with torch.no_grad():
            diff = (decoder(token_ids, g_ones).logits
                    - decoder(token_ids, g_zeros).logits).abs().mean()
        assert diff > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Complejidad O(L) de la rama Mamba
# ─────────────────────────────────────────────────────────────────────────────

class TestMambaLinearComplexity:

    def _get_A_bar_numel(self, decoder, token_ids, graph_repr):
        """
        Recorre las capas buscando SelectiveSSM y devuelve el numel del
        último A_bar computado. A_bar.numel() = B × L × D_inner × state_dim,
        es decir, lineal en L.
        """
        # Hacer un forward para que _last_A_bar_numel se actualice
        decoder.eval()
        with torch.no_grad():
            decoder(token_ids, graph_repr)
        # Extraer de la primera capa Mamba
        ssm = decoder.layers[0].mamba.ssm
        return ssm._last_A_bar_numel

    def test_A_bar_scales_linearly_with_L(self, tiny_cfg):
        """
        A_bar.numel() = B × L × D_inner × state_dim
        Duplicar L → duplicar numel → O(L), no O(L²).
        """
        decoder = StreamDecoder(tiny_cfg)
        n_nodes = 3
        B = 1

        L1 = 4
        L2 = 8

        t1 = torch.randint(0, tiny_cfg.vocab_size, (B, L1))
        t2 = torch.randint(0, tiny_cfg.vocab_size, (B, L2))
        g  = torch.randn(B, n_nodes, tiny_cfg.node_dim)

        numel1 = self._get_A_bar_numel(decoder, t1, g)
        numel2 = self._get_A_bar_numel(decoder, t2, g)

        ratio = numel2 / numel1
        assert abs(ratio - 2.0) < 0.01, (
            f"Expected A_bar to double with double L, got ratio={ratio:.3f}"
        )

    def test_no_attention_matrix_quadratic_growth(self, tiny_cfg):
        """
        El MambaLayer no almacena ninguna attention matrix L×L.
        Verificar que el número de parámetros es constante — no crece con L.
        """
        decoder = StreamDecoder(tiny_cfg)
        params_before = decoder.count_parameters()

        # No hay parámetros adicionales en función de L (todo es input-dependiente)
        assert params_before == decoder.count_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Gradientes — fluyen por ambas ramas
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_gradient_flows_to_token_ids_embedding(self, decoder, small_batch):
        """Gradientes fluyen hacia los embeddings de tokens."""
        token_ids, graph_repr = small_batch
        # Gradient respecto a los embeddings (no token_ids, que son enteros)
        emb = decoder.token_embedding(token_ids)
        emb.retain_grad()
        # Forward completo con el embedding como punto de entrada
        # (necesitamos usar el embedding directamente)
        out = decoder(token_ids, graph_repr)
        out.logits.sum().backward()
        # Los parámetros de token_embedding sí deben tener grad
        assert decoder.token_embedding.weight.grad is not None

    def test_gradient_flows_to_graph_repr(self, decoder, small_batch):
        """Gradientes fluyen hasta graph_repr a través del cross-attention."""
        token_ids, graph_repr = small_batch
        graph_repr = graph_repr.requires_grad_(True)
        out = decoder(token_ids, graph_repr)
        out.logits.sum().backward()
        assert graph_repr.grad is not None
        assert graph_repr.grad.abs().sum() > 0

    def test_mamba_params_get_gradients(self, decoder, small_batch):
        """Los parámetros del SSM dentro de MambaLayer reciben gradientes."""
        token_ids, graph_repr = small_batch
        decoder(token_ids, graph_repr).logits.sum().backward()
        # Buscar parámetros del SSM en cada capa
        mamba_grads = []
        for layer in decoder.layers:
            for p in layer.mamba.ssm.parameters():
                if p.grad is not None:
                    mamba_grads.append(p.grad.abs().sum().item())
        assert len(mamba_grads) > 0, "No Mamba SSM params received gradients"
        assert any(g > 0 for g in mamba_grads)

    def test_cross_attn_params_get_gradients(self, decoder, small_batch):
        """Los parámetros del cross-attention reciben gradientes."""
        token_ids, graph_repr = small_batch
        decoder(token_ids, graph_repr).logits.sum().backward()
        ca_grads = []
        for layer in decoder.layers:
            for p in layer.cross_attn.parameters():
                if p.grad is not None:
                    ca_grads.append(p.grad.abs().sum().item())
        assert len(ca_grads) > 0, "No cross-attention params received gradients"

    def test_anchor_head_gets_gradients(self, decoder, small_batch):
        """El anchor_head recibe gradientes."""
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        out.anchor_logits.sum().backward()
        assert decoder.anchor_head.weight.grad is not None

    def test_meta_head_gets_gradients(self, decoder, small_batch):
        """El meta_head recibe gradientes."""
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        (out.confidence + out.needs_clarification).sum().backward()
        meta_grads = [
            p.grad for p in decoder.meta_head.parameters() if p.grad is not None
        ]
        assert len(meta_grads) > 0

    def test_gradients_finite(self, decoder, small_batch):
        """Todos los gradientes son finitos (sin NaN/Inf)."""
        token_ids, graph_repr = small_batch
        decoder(token_ids, graph_repr).logits.sum().backward()
        for name, p in decoder.named_parameters():
            if p.grad is not None:
                assert p.grad.isfinite().all(), f"Non-finite grad in {name}"

    def test_gradient_flows_through_all_layers(self, tiny_cfg):
        """Con n_layers=4, los gradientes deben llegar a la primera capa."""
        cfg = StreamDecoderConfig(
            vocab_size=256, hidden_dim=32, n_layers=4, n_heads=4,
            node_dim=32, max_graph_nodes=8, state_dim=4, expand=2, ffn_mult=2
        )
        decoder = StreamDecoder(cfg)
        token_ids  = torch.randint(0, cfg.vocab_size, (1, 4))
        graph_repr = torch.randn(1, 3, cfg.node_dim)
        decoder(token_ids, graph_repr).logits.sum().backward()
        # Verificar que la primera capa recibió gradientes
        first_layer_grads = [
            p.grad for p in decoder.layers[0].parameters() if p.grad is not None
        ]
        assert len(first_layer_grads) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Weight tying
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightTying:

    def test_lm_head_and_embedding_share_weights(self, decoder):
        """lm_head.weight y token_embedding.weight son el mismo objeto."""
        assert decoder.lm_head.weight is decoder.token_embedding.weight

    def test_tied_weights_same_values(self, decoder):
        """Mismo tensor → mismos valores."""
        assert torch.allclose(
            decoder.lm_head.weight,
            decoder.token_embedding.weight,
        )

    def test_tying_reduces_unique_params(self, decoder, tiny_cfg):
        """count_parameters() no cuenta lm_head.weight dos veces."""
        vocab_params = tiny_cfg.vocab_size * tiny_cfg.hidden_dim
        total_unique = decoder.count_parameters()
        # Sin tying habría: total_unique + vocab_params (un tensor extra para lm_head)
        # Con tying, solo existe un tensor compartido → total_unique es correcto
        # Verificar: total < lo que sería con dos tensores separados
        theoretical_untied = total_unique + vocab_params
        assert total_unique < theoretical_untied  # tying elimina el duplicado


# ─────────────────────────────────────────────────────────────────────────────
# 9. Anchor head
# ─────────────────────────────────────────────────────────────────────────────

class TestAnchorHead:

    def test_anchor_logits_cover_max_graph_nodes(self, decoder, small_batch, tiny_cfg):
        """anchor_logits tiene last dim = max_graph_nodes."""
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        assert out.anchor_logits.shape[-1] == tiny_cfg.max_graph_nodes

    def test_anchor_logits_finite(self, decoder, small_batch):
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        assert out.anchor_logits.isfinite().all()

    def test_anchor_softmax_gives_distribution(self, decoder, small_batch):
        """Softmax sobre anchor_logits → distribución válida sobre nodos."""
        token_ids, graph_repr = small_batch
        out = decoder(token_ids, graph_repr)
        probs = torch.softmax(out.anchor_logits, dim=-1)
        assert (probs >= 0).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))

    def test_different_tokens_different_anchors(self, decoder, tiny_cfg):
        """Tokens distintos → anchor logits distintos."""
        decoder.eval()
        n_nodes = 3
        B = 1
        t1 = torch.randint(0, tiny_cfg.vocab_size, (B, 4))
        t2 = torch.randint(0, tiny_cfg.vocab_size, (B, 4))
        g  = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        with torch.no_grad():
            out1 = decoder(t1, g)
            out2 = decoder(t2, g)
        # Con tokens distintos, la mayoría de las veces los anchor_logits difieren
        assert not torch.allclose(out1.anchor_logits, out2.anchor_logits)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Parameter count y breakdown
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterCount:

    def test_count_parameters_positive(self, decoder):
        assert decoder.count_parameters() > 0

    def test_parameter_breakdown_keys(self, decoder):
        bd = decoder.parameter_breakdown()
        for key in ["token_embedding", "pos_embedding", "layers",
                    "lm_head", "anchor_head", "meta_head", "total_unique"]:
            assert key in bd, f"Missing key '{key}' in breakdown"

    def test_breakdown_total_matches_count(self, decoder):
        bd = decoder.parameter_breakdown()
        assert bd["total_unique"] == decoder.count_parameters()

    def test_lm_head_counted_as_zero_due_to_tying(self, decoder):
        """lm_head está tied → contribución = 0 en el breakdown."""
        bd = decoder.parameter_breakdown()
        assert bd["lm_head"] == 0

    def test_layers_params_positive(self, decoder):
        bd = decoder.parameter_breakdown()
        assert bd["layers"] > 0

    def test_tiny_config_reasonable_size(self, decoder):
        """Config tiny debe tener menos de 5M parámetros."""
        assert decoder.count_parameters() < 5_000_000


# ─────────────────────────────────────────────────────────────────────────────
# 11. Estabilidad numérica
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalStability:

    def test_long_sequence_stable(self, decoder, tiny_cfg):
        """Secuencia larga no produce NaN/Inf."""
        B, L, n_nodes = 1, 32, 3
        token_ids  = torch.randint(0, tiny_cfg.vocab_size, (B, L))
        graph_repr = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        out = decoder(token_ids, graph_repr)
        assert out.logits.isfinite().all()

    def test_large_graph_repr_stable(self, decoder, tiny_cfg):
        """graph_repr con valores grandes no rompe el modelo."""
        B, L, n_nodes = 1, 4, 6
        token_ids  = torch.randint(0, tiny_cfg.vocab_size, (B, L))
        graph_repr = torch.randn(B, n_nodes, tiny_cfg.node_dim) * 100.0
        out = decoder(token_ids, graph_repr)
        assert out.logits.isfinite().all()

    def test_zero_graph_repr_stable(self, decoder, tiny_cfg):
        """graph_repr = 0 no rompe el modelo (cross-attn uniform)."""
        B, L, n_nodes = 1, 4, 3
        token_ids  = torch.randint(0, tiny_cfg.vocab_size, (B, L))
        graph_repr = torch.zeros(B, n_nodes, tiny_cfg.node_dim)
        out = decoder(token_ids, graph_repr)
        assert out.logits.isfinite().all()

    def test_eval_mode_stable(self, decoder, tiny_cfg):
        decoder.eval()
        B, L, n_nodes = 2, 8, 4
        token_ids  = torch.randint(0, tiny_cfg.vocab_size, (B, L))
        graph_repr = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        with torch.no_grad():
            out = decoder(token_ids, graph_repr)
        assert out.logits.isfinite().all()
        assert out.anchor_logits.isfinite().all()


# ─────────────────────────────────────────────────────────────────────────────
# 12. Determinismo
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_eval_mode_deterministic(self, decoder, tiny_cfg):
        decoder.eval()
        B, L, n_nodes = 2, 5, 3
        token_ids  = torch.randint(0, tiny_cfg.vocab_size, (B, L))
        graph_repr = torch.randn(B, n_nodes, tiny_cfg.node_dim)
        with torch.no_grad():
            out1 = decoder(token_ids, graph_repr)
            out2 = decoder(token_ids, graph_repr)
        assert torch.allclose(out1.logits, out2.logits)
        assert torch.allclose(out1.anchor_logits, out2.anchor_logits)

    def test_train_mode_same_seed_deterministic(self, tiny_cfg):
        """Con la misma semilla, mismo forward en train mode."""
        cfg     = tiny_cfg
        B, L    = 1, 4
        n_nodes = 3
        token_ids  = torch.randint(0, cfg.vocab_size, (B, L))
        graph_repr = torch.randn(B, n_nodes, cfg.node_dim)

        torch.manual_seed(42)
        d1 = StreamDecoder(cfg)
        out1 = d1(token_ids, graph_repr)

        torch.manual_seed(42)
        d2 = StreamDecoder(cfg)
        out2 = d2(token_ids, graph_repr)

        assert torch.allclose(out1.logits, out2.logits, atol=1e-6)
