"""
decoder/hybrid_layer.py — HybridDecoderLayer
=============================================

Capa híbrida del StreamDecoder que combina cuatro sub-módulos en secuencia:

    1. MambaLayer (reutilizado de encoder/mamba_layer.py)
       Procesamiento CAUSAL de la secuencia de tokens.
       O(L) en memoria y compute — hereda todos los beneficios del SSM.

    2. Cross-attention al grafo causal (nn.MultiheadAttention)
       Los tokens "consultan" el grafo para anclar su semántica causal.
       Q = tokens  [B, L, D]
       K = V = nodos del grafo  [B, n_nodes, D]

    3. Cross-attention a los concept vectors del encoder (nn.MultiheadAttention)
       Los tokens "consultan" directamente el input para preservar identidad léxica.
       Q = tokens  [B, L, D]
       K = V = concept vectors del encoder  [B, L_enc, D]
       Resuelve el problema de grounding: sin esta capa, la identidad léxica
       ("fiebre" vs "incendio") se pierde al pasar por crystallizer → CRE.
       El transformer no tiene este problema porque su cross-attention ve
       directamente los tokens del encoder — esta capa replica esa propiedad.

    4. GatedFFN (reutilizado de encoder/mamba_layer.py)
       Procesamiento adicional por token después de integrar ambos contextos.

FLUJO:
    x [B, L, D]
    → MambaLayer (SSM + FFN interno)              → x'   [B, L, D]
    → LN → CrossAttn(graph_repr) + resid          → x''  [B, L, D]
    → LN → CrossAttn(encoder_concepts) + resid   → x''' [B, L, D]
    → LN → GatedFFN + resid                       → x'''' [B, L, D]

REUTILIZACIÓN DE ENCODER:
    MambaLayer y GatedFFN se importan de encoder.mamba_layer sin modificación.
    El HybridDecoderLayer crea un StreamEncoderConfig compatible usando
    los hiperparámetros del StreamDecoderConfig.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from encoder.mamba_layer import GatedFFN, MambaLayer, StreamEncoderConfig
from .config import StreamDecoderConfig


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: construir StreamEncoderConfig desde StreamDecoderConfig
# ─────────────────────────────────────────────────────────────────────────────

def _make_mamba_config(cfg: StreamDecoderConfig) -> StreamEncoderConfig:
    """
    Construye un StreamEncoderConfig compatible para instanciar MambaLayer.
    El MambaLayer del encoder es idéntico al que necesitamos aquí —
    la única diferencia es el contexto (decoder vs encoder).
    """
    return StreamEncoderConfig(
        vocab_size  = cfg.vocab_size,
        hidden_dim  = cfg.hidden_dim,
        n_layers    = 1,             # solo usamos 1 layer a la vez
        state_dim   = cfg.state_dim,
        expand      = cfg.expand,
        d_conv      = cfg.d_conv,
        ffn_mult    = cfg.ffn_mult,
        dropout     = cfg.dropout,
        bias        = cfg.bias,
        rms_eps     = cfg.rms_eps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID DECODER LAYER
# ─────────────────────────────────────────────────────────────────────────────

class HybridDecoderLayer(nn.Module):
    """
    MambaLayer + cross-attention al grafo + GatedFFN.

    Uso:
        layer = HybridDecoderLayer(config)
        x_out = layer(x, graph_repr)
        # x_out: [B, L, hidden_dim]

    Args:
        config: StreamDecoderConfig
    """

    def __init__(self, config: StreamDecoderConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        G = config.node_dim

        # ── 1. Mamba SSM block (con su GatedFFN interno) ──────────────────────
        mamba_cfg = _make_mamba_config(config)
        self.mamba = MambaLayer(mamba_cfg)

        # ── 2. Cross-attention hacia el grafo (estructura causal) ─────────────
        self.cross_attn_norm = nn.LayerNorm(D, eps=config.norm_eps)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = D,
            num_heads   = config.n_heads,
            batch_first = True,
            bias        = config.bias,
            dropout     = config.dropout,
        )

        # Proyecta node features al espacio del decoder si difieren en dimensión
        self.graph_proj: nn.Module = (
            nn.Linear(G, D, bias=False) if G != D else nn.Identity()
        )

        # ── 3. Cross-attention a concepts del encoder (identidad léxica) ───────
        # Permite que cada token generado "lea" directamente los concept vectors
        # del encoder, preservando la identidad léxica original que el
        # crystallizer → CRE puede abstraer o transformar.
        self.enc_attn_norm = nn.LayerNorm(D, eps=config.norm_eps)

        self.enc_attn = nn.MultiheadAttention(
            embed_dim   = D,
            num_heads   = config.n_heads,
            batch_first = True,
            bias        = config.bias,
            dropout     = config.dropout,
        )

        # ── 4. GatedFFN adicional (post-cross-attentions) ─────────────────────
        self.ffn_norm = nn.LayerNorm(D, eps=config.norm_eps)
        self.ffn      = GatedFFN(D, ffn_mult=config.ffn_mult, bias=config.bias)

        self.drop = (
            nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # graph_proj puede ser Identity o Linear
        if isinstance(self.graph_proj, nn.Linear):
            nn.init.normal_(self.graph_proj.weight, std=0.02)

    def forward(
        self,
        x:                torch.Tensor,              # [B, L, D]
        graph_repr:       torch.Tensor,              # [B, n_nodes, G]
        encoder_concepts: torch.Tensor | None = None,  # [B, L_enc, D]
    ) -> torch.Tensor:
        """
        Args:
            x:                [B, L, hidden_dim]     — hidden states de los tokens
            graph_repr:       [B, n_nodes, node_dim] — features de los nodos del grafo
            encoder_concepts: [B, L_enc, hidden_dim] — concept vectors del StreamEncoder
                              (optional; when None, the encoder cross-attention is skipped)

        Returns:
            [B, L, hidden_dim]
        """
        # ── 1. MambaLayer (SSM causal + FFN interno, pre-norm internamente) ───
        x, _ = self.mamba(x)   # [B, L, D]

        # ── 2. Cross-attention al grafo (estructura causal) ───────────────────
        residual = x
        h = self.cross_attn_norm(x)                    # [B, L, D]
        kv = self.graph_proj(graph_repr)               # [B, n_nodes, D]
        attn_out, _ = self.cross_attn(
            query = h,
            key   = kv,
            value = kv,
        )                                              # [B, L, D]
        x = residual + self.drop(attn_out)

        # ── 3. Cross-attention al encoder (identidad léxica) ──────────────────
        if encoder_concepts is not None:
            residual = x
            h = self.enc_attn_norm(x)                      # [B, L, D]
            enc_out, _ = self.enc_attn(
                query = h,
                key   = encoder_concepts,
                value = encoder_concepts,
            )                                              # [B, L, D]
            x = residual + self.drop(enc_out)

        # ── 4. GatedFFN (pre-norm + residual) ─────────────────────────────────
        residual = x
        x = residual + self.drop(self.ffn(self.ffn_norm(x)))

        return x
