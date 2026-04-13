"""
decoder/config.py — StreamDecoderConfig
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class StreamDecoderConfig:
    """
    Configuración del StreamDecoder.

    Tiny (testing):
        hidden_dim=256, n_layers=4, vocab_size=32000, n_heads=4, max_graph_nodes=32

    node_dim debe coincidir con CREConfig.node_dim (representaciones de nodos del grafo).
    max_seq_len es el límite de la tabla de posiciones aprendidas.
    """
    vocab_size:      int   = 32_000
    hidden_dim:      int   = 256      # D: dimensión del modelo
    n_layers:        int   = 4        # Número de HybridDecoderLayers
    n_heads:         int   = 4        # Cabezas del cross-attention
    node_dim:        int   = 256      # Dimensión de los node features del grafo (CREOutput)
    max_graph_nodes: int   = 32       # Máximo de nodos — coincide con CrystallizerConfig.max_nodes
    max_seq_len:     int   = 2048     # Límite de la tabla de posiciones aprendidas
    # ── Parámetros Mamba (compartidos con el encoder SSM interno) ────────────
    state_dim:       int   = 16       # N: dimensión del estado SSM
    expand:          int   = 2        # D_inner = expand × D
    d_conv:          int   = 4        # Ancho de la convolución causal
    ffn_mult:        int   = 4        # GatedFFN inner = ffn_mult × D
    dropout:         float = 0.0
    bias:            bool  = False
    rms_eps:         float = 1e-5
    norm_eps:        float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.max_graph_nodes < 1:
            raise ValueError(f"max_graph_nodes must be >= 1, got {self.max_graph_nodes}")

    @property
    def d_inner(self) -> int:
        """Dimensión interna del SSM (D_inner = expand × D)."""
        return self.expand * self.hidden_dim

    @property
    def dt_rank(self) -> int:
        """Rango de Δ — igual que en el encoder."""
        return math.ceil(self.hidden_dim / 16)
