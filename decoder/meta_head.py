"""
decoder/meta_head.py — OutputMetaHead
======================================

Predice metadatos del output del StreamDecoder:
    - confidence:           ¿qué tan seguro está el modelo de su respuesta?
    - needs_clarification:  ¿necesita pedir más contexto al usuario?

ARQUITECTURA:
    Combina:
        1. Representación media de los tokens generados  (mean-pool sobre L)
        2. Representación media del grafo causal         (mean-pool sobre n_nodes)

    Luego aplica dos heads independientes con sigmoid para escalar a [0,1].

    Input:
        hidden:     [B, L, hidden_dim]   — hidden states del decoder
        graph_repr: [B, n_nodes, node_dim] — features del grafo (del CRE)

    Output:
        MetaOutput con confidence [B] y needs_clarification [B]

NOTA:
    Operar sobre la media de tokens (no solo el último) es más estable
    durante el entrenamiento y captura la intención global del texto generado.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StreamDecoderConfig


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetaOutput:
    """
    Resultado del OutputMetaHead.

    confidence:           [B] ∈ [0, 1] — confianza en el output generado
    needs_clarification:  [B] ∈ [0, 1] — probabilidad de necesitar clarificación
    """
    confidence:          torch.Tensor
    needs_clarification: torch.Tensor


# ─────────────────────────────────────────────────────────────────────────────
# META HEAD
# ─────────────────────────────────────────────────────────────────────────────

class OutputMetaHead(nn.Module):
    """
    Predice confidence y needs_clarification del output generado.

    Uso:
        meta_head = OutputMetaHead(config)
        meta = meta_head(hidden, graph_repr)
        # meta.confidence:          [B]
        # meta.needs_clarification: [B]
    """

    def __init__(self, config: StreamDecoderConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        G = config.node_dim

        # Proyección combinada: (token_pool ‖ graph_pool) → hidden
        self.proj = nn.Linear(D + G, D)

        # Dos heads independientes → escalares por ejemplo
        self.confidence_head    = nn.Linear(D, 1)
        self.clarification_head = nn.Linear(D, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.confidence_head.weight, std=0.02)
        nn.init.zeros_(self.confidence_head.bias)
        nn.init.normal_(self.clarification_head.weight, std=0.02)
        nn.init.zeros_(self.clarification_head.bias)

    def forward(
        self,
        hidden:     torch.Tensor,   # [B, L, hidden_dim]
        graph_repr: torch.Tensor,   # [B, n_nodes, node_dim]
    ) -> MetaOutput:
        """
        Args:
            hidden:     [B, L, hidden_dim]    — hidden states finales del decoder
            graph_repr: [B, n_nodes, node_dim] — features del grafo causal

        Returns:
            MetaOutput con tensores [B] ∈ [0, 1]
        """
        # Mean-pool sobre la dimensión de secuencia / nodos
        h_pool = hidden.mean(dim=1)      # [B, D]
        g_pool = graph_repr.mean(dim=1)  # [B, G]

        combined = torch.cat([h_pool, g_pool], dim=-1)  # [B, D+G]
        h = F.gelu(self.proj(combined))                  # [B, D]

        confidence          = torch.sigmoid(self.confidence_head(h).squeeze(-1))     # [B]
        needs_clarification = torch.sigmoid(self.clarification_head(h).squeeze(-1))  # [B]

        return MetaOutput(
            confidence=confidence,
            needs_clarification=needs_clarification,
        )
