"""
crystallizer/pooler.py — CrossAttentionPooler
=============================================

Agrega información del contexto completo en cada vector de nodo detectado.

PROBLEMA QUE RESUELVE:
    El StreamEncoder ya le asignó un vector a cada posición.
    Pero ese vector solo captura el contexto CAUSAL (tokens anteriores, gracias al SSM).
    El GraphCrystallizer necesita un vector que capture el CONTEXTO COMPLETO del nodo
    — tanto lo que precede como lo que sigue — para decidir la relación con otros nodos.

    Ejemplo: "El fuego [NODO] se extinguió" vs "El fuego [NODO] se propagó"
    El significado de "fuego" como nodo del grafo depende del contexto bidireccional.

SOLUCIÓN:
    Cross-attention donde:
        Queries  = vectores de posiciones detectadas como nodos (lo que queremos enriquecer)
        Keys     = secuencia completa de concept vectors (contexto a consultar)
        Values   = secuencia completa de concept vectors (información a agregar)

    Resultado: cada nodo tiene un vector enriquecido con información global.
    Los vectores de nodo son la input al AsymmetricRelationScorer.

POR QUÉ CROSS-ATTENTION Y NO SELF-ATTENTION:
    Self-attention entre nodos requeriría ya tener los vectores finales de nodo.
    Cross-attention con el contexto completo es el paso que construye esos vectores.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import CrystallizerConfig


class CrossAttentionPooler(nn.Module):
    """
    Agrega contexto en vectores de nodo mediante cross-attention.

    Uso:
        cfg    = CrystallizerConfig()
        pooler = CrossAttentionPooler(cfg)
        queries = torch.randn(2, 8, 256)   # [B, n_nodes, D]
        context = torch.randn(2, 64, 256)  # [B, L, D]
        out = pooler(queries, context)     # [B, n_nodes, D]
    """

    def __init__(self, config: CrystallizerConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        H = config.pooler_heads

        self.n_heads = H
        self.head_dim = D // H
        self.scale = math.sqrt(self.head_dim) ** -1  # = 1/√head_dim

        # Queries vienen de los nodos detectados
        # Keys y Values vienen del contexto completo
        self.q_proj   = nn.Linear(D, D, bias=False)
        self.k_proj   = nn.Linear(D, D, bias=False)
        self.v_proj   = nn.Linear(D, D, bias=False)
        self.out_proj  = nn.Linear(D, D, bias=False)

        # Normalización post-residual (estándar en bloques de cross-attention)
        self.norm = nn.LayerNorm(D)

        self._init_weights()

    def _init_weights(self) -> None:
        # Proyecciones con std mayor que 0.02 para que la cross-attention
        # produzca representaciones distintas por nodo desde el inicio.
        # Con std=0.02 (default LLM), Q≈0 → atención uniforme → todos los nodos
        # reciben el mismo vector agregado, perdiendo la identidad del nodo.
        # std=0.1 produce scores de atención con varianza suficiente para diferenciarlos.
        std = 0.1
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.normal_(proj.weight, std=std)
        # out_proj más conservadora para estabilidad del residual
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(
        self,
        node_queries: torch.Tensor,  # [B, n_nodes, D]
        context: torch.Tensor,       # [B, L, D]
    ) -> torch.Tensor:
        """
        Args:
            node_queries: [B, n_nodes, D] — vectores de nodo (queries)
            context:      [B, L, D]       — secuencia completa (keys + values)

        Returns:
            node_vectors: [B, n_nodes, D] — nodos enriquecidos con contexto
        """
        B, n, D = node_queries.shape
        L = context.shape[1]
        H = self.n_heads
        Hd = self.head_dim

        # Proyectar y separar cabezas
        Q = self.q_proj(node_queries).view(B, n, H, Hd).transpose(1, 2)  # [B, H, n, Hd]
        K = self.k_proj(context).view(B, L, H, Hd).transpose(1, 2)        # [B, H, L, Hd]
        V = self.v_proj(context).view(B, L, H, Hd).transpose(1, 2)        # [B, H, L, Hd]

        # Atención: cada nodo consulta toda la secuencia
        attn_weights = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, n, L]
        attn_weights = torch.softmax(attn_weights, dim=-1)     # [B, H, n, L]

        # Agregación
        attn_out = attn_weights @ V                                      # [B, H, n, Hd]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, n, D)  # [B, n, D]

        # Residual: cada nodo retiene su vector de consulta original.
        # Esto garantiza que nodos distintos produzcan representaciones distintas,
        # incluso cuando la cross-attention está inicializada con pesos pequeños.
        # node_vectors[i] = LN(node_queries[i] + out_proj(attn_out[i]))
        return self.norm(node_queries + self.out_proj(attn_out))         # [B, n, D]
