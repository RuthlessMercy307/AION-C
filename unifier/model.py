"""
unifier/model.py — Unifier: fusión de grafos de múltiples motores
=================================================================

Cuando el Orchestrator activa varios motores, cada uno produce una
representación vectorial del grafo [k_nodes, D]. El Unifier las fusiona
en un único tensor [max_output_nodes, D] para el decoder.

Estrategias de fusión:
  - 1 motor:  identidad — el grafo pasa sin modificación (zero-overhead)
  - N motores: cross-attention entre grafos + proyección a tamaño fijo

Arquitectura (N > 1):
  1. Concatenación de todos los nodos: [total_nodes, D]
     total_nodes = sum(k_nodes_i)
  2. Cross-attention multi-cabeza: cada nodo atiende a todos los demás
     (los nodos de AXIOM pueden "leer" los de MUSE, etc.)
  3. Capa de fusión (MLP): [total_nodes, D] → [total_nodes, D]
  4. Pooling a tamaño fijo: top-K por norma L2 o padding
     → [max_output_nodes, D]

Batching:
  El Unifier opera sobre UN item del batch a la vez (igual que el CRE).
  Para un batch de B items, se llama B veces.

Resolución de conflictos (implícita):
  La cross-attention aprende a ponderar los nodos de diferentes motores.
  Si CORA y AXIOM dan representaciones conflictivas del mismo concepto,
  la atención aprenderá a priorizar la más relevante para el contexto.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UnifierConfig:
    """
    Configuración del Unifier.

    node_dim:         dimensión de los vectores de nodo (= encoder hidden_dim)
    n_heads:          cabezas del cross-attention (debe dividir node_dim)
    max_output_nodes: número de nodos en la representación fusionada
    dropout:          dropout en cross-attention y MLP
    """
    node_dim:         int   = 256
    n_heads:          int   = 4
    max_output_nodes: int   = 32
    dropout:          float = 0.0

    def __post_init__(self) -> None:
        if self.node_dim % self.n_heads != 0:
            raise ValueError(
                f"node_dim ({self.node_dim}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UnifierOutput:
    """
    Resultado del Unifier.

    unified:         [max_output_nodes, D] — representación fusionada
    n_source_motors: número de motores que contribuyeron
    total_nodes:     total de nodos de entrada antes del pooling
    """
    unified:         torch.Tensor
    n_source_motors: int
    total_nodes:     int


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIER
# ─────────────────────────────────────────────────────────────────────────────

class Unifier(nn.Module):
    """
    Unifier: fusiona representaciones de grafos de múltiples motores.

    Para un solo motor, la operación es identidad (sin overhead).
    Para múltiples motores, aplica cross-attention y proyección.

    Uso:
        unifier = Unifier(UnifierConfig(node_dim=64))

        # Un solo motor:
        out = unifier([tensor_shape_k_D])       → unified [max_nodes, D]

        # Dos motores:
        out = unifier([tensor_a, tensor_b])     → unified [max_nodes, D]
    """

    def __init__(self, config: UnifierConfig) -> None:
        super().__init__()
        self.config = config
        D = config.node_dim
        H = config.n_heads

        # Cross-attention multi-cabeza para fusión entre grafos
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = D,
            num_heads   = H,
            dropout     = config.dropout,
            batch_first = True,
        )

        # MLP de fusión post-attention
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D * 2),
            nn.GELU(),
            nn.Linear(D * 2, D),
            nn.LayerNorm(D),
        )

        # LayerNorm de entrada
        self.input_norm = nn.LayerNorm(D)

    def forward(
        self,
        motor_outputs: List[torch.Tensor],   # List of [k_i, D]
    ) -> UnifierOutput:
        """
        Fusiona las representaciones de grafos de múltiples motores.

        Args:
            motor_outputs: lista de tensores [k_i, D], uno por motor activado.
                           Puede contener tensores con k_i = 0 (sin nodos).

        Returns:
            UnifierOutput con tensor unificado [max_output_nodes, D]
        """
        if not motor_outputs:
            raise ValueError("motor_outputs must have at least one tensor")

        # Filtrar motores sin nodos
        valid = [m for m in motor_outputs if m.shape[0] > 0]
        n_motors = len(valid)

        if n_motors == 0:
            # Todos los motores vacíos → ceros
            D = self.config.node_dim
            device = motor_outputs[0].device
            dtype  = motor_outputs[0].dtype
            unified = torch.zeros(self.config.max_output_nodes, D, device=device, dtype=dtype)
            return UnifierOutput(unified=unified, n_source_motors=0, total_nodes=0)

        if n_motors == 1:
            # Un solo motor: identidad + pool a tamaño fijo
            nodes   = valid[0]  # [k, D]
            unified = self._pool_to_fixed_size(nodes)
            return UnifierOutput(
                unified=unified,
                n_source_motors=1,
                total_nodes=nodes.shape[0],
            )

        # Múltiples motores: concatenar → cross-attention → pool
        combined    = torch.cat(valid, dim=0)    # [total, D]
        total_nodes = combined.shape[0]

        # Cross-attention: cada nodo atiende a todos los demás
        # Input shape para MultiheadAttention (batch_first=True): [1, total, D]
        x = self.input_norm(combined).unsqueeze(0)   # [1, total, D]
        attn_out, _ = self.cross_attn(x, x, x)       # [1, total, D]
        attn_out    = attn_out.squeeze(0)             # [total, D]

        # Residual + MLP de fusión
        fused = self.fusion_mlp(attn_out + combined)  # [total, D]

        unified = self._pool_to_fixed_size(fused)

        return UnifierOutput(
            unified=unified,
            n_source_motors=n_motors,
            total_nodes=total_nodes,
        )

    def _pool_to_fixed_size(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        Convierte [N, D] → [max_output_nodes, D] por top-K norma + padding.

        Si N >= max_output_nodes: selecciona los max_output_nodes nodos con
            mayor norma L2 (los más "informativos" semánticamente).
        Si N < max_output_nodes: pad con ceros hasta max_output_nodes.
        """
        N, D  = nodes.shape
        K     = self.config.max_output_nodes
        device = nodes.device
        dtype  = nodes.dtype

        if N >= K:
            top_idx = nodes.norm(dim=-1).topk(K).indices.sort().values
            return nodes[top_idx]                  # [K, D]
        else:
            pad = torch.zeros(K - N, D, device=device, dtype=dtype)
            return torch.cat([nodes, pad], dim=0)  # [K, D]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
