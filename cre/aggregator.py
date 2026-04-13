"""
cre/aggregator.py — AttentiveAggregator
========================================

Agrega mensajes entrantes en cada nodo, ponderando por importancia aprendida.

POR QUÉ ATTENTIVE Y NO MEAN:
    Mean aggregation trata todos los mensajes como igualmente importantes.
    Pero en razonamiento causal:
        - Un mensaje de CAUSES fuerte es más relevante que uno de CORRELATES débil
        - Un mensaje que confirma el estado actual es menos urgente que uno que lo contradice
        - El nodo destino debería poder "filtrar" mensajes irrelevantes

    El AttentiveAggregator aprende a asignar un peso σ ∈ (0,1) a cada mensaje
    basado en (contenido del mensaje, estado actual del nodo destino).

IMPLEMENTACIÓN:
    Para cada mensaje entrante:
        score = sigmoid(MLP([mensaje, estado_nodo_destino]))
        peso  = score  (no softmax — ver nota abajo)

    Luego: agregado = sum(peso × mensaje) / (sum(peso) + ε)

    Por qué sigmoid y no softmax:
        Softmax normaliza sobre los mensajes de UN nodo → requiere per-target segment softmax
        (no disponible directamente en PyTorch sin torch_scatter).
        Sigmoid da pesos independientes → la suma normalizada por count/sum_weights
        es equivalente a una atención suave.
        AMBAS aprenden a ponderar mensajes por importancia — la diferencia es que
        sigmoid puede silenciar todos los mensajes (score → 0) si son irrelevantes.

SCATTER PATTERN:
    Los mensajes están indexados por arista (no por nodo):
        messages:       [E, message_dim]
        target_indices: [E]  — índice del nodo destino de cada mensaje

    Scatter add: para cada nodo i, agrega mensajes donde target_indices == i.
    Implementado con torch.scatter_add_ (disponible en PyTorch ≥ 1.7).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import CREConfig


class AttentiveAggregator(nn.Module):
    """
    Agrega mensajes entrantes ponderados por importancia aprendida.

    Uso:
        cfg  = CREConfig()
        agg  = AttentiveAggregator(cfg)
        # messages: [E, M], targets: [E], nodes: [N, D]
        out  = agg(messages, target_indices, node_features, n_nodes=N)
        # out: [N, M]
    """

    def __init__(self, config: CREConfig) -> None:
        super().__init__()
        M = config.message_dim
        D = config.node_dim

        # Puntúa la importancia de cada mensaje dado el estado del nodo destino.
        # Input: concatenación de [mensaje, estado_destino] → un escalar en (0,1)
        self.attn_scorer = nn.Sequential(
            nn.Linear(M + D, M // 2),
            nn.GELU(),
            nn.Linear(M // 2, 1, bias=False),
        )

        # Normalización post-agregación para estabilidad
        self.norm = nn.LayerNorm(M, eps=config.norm_eps)

        self._message_dim = M
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        messages:       torch.Tensor,  # [E, message_dim]
        target_indices: torch.Tensor,  # [E]  long tensor
        node_features:  torch.Tensor,  # [N, node_dim]
        n_nodes:        int,
    ) -> torch.Tensor:
        """
        Args:
            messages:       [E, M] — uno por arista del grafo
            target_indices: [E]    — a qué nodo va cada mensaje
            node_features:  [N, D] — estado actual de los nodos (como query)
            n_nodes:        int    — N (necesario cuando E=0)

        Returns:
            aggregated: [N, M] — mensajes agregados por nodo
                        zeros para nodos sin mensajes entrantes
        """
        M = self._message_dim
        device = node_features.device
        dtype  = node_features.dtype

        # ── Caso sin aristas ──────────────────────────────────────────────────
        if messages.shape[0] == 0:
            return torch.zeros(n_nodes, M, device=device, dtype=dtype)

        E = messages.shape[0]

        # ── 1. Atención: puntúa cada mensaje vs. estado del nodo destino ─────
        # target_features: [E, D] — estado del nodo que RECIBIRÁ cada mensaje
        target_feats = node_features[target_indices]          # [E, D]
        attn_input   = torch.cat([messages, target_feats], dim=-1)  # [E, M+D]
        raw_scores   = self.attn_scorer(attn_input).squeeze(-1)     # [E]
        weights      = torch.sigmoid(raw_scores)                    # [E] ∈ (0,1)

        # ── 2. Mensajes ponderados ────────────────────────────────────────────
        weighted = messages * weights.unsqueeze(-1)           # [E, M]

        # ── 3. Acumular mensajes por nodo destino ────────────────────────────
        # aggregated[i] = Σ weighted[e] para todo e con target_indices[e] == i
        #
        # Usamos index_add_ en lugar de scatter_add_ porque scatter_add_ 2D
        # no está soportado en torch-directml (backend DirectML / AMD / Intel).
        # index_add_ es equivalente y diferenciable en CPU, CUDA y DML.
        #
        # Use weighted.dtype so index_add_ dtype matches under AMP (FP16/FP32).
        aggregated = torch.zeros(n_nodes, M, device=device, dtype=weighted.dtype)
        aggregated.index_add_(0, target_indices, weighted)

        # ── 4. Normalizar por suma de pesos (en lugar de count) ──────────────
        # sum_weights[i] = Σ weights[e] para todo e con target_indices[e] == i
        sum_weights = torch.zeros(n_nodes, device=device, dtype=weights.dtype)
        sum_weights.index_add_(0, target_indices, weights)
        aggregated   = aggregated / (sum_weights.unsqueeze(-1) + 1e-8)

        # ── 5. LayerNorm para estabilidad numérica ───────────────────────────
        return self.norm(aggregated)                          # [N, M]
