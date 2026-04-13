"""
cre/message_passing.py — CausalMessagePassingLayer
===================================================

Una iteración de typed message passing sobre un CausalGraph.

TYPED MESSAGE PASSING (TMP):
    El message passing estándar usa UNA función de mensaje para todas las aristas.
    TMP usa UNA función por TIPO de arista — 16 funciones para 16 CausalRelations.

    ¿Por qué importa esto?
        "A CAUSES B":   el mensaje debe decir "si A ocurre, B también ocurrirá"
        "A PREVENTS B": el mensaje debe decir "si A ocurre, B NO ocurrirá"

        Con una función única, ambos mensajes serían transformaciones del mismo espacio.
        Con funciones distintas, CAUSES puede aprender transformaciones ATRACTIVAS
        y PREVENTS puede aprender transformaciones REPULSIVAS.

        Esta tensión atracción-repulsión es lo que previene over-smoothing en el GNN:
        los nodos no convergen al mismo valor porque las contradicciones los separan.

PIPELINE POR CAPA:
    1. compute_messages()    — [E, M]: una función por tipo de relación
    2. AttentiveAggregator   — [N, M]: pondera mensajes por importancia
    3. GRUCell               — [N, D]: actualiza estado del nodo (con gate de olvido)
    4. edge_updater          — [E, edge_dim]: actualiza representaciones de aristas
    5. LayerNorm             — estabilidad numérica en nodos y aristas

GRU COMO NODE UPDATER:
    ¿Por qué GRU y no MLP directo?
    - El GRU tiene gate de olvido: puede mantener creencias fuertes ante mensajes débiles
    - El GRU tiene gate de actualización: absorbe mensajes cuando son más informativos
    - Es naturalmente estable para muchas iteraciones (los gates amortiguan explosiones)

EDGE UPDATER:
    Las aristas también tienen estado que evoluciona.
    El edge updater toma (src_features, tgt_features, edge_features) → nuevas edge_features.
    Esto permite que las aristas "aprendan" qué tan activa está la relación en el contexto actual.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from core.graph import CAUSAL_RELATIONS, CausalGraph, CausalRelation
from .aggregator import AttentiveAggregator
from .config import CREConfig


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL GRU CELL
# ─────────────────────────────────────────────────────────────────────────────

class ManualGRUCell(nn.Module):
    """
    Drop-in replacement for nn.GRUCell using only basic tensor ops.

    nn.GRUCell uses the fused kernel 'aten::_thnn_fused_gru_cell' which is
    NOT supported on torch-directml (DirectML backend). This implementation
    uses only matmul + sigmoid + tanh, all supported on DML, CUDA, and CPU.

    Mathematically identical to nn.GRUCell:
        r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)   # reset gate
        z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)   # update gate
        n = tanh(W_in * x + b_in + r * (W_hn * h + b_hn))# candidate
        h' = (1 - z) * n + z * h
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        # Match nn.GRUCell parameter names and layout exactly
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming uniform init — same as nn.GRUCell default."""
        nn.init.kaiming_uniform_(self.weight_ih, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.weight_hh, a=5 ** 0.5)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, input_size]
            h: [N, hidden_size]
        Returns:
            h_new: [N, hidden_size]
        """
        H = self.hidden_size
        gates_x = x  @ self.weight_ih.t()  # [N, 3H]
        gates_h = h  @ self.weight_hh.t()  # [N, 3H]
        if self.bias_ih is not None:
            gates_x = gates_x + self.bias_ih
            gates_h = gates_h + self.bias_hh

        r = torch.sigmoid(gates_x[:, :H]    + gates_h[:, :H])      # reset
        z = torch.sigmoid(gates_x[:, H:2*H] + gates_h[:, H:2*H])   # update
        n = torch.tanh(   gates_x[:, 2*H:]  + r * gates_h[:, 2*H:])# candidate
        return (1.0 - z) * n + z * h


# ─────────────────────────────────────────────────────────────────────────────
# EDGE UPDATER (auxiliar)
# ─────────────────────────────────────────────────────────────────────────────

class _EdgeUpdater(nn.Module):
    """
    Actualiza las features de aristas dado el estado actualizado de los nodos.

    Residual + LayerNorm para estabilidad.
    """

    def __init__(self, config: CREConfig) -> None:
        super().__init__()
        D = config.node_dim
        E = config.edge_dim

        self.mlp = nn.Sequential(
            nn.Linear(D * 2 + E, E),
            nn.GELU(),
            nn.Linear(E, E),
        )
        self.norm = nn.LayerNorm(E, eps=config.norm_eps)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        src_feats:     torch.Tensor,  # [E_count, node_dim]
        tgt_feats:     torch.Tensor,  # [E_count, node_dim]
        edge_features: torch.Tensor,  # [E_count, edge_dim]
    ) -> torch.Tensor:
        """Returns: [E_count, edge_dim]"""
        if edge_features.shape[0] == 0:
            return edge_features
        inp    = torch.cat([src_feats, tgt_feats, edge_features], dim=-1)  # [E, D*2+edge_dim]
        update = self.mlp(inp)
        return self.norm(edge_features + update)                           # residual + norm


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL MESSAGE PASSING LAYER
# ─────────────────────────────────────────────────────────────────────────────

class CausalMessagePassingLayer(nn.Module):
    """
    Una capa de typed message passing causal.

    Esta capa se instancia N_LAYERS veces y se COMPARTE entre todas las iteraciones.
    El CausalReasoningEngine la llama en bucle: mismos pesos, estado del grafo evoluciona.

    Parámetros:
        message_fns:  nn.ModuleDict — una nn.Sequential por CausalRelation (16 total)
        aggregator:   AttentiveAggregator — pondera mensajes por importancia
        node_updater: nn.GRUCell — actualiza estado del nodo con gate de olvido
        edge_updater: _EdgeUpdater — actualiza representación de las aristas
        node_norm:    nn.LayerNorm — normaliza node features post-update
    """

    def __init__(self, config: CREConfig, relation_keys: Optional[List[str]] = None) -> None:
        super().__init__()
        D = config.node_dim
        E = config.edge_dim
        M = config.message_dim

        # ── Una función de mensaje por tipo de relación ──────────────────────
        # Input: [source_features, target_features, edge_features]
        #        = [node_dim, node_dim, edge_dim] → node_dim*2 + edge_dim
        # Output: [message_dim]
        # relation_keys: lista de strings de relación (default: CAUSAL_RELATIONS).
        # Pasa relation_keys distintos para motores con vocabulario de relaciones propio.
        _relation_keys = relation_keys if relation_keys is not None else CAUSAL_RELATIONS
        msg_input_dim = D * 2 + E
        self.message_fns: nn.ModuleDict = nn.ModuleDict({
            rel: nn.Sequential(
                nn.Linear(msg_input_dim, M),
                nn.GELU(),
                nn.Linear(M, M),
            )
            for rel in _relation_keys
        })

        # ── Agregador atencional (scatter-based) ─────────────────────────────
        self.aggregator = AttentiveAggregator(config)

        # ── GRU para actualización de nodo ───────────────────────────────────
        # Input: messages agregados [M]
        # Hidden: estado actual del nodo [D]
        # Output: nuevo estado [D]
        # ManualGRUCell: misma matemática que nn.GRUCell pero usa solo ops básicos
        # (matmul+sigmoid+tanh), compatible con torch-directml (DML).
        self.node_updater = ManualGRUCell(M, D)

        # ── Actualizador de aristas ───────────────────────────────────────────
        self.edge_updater = _EdgeUpdater(config)

        # ── Normalización post-update ─────────────────────────────────────────
        self.node_norm = nn.LayerNorm(D, eps=config.norm_eps)

        self.config = config
        self._init_weights()

    def _init_weights(self) -> None:
        # message_fns: std normal para ambas capas
        for fn in self.message_fns.values():
            for m in fn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # GRUCell: usa init por defecto de PyTorch (kaiming uniform)
        # node_norm: init por defecto (weight=1, bias=0)

    def forward(
        self,
        node_features: torch.Tensor,  # [N, node_dim]
        edge_features: torch.Tensor,  # [E, edge_dim]
        graph:         CausalGraph,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Una pasada completa de message passing.

        Args:
            node_features: [N, node_dim] — estado actual de los nodos
            edge_features: [E, edge_dim] — estado actual de las aristas
            graph:         CausalGraph   — estructura (source_idx, target_idx, relation)

        Returns:
            new_node_features: [N, node_dim]
            new_edge_features: [E, edge_dim]
        """
        N = node_features.shape[0]
        device = node_features.device

        # ── 0. Caso sin aristas ───────────────────────────────────────────────
        # Los nodos no reciben mensajes → GRU con mensaje cero → estado estable
        if len(graph.edges) == 0:
            zero_msgs = torch.zeros(N, self.config.message_dim, device=device,
                                    dtype=node_features.dtype)
            new_nodes = self.node_updater(zero_msgs, node_features)
            new_nodes = self.node_norm(new_nodes)
            return new_nodes, edge_features

        E_count = len(graph.edges)

        # ── 1. Pre-recopilar índices de fuente y destino ─────────────────────
        src_idx = torch.tensor(
            [e.source_idx for e in graph.edges], dtype=torch.long, device=device
        )
        tgt_idx = torch.tensor(
            [e.target_idx for e in graph.edges], dtype=torch.long, device=device
        )
        src_feats = node_features.index_select(0, src_idx)   # [E, D]
        tgt_feats = node_features.index_select(0, tgt_idx)   # [E, D]

        # ── 2. Calcular mensajes por tipo de relación ─────────────────────────
        # Agrupa aristas por tipo para cómputo batched (más eficiente que loop por arista)
        messages = self._compute_messages(
            src_feats, tgt_feats, edge_features, graph
        )                                            # [E, M]

        # ── 3. Agregar mensajes en nodos destino ──────────────────────────────
        aggregated = self.aggregator(
            messages    = messages,
            target_indices = tgt_idx,
            node_features  = node_features,
            n_nodes        = N,
        )                                            # [N, M]

        # ── 4. Actualizar nodos con GRU ───────────────────────────────────────
        # GRUCell(input=aggregated, hx=current_state)
        # El gate de olvido preserva lo que ya se sabe; el de actualización absorbe novedades
        new_nodes = self.node_updater(aggregated, node_features)  # [N, D]
        new_nodes = self.node_norm(new_nodes)

        # ── 5. Actualizar aristas ─────────────────────────────────────────────
        # Usa los nodos YA actualizados para que las aristas reflejen el nuevo estado
        updated_src = new_nodes.index_select(0, src_idx)             # [E, D]
        updated_tgt = new_nodes.index_select(0, tgt_idx)             # [E, D]
        new_edges   = self.edge_updater(updated_src, updated_tgt, edge_features)  # [E, edge_dim]

        return new_nodes, new_edges

    # ── Cómputo de mensajes por tipo de relación ──────────────────────────────

    def forward_tensors(
        self,
        node_features: torch.Tensor,   # [N_total, node_dim]
        edge_features: torch.Tensor,   # [E_total, edge_dim]
        src_idx:       torch.Tensor,   # [E_total] long
        tgt_idx:       torch.Tensor,   # [E_total] long
        edge_rel_vals: List[str],      # E_total relation-type strings
        n_nodes:       int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tensor-only forward para batch processing.
        Misma lógica que forward() pero acepta tensores pre-construidos
        en lugar de un CausalGraph — permite procesar múltiples grafos
        en un único forward concatenando sus nodos con offsets.
        """
        device = node_features.device

        if edge_features.shape[0] == 0:
            zero_msgs = torch.zeros(n_nodes, self.config.message_dim,
                                    device=device, dtype=node_features.dtype)
            new_nodes = self.node_updater(zero_msgs, node_features)
            new_nodes = self.node_norm(new_nodes)
            return new_nodes, edge_features

        src_feats = node_features.index_select(0, src_idx)   # [E, D]
        tgt_feats = node_features.index_select(0, tgt_idx)   # [E, D]

        messages = self._compute_messages_tensors(
            src_feats, tgt_feats, edge_features, edge_rel_vals
        )

        aggregated = self.aggregator(
            messages       = messages,
            target_indices = tgt_idx,
            node_features  = node_features,
            n_nodes        = n_nodes,
        )

        new_nodes = self.node_updater(aggregated, node_features)
        new_nodes = self.node_norm(new_nodes)

        updated_src = new_nodes.index_select(0, src_idx)
        updated_tgt = new_nodes.index_select(0, tgt_idx)
        new_edges   = self.edge_updater(updated_src, updated_tgt, edge_features)

        return new_nodes, new_edges

    def _compute_messages(
        self,
        src_feats:     torch.Tensor,  # [E, node_dim]
        tgt_feats:     torch.Tensor,  # [E, node_dim]
        edge_features: torch.Tensor,  # [E, edge_dim]
        graph:         CausalGraph,
    ) -> torch.Tensor:
        """
        Calcula un mensaje por arista usando la función del tipo de relación correspondiente.

        Agrupa aristas por tipo → batch computation por relación → reensambla en orden original.

        Returns: [E, message_dim]
        """
        rel_vals = [edge.relation.value for edge in graph.edges]
        return self._compute_messages_tensors(src_feats, tgt_feats, edge_features, rel_vals)

    def _compute_messages_tensors(
        self,
        src_feats:     torch.Tensor,   # [E, node_dim]
        tgt_feats:     torch.Tensor,   # [E, node_dim]
        edge_features: torch.Tensor,   # [E, edge_dim]
        edge_rel_vals: List[str],      # E relation-type strings
    ) -> torch.Tensor:
        """
        Versión tensor-pura de _compute_messages.
        Acepta una lista de strings de tipo de relación en lugar de CausalGraph.

        Returns: [E, message_dim]
        """
        device = src_feats.device
        M      = self.config.message_dim
        E      = src_feats.shape[0]

        rel_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, rel_val in enumerate(edge_rel_vals):
            rel_to_indices[rel_val].append(i)

        # Accumulate all indices and messages across relation types, then
        # reorder with a single index_select — ONE GatherBackward instead of
        # E SliceBackward0 ops (E SliceBackward0 per call was the bottleneck).
        all_idx_list:  List[torch.Tensor] = []
        all_msgs_list: List[torch.Tensor] = []
        for rel_val, indices in rel_to_indices.items():
            idx_t = torch.tensor(indices, dtype=torch.long, device=device)
            fn    = self.message_fns[rel_val]
            inp   = torch.cat([src_feats.index_select(0, idx_t),
                               tgt_feats.index_select(0, idx_t),
                               edge_features.index_select(0, idx_t)], dim=-1)
            msgs  = fn(inp)             # [k, M]
            all_idx_list.append(idx_t)
            all_msgs_list.append(msgs)

        all_positions = torch.cat(all_idx_list)           # [E] — permutation of [0..E-1]
        all_computed  = torch.cat(all_msgs_list, dim=0)   # [E, M]
        inv_perm      = torch.argsort(all_positions)      # inverse permutation
        return all_computed.index_select(0, inv_perm)     # [E, M] — one GatherBackward
