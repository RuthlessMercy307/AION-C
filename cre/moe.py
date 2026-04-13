"""
cre/moe.py — SparseMoE (Sparse Mixture of Experts)
====================================================

Multiplicador de capacidad para el CRE.

POSICIÓN EN EL CRE:
    Por cada iteración del CausalReasoningEngine:

        h_new = MessagePassing(h, edges)      # typed MP + GRUCell interno
        moe_out = SparseMoE(h_new)            # ← ESTE MÓDULO: especialización
        h = GRUCell(moe_out, h)               # ← estabilización cross-iter

    El MoE actúa DESPUÉS del message passing para especializar cada nodo
    según su contenido, antes de que el GRU del engine decida cuánto aceptar.

POR QUÉ MoE EN EL CRE:
    El message passing computa representaciones CONTEXTUALES (nodo + vecinos).
    El MoE especializa la representación según el TIPO SEMÁNTICO del nodo:
        - Nodos de tipo HYPOTHESIS activan expertos de evidencia
        - Nodos de tipo EVENT activan expertos temporales
        - Nodos de tipo CONTRADICTION activan expertos lógicos
    Este routing emergente (aprendido, no hardcodeado) da 16x más capacidad
    con solo 2x más compute por paso.

GRUPOS DE EXPERTOS:
    Los n_groups × experts_per_group expertos se organizan en grupos.
    El router produce logits sobre TODOS los expertos (global top-k).
    La organización en grupos no restringe el routing — es solo estructural
    para mantener el código modular y facilitar el scaling.

SPARSE ROUTING (top-k hard):
    Para cada nodo, el router selecciona los active_experts expertos con
    mayor logit. Solo esos expertos corren para ese nodo.

    Parámetros totales: n_groups × experts_per_group × expert_params
    Parámetros activos: active_experts × expert_params
    Ratio:             (n_groups × experts_per_group) / active_experts

    Config tiny (benchmark):  16 expertos, 2 activos → 8x ratio
    Config 3B (producción):   32 expertos, 2 activos → 16x ratio

LOAD BALANCE LOSS:
    Sin regularización, el router aprende a siempre usar los mismos expertos
    (problema de "expert collapse"). La load balance loss penaliza esto:

        lb_loss = E × Σ_i (fraction_tokens_i × mean_routing_prob_i)

    Donde E es el número de expertos. Esta forma penaliza cuando hay alta
    correlación entre "qué experto recibe muchos tokens" y "qué experto
    tiene alta probabilidad de ser seleccionado".

    Se multiplica por moe_load_balance_weight antes de sumarse a la loss total.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CREConfig


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MoEOutput:
    """
    Resultado del SparseMoE.

    output:             [N, output_dim] — representaciones especializadas
    load_balance_loss:  scalar Tensor   — auxiliar de entrenamiento (≥ 0)
    router_probs:       [N, n_experts]  — probabilidades de routing (post-softmax)
    top_k_indices:      [N, active_k]  — índices de expertos seleccionados
    """
    output:             torch.Tensor
    load_balance_loss:  torch.Tensor
    router_probs:       torch.Tensor
    top_k_indices:      torch.Tensor


# ─────────────────────────────────────────────────────────────────────────────
# EXPERT GROUP
# ─────────────────────────────────────────────────────────────────────────────

class ExpertGroup(nn.Module):
    """
    Grupo de N expertos FFN independientes.

    Cada experto es una pequeña red:
        Linear(input_dim → hidden_dim) → GELU → Linear(hidden_dim → output_dim)

    Los expertos del mismo grupo están en el mismo ModuleList para facilitar
    la inspección, pero se ejecutan de forma independiente (no hay parámetros
    compartidos entre expertos).

    Uso:
        group = ExpertGroup(n_experts=4, input_dim=256, output_dim=256)
        # Ejecutar el experto 2 para 3 nodos:
        out = group.forward_expert(expert_idx=2, x=some_nodes)  # [3, 256]
    """

    def __init__(
        self,
        n_experts:  int,
        input_dim:  int,
        output_dim: int,
        hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        hidden_dim = max(input_dim * hidden_mult, input_dim)

        self.experts: nn.ModuleList = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim, bias=False),
            )
            for _ in range(n_experts)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for expert in self.experts:
            for m in expert.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)

    def forward_expert(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta el experto `expert_idx` sobre `x`.

        Args:
            expert_idx: índice del experto en [0, n_experts)
            x:          [n_nodes, input_dim]

        Returns:
            [n_nodes, output_dim]
        """
        return self.experts[expert_idx](x)

    @property
    def n_experts(self) -> int:
        return len(self.experts)


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE MoE
# ─────────────────────────────────────────────────────────────────────────────

class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts para el CRE.

    Organizado en n_groups grupos, cada uno con experts_per_group expertos.
    El routing es GLOBAL: top-k se selecciona de TODOS los expertos.

    Uso:
        config = CREConfig(use_moe=True, moe_n_groups=4, moe_experts_per_group=4,
                           moe_active_experts=2)
        moe = SparseMoE(config)

        # h_new: output del message passing [N, node_dim]
        result = moe(h_new)
        # result.output:            [N, node_dim]
        # result.load_balance_loss: scalar

    Args:
        config: CREConfig con los hiperparámetros del MoE
    """

    def __init__(self, config: CREConfig) -> None:
        super().__init__()
        D = config.node_dim  # input_dim = output_dim = node_dim

        self.n_groups        = config.moe_n_groups
        self.experts_per_group = config.moe_experts_per_group
        self.n_experts       = config.moe_n_groups * config.moe_experts_per_group
        self.active_experts  = config.moe_active_experts
        self.hidden_mult     = config.moe_expert_hidden_mult
        self.lb_weight       = config.moe_load_balance_weight

        assert self.active_experts <= self.n_experts, (
            f"active_experts ({self.active_experts}) must be <= "
            f"n_experts ({self.n_experts})"
        )

        # ── Grupos de expertos ────────────────────────────────────────────────
        self.groups: nn.ModuleList = nn.ModuleList([
            ExpertGroup(
                n_experts   = config.moe_experts_per_group,
                input_dim   = D,
                output_dim  = D,
                hidden_mult = config.moe_expert_hidden_mult,
            )
            for _ in range(config.moe_n_groups)
        ])

        # ── Router ────────────────────────────────────────────────────────────
        # Proyecta node features → logits sobre TODOS los expertos
        self.router = nn.Linear(D, self.n_experts, bias=False)

        # ── Proyección de salida (residual + output norm) ─────────────────────
        self.out_norm = nn.LayerNorm(D, eps=config.norm_eps)

        nn.init.normal_(self.router.weight, std=0.02)

    # ── Indexing helpers ──────────────────────────────────────────────────────

    def _expert_group_and_local(self, expert_idx: int):
        """Returns (group_idx, local_expert_idx) for a global expert index."""
        g = expert_idx // self.experts_per_group
        e = expert_idx %  self.experts_per_group
        return g, e

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> MoEOutput:
        """
        Sparse MoE forward.

        Args:
            x: [N, node_dim] — node features (output del message passing)

        Returns:
            MoEOutput con output [N, node_dim], load_balance_loss scalar,
            router_probs [N, n_experts], top_k_indices [N, active_k]
        """
        N, D = x.shape
        E    = self.n_experts
        K    = self.active_experts

        # ── 1. Router: logits sobre todos los expertos ───────────────────────
        router_logits = self.router(x)                    # [N, E]
        router_probs  = F.softmax(router_logits, dim=-1)  # [N, E], suma 1

        # ── 2. Top-k selection ────────────────────────────────────────────────
        top_k_weights, top_k_indices = torch.topk(router_probs, K, dim=-1)
        # top_k_weights: [N, K] — probabilidades de los K expertos seleccionados
        # top_k_indices: [N, K] — índices globales (0..E-1)

        # Re-normalizar para que los pesos de los K expertos sumen 1
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # ── 3. Sparse expert execution ────────────────────────────────────────
        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)

        for e_global in range(E):
            # Nodos que seleccionaron este experto (en alguna de sus K posiciones)
            expert_mask = (top_k_indices == e_global)          # [N, K] bool
            if not expert_mask.any():
                continue

            node_indices, k_positions = expert_mask.nonzero(as_tuple=True)
            if node_indices.numel() == 0:
                continue

            # Ejecutar el experto
            g, local_e = self._expert_group_and_local(e_global)
            expert_input  = x[node_indices]                    # [n_assigned, D]
            expert_output = self.groups[g].forward_expert(local_e, expert_input)
            # [n_assigned, D]

            # Pesos de cada nodo para este experto
            weights = top_k_weights[node_indices, k_positions].unsqueeze(-1)
            # [n_assigned, 1]

            output.index_add_(0, node_indices, weights * expert_output)

        # ── 4. Residual + norm ────────────────────────────────────────────────
        output = self.out_norm(x + output)

        # ── 5. Load balance loss ──────────────────────────────────────────────
        # Penaliza si los mismos expertos siempre reciben todos los tokens.
        # Basado en: Fedus et al. "Switch Transformers" (2021).
        #
        # fraction_tokens[e] = fracción de nodos asignados al experto e
        # mean_prob[e]        = probabilidad media de routing al experto e
        # lb_loss = E × Σ_e (fraction_tokens[e] × mean_prob[e])
        # Mínimo cuando la distribución es uniforme: lb_loss = 1.0
        # La loss se multiplica por lb_weight antes de sumarse al total.

        with torch.no_grad():
            # Contar asignaciones (flat sobre N×K)
            flat_idx   = top_k_indices.reshape(-1)             # [N*K]
            counts     = torch.zeros(E, device=x.device, dtype=router_probs.dtype)
            counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=router_probs.dtype))
        frac_tokens = counts / (N * K + 1e-9)                 # [E], suma ≈ 1

        mean_probs  = router_probs.mean(dim=0)                 # [E]

        # frac_tokens necesita gradiente solo a través de mean_probs
        # (los índices top-k no son diferenciables — eso está bien)
        lb_loss = self.lb_weight * (E * (frac_tokens * mean_probs).sum())

        return MoEOutput(
            output            = output,
            load_balance_loss = lb_loss,
            router_probs      = router_probs.detach(),
            top_k_indices     = top_k_indices.detach(),
        )

    # ── Utilidades ────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def expert_load(self, top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        Calcula la carga de cada experto dado un batch de routing decisions.

        Args:
            top_k_indices: [N, K] — índices de expertos seleccionados

        Returns:
            [n_experts] — fracción de nodos asignados a cada experto
        """
        N, K = top_k_indices.shape
        flat = top_k_indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.n_experts).float()
        return counts / (N * K)
