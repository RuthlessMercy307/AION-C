"""
decoder/model.py — StreamDecoder
==================================

El decodificador autoregresivo de AION-C condicionado en el grafo causal.

ARQUITECTURA:
    token_ids [B, L]
        ↓ token_embedding + pos_embedding
    x [B, L, D]
        ↓ HybridDecoderLayer × n_layers
          (MambaLayer + cross-attn → grafo + GatedFFN)
    x [B, L, D]
        ↓ final_norm
    ┌─────────────────────────────────────────────┐
    │  lm_head [B, L, vocab_size]   → token logits│
    │  anchor_head [B, L, max_nodes]→ anchor logits│
    │  meta_head → confidence, needs_clarification │
    └─────────────────────────────────────────────┘

CONDITIONING EN EL GRAFO + ENCODER:
    graph_repr       [B, n_nodes, node_dim] — viene del CausalReasoningEngine.
    encoder_concepts [B, L_enc, hidden_dim] — concept vectors del StreamEncoder.
    En cada HybridDecoderLayer, los tokens hacen cross-attention a ambos:
        1. graph_repr       → ancla semántica causal (estructura)
        2. encoder_concepts → ancla identidad léxica (tokens del input)
    Sin encoder_concepts, el decoder puede generar "incendio" cuando el input
    dice "fiebre" porque la identidad léxica se pierde en crystallizer → CRE.

ANCHOR HEAD:
    Predice a qué nodo del grafo "pertenece" cada token generado.
    Durante el entrenamiento se usa como señal auxiliar para forzar
    que los tokens sean fieles al razonamiento causal.

OUTPUT META HEAD:
    Predice metadatos del output:
        confidence:          ¿qué tan seguro está el modelo?
        needs_clarification: ¿necesita pedir más información?

WEIGHT TYING:
    lm_head.weight = token_embedding.weight (práctica estándar en LMs).
    Reduce parámetros y mejora convergencia.

POSICIONES:
    Tabla de embeddings aprendida de tamaño max_seq_len.
    Más simple que RoPE/ALiBi, suficiente para el scope de AION-C.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint as _ckpt

from .config import StreamDecoderConfig
from .hybrid_layer import HybridDecoderLayer
from .meta_head import MetaOutput, OutputMetaHead


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecoderOutput:
    """
    Resultado del StreamDecoder.

    logits:               [B, L, vocab_size]    — distribución sobre vocabulario
    anchor_logits:        [B, L, max_graph_nodes] — qué nodo del grafo produce cada token
    confidence:           [B] ∈ [0,1]          — confianza del modelo
    needs_clarification:  [B] ∈ [0,1]          — solicitar más info al usuario
    """
    logits:              torch.Tensor
    anchor_logits:       torch.Tensor
    confidence:          torch.Tensor
    needs_clarification: torch.Tensor


# ─────────────────────────────────────────────────────────────────────────────
# STREAM DECODER
# ─────────────────────────────────────────────────────────────────────────────

class StreamDecoder(nn.Module):
    """
    Decodificador autoregresivo condicionado en el grafo causal.

    Uso:
        config  = StreamDecoderConfig()
        decoder = StreamDecoder(config)

        # graph_repr del CausalReasoningEngine (CREOutput.node_features)
        token_ids  = torch.randint(0, config.vocab_size, (2, 16))
        graph_repr = torch.randn(2, 8, config.node_dim)

        out = decoder(token_ids, graph_repr)
        # out.logits:         [2, 16, 32000]
        # out.anchor_logits:  [2, 16, 32]
        # out.confidence:     [2]
    """

    def __init__(self, config: StreamDecoderConfig) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim

        # ── Embeddings ────────────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(config.vocab_size, D)
        self.pos_embedding   = nn.Embedding(config.max_seq_len, D)

        # ── Capas híbridas ────────────────────────────────────────────────────
        self.layers: nn.ModuleList = nn.ModuleList([
            HybridDecoderLayer(config)
            for _ in range(config.n_layers)
        ])

        # ── Normalización final ───────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(D, eps=config.norm_eps)

        # ── Heads de salida ───────────────────────────────────────────────────
        # LM head: predice el siguiente token
        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)

        # Anchor head: qué nodo del grafo "produce" cada token
        self.anchor_head = nn.Linear(D, config.max_graph_nodes)

        # Meta head: confianza y necesidad de clarificación
        self.meta_head = OutputMetaHead(config)

        # ── Weight tying ──────────────────────────────────────────────────────
        # token_embedding.weight [vocab_size, D] == lm_head.weight [vocab_size, D]
        # Práctica estándar en LMs: reduce parámetros y mejora convergencia.
        self.lm_head.weight = self.token_embedding.weight

        self.gradient_checkpointing = False
        self._init_weights()

    def enable_gradient_checkpointing(self) -> None:
        """
        Activa gradient checkpointing en las capas HybridDecoderLayer.
        Recomputa las activaciones durante el backward en lugar de guardarlas,
        reduciendo el uso de VRAM del backward ~2-3× a costa de ~15% más compute.
        Solo activo en modo training.
        """
        self.gradient_checkpointing = True

    def _init_weights(self) -> None:
        """Inicialización estándar para embeddings y heads."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        # lm_head.weight es el mismo tensor que token_embedding.weight (tying)
        nn.init.normal_(self.anchor_head.weight, std=0.02)
        nn.init.zeros_(self.anchor_head.bias)

    def forward(
        self,
        token_ids:        torch.Tensor,              # [B, L]
        graph_repr:       torch.Tensor,              # [B, n_nodes, node_dim]
        encoder_concepts: torch.Tensor | None = None,  # [B, L_enc, hidden_dim]
    ) -> DecoderOutput:
        """
        Pasa por todas las capas y genera logits para cada posición.

        Args:
            token_ids:        [B, L]              — índices de tokens (teacher-forced)
            graph_repr:       [B, n_nodes, D]     — representación del grafo causal
            encoder_concepts: [B, L_enc, D]       — concept vectors del StreamEncoder
                              (optional; when None, encoder cross-attention is skipped)

        Returns:
            DecoderOutput con logits [B, L, vocab_size] y metadatos
        """
        B, L = token_ids.shape

        # Positions: [1, L] — broadcast sobre el batch
        pos = torch.arange(L, device=token_ids.device, dtype=torch.long).unsqueeze(0)

        # Combinar token embedding + positional embedding
        x = self.token_embedding(token_ids) + self.pos_embedding(pos)  # [B, L, D]

        # Pasar por todas las HybridDecoderLayers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = _ckpt(layer, x, graph_repr, encoder_concepts,
                          use_reentrant=False)
            else:
                x = layer(x, graph_repr, encoder_concepts)   # [B, L, D]

        # Normalización final
        x = self.final_norm(x)         # [B, L, D]

        # Heads de salida
        logits        = self.lm_head(x)        # [B, L, vocab_size]
        anchor_logits = self.anchor_head(x)    # [B, L, max_graph_nodes]
        meta          = self.meta_head(x, graph_repr)

        return DecoderOutput(
            logits              = logits,
            anchor_logits       = anchor_logits,
            confidence          = meta.confidence,
            needs_clarification = meta.needs_clarification,
        )

    # ── Utilidades ────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Número total de parámetros entrenables."""
        # token_embedding y lm_head comparten pesos → contar una sola vez
        seen: set = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                if p.requires_grad:
                    total += p.numel()
        return total

    def parameter_breakdown(self) -> dict:
        """Desglose de parámetros por sub-módulo (sin doble-contar weight tying)."""
        embed = self.token_embedding.weight.numel()
        pos   = self.pos_embedding.weight.numel()
        layers_total = sum(
            p.numel() for layer in self.layers for p in layer.parameters()
        )
        return {
            "token_embedding":   embed,
            "pos_embedding":     pos,
            "layers":            layers_total,
            "lm_head":           0,  # tied con token_embedding
            "anchor_head":       sum(p.numel() for p in self.anchor_head.parameters()),
            "meta_head":         sum(p.numel() for p in self.meta_head.parameters()),
            "total_unique":      self.count_parameters(),
        }
