"""
encoder/model.py — StreamEncoder: tokens → concept vectors
===========================================================

El StreamEncoder es el primer módulo del pipeline AION-C CEN:

  token_ids [B, L]
       ↓
  token_embedding [B, L, D]
       ↓
  MambaLayer × n_layers
       ↓
  RMSNorm
       ↓
  concept_projector [B, L, concept_dim]
       ↓
  concept_vectors → GraphConstructor

Propiedad clave: memoria O(L), no O(L²).
El scan secuencial del SSM mantiene un estado h[B, D_inner, N]
(tamaño constante) mientras procesa cada token. La información
del pasado queda comprimida en ese vector de estado.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint as _ckpt

from .mamba_layer import GatedFFN, MambaLayer, RMSNorm, StreamEncoderConfig


# ─────────────────────────────────────────────────────────────────────────────
# STREAM ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class StreamEncoder(nn.Module):
    """
    Convierte token IDs en concept vectors usando Mamba-style SSM.

    El concepto de "espacio conceptual" (concept_dim < hidden_dim):
      "the quick brown fox" = 4 tokens → 2-3 conceptos.
      La compresión elimina información léxica superficial,
      dejando solo la semántica relevante para el CEC.
      El CEC opera en concept_dim=128d, no en hidden_dim=256d.
      → 4x menos compute por operación en el razonador.

    Configuración tiny (testing):
      vocab_size=32000, hidden_dim=256, n_layers=4,
      state_dim=16, concept_dim=128
      Parámetros totales: ~13M

    Uso:
        config  = StreamEncoderConfig()
        encoder = StreamEncoder(config)
        ids     = torch.randint(0, config.vocab_size, (2, 512))
        vecs    = encoder(ids)   # [2, 512, 128]
    """

    def __init__(self, config: StreamEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # Embedding de tokens
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Stack de capas Mamba
        self.layers = nn.ModuleList([
            MambaLayer(config) for _ in range(config.n_layers)
        ])

        # Normalización final
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_eps)

        # Proyección al espacio conceptual (hidden_dim → concept_dim)
        # Sin bias — la información viene del embedding + SSM
        self.concept_projector = nn.Linear(
            config.hidden_dim, config.concept_dim, bias=False
        )

        self.gradient_checkpointing = False
        self._init_weights()

    # ── Inicialización ───────────────────────────────────────────────────────

    def enable_gradient_checkpointing(self) -> None:
        """
        Activa gradient checkpointing en las capas MambaLayer.
        Recomputa las activaciones durante el backward en lugar de guardarlas,
        reduciendo el uso de VRAM del backward ~2-3× a costa de ~15% más compute.
        Solo activo en modo training.
        """
        self.gradient_checkpointing = True

    def _init_weights(self) -> None:
        """
        Inicialización estándar para LLMs:
        - Embedding: normal(std=0.02)
        - concept_projector: normal(std=0.02 / sqrt(2 * n_layers))
          El factor 1/sqrt(2L) reduce la varianza de los residuals
          al acumular L capas (similar a GPT-2).
        """
        std_emb  = 0.02
        std_proj = 0.02 / (2 * self.config.n_layers) ** 0.5

        nn.init.normal_(self.token_embedding.weight, std=std_emb)
        nn.init.normal_(self.concept_projector.weight, std=std_proj)

    # ── Forward principal ────────────────────────────────────────────────────

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] — long tensor de IDs de tokens

        Returns:
            concept_vectors: [B, L, concept_dim]
              Cada posición l contiene el concepto extraído del contexto
              [0..l] (gracias al scan causal del SSM).
        """
        x = self.token_embedding(token_ids)  # [B, L, D]

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, _ = _ckpt(layer, x, use_reentrant=False)
            else:
                x, _ = layer(x)

        x = self.norm(x)
        return self.concept_projector(x)     # [B, L, concept_dim]

    # ── Forward con inspección interna ───────────────────────────────────────

    def forward_with_states(
        self,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Igual que forward() pero retorna también la trayectoria de estados SSM.

        Uso:
            concepts, states = encoder.forward_with_states(ids)
            # states[i]: [B, L, D_inner, N] — estados de la capa i
            # states[i][:, t, :, :] — estado en el timestep t, capa i

        Util para:
            - Verificar que el estado SSM cambia entre timesteps
            - Debugging del scan
            - Visualización de qué información retiene el estado
        """
        x = self.token_embedding(token_ids)
        all_states: List[torch.Tensor] = []

        for layer in self.layers:
            x, states = layer(x, return_states=True)
            all_states.append(states)   # states: [B, L, D_inner, N]

        x = self.norm(x)
        concepts = self.concept_projector(x)
        return concepts, all_states

    def get_ssm_A_bar_numel(self, token_ids: torch.Tensor) -> int:
        """
        Ejecuta un forward y retorna el número de elementos del tensor A_bar
        computado en la primera capa SSM.

        Usado en tests de escalado de memoria:
          A_bar.numel() = B × L × D_inner × N  ← lineal en L, NO cuadrático

        Devuelve siempre el valor de la PRIMERA capa (layers[0].ssm).
        """
        with torch.no_grad():
            self.forward(token_ids)
        return self.layers[0].ssm._last_A_bar_numel

    # ── Utilidades ────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> Dict[str, int]:
        """Desglose de parámetros por módulo."""
        return {
            "token_embedding": self.token_embedding.weight.numel(),
            "layers_ssm":      sum(
                p.numel()
                for layer in self.layers
                for p in layer.ssm.parameters()
            ),
            "layers_ffn":      sum(
                p.numel()
                for layer in self.layers
                for p in layer.ffn.parameters()
            ),
            "layers_norms":    sum(
                p.numel()
                for layer in self.layers
                for p in list(layer.norm1.parameters()) + list(layer.norm2.parameters())
            ),
            "concept_projector": self.concept_projector.weight.numel(),
            "total": self.count_parameters(),
        }
