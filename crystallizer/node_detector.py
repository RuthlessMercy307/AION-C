"""
crystallizer/node_detector.py — NodeDetector
=============================================

MLP que clasifica cada posición de la secuencia como nodo o no-nodo.

Por qué no usar attention para detección de nodos:
    La detección de nodos es un problema POR POSICIÓN — ¿es este concepto
    relevante por sí mismo? No depende de su relación con otros.
    Un MLP por posición es suficiente y más eficiente que self-attention.

Tres salidas por posición:
    node_score  — ¿es este concepto un nodo del grafo?
    type_logits — ¿qué tipo de nodo es? (ENTITY, EVENT, STATE, ...)
    confidence  — ¿cuánto confía el detector en su propia predicción?

La separación de `node_score` y `confidence` permite:
    - Nodo con alta probabilidad pero baja confianza (incertidumbre genuina)
    - Nodo con baja probabilidad pero alta confianza (detecta que NO es nodo)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import CrystallizerConfig


class NodeDetector(nn.Module):
    """
    MLP que asigna un puntaje de nodo y tipo a cada posición.

    Uso:
        cfg     = CrystallizerConfig()
        detector = NodeDetector(cfg)
        concepts = torch.randn(2, 64, 256)       # [B, L, D]
        scores, type_logits, conf = detector(concepts)
        # scores:      [2, 64]       — sigmoid, probabilidad de nodo
        # type_logits: [2, 64, 7]    — sin activación, para CrossEntropy
        # conf:        [2, 64]       — sigmoid, confianza del detector
    """

    def __init__(self, config: CrystallizerConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        C = config.node_confidence_hidden_dim

        # ── Puntuación de nodo ─────────────────────────────────────────────────
        # Dos capas: la primera expande para capturar interacciones internas,
        # la segunda colapsa a un escalar. Sin sesgo en la segunda capa
        # para evitar que el bias empuje todos los scores hacia arriba.
        self.node_scorer = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1, bias=False),
        )

        # ── Clasificador de tipo ───────────────────────────────────────────────
        # Proyección directa D → n_types; no sigmoid/softmax aquí
        # (el consumidor aplica softmax o argmax según el uso)
        self.type_classifier = nn.Linear(D, config.n_node_types)

        # ── Estimador de confianza ─────────────────────────────────────────────
        # Separado del scorer para desacoplar "¿es nodo?" de "¿cuánto confío?"
        self.confidence_head = nn.Sequential(
            nn.Linear(D, C),
            nn.GELU(),
            nn.Linear(C, 1, bias=False),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        concepts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            concepts: [B, L, D] — concept vectors del StreamEncoder

        Returns:
            node_scores:  [B, L]           — sigmoid ∈ (0, 1), probability of being a node
            type_logits:  [B, L, n_types]  — sin activación
            confidence:   [B, L]           — sigmoid ∈ (0, 1)
        """
        node_scores = torch.sigmoid(
            self.node_scorer(concepts).squeeze(-1)
        )                                              # [B, L] ∈ (0, 1)

        type_logits = self.type_classifier(concepts)  # [B, L, n_types]

        confidence = torch.sigmoid(
            self.confidence_head(concepts).squeeze(-1)
        )                                          # [B, L]

        return node_scores, type_logits, confidence
