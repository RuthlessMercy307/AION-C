"""
crystallizer/relation_scorer.py — AsymmetricRelationScorer
===========================================================

Puntúa relaciones dirigidas entre pares de nodos.

EL PROBLEMA CON DOT-PRODUCT ATTENTION:
    La atención estándar mide SIMILITUD: score(A, B) = Q(A)·K(B)
    Pero Q y K son la misma proyección → score(A,B) = score(B,A)
    La atención simétrica NO puede representar "A causa B" ≠ "B causa A".

LA SOLUCIÓN — PROYECCIONES ASIMÉTRICAS:
    source_proj: "¿qué rol juega este nodo COMO FUENTE de una relación?"
    target_proj: "¿qué rol juega este nodo COMO DESTINO de una relación?"

    score(A→B, r) = source_proj_r(A) · target_proj_r(B)
    score(B→A, r) = source_proj_r(B) · target_proj_r(A)

    Como source_proj ≠ target_proj:
        source_proj(A) ≠ target_proj(A)   en general
    → score(A→B) ≠ score(B→A)             en general   ✓ ASIMÉTRICO

    Analogía física:
        Una flecha (→) tiene punta y cola — no es lo mismo invertirla.
        source_proj captura la "punta" (causa activa).
        target_proj captura la "cola" (efecto receptivo).

IMPLEMENTACIÓN:
    Por eficiencia, se proyecta a R×H dimensiones en un solo Linear,
    luego se reorganiza como R cabezas de H dimensiones cada una.
    El producto interior por cabeza da el score por relación.

    Finalmente, un MLP refiner ajusta los scores con información
    global del vector de scores completo (interacciones entre relaciones).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import CrystallizerConfig


class AsymmetricRelationScorer(nn.Module):
    """
    Puntúa las R relaciones entre todos los pares (i→j) de nodos.

    Uso:
        cfg    = CrystallizerConfig()
        scorer = AsymmetricRelationScorer(cfg)

        # Input batched [B, n, D]:
        nodes  = torch.randn(2, 8, 256)
        logits = scorer(nodes, nodes)   # [2, 8, 8, 16]

        # Input sin batch [n, D]:
        nodes  = torch.randn(8, 256)
        logits = scorer(nodes, nodes)   # [8, 8, 16]

    Asimetría verificable:
        logits[i, j, :] ≠ logits[j, i, :]  (en general)
        porque source_proj ≠ target_proj
    """

    def __init__(self, config: CrystallizerConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        R = config.n_relation_types
        H = config.relation_hidden_dim

        # Proyecciones asimétricas: mismo input D, roles distintos
        # Cada una produce R sub-vectores de H dimensiones
        # (un sub-espacio por tipo de relación)
        self.source_proj = nn.Linear(D, R * H, bias=False)
        self.target_proj = nn.Linear(D, R * H, bias=False)

        # Refinamiento: ajusta scores finales con contexto inter-relacional
        # (saber que hay CAUSES puede modular la certeza de ENABLES)
        self.refiner = nn.Sequential(
            nn.Linear(R, R * 2),
            nn.GELU(),
            nn.Linear(R * 2, R),
        )

        self._R = R
        self._H = H

        self._init_weights()

    def _init_weights(self) -> None:
        # Proyecciones: init normal estándar
        nn.init.normal_(self.source_proj.weight, std=0.02)
        nn.init.normal_(self.target_proj.weight, std=0.02)
        # Refiner: normal + ceros en biases
        for m in self.refiner.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        source_nodes: torch.Tensor,  # [B, n, D] ó [n, D]
        target_nodes: torch.Tensor,  # [B, m, D] ó [m, D]
    ) -> torch.Tensor:
        """
        Computa logits de relación para todos los pares dirigidos (i→j).

        Args:
            source_nodes: [B, n, D] ó [n, D]
            target_nodes: [B, m, D] ó [m, D]

        Returns:
            relation_logits: [B, n, m, R] ó [n, m, R]
                logits[..., i, j, r] = score de que nodo i tiene relación r con nodo j
                Sin sigmoid/softmax — el consumidor decide la activación.
        """
        batched = source_nodes.dim() == 3
        if not batched:
            source_nodes = source_nodes.unsqueeze(0)  # [1, n, D]
            target_nodes = target_nodes.unsqueeze(0)  # [1, m, D]

        B, n, D = source_nodes.shape
        m = target_nodes.shape[1]
        R, H = self._R, self._H

        # [B, n, R, H] — cada nodo proyectado en su rol de FUENTE
        s = self.source_proj(source_nodes).view(B, n, R, H)

        # [B, m, R, H] — cada nodo proyectado en su rol de DESTINO
        t = self.target_proj(target_nodes).view(B, m, R, H)

        # score(i→j, r) = dot(s[b,i,r,:], t[b,j,r,:])
        # [B, n, m, R] = einsum sobre la dimensión H
        scores = torch.einsum("bnrh,bmrh->bnmr", s, t)

        # Refinamiento post-puntuación
        refined = self.refiner(scores)  # [B, n, m, R]

        return refined if batched else refined.squeeze(0)  # [n, m, R]
