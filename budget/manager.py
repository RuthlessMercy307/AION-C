"""
budget/manager.py — BudgetManager
====================================

Clasifica queries en 4 niveles de complejidad y asigna un presupuesto de
iteraciones máximas para el CRE.

POSICION EN EL PIPELINE:
    token_ids → StreamEncoder → concept_vectors
                                      ↓
                              BudgetManager.forward()
                                      ↓ n_iterations
                              CausalReasoningEngine(n_iterations=budget)

POR QUE:
    No gastar compute en queries triviales ("2+2=?").
    Reservar iteraciones maximas para queries profundas ("diseña microservicios").

NIVELES:
    TRIVIAL → 1 iteracion   (<10 tokens, preguntas de hecho simple)
    SIMPLE  → 3 iteraciones (10-30 tokens, preguntas directas)
    COMPLEX → 10 iteraciones (30-100 tokens, analisis, comparaciones)
    DEEP    → max_iterations (>100 tokens, razonamiento multi-paso)

CLASIFICADOR (QueryComplexityClassifier):
    MLP pequeño sobre mean-pool de concept vectors.
    Linear(D → hidden) → GELU → LayerNorm → Linear(hidden → 4).
    Entrenado end-to-end junto con el pipeline.

FALLBACK HEURISTICO:
    Cuando use_learned=False o concept_vectors=None.
    Reglas por longitud de secuencia (n_tokens).
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# BUDGET LEVEL
# ─────────────────────────────────────────────────────────────────────────────

class BudgetLevel(IntEnum):
    """Niveles de complejidad de query (orden creciente de compute)."""
    TRIVIAL = 0
    SIMPLE  = 1
    COMPLEX = 2
    DEEP    = 3


# Umbrales de tokens para el fallback heuristico
HEURISTIC_THRESHOLDS = (10, 30, 100)
# < 10  → TRIVIAL
# < 30  → SIMPLE
# < 100 → COMPLEX
# ≥ 100 → DEEP

# Iteraciones por nivel; None = usar max_iterations del CRE
_BASE_ITERATIONS = {
    BudgetLevel.TRIVIAL: 1,
    BudgetLevel.SIMPLE:  3,
    BudgetLevel.COMPLEX: 10,
    BudgetLevel.DEEP:    None,
}


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BudgetOutput:
    """
    Resultado del BudgetManager.

    level:          BudgetLevel — nivel de complejidad clasificado
    n_iterations:   int         — iteraciones a pasar al CRE
    used_heuristic: bool        — True si se uso el fallback de longitud
    class_probs:    Optional[Tensor] — [B, 4] softmax del clasificador (None si heuristico)
    """
    level:          BudgetLevel
    n_iterations:   int
    used_heuristic: bool
    class_probs:    Optional[torch.Tensor] = None   # [B, 4], detached


# ─────────────────────────────────────────────────────────────────────────────
# QUERY COMPLEXITY CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class QueryComplexityClassifier(nn.Module):
    """
    MLP pequeño que predice el nivel de complejidad de una query.

    Input:  concept_vectors [B, L, concept_dim] — output del StreamEncoder
    Output: logits [B, 4] — un logit por BudgetLevel

    Arquitectura:
        Mean pool [B, L, D] → [B, D]
        Linear(D → hidden_dim) → GELU → LayerNorm(hidden_dim)
        Linear(hidden_dim → 4)

    El pooling es mean sobre la dimension de secuencia (O(L) operaciones,
    sin parametros extra proporcionales a L).
    """

    def __init__(
        self,
        concept_dim: int,
        hidden_dim:  int   = 64,
        n_classes:   int   = 4,
        norm_eps:    float = 1e-6,
    ) -> None:
        super().__init__()
        self.concept_dim = concept_dim
        self.n_classes   = n_classes

        self.mlp = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=norm_eps),
            nn.Linear(hidden_dim, n_classes, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, concept_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concept_vectors: [B, L, concept_dim]

        Returns:
            logits: [B, n_classes]
        """
        pooled = concept_vectors.mean(dim=1)   # [B, D]
        return self.mlp(pooled)                # [B, 4]

    def predict(self, concept_vectors: torch.Tensor) -> torch.Tensor:
        """
        Predice el nivel de complejidad sin gradientes.

        Returns:
            [B] LongTensor con BudgetLevel indices (0-3)
        """
        with torch.no_grad():
            return self.forward(concept_vectors).argmax(dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# BUDGET MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class BudgetManager(nn.Module):
    """
    Clasifica queries y asigna presupuesto de iteraciones para el CRE.

    Tiene dos modos:
        use_learned=True  (default): usa QueryComplexityClassifier sobre concept_vectors
        use_learned=False:           solo heuristico de longitud de tokens

    El clasificador aprendido se entrena end-to-end. El heuristico es determinista
    y util como baseline o warm-start antes de entrenar el clasificador.

    Uso:
        manager = BudgetManager(concept_dim=256, max_cre_iterations=20)

        # Durante forward del pipeline (concept_vectors disponibles)
        budget = manager(token_ids, concept_vectors)
        cre_out = cre(graph, node_feats, n_iterations=budget.n_iterations)

        # Solo heuristico (sin encoder)
        budget = BudgetManager.classify_heuristic(n_tokens=45, max_iterations=20)
        # → COMPLEX, 10 iters

    Args:
        concept_dim:        dimension de los concept vectors del encoder
        max_cre_iterations: iteraciones maximas del CRE (para nivel DEEP)
        hidden_dim:         dimension oculta del MLP clasificador
        use_learned:        activar clasificador aprendido (vs solo heuristico)
        norm_eps:           epsilon de LayerNorm
    """

    def __init__(
        self,
        concept_dim:        int,
        max_cre_iterations: int   = 20,
        hidden_dim:         int   = 64,
        use_learned:        bool  = True,
        norm_eps:           float = 1e-6,
    ) -> None:
        super().__init__()
        self.max_cre_iterations = max_cre_iterations
        self.use_learned        = use_learned

        self.classifier = QueryComplexityClassifier(
            concept_dim = concept_dim,
            hidden_dim  = hidden_dim,
            n_classes   = 4,
            norm_eps    = norm_eps,
        )

    # ── Heuristico ────────────────────────────────────────────────────────────

    @staticmethod
    def classify_heuristic(
        n_tokens:       int,
        max_iterations: int = 20,
    ) -> BudgetOutput:
        """
        Clasifica por longitud de secuencia de tokens.

        Reglas:
            n_tokens < 10  → TRIVIAL → 1 iter
            10 <= n < 30   → SIMPLE  → 3 iters
            30 <= n < 100  → COMPLEX → min(10, max_iterations) iters
            n >= 100       → DEEP    → max_iterations iters

        Args:
            n_tokens:       longitud de la secuencia de entrada
            max_iterations: iteraciones maximas del CRE

        Returns:
            BudgetOutput con used_heuristic=True
        """
        lo, mid, hi = HEURISTIC_THRESHOLDS
        if n_tokens < lo:
            level = BudgetLevel.TRIVIAL
        elif n_tokens < mid:
            level = BudgetLevel.SIMPLE
        elif n_tokens < hi:
            level = BudgetLevel.COMPLEX
        else:
            level = BudgetLevel.DEEP

        return BudgetOutput(
            level          = level,
            n_iterations   = _level_to_iterations(level, max_iterations),
            used_heuristic = True,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        token_ids:       torch.Tensor,           # [B, L]
        concept_vectors: Optional[torch.Tensor], # [B, L, D], puede ser None
    ) -> BudgetOutput:
        """
        Clasifica la query y devuelve el presupuesto de iteraciones.

        Cuando use_learned=False o concept_vectors=None, usa heuristico.
        Cuando use_learned=True, usa el clasificador MLP sobre el primer item del batch.

        Note sobre batching: el pipeline opera batch=1 en produccion.
        Con batch>1, se usa el nivel del item con mayor complejidad (max) para
        que ningun item del batch sea infra-presupuestado.

        Args:
            token_ids:       [B, L]
            concept_vectors: [B, L, D] o None

        Returns:
            BudgetOutput con level y n_iterations
        """
        L = token_ids.shape[1]

        if not self.use_learned or concept_vectors is None:
            return self.classify_heuristic(L, self.max_cre_iterations)

        # Clasificacion aprendida
        logits = self.classifier(concept_vectors)     # [B, 4]
        probs  = F.softmax(logits, dim=-1)            # [B, 4]

        # Con batch > 1, usar el nivel mas alto del batch (conservador)
        level_per_item = logits.argmax(dim=-1)         # [B]
        level_idx      = int(level_per_item.max().item())
        level          = BudgetLevel(level_idx)
        n_iters        = _level_to_iterations(level, self.max_cre_iterations)

        return BudgetOutput(
            level          = level,
            n_iterations   = n_iters,
            used_heuristic = False,
            class_probs    = probs.detach(),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _level_to_iterations(level: BudgetLevel, max_iterations: int) -> int:
    """
    Convierte BudgetLevel → n_iterations para el CRE.

    Para DEEP retorna max_iterations (sin cap adicional).
    Para COMPLEX retorna min(10, max_iterations) para respetar configs small.
    """
    base = _BASE_ITERATIONS[level]
    if base is None:
        return max_iterations
    return min(base, max_iterations)
