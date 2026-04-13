"""
orchestrator/model.py — Orchestrator: el coordinador de motores MoSE
=====================================================================

El Orchestrator decide qué motor(es) activar para cada query.

3 capacidades:
  1. CLASIFICAR:   ¿es causal, código, creativo, math, social?
  2. PLANIFICAR:   ¿un motor o varios? ¿en qué orden?
  3. PRESUPUESTAR: ¿cuánto compute por motor (n_iterations)?

Arquitectura:
  - Recibe concept_vectors del encoder [B, L, D]
  - Pool temporal (mean) → [B, D]
  - MLP de clasificación (D → hidden → 5 logits)
  - Softmax → 5 probabilidades (una por motor)
  - Top-K selection: activa 1-3 motores según scores y umbral de confianza
  - Asigna n_iterations por motor según score relativo y presupuesto base

Motores (orden canonical):
  0  cora      — Causal Reasoning
  1  forge_c   — Code Reasoning
  2  muse      — Creative/Narrative
  3  axiom     — Mathematical Proof
  4  empathy   — Social/Conversation

Heurísticas de fallback (cuando concept_vectors no está disponible):
  FORGE-C:  function, class, import, bug, error, def, code, código
  AXIOM:    demuestra, ecuación, calcula, teorema, demostrar, solve, equation
  MUSE:     historia, personaje, cuento, poema, story, poem, escribe, crea
  EMPATHY:  siente, cree, quiere, amigo, triste, feels, believes, emoción
  CORA:     default (si ninguna heurística coincide)

Presupuesto base de iteraciones (ajustado por score):
  CORA:     5   — razonamiento causal estándar
  FORGE-C:  3   — análisis de código (rápido)
  MUSE:     5   — generación narrativa (creativa)
  AXIOM:    7   — pruebas matemáticas (profundidad)
  EMPATHY:  3   — respuesta social (concisa)
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

MOTOR_NAMES: List[str] = ["cora", "forge_c", "muse", "axiom", "empathy"]

# Presupuesto base de iteraciones por motor
BASE_ITERATIONS: Dict[str, int] = {
    "cora":    5,
    "forge_c": 3,
    "muse":    5,
    "axiom":   7,
    "empathy": 3,
}

# Palabras clave para el fallback heurístico
KEYWORD_TRIGGERS: Dict[str, List[str]] = {
    "forge_c": [
        "function", "class", "import", "bug", "error", "def", "code",
        "código", "programa", "python", "javascript", "typescript", "java",
        "variable", "loop", "array", "debug", "test", "compile", "syntax",
    ],
    "axiom": [
        "demuestra", "ecuación", "calcula", "teorema", "demostrar",
        "solve", "equation", "theorem", "prueba", "integral", "derivada",
        "álgebra", "geometría", "cálculo", "proof", "math", "fórmula",
    ],
    "muse": [
        "historia", "personaje", "cuento", "poema", "story", "poem",
        "narrativa", "escribe", "crea", "novela", "creative", "creativo",
        "narrativo", "estrofa", "metáfora", "describe", "imagina",
    ],
    "empathy": [
        "siente", "cree", "quiere", "amigo", "triste", "feels", "believes",
        "emoción", "relación", "social", "ayuda", "apoyo", "empatía",
        "conflicto", "persona", "comunicación", "entender", "escuchar",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MotorActivation:
    """
    Descripción de un motor activado por el Orchestrator.

    motor_name:   nombre del motor ("cora", "forge_c", etc.)
    score:        probabilidad asignada por el clasificador (0.0–1.0)
    n_iterations: presupuesto de iteraciones para el CRE de este motor
    rank:         orden de activación (1=primero, 2=segundo, etc.)
    motor_idx:    índice en MOTOR_NAMES (0–4)
    """
    motor_name:   str
    score:        float
    n_iterations: int
    rank:         int
    motor_idx:    int

    def __repr__(self) -> str:
        return (
            f"MotorActivation({self.motor_name}, "
            f"score={self.score:.3f}, iters={self.n_iterations}, rank={self.rank})"
        )


@dataclass
class OrchestratorOutput:
    """
    Resultado del Orchestrator para un batch de queries.

    activations:    lista ordenada de motores a activar (rank 1 primero)
    scores:         [5] tensor de probabilidades (después de softmax)
    logits:         [5] tensor de logits antes de softmax
    routing_mode:   "learned" (MLP) o "heuristic" (keyword fallback)
    n_active:       número de motores activados (1–max_active_motors)
    """
    activations:  List[MotorActivation]
    scores:       torch.Tensor      # [5]
    logits:       torch.Tensor      # [5]
    routing_mode: str
    n_active:     int

    @property
    def primary_motor(self) -> MotorActivation:
        """Motor principal (rank 1)."""
        return self.activations[0]

    @property
    def motor_names(self) -> List[str]:
        """Lista de nombres de motores activados, en orden."""
        return [a.motor_name for a in self.activations]

    def __repr__(self) -> str:
        motors = ", ".join(
            f"{a.motor_name}({a.score:.2f}×{a.n_iterations}it)"
            for a in self.activations
        )
        return f"OrchestratorOutput([{motors}], mode={self.routing_mode})"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorConfig:
    """
    Configuración del Orchestrator.

    hidden_dim:                 dimensión de los concept_vectors del encoder
    n_motors:                   número de motores (siempre 5 en MoSE)
    max_active_motors:          máximo de motores a activar simultáneamente
    min_confidence_to_activate: score mínimo para activar un motor secundario
    mlp_hidden_dim:             dimensión oculta del MLP clasificador
    max_iter_multiplier:        multiplicador del presupuesto base para motor primario
    """
    hidden_dim:                 int   = 256
    n_motors:                   int   = 5
    max_active_motors:          int   = 3
    min_confidence_to_activate: float = 0.3
    mlp_hidden_dim:             int   = 128
    max_iter_multiplier:        float = 2.0   # score alto → más iteraciones

    def __post_init__(self) -> None:
        if self.n_motors != len(MOTOR_NAMES):
            raise ValueError(
                f"n_motors must be {len(MOTOR_NAMES)}, got {self.n_motors}"
            )
        if not 1 <= self.max_active_motors <= self.n_motors:
            raise ValueError(
                f"max_active_motors must be in [1, {self.n_motors}], "
                f"got {self.max_active_motors}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator(nn.Module):
    """
    Orchestrator: decide qué motor(es) activar para cada query.

    Flujo principal (learned routing):
      concept_vectors [B, L, D]
          ↓ mean pool → [B, D] → mean → [D]
          ↓ MLP → [5] logits → softmax → [5] scores
          ↓ top-K con umbral min_confidence_to_activate
          ↓ OrchestratorOutput(activations=[MotorActivation, ...])

    Flujo alternativo (heuristic fallback):
      query_text: str → keyword matching → score artificial → mismo output

    Asignación de iteraciones:
      motor primario (rank 1):   base_iter × max(1.0, score × max_iter_multiplier)
      motores secundarios:       base_iter × max(0.5, score)
      Todo redondeado al entero más cercano (mínimo 1).
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        super().__init__()
        self.config = config
        D   = config.hidden_dim
        H   = config.mlp_hidden_dim
        N   = config.n_motors

        # MLP clasificador: D → H → H//2 → N
        self.classifier = nn.Sequential(
            nn.Linear(D, H),
            nn.GELU(),
            nn.LayerNorm(H),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, N),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FORWARD PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        concept_vectors: torch.Tensor,           # [B, L, D]
        query_text:      Optional[str] = None,   # para heurística de fallback
    ) -> OrchestratorOutput:
        """
        Clasifica la query y decide qué motor(es) activar.

        Args:
            concept_vectors: [B, L, D] — salida del StreamEncoder
            query_text:      texto de la query (para heurística de fallback)

        Returns:
            OrchestratorOutput con lista de motores ordenada por relevancia
        """
        B, L, D = concept_vectors.shape

        # Pool temporal: [B, L, D] → [B, D]
        pooled = concept_vectors.mean(dim=1)   # [B, D]

        # Pool de batch: [B, D] → [1, D]  (keepdim=True → siempre 2-D)
        # Mantener 2D es necesario para compatibilidad con backends como DirectML
        # que no soportan nn.Linear con tensores 1D como entrada.
        # Para batch > 1, promediamos (simplificación: mismo routing para el batch).
        query_repr = pooled.mean(dim=0, keepdim=True)   # [1, D]

        # MLP → logits [1, 5] → squeeze → [5] → softmax → probabilidades
        logits = self.classifier(query_repr).squeeze(0)  # [5]
        scores = F.softmax(logits, dim=-1)               # [5]

        activations = self._select_motors(scores, logits)
        routing_mode = "learned"

        # Si no hay confianza suficiente y tenemos texto, usar heurística
        if (
            scores.max().item() < self.config.min_confidence_to_activate
            and query_text is not None
        ):
            return self.heuristic_route(query_text, logits=logits, scores=scores)

        return OrchestratorOutput(
            activations=activations,
            scores=(scores if self.training else scores.detach()),
            logits=(logits if self.training else logits.detach()),
            routing_mode=routing_mode,
            n_active=len(activations),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SELECCIÓN DE MOTORES
    # ─────────────────────────────────────────────────────────────────────────

    def _select_motors(
        self,
        scores: torch.Tensor,   # [5] probabilities
        logits: torch.Tensor,   # [5] logits
    ) -> List[MotorActivation]:
        """
        Selecciona top-K motores según scores y umbral de confianza.

        - Siempre activa al menos el motor con mayor score (rank 1).
        - Activa motores adicionales si score ≥ min_confidence_to_activate.
        - Máximo max_active_motors motores.
        """
        cfg = self.config
        scores_list = scores.tolist()

        # Ordenar por score descendente
        ranked = sorted(
            enumerate(scores_list),
            key=lambda x: x[1],
            reverse=True,
        )

        activations: List[MotorActivation] = []
        for rank, (motor_idx, score) in enumerate(ranked, start=1):
            # Siempre incluir el primero; los demás necesitan superar el umbral
            if rank > 1 and score < cfg.min_confidence_to_activate:
                break
            if rank > cfg.max_active_motors:
                break

            motor_name = MOTOR_NAMES[motor_idx]
            n_iters    = self._compute_iterations(motor_name, score, rank)
            activations.append(MotorActivation(
                motor_name=motor_name,
                score=score,
                n_iterations=n_iters,
                rank=rank,
                motor_idx=motor_idx,
            ))

        return activations

    def _compute_iterations(
        self,
        motor_name: str,
        score:      float,
        rank:       int,
    ) -> int:
        """
        Calcula n_iterations para este motor según score y rango.

        Motor primario (rank=1): base × max_iter_multiplier × score (capped at base × mult)
        Motores secundarios:     base × score (mínimo 1)
        """
        base = BASE_ITERATIONS[motor_name]
        # Guard against NaN/Inf scores (can occur early in training with small models)
        if not (0.0 <= score <= 1.0):
            score = 0.5
        if rank == 1:
            # Motor principal: más iteraciones si score es alto
            multiplier = min(self.config.max_iter_multiplier, 1.0 + score)
            return max(1, round(base * multiplier))
        else:
            # Motores secundarios: proporcional al score, mínimo 1
            return max(1, round(base * score))

    # ─────────────────────────────────────────────────────────────────────────
    # HEURÍSTICA DE FALLBACK
    # ─────────────────────────────────────────────────────────────────────────

    def heuristic_route(
        self,
        query_text: str,
        logits:     Optional[torch.Tensor] = None,
        scores:     Optional[torch.Tensor] = None,
    ) -> OrchestratorOutput:
        """
        Routing basado en palabras clave cuando el clasificador MLP no tiene confianza.

        Devuelve OrchestratorOutput con routing_mode="heuristic".
        """
        text_lower = query_text.lower()

        # Contar coincidencias por motor
        motor_hits: Dict[str, int] = {name: 0 for name in MOTOR_NAMES}
        for motor_name, keywords in KEYWORD_TRIGGERS.items():
            for kw in keywords:
                if kw in text_lower:
                    motor_hits[motor_name] += 1

        # Motor con más coincidencias (default CORA si empate o sin coincidencias)
        best_motor = max(motor_hits, key=lambda m: motor_hits[m])
        if motor_hits[best_motor] == 0:
            best_motor = "cora"

        # Construir scores artificiales (1.0 para el primario, 0.1 resto)
        device = logits.device if logits is not None else torch.device("cpu")
        dtype  = logits.dtype  if logits is not None else torch.float32
        heur_scores = torch.full((5,), 0.1, device=device, dtype=dtype)
        primary_idx = MOTOR_NAMES.index(best_motor)
        heur_scores[primary_idx] = 0.9

        # Comprobar si hay un segundo motor con suficientes hits
        sorted_hits = sorted(motor_hits.items(), key=lambda x: x[1], reverse=True)
        activations: List[MotorActivation] = []
        rank = 1
        for motor_name, hits in sorted_hits:
            if rank > self.config.max_active_motors:
                break
            if rank > 1 and hits == 0:
                break
            if rank > 1 and motor_hits[motor_name] < motor_hits[sorted_hits[0][0]] * 0.5:
                break
            score = heur_scores[MOTOR_NAMES.index(motor_name)].item()
            n_iters = self._compute_iterations(motor_name, score, rank)
            activations.append(MotorActivation(
                motor_name=motor_name,
                score=score,
                n_iterations=n_iters,
                rank=rank,
                motor_idx=MOTOR_NAMES.index(motor_name),
            ))
            rank += 1

        if not activations:
            # Fallback absoluto: CORA
            activations = [MotorActivation(
                motor_name="cora",
                score=1.0,
                n_iterations=BASE_ITERATIONS["cora"],
                rank=1,
                motor_idx=0,
            )]

        if logits is None:
            logits = torch.zeros(5)
        if scores is None:
            scores = heur_scores

        return OrchestratorOutput(
            activations=activations,
            scores=(heur_scores if self.training else heur_scores.detach()),
            logits=(logits if self.training else logits.detach()),
            routing_mode="heuristic",
            n_active=len(activations),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # UTILIDADES
    # ─────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
