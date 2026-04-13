"""
agent/reasoning_levels.py — Razonamiento adaptativo (Parte 2 del MEGA-PROMPT)
==============================================================================

No todos los queries necesitan razonamiento profundo. El BudgetManager decide
qué nivel aplicar:

  NIVEL 0 — Instantáneo (sin CRE)        "hola", "gracias", "adiós"
  NIVEL 1 — Ligero (1-3 iter CRE)        "quién eres?", "qué hora es?"
  NIVEL 2 — Normal (5-10 iter)           "escribe función Python", "15% de 240"
  NIVEL 3 — Profundo (15-50 iter)        "demuestra teorema", "analiza bug complejo"

Decisión basada en:
  - longitud del query
  - score de complejidad del orchestrator (max softmax → query simple)
  - dominio (axiom/forge_c → tienden a más iteraciones)
  - presencia de skill/MEM inyectado (reduce necesidad de iteraciones)

API mínima:
    decider = LevelDecider()
    level = decider.decide(query, orch_scores={"axiom": 0.85, ...},
                            has_skill=True, has_mem=False)
    iterations = level.iterations  # tuple (min, max)
    show_thinking = level.show_thinking_indicator
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN POR NIVEL
# ─────────────────────────────────────────────────────────────────────────────


class ReasoningLevel(IntEnum):
    INSTANT = 0  # sin CRE
    LIGHT   = 1  # 1-3 iter
    NORMAL  = 2  # 5-10 iter
    DEEP    = 3  # 15-50 iter

    @property
    def iterations(self) -> Tuple[int, int]:
        return {
            ReasoningLevel.INSTANT: (0, 0),
            ReasoningLevel.LIGHT:   (1, 3),
            ReasoningLevel.NORMAL:  (5, 10),
            ReasoningLevel.DEEP:    (15, 50),
        }[self]

    @property
    def show_thinking_indicator(self) -> bool:
        """La UI muestra 'thinking...' SOLO en niveles 2-3 (Parte 2.2)."""
        return self >= ReasoningLevel.NORMAL

    @property
    def label(self) -> str:
        return {
            ReasoningLevel.INSTANT: "instant",
            ReasoningLevel.LIGHT:   "light",
            ReasoningLevel.NORMAL:  "normal",
            ReasoningLevel.DEEP:    "deep",
        }[self]


# Vocabulario marcador de queries instantáneos en es+en
INSTANT_TRIGGERS = {
    "hola", "hi", "hey", "hello", "buenas",
    "gracias", "thanks", "thank you", "thx",
    "adios", "adiós", "bye", "goodbye", "chao",
    "ok", "okay", "vale", "perfecto", "perfect",
    "si", "sí", "no", "yes",
}

# Marcadores que sugieren tarea técnica que necesita más razonamiento
DEEP_TRIGGERS = {
    "demuestra", "demuestre", "prove", "demonstrate",
    "analiza", "analyse", "analyze", "deep analysis",
    "depura", "debug",
    "diseña", "design",
    "refactoriza", "refactor",
    "explica paso a paso", "step by step",
    "teorema", "theorem", "proof",
}

# Dominios que tienden a necesitar más iteraciones
COMPUTE_DOMAINS = {"axiom", "forge_c", "cora"}


# ─────────────────────────────────────────────────────────────────────────────
# DECIDER
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LevelDecision:
    """Resultado de la decisión de nivel."""
    level:  ReasoningLevel
    reason: str
    signals: Dict[str, float]


class LevelDecider:
    """
    Decide el nivel de razonamiento para un query dado.

    Args:
        instant_max_words:  queries con <= N palabras son candidatos a NIVEL 0
        normal_min_words:   queries con >= N palabras son candidatos a NIVEL 2
        deep_min_words:     queries con >= N palabras son candidatos a NIVEL 3
        skill_downshift:    si hay skill inyectada, reduce un nivel (default True)
    """

    def __init__(
        self,
        instant_max_words: int = 4,
        normal_min_words:  int = 6,
        deep_min_words:    int = 30,
        skill_downshift:   bool = True,
    ) -> None:
        self.instant_max_words = instant_max_words
        self.normal_min_words = normal_min_words
        self.deep_min_words = deep_min_words
        self.skill_downshift = skill_downshift

    def decide(
        self,
        query: str,
        orch_scores: Optional[Dict[str, float]] = None,
        has_skill: bool = False,
        has_mem:   bool = False,
    ) -> LevelDecision:
        q = (query or "").strip()
        low = q.lower()
        words = [w for w in low.split() if w]
        n_words = len(words)
        signals: Dict[str, float] = {"n_words": float(n_words)}

        # Heurísticas tempranas
        if not q:
            return LevelDecision(ReasoningLevel.INSTANT, "empty query", signals)

        # Triggers DEEP explícitos → siempre profundo
        for marker in DEEP_TRIGGERS:
            if marker in low:
                signals["deep_trigger"] = 1.0
                return LevelDecision(
                    ReasoningLevel.DEEP,
                    f"deep trigger: '{marker}'",
                    signals,
                )

        # Queries muy largos → DEEP
        if n_words >= self.deep_min_words:
            return LevelDecision(
                ReasoningLevel.DEEP,
                f"long query ({n_words} words >= {self.deep_min_words})",
                signals,
            )

        # Triggers INSTANT (saludos cortos)
        if n_words <= self.instant_max_words:
            for trig in INSTANT_TRIGGERS:
                if low == trig or low.startswith(trig + " ") or low.endswith(" " + trig):
                    signals["instant_trigger"] = 1.0
                    level = ReasoningLevel.INSTANT
                    return self._maybe_downshift(level, signals, "instant trigger", has_skill)

        # Confianza del orchestrator
        max_score = 0.0
        top_domain = None
        if orch_scores:
            top_domain, max_score = max(orch_scores.items(), key=lambda x: x[1])
            signals["orch_max_score"] = float(max_score)
            signals["orch_top_domain_compute"] = float(top_domain in COMPUTE_DOMAINS)

        # Decisión por longitud + dominio + score
        if n_words <= self.instant_max_words and (max_score == 0.0 or max_score >= 0.7):
            level = ReasoningLevel.LIGHT
            reason = f"short query ({n_words} words) and/or high orch confidence"
        elif n_words >= self.normal_min_words or (top_domain in COMPUTE_DOMAINS):
            level = ReasoningLevel.NORMAL
            reason = (f"normal query ({n_words} words)"
                      f" — compute domain={top_domain}" if top_domain in COMPUTE_DOMAINS
                      else f"normal query ({n_words} words)")
        else:
            level = ReasoningLevel.LIGHT
            reason = f"default light ({n_words} words)"

        return self._maybe_downshift(level, signals, reason, has_skill)

    # ── helpers ────────────────────────────────────────────────────────

    def _maybe_downshift(
        self,
        level: ReasoningLevel,
        signals: Dict[str, float],
        reason: str,
        has_skill: bool,
    ) -> LevelDecision:
        """Si hay skill inyectada, baja un nivel (mínimo INSTANT)."""
        if self.skill_downshift and has_skill and level > ReasoningLevel.INSTANT:
            new_level = ReasoningLevel(int(level) - 1)
            signals["skill_downshift"] = 1.0
            return LevelDecision(
                new_level,
                f"{reason} → downshift due to skill injected",
                signals,
            )
        return LevelDecision(level, reason, signals)


__all__ = [
    "ReasoningLevel",
    "LevelDecision",
    "LevelDecider",
    "INSTANT_TRIGGERS",
    "DEEP_TRIGGERS",
    "COMPUTE_DOMAINS",
]
