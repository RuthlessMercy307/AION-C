"""
memory/conversation_history.py — Multi-turn con resumen progresivo (Parte 8.5)
================================================================================

Estructura del historial de una conversación:
  - Últimos N (default 3) turns: completos en contexto.
  - Turns 4-10: resumen comprimido.
  - Turns 10+: solo hechos clave en MEM persistente.

El ConversationHistory expone tres slices distintos:
  - `recent_turns()`     → últimos N turns con contenido completo
  - `summary_block()`    → resumen comprimido de los turns mid-range
  - `key_facts()`        → hechos clave extraídos (lista de strings)

El summarizer es inyectable. Default: heurístico (concatenación
truncada). En producción se inyecta un summarizer que llame al modelo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Turn:
    """Un turno de la conversación."""
    role:      str   # "user" | "assistant"
    content:   str
    timestamp: float = field(default_factory=time.time)
    metadata:  Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role":      self.role,
            "content":   self.content,
            "timestamp": self.timestamp,
            "metadata":  dict(self.metadata),
        }


SummarizerFn = Callable[[List[Turn]], str]


def default_summarizer(turns: List[Turn]) -> str:
    """
    Resumen heurístico (sin modelo): por cada par user/assistant compone
    una línea "Q: ... → A: ...". Útil como fallback y para tests.
    """
    lines: List[str] = []
    pending_user: Optional[Turn] = None
    for t in turns:
        if t.role == "user":
            pending_user = t
        elif t.role == "assistant" and pending_user is not None:
            q = pending_user.content[:60]
            a = t.content[:60]
            lines.append(f"Q: {q} → A: {a}")
            pending_user = None
    if pending_user is not None:
        lines.append(f"Q: {pending_user.content[:60]} → (no answer)")
    return " | ".join(lines)


class ConversationHistory:
    """
    Historial multi-turn con compresión progresiva.

    Args:
        recent_window:   cuántos turns mantener completos (default 3 user+assistant pairs = 6 mensajes)
        mid_window:      hasta cuántos turns mantener en resumen (default 10 mensajes)
        summarizer_fn:   callable para comprimir (default: heurístico)
    """

    def __init__(
        self,
        recent_window: int = 6,    # 3 pares user/assistant
        mid_window:    int = 10,
        summarizer_fn: Optional[SummarizerFn] = None,
    ) -> None:
        if recent_window <= 0 or mid_window < recent_window:
            raise ValueError("mid_window must be >= recent_window > 0")
        self.recent_window = recent_window
        self.mid_window = mid_window
        self.summarizer_fn = summarizer_fn or default_summarizer
        self.turns: List[Turn] = []
        self._key_facts: List[str] = []

    # ── append ─────────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str, **metadata) -> None:
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"invalid role: {role}")
        self.turns.append(Turn(role=role, content=content, metadata=metadata))

    def add_user(self, content: str, **md) -> None:
        self.add_turn("user", content, **md)

    def add_assistant(self, content: str, **md) -> None:
        self.add_turn("assistant", content, **md)

    # ── slices ────────────────────────────────────────────────────────

    def recent_turns(self) -> List[Turn]:
        """Últimos `recent_window` turns en bruto."""
        return list(self.turns[-self.recent_window:])

    def mid_turns(self) -> List[Turn]:
        """Turns en la zona de resumen (entre recent_window y mid_window)."""
        n = len(self.turns)
        if n <= self.recent_window:
            return []
        end = n - self.recent_window
        start = max(0, end - (self.mid_window - self.recent_window))
        return list(self.turns[start:end])

    def old_turns(self) -> List[Turn]:
        """Turns más viejos que mid_window (de los que extraemos hechos clave)."""
        n = len(self.turns)
        if n <= self.mid_window:
            return []
        return list(self.turns[: n - self.mid_window])

    def summary_block(self) -> str:
        """Resumen comprimido de los mid_turns."""
        mt = self.mid_turns()
        if not mt:
            return ""
        return self.summarizer_fn(mt)

    # ── hechos clave ───────────────────────────────────────────────────

    def add_key_fact(self, fact: str) -> None:
        if fact and fact not in self._key_facts:
            self._key_facts.append(fact)

    def key_facts(self) -> List[str]:
        return list(self._key_facts)

    def extract_facts_from_old(self, extractor_fn: Optional[Callable[[Turn], List[str]]] = None) -> int:
        """
        Extrae hechos de los turns más viejos que mid_window y los promueve
        a key_facts. El extractor por defecto identifica cosas como
        "Mi nombre es X", "Estoy trabajando en X", "Prefiero X" — heurístico.
        """
        old = self.old_turns()
        if not old:
            return 0
        extractor = extractor_fn or _default_fact_extractor
        n_added = 0
        for t in old:
            for fact in extractor(t):
                if fact not in self._key_facts:
                    self._key_facts.append(fact)
                    n_added += 1
        return n_added

    # ── render para inyectar en el contexto del modelo ─────────────────

    def render_context(self) -> str:
        """
        Texto canónico para inyectar como contexto, en este orden:
          [FACTS: ...]
          [SUMMARY: ...]
          [HISTORY: ...]
        Cumple con el formato canónico de la Parte 15.
        """
        parts: List[str] = []
        if self._key_facts:
            parts.append("[FACTS: " + " | ".join(self._key_facts) + "]")
        sb = self.summary_block()
        if sb:
            parts.append(f"[SUMMARY: {sb}]")
        rt = self.recent_turns()
        if rt:
            lines = []
            for t in rt:
                tag = "USER" if t.role == "user" else "AION"
                lines.append(f"[{tag}: {t.content}]")
            parts.append("\n".join(lines))
        return "\n".join(parts)

    # ── stats ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.turns)

    def stats(self) -> Dict[str, int]:
        return {
            "total_turns":     len(self.turns),
            "recent_turns":    len(self.recent_turns()),
            "mid_turns":       len(self.mid_turns()),
            "old_turns":       len(self.old_turns()),
            "key_facts":       len(self._key_facts),
        }


def _default_fact_extractor(turn: Turn) -> List[str]:
    """
    Extractor heurístico simple. Marca como hecho cualquier mensaje del
    user que coincida con patrones de auto-identificación.
    """
    if turn.role != "user":
        return []
    text = turn.content.strip()
    low = text.lower()
    patterns_es = ("mi nombre es ", "me llamo ", "soy ", "trabajo en ", "estoy aprendiendo ", "uso ")
    patterns_en = ("my name is ", "i'm ", "i am ", "i work on ", "i'm learning ", "i use ")
    for p in patterns_es + patterns_en:
        if low.startswith(p):
            return [text[:120]]
    return []


__all__ = [
    "Turn",
    "ConversationHistory",
    "default_summarizer",
]
