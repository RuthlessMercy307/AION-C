"""
memory/response_cache.py — Cache inteligente de respuestas (Parte 8.4 del MEGA-PROMPT)
=========================================================================================

Si el modelo ya resolvió "what is javascript?", la próxima vez se devuelve
del cache sin re-generar.

Características:
  - Clave normalizada (lowercase, whitespace colapsado, sin signos de
    puntuación finales) para que "Hola!" y "hola" colisionen.
  - LRU bounded por max_size (default 256)
  - TTL opcional por entrada (segundos)
  - Invalidación por patrón: si MEM sobre "javascript" se actualiza,
    invalidate_by_substring("javascript") borra todas las entradas que
    mencionen "javascript".
  - Hit/miss stats para interpretabilidad.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple


_NORMALIZE_RE = re.compile(r"\s+")
_TRIM_PUNCT = r" .,!?¡¿;:"


def normalize_query(text: str) -> str:
    """
    Normaliza un query para usarlo como clave de cache.
    Lowercase, espacios colapsados, puntuación terminal eliminada.
    """
    if text is None:
        return ""
    s = text.strip().lower()
    s = _NORMALIZE_RE.sub(" ", s)
    s = s.strip(_TRIM_PUNCT)
    return s


@dataclass
class CacheEntry:
    response: str
    stored_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    hits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.ttl is None:
            return False
        now = now if now is not None else time.time()
        return (now - self.stored_at) > self.ttl


class ResponseCache:
    """
    LRU bounded cache con TTL opcional e invalidación por substring.

    Uso:
        cache = ResponseCache(max_size=256)
        cache.set("what is javascript?", "JavaScript is...")
        cache.get("WHAT IS JAVASCRIPT")  # hit (normalizado)
        cache.invalidate_by_substring("javascript")  # borra entradas relacionadas
    """

    def __init__(self, max_size: int = 256, default_ttl: Optional[float] = None) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    # ── operaciones básicas ────────────────────────────────────────────

    def set(
        self,
        query: str,
        response: str,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = normalize_query(query)
        if not key:
            return
        entry = CacheEntry(
            response=response,
            ttl=ttl if ttl is not None else self.default_ttl,
            metadata=dict(metadata or {}),
        )
        if key in self._store:
            del self._store[key]
        self._store[key] = entry
        self._evict_if_needed()

    def get(self, query: str) -> Optional[str]:
        key = normalize_query(query)
        if key not in self._store:
            self.misses += 1
            return None
        entry = self._store[key]
        if entry.is_expired():
            del self._store[key]
            self.misses += 1
            return None
        # LRU touch
        self._store.move_to_end(key)
        entry.hits += 1
        self.hits += 1
        return entry.response

    def has(self, query: str) -> bool:
        key = normalize_query(query)
        return key in self._store and not self._store[key].is_expired()

    def delete(self, query: str) -> bool:
        key = normalize_query(query)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    # ── invalidación inteligente ───────────────────────────────────────

    def invalidate_by_substring(self, substring: str) -> int:
        """
        Borra todas las entradas cuya clave o respuesta contengan `substring`.
        Devuelve el número de entradas eliminadas.
        Útil cuando MEM sobre un tema se actualiza.
        """
        if not substring:
            return 0
        s = substring.lower()
        to_remove = [
            k for k, e in self._store.items()
            if s in k or s in e.response.lower()
        ]
        for k in to_remove:
            del self._store[k]
        return len(to_remove)

    def invalidate_by_predicate(self, predicate) -> int:
        """Borra las entradas para las que `predicate(key, entry)` es True."""
        to_remove = [k for k, e in self._store.items() if predicate(k, e)]
        for k in to_remove:
            del self._store[k]
        return len(to_remove)

    def purge_expired(self) -> int:
        """Elimina todas las entradas expiradas."""
        now = time.time()
        to_remove = [k for k, e in self._store.items() if e.is_expired(now)]
        for k in to_remove:
            del self._store[k]
        return len(to_remove)

    # ── stats ──────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size":     len(self._store),
            "max_size": self.max_size,
            "hits":     self.hits,
            "misses":   self.misses,
            "hit_rate": (self.hits / total) if total > 0 else 0.0,
        }

    def keys(self) -> Iterable[str]:
        return list(self._store.keys())

    # ── internas ───────────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)  # FIFO/LRU bottom


__all__ = [
    "ResponseCache",
    "CacheEntry",
    "normalize_query",
]
