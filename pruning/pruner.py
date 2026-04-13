"""
pruning/pruner.py — MemoryPruner con 4 señales (Parte 24).

Uso típico desde el sleep cycle:

    pruner = MemoryPruner(PruneConfig())
    items = [(id_i, PruneSignals(...)) for id_i, ... in memory.iter_items()]
    report = pruner.prune(items)
    for d in report.decisions:
        if d.action == PruneAction.DELETE:
            memory.delete(d.item_id)
        elif d.action == PruneAction.COMPRESS:
            memory.mark_compress(d.item_id)
        elif d.action == PruneAction.PROMOTE:
            memory.promote(d.item_id)
        memory.set_ttl(d.item_id, d.ttl_seconds)

El módulo es agnóstico del backend de memoria. Trabaja con (id, signals).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════
# Signals
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PruneSignals:
    """Señales crudas para un item de memoria.

    frequency:        accesos acumulados (raw count o suavizado).
    last_access_age:  segundos desde el último acceso.
    utility:          [0, 1] — reward medio del item.
    retrieval_cost:   costo de reconstruirlo (tokens, saltos, tiempo).
    """
    frequency: float
    last_access_age: float
    utility: float
    retrieval_cost: float = 0.0

    def normalize(
        self,
        max_freq: float,
        max_cost: float,
        half_life_sec: float,
    ) -> Tuple[float, float, float, float]:
        """Convierte señales crudas a 4 escalares en [0, 1]."""
        # S1: frecuencia normalizada por la más alta del batch
        s1 = min(self.frequency / max(max_freq, 1e-9), 1.0)
        # S2: recencia — decaimiento exponencial (half_life_sec → 0.5)
        if half_life_sec <= 0:
            s2 = 0.0
        else:
            s2 = math.exp(-self.last_access_age * math.log(2.0) / half_life_sec)
        # S3: utilidad clampada
        s3 = max(0.0, min(1.0, self.utility))
        # S4: costo normalizado por el más caro del batch. Items caros se
        # protegen (mantener ↑) porque son difíciles de reconstruir.
        s4 = min(self.retrieval_cost / max(max_cost, 1e-9), 1.0)
        return s1, s2, s3, s4


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PruneConfig:
    """Pesos y umbrales del pruning.

    Los pesos deben sumar ~1.0 (no lo enforza el código: el caller decide).
    Los umbrales dividen el rango [0,1] en 4 bandas:
        [0, delete]          → DELETE
        [delete, compress]   → COMPRESS
        [compress, promote)  → KEEP
        [promote, 1]         → PROMOTE
    """
    w_frequency: float = 0.25
    w_recency: float = 0.25
    w_utility: float = 0.35
    w_cost: float = 0.15

    delete_threshold: float = 0.15
    compress_threshold: float = 0.35
    promote_threshold: float = 0.80

    half_life_sec: float = 7 * 86400.0   # 1 semana
    ttl_min_sec: float = 3600.0          # 1 hora
    ttl_max_sec: float = 30 * 86400.0    # 30 días

    def __post_init__(self) -> None:
        if not (self.delete_threshold <= self.compress_threshold <= self.promote_threshold):
            raise ValueError(
                "thresholds must be ordered: delete <= compress <= promote"
            )
        for w in (self.w_frequency, self.w_recency, self.w_utility, self.w_cost):
            if w < 0:
                raise ValueError("weights must be non-negative")
        if self.ttl_min_sec < 0 or self.ttl_max_sec < self.ttl_min_sec:
            raise ValueError("ttl bounds invalid")


# ════════════════════════════════════════════════════════════════════════════
# Actions / decisions / report
# ════════════════════════════════════════════════════════════════════════════

class PruneAction(str, Enum):
    KEEP = "keep"
    PROMOTE = "promote"
    COMPRESS = "compress"
    DELETE = "delete"


@dataclass
class PruneDecision:
    item_id: str
    retain_score: float
    action: PruneAction
    ttl_seconds: float
    signals: PruneSignals

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "retain_score": self.retain_score,
            "action": self.action.value,
            "ttl_seconds": self.ttl_seconds,
            "signals": asdict(self.signals),
        }


@dataclass
class PruneReport:
    decisions: List[PruneDecision] = field(default_factory=list)

    def by_action(self, action: PruneAction) -> List[PruneDecision]:
        return [d for d in self.decisions if d.action == action]

    @property
    def kept(self) -> List[str]:
        return [d.item_id for d in self.by_action(PruneAction.KEEP)]

    @property
    def promoted(self) -> List[str]:
        return [d.item_id for d in self.by_action(PruneAction.PROMOTE)]

    @property
    def compressed(self) -> List[str]:
        return [d.item_id for d in self.by_action(PruneAction.COMPRESS)]

    @property
    def deleted(self) -> List[str]:
        return [d.item_id for d in self.by_action(PruneAction.DELETE)]

    def stats(self) -> Dict[str, Any]:
        return {
            "total": len(self.decisions),
            "kept": len(self.kept),
            "promoted": len(self.promoted),
            "compressed": len(self.compressed),
            "deleted": len(self.deleted),
            "mean_retain_score": (
                sum(d.retain_score for d in self.decisions) / len(self.decisions)
                if self.decisions else 0.0
            ),
        }


# ════════════════════════════════════════════════════════════════════════════
# Pruner
# ════════════════════════════════════════════════════════════════════════════

class MemoryPruner:
    """Aplica las reglas de pruning de 4 señales sobre un conjunto de items."""

    def __init__(self, config: Optional[PruneConfig] = None) -> None:
        self.config = config or PruneConfig()

    # ── Scoring ───────────────────────────────────────────────────────────
    def retain_score(
        self,
        signals: PruneSignals,
        max_freq: float,
        max_cost: float,
    ) -> float:
        c = self.config
        s1, s2, s3, s4 = signals.normalize(max_freq, max_cost, c.half_life_sec)
        score = (
            c.w_frequency * s1 +
            c.w_recency * s2 +
            c.w_utility * s3 +
            c.w_cost * s4
        )
        total_w = c.w_frequency + c.w_recency + c.w_utility + c.w_cost
        if total_w <= 0:
            return 0.0
        return max(0.0, min(1.0, score / total_w))

    def _action_for(self, score: float) -> PruneAction:
        c = self.config
        if score < c.delete_threshold:
            return PruneAction.DELETE
        if score < c.compress_threshold:
            return PruneAction.COMPRESS
        if score >= c.promote_threshold:
            return PruneAction.PROMOTE
        return PruneAction.KEEP

    def _ttl_for(self, score: float) -> float:
        """TTL dinámico: interpolación lineal entre ttl_min y ttl_max según score."""
        c = self.config
        clamped = max(0.0, min(1.0, score))
        return c.ttl_min_sec + clamped * (c.ttl_max_sec - c.ttl_min_sec)

    # ── Decide / prune ────────────────────────────────────────────────────
    def decide(
        self,
        item_id: str,
        signals: PruneSignals,
        max_freq: float,
        max_cost: float,
    ) -> PruneDecision:
        score = self.retain_score(signals, max_freq, max_cost)
        return PruneDecision(
            item_id=item_id,
            retain_score=score,
            action=self._action_for(score),
            ttl_seconds=self._ttl_for(score),
            signals=signals,
        )

    def prune(
        self,
        items: Iterable[Tuple[str, PruneSignals]],
    ) -> PruneReport:
        items_list = list(items)
        if not items_list:
            return PruneReport()

        max_freq = max(s.frequency for _, s in items_list)
        max_cost = max(s.retrieval_cost for _, s in items_list)

        decisions = [
            self.decide(item_id, signals, max_freq, max_cost)
            for item_id, signals in items_list
        ]
        return PruneReport(decisions=decisions)


# ════════════════════════════════════════════════════════════════════════════
# Sleep cycle adaptador
# ════════════════════════════════════════════════════════════════════════════

def sleep_prune_hook(
    pruner: Optional[MemoryPruner] = None,
):
    """Devuelve un prune_hook compatible con SleepCycle (Parte 23).

    Se usa como:
        cycle = SleepCycle(buffer, prune_hook=sleep_prune_hook(pruner))

    El hook convierte los episodios + scores previos de la fase 'score' en
    PruneSignals sintéticas y devuelve un dict con kept/removed y el
    informe completo para logging en el sleep log.
    """
    import time as _time
    p = pruner or MemoryPruner()

    def _hook(episodes, prev):
        scores = prev.get("score", {}).get("scores", {})
        items: List[Tuple[str, PruneSignals]] = []
        now = _time.time()
        for i, ep in enumerate(episodes):
            utility = float(scores.get(i, 0.5))
            age = max(0.0, now - ep.timestamp)
            items.append((
                f"ep_{i}",
                PruneSignals(
                    frequency=1.0,  # 1 acceso — es un episodio crudo recién
                    last_access_age=age,
                    utility=utility,
                    retrieval_cost=float(len(ep.user_text) + len(ep.aion_response)),
                ),
            ))
        report = p.prune(items)
        return {
            "kept": len(report.kept),
            "removed": len(report.deleted),
            "promoted": len(report.promoted),
            "compressed": len(report.compressed),
            "stats": report.stats(),
            "source": "pruner",
        }

    return _hook
