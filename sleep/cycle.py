"""
sleep/cycle.py — Ritual de 6 preguntas del Sleep Cycle (Parte 23).

Diseño:
    - Las 6 preguntas tienen orden estricto — jamás cambia.
    - Cada pregunta delega a un hook inyectable. Si no hay hook, corre
      un stub que devuelve un resultado no-op trazable.
    - El ciclo es IDEMPOTENTE: puede reanudarse entre preguntas (cada
      fase se aisla y su resultado persiste antes de pasar a la siguiente).
    - Todo el ciclo produce un SleepCycleLog serializable para UI / memoria
      / interpretabilidad.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════
# Las 6 preguntas — orden inmutable
# ════════════════════════════════════════════════════════════════════════════

SLEEP_QUESTIONS: Tuple[Tuple[str, str], ...] = (
    ("recollect",    "¿Qué viví desde el último sueño?"),
    ("score",        "¿Qué fue útil y qué no?"),
    ("prune",        "¿Qué debo olvidar?"),
    ("compress",     "¿Qué debo comprimir?"),
    ("consolidate",  "¿Qué debo consolidar en los pesos?"),
    ("followups",    "¿Qué debo preguntarme mañana?"),
)


# ════════════════════════════════════════════════════════════════════════════
# Episodio + buffer
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """Una interacción de vigilia que el ciclo de sueño procesará.

    user_text:       texto del usuario.
    aion_response:   respuesta del sistema.
    timestamp:       unix ts.
    motor_sequence:  secuencia de motores usada (Parte 22.5).
    user_feedback:   señal explícita (👍 / 👎 / None) — Parte 25.
    implicit_score:  score implícito calculado por hooks de reward.
    meta:            dict libre para extensiones.
    """
    user_text: str
    aion_response: str
    timestamp: float = field(default_factory=time.time)
    motor_sequence: List[str] = field(default_factory=list)
    user_feedback: Optional[str] = None
    implicit_score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EpisodicBuffer:
    """Cola de episodios pendientes de procesar en el próximo sleep cycle.

    Capacidad máxima; por encima, el más viejo se descarta (FIFO).
    """

    def __init__(self, max_size: int = 1000) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self._episodes: List[Episode] = []

    def add(self, episode: Episode) -> None:
        self._episodes.append(episode)
        if len(self._episodes) > self.max_size:
            self._episodes = self._episodes[-self.max_size:]

    def drain(self) -> List[Episode]:
        """Devuelve TODOS los episodios y vacía el buffer."""
        out = self._episodes
        self._episodes = []
        return out

    def snapshot(self) -> List[Episode]:
        """Copia de los episodios sin vaciar el buffer."""
        return list(self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def clear(self) -> None:
        self._episodes = []


# ════════════════════════════════════════════════════════════════════════════
# Resultado del ciclo
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseResult:
    name: str
    question: str
    summary: str
    data: Dict[str, Any]
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SleepCycleLog:
    started_at: float
    ended_at: float
    trigger: str
    episodes_processed: int
    phases: List[PhaseResult]
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        return (self.ended_at - self.started_at) * 1000.0

    def phase(self, name: str) -> Optional[PhaseResult]:
        for p in self.phases:
            if p.name == name:
                return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "trigger": self.trigger,
            "episodes_processed": self.episodes_processed,
            "phases": [p.to_dict() for p in self.phases],
            "error": self.error,
        }
        return d


# ════════════════════════════════════════════════════════════════════════════
# SleepCycle — el ritual
# ════════════════════════════════════════════════════════════════════════════

# Tipos de los hooks de sub-fase. Cada uno recibe la lista de episodios
# (y outputs previos donde aplique) y devuelve un dict que se guarda en
# PhaseResult.data. Todos son OPCIONALES: si no se pasa hook, se usa stub.
ScoreHook = Callable[[List[Episode]], Dict[str, Any]]
PruneHook = Callable[[List[Episode], Dict[str, Any]], Dict[str, Any]]
CompressHook = Callable[[List[Episode], Dict[str, Any]], Dict[str, Any]]
ConsolidateHook = Callable[[List[Episode], Dict[str, Any]], Dict[str, Any]]
FollowupsHook = Callable[[List[Episode], Dict[str, Any]], List[str]]


class SleepCycle:
    """Ejecuta el ritual de 6 preguntas sobre los episodios del buffer.

    Hooks inyectables (todos opcionales):
        reward_hook:      Parte 25 (reward probabilístico)
        prune_hook:       Parte 24 (pruning 4 señales)
        compress_hook:    Parte 26 (compresión jerárquica)
        consolidate_hook: Parte 9  (auto-learn / anti-forgetting)
        followups_hook:   generación de preguntas abiertas → goals

    Cada hook recibe la lista de episodios filtrada por las fases previas
    y un dict `prev` con los resultados de las fases anteriores (por nombre).
    """

    def __init__(
        self,
        buffer: EpisodicBuffer,
        reward_hook: Optional[ScoreHook] = None,
        prune_hook: Optional[PruneHook] = None,
        compress_hook: Optional[CompressHook] = None,
        consolidate_hook: Optional[ConsolidateHook] = None,
        followups_hook: Optional[FollowupsHook] = None,
    ) -> None:
        self.buffer = buffer
        self.reward_hook = reward_hook
        self.prune_hook = prune_hook
        self.compress_hook = compress_hook
        self.consolidate_hook = consolidate_hook
        self.followups_hook = followups_hook

    # ── Entry point ───────────────────────────────────────────────────────
    def run(self, trigger: str = "manual") -> SleepCycleLog:
        started = time.time()
        episodes = self.buffer.drain()
        phases: List[PhaseResult] = []
        prev: Dict[str, Any] = {}
        error: Optional[str] = None

        try:
            # Los nombres deben corresponder con SLEEP_QUESTIONS en orden.
            for name, question in SLEEP_QUESTIONS:
                result = self._run_phase(name, question, episodes, prev)
                phases.append(result)
                prev[name] = result.data
        except Exception as exc:  # pragma: no cover — defensivo
            error = f"{type(exc).__name__}: {exc}"

        ended = time.time()
        return SleepCycleLog(
            started_at=started,
            ended_at=ended,
            trigger=trigger,
            episodes_processed=len(episodes),
            phases=phases,
            error=error,
        )

    # ── Dispatch por nombre ───────────────────────────────────────────────
    def _run_phase(
        self,
        name: str,
        question: str,
        episodes: List[Episode],
        prev: Dict[str, Any],
    ) -> PhaseResult:
        t0 = time.perf_counter()
        handlers = {
            "recollect":   self._phase_recollect,
            "score":       self._phase_score,
            "prune":       self._phase_prune,
            "compress":    self._phase_compress,
            "consolidate": self._phase_consolidate,
            "followups":   self._phase_followups,
        }
        fn = handlers[name]
        summary, data = fn(episodes, prev)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PhaseResult(
            name=name,
            question=question,
            summary=summary,
            data=data,
            elapsed_ms=elapsed_ms,
        )

    # ── Fase 1: recolección ───────────────────────────────────────────────
    def _phase_recollect(
        self, episodes: List[Episode], prev: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        # Agrupa por primera palabra del user_text como proxy de tema.
        by_topic: Dict[str, int] = {}
        for ep in episodes:
            topic = (ep.user_text.strip().split()[:1] or ["<empty>"])[0].lower()
            by_topic[topic] = by_topic.get(topic, 0) + 1
        summary = f"{len(episodes)} episodios agrupados en {len(by_topic)} temas"
        data = {
            "count": len(episodes),
            "by_topic": by_topic,
            "timestamps": [ep.timestamp for ep in episodes[:20]],
        }
        return summary, data

    # ── Fase 2: reward / utilidad ─────────────────────────────────────────
    def _phase_score(
        self, episodes: List[Episode], prev: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        if self.reward_hook is not None:
            data = self.reward_hook(episodes)
            scores = data.get("scores", {})
            mean = (sum(scores.values()) / len(scores)) if scores else 0.0
        else:
            # Stub: reward desde feedback explícito; sin feedback = 0.5
            scores: Dict[int, float] = {}
            for i, ep in enumerate(episodes):
                if ep.user_feedback == "up":
                    scores[i] = 1.0
                elif ep.user_feedback == "down":
                    scores[i] = 0.0
                else:
                    scores[i] = 0.5
            mean = (sum(scores.values()) / len(scores)) if scores else 0.0
            data = {"scores": scores, "mean": mean, "source": "stub"}
        summary = f"score medio {mean:.2f} sobre {len(episodes)} episodios"
        return summary, data

    # ── Fase 3: pruning ───────────────────────────────────────────────────
    def _phase_prune(
        self, episodes: List[Episode], prev: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        if self.prune_hook is not None:
            data = self.prune_hook(episodes, prev)
            kept = int(data.get("kept", len(episodes)))
            removed = int(data.get("removed", 0))
        else:
            scores = prev.get("score", {}).get("scores", {})
            removed_idx = [i for i, s in scores.items() if s < 0.2]
            kept = len(episodes) - len(removed_idx)
            removed = len(removed_idx)
            data = {
                "kept": kept,
                "removed": removed,
                "removed_indices": removed_idx,
                "source": "stub",
            }
        summary = f"mantenidos {kept}, descartados {removed}"
        return summary, data

    # ── Fase 4: compresión ────────────────────────────────────────────────
    def _phase_compress(
        self, episodes: List[Episode], prev: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        if self.compress_hook is not None:
            data = self.compress_hook(episodes, prev)
            clusters = int(data.get("clusters", 0))
        else:
            # Stub: cuenta clusters tomando la primera palabra como clave,
            # similar a recollect pero consolidado (N>=2 para cluster).
            by_topic: Dict[str, List[int]] = {}
            for i, ep in enumerate(episodes):
                topic = (ep.user_text.strip().split()[:1] or ["<empty>"])[0].lower()
                by_topic.setdefault(topic, []).append(i)
            clusters_data = {t: ixs for t, ixs in by_topic.items() if len(ixs) >= 2}
            clusters = len(clusters_data)
            data = {
                "clusters": clusters,
                "anchors": {t: ixs[:2] for t, ixs in clusters_data.items()},
                "source": "stub",
            }
        summary = f"{clusters} clusters formados"
        return summary, data

    # ── Fase 5: consolidación ─────────────────────────────────────────────
    def _phase_consolidate(
        self, episodes: List[Episode], prev: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        if self.consolidate_hook is not None:
            data = self.consolidate_hook(episodes, prev)
            trained = int(data.get("consolidated", 0))
        else:
            # Stub: marca como candidatos los episodios con score >= 0.7
            scores = prev.get("score", {}).get("scores", {})
            candidates = [i for i, s in scores.items() if s >= 0.7]
            trained = len(candidates)
            data = {
                "consolidated": trained,
                "candidates": candidates,
                "source": "stub",
            }
        summary = f"{trained} conceptos candidatos a consolidación"
        return summary, data

    # ── Fase 6: follow-ups ────────────────────────────────────────────────
    def _phase_followups(
        self, episodes: List[Episode], prev: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        if self.followups_hook is not None:
            questions = self.followups_hook(episodes, prev)
            source = "hook"
        else:
            # Stub: por cada cluster comprimido, genera una pregunta abierta.
            clusters = prev.get("compress", {}).get("anchors", {})
            questions = [f"¿qué más debo saber sobre {topic}?" for topic in list(clusters.keys())[:5]]
            if not questions:
                questions = ["¿hay algo que no entendí bien hoy?"]
            source = "stub"
        summary = f"{len(questions)} preguntas abiertas para mañana"
        data = {"questions": list(questions), "source": source}
        return summary, data
