"""
sleep/daemon.py — SleepDaemon: scheduler de la Parte 23.

Reglas de disparo del sleep cycle (combinables con OR):
    - MANUAL:      llamada explícita a force_run().
    - INACTIVITY:  han pasado ≥ inactivity_seconds sin notify_activity().
    - OVERFLOW:    el EpisodicBuffer superó overflow_threshold episodios.

El daemon NO corre en un thread real — simplemente expone maybe_run(now),
que decide si conviene ejecutar el ciclo AHORA. El backend llama a
maybe_run() periódicamente (p. ej. desde un endpoint de heartbeat o
dentro del WS handler).

Después de un ciclo, se guarda el último log como `last_log` para
consulta desde la UI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sleep.cycle import SleepCycle, SleepCycleLog


class SleepTrigger(str, Enum):
    MANUAL = "manual"
    INACTIVITY = "inactivity"
    OVERFLOW = "overflow"


@dataclass
class SleepDaemon:
    """Scheduler ligero para decidir cuándo correr un SleepCycle.

    cycle:                el SleepCycle a ejecutar
    inactivity_seconds:   segundos de inactividad que disparan el ciclo
    overflow_threshold:   tamaño del buffer que fuerza el ciclo aunque
                          no haya inactividad (0 = deshabilitado)
    """
    cycle: SleepCycle
    inactivity_seconds: float = 3600.0
    overflow_threshold: int = 500

    _last_activity_ts: float = 0.0
    _last_log: Optional[SleepCycleLog] = None

    def __post_init__(self) -> None:
        if self.inactivity_seconds < 0:
            raise ValueError("inactivity_seconds must be >= 0")
        if self.overflow_threshold < 0:
            raise ValueError("overflow_threshold must be >= 0")
        self._last_activity_ts = time.time()
        self._last_log = None

    # ── Activity tracking ─────────────────────────────────────────────────
    def notify_activity(self, ts: Optional[float] = None) -> None:
        """Llamar cada vez que el usuario envía una query."""
        self._last_activity_ts = ts if ts is not None else time.time()

    @property
    def last_activity_ts(self) -> float:
        return self._last_activity_ts

    @property
    def last_log(self) -> Optional[SleepCycleLog]:
        return self._last_log

    # ── Trigger detection ─────────────────────────────────────────────────
    def should_run(self, now: Optional[float] = None) -> Optional[SleepTrigger]:
        """Devuelve el trigger que dispararía el ciclo, o None si no toca."""
        now = now if now is not None else time.time()
        # Overflow: el buffer llegó al umbral
        if self.overflow_threshold > 0 and len(self.cycle.buffer) >= self.overflow_threshold:
            return SleepTrigger.OVERFLOW
        # Inactividad
        if now - self._last_activity_ts >= self.inactivity_seconds:
            # Sólo si hay algo que procesar
            if len(self.cycle.buffer) > 0:
                return SleepTrigger.INACTIVITY
        return None

    # ── Run helpers ───────────────────────────────────────────────────────
    def maybe_run(self, now: Optional[float] = None) -> Optional[SleepCycleLog]:
        """Ejecuta el ciclo si alguna condición lo dispara, si no devuelve None."""
        trigger = self.should_run(now=now)
        if trigger is None:
            return None
        return self._run(trigger)

    def force_run(self) -> SleepCycleLog:
        """Dispara el ciclo manualmente (e.g. desde /api/sleep)."""
        return self._run(SleepTrigger.MANUAL)

    def _run(self, trigger: SleepTrigger) -> SleepCycleLog:
        log = self.cycle.run(trigger=trigger.value)
        self._last_log = log
        # Reset activity — al terminar el ciclo, empieza una nueva vigilia.
        self._last_activity_ts = time.time()
        return log
