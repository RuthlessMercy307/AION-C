"""
agent/lifecycle.py — Estado 24/7 de AION-C (Parte 10.2 del MEGA-PROMPT)
=========================================================================

AION-C no es un chatbot pasivo. Es un sistema VIVO con 4 estados:

  ACTIVE   — respondiendo al usuario
  IDLE     — nadie habla, monitorea, indexa MEM
  LEARNING — auto-entrenamiento en background
  SLEEPING — entrenamiento intensivo (cuando la PC está libre)

Las transiciones permitidas son:

  ACTIVE   → IDLE     (al terminar de responder)
  IDLE     → ACTIVE   (al recibir un query)
  IDLE     → LEARNING (auto-aprendizaje arranca)
  LEARNING → IDLE     (background training termina)
  LEARNING → ACTIVE   (interrumpido por query del usuario)
  IDLE     → SLEEPING (PC libre, training nocturno)
  SLEEPING → ACTIVE   (interrumpido por query)
  SLEEPING → IDLE     (training termina)

El LifecycleManager mantiene el estado actual, el historial de transiciones
y permite registrar callbacks por estado/transición.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class SystemState(str, Enum):
    ACTIVE   = "active"
    IDLE     = "idle"
    LEARNING = "learning"
    SLEEPING = "sleeping"


# Mapa de transiciones permitidas
ALLOWED_TRANSITIONS: Dict[SystemState, Set[SystemState]] = {
    SystemState.ACTIVE:   {SystemState.IDLE},
    SystemState.IDLE:     {SystemState.ACTIVE, SystemState.LEARNING, SystemState.SLEEPING},
    SystemState.LEARNING: {SystemState.IDLE, SystemState.ACTIVE},
    SystemState.SLEEPING: {SystemState.IDLE, SystemState.ACTIVE},
}


@dataclass
class StateTransition:
    """Una transición registrada en el historial."""
    from_state: SystemState
    to_state:   SystemState
    timestamp:  float
    reason:     str = ""


class InvalidTransition(Exception):
    """Se intentó una transición no permitida."""


CallbackFn = Callable[["LifecycleManager", StateTransition], None]


class LifecycleManager:
    """
    Gestor del estado 24/7 de AION-C.

    Args:
        initial_state: estado inicial (default IDLE)
        max_history:   cuántas transiciones recordar (default 100)
    """

    def __init__(
        self,
        initial_state: SystemState = SystemState.IDLE,
        max_history:   int = 100,
    ) -> None:
        self._state = initial_state
        self.max_history = max_history
        self._history: List[StateTransition] = []
        self._on_enter: Dict[SystemState, List[CallbackFn]] = {s: [] for s in SystemState}
        self._on_exit:  Dict[SystemState, List[CallbackFn]] = {s: [] for s in SystemState}
        self._state_started_at = time.time()

    # ── estado actual ──────────────────────────────────────────────────

    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def time_in_state(self) -> float:
        return time.time() - self._state_started_at

    # ── transiciones ───────────────────────────────────────────────────

    def can_transition(self, target: SystemState) -> bool:
        return target in ALLOWED_TRANSITIONS.get(self._state, set())

    def transition(self, target: SystemState, reason: str = "") -> StateTransition:
        if not self.can_transition(target):
            raise InvalidTransition(f"cannot transition {self._state.value} → {target.value}")
        return self._do_transition(target, reason)

    def force_transition(self, target: SystemState, reason: str = "") -> StateTransition:
        """
        Fuerza una transición saltando las reglas (uso para shutdowns,
        recovery o tests). Usar con cuidado.
        """
        return self._do_transition(target, reason or "forced")

    def _do_transition(self, target: SystemState, reason: str) -> StateTransition:
        prev = self._state
        trans = StateTransition(
            from_state=prev,
            to_state=target,
            timestamp=time.time(),
            reason=reason,
        )
        # callbacks de salida
        for cb in self._on_exit.get(prev, []):
            try:
                cb(self, trans)
            except Exception:
                pass
        self._state = target
        self._state_started_at = trans.timestamp
        # historial truncado
        self._history.append(trans)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
        # callbacks de entrada
        for cb in self._on_enter.get(target, []):
            try:
                cb(self, trans)
            except Exception:
                pass
        return trans

    # ── callbacks ──────────────────────────────────────────────────────

    def on_enter(self, state: SystemState, callback: CallbackFn) -> None:
        self._on_enter[state].append(callback)

    def on_exit(self, state: SystemState, callback: CallbackFn) -> None:
        self._on_exit[state].append(callback)

    # ── shortcuts semánticos ───────────────────────────────────────────

    def start_responding(self, reason: str = "user query") -> StateTransition:
        return self.transition(SystemState.ACTIVE, reason)

    def stop_responding(self, reason: str = "response complete") -> StateTransition:
        return self.transition(SystemState.IDLE, reason)

    def start_learning(self, reason: str = "auto-learn triggered") -> StateTransition:
        return self.transition(SystemState.LEARNING, reason)

    def stop_learning(self, reason: str = "learning complete") -> StateTransition:
        return self.transition(SystemState.IDLE, reason)

    def go_to_sleep(self, reason: str = "pc idle") -> StateTransition:
        return self.transition(SystemState.SLEEPING, reason)

    def wake_up(self, reason: str = "user activity") -> StateTransition:
        # Wake puede venir de SLEEPING o LEARNING → siempre va a ACTIVE
        if self._state == SystemState.SLEEPING:
            return self.transition(SystemState.ACTIVE, reason)
        if self._state == SystemState.LEARNING:
            return self.transition(SystemState.ACTIVE, reason)
        if self._state == SystemState.IDLE:
            return self.transition(SystemState.ACTIVE, reason)
        return StateTransition(self._state, self._state, time.time(), "already active")

    # ── historial / introspección ──────────────────────────────────────

    @property
    def history(self) -> List[StateTransition]:
        return list(self._history)

    def stats(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {s.value: 0 for s in SystemState}
        for t in self._history:
            counts[t.to_state.value] += 1
        return {
            "current_state":   self._state.value,
            "time_in_state":   self.time_in_state,
            "history_length":  len(self._history),
            "transitions_to":  counts,
        }


__all__ = [
    "SystemState",
    "StateTransition",
    "InvalidTransition",
    "LifecycleManager",
    "ALLOWED_TRANSITIONS",
]
