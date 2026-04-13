"""
agent/goals.py — Sistema de Goals/Tareas/Rutina (Parte 17 del MEGA-PROMPT)
============================================================================

Estructura:
  MISIÓN  permanente:  "Ser útil, aprender constantemente, mejorar"
  GOALS   largo plazo: usuario crea o auto-genera (con permiso)
  TASKS   diarias:     derivadas de goals
  MISIONES inmediatas: del usuario, pueden pausarse/retomarse

Regla de oro: AION-C propone, el usuario dispone.
  - Usuario puede crear/editar/pausar/cancelar todo
  - AION-C SOLO propone goals nuevos (nunca crea sin permiso)
  - AION-C puede crear tareas internas de housekeeping sin preguntar
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class GoalStatus(str, Enum):
    PENDING   = "pending"
    ACTIVE    = "active"
    COMPLETED = "completed"
    PAUSED    = "paused"
    CANCELED  = "canceled"


class GoalSource(str, Enum):
    USER     = "user"      # creado/aceptado por el usuario
    PROPOSED = "proposed"  # propuesto por AION-C, esperando aprobación
    AUTO     = "auto"      # housekeeping automático (no requiere permiso)


@dataclass
class Goal:
    """Un objetivo de largo plazo."""
    id:          str
    title:       str
    description: str = ""
    status:      str = GoalStatus.ACTIVE.value
    source:      str = GoalSource.USER.value
    progress:    float = 0.0   # 0..1
    deadline:    Optional[float] = None
    created_at:  float = field(default_factory=time.time)
    updated_at:  float = field(default_factory=time.time)
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Task:
    """Una tarea diaria/semanal derivada de un goal o autonoma."""
    id:        str
    title:     str
    goal_id:   Optional[str] = None
    status:    str = GoalStatus.PENDING.value
    source:    str = GoalSource.USER.value
    completed_at: Optional[float] = None
    created_at:   float = field(default_factory=time.time)
    metadata:  Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Mission:
    """Una misión inmediata del usuario, pausable/retomable."""
    id:         str
    title:      str
    status:     str = GoalStatus.ACTIVE.value
    progress:   float = 0.0
    plan_id:    Optional[str] = None  # link al Plan del Planner si aplica
    created_at: float = field(default_factory=time.time)
    paused_at:  Optional[float] = None
    metadata:   Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Constante: la misión permanente
PERMANENT_MISSION = "Be useful, learn constantly, improve."


class GoalsManager:
    """
    Gestor central de Goals / Tasks / Missions / Routine.

    Reglas:
      - add_goal con source=USER     → activo de inmediato
      - add_goal con source=PROPOSED → pendiente de aprobación
      - approve_goal(id)             → PROPOSED → ACTIVE
      - reject_goal(id)              → PROPOSED → CANCELED
      - housekeeping tasks (source=AUTO) se crean sin preguntar
    """

    def __init__(self) -> None:
        self.goals:    Dict[str, Goal]    = {}
        self.tasks:    Dict[str, Task]    = {}
        self.missions: Dict[str, Mission] = {}
        self.routine_log: List[Dict[str, Any]] = []
        self.permanent_mission = PERMANENT_MISSION

    # ── Goals ──────────────────────────────────────────────────────────

    def add_goal(
        self,
        title: str,
        description: str = "",
        source: str = GoalSource.USER.value,
        deadline: Optional[float] = None,
    ) -> Goal:
        gid = "g_" + uuid.uuid4().hex[:8]
        status = GoalStatus.PENDING.value if source == GoalSource.PROPOSED.value else GoalStatus.ACTIVE.value
        g = Goal(id=gid, title=title, description=description,
                 source=source, status=status, deadline=deadline)
        self.goals[gid] = g
        return g

    def approve_goal(self, gid: str) -> bool:
        if gid not in self.goals:
            return False
        g = self.goals[gid]
        if g.source != GoalSource.PROPOSED.value:
            return False
        g.status = GoalStatus.ACTIVE.value
        g.source = GoalSource.USER.value
        g.updated_at = time.time()
        return True

    def reject_goal(self, gid: str) -> bool:
        if gid not in self.goals:
            return False
        g = self.goals[gid]
        g.status = GoalStatus.CANCELED.value
        g.updated_at = time.time()
        return True

    def update_goal_progress(self, gid: str, progress: float) -> bool:
        if gid not in self.goals:
            return False
        self.goals[gid].progress = max(0.0, min(1.0, progress))
        self.goals[gid].updated_at = time.time()
        if self.goals[gid].progress >= 1.0:
            self.goals[gid].status = GoalStatus.COMPLETED.value
        return True

    def list_active_goals(self) -> List[Goal]:
        return [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE.value]

    def list_proposed_goals(self) -> List[Goal]:
        return [g for g in self.goals.values() if g.status == GoalStatus.PENDING.value]

    # ── Tasks ──────────────────────────────────────────────────────────

    def add_task(
        self,
        title: str,
        goal_id: Optional[str] = None,
        source: str = GoalSource.USER.value,
    ) -> Task:
        tid = "t_" + uuid.uuid4().hex[:8]
        t = Task(id=tid, title=title, goal_id=goal_id, source=source)
        self.tasks[tid] = t
        return t

    def add_housekeeping_task(self, title: str) -> Task:
        """Crea una tarea de mantenimiento sin pedir permiso."""
        return self.add_task(title=title, source=GoalSource.AUTO.value)

    def complete_task(self, tid: str) -> bool:
        if tid not in self.tasks:
            return False
        self.tasks[tid].status = GoalStatus.COMPLETED.value
        self.tasks[tid].completed_at = time.time()
        return True

    def list_pending_tasks(self) -> List[Task]:
        return [t for t in self.tasks.values() if t.status == GoalStatus.PENDING.value]

    # ── Missions ───────────────────────────────────────────────────────

    def add_mission(self, title: str, plan_id: Optional[str] = None) -> Mission:
        mid = "m_" + uuid.uuid4().hex[:8]
        # Pausa misiones activas existentes (solo una activa a la vez)
        for m in self.missions.values():
            if m.status == GoalStatus.ACTIVE.value:
                m.status = GoalStatus.PAUSED.value
                m.paused_at = time.time()
        mission = Mission(id=mid, title=title, plan_id=plan_id)
        self.missions[mid] = mission
        return mission

    def pause_mission(self, mid: str) -> bool:
        if mid not in self.missions:
            return False
        self.missions[mid].status = GoalStatus.PAUSED.value
        self.missions[mid].paused_at = time.time()
        return True

    def resume_mission(self, mid: str) -> bool:
        if mid not in self.missions:
            return False
        for m in self.missions.values():
            if m.id != mid and m.status == GoalStatus.ACTIVE.value:
                m.status = GoalStatus.PAUSED.value
                m.paused_at = time.time()
        self.missions[mid].status = GoalStatus.ACTIVE.value
        self.missions[mid].paused_at = None
        return True

    def complete_mission(self, mid: str) -> bool:
        if mid not in self.missions:
            return False
        self.missions[mid].status = GoalStatus.COMPLETED.value
        self.missions[mid].progress = 1.0
        return True

    def list_active_missions(self) -> List[Mission]:
        return [m for m in self.missions.values() if m.status == GoalStatus.ACTIVE.value]

    def list_paused_missions(self) -> List[Mission]:
        return [m for m in self.missions.values() if m.status == GoalStatus.PAUSED.value]

    # ── Routine (housekeeping diario) ──────────────────────────────────

    def log_routine_entry(self, kind: str, summary: str) -> None:
        """Registra una entrada de la rutina diaria (exam, auto-learn, training)."""
        self.routine_log.append({
            "kind":      kind,
            "summary":   summary,
            "timestamp": time.time(),
        })
        if len(self.routine_log) > 100:
            self.routine_log = self.routine_log[-100:]

    def routine_today(self) -> List[Dict[str, Any]]:
        """Devuelve las entradas de routine de las últimas 24h."""
        cutoff = time.time() - 24 * 3600
        return [e for e in self.routine_log if e["timestamp"] >= cutoff]

    # ── Snapshot ───────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        return {
            "permanent_mission": self.permanent_mission,
            "active_missions":   [m.to_dict() for m in self.list_active_missions()],
            "paused_missions":   [m.to_dict() for m in self.list_paused_missions()],
            "active_goals":      [g.to_dict() for g in self.list_active_goals()],
            "proposed_goals":    [g.to_dict() for g in self.list_proposed_goals()],
            "pending_tasks":     [t.to_dict() for t in self.list_pending_tasks()],
            "routine_today":     self.routine_today(),
        }


__all__ = [
    "GoalStatus", "GoalSource",
    "Goal", "Task", "Mission",
    "GoalsManager", "PERMANENT_MISSION",
]
