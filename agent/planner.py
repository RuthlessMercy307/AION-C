"""
agent/planner.py — Planner / Task Decomposition (Parte 6 del MEGA-PROMPT)
==========================================================================

Implementa la Parte 6 del plan:

  6.1  PLANNER
       - Detecta tareas multi-paso, las descompone en una lista numerada
       - Ejecuta paso a paso, verifica cada uno, re-planifica si falla.

  6.2  ESTADO CONVERSACIONAL ESTRUCTURADO
       - Plan persistible (a JSON / a MEM) con:
           current_task, plan, current_step, completed, pending, context
       - Permite retomar una tarea interrumpida.

  6.3  TIMEOUT
       - Un Plan puede tener un wall-clock timeout. Si se excede durante la
         ejecución, el plan se marca como "timed_out" y devuelve estado parcial
         (no se pierde el progreso).

Diseño clave:
  - El Planner NO conoce ningún modelo. Recibe callables inyectados:
       decompose_fn(task, context) -> List[str]   (textos de pasos)
       executor_fn(step, plan)     -> StepResult  (ejecuta un paso)
       verifier_fn(step, result)   -> bool        (opcional, valida resultado)
    Esto lo hace 100% testeable sin tocar el modelo real.
  - El estado del Plan vive en `Plan` (dataclass serializable).
  - `attach_to_mem(plan, mem)` lo escribe en SemanticStore con la clave
    canónica `current_task` para que la conversación pueda recuperarlo.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE ESTADO
# ─────────────────────────────────────────────────────────────────────────────

STATUS_PENDING     = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED   = "completed"
STATUS_FAILED      = "failed"
STATUS_SKIPPED     = "skipped"

PLAN_STATUS_DRAFT     = "draft"
PLAN_STATUS_RUNNING   = "running"
PLAN_STATUS_COMPLETED = "completed"
PLAN_STATUS_FAILED    = "failed"
PLAN_STATUS_TIMED_OUT = "timed_out"


# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StepResult:
    """Resultado de ejecutar un paso del plan."""
    success: bool
    output:  str = ""
    error:   str = ""
    elapsed: float = 0.0


@dataclass
class PlanStep:
    """
    Un paso individual de un plan.

    id:           índice 1-based del paso dentro del plan
    description:  texto humano del paso ("crear DB SQLite")
    status:       pending | in_progress | completed | failed | skipped
    attempts:     número de intentos (incrementado cada re-ejecución)
    result:       último StepResult (opcional)
    """
    id:          int
    description: str
    status:      str = STATUS_PENDING
    attempts:    int = 0
    result:      Optional[StepResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.result is not None:
            d["result"] = asdict(self.result)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        result = data.get("result")
        if result is not None and not isinstance(result, StepResult):
            result = StepResult(**result)
        return cls(
            id=data["id"],
            description=data["description"],
            status=data.get("status", STATUS_PENDING),
            attempts=data.get("attempts", 0),
            result=result,
        )


@dataclass
class Plan:
    """
    Estado completo de una tarea descompuesta.

    Cumple con la Parte 6.2 del MEGA-PROMPT.
    """
    task:           str
    steps:          List[PlanStep] = field(default_factory=list)
    status:         str = PLAN_STATUS_DRAFT
    context:        Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    started_at:     Optional[float] = None
    ended_at:       Optional[float] = None
    replan_count:   int = 0

    # ── propiedades de estado ───────────────────────────────────────────

    @property
    def current_step(self) -> Optional[PlanStep]:
        """El primer paso pending o in_progress."""
        for s in self.steps:
            if s.status in (STATUS_PENDING, STATUS_IN_PROGRESS):
                return s
        return None

    @property
    def completed(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status == STATUS_COMPLETED]

    @property
    def pending(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status == STATUS_PENDING]

    @property
    def failed(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status == STATUS_FAILED]

    @property
    def is_complete(self) -> bool:
        """
        Un plan se considera completado cuando:
          - tiene al menos un paso COMPLETED
          - no hay pasos pending / in_progress / failed
        Los pasos SKIPPED no bloquean la completitud (forman parte del audit
        trail después de un re-plan).
        """
        if not self.steps:
            return False
        blocking = (STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_FAILED)
        if any(s.status in blocking for s in self.steps):
            return False
        return any(s.status == STATUS_COMPLETED for s in self.steps)

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        return len(self.completed) / len(self.steps)

    def time_remaining(self) -> Optional[float]:
        """Segundos restantes antes del timeout. None si no hay timeout."""
        if self.timeout_seconds is None or self.started_at is None:
            return None
        elapsed = time.time() - self.started_at
        return max(0.0, self.timeout_seconds - elapsed)

    def is_timed_out(self) -> bool:
        rem = self.time_remaining()
        return rem is not None and rem <= 0.0

    # ── serialización (para persistir en MEM o disco) ───────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task":            self.task,
            "steps":           [s.to_dict() for s in self.steps],
            "status":          self.status,
            "context":         dict(self.context),
            "timeout_seconds": self.timeout_seconds,
            "started_at":      self.started_at,
            "ended_at":        self.ended_at,
            "replan_count":    self.replan_count,
            # Conveniencia para inspección externa (Parte 6.2)
            "current_step":    self.current_step.id if self.current_step else None,
            "completed_ids":   [s.id for s in self.completed],
            "pending_ids":     [s.id for s in self.pending],
            "progress":        self.progress,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        return cls(
            task=data["task"],
            steps=[PlanStep.from_dict(s) for s in data.get("steps", [])],
            status=data.get("status", PLAN_STATUS_DRAFT),
            context=dict(data.get("context", {})),
            timeout_seconds=data.get("timeout_seconds"),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            replan_count=data.get("replan_count", 0),
        )

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        return cls.from_dict(json.loads(text))

    # ── representación humana (para mostrar en UI / consola) ────────────

    def render(self) -> str:
        """Texto humano del plan, estilo Plan View (Parte 12.6)."""
        if not self.steps:
            return f"(plan vacío para: {self.task})"
        lines = [f"Plan para: {self.task}"]
        for s in self.steps:
            mark = {
                STATUS_PENDING:     "☐",
                STATUS_IN_PROGRESS: "▶",
                STATUS_COMPLETED:   "☑",
                STATUS_FAILED:      "✗",
                STATUS_SKIPPED:     "⊘",
            }.get(s.status, "?")
            suffix = ""
            if s.status == STATUS_FAILED and s.result and s.result.error:
                suffix = f"  (failed: {s.result.error[:60]})"
            lines.append(f"  {mark} Paso {s.id}: {s.description}{suffix}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER
# ─────────────────────────────────────────────────────────────────────────────


# Tipo de las callables inyectables
DecomposeFn = Callable[[str, Dict[str, Any]], List[str]]
ExecutorFn  = Callable[[PlanStep, Plan], StepResult]
VerifierFn  = Callable[[PlanStep, StepResult], bool]


def default_decompose(task: str, context: Dict[str, Any]) -> List[str]:
    """
    Decomposer por defecto — heurística simple.

    Divide por separadores comunes: "luego", "después", "then", "and",
    o salto de línea. NO requiere modelo. En producción inyectar un
    decomposer que llame al modelo real.
    """
    if not task:
        return []
    text = task.strip()
    # Si tiene saltos de línea numerados, respetarlos
    lines = [l.strip(" -•\t") for l in text.split("\n") if l.strip()]
    if len(lines) > 1:
        return lines
    # Dividir por separadores comunes
    seps = [" luego ", " despues ", " después ", " then ", " and ", "; ", ", "]
    parts = [text]
    for sep in seps:
        new_parts: List[str] = []
        for p in parts:
            new_parts.extend([x.strip() for x in p.split(sep) if x.strip()])
        parts = new_parts
    return parts if len(parts) > 1 else [text]


class Planner:
    """
    Orchestrator de Plans. NO conoce el modelo — recibe callables.

    Uso:
        planner = Planner(decompose_fn=my_decomposer)
        plan = planner.plan("Crear app con login")
        # Mostrar al usuario:
        print(plan.render())
        # Ejecutar:
        plan = planner.execute(plan, executor_fn=my_executor, verifier_fn=my_verifier)

    Atributos:
        decompose_fn: callable para descomponer tareas
        max_attempts: intentos máximos por paso antes de marcarlo como failed
        max_replans:  cuántas veces re-planificar después de un failure
    """

    def __init__(
        self,
        decompose_fn: Optional[DecomposeFn] = None,
        max_attempts: int = 2,
        max_replans:  int = 1,
    ) -> None:
        self.decompose_fn = decompose_fn or default_decompose
        self.max_attempts = max_attempts
        self.max_replans  = max_replans

    # ── creación del plan ──────────────────────────────────────────────

    def plan(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Plan:
        """Descompone la tarea en un Plan listo para ejecutar."""
        ctx = dict(context or {})
        descriptions = self.decompose_fn(task, ctx)
        steps = [PlanStep(id=i + 1, description=d) for i, d in enumerate(descriptions)]
        return Plan(
            task=task,
            steps=steps,
            status=PLAN_STATUS_DRAFT,
            context=ctx,
            timeout_seconds=timeout_seconds,
        )

    # ── ejecución ──────────────────────────────────────────────────────

    def execute(
        self,
        plan: Plan,
        executor_fn: ExecutorFn,
        verifier_fn: Optional[VerifierFn] = None,
    ) -> Plan:
        """
        Ejecuta el plan paso a paso.

        - Cada paso se intenta hasta `max_attempts` veces si el verifier
          (o el propio executor) lo rechaza.
        - Si un paso falla definitivamente y quedan replans, llama a
          `replan(plan, step, reason)` y reanuda.
        - Si el wall-clock excede `timeout_seconds`, marca el plan como
          timed_out y devuelve estado parcial.
        """
        if plan.started_at is None:
            plan.started_at = time.time()
        plan.status = PLAN_STATUS_RUNNING

        while True:
            step = plan.current_step
            if step is None:
                break  # nada pendiente
            if plan.is_timed_out():
                plan.status = PLAN_STATUS_TIMED_OUT
                plan.ended_at = time.time()
                return plan

            step.status = STATUS_IN_PROGRESS
            step.attempts += 1
            t0 = time.time()
            try:
                result = executor_fn(step, plan)
            except Exception as exc:
                result = StepResult(success=False, error=f"executor raised: {exc}")
            result.elapsed = time.time() - t0
            step.result = result

            ok = result.success
            if ok and verifier_fn is not None:
                try:
                    ok = bool(verifier_fn(step, result))
                except Exception:
                    ok = False
                if not ok:
                    step.result = StepResult(
                        success=False,
                        output=result.output,
                        error="verifier rejected result",
                        elapsed=result.elapsed,
                    )

            if ok:
                step.status = STATUS_COMPLETED
                continue

            # Falló este intento — reintentar si quedan attempts
            if step.attempts < self.max_attempts:
                step.status = STATUS_PENDING
                continue

            # Sin más attempts → marcamos failed
            step.status = STATUS_FAILED

            # ¿Re-planificar?
            if plan.replan_count < self.max_replans:
                self._replan(plan, step)
                continue

            # Re-plan agotado → falla el plan completo
            plan.status = PLAN_STATUS_FAILED
            plan.ended_at = time.time()
            return plan

        plan.status = PLAN_STATUS_COMPLETED if plan.is_complete else PLAN_STATUS_FAILED
        plan.ended_at = time.time()
        return plan

    def _replan(self, plan: Plan, failed_step: PlanStep) -> None:
        """
        Re-genera los pasos pendientes después de un fallo.

        Mantiene como audit trail TODOS los pasos completados y fallidos
        previos (no se borran). Descarta solo los pendientes que ya no
        aplican y agrega los nuevos pasos del re-plan.
        Incrementa `replan_count`.
        """
        plan.replan_count += 1
        ctx = dict(plan.context)
        ctx["last_failure"] = {
            "step_id":     failed_step.id,
            "description": failed_step.description,
            "error":       failed_step.result.error if failed_step.result else "",
        }
        # Marcar el paso fallido como SKIPPED para preservar el audit trail
        # sin bloquear `is_complete` cuando el re-plan tenga éxito.
        failed_step.status = STATUS_SKIPPED
        # Conservar histórico: completados + skipped (en orden original)
        history = [
            s for s in plan.steps
            if s.status in (STATUS_COMPLETED, STATUS_SKIPPED)
        ]
        new_descriptions = self.decompose_fn(plan.task, ctx)
        if not new_descriptions:
            return
        next_id = (max((s.id for s in history), default=0)) + 1
        new_steps = [
            PlanStep(id=next_id + i, description=d)
            for i, d in enumerate(new_descriptions)
        ]
        plan.steps = history + new_steps

    # ── persistencia / MEM ─────────────────────────────────────────────

    def attach_to_mem(self, plan: Plan, mem: Any, key: str = "current_task") -> None:
        """
        Persiste el plan en MEM bajo `key` (default: 'current_task').
        Cumple Parte 6.2: estado conversacional estructurado en MEM.
        """
        if mem is None:
            return
        mem.store(key, plan.to_json(), domain="planner")

    def load_from_mem(self, mem: Any, key: str = "current_task") -> Optional[Plan]:
        """Recupera el último plan persistido en MEM, o None si no hay."""
        if mem is None:
            return None
        try:
            entries = mem.search(key, top_k=1)
        except Exception:
            return None
        for item in entries or []:
            if isinstance(item, tuple) and len(item) >= 2 and item[0] == key:
                try:
                    return Plan.from_json(item[1])
                except Exception:
                    return None
        return None


__all__ = [
    "STATUS_PENDING", "STATUS_IN_PROGRESS", "STATUS_COMPLETED",
    "STATUS_FAILED", "STATUS_SKIPPED",
    "PLAN_STATUS_DRAFT", "PLAN_STATUS_RUNNING", "PLAN_STATUS_COMPLETED",
    "PLAN_STATUS_FAILED", "PLAN_STATUS_TIMED_OUT",
    "StepResult", "PlanStep", "Plan",
    "Planner", "default_decompose",
    "DecomposeFn", "ExecutorFn", "VerifierFn",
]
