"""
tests/test_planner.py — Tests para agent/planner.py (Parte 6 del MEGA-PROMPT)
==============================================================================

Cubre:
  PlanStep / StepResult — estado, transiciones, serialización
  Plan                  — current_step, completed/pending/failed, progress
                          serialización JSON, timeout, render
  default_decompose     — separadores comunes, edge cases
  Planner               — plan(), execute() con éxito, retries por verifier,
                          re-planificación tras fallo, max_attempts agotado,
                          timeout en ejecución, attach_to_mem / load_from_mem
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from agent.planner import (
    Plan, PlanStep, StepResult,
    Planner, default_decompose,
    STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_COMPLETED,
    STATUS_FAILED, STATUS_SKIPPED,
    PLAN_STATUS_DRAFT, PLAN_STATUS_RUNNING, PLAN_STATUS_COMPLETED,
    PLAN_STATUS_FAILED, PLAN_STATUS_TIMED_OUT,
)


# ─────────────────────────────────────────────────────────────────────────────
# FakeMem para los tests de persistencia
# ─────────────────────────────────────────────────────────────────────────────


class FakeMem:
    def __init__(self):
        self.entries = {}

    def store(self, key, value, domain="general", source="test"):
        self.entries[key] = (value, domain)

    def search(self, query, top_k=5, domain=None):
        out = []
        for k, (v, d) in self.entries.items():
            if k == query or query in k:
                out.append((k, v, 1.0))
        return out[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# StepResult / PlanStep
# ─────────────────────────────────────────────────────────────────────────────


class TestStepResult:
    def test_default_values(self):
        r = StepResult(success=True)
        assert r.success is True
        assert r.output == ""
        assert r.error == ""
        assert r.elapsed == 0.0


class TestPlanStep:
    def test_initial_state(self):
        s = PlanStep(id=1, description="crear DB")
        assert s.status == STATUS_PENDING
        assert s.attempts == 0
        assert s.result is None

    def test_to_from_dict_roundtrip(self):
        s = PlanStep(
            id=2,
            description="x",
            status=STATUS_COMPLETED,
            attempts=1,
            result=StepResult(success=True, output="ok", elapsed=0.5),
        )
        d = s.to_dict()
        s2 = PlanStep.from_dict(d)
        assert s2.id == 2
        assert s2.status == STATUS_COMPLETED
        assert s2.attempts == 1
        assert s2.result.success is True
        assert s2.result.output == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Plan
# ─────────────────────────────────────────────────────────────────────────────


class TestPlan:
    def test_empty_plan_has_no_current_step(self):
        p = Plan(task="x")
        assert p.current_step is None
        assert p.is_complete is False
        assert p.progress == 0.0

    def test_current_step_returns_first_pending(self):
        p = Plan(task="x", steps=[
            PlanStep(id=1, description="a", status=STATUS_COMPLETED),
            PlanStep(id=2, description="b"),
            PlanStep(id=3, description="c"),
        ])
        assert p.current_step.id == 2

    def test_completed_pending_failed_lists(self):
        p = Plan(task="x", steps=[
            PlanStep(id=1, description="a", status=STATUS_COMPLETED),
            PlanStep(id=2, description="b", status=STATUS_FAILED),
            PlanStep(id=3, description="c", status=STATUS_PENDING),
        ])
        assert [s.id for s in p.completed] == [1]
        assert [s.id for s in p.failed]    == [2]
        assert [s.id for s in p.pending]   == [3]

    def test_progress_fraction(self):
        p = Plan(task="x", steps=[
            PlanStep(id=1, description="a", status=STATUS_COMPLETED),
            PlanStep(id=2, description="b", status=STATUS_COMPLETED),
            PlanStep(id=3, description="c"),
            PlanStep(id=4, description="d"),
        ])
        assert p.progress == 0.5

    def test_is_complete_only_when_all_completed(self):
        p = Plan(task="x", steps=[
            PlanStep(id=1, description="a", status=STATUS_COMPLETED),
            PlanStep(id=2, description="b", status=STATUS_COMPLETED),
        ])
        assert p.is_complete

    def test_to_from_json_roundtrip(self):
        p = Plan(
            task="hacer algo",
            steps=[PlanStep(id=1, description="paso uno"), PlanStep(id=2, description="paso dos")],
            context={"k": "v"},
            timeout_seconds=120.0,
            replan_count=1,
        )
        s = p.to_json()
        p2 = Plan.from_json(s)
        assert p2.task == "hacer algo"
        assert len(p2.steps) == 2
        assert p2.context == {"k": "v"}
        assert p2.timeout_seconds == 120.0
        assert p2.replan_count == 1

    def test_to_dict_exposes_helper_fields(self):
        p = Plan(task="x", steps=[
            PlanStep(id=1, description="a", status=STATUS_COMPLETED),
            PlanStep(id=2, description="b"),
        ])
        d = p.to_dict()
        assert d["current_step"] == 2
        assert d["completed_ids"] == [1]
        assert d["pending_ids"]   == [2]
        assert d["progress"] == 0.5

    def test_render_human_friendly(self):
        p = Plan(task="x", steps=[
            PlanStep(id=1, description="crear DB", status=STATUS_COMPLETED),
            PlanStep(id=2, description="backend", status=STATUS_IN_PROGRESS),
            PlanStep(id=3, description="frontend"),
        ])
        out = p.render()
        assert "Plan para: x" in out
        assert "Paso 1: crear DB" in out
        assert "Paso 2: backend" in out
        assert "Paso 3: frontend" in out

    def test_timeout_not_set_means_no_expiry(self):
        p = Plan(task="x", started_at=time.time())
        assert p.time_remaining() is None
        assert not p.is_timed_out()

    def test_timeout_after_zero_seconds(self):
        p = Plan(task="x", timeout_seconds=0.0, started_at=time.time() - 0.1)
        assert p.is_timed_out()


# ─────────────────────────────────────────────────────────────────────────────
# default_decompose
# ─────────────────────────────────────────────────────────────────────────────


class TestDefaultDecompose:
    def test_empty_returns_empty(self):
        assert default_decompose("", {}) == []

    def test_single_step_returns_single(self):
        assert default_decompose("una sola tarea", {}) == ["una sola tarea"]

    def test_splits_on_luego(self):
        out = default_decompose("crear DB luego backend", {})
        assert out == ["crear DB", "backend"]

    def test_splits_on_then(self):
        out = default_decompose("install deps then run tests", {})
        assert out == ["install deps", "run tests"]

    def test_splits_on_newlines_when_multiple(self):
        out = default_decompose("- paso 1\n- paso 2\n- paso 3", {})
        assert out == ["paso 1", "paso 2", "paso 3"]


# ─────────────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────────────


class TestPlannerCreation:
    def test_plan_with_default_decomposer(self):
        planner = Planner()
        plan = planner.plan("crear DB luego backend luego frontend")
        assert len(plan.steps) == 3
        assert plan.steps[0].description == "crear DB"
        assert plan.status == PLAN_STATUS_DRAFT

    def test_plan_with_custom_decomposer(self):
        def fake(task, ctx):
            return ["a", "b", "c"]
        planner = Planner(decompose_fn=fake)
        plan = planner.plan("any task")
        assert [s.description for s in plan.steps] == ["a", "b", "c"]

    def test_plan_with_timeout(self):
        planner = Planner()
        plan = planner.plan("una tarea", timeout_seconds=30.0)
        assert plan.timeout_seconds == 30.0

    def test_plan_with_context(self):
        planner = Planner()
        plan = planner.plan("una tarea", context={"user": "jesus"})
        assert plan.context["user"] == "jesus"


class TestPlannerExecute:
    def test_executes_all_steps_successfully(self):
        executed = []
        def executor(step, plan):
            executed.append(step.id)
            return StepResult(success=True, output=f"done {step.id}")
        planner = Planner(decompose_fn=lambda t, c: ["a", "b", "c"])
        plan = planner.plan("xy")
        plan = planner.execute(plan, executor_fn=executor)
        assert plan.status == PLAN_STATUS_COMPLETED
        assert executed == [1, 2, 3]
        assert plan.is_complete
        assert all(s.attempts == 1 for s in plan.steps)

    def test_verifier_can_reject_and_force_retry(self):
        attempts = {"n": 0}
        def executor(step, plan):
            attempts["n"] += 1
            return StepResult(success=True, output="x")
        def verifier(step, result):
            # Acepta solo en el 2do intento del paso 1
            return step.attempts >= 2
        planner = Planner(decompose_fn=lambda t, c: ["only"], max_attempts=3)
        plan = planner.execute(planner.plan("x"), executor, verifier)
        assert plan.status == PLAN_STATUS_COMPLETED
        assert plan.steps[0].attempts == 2

    def test_step_fails_after_max_attempts(self):
        def executor(step, plan):
            return StepResult(success=False, error="boom")
        planner = Planner(decompose_fn=lambda t, c: ["x"], max_attempts=2, max_replans=0)
        plan = planner.execute(planner.plan("x"), executor)
        assert plan.status == PLAN_STATUS_FAILED
        assert plan.steps[0].status == STATUS_FAILED
        assert plan.steps[0].attempts == 2

    def test_replan_on_failure(self):
        # Decomposer devuelve algo diferente la 2da vez (después del fallo)
        calls = {"n": 0}
        def decomposer(task, ctx):
            calls["n"] += 1
            if calls["n"] == 1:
                return ["dies"]  # primer plan: 1 paso que va a fallar
            return ["alt"]       # re-plan: 1 paso alternativo
        def executor(step, plan):
            if step.description == "dies":
                return StepResult(success=False, error="nope")
            return StepResult(success=True)
        planner = Planner(decompose_fn=decomposer, max_attempts=1, max_replans=1)
        plan = planner.execute(planner.plan("x"), executor)
        assert plan.replan_count == 1
        assert plan.status == PLAN_STATUS_COMPLETED
        # El paso fallido permanece como audit trail (marcado SKIPPED)
        # y los nuevos pasos del re-plan se anexan
        descs = [s.description for s in plan.steps]
        assert "dies" in descs and "alt" in descs
        dies_step = next(s for s in plan.steps if s.description == "dies")
        assert dies_step.status == STATUS_SKIPPED
        alt_step = next(s for s in plan.steps if s.description == "alt")
        assert alt_step.status == STATUS_COMPLETED

    def test_replan_uses_failure_context(self):
        captured_ctx = {}
        def decomposer(task, ctx):
            if "last_failure" in ctx:
                captured_ctx.update(ctx["last_failure"])
                return ["recovery"]
            return ["original"]
        def executor(step, plan):
            if step.description == "original":
                return StepResult(success=False, error="db down")
            return StepResult(success=True)
        planner = Planner(decompose_fn=decomposer, max_attempts=1, max_replans=1)
        plan = planner.execute(planner.plan("x"), executor)
        assert captured_ctx.get("error") == "db down"
        assert plan.status == PLAN_STATUS_COMPLETED

    def test_executor_exception_treated_as_failure(self):
        def executor(step, plan):
            raise RuntimeError("kaboom")
        planner = Planner(decompose_fn=lambda t, c: ["x"], max_attempts=1, max_replans=0)
        plan = planner.execute(planner.plan("x"), executor)
        assert plan.status == PLAN_STATUS_FAILED
        assert "kaboom" in plan.steps[0].result.error

    def test_timeout_during_execution(self):
        def slow_executor(step, plan):
            time.sleep(0.05)
            return StepResult(success=True)
        planner = Planner(decompose_fn=lambda t, c: ["a", "b", "c", "d", "e"])
        plan = planner.plan("x", timeout_seconds=0.06)
        plan = planner.execute(plan, executor_fn=slow_executor)
        assert plan.status == PLAN_STATUS_TIMED_OUT
        # Debe haber completado al menos 1 paso antes del timeout
        assert len(plan.completed) >= 1
        # Y haber dejado pasos pendientes (no se perdió el progreso)
        assert len(plan.pending) >= 1

    def test_records_started_and_ended_time(self):
        def executor(step, plan):
            return StepResult(success=True)
        planner = Planner(decompose_fn=lambda t, c: ["a"])
        plan = planner.execute(planner.plan("x"), executor)
        assert plan.started_at is not None
        assert plan.ended_at   is not None
        assert plan.ended_at >= plan.started_at

    def test_step_result_records_elapsed(self):
        def executor(step, plan):
            time.sleep(0.01)
            return StepResult(success=True, output="ok")
        planner = Planner(decompose_fn=lambda t, c: ["a"])
        plan = planner.execute(planner.plan("x"), executor)
        assert plan.steps[0].result.elapsed > 0.0


class TestPlannerMemPersistence:
    def test_attach_to_mem_writes_json(self):
        mem = FakeMem()
        planner = Planner(decompose_fn=lambda t, c: ["a", "b"])
        plan = planner.plan("hacer cosa")
        planner.attach_to_mem(plan, mem)
        assert "current_task" in mem.entries
        value, domain = mem.entries["current_task"]
        assert domain == "planner"
        assert "hacer cosa" in value

    def test_load_from_mem_roundtrip(self):
        mem = FakeMem()
        planner = Planner(decompose_fn=lambda t, c: ["paso1", "paso2"])
        plan = planner.plan("X", context={"foo": "bar"})
        planner.attach_to_mem(plan, mem)
        loaded = planner.load_from_mem(mem)
        assert loaded is not None
        assert loaded.task == "X"
        assert len(loaded.steps) == 2
        assert loaded.context == {"foo": "bar"}

    def test_load_from_mem_returns_none_when_empty(self):
        planner = Planner()
        assert planner.load_from_mem(FakeMem()) is None

    def test_attach_to_mem_handles_none(self):
        # No debe lanzar
        Planner().attach_to_mem(Plan(task="x"), None)
