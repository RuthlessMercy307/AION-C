"""
tests/test_goals.py — Tests para Parte 17 del MEGA-PROMPT (Goals/Tasks/Missions)
==================================================================================
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from agent.goals import (
    GoalsManager, GoalStatus, GoalSource,
    Goal, Task, Mission, PERMANENT_MISSION,
)


# ─────────────────────────────────────────────────────────────────────────────
# Goals
# ─────────────────────────────────────────────────────────────────────────────


class TestGoals:
    def test_add_user_goal_active_immediately(self):
        gm = GoalsManager()
        g = gm.add_goal(title="learn rust", source=GoalSource.USER.value)
        assert g.status == GoalStatus.ACTIVE.value
        assert g.id in gm.goals

    def test_add_proposed_goal_pending(self):
        gm = GoalsManager()
        g = gm.add_goal(title="improve math", source=GoalSource.PROPOSED.value)
        assert g.status == GoalStatus.PENDING.value
        assert len(gm.list_active_goals()) == 0
        assert len(gm.list_proposed_goals()) == 1

    def test_approve_proposed_goal(self):
        gm = GoalsManager()
        g = gm.add_goal(title="x", source=GoalSource.PROPOSED.value)
        ok = gm.approve_goal(g.id)
        assert ok is True
        assert gm.goals[g.id].status == GoalStatus.ACTIVE.value
        assert gm.goals[g.id].source == GoalSource.USER.value

    def test_reject_proposed_goal(self):
        gm = GoalsManager()
        g = gm.add_goal(title="x", source=GoalSource.PROPOSED.value)
        ok = gm.reject_goal(g.id)
        assert ok is True
        assert gm.goals[g.id].status == GoalStatus.CANCELED.value

    def test_progress_update_completes_goal(self):
        gm = GoalsManager()
        g = gm.add_goal(title="x")
        gm.update_goal_progress(g.id, 0.5)
        assert gm.goals[g.id].progress == 0.5
        assert gm.goals[g.id].status == GoalStatus.ACTIVE.value
        gm.update_goal_progress(g.id, 1.0)
        assert gm.goals[g.id].status == GoalStatus.COMPLETED.value

    def test_progress_clamped(self):
        gm = GoalsManager()
        g = gm.add_goal(title="x")
        gm.update_goal_progress(g.id, 5.0)
        assert gm.goals[g.id].progress == 1.0
        gm.update_goal_progress(g.id, -1.0)
        assert gm.goals[g.id].progress == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────────────────


class TestTasks:
    def test_add_task_pending(self):
        gm = GoalsManager()
        t = gm.add_task(title="quick task")
        assert t.status == GoalStatus.PENDING.value
        assert t in gm.list_pending_tasks()

    def test_complete_task(self):
        gm = GoalsManager()
        t = gm.add_task(title="x")
        ok = gm.complete_task(t.id)
        assert ok is True
        assert gm.tasks[t.id].status == GoalStatus.COMPLETED.value
        assert gm.tasks[t.id].completed_at is not None

    def test_housekeeping_task_auto_source(self):
        gm = GoalsManager()
        t = gm.add_housekeeping_task("indexar MEM")
        assert t.source == GoalSource.AUTO.value


# ─────────────────────────────────────────────────────────────────────────────
# Missions
# ─────────────────────────────────────────────────────────────────────────────


class TestMissions:
    def test_add_mission_active(self):
        gm = GoalsManager()
        m = gm.add_mission(title="build login")
        assert m.status == GoalStatus.ACTIVE.value

    def test_only_one_active_mission_at_a_time(self):
        gm = GoalsManager()
        m1 = gm.add_mission(title="first")
        m2 = gm.add_mission(title="second")
        assert gm.missions[m1.id].status == GoalStatus.PAUSED.value
        assert gm.missions[m2.id].status == GoalStatus.ACTIVE.value

    def test_pause_resume_mission(self):
        gm = GoalsManager()
        m = gm.add_mission(title="x")
        gm.pause_mission(m.id)
        assert gm.missions[m.id].status == GoalStatus.PAUSED.value
        gm.resume_mission(m.id)
        assert gm.missions[m.id].status == GoalStatus.ACTIVE.value

    def test_complete_mission(self):
        gm = GoalsManager()
        m = gm.add_mission(title="x")
        gm.complete_mission(m.id)
        assert gm.missions[m.id].status == GoalStatus.COMPLETED.value
        assert gm.missions[m.id].progress == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Routine
# ─────────────────────────────────────────────────────────────────────────────


class TestRoutine:
    def test_log_entry(self):
        gm = GoalsManager()
        gm.log_routine_entry("exam", "48/50 (96%)")
        gm.log_routine_entry("auto-learn", "3 conceptos")
        today = gm.routine_today()
        assert len(today) == 2

    def test_routine_log_capped(self):
        gm = GoalsManager()
        for i in range(150):
            gm.log_routine_entry("test", f"entry {i}")
        assert len(gm.routine_log) == 100


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot + permanent mission
# ─────────────────────────────────────────────────────────────────────────────


class TestSnapshot:
    def test_snapshot_includes_all_sections(self):
        gm = GoalsManager()
        gm.add_goal(title="g1")
        gm.add_goal(title="g2", source=GoalSource.PROPOSED.value)
        gm.add_task(title="t1")
        gm.add_mission(title="m1")
        gm.log_routine_entry("test", "x")
        snap = gm.snapshot()
        assert snap["permanent_mission"] == PERMANENT_MISSION
        assert len(snap["active_goals"]) == 1
        assert len(snap["proposed_goals"]) == 1
        assert len(snap["pending_tasks"]) == 1
        assert len(snap["active_missions"]) == 1
        assert len(snap["routine_today"]) == 1

    def test_permanent_mission_is_constant(self):
        assert PERMANENT_MISSION
        assert "useful" in PERMANENT_MISSION.lower()
