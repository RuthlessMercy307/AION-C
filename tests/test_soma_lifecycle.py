"""
tests/test_soma_lifecycle.py — Tests para Parte 10 del MEGA-PROMPT
====================================================================

Cubre:
  10.1 SOMA interface — SomaCommand, SomaInterface, MockSomaBackend, SomaCommandTool
  10.2 Estado 24/7   — SystemState, LifecycleManager, transiciones, callbacks
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from soma import (
    SomaCommand, SomaCommandType, SomaResult,
    SomaInterface, MockSomaBackend, SomaCommandTool,
)
from agent.lifecycle import (
    SystemState, StateTransition, InvalidTransition,
    LifecycleManager, ALLOWED_TRANSITIONS,
)
from agent.tools import ToolResult
from agent.tool_executor import ToolCall, ToolExecutor


# ─────────────────────────────────────────────────────────────────────────────
# SOMA interface
# ─────────────────────────────────────────────────────────────────────────────


class TestSomaCommand:
    def test_to_from_dict(self):
        c = SomaCommand(type=SomaCommandType.HIGH_LEVEL,
                        command="open_file", args={"path": "x.py"})
        d = c.to_dict()
        c2 = SomaCommand.from_dict(d)
        assert c2.type == SomaCommandType.HIGH_LEVEL
        assert c2.command == "open_file"
        assert c2.args == {"path": "x.py"}

    def test_from_dict_string_type(self):
        c = SomaCommand.from_dict({"type": "primitive", "command": "stroke"})
        assert c.type == SomaCommandType.PRIMITIVE

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            SomaCommand.from_dict({"type": "alien", "command": "x"})

    def test_missing_command_raises(self):
        with pytest.raises(ValueError):
            SomaCommand.from_dict({"type": "primitive"})


class TestMockBackend:
    def test_records_history(self):
        backend = MockSomaBackend()
        c = SomaCommand(type=SomaCommandType.PRIMITIVE, command="stroke")
        result = backend.execute(c)
        assert result.success
        assert len(backend.history) == 1
        assert backend.history[0] is c


class TestSomaInterface:
    def test_execute_via_payload(self):
        iface = SomaInterface()
        result = iface.execute({
            "type": "high_level",
            "command": "open_file",
            "args": {"path": "app.py"},
        })
        assert result.success
        assert "open_file" in result.output

    def test_invalid_payload_returns_error(self):
        iface = SomaInterface()
        result = iface.execute({"type": "alien", "command": "x"})
        assert not result.success
        assert "invalid" in result.error.lower()

    def test_backend_exception_caught(self):
        class Boom:
            def execute(self, c): raise RuntimeError("kaboom")
        iface = SomaInterface(backend=Boom())
        result = iface.execute({"type": "primitive", "command": "stroke"})
        assert not result.success
        assert "kaboom" in result.error

    def test_default_backend_is_mock(self):
        iface = SomaInterface()
        assert isinstance(iface.backend, MockSomaBackend)


class TestSomaCommandTool:
    def test_runs_via_tool_interface(self):
        tool = SomaCommandTool()
        result = tool.run({"type": "high_level", "command": "open_file"})
        assert isinstance(result, ToolResult)
        assert result.ok

    def test_empty_payload_fails(self):
        tool = SomaCommandTool()
        result = tool.run({})
        assert not result.ok

    def test_integrates_with_tool_executor(self):
        tool = SomaCommandTool()
        executor = ToolExecutor({"soma_command": tool})
        text = (
            '[TOOL: {"action":"soma_command",'
            '"input":{"type":"high_level","command":"open_file","args":{"path":"x.py"}}}]'
        )
        records = executor.run_from_text(text)
        assert len(records) == 1
        assert records[0].success


# ─────────────────────────────────────────────────────────────────────────────
# LifecycleManager
# ─────────────────────────────────────────────────────────────────────────────


class TestLifecycleBasic:
    def test_initial_state_is_idle(self):
        lm = LifecycleManager()
        assert lm.state == SystemState.IDLE

    def test_custom_initial_state(self):
        lm = LifecycleManager(initial_state=SystemState.ACTIVE)
        assert lm.state == SystemState.ACTIVE

    def test_can_transition_from_idle(self):
        lm = LifecycleManager()
        assert lm.can_transition(SystemState.ACTIVE)
        assert lm.can_transition(SystemState.LEARNING)
        assert lm.can_transition(SystemState.SLEEPING)
        assert not lm.can_transition(SystemState.IDLE)  # mismo

    def test_transition_idle_to_active(self):
        lm = LifecycleManager()
        t = lm.transition(SystemState.ACTIVE, reason="user query")
        assert lm.state == SystemState.ACTIVE
        assert t.from_state == SystemState.IDLE
        assert t.to_state == SystemState.ACTIVE
        assert t.reason == "user query"

    def test_invalid_transition_raises(self):
        lm = LifecycleManager(initial_state=SystemState.ACTIVE)
        with pytest.raises(InvalidTransition):
            lm.transition(SystemState.LEARNING)  # ACTIVE no puede ir a LEARNING

    def test_force_transition_skips_rules(self):
        lm = LifecycleManager(initial_state=SystemState.ACTIVE)
        lm.force_transition(SystemState.LEARNING)
        assert lm.state == SystemState.LEARNING


class TestLifecycleSemanticAPI:
    def test_start_responding(self):
        lm = LifecycleManager()
        lm.start_responding()
        assert lm.state == SystemState.ACTIVE

    def test_stop_responding(self):
        lm = LifecycleManager(initial_state=SystemState.ACTIVE)
        lm.stop_responding()
        assert lm.state == SystemState.IDLE

    def test_start_learning_from_idle(self):
        lm = LifecycleManager()
        lm.start_learning()
        assert lm.state == SystemState.LEARNING

    def test_wake_up_from_sleeping(self):
        lm = LifecycleManager()
        lm.go_to_sleep()
        lm.wake_up()
        assert lm.state == SystemState.ACTIVE

    def test_wake_up_from_learning(self):
        lm = LifecycleManager()
        lm.start_learning()
        lm.wake_up()
        assert lm.state == SystemState.ACTIVE

    def test_wake_up_when_already_active_noop(self):
        lm = LifecycleManager(initial_state=SystemState.ACTIVE)
        t = lm.wake_up()
        assert lm.state == SystemState.ACTIVE
        assert t.reason == "already active"


class TestLifecycleHistory:
    def test_history_records_transitions(self):
        lm = LifecycleManager()
        lm.start_responding()
        lm.stop_responding()
        lm.start_learning()
        assert len(lm.history) == 3
        assert lm.history[0].to_state == SystemState.ACTIVE
        assert lm.history[1].to_state == SystemState.IDLE
        assert lm.history[2].to_state == SystemState.LEARNING

    def test_history_truncation(self):
        lm = LifecycleManager(max_history=3)
        for _ in range(10):
            lm.start_responding()
            lm.stop_responding()
        assert len(lm.history) == 3

    def test_stats(self):
        lm = LifecycleManager()
        lm.start_responding()
        lm.stop_responding()
        s = lm.stats()
        assert s["current_state"] == "idle"
        assert s["transitions_to"]["active"] == 1
        assert s["transitions_to"]["idle"] == 1


class TestLifecycleCallbacks:
    def test_on_enter_callback(self):
        lm = LifecycleManager()
        captured = []
        lm.on_enter(SystemState.ACTIVE, lambda mgr, t: captured.append(t.to_state))
        lm.start_responding()
        assert captured == [SystemState.ACTIVE]

    def test_on_exit_callback(self):
        lm = LifecycleManager()
        captured = []
        lm.on_exit(SystemState.IDLE, lambda mgr, t: captured.append(t.from_state))
        lm.start_responding()
        assert captured == [SystemState.IDLE]

    def test_callback_exception_doesnt_block_transition(self):
        lm = LifecycleManager()
        def boom(mgr, t): raise RuntimeError("oops")
        lm.on_enter(SystemState.ACTIVE, boom)
        lm.start_responding()  # no debe propagar
        assert lm.state == SystemState.ACTIVE


class TestAllowedTransitions:
    def test_active_only_to_idle(self):
        assert ALLOWED_TRANSITIONS[SystemState.ACTIVE] == {SystemState.IDLE}

    def test_idle_can_to_three(self):
        assert SystemState.ACTIVE in ALLOWED_TRANSITIONS[SystemState.IDLE]
        assert SystemState.LEARNING in ALLOWED_TRANSITIONS[SystemState.IDLE]
        assert SystemState.SLEEPING in ALLOWED_TRANSITIONS[SystemState.IDLE]

    def test_learning_can_interrupt_to_active(self):
        assert SystemState.ACTIVE in ALLOWED_TRANSITIONS[SystemState.LEARNING]

    def test_sleeping_can_interrupt_to_active(self):
        assert SystemState.ACTIVE in ALLOWED_TRANSITIONS[SystemState.SLEEPING]
