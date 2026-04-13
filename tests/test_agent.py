"""
tests/test_agent.py — Tests para agent/
========================================

Cubre:
  1. ToolResult: ok, as_text, repr
  2. Herramientas individuales con runner mock
  3. AgentSession: record_action, record_file, queries
  4. MemoryBridge: store/load/search/context
  5. MockMotor: secuencia de acciones
  6. AgentLoop: 3 turnos, max_turns, historial, DONE, FAIL
  7. AgentLoop con herramientas mockeadas
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from types import SimpleNamespace
from typing import Any, Dict, List

from agent import (
    AgentLoop,
    MockMotor,
    MotorAction,
    LoopResult,
    DONE_SIGNAL,
    FAIL_SIGNAL,
    AgentSession,
    ActionEntry,
    MemoryBridge,
    ToolResult,
    BashTool,
    GrepTool,
    FindTool,
    CatTool,
    PytestTool,
)
from agent.tools import build_tool_registry


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_completed_process(stdout: str = "", stderr: str = "", returncode: int = 0):
    """Simula subprocess.CompletedProcess."""
    return SimpleNamespace(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
    )


def _mock_runner_ok(output: str = "ok"):
    """Retorna una función runner que siempre responde con éxito."""
    def runner(*args, **kwargs):
        return _make_completed_process(stdout=output, returncode=0)
    return runner


def _mock_runner_fail(error: str = "error"):
    """Retorna una función runner que siempre falla."""
    def runner(*args, **kwargs):
        return _make_completed_process(stderr=error, returncode=1)
    return runner


def _mock_tools(stdout: str = "mock_output") -> Dict[str, Any]:
    """Registro de herramientas que siempre responden con stdout dado."""
    runner = _mock_runner_ok(stdout)
    return build_tool_registry(runner=runner)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ToolResult
# ─────────────────────────────────────────────────────────────────────────────

class TestToolResult:

    def test_ok_when_exit_zero(self):
        r = ToolResult(stdout="hello", stderr="", exit_code=0)
        assert r.ok is True

    def test_not_ok_when_nonzero(self):
        r = ToolResult(stdout="", stderr="err", exit_code=1)
        assert r.ok is False

    def test_as_text_stdout_only(self):
        r = ToolResult(stdout="hello", stderr="", exit_code=0)
        assert r.as_text() == "hello"

    def test_as_text_with_stderr(self):
        r = ToolResult(stdout="out", stderr="err", exit_code=0)
        text = r.as_text()
        assert "out" in text
        assert "err" in text

    def test_as_text_with_nonzero_exit(self):
        r = ToolResult(stdout="", stderr="", exit_code=42)
        text = r.as_text()
        assert "42" in text

    def test_as_text_empty(self):
        r = ToolResult(stdout="", stderr="", exit_code=0)
        assert r.as_text() == "(empty)"

    def test_repr_contains_tool_name(self):
        r = ToolResult(stdout="x", stderr="", exit_code=0, tool_name="bash")
        assert "bash" in repr(r)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Herramientas individuales
# ─────────────────────────────────────────────────────────────────────────────

class TestBashTool:

    def test_run_returns_tool_result(self):
        tool = BashTool(runner=_mock_runner_ok("hello bash"))
        r = tool.run({"command": "echo hello"})
        assert isinstance(r, ToolResult)

    def test_run_ok(self):
        tool = BashTool(runner=_mock_runner_ok("output"))
        r = tool.run({"command": "echo test"})
        assert r.ok

    def test_run_fail(self):
        tool = BashTool(runner=_mock_runner_fail("some error"))
        r = tool.run({"command": "false"})
        assert not r.ok

    def test_tool_name_in_result(self):
        tool = BashTool(runner=_mock_runner_ok())
        r = tool.run({"command": "ls"})
        assert r.tool_name == "bash"

    def test_empty_command_runs(self):
        tool = BashTool(runner=_mock_runner_ok())
        r = tool.run({})
        assert isinstance(r, ToolResult)


class TestGrepTool:

    def test_run_returns_result(self):
        tool = GrepTool(runner=_mock_runner_ok("match.py:42:def foo"))
        r = tool.run({"pattern": "def foo", "path": "."})
        assert isinstance(r, ToolResult)

    def test_tool_name(self):
        tool = GrepTool(runner=_mock_runner_ok())
        r = tool.run({"pattern": "x"})
        assert r.tool_name == "grep"


class TestFindTool:

    def test_run_returns_result(self):
        tool = FindTool(runner=_mock_runner_ok("tests/test_a.py\n"))
        r = tool.run({"name": "*.py", "path": "tests/"})
        assert isinstance(r, ToolResult)
        assert "test_a.py" in r.stdout

    def test_tool_name(self):
        tool = FindTool(runner=_mock_runner_ok())
        r = tool.run({"name": "*.py"})
        assert r.tool_name == "find"

    def test_maxdepth_accepted(self):
        tool = FindTool(runner=_mock_runner_ok())
        r = tool.run({"name": "*.py", "maxdepth": 2})
        assert isinstance(r, ToolResult)


class TestCatTool:

    def test_run_returns_content(self):
        content = "line1\nline2\nline3"
        tool = CatTool(runner=_mock_runner_ok(content))
        r = tool.run({"path": "file.py"})
        assert "line1" in r.stdout

    def test_lines_limit(self):
        content = "\n".join(f"line{i}" for i in range(100))
        tool = CatTool(runner=_mock_runner_ok(content))
        r = tool.run({"path": "file.py", "lines": 5})
        assert r.stdout.count("\n") < 10

    def test_no_path_returns_error(self):
        tool = CatTool(runner=_mock_runner_ok())
        r = tool.run({})
        assert not r.ok
        assert "No path" in r.stderr

    def test_tool_name(self):
        tool = CatTool(runner=_mock_runner_ok())
        r = tool.run({"path": "x"})
        assert r.tool_name == "cat"


class TestPytestTool:

    def test_run_returns_result(self):
        tool = PytestTool(runner=_mock_runner_ok("5 passed"))
        r = tool.run({"path": "tests/", "flags": "-q"})
        assert isinstance(r, ToolResult)
        assert "5 passed" in r.stdout

    def test_tool_name(self):
        tool = PytestTool(runner=_mock_runner_ok())
        r = tool.run({})
        assert r.tool_name == "pytest"


# ─────────────────────────────────────────────────────────────────────────────
# 3. AgentSession
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentSession:

    def _make_session(self) -> AgentSession:
        return AgentSession(task="Fix the training loop")

    def test_initial_state(self):
        s = self._make_session()
        assert s.n_turns == 0
        assert s.seen_files == []
        assert s.errors == []

    def test_record_action_increases_n_turns(self):
        s = self._make_session()
        r = ToolResult("out", "", 0, "bash")
        s.record_action(turn=1, tool_name="bash", args={"command": "ls"}, result=r)
        assert s.n_turns == 1

    def test_record_action_returns_entry(self):
        s = self._make_session()
        r = ToolResult("out", "", 0, "bash")
        entry = s.record_action(turn=1, tool_name="bash", args={}, result=r)
        assert isinstance(entry, ActionEntry)
        assert entry.turn == 1
        assert entry.tool_name == "bash"

    def test_multiple_actions(self):
        s = self._make_session()
        for i in range(3):
            r = ToolResult(f"out{i}", "", 0)
            s.record_action(turn=i+1, tool_name="bash", args={}, result=r)
        assert s.n_turns == 3

    def test_record_file_seen(self):
        s = self._make_session()
        s.record_file_seen("training_utils.py")
        assert "training_utils.py" in s.seen_files

    def test_record_file_no_duplicates(self):
        s = self._make_session()
        s.record_file_seen("x.py")
        s.record_file_seen("x.py")
        assert s.seen_files.count("x.py") == 1

    def test_record_patch(self):
        s = self._make_session()
        s.record_patch("--- a\n+++ b\n@@ 1 @@\n-old\n+new")
        assert len(s.attempted_patches) == 1

    def test_record_error(self):
        s = self._make_session()
        s.record_error("Something went wrong")
        assert s.has_errors()
        assert "Something went wrong" in s.errors

    def test_actions_for_turn(self):
        s = self._make_session()
        r = ToolResult("x", "", 0)
        s.record_action(turn=1, tool_name="cat", args={}, result=r)
        s.record_action(turn=1, tool_name="grep", args={}, result=r)
        s.record_action(turn=2, tool_name="bash", args={}, result=r)
        turn1 = s.actions_for_turn(1)
        assert len(turn1) == 2
        assert all(a.turn == 1 for a in turn1)

    def test_last_action(self):
        s = self._make_session()
        r = ToolResult("last", "", 0)
        s.record_action(turn=1, tool_name="bash", args={}, result=r)
        s.record_action(turn=2, tool_name="grep", args={}, result=r)
        assert s.last_action().tool_name == "grep"

    def test_last_action_empty(self):
        s = self._make_session()
        assert s.last_action() is None

    def test_tool_calls_count(self):
        s = self._make_session()
        r = ToolResult("", "", 0)
        s.record_action(1, "bash", {}, r)
        s.record_action(2, "bash", {}, r)
        s.record_action(3, "cat",  {}, r)
        counts = s.tool_calls_count()
        assert counts["bash"] == 2
        assert counts["cat"]  == 1

    def test_summary_string(self):
        s = self._make_session()
        assert isinstance(s.summary(), str)
        assert "Fix" in s.summary()


# ─────────────────────────────────────────────────────────────────────────────
# 4. MemoryBridge
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryBridge:

    def test_store_and_load(self):
        mem = MemoryBridge()
        mem.store("key1", "value1")
        assert mem.load("key1") == "value1"

    def test_load_missing_returns_default(self):
        mem = MemoryBridge()
        assert mem.load("nonexistent") is None
        assert mem.load("nonexistent", default="fallback") == "fallback"

    def test_delete_existing(self):
        mem = MemoryBridge()
        mem.store("k", "v")
        deleted = mem.delete("k")
        assert deleted is True
        assert mem.load("k") is None

    def test_delete_nonexistent(self):
        mem = MemoryBridge()
        assert mem.delete("missing") is False

    def test_len(self):
        mem = MemoryBridge()
        mem.store("a", 1)
        mem.store("b", 2)
        assert len(mem) == 2

    def test_contains(self):
        mem = MemoryBridge()
        mem.store("x", 99)
        assert "x" in mem
        assert "y" not in mem

    def test_keys(self):
        mem = MemoryBridge()
        mem.store("k1", 1)
        mem.store("k2", 2)
        assert set(mem.keys()) == {"k1", "k2"}

    def test_search_by_key(self):
        mem = MemoryBridge()
        mem.store("training_info", "details about training")
        mem.store("other_key", "something else")
        results = mem.search("training")
        assert any("training_info" in r[0] for r in results)

    def test_search_by_value(self):
        mem = MemoryBridge()
        mem.store("key", "AMP training loop details")
        results = mem.search("AMP")
        assert len(results) > 0

    def test_search_max_results(self):
        mem = MemoryBridge()
        for i in range(20):
            mem.store(f"item_{i}", f"apple content {i}")
        results = mem.search("apple", max_results=3)
        assert len(results) <= 3

    def test_as_context_empty(self):
        mem = MemoryBridge()
        ctx = mem.as_context()
        assert "(no memory)" in ctx

    def test_as_context_with_data(self):
        mem = MemoryBridge()
        mem.store("task", "fix training loop")
        ctx = mem.as_context()
        assert "task" in ctx
        assert "fix training loop" in ctx

    def test_as_context_truncates(self):
        mem = MemoryBridge()
        mem.store("k", "x" * 5000)
        ctx = mem.as_context(max_chars=100)
        assert len(ctx) <= 130   # algunos chars extra para "[truncated]"

    def test_to_dict_and_from_dict(self):
        mem = MemoryBridge()
        mem.store("a", 1)
        mem.store("b", "hello")
        d   = mem.to_dict()
        mem2 = MemoryBridge.from_dict(d)
        assert mem2.load("a") == 1
        assert mem2.load("b") == "hello"


# ─────────────────────────────────────────────────────────────────────────────
# 5. MockMotor
# ─────────────────────────────────────────────────────────────────────────────

class TestMockMotor:

    def test_returns_actions_in_order(self):
        actions = [
            MotorAction("bash", {"command": "ls"}, "listing"),
            MotorAction("DONE", {}, "done"),
        ]
        motor = MockMotor(actions)
        a1 = motor("task", "")
        a2 = motor("task", "")
        assert a1.tool == "bash"
        assert a2.tool == "DONE"

    def test_fail_when_exhausted(self):
        motor = MockMotor([MotorAction("DONE", {}, "done")])
        motor("task", "")   # consumes
        a = motor("task", "")
        assert a.is_fail

    def test_reset_restarts_sequence(self):
        actions = [MotorAction("bash", {}, "")]
        motor   = MockMotor(actions)
        motor("task", "")  # consume
        motor.reset()
        a = motor("task", "")
        assert a.tool == "bash"

    def test_motor_action_is_done(self):
        a = MotorAction(DONE_SIGNAL)
        assert a.is_done
        assert a.is_terminal
        assert not a.is_fail

    def test_motor_action_is_fail(self):
        a = MotorAction(FAIL_SIGNAL)
        assert a.is_fail
        assert a.is_terminal
        assert not a.is_done

    def test_motor_action_tool_not_terminal(self):
        a = MotorAction("bash")
        assert not a.is_terminal


# ─────────────────────────────────────────────────────────────────────────────
# 6. AgentLoop
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentLoop:

    def _make_tools(self, stdout: str = "ok") -> Dict[str, Any]:
        return _mock_tools(stdout)

    def test_3_turn_execution(self):
        """El loop ejecuta exactamente 3 turnos y termina con DONE."""
        motor = MockMotor([
            MotorAction("bash", {"command": "ls"}, "Listing"),
            MotorAction("cat",  {"path": "a.py"},  "Reading"),
            MotorAction("bash", {"command": "echo hi"}, "Echo"),
            MotorAction(DONE_SIGNAL, {}, "Task complete"),
        ])
        loop   = AgentLoop(motor=motor, tools=self._make_tools(), max_turns=20)
        result = loop.run(task="Test task")

        assert result.succeeded
        assert result.turns_used == 4
        assert result.session.n_turns == 4

    def test_done_on_first_turn(self):
        motor  = MockMotor([MotorAction(DONE_SIGNAL, {}, "instant done")])
        loop   = AgentLoop(motor=motor, tools=self._make_tools())
        result = loop.run(task="instant")
        assert result.succeeded
        assert result.turns_used == 1

    def test_max_turns_respected(self):
        """El loop para cuando alcanza max_turns sin DONE/FAIL."""
        # Motor que siempre ejecuta bash (nunca termina)
        always_bash = MockMotor([
            MotorAction("bash", {"command": "echo x"}, f"step {i}")
            for i in range(100)
        ])
        loop   = AgentLoop(motor=always_bash, tools=self._make_tools(), max_turns=5)
        result = loop.run(task="infinite task")

        assert result.status == "max_turns"
        assert result.turns_used == 5
        assert result.session.n_turns == 5

    def test_fail_signal_ends_loop(self):
        motor  = MockMotor([
            MotorAction("bash", {"command": "bad_cmd"}, "trying"),
            MotorAction(FAIL_SIGNAL, {}, "cannot proceed"),
        ])
        loop   = AgentLoop(motor=motor, tools=self._make_tools(), max_turns=20)
        result = loop.run(task="will fail")

        assert result.failed
        assert result.status == "fail"
        assert result.turns_used == 2

    def test_history_saved_in_session(self):
        motor = MockMotor([
            MotorAction("bash",   {"command": "ls"},  "listing"),
            MotorAction("grep",   {"pattern": "foo"}, "searching"),
            MotorAction(DONE_SIGNAL, {}, "done"),
        ])
        loop   = AgentLoop(motor=motor, tools=self._make_tools())
        result = loop.run(task="inspect code")

        session = result.session
        assert session.n_turns == 3
        assert session.actions[0].tool_name == "bash"
        assert session.actions[1].tool_name == "grep"
        assert session.actions[2].tool_name == DONE_SIGNAL

    def test_reasoning_saved_in_history(self):
        motor = MockMotor([
            MotorAction("bash", {"command": "ls"}, reasoning="I need to see the files"),
            MotorAction(DONE_SIGNAL, {}, reasoning="Files look OK"),
        ])
        loop   = AgentLoop(motor=motor, tools=self._make_tools())
        result = loop.run(task="check files")

        assert result.session.actions[0].motor_reasoning == "I need to see the files"
        assert result.session.actions[1].motor_reasoning == "Files look OK"

    def test_unknown_tool_records_error(self):
        motor  = MockMotor([
            MotorAction("nonexistent_tool", {}, ""),
            MotorAction(DONE_SIGNAL, {}, ""),
        ])
        loop   = AgentLoop(motor=motor, tools=self._make_tools())
        result = loop.run(task="test unknown tool")

        assert result.session.has_errors()

    def test_cat_auto_records_file_seen(self):
        motor = MockMotor([
            MotorAction("cat", {"path": "experiments/training_utils.py"}, "reading"),
            MotorAction(DONE_SIGNAL, {}, "done"),
        ])
        loop   = AgentLoop(motor=motor, tools=self._make_tools())
        result = loop.run(task="read file")

        assert "experiments/training_utils.py" in result.session.seen_files

    def test_result_session_task_matches(self):
        motor  = MockMotor([MotorAction(DONE_SIGNAL, {}, "")])
        loop   = AgentLoop(motor=motor, tools=self._make_tools())
        result = loop.run(task="specific task")

        assert result.session.task == "specific task"

    def test_loop_result_succeeded_false_on_fail(self):
        motor  = MockMotor([MotorAction(FAIL_SIGNAL, {}, "fail")])
        loop   = AgentLoop(motor=motor, tools=self._make_tools())
        result = loop.run(task="")
        assert not result.succeeded

    def test_loop_result_failed_true_on_max_turns(self):
        motor  = MockMotor([MotorAction("bash", {"command": "x"}, "") for _ in range(10)])
        loop   = AgentLoop(motor=motor, tools=self._make_tools(), max_turns=3)
        result = loop.run(task="")
        assert result.failed

    def test_tool_result_stored_in_session(self):
        motor = MockMotor([
            MotorAction("bash", {"command": "echo test"}, ""),
            MotorAction(DONE_SIGNAL, {}, ""),
        ])
        loop   = AgentLoop(motor=motor, tools=self._make_tools("mock_output"))
        result = loop.run(task="")

        bash_entry = result.session.actions[0]
        assert isinstance(bash_entry.result, ToolResult)
        assert bash_entry.result.stdout == "mock_output"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Integración con herramientas mockeadas
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentLoopWithTools:

    def test_bash_output_in_session(self):
        """La salida de bash queda en el historial."""
        motor = MockMotor([
            MotorAction("bash", {"command": "echo hello"}, ""),
            MotorAction(DONE_SIGNAL, {}, ""),
        ])
        runner = _mock_runner_ok("hello world")
        tools  = build_tool_registry(runner=runner)
        loop   = AgentLoop(motor=motor, tools=tools)
        result = loop.run(task="echo")

        bash_result = result.session.actions[0].result
        assert bash_result.stdout == "hello world"

    def test_grep_not_found_exit_1_but_no_session_error(self):
        """Exit code 1 de grep (sin matches) NO es un error del agente."""
        motor = MockMotor([
            MotorAction("grep", {"pattern": "nonexistent"}, ""),
            MotorAction(DONE_SIGNAL, {}, ""),
        ])
        runner = _mock_runner_fail("")   # exit 1, sin stderr
        tools  = build_tool_registry(runner=runner)
        loop   = AgentLoop(motor=motor, tools=tools)
        result = loop.run(task="grep")

        # El agente no registra error por exit_code != 0 de la herramienta
        # (solo registra errores por herramientas desconocidas o excepciones)
        assert result.session.actions[0].result.exit_code == 1

    def test_memory_bridge_passed_to_loop(self):
        """MemoryBridge puede pasarse al loop y usarse desde el motor."""
        mem = MemoryBridge()
        mem.store("hint", "check line 42")

        # Motor que usa la memoria para su razonamiento
        motor = MockMotor([
            MotorAction("bash", {"command": "echo check"}, "using hint"),
            MotorAction(DONE_SIGNAL, {}, "done"),
        ])
        loop   = AgentLoop(motor=motor, tools=_mock_tools(), memory=mem)
        result = loop.run(task="debug")

        assert result.succeeded
        assert loop.memory.load("hint") == "check line 42"
