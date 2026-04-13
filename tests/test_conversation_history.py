"""Tests for conversation history and system prompt in AgentLoop."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.loop import AgentLoop, MockMotor, MotorAction
from agent.tools import build_tool_registry
from agent.session import AgentSession


def _mock_runner(cmd, **kwargs):
    from types import SimpleNamespace
    return SimpleNamespace(stdout="ok", stderr="", returncode=0)


def _make_loop(**kwargs):
    motor = MockMotor([MotorAction("DONE", {}, "Done")])
    tools = build_tool_registry(runner=_mock_runner)
    return AgentLoop(motor=motor, tools=tools, **kwargs)


class TestConversationHistory:
    def test_add_user_message(self):
        loop = _make_loop()
        loop.add_user_message("Hello")
        h = loop.get_conversation_history()
        assert len(h) == 1
        assert h[0]["role"] == "user"
        assert h[0]["content"] == "Hello"

    def test_add_assistant_message(self):
        loop = _make_loop()
        loop.add_assistant_message("Hi there")
        h = loop.get_conversation_history()
        assert len(h) == 1
        assert h[0]["role"] == "assistant"

    def test_multi_turn(self):
        loop = _make_loop()
        loop.add_user_message("Q1")
        loop.add_assistant_message("A1")
        loop.add_user_message("Q2")
        loop.add_assistant_message("A2")
        h = loop.get_conversation_history()
        assert len(h) == 4
        assert h[0]["content"] == "Q1"
        assert h[3]["content"] == "A2"

    def test_clear_history(self):
        loop = _make_loop()
        loop.add_user_message("test")
        loop.clear_history()
        assert len(loop.get_conversation_history()) == 0

    def test_truncation(self):
        loop = _make_loop(max_history_tokens=10)
        for i in range(50):
            loop.add_user_message(f"Message number {i} with enough text")
        h = loop.get_conversation_history()
        total = sum(len(m["content"]) // 4 for m in h)
        assert total <= 20  # some slack

    def test_history_in_build_history(self):
        loop = _make_loop()
        loop.add_user_message("Previous question")
        loop.add_assistant_message("Previous answer")
        session = AgentSession(task="test")
        text = loop._build_history(session)
        assert "Previous question" in text
        assert "Previous answer" in text

    def test_history_returns_copy(self):
        loop = _make_loop()
        loop.add_user_message("test")
        h = loop.get_conversation_history()
        h.clear()  # modifying returned list
        assert len(loop.get_conversation_history()) == 1  # original unchanged


class TestSystemPrompt:
    def test_system_prompt_stored(self):
        loop = _make_loop(system_prompt="Be helpful")
        assert loop.system_prompt == "Be helpful"

    def test_system_prompt_in_history(self):
        loop = _make_loop(system_prompt="Eres un tutor de matemáticas")
        session = AgentSession(task="solve x+1=2")
        text = loop._build_history(session)
        assert "tutor de matemáticas" in text

    def test_no_system_prompt(self):
        loop = _make_loop()
        assert loop.system_prompt is None

    def test_system_prompt_none_doesnt_appear(self):
        loop = _make_loop()
        session = AgentSession(task="test")
        text = loop._build_history(session)
        assert "[system]" not in text

    def test_combined_system_and_history(self):
        loop = _make_loop(system_prompt="Be concise")
        loop.add_user_message("What is 2+2?")
        session = AgentSession(task="math")
        text = loop._build_history(session)
        assert "Be concise" in text
        assert "2+2" in text
