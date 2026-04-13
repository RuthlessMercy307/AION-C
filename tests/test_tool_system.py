"""
tests/test_tool_system.py — Tests para el Tool system (Parte 4 del MEGA-PROMPT)
================================================================================

Cubre los 6 tools nuevos + el ToolExecutor + el parser + el sandbox:

  WriteFileTool   — escribe dentro del sandbox /output/, rechaza fuera
  EditFileTool    — old→new dentro del sandbox, valida unicidad
  RunCodeTool     — subprocess con timeout y mockeable
  CallApiTool     — whitelist de dominios, mockeable
  SearchMemTool   — usa SemanticStore inyectado (mock)
  StoreMemTool    — usa SemanticStore inyectado (mock)
  parse_tool_calls — extrae bloques [TOOL: {...}] del output del modelo
  ToolExecutor    — pipeline completo: parse → dispatch → record → format
  build_tool_registry — verifica que registra los 14 tools (incluye aliases)
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from agent.tools import (
    WriteFileTool, EditFileTool, RunCodeTool, CallApiTool,
    SearchMemTool, StoreMemTool,
    build_tool_registry,
)
from agent.tool_executor import (
    ToolCall, ToolExecutor, parse_tool_calls, format_result,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


class FakeMem:
    """Mínimo SemanticStore para los tests de search_mem/store_mem."""
    def __init__(self):
        self.entries = {}

    def store(self, key, value, domain="general", source="test"):
        self.entries[key] = (value, domain)

    def search(self, query, top_k=5, domain=None):
        out = []
        for k, (v, d) in self.entries.items():
            if domain and d != domain:
                continue
            score = 1.0 if query.lower() in v.lower() else 0.5
            out.append((k, v, score))
        out.sort(key=lambda x: x[2], reverse=True)
        return out[:top_k]


def fake_completed(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


# ─────────────────────────────────────────────────────────────────────────────
# WriteFileTool
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteFileTool:
    def test_writes_relative_path(self, tmp_path):
        tool = WriteFileTool(output_root=tmp_path)
        result = tool.run({"path": "hello.py", "content": "print('hi')"})
        assert result.ok
        target = tmp_path / "hello.py"
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "print('hi')"
        assert "File written" in result.stdout

    def test_writes_absolute_path_inside_root(self, tmp_path):
        tool = WriteFileTool(output_root=tmp_path)
        target = tmp_path / "sub" / "out.txt"
        result = tool.run({"path": str(target), "content": "x"})
        assert result.ok
        assert target.exists()

    def test_rejects_path_outside_sandbox(self, tmp_path):
        tool = WriteFileTool(output_root=tmp_path / "sandbox")
        # Intenta escribir en el directorio padre — debe ser rechazado
        evil = tmp_path / "evil.txt"
        result = tool.run({"path": str(evil), "content": "boom"})
        assert not result.ok
        assert "outside sandbox" in result.stderr
        assert not evil.exists()

    def test_rejects_traversal(self, tmp_path):
        sandbox = tmp_path / "sb"
        sandbox.mkdir()
        tool = WriteFileTool(output_root=sandbox)
        result = tool.run({"path": "../escaped.txt", "content": "x"})
        assert not result.ok
        assert "outside sandbox" in result.stderr

    def test_empty_path(self, tmp_path):
        tool = WriteFileTool(output_root=tmp_path)
        result = tool.run({"path": "", "content": "x"})
        assert not result.ok
        assert "No path provided" in result.stderr

    def test_mock_write_fn(self, tmp_path):
        captured = {}
        def fake_write(path, content):
            captured["path"] = path
            captured["content"] = content
        tool = WriteFileTool(output_root=tmp_path, write_fn=fake_write)
        result = tool.run({"path": "x.py", "content": "data"})
        assert result.ok
        assert captured["content"] == "data"
        # File NOT actually written because write_fn is mocked
        assert not (tmp_path / "x.py").exists()


# ─────────────────────────────────────────────────────────────────────────────
# EditFileTool
# ─────────────────────────────────────────────────────────────────────────────


class TestEditFileTool:
    def test_replaces_unique(self, tmp_path):
        target = tmp_path / "code.py"
        target.write_text("def foo():\n    return 1\n", encoding="utf-8")
        tool = EditFileTool(output_root=tmp_path)
        result = tool.run({"path": "code.py", "old": "return 1", "new": "return 2"})
        assert result.ok, result.stderr
        assert target.read_text(encoding="utf-8") == "def foo():\n    return 2\n"

    def test_rejects_non_unique_without_replace_all(self, tmp_path):
        target = tmp_path / "code.py"
        target.write_text("a\na\na\n", encoding="utf-8")
        tool = EditFileTool(output_root=tmp_path)
        result = tool.run({"path": "code.py", "old": "a", "new": "b"})
        assert not result.ok
        assert "not unique" in result.stderr
        # Archivo no debe ser modificado
        assert target.read_text(encoding="utf-8") == "a\na\na\n"

    def test_replace_all(self, tmp_path):
        target = tmp_path / "code.py"
        target.write_text("a\na\na\n", encoding="utf-8")
        tool = EditFileTool(output_root=tmp_path)
        result = tool.run({"path": "code.py", "old": "a", "new": "b", "replace_all": True})
        assert result.ok
        assert target.read_text(encoding="utf-8") == "b\nb\nb\n"

    def test_old_not_found(self, tmp_path):
        target = tmp_path / "x.txt"
        target.write_text("hello", encoding="utf-8")
        tool = EditFileTool(output_root=tmp_path)
        result = tool.run({"path": "x.txt", "old": "missing", "new": "found"})
        assert not result.ok
        assert "not found" in result.stderr

    def test_file_not_exists(self, tmp_path):
        tool = EditFileTool(output_root=tmp_path)
        result = tool.run({"path": "nope.txt", "old": "a", "new": "b"})
        assert not result.ok
        assert "File not found" in result.stderr

    def test_rejects_outside_sandbox(self, tmp_path):
        sandbox = tmp_path / "sb"
        sandbox.mkdir()
        outside = tmp_path / "out.txt"
        outside.write_text("hi", encoding="utf-8")
        tool = EditFileTool(output_root=sandbox)
        result = tool.run({"path": str(outside), "old": "hi", "new": "ho"})
        assert not result.ok
        assert "outside sandbox" in result.stderr
        assert outside.read_text(encoding="utf-8") == "hi"

    def test_mocked_io(self, tmp_path):
        reads = {"x": "hello world"}
        writes = {}
        tool = EditFileTool(
            output_root=tmp_path,
            read_fn=lambda p: reads["x"],
            write_fn=lambda p, c: writes.update({"path": p, "content": c}),
        )
        result = tool.run({"path": "x.txt", "old": "world", "new": "universe"})
        assert result.ok
        assert writes["content"] == "hello universe"


# ─────────────────────────────────────────────────────────────────────────────
# RunCodeTool
# ─────────────────────────────────────────────────────────────────────────────


class TestRunCodeTool:
    def test_python_via_mock(self):
        captured = {}
        def fake_runner(cmd, **kw):
            captured["cmd"] = cmd
            captured["timeout"] = kw.get("timeout")
            return fake_completed(stdout="hello\n", returncode=0)
        tool = RunCodeTool(runner=fake_runner)
        result = tool.run({"language": "python", "code": "print('hello')"})
        assert result.ok
        assert result.stdout == "hello\n"
        assert "print('hello')" in captured["cmd"][-1]
        assert captured["timeout"] == 30

    def test_bash_via_mock(self):
        def fake_runner(cmd, **kw):
            assert cmd[0] == "bash"
            assert cmd[1] == "-c"
            return fake_completed(stdout="ok", returncode=0)
        tool = RunCodeTool(runner=fake_runner)
        result = tool.run({"language": "bash", "code": "echo ok"})
        assert result.ok

    def test_disallowed_language(self):
        tool = RunCodeTool()
        result = tool.run({"language": "ruby", "code": "puts 'x'"})
        assert not result.ok
        assert "not allowed" in result.stderr

    def test_empty_code(self):
        tool = RunCodeTool()
        result = tool.run({"language": "python", "code": ""})
        assert not result.ok
        assert "No code" in result.stderr

    def test_timeout_capped(self):
        captured = {}
        def fake_runner(cmd, **kw):
            captured["timeout"] = kw.get("timeout")
            return fake_completed(stdout="", returncode=0)
        tool = RunCodeTool(runner=fake_runner)
        # Pide 999s, debe ser capado a 60
        tool.run({"language": "python", "code": "pass", "timeout": 999})
        assert captured["timeout"] == 60

    def test_timeout_expired(self):
        def fake_runner(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 1))
        tool = RunCodeTool(runner=fake_runner)
        result = tool.run({"language": "python", "code": "pass", "timeout": 1})
        assert not result.ok
        assert "Timeout" in result.stderr

    def test_truncates_huge_output(self):
        big = "x" * 20000
        def fake_runner(cmd, **kw):
            return fake_completed(stdout=big, returncode=0)
        tool = RunCodeTool(runner=fake_runner)
        result = tool.run({"language": "python", "code": "pass"})
        assert result.ok
        assert "truncated" in result.stdout
        assert len(result.stdout) < 9000


# ─────────────────────────────────────────────────────────────────────────────
# CallApiTool
# ─────────────────────────────────────────────────────────────────────────────


class TestCallApiTool:
    def test_blocks_when_whitelist_empty(self):
        tool = CallApiTool()  # whitelist vacío = deny all
        result = tool.run({"url": "https://example.com/data"})
        assert not result.ok
        assert "not in whitelist" in result.stderr

    def test_blocks_unknown_domain(self):
        tool = CallApiTool(allowed_domains={"api.allowed.com"})
        result = tool.run({"url": "https://evil.com/x"})
        assert not result.ok
        assert "not in whitelist" in result.stderr

    def test_allows_whitelisted_via_mock(self):
        called = {}
        def fake_fetch(url, method, headers, body, timeout):
            called["url"] = url
            called["method"] = method
            return 200, "OK BODY"
        tool = CallApiTool(allowed_domains={"api.example.com"}, fetch_fn=fake_fetch)
        result = tool.run({"url": "https://api.example.com/v1", "method": "GET"})
        assert result.ok
        assert result.stdout == "OK BODY"
        assert called["method"] == "GET"

    def test_allows_subdomain_of_whitelisted(self):
        def fake_fetch(*args):
            return 200, "x"
        tool = CallApiTool(allowed_domains={"example.com"}, fetch_fn=fake_fetch)
        result = tool.run({"url": "https://api.example.com/x"})
        assert result.ok

    def test_invalid_method(self):
        tool = CallApiTool(allowed_domains={"x.com"})
        result = tool.run({"url": "https://x.com/", "method": "TRACE"})
        assert not result.ok
        assert "method not allowed" in result.stderr

    def test_http_status_error_via_mock(self):
        def fake_fetch(*args):
            return 500, "internal err"
        tool = CallApiTool(allowed_domains={"x.com"}, fetch_fn=fake_fetch)
        result = tool.run({"url": "https://x.com/"})
        assert not result.ok  # 500 → exit_code=1

    def test_empty_url(self):
        tool = CallApiTool(allowed_domains={"x.com"})
        result = tool.run({"url": ""})
        assert not result.ok
        assert "No URL" in result.stderr


# ─────────────────────────────────────────────────────────────────────────────
# SearchMemTool / StoreMemTool
# ─────────────────────────────────────────────────────────────────────────────


class TestMemTools:
    def test_store_and_search(self):
        mem = FakeMem()
        store = StoreMemTool(mem=mem)
        search = SearchMemTool(mem=mem)
        r1 = store.run({"key": "jwt", "value": "JWT auth pattern", "domain": "forge_c"})
        assert r1.ok
        assert "Stored" in r1.stdout
        r2 = search.run({"query": "jwt"})
        assert r2.ok
        assert "JWT auth pattern" in r2.stdout

    def test_search_no_mem(self):
        tool = SearchMemTool(mem=None)
        result = tool.run({"query": "anything"})
        assert not result.ok
        assert "MEM not configured" in result.stderr

    def test_store_no_mem(self):
        tool = StoreMemTool(mem=None)
        result = tool.run({"key": "k", "value": "v"})
        assert not result.ok

    def test_search_filters_by_domain(self):
        mem = FakeMem()
        mem.store("a", "java code", domain="forge_c")
        mem.store("b", "java music", domain="muse")
        tool = SearchMemTool(mem=mem)
        result = tool.run({"query": "java", "domain": "forge_c"})
        assert result.ok
        assert "java code" in result.stdout
        assert "music" not in result.stdout

    def test_store_missing_args(self):
        tool = StoreMemTool(mem=FakeMem())
        result = tool.run({"key": "", "value": "x"})
        assert not result.ok

    def test_search_no_matches(self):
        mem = FakeMem()
        tool = SearchMemTool(mem=mem)
        result = tool.run({"query": "anything"})
        assert result.ok
        assert "no matches" in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# parse_tool_calls
# ─────────────────────────────────────────────────────────────────────────────


class TestParser:
    def test_single_simple_call(self):
        text = '[TOOL: {"action":"search_web","input":"javascript"}]'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].action == "search_web"
        assert calls[0].input == "javascript"

    def test_nested_input(self):
        text = '[TOOL: {"action":"write_file","input":{"path":"/output/h.py","content":"print(1)"}}]'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].action == "write_file"
        assert calls[0].input["path"] == "/output/h.py"

    def test_multiple_calls_in_one_text(self):
        text = (
            "Voy a hacer dos cosas. "
            '[TOOL: {"action":"write_file","input":{"path":"a.py","content":"x"}}] '
            "y luego "
            '[TOOL: {"action":"run_code","input":{"language":"python","code":"print(2)"}}] '
            "fin."
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].action == "write_file"
        assert calls[1].action == "run_code"

    def test_handles_quotes_inside_strings(self):
        text = '[TOOL: {"action":"run_code","input":{"language":"python","code":"print(\\"hi\\")"}}]'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert 'print("hi")' in calls[0].input["code"]

    def test_no_tool_calls(self):
        assert parse_tool_calls("Hola, esta es una respuesta normal.") == []

    def test_empty_text(self):
        assert parse_tool_calls("") == []
        assert parse_tool_calls(None) == []

    def test_malformed_json_skipped(self):
        text = '[TOOL: {bad json}] [TOOL: {"action":"x","input":1}]'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].action == "x"

    def test_missing_action_skipped(self):
        text = '[TOOL: {"input":"x"}]'
        assert parse_tool_calls(text) == []

    def test_case_insensitive_tag(self):
        calls = parse_tool_calls('[tool: {"action":"a","input":1}]')
        assert len(calls) == 1
        assert calls[0].action == "a"

    def test_to_args_dict(self):
        c = ToolCall(action="x", input={"a": 1, "b": 2})
        assert c.to_args() == {"a": 1, "b": 2}

    def test_to_args_string_for_search_web(self):
        c = ToolCall(action="search_web", input="python")
        assert c.to_args() == {"query": "python"}

    def test_to_args_string_for_read_file(self):
        c = ToolCall(action="read_file", input="/etc/passwd")
        assert c.to_args() == {"path": "/etc/passwd"}


# ─────────────────────────────────────────────────────────────────────────────
# ToolExecutor — pipeline E2E
# ─────────────────────────────────────────────────────────────────────────────


class TestToolExecutor:
    def test_dispatch_unknown_action(self):
        executor = ToolExecutor({})
        call = ToolCall(action="nope", input={})
        result = executor.execute(call)
        assert not result.ok
        assert "unknown action" in result.stderr
        assert len(executor.history) == 1

    def test_dispatch_to_real_tool(self, tmp_path):
        registry = build_tool_registry(output_root=tmp_path)
        executor = ToolExecutor(registry)
        call = ToolCall(action="write_file", input={"path": "out.txt", "content": "hi"})
        result = executor.execute(call)
        assert result.ok
        assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hi"

    def test_run_from_text_full_pipeline(self, tmp_path):
        registry = build_tool_registry(output_root=tmp_path)
        executor = ToolExecutor(registry)
        model_output = (
            "Voy a crear un archivo. "
            '[TOOL: {"action":"write_file","input":{"path":"hello.py","content":"print(\\"hola\\")"}}]'
            " Listo."
        )
        records = executor.run_from_text(model_output)
        assert len(records) == 1
        assert records[0].success
        assert (tmp_path / "hello.py").exists()

    def test_format_context_returns_results(self, tmp_path):
        registry = build_tool_registry(output_root=tmp_path)
        executor = ToolExecutor(registry)
        records = executor.run_from_text(
            '[TOOL: {"action":"write_file","input":{"path":"a.txt","content":"x"}}]'
        )
        ctx = executor.format_context(records)
        assert ctx.startswith("[RESULT:")
        assert "File written" in ctx

    def test_history_accumulates(self, tmp_path):
        registry = build_tool_registry(output_root=tmp_path)
        executor = ToolExecutor(registry)
        executor.run_from_text(
            '[TOOL: {"action":"write_file","input":{"path":"a.txt","content":"1"}}]'
        )
        executor.run_from_text(
            '[TOOL: {"action":"write_file","input":{"path":"b.txt","content":"2"}}]'
        )
        assert len(executor.history) == 2

    def test_reset_clears_history(self, tmp_path):
        registry = build_tool_registry(output_root=tmp_path)
        executor = ToolExecutor(registry)
        executor.run_from_text(
            '[TOOL: {"action":"write_file","input":{"path":"a.txt","content":"1"}}]'
        )
        executor.reset()
        assert executor.history == []

    def test_tool_exception_captured(self, tmp_path):
        class Boom:
            name = "boom"
            def run(self, args):
                raise RuntimeError("kaboom")
        executor = ToolExecutor({"boom": Boom()})
        result = executor.execute(ToolCall(action="boom", input={}))
        assert not result.ok
        assert "kaboom" in result.stderr

    def test_chained_tool_calls(self, tmp_path):
        registry = build_tool_registry(output_root=tmp_path)
        executor = ToolExecutor(registry)
        text = (
            '[TOOL: {"action":"write_file","input":{"path":"f.py","content":"x = 1"}}] '
            '[TOOL: {"action":"edit_file","input":{"path":"f.py","old":"x = 1","new":"x = 2"}}]'
        )
        records = executor.run_from_text(text)
        assert len(records) == 2
        assert all(r.success for r in records)
        assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 2"


# ─────────────────────────────────────────────────────────────────────────────
# build_tool_registry — verifica que registra todo
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_contains_all_tool_system_tools(self, tmp_path):
        reg = build_tool_registry(output_root=tmp_path, allowed_domains={"x.com"}, mem=FakeMem())
        # Los 6 nuevos del MEGA-PROMPT 4.1
        for name in ("write_file", "edit_file", "run_code", "call_api", "search_mem", "store_mem"):
            assert name in reg, f"missing {name}"
        # Aliases canónicos del MEGA-PROMPT 4.1
        assert "search_web" in reg
        assert "read_file" in reg
        # Las preexistentes
        for name in ("bash", "grep", "find", "cat", "pytest", "web_search", "web_fetch", "file_read"):
            assert name in reg, f"missing legacy {name}"

    def test_search_mem_uses_injected_mem(self, tmp_path):
        mem = FakeMem()
        mem.store("k", "test value")
        reg = build_tool_registry(output_root=tmp_path, mem=mem)
        result = reg["search_mem"].run({"query": "test"})
        assert result.ok
        assert "test value" in result.stdout

    def test_call_api_uses_whitelist(self, tmp_path):
        reg = build_tool_registry(
            output_root=tmp_path,
            allowed_domains={"safe.com"},
            api_fetch_fn=lambda u, m, h, b, t: (200, "ok"),
        )
        bad = reg["call_api"].run({"url": "https://evil.com/x"})
        assert not bad.ok
        good = reg["call_api"].run({"url": "https://safe.com/x"})
        assert good.ok
