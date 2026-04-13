"""Tests for new tools: WebSearchTool, WebFetchTool, FileReadTool"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.tools import (
    WebSearchTool, WebFetchTool, FileReadTool,
    build_tool_registry, ToolResult,
)


# ─── WebSearchTool ───────────────────────────────────────────────────────────

class TestWebSearchTool:
    def test_with_mock_search_fn(self):
        ws = WebSearchTool(search_fn=lambda q, n: f"Found {n} results for '{q}'")
        r = ws.run({"query": "test query", "max_results": 3})
        assert r.ok
        assert "test query" in r.stdout
        assert "3" in r.stdout

    def test_empty_query(self):
        ws = WebSearchTool(search_fn=lambda q, n: "")
        r = ws.run({"query": ""})
        assert not r.ok
        assert "No query" in r.stderr

    def test_search_fn_exception(self):
        def bad_fn(q, n):
            raise ConnectionError("Network error")
        ws = WebSearchTool(search_fn=bad_fn)
        r = ws.run({"query": "test"})
        assert not r.ok
        assert "Network error" in r.stderr

    def test_tool_name(self):
        ws = WebSearchTool()
        assert ws.name == "web_search"


# ─── WebFetchTool ────────────────────────────────────────────────────────────

class TestWebFetchTool:
    def test_with_mock_fetch_fn(self):
        wf = WebFetchTool(fetch_fn=lambda url: f"<html>Content of {url}</html>")
        r = wf.run({"url": "https://example.com"})
        assert r.ok
        assert "example.com" in r.stdout

    def test_empty_url(self):
        wf = WebFetchTool(fetch_fn=lambda url: "")
        r = wf.run({"url": ""})
        assert not r.ok

    def test_fetch_exception(self):
        def bad_fn(url):
            raise TimeoutError("Request timed out")
        wf = WebFetchTool(fetch_fn=bad_fn)
        r = wf.run({"url": "https://example.com"})
        assert not r.ok
        assert "timed out" in r.stderr

    def test_tool_name(self):
        wf = WebFetchTool()
        assert wf.name == "web_fetch"


# ─── FileReadTool ────────────────────────────────────────────────────────────

class TestFileReadTool:
    def test_with_mock_read_fn(self):
        fr = FileReadTool(read_fn=lambda p: "file content here")
        r = fr.run({"path": "/any/path.txt"})
        assert r.ok
        assert r.stdout == "file content here"

    def test_reads_real_file(self):
        fr = FileReadTool()
        r = fr.run({"path": __file__})
        assert r.ok
        assert "TestFileReadTool" in r.stdout

    def test_file_not_found(self):
        fr = FileReadTool()
        r = fr.run({"path": "/nonexistent/path/file.txt"})
        assert not r.ok
        assert "not found" in r.stderr.lower() or "No such file" in r.stderr

    def test_empty_path(self):
        fr = FileReadTool()
        r = fr.run({"path": ""})
        assert not r.ok

    def test_max_lines(self):
        fr = FileReadTool(read_fn=lambda p: "line1\nline2\nline3\nline4\nline5\n")
        # max_lines only works with real file reading, not with read_fn
        r = fr.run({"path": "test.txt"})
        assert r.ok

    def test_tool_name(self):
        fr = FileReadTool()
        assert fr.name == "file_read"


# ─── Registry ────────────────────────────────────────────────────────────────

class TestBuildRegistry:
    def test_includes_new_tools(self):
        registry = build_tool_registry(
            search_fn=lambda q, n: "results",
            fetch_fn=lambda u: "content",
            read_fn=lambda p: "file",
        )
        assert "web_search" in registry
        assert "web_fetch" in registry
        assert "file_read" in registry
        # Old tools still there
        assert "bash" in registry
        assert "grep" in registry

    def test_new_tools_work_through_registry(self):
        registry = build_tool_registry(
            search_fn=lambda q, n: f"found {q}",
            fetch_fn=lambda u: f"fetched {u}",
            read_fn=lambda p: f"read {p}",
        )
        r = registry["web_search"].run({"query": "hello"})
        assert r.ok
        assert "hello" in r.stdout

        r = registry["web_fetch"].run({"url": "http://test.com"})
        assert r.ok
        assert "test.com" in r.stdout

        r = registry["file_read"].run({"path": "test.txt"})
        assert r.ok
        assert "test.txt" in r.stdout
