"""Tests for MemoryBridge.learn() and related methods."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.memory_bridge import MemoryBridge


class TestLearn:
    def test_learn_stores_with_metadata(self):
        mem = MemoryBridge()
        mem.learn("topic", "content here", source="web")
        val = mem.load("topic")
        assert isinstance(val, dict)
        assert val["content"] == "content here"
        assert val["source"] == "web"
        assert "learned_at" in val

    def test_get_learned_extracts_content(self):
        mem = MemoryBridge()
        mem.learn("key1", "value1", source="test")
        assert mem.get_learned("key1") == "value1"

    def test_get_learned_returns_plain_values(self):
        mem = MemoryBridge()
        mem.store("plain", "just a string")
        assert mem.get_learned("plain") == "just a string"

    def test_get_learned_missing_key(self):
        mem = MemoryBridge()
        assert mem.get_learned("nonexistent") is None

    def test_search_learned_finds_by_content(self):
        mem = MemoryBridge()
        mem.learn("pytorch_ver", "2.5 with CUDA 12.1", source="web")
        mem.learn("tensorflow_ver", "2.15", source="docs")
        hits = mem.search_learned("pytorch")
        assert len(hits) == 1
        assert hits[0][0] == "pytorch_ver"
        assert hits[0][1] == "2.5 with CUDA 12.1"
        assert hits[0][2] == "web"

    def test_search_learned_returns_source(self):
        mem = MemoryBridge()
        mem.learn("info", "data", source="user_input")
        hits = mem.search_learned("info")
        assert hits[0][2] == "user_input"

    def test_search_learned_plain_values(self):
        mem = MemoryBridge()
        mem.store("config", "batch_size=32")
        hits = mem.search_learned("batch")
        assert len(hits) == 1
        assert hits[0][2] == "direct"

    def test_learn_overwrites(self):
        mem = MemoryBridge()
        mem.learn("key", "old", source="v1")
        mem.learn("key", "new", source="v2")
        assert mem.get_learned("key") == "new"

    def test_search_learned_max_results(self):
        mem = MemoryBridge()
        for i in range(10):
            mem.learn(f"item_{i}", f"data about topic {i}", source="test")
        hits = mem.search_learned("topic", max_results=3)
        assert len(hits) == 3
