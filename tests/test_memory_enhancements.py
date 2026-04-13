"""
tests/test_memory_enhancements.py — Tests para Parte 8 del MEGA-PROMPT
========================================================================

Cubre:
  UserModel              (8.3) — perfil persistente con setters validados
  ResponseCache          (8.4) — LRU + TTL + invalidación por substring
  ConversationHistory    (8.5) — multi-turn con resumen progresivo
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from memory.user_model import (
    UserModel, USER_MODEL_DOMAIN, USER_MODEL_KEY,
    VALID_LANGUAGES, VALID_LEVELS, VALID_TONES,
)
from memory.response_cache import ResponseCache, normalize_query, CacheEntry
from memory.conversation_history import (
    Turn, ConversationHistory, default_summarizer,
)


# ─────────────────────────────────────────────────────────────────────────────
# FakeMem
# ─────────────────────────────────────────────────────────────────────────────


class FakeMem:
    def __init__(self):
        self.entries = {}

    def store(self, key, value, domain="general", source="test"):
        self.entries[key] = (value, domain)

    def search(self, query, top_k=5, domain=None):
        out = []
        for k, (v, d) in self.entries.items():
            if k == query and (not domain or d == domain):
                out.append((k, v, 1.0))
        return out[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# UserModel
# ─────────────────────────────────────────────────────────────────────────────


class TestUserModel:
    def test_default_values(self):
        u = UserModel()
        assert u.name is None
        assert u.preferred_language == "mixed"
        assert u.technical_level == "unknown"
        assert u.preferred_tone == "neutral"
        assert u.projects == []
        assert u.facts == {}

    def test_set_name_strips(self):
        u = UserModel()
        u.set_name("  Jesus  ")
        assert u.name == "Jesus"

    def test_set_language_validates(self):
        u = UserModel()
        u.set_language("es")
        assert u.preferred_language == "es"
        u.set_language("klingon")  # rejected
        assert u.preferred_language == "es"

    def test_set_technical_level_validates(self):
        u = UserModel()
        u.set_technical_level("advanced")
        assert u.technical_level == "advanced"
        u.set_technical_level("guru")
        assert u.technical_level == "advanced"

    def test_set_tone_validates(self):
        u = UserModel()
        u.set_tone("casual")
        assert u.preferred_tone == "casual"

    def test_add_project_no_duplicates(self):
        u = UserModel()
        u.add_project("aion-c")
        u.add_project("aion-c")
        u.add_project(" soma ")
        assert u.projects == ["aion-c", "soma"]

    def test_remove_project(self):
        u = UserModel()
        u.add_project("p1")
        u.remove_project("p1")
        assert u.projects == []

    def test_facts_set_get(self):
        u = UserModel()
        u.set_fact("editor", "vscode")
        assert u.get_fact("editor") == "vscode"
        assert u.get_fact("missing") is None

    def test_to_from_json_roundtrip(self):
        u = UserModel(name="Jesus", preferred_language="es", technical_level="advanced",
                       preferred_tone="casual", projects=["aion"], facts={"k": "v"})
        s = u.to_json()
        u2 = UserModel.from_json(s)
        assert u2.name == "Jesus"
        assert u2.preferred_language == "es"
        assert u2.technical_level == "advanced"
        assert u2.preferred_tone == "casual"
        assert u2.projects == ["aion"]
        assert u2.facts == {"k": "v"}

    def test_save_load_via_mem(self):
        mem = FakeMem()
        u = UserModel(name="Jesus", preferred_language="es")
        u.save_to_mem(mem)
        assert USER_MODEL_KEY in mem.entries
        assert mem.entries[USER_MODEL_KEY][1] == USER_MODEL_DOMAIN
        u2 = UserModel.load_from_mem(mem)
        assert u2 is not None
        assert u2.name == "Jesus"

    def test_load_from_empty_mem(self):
        assert UserModel.load_from_mem(FakeMem()) is None

    def test_save_to_none_mem_safe(self):
        UserModel().save_to_mem(None)  # no debe lanzar

    def test_render_for_context(self):
        u = UserModel(name="Jesus", preferred_language="es",
                       technical_level="advanced", projects=["aion"])
        out = u.render_for_context()
        assert "name=Jesus" in out
        assert "lang=es" in out
        assert "level=advanced" in out
        assert "aion" in out

    def test_render_empty_returns_empty(self):
        assert UserModel().render_for_context() == ""


# ─────────────────────────────────────────────────────────────────────────────
# normalize_query
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalize:
    def test_lowercase(self):
        assert normalize_query("HELLO") == "hello"

    def test_collapses_whitespace(self):
        assert normalize_query("hello  world\t\nfoo") == "hello world foo"

    def test_strips_punctuation_terminal(self):
        assert normalize_query("hola!!!") == "hola"
        assert normalize_query("¿qué tal?") == "qué tal"

    def test_handles_none(self):
        assert normalize_query(None) == ""


# ─────────────────────────────────────────────────────────────────────────────
# ResponseCache
# ─────────────────────────────────────────────────────────────────────────────


class TestResponseCache:
    def test_set_get_basic(self):
        c = ResponseCache()
        c.set("hola", "respuesta 1")
        assert c.get("hola") == "respuesta 1"

    def test_normalized_key_collisions(self):
        c = ResponseCache()
        c.set("Hola!", "x")
        assert c.get("hola") == "x"
        assert c.get("HOLA?") == "x"

    def test_miss_returns_none(self):
        c = ResponseCache()
        assert c.get("nada") is None

    def test_stats_track_hits_misses(self):
        c = ResponseCache()
        c.set("k", "v")
        c.get("k")
        c.get("missing")
        s = c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5

    def test_lru_eviction(self):
        c = ResponseCache(max_size=2)
        c.set("a", "1")
        c.set("b", "2")
        c.set("c", "3")  # eviction → "a" out
        assert c.get("a") is None
        assert c.get("b") == "2"
        assert c.get("c") == "3"

    def test_lru_touch_on_access(self):
        c = ResponseCache(max_size=2)
        c.set("a", "1")
        c.set("b", "2")
        c.get("a")  # touch a, b becomes the LRU
        c.set("c", "3")  # eviction → b out
        assert c.get("a") == "1"
        assert c.get("b") is None

    def test_ttl_expiration(self):
        c = ResponseCache()
        c.set("k", "v", ttl=0.05)
        assert c.get("k") == "v"
        time.sleep(0.06)
        assert c.get("k") is None

    def test_invalidate_by_substring(self):
        c = ResponseCache()
        c.set("what is javascript", "JavaScript is a programming language")
        c.set("what is python", "Python is a programming language")
        c.set("hello", "hi")
        n = c.invalidate_by_substring("javascript")
        assert n == 1
        assert c.get("what is javascript") is None
        assert c.get("what is python") == "Python is a programming language"

    def test_invalidate_by_predicate(self):
        c = ResponseCache()
        c.set("a", "x")
        c.set("b", "y")
        n = c.invalidate_by_predicate(lambda k, e: k == "a")
        assert n == 1
        assert c.get("a") is None

    def test_purge_expired(self):
        c = ResponseCache()
        c.set("a", "1", ttl=0.05)
        c.set("b", "2", ttl=10)
        time.sleep(0.06)
        n = c.purge_expired()
        assert n == 1
        assert c.get("b") == "2"

    def test_clear(self):
        c = ResponseCache()
        c.set("a", "1")
        c.clear()
        assert len(c) == 0
        assert c.stats()["hits"] == 0

    def test_invalid_max_size(self):
        with pytest.raises(ValueError):
            ResponseCache(max_size=0)

    def test_set_empty_query_ignored(self):
        c = ResponseCache()
        c.set("", "x")
        assert len(c) == 0


# ─────────────────────────────────────────────────────────────────────────────
# ConversationHistory
# ─────────────────────────────────────────────────────────────────────────────


class TestConversationHistory:
    def test_add_turns(self):
        h = ConversationHistory()
        h.add_user("hola")
        h.add_assistant("hola, qué tal")
        assert len(h) == 2

    def test_recent_window_limit(self):
        h = ConversationHistory(recent_window=4, mid_window=20)
        for i in range(10):
            h.add_user(f"q{i}")
            h.add_assistant(f"a{i}")
        recent = h.recent_turns()
        assert len(recent) == 4

    def test_mid_and_old_partition(self):
        h = ConversationHistory(recent_window=2, mid_window=6)
        # Total 10 mensajes:
        #   recent (last 2)
        #   mid (4 before recent → last 4 of the older 8)
        #   old (4 oldest)
        for i in range(10):
            h.add_user(f"u{i}")
        assert len(h.recent_turns()) == 2
        assert len(h.mid_turns()) == 4
        assert len(h.old_turns()) == 4

    def test_summary_block_uses_summarizer(self):
        called = {"n": 0}
        def fake_sum(turns):
            called["n"] += 1
            return f"SUMMARY of {len(turns)}"
        h = ConversationHistory(recent_window=2, mid_window=4, summarizer_fn=fake_sum)
        for i in range(6):
            h.add_user(f"u{i}")
        out = h.summary_block()
        assert called["n"] == 1
        assert "SUMMARY" in out

    def test_default_summarizer_pairs_user_assistant(self):
        turns = [
            Turn(role="user", content="hola"),
            Turn(role="assistant", content="hola, qué tal"),
            Turn(role="user", content="2+2"),
            Turn(role="assistant", content="4"),
        ]
        out = default_summarizer(turns)
        assert "hola" in out
        assert "2+2" in out
        assert "→" in out

    def test_extract_facts_from_old(self):
        h = ConversationHistory(recent_window=2, mid_window=4)
        h.add_user("Mi nombre es Jesus")
        h.add_assistant("hola Jesus")
        for i in range(6):
            h.add_user(f"q{i}")
        n = h.extract_facts_from_old()
        assert n >= 1
        assert any("Jesus" in f for f in h.key_facts())

    def test_render_context_layered(self):
        h = ConversationHistory(recent_window=2, mid_window=4)
        h.add_key_fact("user is jesus")
        h.add_user("hola")
        h.add_assistant("hola tú")
        out = h.render_context()
        assert "[FACTS:" in out
        assert "[USER: hola]" in out
        assert "[AION: hola tú]" in out

    def test_invalid_role_raises(self):
        h = ConversationHistory()
        with pytest.raises(ValueError):
            h.add_turn("alien", "x")

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            ConversationHistory(recent_window=0)
        with pytest.raises(ValueError):
            ConversationHistory(recent_window=5, mid_window=2)

    def test_stats(self):
        h = ConversationHistory(recent_window=2, mid_window=4)
        for i in range(8):
            h.add_user(f"u{i}")
        s = h.stats()
        assert s["total_turns"] == 8
        assert s["recent_turns"] == 2
        assert s["old_turns"] == 4

    def test_add_key_fact_no_duplicates(self):
        h = ConversationHistory()
        h.add_key_fact("a")
        h.add_key_fact("a")
        assert h.key_facts() == ["a"]
