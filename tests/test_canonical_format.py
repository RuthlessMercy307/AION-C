"""
tests/test_canonical_format.py — Tests para Fase B del MEGA-PROMPT
====================================================================

Cubre:
  CanonicalTurn / CanonicalRecord — serialización
  format_record() / build_record() — todos los slots, multi-turn, EOS garantizado
  has_eos / strip_eos / count_tags
  parse_canonical — parser tolerante con balanceo
  canonicalize_legacy — conversión del formato {input, output, ...}
  Generators — los 5 (conversational, tool, skill, mem, identity)
  Unifier — read/write/merge, verify_eos_all, fix_eos
  Diversity stats
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from synth.canonical_format import (
    CanonicalRecord, CanonicalTurn,
    format_record, build_record,
    has_eos, strip_eos, count_tags,
    parse_canonical, canonicalize_legacy,
    EOS_MARKER, TAG_USER, TAG_AION, TAG_TOOL, TAG_RESULT,
)


# ─────────────────────────────────────────────────────────────────────────────
# format_record / EOS / build_record
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatRecord:
    def test_minimal_user_aion(self):
        text = format_record(user="hola", aion="hola, qué tal")
        assert text.startswith("[USER: hola]")
        assert "[AION: hola, qué tal]" in text
        assert text.endswith(EOS_MARKER)

    def test_with_skill_and_mem(self):
        text = format_record(
            user="x", aion="y",
            skill="be concise",
            mem="user=jesus",
        )
        assert "[SKILL: be concise]" in text
        assert "[MEM: user=jesus]" in text
        # Order: SKILL → MEM → USER → AION → EOS
        idx_skill = text.find("[SKILL:")
        idx_mem   = text.find("[MEM:")
        idx_user  = text.find("[USER:")
        idx_aion  = text.find("[AION:")
        assert idx_skill < idx_mem < idx_user < idx_aion

    def test_with_tool_result(self):
        text = format_record(
            user="run x", aion="done",
            tool='{"action":"run_code","input":"x"}',
            result="ok",
        )
        assert "[TOOL:" in text
        assert "[RESULT: ok]" in text
        # USER → TOOL → RESULT → AION
        idx_user = text.find("[USER:")
        idx_tool = text.find("[TOOL:")
        idx_result = text.find("[RESULT:")
        idx_aion = text.find("[AION:")
        assert idx_user < idx_tool < idx_result < idx_aion

    def test_multi_turn(self):
        text = format_record(
            user="hola", aion="hola tú",
            extra_turns=[
                CanonicalTurn(user="2+2", aion="4"),
                CanonicalTurn(user="gracias", aion="de nada"),
            ],
        )
        # 3 USER blocks total
        assert text.count("[USER:") == 3
        assert text.count("[AION:") == 3
        assert text.endswith(EOS_MARKER)

    def test_always_has_eos(self):
        text = format_record(user="x", aion="y")
        assert has_eos(text)


class TestBuildRecord:
    def test_returns_canonical_record(self):
        r = build_record(user="x", aion="y", domain="forge_c", language="es")
        assert isinstance(r, CanonicalRecord)
        assert r.domain == "forge_c"
        assert r.language == "es"
        assert not r.has_skill
        assert not r.has_mem
        assert not r.has_tool

    def test_flags_set_correctly(self):
        r = build_record(user="x", aion="y", skill="s", mem="m", tool="t", result="r")
        assert r.has_skill
        assert r.has_mem
        assert r.has_tool

    def test_multi_turn_flag(self):
        r = build_record(user="x", aion="y", extra_turns=[CanonicalTurn("a", "b")])
        assert r.is_multi_turn
        assert r.turn_count == 2

    def test_dict_roundtrip(self):
        r = build_record(user="x", aion="y", domain="cora", language="en", skill="s")
        d = r.to_dict()
        r2 = CanonicalRecord.from_dict(d)
        assert r2.text == r.text
        assert r2.has_skill
        assert r2.domain == "cora"


# ─────────────────────────────────────────────────────────────────────────────
# EOS helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestEosHelpers:
    def test_has_eos_true(self):
        assert has_eos("foo\n[EOS]")
        assert has_eos("foo [EOS]")

    def test_has_eos_false(self):
        assert not has_eos("foo")
        assert not has_eos("")
        assert not has_eos("[EOS] foo")  # not at end

    def test_strip_eos(self):
        assert strip_eos("foo\n[EOS]") == "foo"
        assert strip_eos("foo") == "foo"

    def test_count_tags(self):
        text = "[USER: a] [AION: b] [USER: c] [AION: d]"
        counts = count_tags(text)
        assert counts["USER"] == 2
        assert counts["AION"] == 2
        assert counts["SKILL"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────


class TestParser:
    def test_parses_simple(self):
        text = format_record(user="hola", aion="qué tal")
        blocks = parse_canonical(text)
        assert ("USER", "hola") in blocks
        assert ("AION", "qué tal") in blocks

    def test_parses_full_record(self):
        text = format_record(
            user="x", aion="y",
            skill="s", mem="m",
            tool='{"action":"x"}', result="ok",
        )
        blocks = parse_canonical(text)
        tags = [b[0] for b in blocks]
        assert tags == ["SKILL", "MEM", "USER", "TOOL", "RESULT", "AION"]

    def test_parses_multi_turn(self):
        text = format_record(
            user="a", aion="b",
            extra_turns=[CanonicalTurn("c", "d")],
        )
        blocks = parse_canonical(text)
        users = [b[1] for b in blocks if b[0] == "USER"]
        assert users == ["a", "c"]

    def test_handles_nested_brackets_in_content(self):
        # tool JSON contiene corchetes/llaves
        text = format_record(
            user="run", aion="done",
            tool='{"input":{"path":"a.py","args":[1,2]}}',
        )
        blocks = parse_canonical(text)
        tool_block = [b for b in blocks if b[0] == "TOOL"][0]
        assert "args" in tool_block[1]

    def test_empty_text(self):
        assert parse_canonical("") == []
        assert parse_canonical(None) == []

    def test_strips_eos(self):
        text = "[USER: x]\n[AION: y]\n[EOS]"
        blocks = parse_canonical(text)
        assert len(blocks) == 2
        # EOS should not appear as a block
        assert not any(b[0] == "EOS" for b in blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy conversion
# ─────────────────────────────────────────────────────────────────────────────


class TestCanonicalizeLegacy:
    def test_basic_legacy_record(self):
        rec = {"input": "hola", "output": "hola, qué tal", "domain": "general", "language": "es"}
        cr = canonicalize_legacy(rec)
        assert has_eos(cr.text)
        assert "[USER: hola]" in cr.text
        assert "[AION: hola, qué tal]" in cr.text
        assert cr.domain == "general"
        assert cr.language == "es"

    def test_legacy_with_graph_becomes_mem(self):
        rec = {
            "input": "rain causes wet soil",
            "output": "yes",
            "domain": "cora",
            "language": "en",
            "graph": {
                "nodes": [{"id": "n0", "label": "rain"}, {"id": "n1", "label": "soil"}],
                "edges": [{"source": "n0", "target": "n1", "relation": "causes"}],
            },
        }
        cr = canonicalize_legacy(rec)
        assert cr.has_mem
        assert "rain" in cr.text
        assert "causes" in cr.text

    def test_legacy_empty_graph_no_mem(self):
        rec = {"input": "hi", "output": "hello", "domain": "general", "language": "en", "graph": {"nodes": [], "edges": []}}
        cr = canonicalize_legacy(rec)
        assert not cr.has_mem

    def test_eos_always_present(self):
        rec = {"input": "x", "output": "y"}
        cr = canonicalize_legacy(rec)
        assert has_eos(cr.text)


# ─────────────────────────────────────────────────────────────────────────────
# Generators (sanity, no full 12.5K aquí)
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerators:
    def test_conversational_small_batch(self):
        from synth.conversational_gen import generate_conversational
        records = list(generate_conversational(n=20, seed=1))
        assert len(records) == 20
        for r in records:
            assert has_eos(r.text)
            assert r.is_multi_turn  # all are multi-turn
            assert r.turn_count >= 2
            assert r.type == "multi_turn"

    def test_tool_calls_small_batch(self):
        from synth.tool_gen import generate_tool_calls
        records = list(generate_tool_calls(n=30, seed=1))
        assert len(records) == 30
        for r in records:
            assert has_eos(r.text)
            assert r.has_tool
            assert "[TOOL:" in r.text
            assert "[RESULT:" in r.text

    def test_tool_calls_cover_multiple_actions(self):
        from synth.tool_gen import generate_tool_calls
        records = list(generate_tool_calls(n=200, seed=1))
        actions = {r.metadata.get("action") for r in records}
        assert len(actions) >= 5  # debe cubrir al menos 5 tools distintos

    def test_skill_injected_small_batch(self):
        from synth.skill_injected_gen import generate_skill_injected
        records = list(generate_skill_injected(n=20, seed=1))
        assert len(records) == 20
        for r in records:
            assert has_eos(r.text)
            assert r.has_skill
            assert "[SKILL:" in r.text
            assert r.metadata.get("skill")

    def test_skill_uses_real_skill_files(self):
        from synth.skill_injected_gen import generate_skill_injected
        records = list(generate_skill_injected(n=100, seed=1))
        skills_used = {r.metadata.get("skill") for r in records}
        # Debe haber usado al menos 5 de los 11 skills
        assert len(skills_used) >= 5

    def test_mem_injected_small_batch(self):
        from synth.mem_injected_gen import generate_mem_injected
        records = list(generate_mem_injected(n=20, seed=1))
        assert len(records) == 20
        for r in records:
            assert has_eos(r.text)
            assert r.has_mem
            assert "[MEM:" in r.text

    def test_identity_small_batch(self):
        from synth.identity_gen import generate_identity
        records = list(generate_identity(n=20, seed=1))
        assert len(records) == 20
        for r in records:
            assert has_eos(r.text)
            assert r.type == "identity"

    def test_identity_covers_traits(self):
        from synth.identity_gen import generate_identity
        records = list(generate_identity(n=200, seed=1))
        traits = {r.metadata.get("trait") for r in records}
        assert len(traits) >= 6  # al menos 6 traits distintos

    def test_generators_deterministic_by_seed(self):
        from synth.conversational_gen import generate_conversational
        a = list(generate_conversational(n=10, seed=99))
        b = list(generate_conversational(n=10, seed=99))
        for ra, rb in zip(a, b):
            assert ra.text == rb.text


# ─────────────────────────────────────────────────────────────────────────────
# Unifier + diversity
# ─────────────────────────────────────────────────────────────────────────────


class TestUnifier:
    def test_compute_diversity_exact(self):
        from synth.dataset_unifier import compute_diversity_exact
        records = [
            build_record(user="a", aion="b", skill="s", domain="forge_c", language="es", type="skill"),
            build_record(user="a", aion="b", mem="m", domain="cora", language="en", type="mem"),
            build_record(user="a", aion="b", domain="general", language="es", type="single"),
            build_record(user="a", aion="b", tool="t", result="r", domain="forge_c", language="en", type="tool"),
        ]
        stats = compute_diversity_exact(records)
        assert stats.total == 4
        assert stats.with_skill == 1
        assert stats.with_mem == 1
        assert stats.with_tool == 1
        assert stats.eos_count == 4
        assert stats.eos_missing == 0
        assert stats.skill_or_mem_pct == 0.5  # 2 of 4 have skill or mem
        assert "forge_c" in stats.by_domain
        assert "es" in stats.by_language

    def test_verify_eos_all_passes(self):
        from synth.dataset_unifier import verify_eos_all
        records = [build_record(user="a", aion="b") for _ in range(10)]
        assert verify_eos_all(records) == 0

    def test_fix_eos_repairs(self):
        from synth.dataset_unifier import fix_eos, verify_eos_all
        from synth.canonical_format import strip_eos
        # Crea un record con EOS, lo strip-eamos a mano
        r = build_record(user="x", aion="y")
        r.text = strip_eos(r.text)
        assert not has_eos(r.text)
        fixed = list(fix_eos([r]))
        assert has_eos(fixed[0].text)

    def test_write_read_roundtrip(self, tmp_path):
        from synth.dataset_unifier import write_canonical_jsonl, read_canonical_jsonl
        records = [build_record(user=f"q{i}", aion=f"a{i}") for i in range(5)]
        path = tmp_path / "out.jsonl"
        n = write_canonical_jsonl(records, path)
        assert n == 5
        loaded = list(read_canonical_jsonl(path))
        assert len(loaded) == 5
        assert loaded[0].text == records[0].text

    def test_merge_and_shuffle(self):
        from synth.dataset_unifier import merge_and_shuffle
        a = [build_record(user="a", aion="x") for _ in range(5)]
        b = [build_record(user="b", aion="y") for _ in range(5)]
        merged = merge_and_shuffle([a, b], seed=42)
        assert len(merged) == 10

    def test_canonicalize_legacy_dataset(self):
        from synth.dataset_unifier import canonicalize_legacy_dataset
        legacy = [
            {"input": "hi", "output": "hello", "domain": "general", "language": "en"},
            {"input": "hola", "output": "qué tal", "domain": "general", "language": "es"},
        ]
        cr_list = list(canonicalize_legacy_dataset(legacy))
        assert len(cr_list) == 2
        assert all(has_eos(r.text) for r in cr_list)
