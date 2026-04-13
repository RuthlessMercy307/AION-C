"""
tests/test_canonical_dataloader.py — Tests para Fase C helpers
================================================================

Cubre los helpers que el training script usa para muestrear el dataset
canónico:
  - load_canonical_records      — read del .jsonl
  - balanced_indices            — 50/50 SKILL/MEM-or-not
  - weighted_sampler_indices    — muestreo continuo balanceado
  - domain_to_motor_idx         — mapeo dominio → motor index
  - encode_record               — tokenización + EOS append
  - quick_stats                 — sanity stats
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from synth.canonical_format import build_record, CanonicalRecord
from synth.canonical_dataloader import (
    DOMAIN_TO_MOTOR_IDX, MOTOR_NAMES, EOS_TOKEN_ID,
    load_canonical_records, domain_to_motor_idx,
    encode_record, balanced_indices, weighted_sampler_indices,
    quick_stats,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mocks
# ─────────────────────────────────────────────────────────────────────────────


class FakeTokenizer:
    """Mock muy simple: cada char → su ord (mod 1000)."""
    vocab_size = 1000

    def encode(self, text, max_len=None):
        ids = [(ord(c) % 999) + 1 for c in (text or "")]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids):
        return "".join(chr((i - 1) % 999) for i in ids)


def _make_records():
    return [
        build_record(user="a", aion="1", domain="forge_c"),                       # no skill/mem
        build_record(user="b", aion="2", domain="cora", skill="s"),               # skill
        build_record(user="c", aion="3", domain="axiom", mem="m"),                # mem
        build_record(user="d", aion="4", domain="muse", skill="s", mem="m"),      # both
        build_record(user="e", aion="5", domain="empathy"),                       # neither
        build_record(user="f", aion="6", domain="general"),                       # neither
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Domain mapping
# ─────────────────────────────────────────────────────────────────────────────


class TestDomainMapping:
    def test_known_domains(self):
        assert domain_to_motor_idx("cora") == 0
        assert domain_to_motor_idx("forge_c") == 1
        assert domain_to_motor_idx("muse") == 2
        assert domain_to_motor_idx("axiom") == 3
        assert domain_to_motor_idx("empathy") == 4

    def test_general_falls_back_to_cora(self):
        assert domain_to_motor_idx("general") == 0

    def test_unknown_domain_falls_back(self):
        assert domain_to_motor_idx("nonexistent") == 0

    def test_motor_names_count(self):
        assert len(MOTOR_NAMES) == 5


# ─────────────────────────────────────────────────────────────────────────────
# encode_record
# ─────────────────────────────────────────────────────────────────────────────


class TestEncodeRecord:
    def test_appends_eos(self):
        tok = FakeTokenizer()
        r = build_record(user="a", aion="b")
        ids = encode_record(tok, r)
        assert ids[-1] == EOS_TOKEN_ID

    def test_does_not_double_append_eos(self):
        # Si la tokenización ya devuelve EOS al final, NO se anexa otro EOS
        class TokWithEos:
            vocab_size = 1000
            def encode(self, text, ml=None):
                return [10, 11, 12, EOS_TOKEN_ID]
        ids = encode_record(TokWithEos(), build_record(user="x", aion="y"))
        assert ids[-1] == EOS_TOKEN_ID
        # No double-EOS at the end (no two EOS in a row)
        assert ids[-2] != EOS_TOKEN_ID
        # No se anexó nada extra
        assert ids == [10, 11, 12, EOS_TOKEN_ID]

    def test_respects_max_len(self):
        tok = FakeTokenizer()
        r = build_record(user="a" * 500, aion="b")
        ids = encode_record(tok, r, max_len=64)
        assert len(ids) <= 64


# ─────────────────────────────────────────────────────────────────────────────
# balanced_indices
# ─────────────────────────────────────────────────────────────────────────────


class TestBalancedIndices:
    def test_default_50_50(self):
        records = _make_records()
        idx = balanced_indices(records, target_ratio=0.5, seed=42)
        assert len(idx) == len(records)
        n_with = sum(1 for i in idx if records[i].has_skill or records[i].has_mem)
        n_without = len(idx) - n_with
        # Con 6 records: target 3/3
        assert n_with == 3
        assert n_without == 3

    def test_target_ratio_zero(self):
        records = _make_records()
        idx = balanced_indices(records, target_ratio=0.0, seed=42)
        for i in idx:
            assert not (records[i].has_skill or records[i].has_mem)

    def test_target_ratio_one(self):
        records = _make_records()
        idx = balanced_indices(records, target_ratio=1.0, seed=42)
        for i in idx:
            assert records[i].has_skill or records[i].has_mem

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            balanced_indices(_make_records(), target_ratio=1.5)
        with pytest.raises(ValueError):
            balanced_indices(_make_records(), target_ratio=-0.1)

    def test_empty_records(self):
        assert balanced_indices([]) == []

    def test_deterministic_with_seed(self):
        records = _make_records()
        a = balanced_indices(records, seed=99)
        b = balanced_indices(records, seed=99)
        assert a == b


# ─────────────────────────────────────────────────────────────────────────────
# weighted_sampler_indices
# ─────────────────────────────────────────────────────────────────────────────


class TestWeightedSampler:
    def test_produces_n_steps(self):
        records = _make_records()
        idx = weighted_sampler_indices(records, n_steps=100)
        assert len(idx) == 100

    def test_approximate_50_50(self):
        records = _make_records()
        idx = weighted_sampler_indices(records, n_steps=1000, target_ratio=0.5, seed=1)
        n_with = sum(1 for i in idx if records[i].has_skill or records[i].has_mem)
        ratio = n_with / 1000
        # Tolerancia ±5%
        assert 0.45 <= ratio <= 0.55

    def test_target_ratio_75(self):
        records = _make_records()
        idx = weighted_sampler_indices(records, n_steps=1000, target_ratio=0.75, seed=1)
        n_with = sum(1 for i in idx if records[i].has_skill or records[i].has_mem)
        ratio = n_with / 1000
        assert 0.70 <= ratio <= 0.80

    def test_handles_empty_with_pool(self):
        records = [
            build_record(user="a", aion="1"),
            build_record(user="b", aion="2"),
        ]
        # Ningún record tiene skill/mem
        idx = weighted_sampler_indices(records, n_steps=50)
        assert len(idx) == 50

    def test_empty_records(self):
        assert weighted_sampler_indices([], n_steps=10) == []


# ─────────────────────────────────────────────────────────────────────────────
# load_canonical_records / quick_stats
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadAndStats:
    def test_load_and_count(self, tmp_path):
        from synth.dataset_unifier import write_canonical_jsonl
        records = _make_records()
        path = tmp_path / "x.jsonl"
        write_canonical_jsonl(records, path)
        loaded = load_canonical_records(path)
        assert len(loaded) == len(records)

    def test_quick_stats(self):
        records = _make_records()
        s = quick_stats(records)
        assert s.total == 6
        assert s.n_with == 3   # 3 records con skill o mem
        assert s.n_without == 3
        assert "forge_c" in s.by_domain

    def test_load_skips_blank_lines(self, tmp_path):
        path = tmp_path / "x.jsonl"
        path.write_text(
            json.dumps({"text": "[USER: a]\n[AION: b]\n[EOS]"}) + "\n\n\n" +
            json.dumps({"text": "[USER: c]\n[AION: d]\n[EOS]"}) + "\n",
            encoding="utf-8",
        )
        loaded = load_canonical_records(path)
        assert len(loaded) == 2
