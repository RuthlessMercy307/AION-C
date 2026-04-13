"""
tests/test_self_check.py — Tests para Parte 7 del MEGA-PROMPT
==============================================================

Cubre:
  confidence_from_probs / classify_confidence / policy_for_confidence
  SelfChecker — longitud, eco, sintaxis Python, brackets, numérica
  ErrorLog — record + recall, con MEM y sin MEM (fallback local)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from agent.self_check import (
    ConfidenceLevel, SelfChecker, SelfCheckResult,
    confidence_from_probs, classify_confidence, policy_for_confidence,
    ErrorLog, ErrorRecord, ERROR_LOG_DOMAIN,
    HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD,
    _check_brackets_balanced, _check_python_syntax, _looks_like_python_code,
    _extract_numbers,
)


# ─────────────────────────────────────────────────────────────────────────────
# FakeMem
# ─────────────────────────────────────────────────────────────────────────────


class FakeMem:
    def __init__(self):
        self.entries = {}
        self.search_calls = []

    def store(self, key, value, domain="general", source="test"):
        self.entries[key] = (value, domain)

    def search(self, query, top_k=5, domain=None):
        self.search_calls.append((query, domain))
        out = []
        for k, (v, d) in self.entries.items():
            if domain and d != domain:
                continue
            out.append((k, v, 0.9))
        return out[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidence:
    def test_confidence_from_probs_average(self):
        assert confidence_from_probs([0.9, 0.8, 0.7, 0.6, 0.5]) == pytest.approx(0.7)

    def test_confidence_window_caps(self):
        # Solo los primeros 5 cuentan
        score = confidence_from_probs([1.0] * 5 + [0.0] * 100)
        assert score == 1.0

    def test_confidence_empty_list(self):
        assert confidence_from_probs([]) == 0.0

    def test_classify_high(self):
        assert classify_confidence(0.95) == ConfidenceLevel.HIGH
        assert classify_confidence(HIGH_CONFIDENCE_THRESHOLD) == ConfidenceLevel.HIGH

    def test_classify_medium(self):
        assert classify_confidence(0.65) == ConfidenceLevel.MEDIUM
        assert classify_confidence(LOW_CONFIDENCE_THRESHOLD) == ConfidenceLevel.MEDIUM

    def test_classify_low(self):
        assert classify_confidence(0.3) == ConfidenceLevel.LOW
        assert classify_confidence(0.0) == ConfidenceLevel.LOW

    def test_policy_for_each_level(self):
        assert policy_for_confidence(ConfidenceLevel.HIGH)   == "respond_directly"
        assert policy_for_confidence(ConfidenceLevel.MEDIUM) == "respond_with_disclaimer"
        assert policy_for_confidence(ConfidenceLevel.LOW)    == "search_then_respond"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_brackets_balanced_ok(self):
        assert _check_brackets_balanced("def f(x): return [1, {2: 3}]")

    def test_brackets_unbalanced(self):
        assert not _check_brackets_balanced("def f(x: return 1")
        assert not _check_brackets_balanced("[1, 2")
        assert not _check_brackets_balanced("{a: 1")

    def test_brackets_inside_strings_ignored(self):
        assert _check_brackets_balanced('print("(") and print(")")')

    def test_python_syntax_ok(self):
        assert _check_python_syntax("def f(x): return x + 1") is None

    def test_python_syntax_bad(self):
        err = _check_python_syntax("def f(x return")
        assert err is not None
        assert "syntax" in err.lower()

    def test_looks_like_python_positive(self):
        assert _looks_like_python_code("def f(): pass")
        assert _looks_like_python_code("import os")
        assert _looks_like_python_code("from x import y")

    def test_looks_like_python_negative(self):
        assert not _looks_like_python_code("hola, cómo estás")
        assert not _looks_like_python_code("")

    def test_extract_numbers(self):
        assert _extract_numbers("hay 3 perros y 2.5 gatos") == [3.0, 2.5]
        assert _extract_numbers("sin números") == []


# ─────────────────────────────────────────────────────────────────────────────
# SelfChecker
# ─────────────────────────────────────────────────────────────────────────────


class TestSelfChecker:
    def test_passes_normal_response(self):
        c = SelfChecker()
        r = c.check("hola", "hola, en qué te ayudo?")
        assert r.passed
        assert r.issues == []

    def test_fails_on_empty_response(self):
        c = SelfChecker()
        r = c.check("hola", "")
        assert not r.passed
        assert any("empty" in i or "short" in i for i in r.issues)

    def test_fails_on_too_long(self):
        c = SelfChecker(max_response_length=20)
        r = c.check("x", "a" * 100)
        assert not r.passed
        assert any("too long" in i for i in r.issues)

    def test_fails_on_echo(self):
        c = SelfChecker()
        r = c.check("hola", "Hola")
        assert not r.passed
        assert any("echo" in i for i in r.issues)

    def test_fails_on_python_syntax_error(self):
        c = SelfChecker()
        r = c.check("write a function", "def f(x return x + 1")
        assert not r.passed
        assert any("syntax" in i or "bracket" in i for i in r.issues)

    def test_passes_valid_python(self):
        c = SelfChecker()
        r = c.check("write a function", "def add(a, b):\n    return a + b")
        assert r.passed

    def test_fails_on_unbalanced_brackets(self):
        c = SelfChecker(check_code_syntax=True)
        # No "looks like python" pero sí brackets desbalanceados
        r = c.check("foo", "[1, 2")
        assert not r.passed

    def test_numeric_consistency_pass(self):
        c = SelfChecker()
        # Respuesta menciona los números del query → OK
        r = c.check("cuánto es 25% de 200?", "el 25% de 200 es 50")
        assert r.passed

    def test_numeric_consistency_fail_unrelated_numbers(self):
        c = SelfChecker()
        # query tiene 25 y 200, respuesta tiene números completamente distintos
        r = c.check("cuánto es 25% de 200?", "tengo 7 perros y 9 gatos")
        assert not r.passed
        assert any("numeric" in i for i in r.issues)

    def test_confidence_attached_when_provided(self):
        c = SelfChecker()
        r = c.check("x", "y", probs=[0.9, 0.85, 0.8, 0.78, 0.82])
        assert r.confidence is not None
        assert r.confidence_level == ConfidenceLevel.HIGH

    def test_confidence_not_attached_when_omitted(self):
        c = SelfChecker()
        r = c.check("x", "y")
        assert r.confidence is None
        assert r.confidence_level is None

    def test_low_confidence_does_not_fail_check(self):
        c = SelfChecker()
        r = c.check("x", "y", probs=[0.1, 0.2, 0.15])
        assert r.passed  # confidence baja NO marca passed=False
        assert r.confidence_level == ConfidenceLevel.LOW

    def test_self_check_result_add_issue(self):
        r = SelfCheckResult(passed=True)
        r.add_issue("test")
        assert not r.passed
        assert "test" in r.issues


# ─────────────────────────────────────────────────────────────────────────────
# ErrorLog
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorLog:
    def test_record_with_mem(self):
        mem = FakeMem()
        log = ErrorLog(mem)
        rec = log.record(
            error="syntax error in generated code",
            cause="unclosed bracket",
            prevention="check brackets",
            domain="forge_c",
        )
        assert isinstance(rec, ErrorRecord)
        # Debe haberse persistido en MEM con el dominio correcto
        assert any(d == ERROR_LOG_DOMAIN for _, d in mem.entries.values())

    def test_record_without_mem_uses_local(self):
        log = ErrorLog(mem=None)
        log.record("e", "c", "p", "d")
        assert len(log) == 1

    def test_recall_filters_by_domain_local(self):
        log = ErrorLog(mem=None)
        log.record("e1", "c1", "p1", "forge_c")
        log.record("e2", "c2", "p2", "muse")
        out = log.recall(domain="forge_c")
        assert len(out) == 1
        assert out[0].error == "e1"

    def test_recall_with_mem_uses_search(self):
        mem = FakeMem()
        log = ErrorLog(mem)
        log.record("e1", "c1", "p1", "forge_c")
        out = log.recall(query="syntax")
        assert len(out) >= 1
        # Debe haber pasado domain=ERROR_LOG_DOMAIN al search
        assert any(d == ERROR_LOG_DOMAIN for _, d in mem.search_calls)

    def test_record_preserves_text_format(self):
        rec = ErrorRecord(error="e", cause="c", prevention="p", domain="d")
        text = rec.to_text()
        assert "error: e" in text
        assert "cause: c" in text
        assert "prevention: p" in text
