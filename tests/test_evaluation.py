"""
tests/test_evaluation.py — Tests para evaluation/eval_prompts + evaluation/metrics
====================================================================================
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from evaluation.eval_prompts import (
    EvalPrompt, EVAL_PROMPTS,
    CORA_PROMPTS, FORGE_C_PROMPTS, AXIOM_PROMPTS, MUSE_PROMPTS, EMPATHY_PROMPTS,
    prompts_by_domain, prompts_for_domain,
)
from evaluation.metrics import (
    tokenize, ngrams, bleu_score, multi_reference_bleu,
    exact_match, contains_any,
    GenerationQualityResult, generation_quality_score,
    WEIGHT_EXACT_MATCH, WEIGHT_BLEU, WEIGHT_ROUTING_ACCURACY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Eval prompts: estructura y completitud
# ─────────────────────────────────────────────────────────────────────────────


class TestEvalPromptsStructure:
    def test_total_count_50(self):
        assert len(EVAL_PROMPTS) == 50

    def test_10_per_domain(self):
        assert len(CORA_PROMPTS) == 10
        assert len(FORGE_C_PROMPTS) == 10
        assert len(AXIOM_PROMPTS) == 10
        assert len(MUSE_PROMPTS) == 10
        assert len(EMPATHY_PROMPTS) == 10

    def test_all_have_required_fields(self):
        for p in EVAL_PROMPTS:
            assert isinstance(p, EvalPrompt)
            assert p.query.strip()
            assert p.domain in ("cora", "forge_c", "axiom", "muse", "empathy")
            assert p.expected_substring.strip()
            assert p.language in ("es", "en")

    def test_all_have_at_least_one_reference(self):
        for p in EVAL_PROMPTS:
            assert len(p.references) >= 1, f"{p.query} has no references"

    def test_expected_substring_in_at_least_one_reference(self):
        """El expected_substring debería estar presente en al menos una reference."""
        for p in EVAL_PROMPTS:
            found = any(p.expected_substring.lower() in r.lower() for r in p.references)
            assert found, (
                f"prompt '{p.query[:40]}' has expected_substring '{p.expected_substring}' "
                f"that doesn't appear in any reference"
            )

    def test_to_dict_serializable(self):
        d = EVAL_PROMPTS[0].to_dict()
        import json
        json.dumps(d)  # no debe lanzar


class TestPromptsHelpers:
    def test_prompts_by_domain(self):
        groups = prompts_by_domain()
        assert set(groups.keys()) == {"cora", "forge_c", "axiom", "muse", "empathy"}
        for dom, plist in groups.items():
            assert len(plist) == 10

    def test_prompts_for_domain(self):
        cora = prompts_for_domain("cora")
        assert len(cora) == 10
        for p in cora:
            assert p.domain == "cora"

    def test_prompts_for_unknown_domain(self):
        assert prompts_for_domain("nonexistent") == []


class TestLanguageBalance:
    def test_each_domain_has_both_languages(self):
        groups = prompts_by_domain()
        for dom, plist in groups.items():
            langs = {p.language for p in plist}
            assert "es" in langs and "en" in langs, f"{dom} missing one language"


# ─────────────────────────────────────────────────────────────────────────────
# Métricas: tokenize / ngrams
# ─────────────────────────────────────────────────────────────────────────────


class TestTokenize:
    def test_basic_words(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_lowercase(self):
        assert tokenize("HELLO World") == ["hello", "world"]

    def test_punctuation_separated(self):
        toks = tokenize("hola, qué tal?")
        assert "hola" in toks
        assert "," in toks
        assert "qué" in toks
        assert "?" in toks

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize(None) == []

    def test_unicode_spanish(self):
        toks = tokenize("niño año")
        assert "niño" in toks
        assert "año" in toks


class TestNgrams:
    def test_unigrams(self):
        assert ngrams(["a", "b", "c"], 1) == [("a",), ("b",), ("c",)]

    def test_bigrams(self):
        assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]

    def test_trigrams(self):
        assert ngrams(["a", "b", "c", "d"], 3) == [("a", "b", "c"), ("b", "c", "d")]

    def test_empty(self):
        assert ngrams([], 1) == []
        assert ngrams(["a"], 2) == []  # not enough tokens

    def test_invalid_n(self):
        assert ngrams(["a", "b"], 0) == []
        assert ngrams(["a", "b"], -1) == []


# ─────────────────────────────────────────────────────────────────────────────
# BLEU
# ─────────────────────────────────────────────────────────────────────────────


class TestBleu:
    def test_perfect_match_high(self):
        # Identical sentences should give high BLEU
        score = bleu_score("hello world", "hello world")
        assert score > 0.9

    def test_no_overlap_low(self):
        score = bleu_score("hello world", "foo bar baz")
        assert score < 0.1

    def test_partial_overlap(self):
        score = bleu_score("the quick brown fox", "the slow brown dog")
        # 2 tokens overlap of 4 → ~0.5 unigram precision, lower bigram
        assert 0.0 < score < 0.7

    def test_empty_inputs(self):
        assert bleu_score("", "hello") == 0.0
        assert bleu_score("hello", "") == 0.0
        assert bleu_score("", "") == 0.0

    def test_case_insensitive(self):
        s1 = bleu_score("Hello World", "HELLO WORLD")
        s2 = bleu_score("hello world", "hello world")
        assert abs(s1 - s2) < 0.01

    def test_brevity_penalty_short_hypothesis(self):
        # Hypothesis much shorter than reference → BP penalizes
        long_score  = bleu_score("the quick brown fox jumps", "the quick brown fox jumps")
        short_score = bleu_score("the quick brown fox jumps", "the quick")
        assert short_score < long_score

    def test_max_n_unigram_only(self):
        score = bleu_score("hello world", "hello", max_n=1)
        # Only 1-gram, hyp has 1 of 2 ref unigrams matching
        assert score > 0.0

    def test_returns_float_in_range(self):
        score = bleu_score("any text here", "completely different words")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestMultiReferenceBleu:
    def test_picks_best_reference(self):
        refs = ["completely wrong reference one", "the cat sat on the mat"]
        hyp = "the cat sat on the mat"
        score = multi_reference_bleu(refs, hyp)
        # Debe usar la mejor reference (la segunda) → score alto
        assert score > 0.9

    def test_empty_references_returns_zero(self):
        assert multi_reference_bleu([], "hello") == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Exact match / contains_any
# ─────────────────────────────────────────────────────────────────────────────


class TestExactMatch:
    def test_substring_present(self):
        assert exact_match("the answer is 42 indeed", "42")

    def test_case_insensitive(self):
        assert exact_match("HELLO WORLD", "hello")

    def test_substring_absent(self):
        assert not exact_match("the answer is 42", "43")

    def test_empty(self):
        assert not exact_match("", "x")
        assert not exact_match("hello", "")


class TestContainsAny:
    def test_finds_one(self):
        assert contains_any("hello world", ["foo", "world", "bar"])

    def test_no_match(self):
        assert not contains_any("hello world", ["foo", "bar"])

    def test_empty_text(self):
        assert not contains_any("", ["foo"])


# ─────────────────────────────────────────────────────────────────────────────
# generation_quality_score
# ─────────────────────────────────────────────────────────────────────────────


def _make_prompts():
    return [
        EvalPrompt(
            query="what is 2+2",
            domain="axiom",
            expected_substring="4",
            references=["4", "the answer is 4"],
        ),
        EvalPrompt(
            query="hola",
            domain="cora",
            expected_substring="hola",
            references=["hola, en qué te ayudo"],
        ),
    ]


class TestGenerationQualityScore:
    def test_perfect_score(self):
        prompts = _make_prompts()
        def gen(query):
            if "2+2" in query:
                return "the answer is 4", "axiom"
            return "hola, en qué te ayudo", "cora"
        result = generation_quality_score(prompts, gen)
        assert result.n_prompts == 2
        assert result.exact_match == 1.0
        assert result.routing_accuracy == 1.0
        assert result.bleu > 0.5
        assert result.combined > 0.7

    def test_zero_score(self):
        prompts = _make_prompts()
        def gen(query):
            return "completely wrong", "muse"  # wrong text + wrong motor
        result = generation_quality_score(prompts, gen)
        assert result.exact_match == 0.0
        assert result.routing_accuracy == 0.0

    def test_partial_routing_correct(self):
        prompts = _make_prompts()
        def gen(query):
            if "2+2" in query:
                return "5", "axiom"  # wrong answer but correct routing
            return "wrong", "wrong_motor"
        result = generation_quality_score(prompts, gen)
        assert result.routing_accuracy == 0.5  # 1 of 2 correct
        assert result.exact_match == 0.0

    def test_per_domain_breakdown(self):
        prompts = _make_prompts()
        def gen(query):
            return "4", "axiom" if "2+2" in query else "cora"
        result = generation_quality_score(prompts, gen)
        assert "axiom" in result.per_domain
        assert "cora" in result.per_domain
        assert result.per_domain["axiom"]["n"] == 1
        assert result.per_domain["cora"]["n"] == 1

    def test_per_prompt_records(self):
        prompts = _make_prompts()
        def gen(query):
            return "4", "axiom"
        result = generation_quality_score(prompts, gen)
        assert len(result.per_prompt) == 2
        assert all("query" in p for p in result.per_prompt)
        assert all("exact" in p for p in result.per_prompt)

    def test_generator_exception_treated_as_failure(self):
        prompts = _make_prompts()
        def gen(query):
            raise RuntimeError("model crashed")
        result = generation_quality_score(prompts, gen)
        assert result.exact_match == 0.0
        assert result.routing_accuracy == 0.0
        assert all("error" in p for p in result.per_prompt)

    def test_empty_prompts(self):
        def gen(query): return "", None
        result = generation_quality_score([], gen)
        assert result.n_prompts == 0
        assert result.combined == 0.0

    def test_combined_uses_documented_weights(self):
        prompts = _make_prompts()
        def gen(query):
            # Always exact match, never routing match, low bleu
            if "2+2" in query: return "4", "wrong"
            return "hola", "wrong"
        result = generation_quality_score(prompts, gen)
        # exact_match = 1.0, routing = 0.0, bleu varies
        expected = (
            WEIGHT_EXACT_MATCH * 1.0
            + WEIGHT_BLEU * result.bleu
            + WEIGHT_ROUTING_ACCURACY * 0.0
        )
        assert abs(result.combined - expected) < 1e-6

    def test_to_dict_serializable(self):
        import json
        prompts = _make_prompts()
        result = generation_quality_score(prompts, lambda q: ("4", "axiom"))
        d = result.to_dict()
        json.dumps(d)  # no debe lanzar


class TestWeightsSumToOne:
    def test_weights_sum_to_one(self):
        total = WEIGHT_EXACT_MATCH + WEIGHT_BLEU + WEIGHT_ROUTING_ACCURACY
        assert abs(total - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Integration: usar los 50 prompts reales con un gen mock
# ─────────────────────────────────────────────────────────────────────────────


class TestEvalPromptsIntegration:
    def test_real_prompts_mock_generator(self):
        # Mock que devuelve la primera reference de cada prompt
        prompts = list(EVAL_PROMPTS)
        # Map query → expected reference
        ref_map = {p.query: (p.references[0], p.domain) for p in prompts}
        def gen(q): return ref_map.get(q, ("", None))
        result = generation_quality_score(prompts, gen)
        assert result.n_prompts == 50
        # Con respuestas perfectas, exact_match y routing deben ser ≥0.9
        assert result.exact_match > 0.9
        assert result.routing_accuracy > 0.9
        assert result.combined > 0.7

    def test_real_prompts_random_generator(self):
        prompts = list(EVAL_PROMPTS)
        def gen(q): return ("nonsense response", "wrong_motor")
        result = generation_quality_score(prompts, gen)
        assert result.exact_match == 0.0
        assert result.routing_accuracy == 0.0
