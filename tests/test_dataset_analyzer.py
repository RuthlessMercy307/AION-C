"""
tests/test_dataset_analyzer.py — Tests para DatasetQualityAnalyzer
===================================================================

Cubre:
    - Metricas individuales: correctness, diversity, level_balance,
      relation_coverage, entity_spans
    - QualityReport: overall_score, grade, summary, WEIGHTS
    - DatasetQualityAnalyzer: analyze, recommend, pesos custom
    - Helpers: _normalized_entropy, _score_to_grade, _level_to_iterations
    - CLI: main() con argumentos
    - Integracion: dataset sintetico real de CausalGraphGenerator
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import pytest
from unittest.mock import patch

from core.graph import CausalRelation, CausalGraph, CausalNode, CausalEdge, NodeType
from synth.causal_graph_gen import (
    AnswerType,
    CausalExample,
    CausalGraphGenerator,
    verify_example,
)
from tools.dataset_analyzer import (
    ALL_DOMAINS,
    ALL_LEVELS,
    ALL_RELATIONS,
    GRADE_THRESHOLDS,
    N_ALL_ANSWER_TYPES,
    N_ALL_DOMAINS,
    N_ALL_RELATIONS,
    CorrectnessMetric,
    DatasetQualityAnalyzer,
    DiversityMetric,
    EntitySpansMetric,
    LevelBalanceMetric,
    QualityReport,
    RelationCoverageMetric,
    _dominant_key,
    _normalized_entropy,
    _score_to_grade,
    main,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_dataset():
    """50 ejemplos verificados — reutilizados por toda la suite del modulo."""
    gen = CausalGraphGenerator(seed=42)
    return gen.generate_batch(n=50, verify=True)


@pytest.fixture(scope="module")
def analyzer():
    return DatasetQualityAnalyzer()


@pytest.fixture(scope="module")
def report(analyzer, small_dataset):
    return analyzer.analyze(small_dataset)


def _make_example(
    level: int = 1,
    answer_type: AnswerType = AnswerType.TRANSITIVITY,
    domain: str = "clima",
    n_nodes: int = 2,
    relation: CausalRelation = CausalRelation.CAUSES,
    valid_spans: bool = True,
) -> CausalExample:
    """Fabrica un CausalExample minimal para tests unitarios."""
    g = CausalGraph()
    for i in range(n_nodes):
        g.add_node(CausalNode(node_id=f"n{i}", label=f"nodo{i}", node_type=NodeType.ENTITY))
    if n_nodes >= 2:
        g.add_edge(CausalEdge(source_id="n0", target_id="n1", relation=relation))

    spans = [(0, 1), (1, 2)] if valid_spans else [(-1, -1), (-1, -1)]
    return CausalExample(
        problem_text    = "nodo0 causa nodo1",
        graph           = g,
        answer          = "si",
        complexity_level= level,
        answer_type     = answer_type,
        metadata        = {"domain": domain, "source_id": "n0", "target_id": "n1",
                           "expected_reachable": True},
        entity_spans    = spans,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_n_all_relations(self):
        assert N_ALL_RELATIONS == 16
        assert len(ALL_RELATIONS) == 16

    def test_n_all_answer_types(self):
        assert N_ALL_ANSWER_TYPES == 7

    def test_n_all_domains(self):
        assert N_ALL_DOMAINS == 7

    def test_all_levels(self):
        assert ALL_LEVELS == [1, 2, 3, 4, 5]

    def test_grade_thresholds(self):
        assert GRADE_THRESHOLDS["A"] == 0.90
        assert GRADE_THRESHOLDS["B"] == 0.75
        assert GRADE_THRESHOLDS["C"] == 0.50


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizedEntropy:
    def test_uniform_is_1(self):
        assert abs(_normalized_entropy([25, 25, 25, 25], 4) - 1.0) < 1e-6

    def test_concentrated_is_low(self):
        # Todos en una categoria
        score = _normalized_entropy([100, 0, 0, 0], 4)
        assert score == 0.0

    def test_empty_is_zero(self):
        assert _normalized_entropy([], 4) == 0.0

    def test_single_category_is_zero(self):
        assert _normalized_entropy([10], 1) == 0.0

    def test_max_is_one(self):
        score = _normalized_entropy([1] * 8, 8)
        assert abs(score - 1.0) < 1e-6

    def test_partial_diversity(self):
        # Dos categorias de 4 con igual peso, n_categories=4 → H = log2(2)/log2(4) = 0.5
        score = _normalized_entropy([50, 50, 0, 0], 4)
        assert abs(score - 0.5) < 1e-6

    def test_all_zeros_is_zero(self):
        assert _normalized_entropy([0, 0, 0], 3) == 0.0


class TestScoreToGrade:
    def test_a(self):
        assert _score_to_grade(0.95) == "A"
        assert _score_to_grade(0.90) == "A"

    def test_b(self):
        assert _score_to_grade(0.85) == "B"
        assert _score_to_grade(0.75) == "B"

    def test_c(self):
        assert _score_to_grade(0.70) == "C"
        assert _score_to_grade(0.50) == "C"

    def test_d(self):
        assert _score_to_grade(0.49) == "D"
        assert _score_to_grade(0.0)  == "D"

    def test_boundary_a(self):
        assert _score_to_grade(0.8999) == "B"

    def test_boundary_b(self):
        assert _score_to_grade(0.7499) == "C"

    def test_boundary_c(self):
        assert _score_to_grade(0.4999) == "D"


class TestDominantKey:
    def test_single(self):
        assert _dominant_key({"a": 5}) == "a"

    def test_multiple(self):
        assert _dominant_key({"a": 3, "b": 10, "c": 1}) == "b"

    def test_empty(self):
        assert _dominant_key({}) == "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTNESS METRIC
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrectnessMetric:
    def test_perfect(self, small_dataset, analyzer):
        m = analyzer._compute_correctness(small_dataset)
        assert m.n_total == len(small_dataset)
        assert 0.0 <= m.score <= 1.0
        assert m.label != ""

    def test_all_verified(self, small_dataset, analyzer):
        """Todos los ejemplos del small_dataset fueron generados con verify=True."""
        m = analyzer._compute_correctness(small_dataset)
        assert m.score == 1.0

    def test_mixed(self, analyzer):
        """Algunos pasan, algunos no."""
        good = [_make_example(level=1, domain="clima") for _ in range(8)]
        # Crear ejemplo que NO pasa verify: sin aristas → TRANSITIVITY falla
        g = CausalGraph()
        g.add_node(CausalNode(node_id="x", label="x", node_type=NodeType.ENTITY))
        bad_ex = CausalExample(
            problem_text="x",
            graph=g,
            answer="si",
            complexity_level=1,
            answer_type=AnswerType.TRANSITIVITY,
            metadata={"source_id": "x", "target_id": "x", "expected_reachable": True},
            entity_spans=[],
        )
        examples = good + [bad_ex] + good  # 17 total, >=1 falla
        m = analyzer._compute_correctness(examples)
        assert m.n_total == 17
        assert m.score < 1.0

    def test_score_label_format(self):
        m = CorrectnessMetric(n_total=100, n_passed=95, score=0.95)
        assert "95/100" in m.label
        assert "95.0%" in m.label


# ─────────────────────────────────────────────────────────────────────────────
# DIVERSITY METRIC
# ─────────────────────────────────────────────────────────────────────────────

class TestDiversityMetric:
    def test_high_diversity_on_real_dataset(self, small_dataset, analyzer):
        m = analyzer._compute_diversity(small_dataset)
        assert m.score > 0.5   # dataset sintetico es diverso

    def test_all_same_type_is_low_diversity(self, analyzer):
        examples = [
            _make_example(answer_type=AnswerType.TRANSITIVITY, domain="clima", level=1)
            for _ in range(20)
        ]
        m = analyzer._compute_diversity(examples)
        # Todos el mismo tipo → answer_type_entropy = 0
        assert m.answer_type_entropy == 0.0

    def test_uniform_types_is_high_diversity(self, analyzer):
        types = list(AnswerType)
        examples = [
            _make_example(answer_type=t, domain="clima", level=i % 5 + 1)
            for i, t in enumerate(types * 3)   # 3 de cada tipo
        ]
        m = analyzer._compute_diversity(examples)
        assert m.answer_type_entropy > 0.9

    def test_domain_entropy_without_metadata(self, analyzer):
        """Sin metadata['domain'], domain entropy deberia ser baja."""
        examples = [
            CausalExample(
                problem_text="x",
                graph=CausalGraph(),
                answer="x",
                complexity_level=1,
                answer_type=AnswerType.DIRECT_CAUSE,
                metadata={},   # sin domain
                entity_spans=[],
            )
            for _ in range(5)
        ]
        m = analyzer._compute_diversity(examples)
        assert 0.0 <= m.score <= 1.0   # no debe crashear

    def test_all_scores_in_01(self, small_dataset, analyzer):
        m = analyzer._compute_diversity(small_dataset)
        assert 0.0 <= m.answer_type_entropy <= 1.0
        assert 0.0 <= m.level_entropy       <= 1.0
        assert 0.0 <= m.domain_entropy      <= 1.0
        assert 0.0 <= m.score               <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL BALANCE METRIC
# ─────────────────────────────────────────────────────────────────────────────

class TestLevelBalanceMetric:
    def test_perfect_balance_is_1(self, analyzer):
        # 4 ejemplos de cada nivel → perfectamente uniforme
        examples = [_make_example(level=l) for l in ALL_LEVELS for _ in range(4)]
        m = analyzer._compute_level_balance(examples)
        assert abs(m.score - 1.0) < 1e-6

    def test_all_same_level_is_low(self, analyzer):
        examples = [_make_example(level=1) for _ in range(20)]
        m = analyzer._compute_level_balance(examples)
        assert m.score < 0.5

    def test_level_fractions_sum_to_1(self, small_dataset, analyzer):
        m = analyzer._compute_level_balance(small_dataset)
        total = sum(m.level_fractions.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_levels_present_in_fractions(self, analyzer):
        examples = [_make_example(level=l) for l in ALL_LEVELS for _ in range(2)]
        m = analyzer._compute_level_balance(examples)
        for l in ALL_LEVELS:
            assert l in m.level_fractions

    def test_score_in_01(self, small_dataset, analyzer):
        m = analyzer._compute_level_balance(small_dataset)
        assert 0.0 <= m.score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# RELATION COVERAGE METRIC
# ─────────────────────────────────────────────────────────────────────────────

class TestRelationCoverageMetric:
    def test_score_post_init(self):
        m = RelationCoverageMetric(n_covered=12, n_total=16)
        assert abs(m.score - 12/16) < 1e-6

    def test_real_dataset_coverage(self, small_dataset, analyzer):
        m = analyzer._compute_relation_coverage(small_dataset)
        assert m.n_total == 16
        assert 0 <= m.n_covered <= 16
        assert 0.0 <= m.score <= 1.0

    def test_missing_relations_listed(self, analyzer):
        # Solo CAUSES
        examples = [_make_example(relation=CausalRelation.CAUSES) for _ in range(10)]
        m = analyzer._compute_relation_coverage(examples)
        assert CausalRelation.CAUSES.value not in m.missing
        assert len(m.missing) == 15   # 16 - 1 = 15 faltan

    def test_empty_graph_counts_nothing(self, analyzer):
        empty_ex = CausalExample(
            problem_text="x", graph=CausalGraph(), answer="x",
            complexity_level=1, answer_type=AnswerType.DIRECT_CAUSE,
        )
        m = analyzer._compute_relation_coverage([empty_ex])
        assert m.n_covered == 0
        assert len(m.missing) == 16

    def test_all_relations_covered(self, analyzer):
        """Un ejemplo con cada tipo de relacion → cobertura 100%."""
        examples = [_make_example(relation=rel) for rel in CausalRelation]
        m = analyzer._compute_relation_coverage(examples)
        assert m.n_covered == 16
        assert m.score == 1.0
        assert m.missing == []


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY SPANS METRIC
# ─────────────────────────────────────────────────────────────────────────────

class TestEntitySpansMetric:
    def test_all_valid(self, analyzer):
        examples = [_make_example(valid_spans=True) for _ in range(10)]
        m = analyzer._compute_entity_spans(examples)
        assert m.score == 1.0
        assert m.n_with_valid_spans == 10

    def test_none_valid(self, analyzer):
        examples = [_make_example(valid_spans=False) for _ in range(10)]
        m = analyzer._compute_entity_spans(examples)
        assert m.score == 0.0
        assert m.n_with_valid_spans == 0

    def test_real_dataset(self, small_dataset, analyzer):
        m = analyzer._compute_entity_spans(small_dataset)
        assert 0.0 <= m.score <= 1.0

    def test_empty_spans_treated_as_invalid(self, analyzer):
        ex = CausalExample(
            problem_text="x", graph=CausalGraph(), answer="x",
            complexity_level=1, answer_type=AnswerType.DIRECT_CAUSE,
            entity_spans=[],  # lista vacia → sin spans validos
        )
        m = analyzer._compute_entity_spans([ex])
        assert m.score == 0.0

    def test_label_format(self):
        m = EntitySpansMetric(n_with_valid_spans=80, n_total=100, score=0.80)
        assert "80/100" in m.label
        assert "80.0%" in m.label


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityReport:
    def test_grade_in_abcd(self, report):
        assert report.grade in {"A", "B", "C", "D"}

    def test_overall_in_01(self, report):
        assert 0.0 <= report.overall_score <= 1.0

    def test_summary_contains_grade(self, report):
        s = report.summary()
        assert report.grade in s

    def test_summary_contains_overall(self, report):
        s = report.summary()
        assert "overall" in s.lower() or str(round(report.overall_score, 2)) in s

    def test_summary_contains_all_metric_names(self, report):
        s = report.summary()
        for name in ["Correctness", "Diversity", "Level balance",
                     "Relation coverage", "Entity spans"]:
            assert name.lower() in s.lower(), f"Missing '{name}' in summary"

    def test_empty_dataset(self, analyzer):
        report = analyzer.analyze([])
        assert report.grade  == "D"
        assert report.overall_score == 0.0
        assert report.n_examples    == 0


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetQualityAnalyzer:
    def test_analyze_returns_report(self, small_dataset, analyzer):
        r = analyzer.analyze(small_dataset)
        assert isinstance(r, QualityReport)

    def test_n_examples_correct(self, small_dataset, analyzer):
        r = analyzer.analyze(small_dataset)
        assert r.n_examples == len(small_dataset)

    def test_custom_weights(self):
        """Pesos custom deben normalizarse y dar distinto overall_score."""
        gen  = CausalGraphGenerator(seed=0)
        exs  = gen.generate_batch(n=30)
        a1   = DatasetQualityAnalyzer()
        a2   = DatasetQualityAnalyzer(weights={"correctness": 1.0, "diversity": 0.0,
                                                "level_balance": 0.0,
                                                "relation_coverage": 0.0,
                                                "entity_spans": 0.0})
        r1 = a1.analyze(exs)
        r2 = a2.analyze(exs)
        # r2 solo mide correctness → overall = correctness.score
        assert abs(r2.overall_score - r2.correctness.score) < 1e-6

    def test_weights_normalized(self):
        a = DatasetQualityAnalyzer(weights={"correctness": 2.0, "diversity": 2.0,
                                             "level_balance": 2.0,
                                             "relation_coverage": 2.0,
                                             "entity_spans": 2.0})
        total = sum(a.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_level1_gives_low_balance(self):
        gen  = CausalGraphGenerator(seed=1)
        exs  = [gen.generate(level=1) for _ in range(40)]
        a    = DatasetQualityAnalyzer()
        r    = a.analyze(exs)
        assert r.level_balance.score < 0.6   # muy desbalanceado

    def test_high_quality_real_dataset(self):
        gen = CausalGraphGenerator(seed=42)
        exs = gen.generate_batch(n=100, verify=True)
        a   = DatasetQualityAnalyzer()
        r   = a.analyze(exs)
        assert r.grade in {"A", "B"}   # dataset sintetico debe ser de calidad alta


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMEND
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommend:
    def test_grade_a_says_excellent(self, analyzer):
        # Construir report con grade A
        perfect = QualityReport(
            n_examples        = 100,
            correctness       = CorrectnessMetric(100, 100, 1.0),
            diversity         = DiversityMetric(1.0, 1.0, 1.0, 1.0),
            level_balance     = LevelBalanceMetric({l: 0.2 for l in ALL_LEVELS}, 1.0),
            relation_coverage = RelationCoverageMetric(16, 16, missing=[]),
            entity_spans      = EntitySpansMetric(100, 100, 1.0),
            overall_score     = 0.95,
            grade             = "A",
        )
        rec = analyzer.recommend(perfect)
        assert "excelente" in rec.lower() or "listo" in rec.lower()

    def test_low_correctness_mentions_verify(self, analyzer):
        bad_correctness = QualityReport(
            n_examples        = 100,
            correctness       = CorrectnessMetric(100, 70, 0.70),
            diversity         = DiversityMetric(1.0, 1.0, 1.0, 1.0),
            level_balance     = LevelBalanceMetric({l: 0.2 for l in ALL_LEVELS}, 1.0),
            relation_coverage = RelationCoverageMetric(16, 16, missing=[]),
            entity_spans      = EntitySpansMetric(100, 100, 1.0),
            overall_score     = 0.75,
            grade             = "B",
        )
        rec = analyzer.recommend(bad_correctness)
        assert "CORRECTNESS" in rec or "verify" in rec.lower()

    def test_missing_relations_mentioned(self, analyzer):
        missing = ["equivalent", "part_of", "instance_of", "analogous_to"]
        low_coverage = QualityReport(
            n_examples        = 100,
            correctness       = CorrectnessMetric(100, 100, 1.0),
            diversity         = DiversityMetric(1.0, 1.0, 1.0, 1.0),
            level_balance     = LevelBalanceMetric({l: 0.2 for l in ALL_LEVELS}, 1.0),
            relation_coverage = RelationCoverageMetric(12, 16, missing=missing),
            entity_spans      = EntitySpansMetric(100, 100, 1.0),
            overall_score     = 0.77,
            grade             = "B",
        )
        rec = analyzer.recommend(low_coverage)
        assert "RELATIONS" in rec
        for rel in missing:
            assert rel in rec

    def test_low_spans_mentioned(self, analyzer):
        low_spans = QualityReport(
            n_examples        = 100,
            correctness       = CorrectnessMetric(100, 100, 1.0),
            diversity         = DiversityMetric(1.0, 1.0, 1.0, 1.0),
            level_balance     = LevelBalanceMetric({l: 0.2 for l in ALL_LEVELS}, 1.0),
            relation_coverage = RelationCoverageMetric(16, 16, missing=[]),
            entity_spans      = EntitySpansMetric(30, 100, 0.30),
            overall_score     = 0.78,
            grade             = "B",
        )
        rec = analyzer.recommend(low_spans)
        assert "ENTITY SPANS" in rec or "spans" in rec.lower()

    def test_returns_string(self, report, analyzer):
        rec = analyzer.recommend(report)
        assert isinstance(rec, str)
        assert len(rec) > 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_main_runs_without_error(self, capsys):
        main(["--n", "30", "--seed", "0"])
        out = capsys.readouterr().out
        assert "DATASET QUALITY REPORT" in out

    def test_main_verbose_runs(self, capsys):
        main(["--n", "30", "--seed", "1", "--verbose"])
        out = capsys.readouterr().out
        assert "DETALLE" in out

    def test_main_no_verify(self, capsys):
        main(["--n", "20", "--seed", "2", "--no-verify"])
        out = capsys.readouterr().out
        assert "DATASET QUALITY REPORT" in out

    def test_main_output_contains_grade(self, capsys):
        main(["--n", "50", "--seed", "99"])
        out = capsys.readouterr().out
        assert any(grade in out for grade in ["A", "B", "C", "D"])

    def test_main_output_contains_n(self, capsys):
        main(["--n", "40", "--seed", "3"])
        out = capsys.readouterr().out
        assert "40" in out


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — DATASET REAL
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_100_examples(self):
        gen      = CausalGraphGenerator(seed=7)
        examples = gen.generate_batch(n=100, verify=True)
        analyzer = DatasetQualityAnalyzer()
        report   = analyzer.analyze(examples)

        assert report.n_examples == 100
        assert report.grade in {"A", "B", "C", "D"}
        assert 0.0 <= report.overall_score <= 1.0
        # Dataset sintetico verificado debe ser alta calidad
        assert report.correctness.score == 1.0

    def test_all_metrics_have_valid_scores(self, report):
        for score in [
            report.correctness.score,
            report.diversity.score,
            report.level_balance.score,
            report.relation_coverage.score,
            report.entity_spans.score,
        ]:
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_grade_consistent_with_overall(self, report):
        g = _score_to_grade(report.overall_score)
        assert g == report.grade

    def test_recommend_returns_for_any_grade(self):
        gen  = CausalGraphGenerator(seed=0)
        exs  = gen.generate_batch(n=60)
        a    = DatasetQualityAnalyzer()
        r    = a.analyze(exs)
        rec  = a.recommend(r)
        assert isinstance(rec, str) and len(rec) > 0
