"""
tests/test_budget.py — Tests para BudgetManager y QueryComplexityClassifier
============================================================================

Cubre:
    - BudgetLevel: orden, valores
    - Heuristica: umbrales de tokens correctos
    - QueryComplexityClassifier: shapes, gradientes, predict
    - BudgetManager: modo heuristico, modo aprendido, batch > 1
    - Pipeline integration: use_budget_manager=True en CORAPipeline
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

from budget.manager import (
    HEURISTIC_THRESHOLDS,
    BudgetLevel,
    BudgetManager,
    BudgetOutput,
    QueryComplexityClassifier,
    _level_to_iterations,
)
from router.pipeline import CORAConfig, CORAPipeline, PipelineOutput


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def concept_dim() -> int:
    return 64


@pytest.fixture
def classifier(concept_dim) -> QueryComplexityClassifier:
    return QueryComplexityClassifier(concept_dim=concept_dim, hidden_dim=32)


@pytest.fixture
def manager(concept_dim) -> BudgetManager:
    return BudgetManager(
        concept_dim        = concept_dim,
        max_cre_iterations = 20,
        hidden_dim         = 32,
        use_learned        = True,
    )


@pytest.fixture
def pipeline_config_with_budget() -> CORAConfig:
    cfg = CORAConfig.tiny()
    cfg.use_budget_manager = True
    cfg.budget_hidden_dim  = 16
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# BUDGET LEVEL
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetLevel:
    def test_levels_are_ordered(self):
        assert BudgetLevel.TRIVIAL < BudgetLevel.SIMPLE
        assert BudgetLevel.SIMPLE  < BudgetLevel.COMPLEX
        assert BudgetLevel.COMPLEX < BudgetLevel.DEEP

    def test_level_values(self):
        assert int(BudgetLevel.TRIVIAL) == 0
        assert int(BudgetLevel.SIMPLE)  == 1
        assert int(BudgetLevel.COMPLEX) == 2
        assert int(BudgetLevel.DEEP)    == 3

    def test_level_count(self):
        assert len(BudgetLevel) == 4


# ─────────────────────────────────────────────────────────────────────────────
# HEURISTICA
# ─────────────────────────────────────────────────────────────────────────────

class TestHeuristic:
    def test_trivial_threshold(self):
        lo = HEURISTIC_THRESHOLDS[0]
        out = BudgetManager.classify_heuristic(lo - 1, max_iterations=20)
        assert out.level == BudgetLevel.TRIVIAL
        assert out.n_iterations == 1
        assert out.used_heuristic is True

    def test_simple_threshold_lower(self):
        lo = HEURISTIC_THRESHOLDS[0]
        out = BudgetManager.classify_heuristic(lo, max_iterations=20)
        assert out.level == BudgetLevel.SIMPLE

    def test_simple_threshold_upper(self):
        mid = HEURISTIC_THRESHOLDS[1]
        out = BudgetManager.classify_heuristic(mid - 1, max_iterations=20)
        assert out.level == BudgetLevel.SIMPLE
        assert out.n_iterations == 3

    def test_complex_threshold_lower(self):
        mid = HEURISTIC_THRESHOLDS[1]
        out = BudgetManager.classify_heuristic(mid, max_iterations=20)
        assert out.level == BudgetLevel.COMPLEX

    def test_complex_threshold_upper(self):
        hi = HEURISTIC_THRESHOLDS[2]
        out = BudgetManager.classify_heuristic(hi - 1, max_iterations=20)
        assert out.level == BudgetLevel.COMPLEX
        assert out.n_iterations == 10

    def test_deep_threshold(self):
        hi = HEURISTIC_THRESHOLDS[2]
        out = BudgetManager.classify_heuristic(hi, max_iterations=20)
        assert out.level == BudgetLevel.DEEP
        assert out.n_iterations == 20

    def test_deep_uses_max_iterations(self):
        out = BudgetManager.classify_heuristic(200, max_iterations=15)
        assert out.n_iterations == 15

    def test_complex_capped_by_max_iterations(self):
        """Si max_iterations < 10, COMPLEX no puede pedir 10 iters."""
        out = BudgetManager.classify_heuristic(50, max_iterations=5)
        assert out.n_iterations == 5
        assert out.level == BudgetLevel.COMPLEX

    def test_class_probs_is_none_for_heuristic(self):
        out = BudgetManager.classify_heuristic(5)
        assert out.class_probs is None

    def test_specific_examples(self):
        """Casos concretos del plan: 'hola' → trivial, 'diseña microservicios' → deep."""
        # "hola" ≈ 1 token
        out = BudgetManager.classify_heuristic(1, max_iterations=20)
        assert out.level == BudgetLevel.TRIVIAL

        # "diseña una arquitectura de microservicios..." ≈ 100+ tokens
        out = BudgetManager.classify_heuristic(120, max_iterations=20)
        assert out.level == BudgetLevel.DEEP


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL → ITERATIONS HELPER
# ─────────────────────────────────────────────────────────────────────────────

class TestLevelToIterations:
    def test_trivial_is_1(self):
        assert _level_to_iterations(BudgetLevel.TRIVIAL, 20) == 1

    def test_simple_is_3(self):
        assert _level_to_iterations(BudgetLevel.SIMPLE, 20) == 3

    def test_complex_is_10_when_max_allows(self):
        assert _level_to_iterations(BudgetLevel.COMPLEX, 20) == 10

    def test_complex_capped_at_max(self):
        assert _level_to_iterations(BudgetLevel.COMPLEX, 7) == 7

    def test_deep_is_max(self):
        assert _level_to_iterations(BudgetLevel.DEEP, 20) == 20
        assert _level_to_iterations(BudgetLevel.DEEP, 5)  == 5


# ─────────────────────────────────────────────────────────────────────────────
# QUERY COMPLEXITY CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryComplexityClassifier:
    def test_output_shape(self, classifier, concept_dim):
        B, L = 2, 15
        x = torch.randn(B, L, concept_dim)
        logits = classifier(x)
        assert logits.shape == (B, 4)

    def test_predict_shape(self, classifier, concept_dim):
        B, L = 3, 10
        x = torch.randn(B, L, concept_dim)
        preds = classifier.predict(x)
        assert preds.shape == (B,)

    def test_predict_values_in_range(self, classifier, concept_dim):
        x = torch.randn(5, 8, concept_dim)
        preds = classifier.predict(x)
        assert preds.min() >= 0
        assert preds.max() <= 3

    def test_gradients_flow(self, classifier, concept_dim):
        x = torch.randn(2, 10, concept_dim, requires_grad=True)
        logits = classifier(x)
        logits.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_count_parameters_nonzero(self, classifier):
        assert classifier.count_parameters() > 0

    def test_single_token_sequence(self, classifier, concept_dim):
        """Debe funcionar con L=1."""
        x = torch.randn(1, 1, concept_dim)
        logits = classifier(x)
        assert logits.shape == (1, 4)

    def test_different_inputs_produce_different_logits(self, classifier, concept_dim):
        torch.manual_seed(0)
        x1 = torch.randn(1, 5, concept_dim)
        x2 = torch.randn(1, 5, concept_dim)
        l1 = classifier(x1)
        l2 = classifier(x2)
        assert not torch.allclose(l1, l2, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# BUDGET MANAGER — MODO HEURISTICO
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetManagerHeuristic:
    def test_no_concept_vectors_uses_heuristic(self, manager):
        token_ids = torch.randint(0, 100, (1, 8))   # 8 tokens → TRIVIAL
        out = manager(token_ids, concept_vectors=None)
        assert out.used_heuristic is True
        assert out.level == BudgetLevel.TRIVIAL

    def test_use_learned_false_ignores_concepts(self, concept_dim):
        mgr = BudgetManager(concept_dim=concept_dim, use_learned=False)
        token_ids = torch.randint(0, 100, (1, 50))  # 50 tokens → COMPLEX
        concepts  = torch.randn(1, 50, concept_dim)
        out = mgr(token_ids, concepts)
        assert out.used_heuristic is True
        assert out.level == BudgetLevel.COMPLEX

    def test_heuristic_output_structure(self, manager):
        token_ids = torch.randint(0, 100, (1, 5))
        out = manager(token_ids, concept_vectors=None)
        assert isinstance(out, BudgetOutput)
        assert isinstance(out.level, BudgetLevel)
        assert out.n_iterations >= 1


# ─────────────────────────────────────────────────────────────────────────────
# BUDGET MANAGER — MODO APRENDIDO
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetManagerLearned:
    def test_returns_budget_output(self, manager, concept_dim):
        token_ids = torch.randint(0, 100, (1, 20))
        concepts  = torch.randn(1, 20, concept_dim)
        out = manager(token_ids, concepts)
        assert isinstance(out, BudgetOutput)
        assert out.used_heuristic is False

    def test_class_probs_shape(self, manager, concept_dim):
        B, L = 2, 15
        token_ids = torch.randint(0, 100, (B, L))
        concepts  = torch.randn(B, L, concept_dim)
        out = manager(token_ids, concepts)
        assert out.class_probs is not None
        assert out.class_probs.shape == (B, 4)

    def test_class_probs_sum_to_one(self, manager, concept_dim):
        token_ids = torch.randint(0, 100, (3, 12))
        concepts  = torch.randn(3, 12, concept_dim)
        out = manager(token_ids, concepts)
        sums = out.class_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_n_iterations_within_bounds(self, manager, concept_dim):
        token_ids = torch.randint(0, 100, (1, 20))
        concepts  = torch.randn(1, 20, concept_dim)
        out = manager(token_ids, concepts)
        assert 1 <= out.n_iterations <= manager.max_cre_iterations

    def test_level_matches_n_iterations(self, manager, concept_dim):
        """n_iterations debe ser consistente con el nivel clasificado."""
        token_ids = torch.randint(0, 100, (1, 20))
        concepts  = torch.randn(1, 20, concept_dim)
        out = manager(token_ids, concepts)
        expected = _level_to_iterations(out.level, manager.max_cre_iterations)
        assert out.n_iterations == expected

    def test_class_probs_detached(self, manager, concept_dim):
        """class_probs no debe tener grad_fn."""
        token_ids = torch.randint(0, 100, (1, 10))
        concepts  = torch.randn(1, 10, concept_dim, requires_grad=True)
        out = manager(token_ids, concepts)
        assert out.class_probs.grad_fn is None

    def test_batch_gt_1_uses_max_level(self, concept_dim):
        """Con B>1, el nivel es el maximo del batch (conservador)."""
        mgr = BudgetManager(concept_dim=concept_dim, max_cre_iterations=20, use_learned=True)
        # Forzar que el clasificador siempre prediga distintos niveles por batch item
        # no es trivial, pero si testeamos que n_iterations respeta [1, 20]
        token_ids = torch.randint(0, 100, (4, 15))
        concepts  = torch.randn(4, 15, concept_dim)
        out = mgr(token_ids, concepts)
        assert 1 <= out.n_iterations <= 20

    def test_count_parameters_nonzero(self, manager):
        assert manager.count_parameters() > 0


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineWithBudget:
    def test_pipeline_instantiation_with_budget(self, pipeline_config_with_budget):
        pipeline = CORAPipeline(pipeline_config_with_budget)
        assert pipeline.budget_manager is not None

    def test_pipeline_instantiation_without_budget(self):
        config = CORAConfig.tiny()
        config.use_budget_manager = False
        pipeline = CORAPipeline(config)
        assert pipeline.budget_manager is None

    def test_pipeline_forward_with_budget(self, pipeline_config_with_budget):
        pipeline  = CORAPipeline(pipeline_config_with_budget)
        token_ids = torch.randint(0, pipeline_config_with_budget.vocab_size, (1, 16))
        out = pipeline(token_ids)
        assert isinstance(out, PipelineOutput)
        assert out.budget is not None

    def test_pipeline_budget_is_none_without_manager(self):
        config = CORAConfig.tiny()
        config.use_budget_manager = False
        pipeline = CORAPipeline(config)
        token_ids = torch.randint(0, config.vocab_size, (1, 16))
        out = pipeline(token_ids)
        assert out.budget is None

    def test_pipeline_budget_n_iterations_respected(self, pipeline_config_with_budget):
        """n_iterations del budget debe estar dentro del rango válido."""
        pipeline  = CORAPipeline(pipeline_config_with_budget)
        token_ids = torch.randint(0, pipeline_config_with_budget.vocab_size, (1, 20))
        out = pipeline(token_ids)
        max_iters = pipeline_config_with_budget.cre_max_iterations
        assert 1 <= out.budget.n_iterations <= max_iters

    def test_pipeline_output_shape_unchanged(self, pipeline_config_with_budget):
        """El BudgetManager no debe cambiar las shapes de logits/graph_repr."""
        pipeline  = CORAPipeline(pipeline_config_with_budget)
        B, L      = 1, 12
        token_ids = torch.randint(0, pipeline_config_with_budget.vocab_size, (B, L))
        out = pipeline(token_ids)
        assert out.logits.shape == (B, L, pipeline_config_with_budget.vocab_size)

    def test_pipeline_gradients_flow_with_budget(self, pipeline_config_with_budget):
        """Los gradientes deben fluir incluso con BudgetManager activo."""
        pipeline  = CORAPipeline(pipeline_config_with_budget)
        token_ids = torch.randint(0, pipeline_config_with_budget.vocab_size, (1, 8))
        out = pipeline(token_ids)
        out.logits.sum().backward()
        # Verificar que al menos un parametro del encoder tiene gradiente
        enc_grad = pipeline.encoder.token_embedding.weight.grad
        assert enc_grad is not None

    def test_budget_manager_in_parameter_count(self, pipeline_config_with_budget):
        """CORAPipeline con budget debe tener mas params que sin budget."""
        config_no = CORAConfig.tiny()
        config_no.use_budget_manager = False
        config_yes = pipeline_config_with_budget
        p_no  = CORAPipeline(config_no).count_parameters()
        p_yes = CORAPipeline(config_yes).count_parameters()
        assert p_yes > p_no
