"""
tests/test_val.py — Tests para AionCValidator
==============================================

Cubre:
    - ValidatorConfig: validacion, defaults
    - ValidationIssue: campos y severidades
    - ValidationResult: passed, overall_score, issues
    - AionCValidator: shapes, 4 checks, gradientes, batch
    - Pipeline integration: use_validator=True en CORAPipeline,
      rereason_on_fail=True (val_rereason)
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

from validation.model import (
    AionCValidator,
    ValidatorConfig,
    ValidationIssue,
    ValidationResult,
    _geometric_mean,
    _build_issues,
)
from router.pipeline import CORAConfig, CORAPipeline, PipelineOutput


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

VOCAB   = 128
INPUT_D = 64
H       = 64     # val hidden_dim; must be divisible by n_heads
N_HEADS = 4


@pytest.fixture
def cfg() -> ValidatorConfig:
    return ValidatorConfig(
        input_dim      = INPUT_D,
        hidden_dim     = H,
        n_heads        = N_HEADS,
        n_layers       = 2,
        pass_threshold = 0.5,
        issue_threshold= 0.4,
    )


@pytest.fixture
def validator(cfg) -> AionCValidator:
    return AionCValidator(cfg, vocab_size=VOCAB)


@pytest.fixture
def pipeline_cfg_with_val() -> CORAConfig:
    cfg = CORAConfig.tiny()
    cfg.use_validator   = True
    cfg.val_hidden_dim  = 64
    cfg.val_n_heads     = 4
    cfg.val_n_layers    = 2
    cfg.val_rereason    = False
    return cfg


@pytest.fixture
def pipeline_cfg_with_rereason() -> CORAConfig:
    cfg = CORAConfig.tiny()
    cfg.use_validator        = True
    cfg.val_hidden_dim       = 64
    cfg.val_n_heads          = 4
    cfg.val_n_layers         = 1
    cfg.val_rereason         = True
    cfg.cre_max_iterations   = 4   # small so retry stays cheap
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class TestValidatorConfig:
    def test_defaults(self):
        cfg = ValidatorConfig()
        assert cfg.input_dim   == 256
        assert cfg.hidden_dim  == 128
        assert cfg.n_layers    == 2
        assert cfg.n_heads     == 4

    def test_invalid_n_heads_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            ValidatorConfig(hidden_dim=64, n_heads=3)

    def test_valid_config_no_raise(self):
        ValidatorConfig(hidden_dim=64, n_heads=4)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION ISSUE
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationIssue:
    def test_fields(self):
        issue = ValidationIssue(check="faithfulness", score=0.25, severity="error")
        assert issue.check    == "faithfulness"
        assert issue.score    == 0.25
        assert issue.severity == "error"

    def test_repr(self):
        issue = ValidationIssue(check="consistency", score=0.35, severity="warning")
        r = repr(issue)
        assert "consistency" in r
        assert "warning" in r

    def test_build_issues_empty_when_all_pass(self):
        scores = {"faithfulness": 0.8, "consistency": 0.9, "completeness": 0.7, "hallucination": 0.6}
        issues = _build_issues(scores, threshold=0.4)
        assert issues == []

    def test_build_issues_warning(self):
        scores = {"faithfulness": 0.35, "consistency": 0.9, "completeness": 0.9, "hallucination": 0.9}
        issues = _build_issues(scores, threshold=0.4)
        assert len(issues) == 1
        assert issues[0].check    == "faithfulness"
        assert issues[0].severity == "warning"

    def test_build_issues_error(self):
        scores = {"faithfulness": 0.1, "consistency": 0.9, "completeness": 0.9, "hallucination": 0.9}
        issues = _build_issues(scores, threshold=0.4)
        assert len(issues) == 1
        assert issues[0].severity == "error"

    def test_build_issues_multiple(self):
        scores = {"faithfulness": 0.1, "consistency": 0.2, "completeness": 0.8, "hallucination": 0.8}
        issues = _build_issues(scores, threshold=0.4)
        assert len(issues) == 2


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class TestGeometricMean:
    def test_uniform(self):
        assert abs(_geometric_mean([0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-6

    def test_single(self):
        assert abs(_geometric_mean([0.8]) - 0.8) < 1e-6

    def test_one_zero_drags_down(self):
        # geometric mean should be much lower than arithmetic when one is near 0
        geo = _geometric_mean([1.0, 1.0, 1.0, 1e-6])
        assert geo < 0.1

    def test_empty(self):
        assert _geometric_mean([]) == 0.0

    def test_penalizes_more_than_arithmetic(self):
        scores = [0.9, 0.9, 0.9, 0.1]
        geo   = _geometric_mean(scores)
        arith = sum(scores) / len(scores)
        assert geo < arith


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR — SHAPES
# ─────────────────────────────────────────────────────────────────────────────

class TestValidatorShapes:
    def test_result_is_validation_result(self, validator):
        logits = torch.randn(1, 8, VOCAB)
        graph  = torch.randn(1, 6, INPUT_D)
        result = validator(logits, graph)
        assert isinstance(result, ValidationResult)

    def test_scores_in_01(self, validator):
        logits = torch.randn(2, 10, VOCAB)
        graph  = torch.randn(2, 8, INPUT_D)
        r = validator(logits, graph)
        for score in [r.overall_score, r.faithfulness, r.consistency,
                      r.completeness, r.hallucination]:
            assert 0.0 <= score <= 1.0

    def test_passed_is_bool(self, validator):
        logits = torch.randn(1, 6, VOCAB)
        graph  = torch.randn(1, 4, INPUT_D)
        r = validator(logits, graph)
        assert isinstance(r.passed, bool)

    def test_issues_is_list(self, validator):
        logits = torch.randn(1, 5, VOCAB)
        graph  = torch.randn(1, 4, INPUT_D)
        r = validator(logits, graph)
        assert isinstance(r.issues, list)

    def test_with_input_concepts(self, validator):
        logits = torch.randn(1, 8, VOCAB)
        graph  = torch.randn(1, 6, INPUT_D)
        inp    = torch.randn(1, 8, INPUT_D)
        r = validator(logits, graph, inp)
        assert isinstance(r, ValidationResult)
        assert 0.0 <= r.overall_score <= 1.0

    def test_without_input_concepts(self, validator):
        """completeness usa graph_pool como proxy cuando no hay input_concepts."""
        logits = torch.randn(1, 8, VOCAB)
        graph  = torch.randn(1, 6, INPUT_D)
        r = validator(logits, graph, input_concepts=None)
        assert isinstance(r, ValidationResult)

    def test_single_node_graph(self, validator):
        logits = torch.randn(1, 4, VOCAB)
        graph  = torch.randn(1, 1, INPUT_D)
        r = validator(logits, graph)
        assert 0.0 <= r.overall_score <= 1.0

    def test_batch_gt_1(self, validator):
        logits = torch.randn(3, 8, VOCAB)
        graph  = torch.randn(3, 6, INPUT_D)
        r = validator(logits, graph)
        assert isinstance(r, ValidationResult)
        assert 0.0 <= r.overall_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR — 4 CHECKS
# ─────────────────────────────────────────────────────────────────────────────

class TestValidatorChecks:
    def test_four_check_scores_present(self, validator):
        logits = torch.randn(1, 8, VOCAB)
        graph  = torch.randn(1, 6, INPUT_D)
        r = validator(logits, graph)
        assert hasattr(r, "faithfulness")
        assert hasattr(r, "consistency")
        assert hasattr(r, "completeness")
        assert hasattr(r, "hallucination")

    def test_all_checks_in_01(self, validator):
        logits = torch.randn(1, 10, VOCAB)
        graph  = torch.randn(1, 8, INPUT_D)
        r = validator(logits, graph)
        for name in ["faithfulness", "consistency", "completeness", "hallucination"]:
            s = getattr(r, name)
            assert 0.0 <= s <= 1.0, f"{name} = {s}"

    def test_overall_is_geometric_mean(self, validator):
        logits = torch.randn(1, 8, VOCAB)
        graph  = torch.randn(1, 6, INPUT_D)
        r = validator(logits, graph)
        expected = _geometric_mean([r.faithfulness, r.consistency,
                                    r.completeness, r.hallucination])
        assert abs(r.overall_score - expected) < 1e-5

    def test_passed_consistent_with_threshold(self, cfg, validator):
        logits = torch.randn(1, 8, VOCAB)
        graph  = torch.randn(1, 6, INPUT_D)
        r = validator(logits, graph)
        if r.overall_score >= cfg.pass_threshold:
            assert r.passed is True
        else:
            assert r.passed is False

    def test_issues_match_threshold(self, cfg, validator):
        logits = torch.randn(1, 8, VOCAB)
        graph  = torch.randn(1, 6, INPUT_D)
        r = validator(logits, graph)
        for issue in r.issues:
            assert issue.score < cfg.issue_threshold
            assert issue.check in {"faithfulness", "consistency", "completeness", "hallucination"}
            assert issue.severity in {"warning", "error"}

    def test_repr_contains_status(self, validator):
        logits = torch.randn(1, 8, VOCAB)
        graph  = torch.randn(1, 6, INPUT_D)
        r = validator(logits, graph)
        s = repr(r)
        assert "PASS" in s or "FAIL" in s


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR — GRADIENTES
# ─────────────────────────────────────────────────────────────────────────────

class TestValidatorGradients:
    def test_gradients_flow_from_logits(self, validator):
        logits = torch.randn(1, 8, VOCAB, requires_grad=True)
        graph  = torch.randn(1, 6, INPUT_D)
        r = validator(logits, graph)
        # overall_score es float Python, debemos usar el tensor intermedio
        # Recomputamos con un proxy diferenciable
        logits2 = torch.randn(1, 8, VOCAB, requires_grad=True)
        r2 = validator(logits2, graph)
        # Reconstruir loss a partir de scores internos del modelo
        # Hacemos backward a través de la salida de las cabezas
        val2 = AionCValidator(validator.config, VOCAB)
        val2.load_state_dict(validator.state_dict())
        logits3 = torch.randn(1, 8, VOCAB, requires_grad=True)
        graph3  = torch.randn(1, 6, INPUT_D)
        # El validator devuelve floats Python; testeamos que los params tienen grad
        loss = _differentiable_proxy(val2, logits3, graph3)
        loss.backward()
        params_with_grad = [p for p in val2.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0

    def test_count_parameters_nonzero(self, validator):
        assert validator.count_parameters() > 0


def _differentiable_proxy(
    val: AionCValidator,
    logits: torch.Tensor,
    graph: torch.Tensor,
) -> torch.Tensor:
    """
    Ejecuta el validator pero retorna un tensor diferenciable para backward.
    (ValidationResult usa floats Python, pero los tensores internos tienen grad_fn.)
    """
    probs       = torch.softmax(logits, dim=-1)
    response    = val.response_proj(probs)
    for block in val.response_encoder:
        response = block(response)
    response    = val.response_norm(response)
    resp_pool   = response.mean(dim=1)
    graph_enc   = val.graph_norm(val.graph_proj(graph))
    graph_pool  = graph_enc.mean(dim=1)
    h           = val.cross_attn_norm(resp_pool.unsqueeze(1))
    cross_out, _= val.cross_attn(h, graph_enc, graph_enc)
    cross_out   = cross_out.squeeze(1)
    faith = torch.sigmoid(val.head_faithfulness.net(cross_out))
    return faith.sum()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineWithValidator:
    def test_instantiation_with_validator(self, pipeline_cfg_with_val):
        pipe = CORAPipeline(pipeline_cfg_with_val)
        assert pipe.validator is not None

    def test_instantiation_without_validator(self):
        cfg  = CORAConfig.tiny()
        pipe = CORAPipeline(cfg)
        assert pipe.validator is None

    def test_forward_returns_validation(self, pipeline_cfg_with_val):
        pipe      = CORAPipeline(pipeline_cfg_with_val)
        token_ids = torch.randint(0, pipeline_cfg_with_val.vocab_size, (1, 10))
        out       = pipe(token_ids)
        assert out.validation is not None
        assert isinstance(out.validation, ValidationResult)

    def test_forward_no_validator_returns_none(self):
        cfg       = CORAConfig.tiny()
        pipe      = CORAPipeline(cfg)
        token_ids = torch.randint(0, cfg.vocab_size, (1, 10))
        out       = pipe(token_ids)
        assert out.validation is None

    def test_validation_scores_in_01(self, pipeline_cfg_with_val):
        pipe      = CORAPipeline(pipeline_cfg_with_val)
        token_ids = torch.randint(0, pipeline_cfg_with_val.vocab_size, (1, 8))
        out       = pipe(token_ids)
        r = out.validation
        for score in [r.overall_score, r.faithfulness, r.consistency,
                      r.completeness, r.hallucination]:
            assert 0.0 <= score <= 1.0

    def test_logits_shape_unchanged(self, pipeline_cfg_with_val):
        pipe      = CORAPipeline(pipeline_cfg_with_val)
        B, L      = 1, 10
        token_ids = torch.randint(0, pipeline_cfg_with_val.vocab_size, (B, L))
        out       = pipe(token_ids)
        assert out.logits.shape == (B, L, pipeline_cfg_with_val.vocab_size)

    def test_gradients_flow_with_validator(self, pipeline_cfg_with_val):
        pipe      = CORAPipeline(pipeline_cfg_with_val)
        token_ids = torch.randint(0, pipeline_cfg_with_val.vocab_size, (1, 8))
        out       = pipe(token_ids)
        out.logits.sum().backward()
        enc_grad = pipe.encoder.token_embedding.weight.grad
        assert enc_grad is not None

    def test_validator_adds_parameters(self, pipeline_cfg_with_val):
        cfg_no  = CORAConfig.tiny()
        cfg_yes = pipeline_cfg_with_val
        p_no    = CORAPipeline(cfg_no).count_parameters()
        p_yes   = CORAPipeline(cfg_yes).count_parameters()
        assert p_yes > p_no

    def test_parameter_breakdown_includes_validator(self, pipeline_cfg_with_val):
        pipe = CORAPipeline(pipeline_cfg_with_val)
        bd   = pipe.parameter_breakdown()
        assert "validator" in bd
        assert bd["validator"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# REREASON ON FAIL
# ─────────────────────────────────────────────────────────────────────────────

class TestRereasonOnFail:
    def test_rereason_flag_instantiation(self, pipeline_cfg_with_rereason):
        pipe = CORAPipeline(pipeline_cfg_with_rereason)
        assert pipe.validator is not None
        assert pipe.config.val_rereason is True

    def test_rereason_forward_completes(self, pipeline_cfg_with_rereason):
        """Con val_rereason=True, el forward debe completar sin errores."""
        pipe      = CORAPipeline(pipeline_cfg_with_rereason)
        token_ids = torch.randint(0, pipeline_cfg_with_rereason.vocab_size, (1, 8))
        out       = pipe(token_ids)
        assert out.validation is not None
        assert out.logits.shape[0] == 1

    def test_rereason_logits_valid(self, pipeline_cfg_with_rereason):
        pipe      = CORAPipeline(pipeline_cfg_with_rereason)
        token_ids = torch.randint(0, pipeline_cfg_with_rereason.vocab_size, (1, 10))
        out       = pipe(token_ids)
        assert not torch.isnan(out.logits).any()
        assert not torch.isinf(out.logits).any()

    def test_rereason_validation_scores_valid(self, pipeline_cfg_with_rereason):
        pipe      = CORAPipeline(pipeline_cfg_with_rereason)
        token_ids = torch.randint(0, pipeline_cfg_with_rereason.vocab_size, (1, 8))
        out       = pipe(token_ids)
        r = out.validation
        assert 0.0 <= r.overall_score <= 1.0
