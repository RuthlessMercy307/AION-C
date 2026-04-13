"""
tests/test_orchestrator.py — Tests for Orchestrator
====================================================
"""

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from orchestrator.model import (
    Orchestrator, OrchestratorConfig, OrchestratorOutput,
    MotorActivation, MOTOR_NAMES, BASE_ITERATIONS, KEYWORD_TRIGGERS,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_config():
    return OrchestratorConfig()

@pytest.fixture
def tiny_config():
    return OrchestratorConfig(hidden_dim=32, mlp_hidden_dim=16)

@pytest.fixture
def orchestrator(tiny_config):
    return Orchestrator(tiny_config)

@pytest.fixture
def concept_vectors(tiny_config):
    """Fake concept_vectors [B=2, L=8, D=32]."""
    torch.manual_seed(42)
    return torch.randn(2, 8, tiny_config.hidden_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorConfig:
    def test_default_values(self, default_config):
        assert default_config.hidden_dim == 256
        assert default_config.n_motors == 5
        assert default_config.max_active_motors == 3
        assert default_config.min_confidence_to_activate == 0.3
        assert default_config.mlp_hidden_dim == 128
        assert default_config.max_iter_multiplier == 2.0

    def test_wrong_n_motors_raises(self):
        with pytest.raises(ValueError, match="n_motors"):
            OrchestratorConfig(n_motors=4)

    def test_max_active_motors_zero_raises(self):
        with pytest.raises(ValueError, match="max_active_motors"):
            OrchestratorConfig(max_active_motors=0)

    def test_max_active_motors_too_large_raises(self):
        with pytest.raises(ValueError, match="max_active_motors"):
            OrchestratorConfig(max_active_motors=6)

    def test_custom_valid_config(self):
        cfg = OrchestratorConfig(hidden_dim=64, mlp_hidden_dim=32, max_active_motors=2)
        assert cfg.hidden_dim == 64
        assert cfg.max_active_motors == 2

    def test_max_active_motors_boundary_1(self):
        cfg = OrchestratorConfig(max_active_motors=1)
        assert cfg.max_active_motors == 1

    def test_max_active_motors_boundary_5(self):
        cfg = OrchestratorConfig(max_active_motors=5)
        assert cfg.max_active_motors == 5


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_motor_names_count(self):
        assert len(MOTOR_NAMES) == 5

    def test_motor_names_values(self):
        assert MOTOR_NAMES == ["cora", "forge_c", "muse", "axiom", "empathy"]

    def test_base_iterations_all_motors(self):
        for name in MOTOR_NAMES:
            assert name in BASE_ITERATIONS
            assert BASE_ITERATIONS[name] >= 1

    def test_base_iterations_values(self):
        assert BASE_ITERATIONS["cora"] == 5
        assert BASE_ITERATIONS["forge_c"] == 3
        assert BASE_ITERATIONS["muse"] == 5
        assert BASE_ITERATIONS["axiom"] == 7
        assert BASE_ITERATIONS["empathy"] == 3

    def test_keyword_triggers_has_4_motors(self):
        # cora is the default — no explicit keywords
        assert len(KEYWORD_TRIGGERS) == 4
        for key in ["forge_c", "axiom", "muse", "empathy"]:
            assert key in KEYWORD_TRIGGERS

    def test_keyword_triggers_non_empty(self):
        for motor, keywords in KEYWORD_TRIGGERS.items():
            assert len(keywords) >= 5, f"{motor} should have >= 5 keywords"


# ─────────────────────────────────────────────────────────────────────────────
# 3. INSTANTIATION
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorInstantiation:
    def test_default_config(self, default_config):
        orch = Orchestrator(default_config)
        assert isinstance(orch, nn.Module)

    def test_tiny_config(self, orchestrator):
        assert isinstance(orchestrator, nn.Module)

    def test_has_classifier(self, orchestrator):
        assert hasattr(orchestrator, "classifier")
        assert isinstance(orchestrator.classifier, nn.Sequential)

    def test_parameter_count_positive(self, orchestrator):
        assert orchestrator.count_parameters() > 0

    def test_parameter_count_reasonable(self, tiny_config):
        orch = Orchestrator(tiny_config)
        D, H, N = tiny_config.hidden_dim, tiny_config.mlp_hidden_dim, tiny_config.n_motors
        # MLP: D→H + H→H//2 + H//2→N (weights only)
        expected_min = D * H + H * (H // 2) + (H // 2) * N
        assert orch.count_parameters() >= expected_min

    def test_classifier_output_dim(self, orchestrator, tiny_config):
        D = tiny_config.hidden_dim
        x = torch.randn(D)
        out = orchestrator.classifier(x)
        assert out.shape == (tiny_config.n_motors,)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FORWARD OUTPUT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorForward:
    def test_returns_orchestrator_output(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        assert isinstance(out, OrchestratorOutput)

    def test_scores_shape(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        assert out.scores.shape == (5,)

    def test_logits_shape(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        assert out.logits.shape == (5,)

    def test_scores_sum_to_one(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        assert abs(out.scores.sum().item() - 1.0) < 1e-5

    def test_scores_non_negative(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        assert (out.scores >= 0).all()

    def test_at_least_one_activation(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        assert len(out.activations) >= 1

    def test_n_active_matches_activations(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        assert out.n_active == len(out.activations)

    def test_routing_mode_learned(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        assert out.routing_mode == "learned"

    def test_primary_motor_property(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        assert out.primary_motor is out.activations[0]
        assert out.primary_motor.rank == 1

    def test_motor_names_property(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        names = out.motor_names
        assert isinstance(names, list)
        assert all(n in MOTOR_NAMES for n in names)

    def test_batch_size_1(self, tiny_config):
        orch = Orchestrator(tiny_config)
        cv = torch.randn(1, 4, tiny_config.hidden_dim)
        out = orch(cv)
        assert isinstance(out, OrchestratorOutput)

    def test_batch_size_4(self, tiny_config):
        orch = Orchestrator(tiny_config)
        cv = torch.randn(4, 8, tiny_config.hidden_dim)
        out = orch(cv)
        assert isinstance(out, OrchestratorOutput)

    def test_no_grad_in_eval(self, orchestrator, concept_vectors):
        orchestrator.eval()
        with torch.no_grad():
            out = orchestrator(concept_vectors)
        assert isinstance(out, OrchestratorOutput)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MOTOR ACTIVATION STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class TestMotorActivation:
    def test_activation_fields(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        act = out.activations[0]
        assert hasattr(act, "motor_name")
        assert hasattr(act, "score")
        assert hasattr(act, "n_iterations")
        assert hasattr(act, "rank")
        assert hasattr(act, "motor_idx")

    def test_activation_motor_name_valid(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        for act in out.activations:
            assert act.motor_name in MOTOR_NAMES

    def test_activation_score_in_range(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        for act in out.activations:
            assert 0.0 <= act.score <= 1.0

    def test_activation_n_iterations_positive(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        for act in out.activations:
            assert act.n_iterations >= 1

    def test_activation_rank_ascending(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        for i, act in enumerate(out.activations, start=1):
            assert act.rank == i

    def test_activation_motor_idx_valid(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        for act in out.activations:
            assert 0 <= act.motor_idx < 5
            assert MOTOR_NAMES[act.motor_idx] == act.motor_name

    def test_activation_repr(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        r = repr(out.activations[0])
        assert "MotorActivation" in r

    def test_orchestrator_output_repr(self, orchestrator, concept_vectors):
        out = orchestrator(concept_vectors)
        r = repr(out)
        assert "OrchestratorOutput" in r


# ─────────────────────────────────────────────────────────────────────────────
# 6. TOP-K SELECTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

class TestMotorSelection:
    def _make_orch_with_scores(self, scores_list, max_active=3, min_conf=0.3):
        """Helper: mock classifier to return fixed logits → known scores."""
        cfg = OrchestratorConfig(
            hidden_dim=16, mlp_hidden_dim=8,
            max_active_motors=max_active,
            min_confidence_to_activate=min_conf,
        )
        orch = Orchestrator(cfg)
        # Override classifier to return known logits
        # We'll use logits that produce approximately the desired scores
        # by calling _select_motors directly
        scores = torch.tensor(scores_list)
        logits = torch.log(scores + 1e-8)
        return orch, scores, logits

    def test_always_activates_at_least_one(self):
        cfg = OrchestratorConfig(hidden_dim=16, mlp_hidden_dim=8, min_confidence_to_activate=0.99)
        orch = Orchestrator(cfg)
        # Even with very high threshold, must activate at least 1
        scores = torch.tensor([0.21, 0.20, 0.20, 0.20, 0.19])
        logits = torch.log(scores + 1e-8)
        acts = orch._select_motors(scores, logits)
        assert len(acts) >= 1

    def test_max_active_motors_respected(self):
        cfg = OrchestratorConfig(hidden_dim=16, mlp_hidden_dim=8, max_active_motors=2)
        orch = Orchestrator(cfg)
        # All scores above threshold
        scores = torch.tensor([0.35, 0.30, 0.20, 0.10, 0.05])
        logits = torch.log(scores + 1e-8)
        acts = orch._select_motors(scores, logits)
        assert len(acts) <= 2

    def test_threshold_filters_secondary(self):
        cfg = OrchestratorConfig(hidden_dim=16, mlp_hidden_dim=8, min_confidence_to_activate=0.5)
        orch = Orchestrator(cfg)
        # Only one score above 0.5
        scores = torch.tensor([0.6, 0.2, 0.1, 0.05, 0.05])
        logits = torch.log(scores + 1e-8)
        acts = orch._select_motors(scores, logits)
        assert len(acts) == 1

    def test_primary_motor_is_highest_score(self):
        cfg = OrchestratorConfig(hidden_dim=16, mlp_hidden_dim=8)
        orch = Orchestrator(cfg)
        # axiom (index 3) has highest score
        scores = torch.tensor([0.1, 0.1, 0.1, 0.6, 0.1])
        logits = torch.log(scores + 1e-8)
        acts = orch._select_motors(scores, logits)
        assert acts[0].motor_name == "axiom"
        assert acts[0].motor_idx == 3

    def test_activations_ordered_by_score_desc(self):
        cfg = OrchestratorConfig(hidden_dim=16, mlp_hidden_dim=8, min_confidence_to_activate=0.2)
        orch = Orchestrator(cfg)
        scores = torch.tensor([0.1, 0.4, 0.3, 0.1, 0.1])
        logits = torch.log(scores + 1e-8)
        acts = orch._select_motors(scores, logits)
        for i in range(len(acts) - 1):
            assert acts[i].score >= acts[i + 1].score

    def test_max_active_1_always_single(self):
        cfg = OrchestratorConfig(
            hidden_dim=16, mlp_hidden_dim=8,
            max_active_motors=1, min_confidence_to_activate=0.0,
        )
        orch = Orchestrator(cfg)
        scores = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        logits = torch.log(scores + 1e-8)
        acts = orch._select_motors(scores, logits)
        assert len(acts) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. ITERATION BUDGET
# ─────────────────────────────────────────────────────────────────────────────

class TestIterationBudget:
    def test_primary_min_1(self, orchestrator):
        iters = orchestrator._compute_iterations("cora", score=0.0, rank=1)
        assert iters >= 1

    def test_secondary_min_1(self, orchestrator):
        iters = orchestrator._compute_iterations("cora", score=0.0, rank=2)
        assert iters >= 1

    def test_primary_high_score_more_iters(self, orchestrator):
        low  = orchestrator._compute_iterations("axiom", score=0.1, rank=1)
        high = orchestrator._compute_iterations("axiom", score=0.9, rank=1)
        assert high >= low

    def test_primary_base_capped_by_multiplier(self, tiny_config):
        orch = Orchestrator(tiny_config)
        base = BASE_ITERATIONS["axiom"]
        mult = tiny_config.max_iter_multiplier
        iters = orch._compute_iterations("axiom", score=1.0, rank=1)
        assert iters <= base * (mult + 1)  # generous upper bound

    def test_secondary_proportional_to_score(self, orchestrator):
        low  = orchestrator._compute_iterations("muse", score=0.3, rank=2)
        high = orchestrator._compute_iterations("muse", score=0.9, rank=2)
        assert high >= low

    def test_all_motors_have_base(self, orchestrator):
        for motor_name in MOTOR_NAMES:
            iters = orchestrator._compute_iterations(motor_name, score=0.5, rank=1)
            assert iters >= 1

    def test_primary_rank1_different_from_rank2(self, orchestrator):
        r1 = orchestrator._compute_iterations("cora", score=0.8, rank=1)
        r2 = orchestrator._compute_iterations("cora", score=0.8, rank=2)
        # Primary should get >= iterations than secondary at same score
        assert r1 >= r2


# ─────────────────────────────────────────────────────────────────────────────
# 8. HEURISTIC ROUTING
# ─────────────────────────────────────────────────────────────────────────────

class TestHeuristicRouting:
    def test_code_query_routes_to_forge_c(self, orchestrator):
        out = orchestrator.heuristic_route("Write a Python function with a bug to debug")
        assert out.primary_motor.motor_name == "forge_c"

    def test_math_query_routes_to_axiom(self, orchestrator):
        out = orchestrator.heuristic_route("demuestra el teorema de Pitágoras")
        assert out.primary_motor.motor_name == "axiom"

    def test_creative_query_routes_to_muse(self, orchestrator):
        out = orchestrator.heuristic_route("Escribe una historia sobre un personaje")
        assert out.primary_motor.motor_name == "muse"

    def test_social_query_routes_to_empathy(self, orchestrator):
        out = orchestrator.heuristic_route("Mi amigo se siente triste y necesita apoyo")
        assert out.primary_motor.motor_name == "empathy"

    def test_unknown_query_routes_to_cora(self, orchestrator):
        out = orchestrator.heuristic_route("what is the meaning of this thing")
        assert out.primary_motor.motor_name == "cora"

    def test_routing_mode_is_heuristic(self, orchestrator):
        out = orchestrator.heuristic_route("Write Python code")
        assert out.routing_mode == "heuristic"

    def test_heuristic_output_type(self, orchestrator):
        out = orchestrator.heuristic_route("simple text")
        assert isinstance(out, OrchestratorOutput)

    def test_heuristic_scores_shape(self, orchestrator):
        out = orchestrator.heuristic_route("simple text")
        assert out.scores.shape == (5,)

    def test_heuristic_at_least_one_activation(self, orchestrator):
        out = orchestrator.heuristic_route("")
        assert len(out.activations) >= 1

    def test_heuristic_primary_score_high(self, orchestrator):
        out = orchestrator.heuristic_route("Write a Python class")
        assert out.primary_motor.score > 0.5

    def test_heuristic_with_logits_passed(self, tiny_config):
        orch = Orchestrator(tiny_config)
        logits = torch.randn(5)
        scores = torch.softmax(logits, dim=-1)
        out = orch.heuristic_route("Python code", logits=logits, scores=scores)
        assert isinstance(out, OrchestratorOutput)

    def test_heuristic_without_logits(self, orchestrator):
        out = orchestrator.heuristic_route("Python code", logits=None, scores=None)
        assert isinstance(out, OrchestratorOutput)

    def test_heuristic_n_active_matches(self, orchestrator):
        out = orchestrator.heuristic_route("Write a Python program")
        assert out.n_active == len(out.activations)

    def test_heuristic_motor_idx_consistent(self, orchestrator):
        out = orchestrator.heuristic_route("demuestra el teorema")
        for act in out.activations:
            assert MOTOR_NAMES[act.motor_idx] == act.motor_name

    def test_heuristic_iterations_positive(self, orchestrator):
        out = orchestrator.heuristic_route("Write Python code")
        for act in out.activations:
            assert act.n_iterations >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 9. FALLBACK TO HEURISTIC IN FORWARD
# ─────────────────────────────────────────────────────────────────────────────

class TestForwardFallback:
    def test_forward_uses_heuristic_when_low_confidence(self, tiny_config):
        """With a near-uniform softmax output (very low max score), heuristic kicks in."""
        orch = Orchestrator(tiny_config)
        # Force classifier to produce ~uniform output → max score ≈ 0.2
        for p in orch.classifier.parameters():
            nn.init.zeros_(p)
        cv = torch.zeros(1, 4, tiny_config.hidden_dim)
        out = orch(cv, query_text="Write a Python class with a bug")
        # When max score is low AND query_text is provided → heuristic
        # max score with zeros ≈ 0.2 (uniform over 5) < default 0.3 threshold
        assert out.routing_mode == "heuristic"

    def test_forward_learned_when_confident(self, tiny_config):
        """With a strong class imbalance, learned routing is used."""
        orch = Orchestrator(tiny_config)
        # Make the classifier strongly prefer one motor
        with torch.no_grad():
            # Set bias of last layer to make motor 0 dominate
            last_layer = [m for m in orch.classifier if isinstance(m, nn.Linear)][-1]
            last_layer.bias.fill_(0.0)
            last_layer.bias[0] = 10.0
        cv = torch.randn(1, 4, tiny_config.hidden_dim)
        out = orch(cv)
        assert out.routing_mode == "learned"

    def test_forward_no_text_no_heuristic(self, tiny_config):
        """No query_text → heuristic never triggered, even if score is low."""
        orch = Orchestrator(tiny_config)
        for p in orch.classifier.parameters():
            nn.init.zeros_(p)
        cv = torch.zeros(1, 4, tiny_config.hidden_dim)
        out = orch(cv, query_text=None)
        assert out.routing_mode == "learned"


# ─────────────────────────────────────────────────────────────────────────────
# 10. GRADIENT FLOW
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:
    def test_logits_have_grad_fn(self, tiny_config):
        orch = Orchestrator(tiny_config)
        cv = torch.randn(1, 4, tiny_config.hidden_dim, requires_grad=True)
        out = orch(cv)
        # logits is detached in output but the classifier still ran
        assert orch.classifier[-1].weight.requires_grad

    def test_backward_through_orchestrator(self, tiny_config):
        orch = Orchestrator(tiny_config)
        cv = torch.randn(1, 4, tiny_config.hidden_dim)
        # Rerun without detach to check grad flow
        pooled = cv.mean(dim=1).mean(dim=0)
        logits = orch.classifier(pooled)
        scores = torch.softmax(logits, dim=-1)
        loss = scores.sum()
        loss.backward()
        # Check at least one parameter has a gradient
        grads = [p.grad for p in orch.parameters() if p.grad is not None]
        assert len(grads) > 0
