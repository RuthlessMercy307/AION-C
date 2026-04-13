"""
tests/test_anti_forgetting.py — Tests para Parte 9.3 del MEGA-PROMPT
=====================================================================

Cubre las 5 capas:
  Capa 2 — MotorIsolation (congela motores no seleccionados)
  Capa 3 — WeightImportanceTracker (running mean/var, mascara de protección)
  Capa 4 — ExamRunner + RollbackManager + should_rollback
  Capa 5 — SelectiveReplay + compute_weight_delta
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from training.anti_forgetting import (
    MotorIsolation,
    WeightImportanceTracker,
    ExamItem, ExamResult, ExamRunner,
    RollbackManager, should_rollback,
    SelectiveReplay, compute_weight_delta,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: modelo dummy con motores
# ─────────────────────────────────────────────────────────────────────────────


class DummyMotor(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.motors = nn.ModuleDict({
            "cora":     DummyMotor(),
            "forge_c":  DummyMotor(),
            "axiom":    DummyMotor(),
            "muse":     DummyMotor(),
            "empathy":  DummyMotor(),
        })


# ─────────────────────────────────────────────────────────────────────────────
# Capa 2 — MotorIsolation
# ─────────────────────────────────────────────────────────────────────────────


class TestMotorIsolation:
    def test_freezes_non_target_motors(self):
        m = DummyModel()
        iso = MotorIsolation(m, train_motors=["forge_c"])
        n_frozen = iso.apply()
        assert n_frozen > 0
        # forge_c debe estar libre
        for p in m.motors["forge_c"].parameters():
            assert p.requires_grad
        # los demás motores deben estar congelados
        for name in ("cora", "axiom", "muse", "empathy"):
            for p in m.motors[name].parameters():
                assert not p.requires_grad
        iso.restore()
        for p in m.parameters():
            assert p.requires_grad

    def test_context_manager(self):
        m = DummyModel()
        with MotorIsolation(m, train_motors=["axiom"]):
            for p in m.motors["cora"].parameters():
                assert not p.requires_grad
        # restored
        for p in m.parameters():
            assert p.requires_grad

    def test_no_motors_attribute_safe(self):
        # Modelo sin .motors no debe explotar
        class Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)
        iso = MotorIsolation(Plain(), train_motors=["forge_c"])
        assert iso.apply() == 0  # nada que congelar
        iso.restore()

    def test_empty_train_motors_freezes_all(self):
        m = DummyModel()
        iso = MotorIsolation(m, train_motors=[])
        iso.apply()
        for motor in m.motors.values():
            for p in motor.parameters():
                assert not p.requires_grad


# ─────────────────────────────────────────────────────────────────────────────
# Capa 3 — WeightImportanceTracker
# ─────────────────────────────────────────────────────────────────────────────


class TestWeightImportanceTracker:
    def test_initial_variance_is_zero(self):
        m = nn.Linear(4, 4)
        tracker = WeightImportanceTracker(m, momentum=0.9)
        for name in dict(m.named_parameters()):
            v = tracker.variance(name)
            assert v is not None
            assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)

    def test_variance_grows_when_weight_changes(self):
        m = nn.Linear(4, 4)
        tracker = WeightImportanceTracker(m, momentum=0.5)
        # Cambia los pesos drásticamente, llama update varias veces
        with torch.no_grad():
            for _ in range(5):
                m.weight.add_(torch.randn_like(m.weight))
                tracker.update()
        v = tracker.variance("weight")
        assert v.sum() > 0  # debe haber crecido

    def test_importance_mask_shape_matches(self):
        m = nn.Linear(4, 4)
        tracker = WeightImportanceTracker(m)
        mask = tracker.importance_mask(threshold=10.0)
        # Threshold alto → todos los pesos son "estables"
        assert mask["weight"].shape == m.weight.shape
        assert (mask["weight"] == 1.0).all()

    def test_protection_factor_caps(self):
        m = nn.Linear(4, 4)
        tracker = WeightImportanceTracker(m)
        # Inicial: var=0 → todos protegidos
        factors = tracker.protection_factor(threshold=10.0, min_factor=0.1)
        # Todos en min_factor
        assert torch.allclose(factors["weight"],
                              torch.full_like(factors["weight"], 0.1))

    def test_invalid_momentum(self):
        m = nn.Linear(4, 4)
        with pytest.raises(ValueError):
            WeightImportanceTracker(m, momentum=0.0)
        with pytest.raises(ValueError):
            WeightImportanceTracker(m, momentum=1.0)

    def test_n_updates_counter(self):
        m = nn.Linear(4, 4)
        tracker = WeightImportanceTracker(m)
        for _ in range(3):
            tracker.update()
        assert tracker.n_updates == 3


# ─────────────────────────────────────────────────────────────────────────────
# Capa 4 — ExamRunner + RollbackManager
# ─────────────────────────────────────────────────────────────────────────────


class TestExamRunner:
    def test_perfect_score(self):
        items = [
            ExamItem(query="2+2", expected="4"),
            ExamItem(query="hola", expected="hola"),
        ]
        def gen(q, item):
            return item.expected
        result = ExamRunner(items, gen).run()
        assert result.score == 1.0
        assert result.correct == 2

    def test_partial_score(self):
        items = [
            ExamItem(query="a", expected="A"),
            ExamItem(query="b", expected="B"),
            ExamItem(query="c", expected="C"),
        ]
        def gen(q, item):
            return item.expected if item.query != "b" else "wrong"
        result = ExamRunner(items, gen).run()
        assert result.correct == 2
        assert result.score == pytest.approx(2 / 3)

    def test_substring_match(self):
        items = [ExamItem(query="x", expected="hello")]
        def gen(q, i): return "well, hello there!"
        result = ExamRunner(items, gen).run()
        assert result.score == 1.0

    def test_generator_exception_treated_as_wrong(self):
        items = [ExamItem(query="x", expected="y")]
        def gen(q, i): raise RuntimeError("boom")
        result = ExamRunner(items, gen).run()
        assert result.score == 0.0

    def test_custom_matcher(self):
        items = [ExamItem(query="x", expected="42")]
        def gen(q, i): return "42.0"
        def numeric_match(g, e): return float(g) == float(e)
        result = ExamRunner(items, gen, matcher_fn=numeric_match).run()
        assert result.score == 1.0


class TestRollbackManager:
    def test_snapshot_and_rollback(self):
        m = nn.Linear(4, 4)
        rb = RollbackManager(m)
        rb.snapshot()
        original_weight = m.weight.detach().clone()
        with torch.no_grad():
            m.weight.fill_(99.0)
        rb.rollback()
        assert torch.allclose(m.weight, original_weight)

    def test_rollback_without_snapshot_raises(self):
        rb = RollbackManager(nn.Linear(4, 4))
        with pytest.raises(RuntimeError):
            rb.rollback()

    def test_has_snapshot(self):
        rb = RollbackManager(nn.Linear(4, 4))
        assert not rb.has_snapshot()
        rb.snapshot()
        assert rb.has_snapshot()
        rb.discard()
        assert not rb.has_snapshot()


class TestShouldRollback:
    def test_drop_above_threshold(self):
        assert should_rollback(0.95, 0.92, max_drop=0.02)

    def test_drop_below_threshold(self):
        assert not should_rollback(0.95, 0.94, max_drop=0.02)

    def test_score_improved(self):
        assert not should_rollback(0.80, 0.90)


# ─────────────────────────────────────────────────────────────────────────────
# Capa 5 — SelectiveReplay
# ─────────────────────────────────────────────────────────────────────────────


class TestSelectiveReplay:
    def test_register_and_select(self):
        replay = SelectiveReplay()
        # Ejemplo 1: usa "weight" mucho, "bias" poco
        replay.register_example("ex1", {"weight": torch.tensor(2.0), "bias": torch.tensor(0.1)})
        replay.register_example("ex2", {"weight": torch.tensor(0.1), "bias": torch.tensor(2.0)})
        # Si el delta dice que "weight" cambió mucho → ex1 es prioritario
        delta = {"weight": 5.0, "bias": 0.01}
        sel = replay.select(delta, top_k=1)
        assert sel == ["ex1"]

    def test_select_empty_when_no_examples(self):
        assert SelectiveReplay().select({}, top_k=5) == []

    def test_select_filters_zero_score(self):
        replay = SelectiveReplay()
        replay.register_example("a", {"weight": torch.tensor(0.0)})
        sel = replay.select({"weight": 5.0}, top_k=10)
        assert sel == []

    def test_top_k_limits(self):
        replay = SelectiveReplay()
        for i in range(5):
            replay.register_example(f"ex{i}", {"weight": torch.tensor(float(i + 1))})
        sel = replay.select({"weight": 1.0}, top_k=3)
        assert len(sel) == 3
        # Los más relevantes deben venir primero
        assert sel[0] == "ex4"

    def test_compute_weight_delta(self):
        before = {"a": torch.zeros(4), "b": torch.zeros(2)}
        after  = {"a": torch.ones(4),  "b": torch.zeros(2)}
        delta = compute_weight_delta(before, after)
        assert delta["a"] == pytest.approx(2.0)  # ||ones(4)||_2 = 2
        assert delta["b"] == 0.0

    def test_clear(self):
        replay = SelectiveReplay()
        replay.register_example("a", {"w": torch.tensor(1.0)})
        replay.clear()
        assert len(replay) == 0
