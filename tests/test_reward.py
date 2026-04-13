"""
tests/test_reward.py — Parte 25 (Reward probabilístico).

Cubre:
    - ExplicitSignal values (UP/DOWN/CORRECTION/NONE)
    - ImplicitSignals + _implicit_score peso de no_correction > thanks
    - IntrinsicSignals.to_mean_std
    - RewardConfig: validación
    - RewardEstimator.compute: fórmula + varianza combinada + conservative
    - ImplicitDetector: thanks, re_asked_similar, code_copied, abandoned, no_correction
    - RewardLedger: acumulación + stats
    - sleep_reward_hook: integración con SleepCycle
"""

from __future__ import annotations

import math

import pytest

from reward import (
    ExplicitSignal,
    ImplicitSignals,
    IntrinsicSignals,
    RewardSignals,
    RewardConfig,
    RewardEstimator,
    RewardEstimate,
    ImplicitDetector,
    RewardLedger,
    sleep_reward_hook,
)
from sleep import Episode, EpisodicBuffer, SleepCycle


# ════════════════════════════════════════════════════════════════════════════
# RewardConfig
# ════════════════════════════════════════════════════════════════════════════

class TestRewardConfig:
    def test_defaults_valid(self):
        RewardConfig()

    def test_zero_total_weight_rejected(self):
        with pytest.raises(ValueError):
            RewardConfig(alpha_explicit=0, beta_implicit=0, gamma_intrinsic=0)

    def test_negative_k_rejected(self):
        with pytest.raises(ValueError):
            RewardConfig(k_std=-1)


# ════════════════════════════════════════════════════════════════════════════
# RewardEstimator — fórmula
# ════════════════════════════════════════════════════════════════════════════

class TestRewardEstimator:
    def test_explicit_up_gives_high_mean(self):
        est = RewardEstimator()
        e = est.compute(RewardSignals(explicit=ExplicitSignal.UP))
        assert e.mean > 0.65

    def test_explicit_down_gives_low_mean(self):
        est = RewardEstimator()
        e = est.compute(RewardSignals(explicit=ExplicitSignal.DOWN))
        assert e.mean < 0.40

    def test_none_signal_near_neutral(self):
        est = RewardEstimator()
        e = est.compute(RewardSignals(explicit=ExplicitSignal.NONE))
        assert 0.35 <= e.mean <= 0.65

    def test_no_correction_weighs_more_than_thanks(self):
        est = RewardEstimator()
        thanks_only = est.compute(RewardSignals(
            explicit=ExplicitSignal.NONE,
            implicit=ImplicitSignals(thanks=True),
        ))
        continue_only = est.compute(RewardSignals(
            explicit=ExplicitSignal.NONE,
            implicit=ImplicitSignals(no_correction_continue=True),
        ))
        assert continue_only.mean > thanks_only.mean

    def test_re_asked_penalizes(self):
        est = RewardEstimator()
        neutral = est.compute(RewardSignals())
        re_ask = est.compute(RewardSignals(
            implicit=ImplicitSignals(re_asked_similar=True)
        ))
        assert re_ask.mean < neutral.mean

    def test_conservative_below_or_equal_mean(self):
        est = RewardEstimator()
        e = est.compute(RewardSignals(explicit=ExplicitSignal.UP))
        assert e.conservative <= e.mean
        assert 0.0 <= e.conservative <= 1.0

    def test_std_non_negative(self):
        e = RewardEstimator().compute(RewardSignals())
        assert e.std >= 0.0

    def test_components_breakdown_present(self):
        e = RewardEstimator().compute(RewardSignals(explicit=ExplicitSignal.UP))
        assert "explicit" in e.components
        assert "implicit" in e.components
        assert "intrinsic" in e.components
        assert e.components["explicit"] == pytest.approx(1.0, abs=0.01)

    def test_intrinsic_symbolic_inconsistent_penalizes(self):
        est = RewardEstimator()
        a = est.compute(RewardSignals(
            intrinsic=IntrinsicSignals(symbolic_consistent=True)
        ))
        b = est.compute(RewardSignals(
            intrinsic=IntrinsicSignals(symbolic_consistent=False)
        ))
        assert a.mean > b.mean


# ════════════════════════════════════════════════════════════════════════════
# ImplicitDetector
# ════════════════════════════════════════════════════════════════════════════

class TestImplicitDetector:
    def setup_method(self):
        self.det = ImplicitDetector()

    def test_detect_thanks(self):
        s = self.det.detect(
            assistant_response="here is the answer",
            next_user_text="gracias!",
            previous_user_text="help me please",
        )
        assert s.thanks is True

    def test_detect_re_asked_similar(self):
        s = self.det.detect(
            assistant_response="the answer is 42",
            next_user_text="what is the answer please",
            previous_user_text="what is the answer",
        )
        assert s.re_asked_similar is True
        assert s.no_correction_continue is False

    def test_detect_no_correction_continue(self):
        s = self.det.detect(
            assistant_response="here",
            next_user_text="ok next let's talk about something else",
            previous_user_text="first question",
        )
        assert s.no_correction_continue is True
        assert s.thanks is False

    def test_detect_correction_blocks_continue(self):
        s = self.det.detect(
            assistant_response="the capital is Lyon",
            next_user_text="no, actually wrong, it's Paris",
            previous_user_text="what's the capital of france",
        )
        assert s.no_correction_continue is False

    def test_detect_code_copied(self):
        s = self.det.detect(
            assistant_response="```python\ndef f(): pass\n```",
            next_user_text="thanks this works",
            previous_user_text="write a python function",
        )
        assert s.code_copied is True

    def test_detect_abandoned_no_next(self):
        s = self.det.detect(
            assistant_response="here",
            next_user_text=None,
            previous_user_text="something",
        )
        assert s.abandoned is True

    def test_detect_abandoned_time_threshold(self):
        det = ImplicitDetector(abandon_threshold_sec=10)
        # Mucho tiempo sin responder + no hay next → abandoned
        s = det.detect("x", None, "p", time_to_next_turn_sec=60)
        assert s.abandoned is True
        # Poco tiempo, todavía están tal vez escribiendo → no abandonado
        s2 = det.detect("x", None, "p", time_to_next_turn_sec=5)
        assert s2.abandoned is False


# ════════════════════════════════════════════════════════════════════════════
# RewardLedger
# ════════════════════════════════════════════════════════════════════════════

class TestRewardLedger:
    def test_add_and_mean(self):
        ledger = RewardLedger()
        est = RewardEstimator()
        ledger.add("forge_c", est.compute(RewardSignals(explicit=ExplicitSignal.UP)))
        ledger.add("forge_c", est.compute(RewardSignals(explicit=ExplicitSignal.UP)))
        assert ledger.count_for("forge_c") == 2
        assert ledger.mean_for("forge_c") > 0.6

    def test_snapshot_multiple_keys(self):
        ledger = RewardLedger()
        est = RewardEstimator()
        ledger.add("forge_c:python", est.compute(RewardSignals(explicit=ExplicitSignal.UP)))
        ledger.add("muse:poem", est.compute(RewardSignals(explicit=ExplicitSignal.DOWN)))
        snap = ledger.snapshot()
        assert "forge_c:python" in snap
        assert "muse:poem" in snap
        assert snap["forge_c:python"]["n"] == 1

    def test_missing_key_defaults_zero(self):
        ledger = RewardLedger()
        assert ledger.mean_for("unknown") == 0.0
        assert ledger.count_for("unknown") == 0


# ════════════════════════════════════════════════════════════════════════════
# SleepCycle integration
# ════════════════════════════════════════════════════════════════════════════

class TestRewardLedgerPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        l1 = RewardLedger()
        est = RewardEstimator()
        l1.add("forge_c", est.compute(RewardSignals(explicit=ExplicitSignal.UP)))
        l1.add("forge_c", est.compute(RewardSignals(explicit=ExplicitSignal.DOWN)))
        l1.add("muse", est.compute(RewardSignals(explicit=ExplicitSignal.UP)))
        path = tmp_path / "ledger.jsonl"
        l1.save_jsonl(path)
        assert path.exists()

        l2 = RewardLedger()
        l2.load_jsonl(path)
        assert l2.count_for("forge_c") == 2
        assert l2.count_for("muse") == 1
        assert l2.mean_for("forge_c") == pytest.approx(l1.mean_for("forge_c"))

    def test_load_missing_file_noop(self, tmp_path):
        l = RewardLedger()
        l.load_jsonl(tmp_path / "nothing.jsonl")  # no raise
        assert l.keys() == []

    def test_save_empty_ledger(self, tmp_path):
        l = RewardLedger()
        path = tmp_path / "empty.jsonl"
        l.save_jsonl(path)
        l2 = RewardLedger()
        l2.load_jsonl(path)
        assert l2.keys() == []


class TestSleepIntegration:
    def test_reward_hook_used_in_cycle(self):
        buf = EpisodicBuffer()
        buf.add(Episode("q1", "a1", user_feedback="up"))
        buf.add(Episode("q2", "a2", user_feedback="down"))
        buf.add(Episode("q3", "a3"))
        cycle = SleepCycle(buf, reward_hook=sleep_reward_hook())
        log = cycle.run()
        score = log.phase("score")
        assert score.data["source"] == "reward_estimator"
        # El 0 tuvo "up" → alto; el 1 "down" → bajo
        assert score.data["scores"][0] > score.data["scores"][1]
