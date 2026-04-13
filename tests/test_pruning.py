"""
tests/test_pruning.py — Parte 24 (Pruning con 4 señales).

Cubre:
    - PruneSignals.normalize: 4 señales en [0, 1]
    - PruneConfig: validación de umbrales y pesos
    - MemoryPruner.retain_score: fórmula ponderada
    - MemoryPruner.decide: KEEP / PROMOTE / COMPRESS / DELETE
    - TTL dinámico proporcional al retain_score
    - prune() sobre 1000 items sintéticos: distribución razonable
    - sleep_prune_hook: integración con SleepCycle
"""

from __future__ import annotations

import math
import random
import time

import pytest

from pruning import (
    PruneSignals,
    PruneConfig,
    PruneAction,
    PruneDecision,
    PruneReport,
    MemoryPruner,
    sleep_prune_hook,
)
from sleep import Episode, EpisodicBuffer, SleepCycle


# ════════════════════════════════════════════════════════════════════════════
# PruneSignals
# ════════════════════════════════════════════════════════════════════════════

class TestPruneSignals:
    def test_normalize_range(self):
        s = PruneSignals(frequency=10, last_access_age=3600, utility=0.7, retrieval_cost=50)
        s1, s2, s3, s4 = s.normalize(max_freq=20, max_cost=100, half_life_sec=7 * 86400)
        assert 0.0 <= s1 <= 1.0
        assert 0.0 <= s2 <= 1.0
        assert 0.0 <= s3 <= 1.0
        assert 0.0 <= s4 <= 1.0
        assert s1 == 0.5  # 10 / 20
        assert s3 == 0.7

    def test_recency_decay_half_life(self):
        # age == half_life_sec → s2 == 0.5
        s = PruneSignals(frequency=0, last_access_age=100.0, utility=0, retrieval_cost=0)
        _, s2, _, _ = s.normalize(1, 1, half_life_sec=100.0)
        assert s2 == pytest.approx(0.5, abs=1e-6)

    def test_recency_fresh_is_one(self):
        s = PruneSignals(frequency=0, last_access_age=0, utility=0, retrieval_cost=0)
        _, s2, _, _ = s.normalize(1, 1, half_life_sec=100.0)
        assert s2 == pytest.approx(1.0)

    def test_recency_ancient_goes_to_zero(self):
        s = PruneSignals(frequency=0, last_access_age=1e6, utility=0, retrieval_cost=0)
        _, s2, _, _ = s.normalize(1, 1, half_life_sec=100.0)
        assert s2 < 1e-6

    def test_utility_clamped(self):
        s = PruneSignals(0, 0, 1.5, 0)
        _, _, s3, _ = s.normalize(1, 1, 100)
        assert s3 == 1.0
        s2 = PruneSignals(0, 0, -0.1, 0)
        _, _, s3b, _ = s2.normalize(1, 1, 100)
        assert s3b == 0.0


# ════════════════════════════════════════════════════════════════════════════
# PruneConfig
# ════════════════════════════════════════════════════════════════════════════

class TestPruneConfig:
    def test_default_valid(self):
        PruneConfig()  # no raise

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError):
            PruneConfig(w_frequency=-0.1)

    def test_bad_threshold_order_rejected(self):
        with pytest.raises(ValueError):
            PruneConfig(delete_threshold=0.8, compress_threshold=0.4, promote_threshold=0.9)

    def test_bad_ttl_bounds_rejected(self):
        with pytest.raises(ValueError):
            PruneConfig(ttl_min_sec=1000, ttl_max_sec=100)


# ════════════════════════════════════════════════════════════════════════════
# MemoryPruner — scoring y decisiones
# ════════════════════════════════════════════════════════════════════════════

class TestMemoryPruner:
    def test_high_signals_promote(self):
        pruner = MemoryPruner()
        signals = PruneSignals(
            frequency=100, last_access_age=0,
            utility=1.0, retrieval_cost=100,
        )
        d = pruner.decide("x", signals, max_freq=100, max_cost=100)
        assert d.action == PruneAction.PROMOTE
        assert d.retain_score >= 0.80

    def test_low_signals_delete(self):
        pruner = MemoryPruner()
        signals = PruneSignals(
            frequency=0.01, last_access_age=1e8,
            utility=0.0, retrieval_cost=0.01,
        )
        d = pruner.decide("x", signals, max_freq=100, max_cost=100)
        assert d.action == PruneAction.DELETE

    def test_mid_signals_keep(self):
        pruner = MemoryPruner()
        signals = PruneSignals(
            frequency=40, last_access_age=0,
            utility=0.6, retrieval_cost=50,
        )
        d = pruner.decide("x", signals, max_freq=100, max_cost=100)
        assert d.action == PruneAction.KEEP

    def test_ttl_monotone_in_score(self):
        pruner = MemoryPruner()
        low = PruneSignals(0.1, 1e6, 0.0, 0.1)
        high = PruneSignals(100, 0, 1.0, 100)
        d_low = pruner.decide("a", low, 100, 100)
        d_high = pruner.decide("b", high, 100, 100)
        assert d_high.ttl_seconds > d_low.ttl_seconds

    def test_ttl_bounds_respected(self):
        cfg = PruneConfig(ttl_min_sec=60, ttl_max_sec=3600)
        p = MemoryPruner(cfg)
        # score 0 → ttl_min
        s = p._ttl_for(0.0)
        assert s == 60
        s = p._ttl_for(1.0)
        assert s == 3600
        s = p._ttl_for(0.5)
        assert s == pytest.approx((60 + 3600) / 2, abs=1.0)

    def test_prune_empty(self):
        r = MemoryPruner().prune([])
        assert r.decisions == []
        assert r.stats()["total"] == 0

    def test_prune_report_partitions(self):
        pruner = MemoryPruner()
        items = [
            ("top", PruneSignals(100, 0, 1.0, 100)),
            ("keep", PruneSignals(50, 0, 0.6, 50)),
            ("mid", PruneSignals(5, 86400, 0.4, 5)),
            ("gone", PruneSignals(0.01, 1e8, 0.0, 0.01)),
        ]
        report = pruner.prune(items)
        assert "top" in report.promoted
        assert "gone" in report.deleted
        # Cada item tiene exactamente una acción
        seen = set()
        for d in report.decisions:
            assert d.item_id not in seen
            seen.add(d.item_id)
        assert len(seen) == 4


# ════════════════════════════════════════════════════════════════════════════
# Stress test: 1000 items sintéticos
# ════════════════════════════════════════════════════════════════════════════

class TestStress1000:
    def test_thousand_items_distribution(self):
        rng = random.Random(42)
        now = 0.0
        items = []
        for i in range(1000):
            # Sesgado: 10% "calientes", 20% medios, 70% fríos
            roll = rng.random()
            if roll < 0.1:
                freq = rng.uniform(50, 100)
                age = rng.uniform(0, 3600)
                util = rng.uniform(0.8, 1.0)
                cost = rng.uniform(50, 100)
            elif roll < 0.3:
                freq = rng.uniform(10, 40)
                age = rng.uniform(3600, 86400 * 7)
                util = rng.uniform(0.4, 0.7)
                cost = rng.uniform(10, 40)
            else:
                freq = rng.uniform(0, 5)
                age = rng.uniform(86400 * 30, 86400 * 365)
                util = rng.uniform(0.0, 0.3)
                cost = rng.uniform(0, 10)
            items.append((f"i{i}", PruneSignals(freq, age, util, cost)))

        pruner = MemoryPruner(PruneConfig())
        report = pruner.prune(items)
        stats = report.stats()

        # Deben quedar 1000 decisiones
        assert stats["total"] == 1000
        # Una mayoría debe descartarse (los fríos)
        assert stats["deleted"] >= 500
        # Alguna fracción debe promoverse (los calientes)
        assert stats["promoted"] >= 50
        # No debe promoverse la mayoría
        assert stats["promoted"] <= 200
        # Suma consistente
        assert (
            stats["kept"] + stats["promoted"] + stats["compressed"] + stats["deleted"]
            == stats["total"]
        )

    def test_report_identity_preserved(self):
        pruner = MemoryPruner()
        items = [
            (f"id_{i}", PruneSignals(i, 0, 0.5, 0))
            for i in range(50)
        ]
        r = pruner.prune(items)
        returned_ids = {d.item_id for d in r.decisions}
        assert returned_ids == {f"id_{i}" for i in range(50)}


# ════════════════════════════════════════════════════════════════════════════
# Integración con SleepCycle
# ════════════════════════════════════════════════════════════════════════════

class TestSleepIntegration:
    def test_sleep_prune_hook_used_in_cycle(self):
        buf = EpisodicBuffer()
        buf.add(Episode("python", "ok", user_feedback="up"))
        buf.add(Episode("rust", "ok", user_feedback="down"))
        buf.add(Episode("hola", "hi"))

        hook = sleep_prune_hook(MemoryPruner())
        cycle = SleepCycle(buf, prune_hook=hook)
        log = cycle.run()

        prune_phase = log.phase("prune")
        assert prune_phase is not None
        assert prune_phase.data["source"] == "pruner"
        assert "stats" in prune_phase.data
        assert prune_phase.data["stats"]["total"] == 3
