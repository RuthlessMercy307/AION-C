"""
tests/test_sleep_cycle.py — Parte 23 (Sleep Cycle ritual).

Cubre:
    - Episode / EpisodicBuffer: add, drain, snapshot, max_size FIFO
    - SleepCycle ritual: orden estricto de las 6 preguntas, stubs, hooks
    - SleepCycleLog: serialización y acceso por fase
    - SleepDaemon: triggers (manual, inactivity, overflow), last_log
"""

from __future__ import annotations

import time

import pytest

from sleep.cycle import (
    Episode,
    EpisodicBuffer,
    PhaseResult,
    SleepCycle,
    SleepCycleLog,
    SLEEP_QUESTIONS,
)
from sleep.daemon import SleepDaemon, SleepTrigger


# ════════════════════════════════════════════════════════════════════════════
# EpisodicBuffer
# ════════════════════════════════════════════════════════════════════════════

class TestEpisodicBuffer:
    def test_add_and_len(self):
        b = EpisodicBuffer()
        b.add(Episode(user_text="hola", aion_response="hi"))
        b.add(Episode(user_text="qué tal", aion_response="bien"))
        assert len(b) == 2

    def test_drain_empties(self):
        b = EpisodicBuffer()
        b.add(Episode("a", "A"))
        b.add(Episode("b", "B"))
        out = b.drain()
        assert len(out) == 2
        assert len(b) == 0

    def test_snapshot_does_not_empty(self):
        b = EpisodicBuffer()
        b.add(Episode("a", "A"))
        out = b.snapshot()
        assert out[0].user_text == "a"
        assert len(b) == 1

    def test_max_size_fifo(self):
        b = EpisodicBuffer(max_size=3)
        for i in range(5):
            b.add(Episode(f"q{i}", f"a{i}"))
        assert len(b) == 3
        ep = b.snapshot()
        # Los 2 primeros se cayeron — quedan q2, q3, q4
        assert [e.user_text for e in ep] == ["q2", "q3", "q4"]

    def test_invalid_max_size(self):
        with pytest.raises(ValueError):
            EpisodicBuffer(max_size=0)

    def test_clear(self):
        b = EpisodicBuffer()
        b.add(Episode("x", "y"))
        b.clear()
        assert len(b) == 0


# ════════════════════════════════════════════════════════════════════════════
# SleepCycle — orden y resultados
# ════════════════════════════════════════════════════════════════════════════

def _seed_buffer():
    b = EpisodicBuffer()
    # Cinco episodios con distintos feedback/topics
    b.add(Episode("python es lento",   "sí depende", user_feedback="up"))
    b.add(Episode("python types hint", "usa mypy",    user_feedback="up"))
    b.add(Episode("rust borrow",       "ownership",   user_feedback="down"))
    b.add(Episode("rust lifetimes",    "confuso",     user_feedback=None))
    b.add(Episode("hola",              "hi",          user_feedback=None))
    return b


class TestSleepCycleStubs:
    def test_order_is_strict(self):
        cycle = SleepCycle(_seed_buffer())
        log = cycle.run(trigger="manual")
        names = [p.name for p in log.phases]
        expected = [n for n, _ in SLEEP_QUESTIONS]
        assert names == expected

    def test_drains_buffer(self):
        buf = _seed_buffer()
        assert len(buf) == 5
        SleepCycle(buf).run()
        assert len(buf) == 0

    def test_log_is_serializable(self):
        import json
        log = SleepCycle(_seed_buffer()).run()
        d = log.to_dict()
        json.dumps(d)
        assert d["episodes_processed"] == 5
        assert len(d["phases"]) == 6
        assert d["duration_ms"] >= 0
        assert d["error"] is None

    def test_recollect_counts_topics(self):
        log = SleepCycle(_seed_buffer()).run()
        r = log.phase("recollect")
        assert r is not None
        assert r.data["count"] == 5
        # python, rust, hola → 3 temas distintos
        assert len(r.data["by_topic"]) == 3

    def test_score_stub_uses_explicit_feedback(self):
        log = SleepCycle(_seed_buffer()).run()
        s = log.phase("score")
        assert s is not None
        scores = s.data["scores"]
        assert scores[0] == 1.0  # up
        assert scores[1] == 1.0  # up
        assert scores[2] == 0.0  # down
        assert scores[3] == 0.5  # none
        assert s.data["mean"] == pytest.approx((1 + 1 + 0 + 0.5 + 0.5) / 5)

    def test_prune_stub_removes_low_score(self):
        log = SleepCycle(_seed_buffer()).run()
        p = log.phase("prune")
        assert p is not None
        # El episodio con score 0.0 (rust borrow) queda removido
        assert p.data["removed"] == 1
        assert 2 in p.data["removed_indices"]

    def test_compress_stub_forms_clusters(self):
        log = SleepCycle(_seed_buffer()).run()
        c = log.phase("compress")
        assert c is not None
        # "python" aparece 2x → cluster; "rust" 2x → cluster; "hola" 1x → no
        assert c.data["clusters"] == 2

    def test_consolidate_stub_picks_high_score(self):
        log = SleepCycle(_seed_buffer()).run()
        c = log.phase("consolidate")
        assert c is not None
        # Episodios con score >= 0.7 → los 2 con "up"
        assert c.data["consolidated"] == 2

    def test_followups_stub_per_cluster(self):
        log = SleepCycle(_seed_buffer()).run()
        f = log.phase("followups")
        assert f is not None
        assert len(f.data["questions"]) >= 1
        assert all(isinstance(q, str) for q in f.data["questions"])

    def test_empty_buffer_still_runs(self):
        cycle = SleepCycle(EpisodicBuffer())
        log = cycle.run()
        assert log.episodes_processed == 0
        assert len(log.phases) == 6


# ════════════════════════════════════════════════════════════════════════════
# SleepCycle — hooks inyectables
# ════════════════════════════════════════════════════════════════════════════

class TestSleepCycleHooks:
    def test_reward_hook_called(self):
        called = {}
        def reward(eps):
            called["n"] = len(eps)
            return {"scores": {0: 0.9, 1: 0.9}, "mean": 0.9, "source": "custom"}

        buf = _seed_buffer()
        log = SleepCycle(buf, reward_hook=reward).run()
        assert called["n"] == 5
        assert log.phase("score").data["source"] == "custom"

    def test_followups_hook_overrides_stub(self):
        def followups(eps, prev):
            return ["pregunta custom 1", "pregunta custom 2"]
        log = SleepCycle(_seed_buffer(), followups_hook=followups).run()
        assert log.phase("followups").data["source"] == "hook"
        assert log.phase("followups").data["questions"] == ["pregunta custom 1", "pregunta custom 2"]

    def test_prune_hook_receives_prev(self):
        captured = {}
        def prune(eps, prev):
            captured["has_score"] = "score" in prev
            return {"kept": len(eps), "removed": 0}
        log = SleepCycle(_seed_buffer(), prune_hook=prune).run()
        # El hook debe ver el resultado de la fase previa 'score'
        assert captured["has_score"] is True


# ════════════════════════════════════════════════════════════════════════════
# SleepDaemon — triggers
# ════════════════════════════════════════════════════════════════════════════

class TestSleepDaemon:
    def test_force_run_manual_trigger(self):
        d = SleepDaemon(SleepCycle(_seed_buffer()), inactivity_seconds=999, overflow_threshold=999)
        log = d.force_run()
        assert log.trigger == "manual"
        assert d.last_log is log

    def test_no_trigger_when_fresh_activity(self):
        d = SleepDaemon(
            SleepCycle(_seed_buffer()),
            inactivity_seconds=3600,
            overflow_threshold=999,
        )
        d.notify_activity()
        assert d.should_run() is None
        assert d.maybe_run() is None

    def test_inactivity_trigger(self):
        buf = _seed_buffer()
        d = SleepDaemon(SleepCycle(buf), inactivity_seconds=0.01, overflow_threshold=999)
        d.notify_activity(ts=time.time() - 10.0)
        t = d.should_run()
        assert t == SleepTrigger.INACTIVITY
        log = d.maybe_run()
        assert log is not None
        assert log.trigger == "inactivity"

    def test_inactivity_without_episodes_no_run(self):
        buf = EpisodicBuffer()
        d = SleepDaemon(SleepCycle(buf), inactivity_seconds=0.01, overflow_threshold=999)
        d.notify_activity(ts=time.time() - 10.0)
        assert d.should_run() is None

    def test_overflow_trigger(self):
        buf = EpisodicBuffer()
        for i in range(10):
            buf.add(Episode(f"q{i}", "a"))
        d = SleepDaemon(SleepCycle(buf), inactivity_seconds=9999, overflow_threshold=5)
        assert d.should_run() == SleepTrigger.OVERFLOW
        log = d.maybe_run()
        assert log is not None
        assert log.trigger == "overflow"

    def test_run_resets_activity(self):
        buf = _seed_buffer()
        d = SleepDaemon(SleepCycle(buf), inactivity_seconds=0.01, overflow_threshold=999)
        d.notify_activity(ts=time.time() - 10.0)
        before = d.last_activity_ts
        d.maybe_run()
        assert d.last_activity_ts > before

    def test_maybe_run_returns_none_until_triggered(self):
        d = SleepDaemon(
            SleepCycle(EpisodicBuffer()),
            inactivity_seconds=99999,
            overflow_threshold=99999,
        )
        assert d.maybe_run() is None

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            SleepDaemon(SleepCycle(EpisodicBuffer()), inactivity_seconds=-1)
        with pytest.raises(ValueError):
            SleepDaemon(SleepCycle(EpisodicBuffer()), overflow_threshold=-1)
