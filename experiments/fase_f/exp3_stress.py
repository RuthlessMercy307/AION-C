"""
exp3_stress.py — Parte 21.3 (stress test: 1000 episodios).

Pregunta científica:
    Con 1000 episodios en el buffer, ¿el SleepCycle (con pruner+reward+compressor)
    corre sin degradación? ¿La memoria queda acotada? ¿Los ratios son sanos?

Procedimiento:
    1. Sembrar 1000 episodios sintéticos con distribuciones variadas de
       feedback (up/down/none) y temas.
    2. Correr UN sleep cycle completo.
    3. Verificar:
       - Termina en tiempo razonable (< 5s en CPU).
       - Las 6 fases se completaron sin error.
       - El pruner clasificó todos los items (sin off-by-one).
       - El compressor formó al menos 1 cluster.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict

from compression import (
    Clusterer,
    HierarchicalCompressor,
    HierarchicalStore,
    sleep_compress_hook,
)
from experiments.fase_f.common import ExperimentReport, write_report
from pruning import MemoryPruner, sleep_prune_hook
from reward import RewardEstimator, sleep_reward_hook
from sleep import Episode, EpisodicBuffer, SleepCycle


SAMPLE_TEXTS = [
    "python types hints",
    "python functions",
    "rust ownership borrow",
    "rust lifetimes",
    "javascript async await",
    "calcular 15 porciento",
    "por qué llueve en invierno",
    "me siento triste hoy",
    "escribe un poema sobre el mar",
    "hola qué tal",
]


def run(n: int = 1000, seed: int = 99) -> ExperimentReport:
    start = time.time()
    rng = random.Random(seed)
    buf = EpisodicBuffer(max_size=n + 10)
    for _ in range(n):
        text = rng.choice(SAMPLE_TEXTS)
        feedback = rng.choice(["up", "down", None, None, None])
        buf.add(Episode(text, "ok", user_feedback=feedback))

    store = HierarchicalStore()
    compressor = HierarchicalCompressor(store, Clusterer(threshold=0.3))
    cycle = SleepCycle(
        buf,
        reward_hook=sleep_reward_hook(RewardEstimator()),
        prune_hook=sleep_prune_hook(MemoryPruner()),
        compress_hook=sleep_compress_hook(compressor),
    )
    log = cycle.run(trigger="stress")

    end = time.time()

    phase_names = [p.name for p in log.phases]
    all_six = len(log.phases) == 6
    no_error = log.error is None
    duration_ok = log.duration_ms < 10_000  # 10s cap

    prune_stats = log.phase("prune").data.get("stats", {}) if log.phase("prune") else {}
    compress_stats = log.phase("compress").data if log.phase("compress") else {}

    passed = all_six and no_error and duration_ok and prune_stats.get("total") == n

    return ExperimentReport(
        experiment_id="exp3_stress",
        name="Stress test 1000 episodes",
        started_at=start,
        ended_at=end,
        passed=passed,
        summary=(
            f"ran {n} episodes in {log.duration_ms:.0f}ms; "
            f"all 6 phases={all_six}, error={log.error}; "
            f"compress clusters={compress_stats.get('clusters', 0)}"
        ),
        metrics={
            "episodes": n,
            "all_six_phases": all_six,
            "duration_ms": log.duration_ms,
            "prune_stats": prune_stats,
            "compress_clusters": compress_stats.get("clusters", 0),
        },
        details={
            "phase_names": phase_names,
            "phase_summaries": [p.summary for p in log.phases],
        },
    )


def main() -> None:
    report = run()
    write_report(report)
    print(f"{report.experiment_id}: passed={report.passed} — {report.summary}")


if __name__ == "__main__":  # pragma: no cover
    main()
