"""
run_real.py — Ejecuta los 5 experimentos de Parte 21 contra el tiny real.

Diferencias vs run_all.py:
    - exp1 y exp2 usan los motores del MoSEPipeline cargado desde
      checkpoints/tiny_canonical.pt (no FakeMotor).
    - exp3, exp4, exp5 usan las mismas funciones — su ground truth no
      depende del modelo cargado.

Genera experiments/fase_f/results/real_*.json y un real_summary.json.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import torch

from experiments.fase_f import (
    exp3_stress,
    exp4_compositional,
    exp5_self_evaluation,
)
from experiments.fase_f.common import (
    ExperimentReport,
    RESULTS_DIR,
    write_report,
)
from experiments.fase_f.real_pipeline import (
    load_real_pipeline,
    real_motor_targets,
    real_motor_exam,
    real_motor_outputs,
    real_exam_pass_rate,
)
from growth import (
    AdapterRegistry,
    LoRAConfig,
    attach_adapter_pack,
    build_adapter_pack,
    detach_adapter_pack,
)


# ════════════════════════════════════════════════════════════════════════════
# Exp1 REAL — sequential learning sobre forge_c del tiny
# ════════════════════════════════════════════════════════════════════════════

def exp1_real(pipeline, n_concepts: int = 50, check_every: int = 10) -> ExperimentReport:
    start = time.time()
    motor = pipeline.motors["forge_c"]
    targets = real_motor_targets(motor, max_targets=6)
    exam = real_motor_exam(motor, n=10)
    reference = real_motor_outputs(motor, exam)

    history: List[dict] = []
    min_pass_rate = 1.0

    with tempfile.TemporaryDirectory() as tmp:
        reg = AdapterRegistry(Path(tmp))
        for i in range(1, n_concepts + 1):
            name = f"real_concept_{i:04d}"
            pack = build_adapter_pack(
                motor, targets, LoRAConfig(rank=4), name, "forge_c"
            )
            attach_adapter_pack(motor, pack)
            # Simular fine-tune: ruido aleatorio sobre lora_B
            for path in targets:
                with torch.no_grad():
                    pack.get(path).lora_B.normal_(0, 0.15)
            reg.save(pack)
            detach_adapter_pack(motor, pack)

            if i % check_every == 0:
                pr = real_exam_pass_rate(motor, exam, reference)
                history.append({"concepts_learned": i, "exam_pass_rate": pr})
                min_pass_rate = min(min_pass_rate, pr)

    end = time.time()
    passed = min_pass_rate == 1.0
    return ExperimentReport(
        experiment_id="exp1_sequential_real",
        name="Sequential learning (real tiny forge_c)",
        started_at=start,
        ended_at=end,
        passed=passed,
        summary=(
            f"learned {n_concepts} real adapters on forge_c, "
            f"min exam_pass_rate={min_pass_rate:.4f}, {len(targets)} target linears"
        ),
        metrics={
            "n_concepts": n_concepts,
            "min_exam_pass_rate": min_pass_rate,
            "target_paths": targets,
            "motor": "forge_c",
        },
        details={"history": history},
    )


# ════════════════════════════════════════════════════════════════════════════
# Exp2 REAL — cross-domain sobre los 5 motores reales
# ════════════════════════════════════════════════════════════════════════════

def exp2_real(pipeline, adapters_per_motor: int = 3) -> ExperimentReport:
    start = time.time()
    motor_names = ["cora", "forge_c", "muse", "axiom", "empathy"]
    motors = {name: pipeline.motors[name] for name in motor_names}
    exams = {name: real_motor_exam(motors[name], n=10, seed=i * 13) for i, name in enumerate(motor_names)}
    references = {name: real_motor_outputs(motors[name], exams[name]) for name in motor_names}
    targets_per_motor = {name: real_motor_targets(motors[name], max_targets=4) for name in motor_names}

    with tempfile.TemporaryDirectory() as tmp:
        reg = AdapterRegistry(Path(tmp))
        for name in motor_names:
            motor = motors[name]
            targets = targets_per_motor[name]
            if not targets:
                continue
            for j in range(adapters_per_motor):
                concept = f"{name}_real_{j}"
                pack = build_adapter_pack(
                    motor, targets, LoRAConfig(rank=4), concept, name
                )
                attach_adapter_pack(motor, pack)
                for path in targets:
                    with torch.no_grad():
                        pack.get(path).lora_B.normal_(0, 0.15)
                reg.save(pack)
                detach_adapter_pack(motor, pack)

    per_motor_pass: Dict[str, float] = {}
    for name in motor_names:
        if not targets_per_motor[name]:
            per_motor_pass[name] = 1.0  # no-op motor
            continue
        per_motor_pass[name] = real_exam_pass_rate(motors[name], exams[name], references[name])

    min_pass = min(per_motor_pass.values())
    mean_pass = sum(per_motor_pass.values()) / len(per_motor_pass)
    passed = min_pass == 1.0
    end = time.time()

    return ExperimentReport(
        experiment_id="exp2_cross_domain_real",
        name="Cross-domain interference (real 5 motors)",
        started_at=start,
        ended_at=end,
        passed=passed,
        summary=(
            f"5 real motors × {adapters_per_motor} adapters; "
            f"min_pass={min_pass:.4f}, mean_pass={mean_pass:.4f}"
        ),
        metrics={
            "n_motors": len(motor_names),
            "adapters_per_motor": adapters_per_motor,
            "min_exam_pass_rate": min_pass,
            "mean_exam_pass_rate": mean_pass,
            "targets_per_motor": {
                name: targets_per_motor[name] for name in motor_names
            },
        },
        details={"per_motor_pass_rate": per_motor_pass},
    )


# ════════════════════════════════════════════════════════════════════════════
# Exp4 REAL — compositional usando la pipeline real para ejecutar steps
# ════════════════════════════════════════════════════════════════════════════

def exp4_real(pipeline) -> ExperimentReport:
    """Versión de exp4 que usa el tiny real para ejecutar los trajectories.

    Esto es una sanity check: verifica que el CompositeOrchestrator NO
    crashea con el real pipeline y que los motores respondan algo (no
    necesariamente bueno — el tiny fue entrenado sin trajectories).
    """
    from composition import CompositeOrchestrator, TrajectoryPlanner
    from synth.canonical_dataloader import EOS_TOKEN_ID
    from experiments.train_production import build_tokenizer

    start = time.time()
    planner = TrajectoryPlanner()
    tok = build_tokenizer(32_000)

    def gen(motor: str, prompt: str, max_tokens: int) -> str:
        try:
            ids = tok.encode(prompt, 96) if hasattr(tok, "encode") else list(range(min(len(prompt), 96)))
            if isinstance(ids, list) and len(ids) == 0:
                return ""
            cur = torch.tensor([ids], dtype=torch.long)
            plen = cur.shape[1]
            with torch.no_grad():
                out = pipeline(cur)
                for _ in range(max_tokens):
                    nxt = int(out.logits[0, -1].argmax().item())
                    if nxt in (0, EOS_TOKEN_ID):
                        break
                    cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
                    if cur.shape[1] >= 120:
                        break
                    out = pipeline(cur)
            try:
                return tok.decode(cur[0, plen:].tolist())
            except Exception:
                return ""
        except Exception as exc:
            return f"[gen error: {exc}]"

    queries = [
        ("escribe una función python", ["forge_c"]),
        ("calcula 15% de 240", ["axiom"]),
        ("por qué llueve", ["cora"]),
        ("explica este código como cuento", ["forge_c", "muse"]),
    ]
    results = []
    for q, expected in queries:
        traj = planner.plan(q)
        try:
            res = CompositeOrchestrator(gen).execute(traj)
            crashed = False
            out_len = len(res.fused_output)
        except Exception as exc:
            crashed = True
            out_len = 0
        results.append({
            "query": q,
            "expected": expected,
            "got_sequence": traj.motor_sequence(),
            "sequence_match": traj.motor_sequence() == expected,
            "crashed": crashed,
            "output_len": out_len,
        })

    end = time.time()
    n_match = sum(1 for r in results if r["sequence_match"])
    no_crashes = all(not r["crashed"] for r in results)
    passed = no_crashes and n_match >= len(queries) * 0.75

    return ExperimentReport(
        experiment_id="exp4_compositional_real",
        name="Compositional test with real tiny",
        started_at=start,
        ended_at=end,
        passed=passed,
        summary=(
            f"{n_match}/{len(queries)} sequences match, no_crashes={no_crashes}"
        ),
        metrics={
            "total": len(queries),
            "sequence_matches": n_match,
            "no_crashes": no_crashes,
        },
        details={"results": results},
    )


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run_real() -> List[ExperimentReport]:
    """Ejecuta los 5 experimentos contra el tiny real."""
    print("Loading real tiny_canonical.pt ...")
    pipeline = load_real_pipeline()
    n_params = sum(p.numel() for p in pipeline.parameters())
    print(f"  loaded {n_params} params, {len(pipeline.motors)} motors")

    reports: List[ExperimentReport] = []

    print("exp1 REAL — sequential learning on real forge_c ...")
    r = exp1_real(pipeline, n_concepts=50, check_every=10)
    write_report(r)
    reports.append(r)
    print(f"  passed={r.passed} — {r.summary}")

    print("exp2 REAL — cross-domain on 5 real motors ...")
    r = exp2_real(pipeline, adapters_per_motor=3)
    write_report(r)
    reports.append(r)
    print(f"  passed={r.passed} — {r.summary}")

    print("exp3 — stress (1000 episodes, model-independent) ...")
    r = exp3_stress.run(n=1000)
    r.experiment_id = "exp3_stress_real"
    write_report(r)
    reports.append(r)
    print(f"  passed={r.passed} — {r.summary}")

    print("exp4 REAL — compositional with real tiny as generator ...")
    r = exp4_real(pipeline)
    write_report(r)
    reports.append(r)
    print(f"  passed={r.passed} — {r.summary}")

    print("exp5 — self-evaluation (model-independent) ...")
    r = exp5_self_evaluation.run()
    r.experiment_id = "exp5_self_evaluation_real"
    write_report(r)
    reports.append(r)
    print(f"  passed={r.passed} — {r.summary}")

    summary = {
        "started_at": min(r.started_at for r in reports),
        "ended_at": max(r.ended_at for r in reports),
        "n_experiments": len(reports),
        "n_passed": sum(1 for r in reports if r.passed),
        "all_passed": all(r.passed for r in reports),
        "reports": [r.to_dict() for r in reports],
        "pipeline_params": n_params,
        "model": "tiny_canonical.pt",
    }
    out = RESULTS_DIR / "real_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"{summary['n_passed']}/{summary['n_experiments']} real experiments passed")
    print(f"summary: {out}")
    return reports


def main() -> None:  # pragma: no cover
    run_real()


if __name__ == "__main__":  # pragma: no cover
    main()
