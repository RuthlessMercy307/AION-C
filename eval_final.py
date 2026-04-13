"""
eval_final.py — Evaluación del modelo 1.1B entrenado con Motor-Sequential.

Corre después de las 4 fases terminadas:
    1. Carga el checkpoint final del sequential training
    2. Corre los 50 canonical eval prompts
    3. Corre los 5 experimentos de Parte 21 contra el modelo real
    4. Genera reporte JSON + resumen consola

Uso:
    python eval_final.py --checkpoint checkpoints/aion_1b_sequential.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from experiments.benchmark_local import build_pipeline
from evaluation.eval_prompts import EVAL_PROMPTS
from evaluation.metrics import generation_quality_score


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(checkpoint: Path, vocab_size: int):
    log(f"Loading checkpoint from {checkpoint}...")
    pipeline, cfg = build_pipeline("1b", vocab_size=vocab_size)
    ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck)
    missing, unexpected = pipeline.load_state_dict(state, strict=False)
    log(f"  loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        log(f"  (first missing): {missing[:3]}")
    pipeline.eval()
    return pipeline, cfg


def run_eval_prompts(pipeline, tok, device: torch.device) -> Dict[str, Any]:
    log("Running 50 canonical eval prompts...")
    from train_1b_canonical import greedy_generate

    def gen_fn(query: str):
        return greedy_generate(pipeline, tok, query, device, max_new=40, use_amp=False)

    t0 = time.perf_counter()
    result = generation_quality_score(list(EVAL_PROMPTS), gen_fn)
    elapsed = time.perf_counter() - t0
    log(f"  done in {elapsed:.1f}s")
    log(f"  exact_match: {result.exact_match:.3f}")
    log(f"  bleu:        {result.bleu:.3f}")
    log(f"  routing_acc: {result.routing_accuracy:.3f}")
    log(f"  COMBINED:    {result.combined:.4f}")
    log(f"  per_domain:  {result.per_domain}")
    return {
        "exact_match": round(result.exact_match, 4),
        "bleu": round(result.bleu, 4),
        "routing_accuracy": round(result.routing_accuracy, 4),
        "combined": round(result.combined, 4),
        "per_domain": result.per_domain,
        "elapsed_sec": round(elapsed, 2),
    }


def run_experiments(pipeline) -> Dict[str, Any]:
    log("Running 5 validation experiments (Parte 21) against real pipeline...")
    from experiments.fase_f.run_real import exp1_real, exp2_real, exp4_real
    from experiments.fase_f import exp3_stress, exp5_self_evaluation

    results = {}
    t0 = time.perf_counter()

    log("  exp1 sequential learning (50 adapters on forge_c)...")
    r = exp1_real(pipeline, n_concepts=50, check_every=10)
    results["exp1"] = r.to_dict()
    log(f"    passed={r.passed} min_pass_rate={r.metrics.get('min_exam_pass_rate')}")

    log("  exp2 cross-domain (5 motors x 3 adapters)...")
    r = exp2_real(pipeline, adapters_per_motor=3)
    results["exp2"] = r.to_dict()
    log(f"    passed={r.passed} min_pass={r.metrics.get('min_exam_pass_rate')}")

    log("  exp3 stress 1000 episodes...")
    r = exp3_stress.run(n=1000)
    results["exp3"] = r.to_dict()
    log(f"    passed={r.passed}")

    log("  exp4 compositional (real pipeline as generator)...")
    r = exp4_real(pipeline)
    results["exp4"] = r.to_dict()
    log(f"    passed={r.passed}")

    log("  exp5 self-evaluation...")
    r = exp5_self_evaluation.run()
    results["exp5"] = r.to_dict()
    log(f"    passed={r.passed} accuracy={r.metrics.get('accuracy')}")

    elapsed = time.perf_counter() - t0
    n_passed = sum(1 for k, v in results.items() if v.get("passed"))
    log(f"  done in {elapsed:.1f}s — {n_passed}/5 experiments passed")
    return {"experiments": results, "n_passed": n_passed, "elapsed_sec": round(elapsed, 2)}


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "checkpoints" / "aion_1b_sequential.pt")
    p.add_argument("--out", type=Path,
                   default=ROOT / "checkpoints" / "eval_final_report.json")
    p.add_argument("--skip-prompts", action="store_true")
    p.add_argument("--skip-experiments", action="store_true")
    args = p.parse_args()

    log("=" * 72)
    log("FINAL EVAL — AION-C 1.1B post-sequential-training")
    log("=" * 72)

    if not args.checkpoint.exists():
        log(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    from experiments.train_production import build_tokenizer
    tok = build_tokenizer(32_000)
    pipeline, cfg = load_model(args.checkpoint, tok.vocab_size)

    device = torch.device("cpu")  # eval runs on CPU — local machine
    pipeline.to(device)

    report: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "timestamp": time.time(),
        "n_params": sum(p.numel() for p in pipeline.parameters()),
    }

    if not args.skip_prompts:
        report["canonical_eval"] = run_eval_prompts(pipeline, tok, device)

    if not args.skip_experiments:
        report["experiments"] = run_experiments(pipeline)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log("")
    log(f"Report: {args.out}")

    # Summary
    log("")
    log("=" * 72)
    log("SUMMARY")
    log("=" * 72)
    if "canonical_eval" in report:
        ce = report["canonical_eval"]
        log(f"  Canonical eval: combined={ce['combined']} "
            f"(exact={ce['exact_match']}, bleu={ce['bleu']}, routing={ce['routing_accuracy']})")
    if "experiments" in report:
        log(f"  Experiments: {report['experiments']['n_passed']}/5 passed")


if __name__ == "__main__":
    main()
