#!/usr/bin/env python3
"""
train_tiny_canonical.py — Fase C: entrena tiny 5.5M con el dataset canónico
==============================================================================

Recipe (heredada de train_4090.py + auto_learn_demo.py + fixes EOS):
  - Tokenizer aion_32k BPE (32000 vocab)
  - Tiny config: hidden_dim=64, dec_layers=2, enc_layers=2 (~5.5M params)
  - LM loss + Routing loss + Balance loss
  - WeightedRandomSampler 50/50 SKILL/MEM-or-not (Fase B opción c)
  - Batch=1 (CPU), grad_accum=4
  - 3000 steps default, 4-7% del dataset cubierto
  - Save: checkpoints/tiny_canonical.pt

Uso:
    python train_tiny_canonical.py
    python train_tiny_canonical.py --steps 5000 --lr 5e-4
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F

from synth.canonical_dataloader import (
    load_canonical_records, encode_record, weighted_sampler_indices,
    domain_to_motor_idx, MOTOR_NAMES, EOS_TOKEN_ID, quick_stats,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_tiny_pipeline():
    from router.pipeline import MoSEPipeline, MoSEConfig
    from experiments.train_production import build_tokenizer
    tok = build_tokenizer(32_000)
    cfg = MoSEConfig(
        hidden_dim=64, vocab_size=tok.vocab_size,
        enc_n_layers=2, enc_state_dim=4, enc_expand=2, enc_d_conv=4, enc_ffn_mult=2,
        orch_mlp_hidden=32, orch_max_motors=3, orch_min_confidence=0.3,
        motor_max_nodes=8, motor_n_heads=4, motor_threshold=0.01, unif_n_heads=4,
        dec_n_layers=2, dec_n_heads=4, dec_max_seq_len=128,
        dec_state_dim=4, dec_expand=2, dec_d_conv=4, dec_ffn_mult=2,
    )
    return MoSEPipeline(cfg), tok, cfg


def train(
    dataset_path: Path,
    ckpt_path:    Path,
    n_steps:      int = 3000,
    lr:           float = 1e-3,
    grad_accum:   int = 4,
    routing_w:    float = 1.0,
    balance_w:    float = 0.3,
    clip:         float = 1.0,
    log_every:    int = 100,
    seed:         int = 42,
) -> dict:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    random.seed(seed)

    log("=" * 64)
    log("FASE C — Entrenamiento tiny canónico")
    log("=" * 64)

    log(f"[1/4] Loading dataset {dataset_path.name}...")
    records = load_canonical_records(dataset_path)
    stats = quick_stats(records)
    log(f"  Records: {stats.total:,}  with={stats.n_with:,}  without={stats.n_without:,}")
    log(f"  by_domain: {stats.by_domain}")
    log(f"  by_type:   {stats.by_type}")

    log("[2/4] Building tiny pipeline...")
    pipeline, tok, cfg = build_tiny_pipeline()
    n_params = sum(p.numel() for p in pipeline.parameters())
    log(f"  Params: {n_params:,} ({n_params/1e6:.2f}M)")

    log("[3/4] Sampling step indices (WeightedRandomSampler 50/50)...")
    indices = weighted_sampler_indices(records, n_steps=n_steps * grad_accum, target_ratio=0.5, seed=seed)
    sampled_with = sum(1 for i in indices if records[i].has_skill or records[i].has_mem)
    log(f"  Total samples to draw: {len(indices):,}")
    log(f"  with_skill_or_mem: {sampled_with}/{len(indices)} ({100*sampled_with/len(indices):.1f}%)")

    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=lr, weight_decay=1e-2)
    activation_ema = torch.ones(5) / 5.0
    routing_correct = 0
    routing_total = 0
    motor_counts = {m: 0 for m in MOTOR_NAMES}
    sample_pos = 0

    log(f"[4/4] Training {n_steps} steps  lr={lr}  grad_accum={grad_accum}\n")

    for step in range(1, n_steps + 1):
        pipeline.train()
        optimizer.zero_grad()
        accum_lm = 0.0
        accum_route = 0.0

        for _ in range(grad_accum):
            idx = indices[sample_pos % len(indices)]
            sample_pos += 1
            record = records[idx]
            ids = encode_record(tok, record, max_len=128)
            if len(ids) < 4:
                continue
            ids_t = torch.tensor([ids], dtype=torch.long)

            out = pipeline(ids_t)
            lm = F.cross_entropy(
                out.logits[0, :-1], ids_t[0, 1:], ignore_index=0
            ) / grad_accum

            # Routing supervision
            concepts = pipeline.encoder(ids_t)
            pooled = concepts.mean(1).mean(0, keepdim=True)
            orch_log = pipeline.orchestrator.classifier(pooled)
            tgt = domain_to_motor_idx(record.domain)
            rl = F.cross_entropy(
                orch_log, torch.tensor([tgt], dtype=torch.long)
            ) / grad_accum

            # Balance loss
            probs = F.softmax(orch_log.squeeze(0), dim=-1)
            bl = 5.0 * (activation_ema * probs).sum() / grad_accum

            loss = lm + routing_w * rl + balance_w * bl
            if not math.isfinite(loss.item()):
                continue
            loss.backward()
            accum_lm += lm.item() * grad_accum
            accum_route += rl.item() * grad_accum

            activation_ema = 0.99 * activation_ema + 0.01 * probs.detach()
            for m in out.active_motors:
                if m in motor_counts:
                    motor_counts[m] += 1
            expected_motor = MOTOR_NAMES[tgt]
            if expected_motor in out.active_motors:
                routing_correct += 1
            routing_total += 1

        nn.utils.clip_grad_norm_(pipeline.parameters(), clip)
        optimizer.step()

        if step % log_every == 0:
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            eta_min = (n_steps - step) / max(sps, 1e-6) / 60
            racc = 100 * routing_correct / max(1, routing_total)
            log(f"  step {step:>5}/{n_steps}  "
                f"lm={accum_lm/grad_accum:.3f}  "
                f"route={accum_route/grad_accum:.3f}  "
                f"acc={racc:.0f}%  "
                f"{sps:.2f}sps  ETA {eta_min:.0f}m")

    elapsed = time.perf_counter() - t0
    racc = 100 * routing_correct / max(1, routing_total)

    log(f"\nDONE in {elapsed/60:.1f} min")
    log(f"Final routing accuracy: {racc:.1f}%")
    log(f"Motor activation counts: {motor_counts}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state":     pipeline.state_dict(),
        "config_name":     "tiny_canonical",
        "n_params":        n_params,
        "n_steps":         n_steps,
        "routing_acc":     racc,
        "motor_counts":    motor_counts,
        "elapsed_minutes": elapsed / 60,
    }, str(ckpt_path))
    log(f"Checkpoint saved: {ckpt_path}")

    # Punto 9: BrainVersionManager — guardar como brain/v* con metadata
    try:
        from brain.version_manager import BrainVersionManager
        bvm = BrainVersionManager(root_dir=ROOT / "brain")
        version = bvm.save_version(
            state_dict=pipeline.state_dict(),
            notes=f"tiny_canonical {n_steps} steps",
            metrics={
                "routing_acc": racc / 100.0,
                "lm_loss":     0.0,  # placeholder, the last value is in motor_counts
                "n_steps":     float(n_steps),
            },
            metadata={
                "config_name":     "tiny_canonical",
                "n_params":        n_params,
                "elapsed_minutes": elapsed / 60,
                "motor_counts":    motor_counts,
            },
        )
        log(f"BrainVersionManager: saved as {version.id} (parent={version.parent_id})")
    except Exception as exc:
        log(f"BrainVersionManager save failed: {exc}")
    return {
        "routing_acc": racc,
        "motor_counts": motor_counts,
        "elapsed_minutes": elapsed / 60,
        "n_steps": n_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "dataset_canonical_70k.jsonl")
    ap.add_argument("--ckpt",    type=Path, default=ROOT / "checkpoints" / "tiny_canonical.pt")
    ap.add_argument("--steps",   type=int,  default=3000)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()
    train(
        dataset_path=args.dataset,
        ckpt_path=args.ckpt,
        n_steps=args.steps,
        lr=args.lr,
        grad_accum=args.grad_accum,
    )


if __name__ == "__main__":
    main()
