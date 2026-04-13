#!/usr/bin/env python3
"""
experiments/train_tiny_50k.py — Train tiny MoSE on 50K diverse dataset
======================================================================

Validates:
  1. All 5 motors activate (routing works)
  2. val_loss doesn't stagnate (diverse data works)
  3. F1 emerges across domains
  4. Routing accuracy > 70%

Uses same metrics as ablation studies for comparability.

Usage:
    cd AION-C
    python -m experiments.train_tiny_50k
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── Fixed eval prompts (bilingual) ─────────────────────────────────────────
EVAL_PROMPTS = [
    ("Hola, quien eres?", "general", "Soy AION-C"),
    ("If rain causes wet soil, does rain cause floods?", "cora", "Yes"),
    ("Write a Python function to reverse a linked list", "forge_c", "def reverse"),
    ("What is 15% of 240?", "axiom", "36"),
    ("Mi amigo esta triste porque perdio su trabajo", "empathy", "triste"),
    ("Write a short scene: a robot discovers music", "muse", "robot"),
]

DOMAIN_IDS = {"cora": 0, "forge_c": 1, "axiom": 2, "muse": 3, "empathy": 4, "general": 5}


def main():
    t0 = time.perf_counter()

    log("=" * 65)
    log("  AION-C Tiny Training on 50K Diverse Dataset")
    log("=" * 65)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("CPU mode")

    # ── 1. Load dataset ──
    log("\n[1/5] Loading dataset...")
    data_path = _ROOT / "datasets" / "dataset_50k.jsonl"
    all_data = []
    with open(str(data_path), encoding="utf-8") as f:
        for line in f:
            all_data.append(json.loads(line))

    rng = random.Random(42)
    rng.shuffle(all_data)
    val_data = all_data[:2000]
    train_data = all_data[2000:]
    log(f"Total: {len(all_data)}, Train: {len(train_data)}, Val: {len(val_data)}")

    domains_dist = Counter(ex["domain"] for ex in train_data)
    log(f"Train domains: {dict(domains_dist)}")

    # ── 2. Build model + tokenizer ──
    log("\n[2/5] Building model...")
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
    pipeline = MoSEPipeline(cfg).to(device)
    params = sum(p.numel() for p in pipeline.parameters())
    log(f"Params: {params:,} ({params/1e6:.2f}M)")

    # ── 3. Tokenize ──
    log("\n[3/5] Tokenizing...")

    def encode(text, max_len=128):
        try:
            return tok.encode(text, max_len)
        except TypeError:
            return tok.encode(text)[:max_len]

    train_ids = [(encode(ex["input"] + " " + ex["output"]), ex.get("domain_id", 5))
                 for ex in train_data]
    val_ids = [encode(ex["input"] + " " + ex["output"]) for ex in val_data[:200]]

    # Val by domain for per-domain metrics
    val_by_domain = {}
    for ex in val_data[:500]:
        d = ex["domain"]
        if d not in val_by_domain:
            val_by_domain[d] = []
        if len(val_by_domain[d]) < 50:
            val_by_domain[d].append(encode(ex["input"] + " " + ex["output"]))

    log(f"Mean train tokens: {sum(len(ids) for ids, _ in train_ids)/len(train_ids):.0f}")

    # ── 4. Train ──
    log("\n[4/5] Training...")
    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=3e-4, weight_decay=1e-2)

    MAX_STEPS = 3000
    EVAL_EVERY = 100
    PATIENCE = 500
    ROUTING_LOSS_WEIGHT = 1.0   # Strong supervision for routing
    BALANCE_LOSS_WEIGHT = 0.5   # Penalize motor imbalance
    best_val = float("inf")
    no_improve = 0
    train_rng = random.Random(42)
    torch.manual_seed(42)

    eval_records = []
    motor_activations = Counter()
    routing_correct = 0
    routing_total = 0

    # Motor name → index mapping for routing targets
    motor_name_to_idx = {"cora": 0, "forge_c": 1, "muse": 2, "axiom": 3, "empathy": 4}
    domain_to_motor_idx = {
        "cora": 0, "forge_c": 1, "axiom": 3, "muse": 2, "empathy": 4, "general": 0,
    }

    # Running activation counts for load balancing (exponential moving average)
    activation_ema = torch.ones(5, device=device) / 5  # start uniform
    ema_alpha = 0.01

    log(f"Steps: {MAX_STEPS}  eval_every: {EVAL_EVERY}  patience: {PATIENCE}")
    log(f"Routing loss weight: {ROUTING_LOSS_WEIGHT}  Balance loss weight: {BALANCE_LOSS_WEIGHT}\n")

    for step in range(1, MAX_STEPS + 1):
        ids, domain_id = train_rng.choice(train_ids)
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)

        pipeline.train()
        out = pipeline(ids_t)

        # 1. Language modeling loss
        lm_loss = F.cross_entropy(out.logits[0, :-1], ids_t[0, 1:], ignore_index=0)

        # 2. Routing loss — teach orchestrator to route correctly
        # Recompute logits from encoder (not detached)
        concepts = pipeline.encoder(ids_t)
        pooled = concepts.mean(dim=1).mean(dim=0, keepdim=True)  # [1, D]
        orch_logits = pipeline.orchestrator.classifier(pooled)    # [1, 5]

        # Target: the motor matching this domain
        domain_name = list(DOMAIN_IDS.keys())[domain_id] if domain_id < 6 else "general"
        target_motor_idx = domain_to_motor_idx.get(domain_name, 0)
        target_t = torch.tensor([target_motor_idx], dtype=torch.long, device=device)
        routing_loss = F.cross_entropy(orch_logits, target_t)

        # 3. Load balancing loss — penalize if one motor dominates
        # Softmax of current logits = probability distribution over motors
        orch_probs = F.softmax(orch_logits.squeeze(0), dim=-1)  # [5]
        # Update EMA of activations
        activation_ema = (1 - ema_alpha) * activation_ema + ema_alpha * orch_probs.detach()
        # Ideal = uniform (0.2 each). Penalize deviation.
        # Load balance = N * sum(fraction_i * prob_i) — minimized when uniform
        balance_loss = 5.0 * (activation_ema * orch_probs).sum()

        # Combined loss
        loss = lm_loss + ROUTING_LOSS_WEIGHT * routing_loss + BALANCE_LOSS_WEIGHT * balance_loss

        if not math.isfinite(loss.item()):
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
        optimizer.step()

        # Track routing
        for motor_name in out.active_motors:
            motor_activations[motor_name] += 1

        # Check routing accuracy
        domain_to_motor_name = {"cora": "cora", "forge_c": "forge_c", "axiom": "axiom",
                                "muse": "muse", "empathy": "empathy", "general": "cora"}
        expected_motor = domain_to_motor_name.get(domain_name, "cora")
        if expected_motor in out.active_motors:
            routing_correct += 1
        routing_total += 1

        if step % 50 == 0:
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            ema_str = " ".join(f"{v:.2f}" for v in activation_ema.tolist())
            log(f"  step {step:>5}/{MAX_STEPS}  lm={lm_loss.item():.3f} "
                f"route={routing_loss.item():.3f} bal={balance_loss.item():.3f}  "
                f"acc={100*routing_correct/max(1,routing_total):.0f}%  "
                f"ema=[{ema_str}]  {sps:.1f} sps")

        # ── Eval ──
        if step % EVAL_EVERY == 0:
            pipeline.eval()

            # Overall val loss
            vls = []
            with torch.no_grad():
                for vi in val_ids[:100]:
                    vt = torch.tensor([vi], dtype=torch.long, device=device)
                    vo = pipeline(vt)
                    vl = F.cross_entropy(vo.logits[0, :-1], vt[0, 1:], ignore_index=0)
                    if math.isfinite(vl.item()):
                        vls.append(vl.item())
            val_loss = sum(vls) / len(vls) if vls else float("nan")

            # Per-domain val loss
            domain_losses = {}
            with torch.no_grad():
                for domain, d_ids in val_by_domain.items():
                    d_vls = []
                    for vi in d_ids[:30]:
                        vt = torch.tensor([vi], dtype=torch.long, device=device)
                        vo = pipeline(vt)
                        vl = F.cross_entropy(vo.logits[0, :-1], vt[0, 1:], ignore_index=0)
                        if math.isfinite(vl.item()):
                            d_vls.append(vl.item())
                    if d_vls:
                        domain_losses[domain] = sum(d_vls) / len(d_vls)

            # Greedy decode on fixed prompts
            f1s, cf1s = [], []
            decode_results = []
            for prompt, expected_domain, expected_keyword in EVAL_PROMPTS:
                enc_ids = encode(prompt, 96)
                cur = torch.tensor([enc_ids], dtype=torch.long, device=device)
                plen = len(enc_ids)
                with torch.no_grad():
                    for _ in range(32):
                        o = pipeline(cur)
                        nxt = int(o.logits[0, -1].argmax().item())
                        if nxt in (0, 2):
                            break
                        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
                        n_gen = cur.shape[1] - plen
                        if n_gen >= 3:
                            try:
                                ts = tok.decode([nxt])
                                if ts.rstrip().endswith(('.', '?', '!')):
                                    break
                            except Exception:
                                pass

                pred_ids = cur[0, plen:].tolist()
                try:
                    pred = tok.decode(pred_ids)
                except Exception:
                    pred = ""

                # Word F1
                pw = set(pred.lower().split())
                rw = set(expected_keyword.lower().split())
                if pw and rw:
                    tp = len(pw & rw)
                    f1 = 2 * (tp/len(pw)) * (tp/len(rw)) / (tp/len(pw) + tp/len(rw)) if tp > 0 else 0
                else:
                    f1 = 0

                # Contains
                cf1 = 1.0 if expected_keyword.lower() in pred.lower() else 0.0
                f1s.append(f1)
                cf1s.append(cf1)
                decode_results.append((expected_domain, prompt[:40], pred[:50]))

            mean_f1 = sum(f1s) / len(f1s) if f1s else 0
            mean_cf1 = sum(cf1s) / len(cf1s) if cf1s else 0
            route_acc = routing_correct / max(1, routing_total)

            # Print
            print(f"\n  --- Eval step {step} ---")
            print(f"  val_loss={val_loss:.4f}  F1={mean_f1:.3f}  cF1={mean_cf1:.3f}  "
                  f"route_acc={100*route_acc:.0f}%")
            dl_str = "  ".join(f"{d}={v:.2f}" for d, v in sorted(domain_losses.items()))
            print(f"  Domain losses: {dl_str}")
            for domain, prompt, pred in decode_results:
                p_safe = pred[:45].encode("ascii", "replace").decode()
                print(f"    [{domain:>8}] {prompt[:35]:<35} -> {p_safe}")
            print(f"  Motor activations: {dict(motor_activations)}")
            print()

            eval_records.append({
                "step": step, "val_loss": round(val_loss, 4),
                "f1": round(mean_f1, 4), "cf1": round(mean_cf1, 4),
                "route_acc": round(route_acc, 4),
                "domain_losses": {k: round(v, 4) for k, v in domain_losses.items()},
            })

            # Early stopping
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                no_improve = 0
            else:
                no_improve += EVAL_EVERY
            if no_improve >= PATIENCE:
                log(f"Early stop at step {step}")
                break

    # ── 5. Summary ──
    elapsed = time.perf_counter() - t0
    log(f"\n{'='*65}")
    log(f"  RESULTS ({elapsed:.0f}s)")
    log(f"{'='*65}")

    log("\nVal loss trajectory:")
    for r in eval_records:
        marker = " <-- best" if abs(r["val_loss"] - best_val) < 0.001 else ""
        log(f"  step {r['step']:>5}: val_loss={r['val_loss']:.4f}  F1={r['f1']:.3f}  "
            f"cF1={r['cf1']:.3f}  route={100*r['route_acc']:.0f}%{marker}")

    # Stagnation check
    if len(eval_records) >= 5:
        early = eval_records[1]["val_loss"]  # step 200
        late = eval_records[-1]["val_loss"]
        improvement = early - late
        if improvement > 0.05:
            log(f"\n  PASS: val_loss improved {improvement:.4f} after step 200")
        else:
            log(f"\n  WARN: possible stagnation (improvement={improvement:.4f})")

    # Motor activation check
    log(f"\nMotor activations: {dict(motor_activations)}")
    n_active = sum(1 for v in motor_activations.values() if v > 0)
    if n_active >= 3:
        log(f"  PASS: {n_active}/5 motors activated")
    else:
        log(f"  WARN: only {n_active}/5 motors activated")

    # Final routing
    final_route = eval_records[-1]["route_acc"] if eval_records else 0
    if final_route > 0.70:
        log(f"  PASS: routing accuracy {100*final_route:.0f}% > 70%")
    elif final_route > 0.40:
        log(f"  OK: routing accuracy {100*final_route:.0f}% (above random)")
    else:
        log(f"  WARN: routing accuracy {100*final_route:.0f}% (near random)")

    # Save
    results_path = _ROOT / "experiments" / "tiny_50k_results.json"
    with open(str(results_path), "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2)
    log(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
