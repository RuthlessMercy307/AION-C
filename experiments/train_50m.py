#!/usr/bin/env python3
"""
experiments/train_50m.py — Train 50M MoSE on 57.5K diverse data (CPU)
=====================================================================

Target: model that speaks. "Hola" → coherent, "2+2" → AXIOM, F1 > 0.
Uses Phase 1.5 style: all params unfrozen, routing loss, load balance.

~2 hours on Ryzen 5 3600 CPU.
"""

from __future__ import annotations
import json, math, os, random, sys, time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn as nn
import torch.nn.functional as F


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


EVAL_PROMPTS = [
    ("Hola", "general", "Hola"),
    ("Who are you?", "general", "AION-C"),
    ("If rain causes floods, what happens?", "cora", "floods"),
    ("2 + 2", "axiom", "4"),
    ("Write a sort function", "forge_c", "def"),
    ("Mi amigo esta triste", "empathy", "triste"),
    ("Write a scene about discovery", "muse", "discover"),
]

MOTOR_NAMES = ["cora", "forge_c", "muse", "axiom", "empathy"]
DOMAIN_TO_MOTOR = {"cora": 0, "forge_c": 1, "axiom": 3, "muse": 2, "empathy": 4, "general": 0}


def main():
    t0 = time.perf_counter()
    log("=" * 65)
    log("  AION-C 50M Training — CPU")
    log("=" * 65)

    device = torch.device("cpu")
    torch.manual_seed(42)
    random.seed(42)

    # ── Model ──
    from router.pipeline import MoSEPipeline, MoSEConfig
    from experiments.train_production import build_tokenizer

    tok = build_tokenizer(32_000)
    cfg = MoSEConfig(
        hidden_dim=256, vocab_size=tok.vocab_size,
        enc_n_layers=6, enc_state_dim=8, enc_expand=2, enc_d_conv=4, enc_ffn_mult=2,
        orch_mlp_hidden=128, orch_max_motors=3, orch_min_confidence=0.3,
        motor_max_nodes=8, motor_n_heads=4, motor_threshold=0.01, unif_n_heads=4,
        dec_n_layers=10, dec_n_heads=4, dec_max_seq_len=128,
        dec_state_dim=8, dec_expand=2, dec_d_conv=4, dec_ffn_mult=2,
    )
    pipeline = MoSEPipeline(cfg).to(device)
    params = sum(p.numel() for p in pipeline.parameters())
    log(f"Model: {params:,} ({params/1e6:.1f}M)")

    # ── Data ──
    log("Loading data...")
    data = []
    with open(str(_ROOT / "datasets" / "dataset_50k.jsonl"), encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    random.shuffle(data)
    val_data = data[:2000]
    train_data = data[2000:]
    log(f"Train: {len(train_data)}  Val: {len(val_data)}")

    def encode(text, max_len=128):
        try: return tok.encode(text, max_len)
        except TypeError: return tok.encode(text)[:max_len]

    train_ids = [(encode(ex["input"] + " " + ex["output"]), ex.get("domain_id", 5)) for ex in train_data]
    val_ids = [encode(ex["input"] + " " + ex["output"]) for ex in val_data[:300]]

    # ── Training config ──
    MAX_STEPS = 8000
    EVAL_EVERY = 200
    PATIENCE = 600
    ROUTING_W = 1.0
    BALANCE_W = 0.3
    LR = 3e-4
    SAVE_PATH = _ROOT / "checkpoints" / "aion_50m.pt"
    SAVE_PATH.parent.mkdir(exist_ok=True)

    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=LR, weight_decay=1e-2)
    best_val = float("inf")
    no_improve = 0
    train_rng = random.Random(42)
    activation_ema = torch.ones(5) / 5
    routing_correct = 0
    routing_total = 0
    eval_records = []
    motor_counts = Counter()

    log(f"Steps: {MAX_STEPS}  LR: {LR}  Routing_W: {ROUTING_W}  Balance_W: {BALANCE_W}")
    log(f"Checkpoint: {SAVE_PATH}\n")

    for step in range(1, MAX_STEPS + 1):
        ids, domain_id = train_rng.choice(train_ids)
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)

        pipeline.train()
        out = pipeline(ids_t)

        # LM loss
        lm_loss = F.cross_entropy(out.logits[0, :-1], ids_t[0, 1:], ignore_index=0)

        # Routing loss
        concepts = pipeline.encoder(ids_t)
        pooled = concepts.mean(dim=1).mean(dim=0, keepdim=True)
        orch_logits = pipeline.orchestrator.classifier(pooled)
        domain_name = ["cora", "forge_c", "axiom", "muse", "empathy", "general"][min(domain_id, 5)]
        target_idx = DOMAIN_TO_MOTOR.get(domain_name, 0)
        routing_loss = F.cross_entropy(orch_logits, torch.tensor([target_idx]))

        # Load balance
        probs = F.softmax(orch_logits.squeeze(0), dim=-1)
        activation_ema = 0.99 * activation_ema + 0.01 * probs.detach()
        balance_loss = 5.0 * (activation_ema * probs).sum()

        loss = lm_loss + ROUTING_W * routing_loss + BALANCE_W * balance_loss

        if not math.isfinite(loss.item()):
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
        optimizer.step()

        # Track
        for m in out.active_motors:
            motor_counts[m] += 1
        expected = ["cora", "forge_c", "muse", "axiom", "empathy", "cora"][min(domain_id, 5)]
        if expected in out.active_motors:
            routing_correct += 1
        routing_total += 1

        if step % 100 == 0:
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            eta_m = (MAX_STEPS - step) / sps / 60
            log(f"  step {step:>5}/{MAX_STEPS}  lm={lm_loss.item():.3f} "
                f"route={routing_loss.item():.3f}  "
                f"acc={100*routing_correct/max(1,routing_total):.0f}%  "
                f"{sps:.2f} sps  ETA {eta_m:.0f}m")

        # ── Eval ──
        if step % EVAL_EVERY == 0:
            pipeline.eval()

            # Val loss
            vls = []
            with torch.no_grad():
                for vi in val_ids[:100]:
                    vt = torch.tensor([vi], dtype=torch.long, device=device)
                    vo = pipeline(vt)
                    vl = F.cross_entropy(vo.logits[0, :-1], vt[0, 1:], ignore_index=0)
                    if math.isfinite(vl.item()):
                        vls.append(vl.item())
            val_loss = sum(vls) / len(vls) if vls else float("nan")

            # Decode fixed prompts
            f1s, cf1s = [], []
            print(f"\n  --- Eval step {step} --- val_loss={val_loss:.4f}")
            for prompt, domain, keyword in EVAL_PROMPTS:
                enc = encode(prompt, 96)
                cur = torch.tensor([enc], dtype=torch.long, device=device)
                plen = len(enc)
                with torch.no_grad():
                    for _ in range(40):
                        o = pipeline(cur)
                        nxt = int(o.logits[0, -1].argmax().item())
                        if nxt in (0, 2): break
                        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
                        if cur.shape[1] - plen >= 3:
                            try:
                                ts = tok.decode([nxt])
                                if ts.rstrip().endswith(('.', '?', '!')): break
                            except: pass
                        if cur.shape[1] >= 128: break

                pred_ids = cur[0, plen:].tolist()
                try: pred = tok.decode(pred_ids)
                except: pred = ""

                # Route check
                with torch.no_grad():
                    o2 = pipeline(torch.tensor([enc], dtype=torch.long, device=device))
                    routed = o2.active_motors[0] if o2.active_motors else "?"
                    scores = o2.orchestrator.scores.tolist()

                # F1
                pw, rw = set(pred.lower().split()), set(keyword.lower().split())
                if pw and rw:
                    tp = len(pw & rw)
                    f1 = 2*(tp/len(pw))*(tp/len(rw))/(tp/len(pw)+tp/len(rw)) if tp else 0
                else: f1 = 0
                cf1 = 1.0 if keyword.lower() in pred.lower() else 0.0
                f1s.append(f1)
                cf1s.append(cf1)

                p_safe = pred[:50].encode("ascii", "replace").decode()
                sc = " ".join(f"{MOTOR_NAMES[i]}={s:.2f}" for i, s in enumerate(scores[:5]))
                print(f"    [{domain:>8} -> {routed:>8}] F1={f1:.2f} cF1={cf1:.0f} | {prompt[:30]:<30} -> {p_safe}")

            mean_f1 = sum(f1s) / len(f1s)
            mean_cf1 = sum(cf1s) / len(cf1s)
            route_acc = routing_correct / max(1, routing_total)
            print(f"  val_loss={val_loss:.4f}  F1={mean_f1:.3f}  cF1={mean_cf1:.3f}  route={100*route_acc:.0f}%")
            print(f"  motors: {dict(motor_counts)}\n")

            eval_records.append({
                "step": step, "val_loss": round(val_loss, 4),
                "f1": round(mean_f1, 4), "cf1": round(mean_cf1, 4),
                "route_acc": round(route_acc, 4),
            })

            # Early stopping
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                no_improve = 0
                # Save best
                torch.save({"model_state": pipeline.state_dict(), "config_name": "50m",
                            "step": step, "val_loss": val_loss}, str(SAVE_PATH))
                log(f"  Saved best checkpoint (val_loss={val_loss:.4f})")
            else:
                no_improve += EVAL_EVERY
            if no_improve >= PATIENCE:
                log(f"Early stop at step {step}")
                break

    # ── Summary ──
    elapsed = time.perf_counter() - t0
    log(f"\n{'='*65}")
    log(f"  DONE ({elapsed/60:.0f} min)")
    log(f"{'='*65}\n")

    for r in eval_records:
        marker = " <-- best" if abs(r["val_loss"] - best_val) < 0.001 else ""
        log(f"  step {r['step']:>5}: vl={r['val_loss']:.4f} F1={r['f1']:.3f} cF1={r['cf1']:.3f} route={100*r['route_acc']:.0f}%{marker}")

    log(f"\nMotors: {dict(motor_counts)}")
    log(f"Best val_loss: {best_val:.4f}")
    log(f"Checkpoint: {SAVE_PATH}")

    # Save results
    with open(str(_ROOT / "experiments" / "train_50m_results.json"), "w") as f:
        json.dump(eval_records, f, indent=2)


if __name__ == "__main__":
    main()
