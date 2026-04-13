#!/usr/bin/env python3
"""
finetune_direct.py — Fine-tune 1.1B on 10K direct Q&A with EOS
================================================================
Loads checkpoint, trains with very low LR to not destroy learned representations.
"""
import json, math, os, random, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn as nn
import torch.nn.functional as F

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

MOTOR_NAMES = ["cora", "forge_c", "muse", "axiom", "empathy"]
DOMAIN_TO_MOTOR = {"cora": 0, "forge_c": 1, "axiom": 3, "muse": 2, "empathy": 4, "general": 0}
EOS_ID = 2

EVAL_PROMPTS = [
    ("Hola, quien eres?", "general"),
    ("If rain causes floods, what happens?", "cora"),
    ("Write a Python function to add two numbers", "forge_c"),
    ("What is 25% of 200?", "axiom"),
    ("Mi amigo esta triste porque perdio su trabajo", "empathy"),
    ("Write a short poem about the moon", "muse"),
    ("How are you?", "general"),
    ("Cuanto es 7 * 8?", "axiom"),
    ("Gracias", "general"),
    ("What can you do?", "general"),
]

def main():
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); random.seed(42)

    log(f"Device: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    from router.pipeline import MoSEPipeline, MoSEConfig
    from experiments.train_production import build_tokenizer

    tok = build_tokenizer(32_000)
    cfg = MoSEConfig(
        hidden_dim=1024, vocab_size=tok.vocab_size,
        enc_n_layers=12, enc_state_dim=16, enc_expand=2, enc_d_conv=4, enc_ffn_mult=4,
        orch_mlp_hidden=512, orch_max_motors=3, orch_min_confidence=0.3,
        motor_max_nodes=8, motor_n_heads=8, motor_threshold=0.01, unif_n_heads=8,
        dec_n_layers=16, dec_n_heads=8, dec_max_seq_len=512,
        dec_state_dim=16, dec_expand=2, dec_d_conv=4, dec_ffn_mult=4,
    )
    pipeline = MoSEPipeline(cfg)

    # Load checkpoint
    ckpt_path = ROOT / "checkpoints" / "aion_1b.pt"
    log(f"Loading {ckpt_path}...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    pipeline.load_state_dict(ckpt["model_state"], strict=False)
    log(f"Loaded step={ckpt.get('step','?')}, val_loss={ckpt.get('val_loss','?')}")
    pipeline.to(device)

    # Load direct Q&A data
    log("Loading direct Q&A data...")
    data = []
    with open(str(ROOT / "datasets" / "direct_qa_10k.jsonl"), encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    random.shuffle(data)
    val_data, train_data = data[:500], data[500:]
    log(f"Train: {len(train_data)}  Val: {len(val_data)}")

    def encode_pair(inp, out, ml=128):
        try: ids = tok.encode(inp + " " + out, ml - 1)
        except TypeError: ids = tok.encode(inp + " " + out)[:ml - 1]
        return ids + [EOS_ID]

    train_ids = [encode_pair(ex["input"], ex["output"]) for ex in train_data]
    val_ids = [encode_pair(ex["input"], ex["output"]) for ex in val_data]

    # Fine-tune config: LOW LR to preserve learned representations
    MAX_STEPS = 2000
    EVAL_EVERY = 200
    PATIENCE = 400
    GRAD_ACCUM = 4
    LR = 5e-5
    CLIP = 0.5

    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=LR, weight_decay=1e-2)
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    best_val = float("inf")
    no_improve = 0
    train_rng = random.Random(42)
    CKPT_OUT = ROOT / "checkpoints" / "aion_1b_direct.pt"

    log(f"Steps: {MAX_STEPS}  LR: {LR}  grad_accum: {GRAD_ACCUM}  patience: {PATIENCE}\n")

    for step in range(1, MAX_STEPS + 1):
        pipeline.train()
        optimizer.zero_grad()
        accum_loss = 0

        for _ in range(GRAD_ACCUM):
            ids = train_rng.choice(train_ids)
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)

            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = pipeline(ids_t)
                    loss = F.cross_entropy(out.logits[0, :-1], ids_t[0, 1:], ignore_index=0) / GRAD_ACCUM
                if math.isfinite(loss.item()):
                    scaler.scale(loss).backward()
                    accum_loss += loss.item() * GRAD_ACCUM
            else:
                out = pipeline(ids_t)
                loss = F.cross_entropy(out.logits[0, :-1], ids_t[0, 1:], ignore_index=0) / GRAD_ACCUM
                if math.isfinite(loss.item()):
                    loss.backward()
                    accum_loss += loss.item() * GRAD_ACCUM

        if device.type == "cuda":
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(pipeline.parameters(), CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(pipeline.parameters(), CLIP)
            optimizer.step()

        if step % 50 == 0:
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            log(f"  step {step:>5}/{MAX_STEPS}  loss={accum_loss/GRAD_ACCUM:.4f}  "
                f"{sps:.2f} sps  ETA {(MAX_STEPS-step)/sps/60:.0f}m")

        # Eval
        if step % EVAL_EVERY == 0:
            pipeline.eval()
            vls = []
            with torch.no_grad():
                for vi in val_ids[:100]:
                    vt = torch.tensor([vi], dtype=torch.long, device=device)
                    if device.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            vo = pipeline(vt)
                    else:
                        vo = pipeline(vt)
                    vl = F.cross_entropy(vo.logits[0, :-1], vt[0, 1:], ignore_index=0)
                    if math.isfinite(vl.item()): vls.append(vl.item())
            val_loss = sum(vls)/len(vls) if vls else float("nan")

            # Decode prompts
            print(f"\n  --- Eval step {step} --- val_loss={val_loss:.4f}")
            for prompt, domain in EVAL_PROMPTS:
                try: enc = tok.encode(prompt, 96)
                except TypeError: enc = tok.encode(prompt)[:96]
                cur = torch.tensor([enc], dtype=torch.long, device=device)
                pl = len(enc)
                with torch.no_grad():
                    for _ in range(60):
                        if device.type == "cuda":
                            with torch.amp.autocast("cuda", dtype=torch.float16):
                                o = pipeline(cur)
                        else:
                            o = pipeline(cur)
                        nxt = int(o.logits[0, -1].float().argmax().item())
                        if nxt in (0, EOS_ID): break
                        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], 1)
                        if cur.shape[1] >= 160: break
                pred = tok.decode(cur[0, pl:].tolist()) if cur.shape[1] > pl else "(empty)"
                routed = o.active_motors[0] if hasattr(o, 'active_motors') and o.active_motors else "?"
                print(f"    [{domain:>8}->{routed:>8}] {prompt[:35]:<35} -> {pred[:55]}")
            print()

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                no_improve = 0
                torch.save({"model_state": pipeline.state_dict(), "config_name": "1b_direct",
                            "step": step, "val_loss": val_loss}, str(CKPT_OUT))
                log(f"  Saved best (val_loss={val_loss:.4f})")
            else:
                no_improve += EVAL_EVERY
            if no_improve >= PATIENCE:
                log(f"Early stop at step {step}")
                break

    elapsed = time.perf_counter() - t0
    log(f"\nDone in {elapsed/60:.0f} min. Best val_loss={best_val:.4f}")
    log(f"Checkpoint: {CKPT_OUT}")

if __name__ == "__main__":
    main()
