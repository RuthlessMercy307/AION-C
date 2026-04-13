#!/usr/bin/env python3
"""
train_4090.py — Train ~1.1B MoSE on RTX 4090 (24GB)
====================================================
Phase 1.5: all params, mixed data, routing+balance loss.
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
EVAL_PROMPTS = [
    ("Hola, quien eres?", "general", "AION-C"),
    ("If rain causes wet soil, does rain cause floods?", "cora", "Yes"),
    ("Write a Python function to reverse a linked list", "forge_c", "def"),
    ("What is 15% of 240?", "axiom", "36"),
    ("Mi amigo esta triste porque perdio su trabajo", "empathy", "triste"),
    ("Write a short scene: a robot discovers music", "muse", "robot"),
]

def main():
    t0 = time.perf_counter()
    device = torch.device("cuda")
    torch.manual_seed(42); random.seed(42)

    log("=" * 65)
    log(f"  AION-C 1.1B Training — RTX 4090")
    log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log("=" * 65)

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
    pipeline = MoSEPipeline(cfg).to(device)
    params = sum(p.numel() for p in pipeline.parameters())
    log(f"Params: {params:,} ({params/1e6:.0f}M)")

    # Resume from checkpoint if exists
    CKPT = ROOT / "checkpoints" / "aion_1b.pt"
    CKPT.parent.mkdir(exist_ok=True)
    start_step = 0
    if CKPT.exists():
        ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
        pipeline.load_state_dict(ckpt["model_state"], strict=False)
        start_step = ckpt.get("step", 0)
        log(f"Resumed from checkpoint: step={start_step}, val_loss={ckpt.get('val_loss', '?')}")
    pipeline.to(device)
    log(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Data
    log("Loading data...")
    data = []
    with open(str(ROOT / "datasets" / "dataset_50k.jsonl"), encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    random.shuffle(data)
    val_data, train_data = data[:2000], data[2000:]
    log(f"Train: {len(train_data)}  Val: {len(val_data)}")

    EOS_ID = 2

    def encode(text, ml=128):
        try: return tok.encode(text, ml)
        except TypeError: return tok.encode(text)[:ml]

    def encode_pair(inp, out, ml=128):
        """Encode input + output + EOS token."""
        ids = encode(inp + " " + out, ml - 1)
        return ids + [EOS_ID]

    train_ids = [(encode_pair(ex["input"], ex["output"]), ex.get("domain_id", 5)) for ex in train_data]
    val_ids = [encode_pair(ex["input"], ex["output"]) for ex in val_data[:200]]

    # Config
    MAX_STEPS = 15000
    EVAL_EVERY = 200
    PATIENCE = 500
    GRAD_ACCUM = 8
    ROUTING_W = 1.0
    BALANCE_W = 0.3
    LR = 1e-4
    WARMUP = 300
    CLIP = 0.5

    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=LR, weight_decay=1e-2)

    # Cosine with warmup
    def lr_lambda(step):
        if step < WARMUP:
            return step / max(1, WARMUP)
        progress = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler("cuda")
    best_val = float("inf")
    no_improve = 0
    train_rng = random.Random(42)
    activation_ema = torch.ones(5, device=device) / 5
    routing_correct = 0
    routing_total = 0
    motor_counts = Counter()
    eval_records = []

    log(f"Steps: {MAX_STEPS}  grad_accum: {GRAD_ACCUM}  LR: {LR}  warmup: {WARMUP}")
    log(f"Routing_W: {ROUTING_W}  Balance_W: {BALANCE_W}  patience: {PATIENCE}\n")

    for step in range(start_step + 1, MAX_STEPS + 1):
        pipeline.train()
        optimizer.zero_grad()
        accum_lm = 0; accum_route = 0

        for _ in range(GRAD_ACCUM):
            ids, did = train_rng.choice(train_ids)
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = pipeline(ids_t)
                lm = F.cross_entropy(out.logits[0, :-1], ids_t[0, 1:], ignore_index=0) / GRAD_ACCUM

                concepts = pipeline.encoder(ids_t)
                pooled = concepts.mean(1).mean(0, keepdim=True)
                orch_log = pipeline.orchestrator.classifier(pooled)
                dn = ["cora","forge_c","axiom","muse","empathy","general"][min(did,5)]
                tgt = DOMAIN_TO_MOTOR.get(dn, 0)
                rl = F.cross_entropy(orch_log, torch.tensor([tgt], device=device)) / GRAD_ACCUM

                probs = F.softmax(orch_log.squeeze(0), dim=-1)
                bl = 5.0 * (activation_ema * probs).sum() / GRAD_ACCUM

                loss = lm + ROUTING_W * rl + BALANCE_W * bl

            if not math.isfinite(loss.item()):
                continue
            scaler.scale(loss).backward()
            accum_lm += lm.item() * GRAD_ACCUM
            accum_route += rl.item() * GRAD_ACCUM

            activation_ema = 0.99 * activation_ema + 0.01 * probs.detach()
            for m in out.active_motors: motor_counts[m] += 1
            exp = ["cora","forge_c","muse","axiom","empathy","cora"][min(did,5)]
            if exp in out.active_motors: routing_correct += 1
            routing_total += 1

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(pipeline.parameters(), CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 50 == 0:
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            eta = (MAX_STEPS - step) / sps / 60
            lr_now = scheduler.get_last_lr()[0]
            racc = 100 * routing_correct / max(1, routing_total)
            log(f"  step {step:>5}/{MAX_STEPS}  lm={accum_lm/GRAD_ACCUM:.3f} "
                f"route={accum_route/GRAD_ACCUM:.3f}  acc={racc:.0f}%  "
                f"lr={lr_now:.1e}  {sps:.2f}sps  ETA {eta:.0f}m  "
                f"VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

        # Eval
        if step % EVAL_EVERY == 0:
            pipeline.eval()
            vls = []
            with torch.no_grad():
                for vi in val_ids[:80]:
                    vt = torch.tensor([vi], dtype=torch.long, device=device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        vo = pipeline(vt)
                    vl = F.cross_entropy(vo.logits[0,:-1], vt[0,1:], ignore_index=0)
                    if math.isfinite(vl.item()): vls.append(vl.item())
            val_loss = sum(vls)/len(vls) if vls else float("nan")

            # Decode
            print(f"\n  --- Eval step {step} --- val_loss={val_loss:.4f}")
            for prompt, domain, kw in EVAL_PROMPTS:
                enc = encode(prompt, 96)
                cur = torch.tensor([enc], dtype=torch.long, device=device)
                pl = len(enc)
                with torch.no_grad():
                    for _ in range(48):
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            o = pipeline(cur)
                        nxt = int(o.logits[0,-1].float().argmax().item())
                        if nxt in (0,2): break
                        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], 1)
                        if cur.shape[1]-pl >= 3:
                            try:
                                ts = tok.decode([nxt])
                                if ts.rstrip().endswith(('.','?','!')): break
                            except: pass
                        if cur.shape[1] >= 128: break
                pred = tok.decode(cur[0,pl:].tolist()) if cur.shape[1] > pl else ""
                with torch.no_grad():
                    o2 = pipeline(torch.tensor([enc], dtype=torch.long, device=device))
                    routed = o2.active_motors[0] if o2.active_motors else "?"
                ps = pred[:55].encode("ascii","replace").decode()
                print(f"    [{domain:>8}->{routed:>8}] {prompt[:30]:<30} -> {ps}")

            racc = routing_correct / max(1, routing_total)
            print(f"  route_acc={100*racc:.0f}%  motors={dict(motor_counts)}\n")

            eval_records.append({"step": step, "val_loss": round(val_loss, 4), "route_acc": round(racc, 4)})

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                no_improve = 0
                torch.save({"model_state": pipeline.state_dict(), "config_name": "1b_4090",
                            "step": step, "val_loss": val_loss}, str(CKPT))
                log(f"  Saved best (val_loss={val_loss:.4f})")
            else:
                no_improve += EVAL_EVERY
            if no_improve >= PATIENCE:
                log(f"Early stop at step {step}")
                break

    elapsed = time.perf_counter() - t0
    log(f"\nDONE in {elapsed/60:.0f} min. Best val_loss={best_val:.4f}")
    log(f"Checkpoint: {CKPT}")
    with open(str(ROOT / "experiments" / "train_1b_results.json"), "w") as f:
        json.dump(eval_records, f, indent=2)

if __name__ == "__main__":
    main()
