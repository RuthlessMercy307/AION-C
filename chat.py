#!/usr/bin/env python3
"""
chat.py — AION-C Interactive Chat with routing, MEM, and graph display
======================================================================

Features:
  - Routing visible: shows which motor handles each query
  - MEM active: searches before answering, stores after
  - Bilingual ES+EN
  - Commands: /mem, /graph, /route, /eval, /stats, /quit

Usage:
    cd AION-C
    python chat.py                          # tiny model, CPU
    python chat.py --checkpoint model.pt    # load trained model
    python chat.py --eval                   # run benchmark evaluation only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch
import torch.nn.functional as F

from router.pipeline import MoSEPipeline, MoSEConfig
from experiments.train_production import build_tokenizer
from memory.semantic_store import SemanticStore


# ─── Eval prompts ────────────────────────────────────────────────────────────
EVAL_PROMPTS = [
    ("Hola, quien eres?", "general", "identity/greeting"),
    ("If rain causes wet soil, does rain cause floods?", "cora", "causal chain"),
    ("Write a Python function to reverse a linked list", "forge_c", "code generation"),
    ("What is 15% of 240?", "axiom", "arithmetic"),
    ("Mi amigo esta triste porque perdio su trabajo", "empathy", "emotional support"),
    ("Write a short scene: a robot discovers music for the first time", "muse", "creative writing"),
]


def build_model(checkpoint: Optional[str] = None, device_str: str = "auto"):
    """Build model, tokenizer, and MEM."""
    device = torch.device("cuda" if device_str == "auto" and torch.cuda.is_available()
                          else device_str if device_str != "auto" else "cpu")

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

    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        pipeline.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        print(f"  Loaded checkpoint: {checkpoint}")

    # Initialize MEM with encoder
    mem = SemanticStore(
        encoder=pipeline.encoder,
        tokenizer=tok,
        similarity_threshold=0.3,
    )
    mem._device = device

    # Seed MEM with basic facts
    mem.store("identity", "I am AION-C, an AI created by Jesus with MoSE architecture.", "general")
    mem.store("architecture", "MoSE has 5 motors: CORA causal, FORGE-C code, AXIOM math, MUSE creative, EMPATHY social.", "general")
    mem.store("limitations", "I cannot see images, audio, or video. I only work with text.", "general")

    params = sum(p.numel() for p in pipeline.parameters())
    print(f"  Model: {params:,} params on {device}")
    print(f"  MEM: {len(mem)} entries")

    return pipeline, tok, mem, cfg, device


def greedy_decode(pipeline, tok, text, device, max_new=48, mem_context=""):
    """Generate response with early stop on punctuation."""
    full_text = text
    if mem_context:
        full_text = mem_context + "\n" + text

    try:
        ids = tok.encode(full_text, 96)
    except TypeError:
        ids = tok.encode(full_text)[:96]

    plen = len(ids)
    cur = torch.tensor([ids], dtype=torch.long, device=device)

    pipeline.eval()
    with torch.no_grad():
        # Get routing info from first forward
        out = pipeline(cur)
        active_motors = list(out.active_motors)
        scores = out.orchestrator.scores.tolist() if hasattr(out.orchestrator, 'scores') else []

        # Generate tokens
        for _ in range(max_new):
            nxt = int(out.logits[0, -1].argmax().item())
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
            if cur.shape[1] < 128:
                out = pipeline(cur)

    pred_ids = cur[0, plen:].tolist()
    try:
        response = tok.decode(pred_ids)
    except Exception:
        response = ""

    return response, active_motors, scores


def run_eval(pipeline, tok, mem, device):
    """Run benchmark evaluation on fixed prompts."""
    print("\n" + "=" * 65)
    print("  AION-C Evaluation — Tiny Model Baseline")
    print("=" * 65)

    motor_names = ["cora", "forge_c", "muse", "axiom", "empathy"]
    routing_correct = 0

    for prompt, expected_domain, desc in EVAL_PROMPTS:
        # Search MEM
        mem_ctx = mem.search_as_context(prompt, top_k=2)

        response, motors, scores = greedy_decode(pipeline, tok, prompt, device, mem_context=mem_ctx)
        routed_to = motors[0] if motors else "?"

        # Routing accuracy
        domain_to_motor = {"cora": "cora", "forge_c": "forge_c", "axiom": "axiom",
                          "muse": "muse", "empathy": "empathy", "general": "cora"}
        expected_motor = domain_to_motor.get(expected_domain, "cora")
        correct = expected_motor in motors
        if correct:
            routing_correct += 1

        # Format scores
        score_str = " ".join(f"{motor_names[i]}={s:.2f}" for i, s in enumerate(scores[:5]))

        print(f"\n  [{desc}]")
        print(f"  Q: {prompt[:60]}")
        print(f"  A: {response[:80] if response else '(empty)'}")
        print(f"  Route: {routed_to} {'OK' if correct else 'MISS'} | Scores: {score_str}")
        if mem_ctx:
            print(f"  MEM: {mem_ctx[:60]}")

    total = len(EVAL_PROMPTS)
    print(f"\n  Routing accuracy: {routing_correct}/{total} ({100*routing_correct/total:.0f}%)")
    print(f"  MEM entries: {len(mem)}")
    print("=" * 65)


def chat_loop(pipeline, tok, mem, device):
    """Interactive chat with commands."""
    history: List[Tuple[str, str]] = []
    motor_names = ["cora", "forge_c", "muse", "axiom", "empathy"]
    last_motors = []
    last_scores = []

    print("\n" + "=" * 65)
    print("  AION-C Interactive Chat")
    print("  Commands: /mem /route /stats /eval /quit")
    print("  Type naturally in English or Spanish")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if user_input.lower() == "/mem":
            entries = mem.list_entries()
            print(f"\n  MEM ({len(entries)} entries):")
            for e in entries:
                print(f"    [{e['domain']:>8}] {e['key']}: {e['value']}")
            print()
            continue

        if user_input.lower() == "/route":
            if last_scores:
                print(f"\n  Last routing scores:")
                for i, s in enumerate(last_scores[:5]):
                    bar = "#" * int(s * 40)
                    print(f"    {motor_names[i]:>8}: {s:.3f} |{bar}")
                print(f"  Active: {last_motors}")
            else:
                print("  No routing data yet.")
            print()
            continue

        if user_input.lower() == "/stats":
            stats = mem.stats()
            params = sum(p.numel() for p in pipeline.parameters())
            print(f"\n  Model: {params:,} params")
            print(f"  MEM: {stats}")
            print(f"  History: {len(history)} turns")
            print()
            continue

        if user_input.lower() == "/eval":
            run_eval(pipeline, tok, mem, device)
            continue

        # ── Generate response ──
        # 1. Search MEM
        mem_ctx = mem.search_as_context(user_input, top_k=2)

        # 2. Build context from history
        ctx_parts = []
        if mem_ctx:
            ctx_parts.append(mem_ctx)
        for q, a in history[-3:]:
            ctx_parts.append(f"Q: {q}\nA: {a}")
        context = "\n".join(ctx_parts)

        full_input = user_input
        if context:
            full_input = context + "\n" + user_input

        # 3. Generate
        response, motors, scores = greedy_decode(pipeline, tok, full_input, device)
        last_motors = motors
        last_scores = scores

        # 4. Display
        motor_str = ", ".join(motors) if motors else "?"
        if response:
            print(f"AION-C [{motor_str}]: {response}")
        else:
            print(f"AION-C [{motor_str}]: (no response generated)")

        # Show MEM hit if found
        if mem_ctx:
            print(f"  [MEM found: {mem_ctx[:50]}...]")

        # 5. Learn from interaction
        if response and len(response.split()) > 3:
            # Auto-learn: store the Q&A if it seems factual
            key = f"qa_{len(history)}"
            mem.learn(key, f"Q: {user_input[:50]} A: {response[:50]}",
                     domain=motors[0] if motors else "general")

        history.append((user_input, response))
        print()


def main():
    parser = argparse.ArgumentParser(description="AION-C Interactive Chat")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    print("Loading AION-C...")
    pipeline, tok, mem, cfg, device = build_model(args.checkpoint, args.device)

    if args.eval:
        run_eval(pipeline, tok, mem, device)
    else:
        # Run eval first to show baseline
        run_eval(pipeline, tok, mem, device)
        # Then enter chat
        chat_loop(pipeline, tok, mem, device)


if __name__ == "__main__":
    main()
