#!/usr/bin/env python3
"""
diagnose_1b.py — Compare decoder-only vs full pipeline vs sampling
===================================================================
Tests the 1.1B checkpoint 3 ways to find where generation breaks.
"""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTHONIOENCODING"] = "utf-8"

PROMPTS = [
    ("Hola, quien eres?", "general"),
    ("If rain causes wet soil, does rain cause floods?", "cora"),
    ("Write a Python function to reverse a linked list", "forge_c"),
    ("What is 15% of 240?", "axiom"),
    ("Mi amigo esta triste porque perdio su trabajo", "empathy"),
    ("Write a short scene: a robot discovers music", "muse"),
]

def build_model():
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

    ckpt_path = "checkpoints/aion_1b.pt"
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pipeline.load_state_dict(ckpt["model_state"], strict=False)
    print(f"Loaded step={ckpt.get('step','?')}, val_loss={ckpt.get('val_loss','?')}")

    return pipeline, tok, cfg


def encode(tok, text, ml=96):
    try: return tok.encode(text, ml)
    except TypeError: return tok.encode(text)[:ml]


def decode_pure_decoder(pipeline, tok, prompt, max_new=50):
    """Mode 1: Encoder → Decoder ONLY, no CRE/motors."""
    pipeline.eval()
    ids = encode(tok, prompt)
    cur = torch.tensor([ids])
    plen = len(ids)

    with torch.no_grad():
        for _ in range(max_new):
            concepts = pipeline.encoder(cur)
            D = concepts.shape[-1]
            K = 8  # motor_max_nodes
            graph = torch.zeros(1, K, D)
            dec_out = pipeline.decoder(cur, graph, encoder_concepts=concepts)
            nxt = int(dec_out.logits[0, -1].argmax().item())
            if nxt in (0, 2): break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
            if cur.shape[1] >= 128: break

    return tok.decode(cur[0, plen:].tolist()) if cur.shape[1] > plen else "(empty)"


def decode_full_pipeline(pipeline, tok, prompt, max_new=50):
    """Mode 2: Full pipeline with CRE and motors (greedy)."""
    pipeline.eval()
    ids = encode(tok, prompt)
    cur = torch.tensor([ids])
    plen = len(ids)

    with torch.no_grad():
        for _ in range(max_new):
            out = pipeline(cur)
            nxt = int(out.logits[0, -1].argmax().item())
            if nxt in (0, 2): break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
            if cur.shape[1] >= 128: break

    motors = list(out.active_motors) if hasattr(out, 'active_motors') else []
    return tok.decode(cur[0, plen:].tolist()) if cur.shape[1] > plen else "(empty)", motors


def decode_sampling(pipeline, tok, prompt, max_new=50, temperature=0.3, top_p=0.9):
    """Mode 3: Full pipeline with temperature + top-p sampling."""
    pipeline.eval()
    ids = encode(tok, prompt)
    cur = torch.tensor([ids])
    plen = len(ids)

    with torch.no_grad():
        for _ in range(max_new):
            out = pipeline(cur)
            logits = out.logits[0, -1].float() / temperature

            # Top-p filtering
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum - probs > top_p
            sorted_logits[mask] = float('-inf')
            probs = torch.softmax(sorted_logits, dim=-1)

            nxt_idx = torch.multinomial(probs, 1).item()
            nxt = sorted_idx[nxt_idx].item()

            if nxt in (0, 2): break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
            if cur.shape[1] >= 128: break

    return tok.decode(cur[0, plen:].tolist()) if cur.shape[1] > plen else "(empty)"


def main():
    pipeline, tok, cfg = build_model()

    print("\n" + "=" * 90)
    print("  DIAGNOSIS: 1.1B Checkpoint — 3 Decode Modes")
    print("=" * 90)

    # Header
    print(f"\n{'Prompt':<45} | {'Mode':<12} | Response")
    print("-" * 120)

    for prompt, domain in PROMPTS:
        short_p = prompt[:42] + "..." if len(prompt) > 42 else prompt

        # Mode 1: Pure decoder
        r1 = decode_pure_decoder(pipeline, tok, prompt)

        # Mode 2: Full pipeline greedy
        r2, motors = decode_full_pipeline(pipeline, tok, prompt)
        motor_str = motors[0] if motors else "?"

        # Mode 3: Sampling
        r3 = decode_sampling(pipeline, tok, prompt)

        print(f"{short_p:<45} | {'1.Decoder':<12} | {r1[:60]}")
        print(f"{'':45} | {'2.Pipeline':<12} | {r2[:60]} [{motor_str}]")
        print(f"{'':45} | {'3.Sample':<12} | {r3[:60]}")
        print()

    # Summary
    print("=" * 90)
    print("  INTERPRETATION:")
    print("  If Mode 1 >> Mode 2: CRE/motors interfere with decoder output")
    print("  If Mode 3 >> Mode 2: Greedy decoding is the problem, sampling helps")
    print("  If all 3 bad:        Model needs more training data or steps")
    print("=" * 90)


if __name__ == "__main__":
    main()
