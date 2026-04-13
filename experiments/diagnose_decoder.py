#!/usr/bin/env python3
"""
diagnose_decoder.py — Minimal test: can the decoder learn 10 trivial examples?
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn.functional as F

EXAMPLES = [
    ("hola", "hola, como estas?"),
    ("como estas?", "bien, y tu?"),
    ("que haces?", "nada, aqui pensando"),
    ("quien eres?", "soy AION-C"),
    ("2+2", "4"),
    ("que es python?", "un lenguaje de programacion"),
    ("gracias", "de nada"),
    ("adios", "hasta luego"),
    ("hola en ingles", "hello"),
    ("que dia es hoy?", "no lo se"),
]

EOS = 2  # BPE EOS token id

def encode(tok, text, ml=64):
    try: ids = tok.encode(text, ml)
    except TypeError: ids = tok.encode(text)[:ml]
    return ids

def encode_with_eos(tok, inp, out, ml=64):
    """Encode as: [input tokens] [output tokens] [EOS]"""
    try:
        inp_ids = tok.encode(inp, ml // 2)
        out_ids = tok.encode(out, ml // 2)
    except TypeError:
        inp_ids = tok.encode(inp)[:ml // 2]
        out_ids = tok.encode(out)[:ml // 2]
    return inp_ids + out_ids + [EOS]

def decode_greedy(model, tok, prompt, device, max_new=20, use_pipeline=False):
    model.eval()
    ids = encode(tok, prompt)
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    plen = len(ids)
    with torch.no_grad():
        for _ in range(max_new):
            if use_pipeline:
                out = model(cur)
                logits = out.logits
            else:
                concepts = model.encoder(cur)
                D = concepts.shape[-1]
                graph = torch.zeros(1, 8, D, device=device)
                dec_out = model.decoder(cur, graph, encoder_concepts=concepts)
                logits = dec_out.logits
            nxt = int(logits[0, -1].argmax().item())
            if nxt in (0, 2): break
            cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
            if cur.shape[1] >= 64: break
    gen_ids = cur[0, plen:].tolist()
    try: return tok.decode(gen_ids)
    except: return ""

def run_test(name, model, tok, device, use_pipeline=False, steps=500, use_eos=False):
    print(f"\n{'='*55}")
    print(f"  TEST: {name} {'(+EOS)' if use_eos else '(no EOS)'}")
    print(f"{'='*55}")

    # Tokenize
    train_pairs = []
    for inp, out in EXAMPLES:
        if use_eos:
            ids = encode_with_eos(tok, inp, out)
        else:
            full = inp + " " + out
            ids = encode(tok, full)
        train_pairs.append(ids)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    model.train()

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        ids = train_pairs[step % len(train_pairs)]
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)

        if use_pipeline:
            out = model(ids_t)
            logits = out.logits
        else:
            concepts = model.encoder(ids_t)
            D = concepts.shape[-1]
            graph = torch.zeros(1, 8, D, device=device)
            dec_out = model.decoder(ids_t, graph, encoder_concepts=concepts)
            logits = dec_out.logits

        loss = F.cross_entropy(logits[0, :-1], ids_t[0, 1:], ignore_index=0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            print(f"  step {step:>4}  loss={loss.item():.4f}")

    elapsed = time.perf_counter() - t0
    print(f"  Training: {elapsed:.1f}s")

    # Test all 10 examples
    print(f"\n  Results:")
    correct = 0
    for inp, expected in EXAMPLES:
        pred = decode_greedy(model, tok, inp, device, use_pipeline=use_pipeline)
        match = expected.lower().strip() == pred.lower().strip()
        contains = expected.lower() in pred.lower() or pred.lower() in expected.lower()
        if match: correct += 1
        status = "EXACT" if match else "PARTIAL" if contains else "WRONG"
        print(f"    [{status:>7}] '{inp}' -> '{pred[:40]}' (expected: '{expected[:30]}')")

    print(f"\n  Exact matches: {correct}/10")
    return correct

def main():
    device = torch.device("cpu")
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

    # TEST 1: No EOS (baseline — reproduces the problem)
    pipeline1 = MoSEPipeline(cfg).to(device)
    c1 = run_test("PURE DECODER no EOS", pipeline1, tok, device,
                   use_pipeline=False, steps=500, use_eos=False)

    # TEST 2: WITH EOS (the fix)
    pipeline2 = MoSEPipeline(cfg).to(device)
    c2 = run_test("PURE DECODER +EOS", pipeline2, tok, device,
                   use_pipeline=False, steps=500, use_eos=True)

    # TEST 3: Full pipeline + EOS
    pipeline3 = MoSEPipeline(cfg).to(device)
    c3 = run_test("FULL PIPELINE +EOS", pipeline3, tok, device,
                   use_pipeline=True, steps=500, use_eos=True)

    print(f"\n{'='*55}")
    print(f"  DIAGNOSIS")
    print(f"{'='*55}")
    print(f"  No EOS 500 steps:        {c1}/10 exact")
    print(f"  +EOS 500 steps:          {c2}/10 exact")
    print(f"  +EOS full pipeline:      {c3}/10 exact")
    if c2 > c1:
        print(f"\n  EOS FIX WORKS! ({c1} -> {c2} exact matches)")
    else:
        print(f"\n  EOS didn't help — deeper issue")

if __name__ == "__main__":
    main()
