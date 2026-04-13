#!/usr/bin/env python3
"""
auto_learn_demo.py — End-to-end self-learning loop on the tiny 5.5M model
==========================================================================
Flow:
  1. Pre-train tiny model on 10 known examples (the ones from diagnose_decoder.py)
  2. Verify baseline 10/10
  3. Ask "what is javascript?" — model does NOT know
  4. Detect ignorance (low first-token confidence + gibberish heuristic)
  5. "Search" the local knowledge base for an answer
  6. Spawn a BACKGROUND thread that fine-tunes ONLY the last decoder layer
     on (new example + rehearsal of the original 10) to avoid forgetting.
     Main thread keeps answering known queries while training runs.
  7. Re-ask "what is javascript?" — should now answer correctly
  8. Re-verify all 10 originals — should still be 10/10
"""
import sys, os, time, random, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn.functional as F

EOS = 2

# 10 known training examples (from diagnose_decoder.py — proven 10/10 with EOS)
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

# Local knowledge base — simulates an internet search lookup
KNOWLEDGE_BASE = {
    "javascript": "JavaScript is a programming language used for web development.",
    "rust":       "Rust is a systems programming language focused on safety.",
    "html":       "HTML is the standard markup language for web pages.",
    "css":        "CSS is a style sheet language used for web design.",
    "git":        "Git is a distributed version control system.",
}


# ── Tokenization helpers ────────────────────────────────────────────────────

def encode(tok, text, ml=64):
    try: return tok.encode(text, ml)
    except TypeError: return tok.encode(text)[:ml]

def encode_pair_with_eos(tok, inp, out):
    try:
        ip = tok.encode(inp, 32)
        op = tok.encode(out, 32)
    except TypeError:
        ip = tok.encode(inp)[:32]
        op = tok.encode(out)[:32]
    return ip + op + [EOS]


# ── Model build ─────────────────────────────────────────────────────────────

def build_tiny():
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
    return MoSEPipeline(cfg), tok


# ── Pre-training (use the proven recipe from diagnose_decoder.py) ───────────

def pretrain(model, tok, steps=500):
    pairs = [encode_pair_with_eos(tok, i, o) for i, o in EXAMPLES]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    model.train()
    for step in range(1, steps + 1):
        ids = pairs[step % len(pairs)]
        ids_t = torch.tensor([ids], dtype=torch.long)
        out = model(ids_t)
        loss = F.cross_entropy(out.logits[0, :-1], ids_t[0, 1:], ignore_index=0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"  pretrain {step}/{steps}  loss={loss.item():.4f}")


# ── Inference with confidence ───────────────────────────────────────────────

@torch.no_grad()
def query(model, tok, prompt, max_new=24):
    """Greedy decode + return (text, mean_first5_confidence)."""
    model.eval()
    ids = encode(tok, prompt)
    cur = torch.tensor([ids], dtype=torch.long)
    plen = len(ids)
    confs = []
    for _ in range(max_new):
        out = model(cur)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        top_p, top_i = probs.max(0)
        confs.append(float(top_p))
        nxt = int(top_i)
        if nxt in (0, EOS): break
        cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
        if cur.shape[1] >= 64: break
    text = tok.decode(cur[0, plen:].tolist()) if cur.shape[1] > plen else ""
    mean_conf = sum(confs[:5]) / max(1, len(confs[:5]))
    return text, mean_conf


def looks_like_gibberish(text):
    s = text.strip()
    if not s or len(s) < 2: return True
    words = s.split()
    if len(words) >= 3 and len(set(words)) <= 1: return True
    return False


def matches(pred, expected):
    p, e = pred.lower().strip(), expected.lower().strip()
    return e in p or p in e


# ── "Internet search" ───────────────────────────────────────────────────────

def search_knowledge(query_text):
    q = query_text.lower()
    for keyword, fact in KNOWLEDGE_BASE.items():
        if keyword in q:
            return fact
    return None


# ── Background fine-tune: last decoder layer only + rehearsal ───────────────

def background_finetune(model, tok, new_pair, rehearsal_pairs, warmup_steps, mix_steps, lr, lock, status):
    """Curriculum + Anti-forgetting (Punto 14):
       Phase A (warmup):  pure new example, ALL params unfrozen
       Phase B (rehearse): mix new + rehearsal, RollbackManager snapshot before
                           starting; ExamRunner valida después; rollback si baja
                           >2% el score en los rehearsal_pairs
       The whole tiny model is trainable; rehearsal handles forgetting.
    """
    # Punto 14: anti-forgetting layers
    from training.anti_forgetting import (
        RollbackManager, ExamRunner, ExamItem, should_rollback,
    )

    rb = RollbackManager(model)
    rb.snapshot()  # snapshot pre-training

    # Exam con los rehearsal pairs (los 10 originales)
    def _generator(query, item):
        # genera con el modelo actual usando query() ya definido en el módulo
        text, _ = query(model, tok, query.__defaults__[0] if False else item.query, max_new=12)
        return text
    # Construir ExamItems desde los rehearsal pairs
    exam_items = [
        ExamItem(query=inp, expected=out)
        for inp, out in rehearsal_pairs
    ]

    # Score baseline (antes del fine-tune)
    def _gen(q, item):
        text, _ = query(model, tok, q, max_new=12)
        return text
    baseline_exam = ExamRunner(exam_items, _gen).run()
    status['baseline_exam'] = baseline_exam.score

    for p in model.parameters():
        p.requires_grad = True
    trainable = list(model.parameters())
    status['trainable'] = sum(p.numel() for p in trainable)
    status['stage'] = 'warmup'

    new_ids = encode_pair_with_eos(tok, *new_pair)
    rehearsal_ids = [encode_pair_with_eos(tok, i, o) for i, o in rehearsal_pairs]

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0)
    rng = random.Random(0)
    new_t = torch.tensor([new_ids], dtype=torch.long)

    # ── Phase A: warmup on the new fact alone ────────────────────────────
    for step in range(1, warmup_steps + 1):
        with lock:
            model.train()
            out = model(new_t)
            loss = F.cross_entropy(out.logits[0, :-1], new_t[0, 1:], ignore_index=0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            model.eval()
        status['step'] = step
        status['loss'] = float(loss.item())
        status['phase'] = 'A:warmup'
        time.sleep(0.003)

    # ── Phase B: mix new fact with rehearsal of the original 10 ─────────
    status['stage'] = 'rehearse'
    for step in range(1, mix_steps + 1):
        # 50/50 mix; rehearsal keeps old skills alive without erasing the fact
        ids = new_ids if (rng.random() < 0.5) else rng.choice(rehearsal_ids)
        ids_t = torch.tensor([ids], dtype=torch.long)
        with lock:
            model.train()
            out = model(ids_t)
            loss = F.cross_entropy(out.logits[0, :-1], ids_t[0, 1:], ignore_index=0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            model.eval()
        status['step'] = warmup_steps + step
        status['loss'] = float(loss.item())
        status['phase'] = 'B:rehearse'
        time.sleep(0.003)

    # Punto 14: post-training exam + rollback si bajó >2%
    model.eval()
    after_exam = ExamRunner(exam_items, _gen).run()
    status['after_exam']  = after_exam.score
    status['exam_delta']  = after_exam.score - baseline_exam.score
    if should_rollback(baseline_exam.score, after_exam.score, max_drop=0.02):
        rb.rollback()
        status['rollback'] = True
        status['stage'] = 'rolled_back'
    else:
        status['rollback'] = False
        status['stage'] = 'done'


# ── Main demo ───────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  AION-C AUTO-LEARN DEMO  (tiny 5.5M, CPU)")
    print("=" * 64)

    print("\n[1/7] Build + pre-train tiny model on 10 examples...")
    model, tok = build_tiny()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")
    pretrain(model, tok, steps=500)

    print("\n[2/7] Verify baseline (expect 10/10)...")
    base_correct = 0
    for inp, expected in EXAMPLES:
        pred, conf = query(model, tok, inp)
        ok = matches(pred, expected)
        base_correct += int(ok)
        print(f"  [{'OK' if ok else 'X '}] '{inp:<14}' -> '{pred[:35]:<35}' conf={conf:.2f}")
    print(f"  Baseline: {base_correct}/10")

    UNKNOWN = "what is javascript?"
    print(f"\n[3/7] Ask unknown: '{UNKNOWN}'")
    pred, conf = query(model, tok, UNKNOWN, max_new=24)
    print(f"  Response: '{pred[:60]}'")
    print(f"  Mean confidence (first 5 tokens): {conf:.3f}")

    CONF_THRESHOLD = 0.40
    knows = (conf >= CONF_THRESHOLD) and not looks_like_gibberish(pred) and "javascript" in pred.lower()
    print(f"  Detection: {'KNOWS' if knows else 'DOES NOT KNOW'} (threshold={CONF_THRESHOLD})")
    if knows:
        print("  Model already knows. Demo aborts.")
        return

    print(f"\n[4/7] Auto-search knowledge base for '{UNKNOWN}'...")
    answer = search_knowledge(UNKNOWN)
    if answer is None:
        print("  No KB hit. Demo aborts.")
        return
    print(f"  Found: '{answer}'")

    print(f"\n[5/7] Spawn BACKGROUND fine-tune (full model + curriculum)...")
    new_pair = (UNKNOWN, answer)
    lock = threading.RLock()
    status = {'step': 0, 'stage': 'starting', 'loss': float('nan'), 'trainable': 0, 'phase': '-'}
    WARMUP, MIX, LR = 80, 200, 1e-3
    TOTAL = WARMUP + MIX
    t = threading.Thread(
        target=background_finetune,
        args=(model, tok, new_pair, EXAMPLES, WARMUP, MIX, LR, lock, status),
        daemon=True,
    )
    t.start()
    time.sleep(0.3)
    print(f"  Trainable params: {status['trainable']:,} of {n_params:,}"
          f"  ({100*status['trainable']/n_params:.1f}%)")
    print(f"  Curriculum: warmup={WARMUP} steps on new fact, then {MIX} steps mixed rehearsal")
    print(f"  Main thread keeps answering 'hola' WHILE training runs:")
    last_logged = -1
    while t.is_alive():
        with lock:
            pred, conf = query(model, tok, "hola", max_new=10)
        s = status['step']
        if s != last_logged and s % 10 == 0:
            print(f"    [{status['phase']} step {s:>3}/{TOTAL} loss={status['loss']:.4f}]"
                  f" 'hola' -> '{pred[:30]}'")
            last_logged = s
        time.sleep(0.3)
        if status['stage'] == 'done': break
    t.join()
    print(f"  Background training done.")

    print(f"\n[6/7] Re-ask '{UNKNOWN}' after auto-learn...")
    pred, conf = query(model, tok, UNKNOWN, max_new=32)
    print(f"  Response: '{pred[:80]}' conf={conf:.2f}")
    learned = any(w in pred.lower() for w in ["javascript", "programming", "web"])
    print(f"  Learned: {'YES' if learned else 'NO'}")

    print(f"\n[7/7] Verify NO catastrophic forgetting on original 10...")
    after_correct = 0
    for inp, expected in EXAMPLES:
        pred, conf = query(model, tok, inp)
        ok = matches(pred, expected)
        after_correct += int(ok)
        flag = 'OK' if ok else 'X '
        print(f"  [{flag}] '{inp:<14}' -> '{pred[:35]:<35}'")
    print(f"  After-learn: {after_correct}/10  (baseline was {base_correct}/10)")

    print("\n" + "=" * 64)
    print(f"  RESULT")
    print(f"    baseline:        {base_correct}/10")
    print(f"    learned new fact: {'YES' if learned else 'NO'}")
    print(f"    retained old:    {after_correct}/10")
    if learned and after_correct >= base_correct - 1:
        print("  >>> END-TO-END AUTO-LEARN SUCCESS <<<")
    else:
        print("  >>> PARTIAL — see metrics above <<<")
    print("=" * 64)


if __name__ == "__main__":
    main()
