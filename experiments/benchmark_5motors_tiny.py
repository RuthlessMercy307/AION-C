"""
experiments/benchmark_5motors_tiny.py
======================================
Benchmark rápido de los 5 motores MoSE.

Config idéntica al benchmark exitoso de CORA:
  - hidden_dim=64, vocab pequeño construido del dataset
  - batch=1, pipeline() directo (NO PyGStyleBatcher)
  - 2000 steps por motor, nivel 1 (respuestas cortas)
  - Mismo forward path en train e inference → sin mismatch

Objetivo: demostrar que los 5 motores APRENDEN (Word F1 > 0.3 en train data).
Tiempo esperado: ~60-90s por motor, ~5-8 min total.

Uso:
    cd AION-C
    python -m experiments.benchmark_5motors_tiny
"""
from __future__ import annotations

import sys, os, math, random, re as _re, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from router.pipeline           import MoSEPipeline, MoSEConfig
from synth.causal_graph_gen    import CausalGraphGenerator
from synth.code_graph_gen      import CodeGraphGenerator
from synth.math_graph_gen      import MathGraphGenerator
from synth.narrative_graph_gen import NarrativeGraphGenerator
from synth.social_graph_gen    import SocialGraphGenerator
from orchestrator.model        import MOTOR_NAMES

torch.set_num_threads(4)
torch.manual_seed(42)
random.seed(42)

PAD, BOS, EOS = 0, 1, 2
VOCAB_SIZE  = 256
MAX_LEN     = 64
N_EXAMPLES  = 500   # ejemplos generados por motor
N_STEPS     = 2000  # steps de entrenamiento por motor
N_EVAL      = 20    # ejemplos de evaluación (nuevos, del generador)
LR          = 3e-4
PRINT_EVERY = 200

GENS = {
    "cora":    CausalGraphGenerator(),
    "forge_c": CodeGraphGenerator(),
    "axiom":   MathGraphGenerator(),
    "muse":    NarrativeGraphGenerator(),
    "empathy": SocialGraphGenerator(),
}
HINTS = {
    "cora":    "",
    "forge_c": "function ",
    "axiom":   "theorem ",
    "muse":    "historia ",
    "empathy": "siente ",
}


# ─── Tokenizador word-level mínimo ───────────────────────────────────────────

class Tokenizer:
    def __init__(self, vocab_size: int = VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size
        self._w2i: Dict[str, int] = {}
        self._i2w: List[str] = []

    def build(self, texts: List[str]) -> "Tokenizer":
        freq: Counter = Counter()
        for t in texts:
            freq.update(_re.findall(r"\w+", t.lower()))
        cap = self.vocab_size - 3  # reserva PAD BOS EOS
        self._i2w = [w for w, _ in freq.most_common(cap)]
        self._w2i = {w: i + 3 for i, w in enumerate(self._i2w)}
        return self

    def encode(self, text: str) -> List[int]:
        toks = _re.findall(r"\w+", text.lower())
        ids = [BOS] + [self._w2i.get(t, 2) for t in toks] + [EOS]
        return ids[:MAX_LEN]

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i == EOS:
                break
            if i >= 3:
                idx = i - 3
                out.append(self._i2w[idx] if idx < len(self._i2w) else "<unk>")
        return " ".join(out)

    def to_tensor(self, ids: List[int]) -> torch.Tensor:
        return torch.tensor([ids], dtype=torch.long)   # [1, L]


# ─── Word F1 ─────────────────────────────────────────────────────────────────

def word_f1(pred: str, ref: str) -> float:
    p, r = set(pred.lower().split()), set(ref.lower().split())
    if not p or not r:
        return 0.0
    tp = len(p & r)
    pr = tp / len(p)
    rc = tp / len(r)
    return round(2 * pr * rc / (pr + rc), 3) if (pr + rc) else 0.0


# ─── Greedy decode (mismo pipeline() que en training) ────────────────────────

def greedy_decode(pipeline: MoSEPipeline, prompt: str, hint: str,
                  tok: Tokenizer, max_new: int = 20) -> str:
    pipeline.eval()
    ids  = tok.encode(prompt)
    cur  = tok.to_tensor(ids)
    with torch.no_grad():
        for _ in range(max_new):
            if cur.shape[1] >= MAX_LEN:
                break
            out = pipeline(cur, query_text=hint or None)
            nxt = out.logits[0, -1].argmax().item()
            if nxt == EOS:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    pipeline.train()
    return tok.decode(cur[0, len(ids):].tolist())


# ─── Benchmark de un motor ───────────────────────────────────────────────────

def run_motor(domain: str) -> Tuple[float, float, float]:
    """
    Entrena un pipeline fresco en el dominio y evalúa Word F1.
    Retorna (mean_f1_train, mean_f1_eval, elapsed_seconds).
    """
    SEP = "─" * 60
    print(f"\n{SEP}")
    print(f"  Motor: {domain.upper()}")
    print(SEP)

    gen  = GENS[domain]
    hint = HINTS[domain]

    # ── Generar ejemplos (solo nivel 1 — respuestas cortas) ─────────────────
    t0 = time.perf_counter()
    examples: List[Tuple[str, str]] = []
    for _ in range(N_EXAMPLES):
        ex = gen.generate(level=1)
        examples.append((ex.problem_text, ex.answer))

    # ── Tokenizador construido del dataset ───────────────────────────────────
    all_text = [q + " " + a for q, a in examples]
    tok = Tokenizer(VOCAB_SIZE).build(all_text)
    ids_list = [tok.encode(q + " " + a) for q, a in examples]
    print(f"  Vocab: {len(tok._i2w)} palabras | "
          f"{N_EXAMPLES} ejemplos generados en {time.perf_counter()-t0:.1f}s")

    # ── Pipeline fresh con hidden_dim=64 ────────────────────────────────────
    base = MoSEConfig.tiny()
    cfg  = MoSEConfig(
        hidden_dim       = 64,
        vocab_size       = VOCAB_SIZE,
        enc_n_layers     = base.enc_n_layers,
        enc_state_dim    = base.enc_state_dim,
        enc_expand       = base.enc_expand,
        enc_d_conv       = base.enc_d_conv,
        enc_ffn_mult     = base.enc_ffn_mult,
        orch_mlp_hidden  = 32,
        orch_max_motors  = base.orch_max_motors,
        orch_min_confidence = base.orch_min_confidence,
        motor_max_nodes  = base.motor_max_nodes,
        motor_n_heads    = 2,            # head_dim = 64/2 = 32, ok
        motor_threshold  = base.motor_threshold,
        unif_n_heads     = 2,
        dec_n_layers     = base.dec_n_layers,
        dec_n_heads      = 2,
        dec_max_seq_len  = MAX_LEN,
        dec_state_dim    = base.dec_state_dim,
        dec_expand       = base.dec_expand,
        dec_d_conv       = base.dec_d_conv,
        dec_ffn_mult     = base.dec_ffn_mult,
    )
    pipeline = MoSEPipeline(cfg)
    bd = pipeline.parameter_breakdown()
    print(f"  Params: {bd['total_unique']:,}")

    opt = torch.optim.AdamW(pipeline.parameters(), lr=LR, weight_decay=1e-2)
    pipeline.train()

    # ── Training loop: batch=1, pipeline() directo ───────────────────────────
    t_train = time.perf_counter()
    losses: List[float] = []
    for step in range(1, N_STEPS + 1):
        q, a = random.choice(examples)
        text  = q + " " + a
        ids   = tok.to_tensor(tok.encode(text))   # [1, L]

        out  = pipeline(ids, query_text=hint or None)
        loss = F.cross_entropy(
            out.logits[0, :-1],
            ids[0, 1:],
            ignore_index=PAD,
        )

        if math.isfinite(loss.item()):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        else:
            losses.append(float("nan"))

        if step % PRINT_EVERY == 0:
            valid = [x for x in losses[-PRINT_EVERY:] if math.isfinite(x)]
            avg = sum(valid) / len(valid) if valid else float("nan")
            elapsed = time.perf_counter() - t_train
            print(f"  step {step:>4}  loss={avg:.4f}  {elapsed:.0f}s", flush=True)

    t_elapsed = time.perf_counter() - t_train
    valid_losses = [x for x in losses if math.isfinite(x)]
    final_loss = sum(valid_losses[-20:]) / min(20, len(valid_losses))
    print(f"  Training done: {t_elapsed:.0f}s  final_loss={final_loss:.4f}")

    # ── Eval en TRAIN data (verifica que aprendió) ──────────────────────────
    print(f"\n  Eval train data ({N_EVAL} ejemplos):")
    train_samples = random.sample(examples, min(N_EVAL, len(examples)))
    train_f1s = []
    for i, (q, a) in enumerate(train_samples):
        max_new = min(len(a.split()) * 2 + 4, 24)
        p = greedy_decode(pipeline, q, hint, tok, max_new=max_new)
        f1 = word_f1(p, a)
        train_f1s.append(f1)
        if i < 3:
            q_s = q[:55].encode("ascii", "replace").decode("ascii")
            a_s = a[:55].encode("ascii", "replace").decode("ascii")
            p_s = p[:55].encode("ascii", "replace").decode("ascii")
            print(f"    Q: {q_s}")
            print(f"    A: {a_s}")
            print(f"    P: {p_s}  F1={f1:.2f}")
    mean_f1_train = sum(train_f1s) / len(train_f1s) if train_f1s else 0.0
    print(f"  mean Word F1 (train) = {mean_f1_train:.3f}")

    # ── Eval en NUEVOS ejemplos (generalización) ────────────────────────────
    print(f"\n  Eval nuevos ejemplos ({N_EVAL}):")
    new_f1s = []
    for i in range(N_EVAL):
        ex = gen.generate(level=1)
        q, a = ex.problem_text, ex.answer
        max_new = min(len(a.split()) * 2 + 4, 24)
        p = greedy_decode(pipeline, q, hint, tok, max_new=max_new)
        f1 = word_f1(p, a)
        new_f1s.append(f1)
        if i < 3:
            q_s = q[:55].encode("ascii", "replace").decode("ascii")
            a_s = a[:55].encode("ascii", "replace").decode("ascii")
            p_s = p[:55].encode("ascii", "replace").decode("ascii")
            print(f"    Q: {q_s}")
            print(f"    A: {a_s}")
            print(f"    P: {p_s}  F1={f1:.2f}")
    mean_f1_new = sum(new_f1s) / len(new_f1s) if new_f1s else 0.0
    print(f"  mean Word F1 (nuevos) = {mean_f1_new:.3f}")

    total_elapsed = time.perf_counter() - t0
    return mean_f1_train, mean_f1_new, total_elapsed


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP = "=" * 60

    print(SEP)
    print("  benchmark_5motors_tiny")
    print(f"  hidden_dim=64 | batch=1 | {N_STEPS} steps/motor | nivel=1")
    print(SEP)

    results: Dict[str, Tuple[float, float, float]] = {}
    t_total = time.perf_counter()

    for domain in MOTOR_NAMES:
        f1_train, f1_new, elapsed = run_motor(domain)
        results[domain] = (f1_train, f1_new, elapsed)

    # ── Resumen final ─────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total
    print(f"\n{SEP}")
    print("  RESUMEN FINAL")
    print(SEP)
    print(f"  {'Motor':<10}  {'F1 train':>9}  {'F1 nuevos':>9}  {'Tiempo':>8}")
    print(f"  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*8}")

    all_train_f1s = []
    all_new_f1s   = []
    for domain, (f1t, f1n, elapsed) in results.items():
        bar = "#" * int(f1t * 20)
        print(f"  {domain:<10}  {f1t:>9.3f}  {f1n:>9.3f}  {elapsed:>7.0f}s"
              f"  [{bar:<20}]")
        all_train_f1s.append(f1t)
        all_new_f1s.append(f1n)

    mean_train = sum(all_train_f1s) / len(all_train_f1s)
    mean_new   = sum(all_new_f1s)   / len(all_new_f1s)
    print(f"  {'MEDIA':<10}  {mean_train:>9.3f}  {mean_new:>9.3f}  {total_elapsed:>7.0f}s")
    print(SEP)

    print()
    if mean_train > 0.3:
        print("  >> RESULTADO: Arquitectura OK.")
        print(f"     F1 train={mean_train:.3f} > 0.3 — los 5 motores aprenden.")
        print("     Siguiente paso: dataset grande (50k+ ejemplos) para generalizar.")
    elif mean_train > 0.05:
        print("  >> RESULTADO: Aprendizaje parcial.")
        print(f"     F1 train={mean_train:.3f} — algunos motores aprenden, otros no.")
        print("     Revisar motores con F1 < 0.1.")
    else:
        print("  >> ALERTA: F1 ~ 0 en todos los motores.")
        print("     Problema en decoder o greedy_decode. Revisar pipeline().")
    print(SEP)


if __name__ == "__main__":
    main()
