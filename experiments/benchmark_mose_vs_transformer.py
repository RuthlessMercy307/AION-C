"""
experiments/benchmark_mose_vs_transformer.py
=============================================
Comparación REAL de sistema completo:

    MoSE (orchestrator + 5 motores + unifier, ~1.49M params)
        vs
    Transformer generalista (1 modelo, mismos params totales)

Protocolo de entrenamiento (mismos datos, mismos steps totales = 3500):
  MoSE   — Phase 1: 1000 steps con los 5 dominios mezclados (shared backbone)
           Phase 2: 500 steps por dominio con su generador (fine-tuning motores)
  TF     — 3500 steps con los 5 dominios mezclados uniformemente

Evaluación:
  100 queries nuevas (20 por dominio).
  Para MoSE: imprime qué motor eligió el orquestador.
  Word F1 total y por dominio para ambos.

Config: hidden_dim=64, vocab compartido de los 5 dominios, nivel=1, lr=3e-4, batch=1.
Fix EOS: logits[EOS] = -inf en los primeros 3 tokens generados.

Uso:
    cd AION-C
    python -m experiments.benchmark_mose_vs_transformer
"""
from __future__ import annotations

import sys, os, math, random, re as _re, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

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

# ─── Hyperparams ─────────────────────────────────────────────────────────────
PAD, BOS, EOS = 0, 1, 2
VOCAB_SIZE   = 256
MAX_LEN      = 64
HIDDEN_DIM   = 64
N_EXAMPLES   = 500   # ejemplos por dominio para entrenamiento
N_EVAL       = 20    # queries nuevas por dominio (100 total)
LR           = 3e-4
STEPS_PHASE1 = 1000  # steps shared (MoSE Phase 1) / mezclados (TF)
STEPS_PHASE2 = 500   # steps por motor (MoSE Phase 2)
PRINT_EVERY  = 250

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


# ─── MoSE config (misma que benchmark_5motors_tiny) ──────────────────────────

def make_mose_cfg() -> MoSEConfig:
    base = MoSEConfig.tiny()
    return MoSEConfig(
        hidden_dim        = HIDDEN_DIM,
        vocab_size        = VOCAB_SIZE,
        enc_n_layers      = base.enc_n_layers,
        enc_state_dim     = base.enc_state_dim,
        enc_expand        = base.enc_expand,
        enc_d_conv        = base.enc_d_conv,
        enc_ffn_mult      = base.enc_ffn_mult,
        orch_mlp_hidden   = 32,
        orch_max_motors   = base.orch_max_motors,
        orch_min_confidence = base.orch_min_confidence,
        motor_max_nodes   = base.motor_max_nodes,
        motor_n_heads     = 2,
        motor_threshold   = base.motor_threshold,
        unif_n_heads      = 2,
        dec_n_layers      = base.dec_n_layers,
        dec_n_heads       = 2,
        dec_max_seq_len   = MAX_LEN,
        dec_state_dim     = base.dec_state_dim,
        dec_expand        = base.dec_expand,
        dec_d_conv        = base.dec_d_conv,
        dec_ffn_mult      = base.dec_ffn_mult,
    )


# ─── Tokenizador compartido ───────────────────────────────────────────────────

class Tokenizer:
    def __init__(self, vocab_size: int = VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size
        self._w2i: Dict[str, int] = {}
        self._i2w: List[str] = []

    def build(self, texts: List[str]) -> "Tokenizer":
        freq: Counter = Counter()
        for t in texts:
            freq.update(_re.findall(r"\w+", t.lower()))
        cap = self.vocab_size - 3
        self._i2w = [w for w, _ in freq.most_common(cap)]
        self._w2i = {w: i + 3 for i, w in enumerate(self._i2w)}
        return self

    def encode(self, text: str) -> List[int]:
        toks = _re.findall(r"\w+", text.lower())
        ids = [BOS] + [self._w2i.get(t, EOS) for t in toks] + [EOS]
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
        return torch.tensor([ids], dtype=torch.long)


# ─── Transformer generalista (mismo tamaño que MoSE) ─────────────────────────

class GeneralistTransformer(nn.Module):
    """Decoder-only causal, tamaño ajustado para igualar params de MoSE."""
    def __init__(self, vocab_size: int, hidden_dim: int, n_layers: int,
                 n_heads: int = 2, max_len: int = MAX_LEN,
                 ffn_mult: int = 4) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD)
        self.pos_emb   = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=0.0, batch_first=True,
        )
        self.layers  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        pos  = torch.arange(L, device=ids.device).unsqueeze(0)
        x    = self.token_emb(ids) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ids.device),
                          diagonal=1)
        x    = self.layers(x, mask=mask, is_causal=True)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        seen: set = set()
        n = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                n += p.numel()
        return n


def build_matching_tf(target: int) -> GeneralistTransformer:
    best, best_d = None, float("inf")
    for n in range(1, 50):
        m = GeneralistTransformer(VOCAB_SIZE, HIDDEN_DIM, n)
        d = abs(m.count_parameters() - target)
        if d < best_d:
            best_d, best = d, m
        if m.count_parameters() > target + 150_000:
            break
    return best


# ─── Word F1 ─────────────────────────────────────────────────────────────────

def word_f1(pred: str, ref: str) -> float:
    p, r = set(pred.lower().split()), set(ref.lower().split())
    if not p or not r:
        return 0.0
    tp = len(p & r)
    pr, rc = tp / len(p), tp / len(r)
    return round(2 * pr * rc / (pr + rc), 3) if (pr + rc) else 0.0


# ─── Greedy decode con fix EOS ────────────────────────────────────────────────

MIN_NEW = 3  # suprimir EOS en los primeros N tokens generados


def greedy_mose(mose: MoSEPipeline, prompt: str, hint: str,
                tok: Tokenizer, max_new: int = 20) -> Tuple[str, List[str]]:
    """Retorna (texto_generado, motores_usados)."""
    mose.eval()
    ids  = tok.encode(prompt)
    cur  = tok.to_tensor(ids)
    motors_used: List[str] = []
    with torch.no_grad():
        for step in range(max_new):
            if cur.shape[1] >= MAX_LEN:
                break
            out    = mose(cur, query_text=hint or None)
            if not motors_used:
                motors_used = out.active_motors[:]
            logits = out.logits[0, -1].clone()
            if step < MIN_NEW:
                logits[EOS] = float("-inf")
            nxt = logits.argmax().item()
            if nxt == EOS:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    mose.train()
    return tok.decode(cur[0, len(ids):].tolist()), motors_used


def greedy_tf(tf: GeneralistTransformer, prompt: str,
              tok: Tokenizer, max_new: int = 20) -> str:
    tf.eval()
    ids = tok.encode(prompt)
    cur = tok.to_tensor(ids)
    with torch.no_grad():
        for step in range(max_new):
            if cur.shape[1] >= MAX_LEN:
                break
            logits = tf(cur)[0, -1].clone()
            if step < MIN_NEW:
                logits[EOS] = float("-inf")
            nxt = logits.argmax().item()
            if nxt == EOS:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    tf.train()
    return tok.decode(cur[0, len(ids):].tolist())


# ─── Training step genérico ───────────────────────────────────────────────────

def step_mose(mose: MoSEPipeline, opt: torch.optim.Optimizer,
              ids_t: torch.Tensor, hint: str) -> float:
    out  = mose(ids_t, query_text=hint or None)
    loss = F.cross_entropy(out.logits[0, :-1], ids_t[0, 1:], ignore_index=PAD)
    if math.isfinite(loss.item()):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(mose.parameters(), 1.0)
        opt.step()
        return loss.item()
    return float("nan")


def step_tf(tf: GeneralistTransformer, opt: torch.optim.Optimizer,
            ids_t: torch.Tensor) -> float:
    logits = tf(ids_t)
    loss   = F.cross_entropy(logits[0, :-1], ids_t[0, 1:], ignore_index=PAD)
    if math.isfinite(loss.item()):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(tf.parameters(), 1.0)
        opt.step()
        return loss.item()
    return float("nan")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP = "=" * 65
    total_steps = STEPS_PHASE1 + len(MOTOR_NAMES) * STEPS_PHASE2

    print(SEP)
    print("  benchmark_mose_vs_transformer  [sistema completo]")
    print(f"  hidden_dim={HIDDEN_DIM} | batch=1 | nivel=1")
    print(f"  MoSE: Phase1={STEPS_PHASE1}s + Phase2={STEPS_PHASE2}s x5 = {total_steps}s total")
    print(f"  TF:   {total_steps} steps mezclados")
    print(f"  Eval: {N_EVAL} queries x {len(MOTOR_NAMES)} dominios = {N_EVAL*len(MOTOR_NAMES)} total")
    print(f"  Fix:  EOS suprimido en primeros {MIN_NEW} tokens generados")
    print(SEP)

    # ── 1. Generar ejemplos por dominio ───────────────────────────────────────
    print("\nGenerando ejemplos...")
    examples: Dict[str, List[Tuple[str, str]]] = {}
    all_texts: List[str] = []
    for domain in MOTOR_NAMES:
        gen = GENS[domain]
        exs = [(gen.generate(level=1).problem_text,
                gen.generate(level=1).answer) for _ in range(N_EXAMPLES)]
        # re-generate properly (generate() returns a single Example object)
        exs = []
        for _ in range(N_EXAMPLES):
            ex = gen.generate(level=1)
            exs.append((ex.problem_text, ex.answer))
        examples[domain] = exs
        all_texts += [q + " " + a for q, a in exs]
        print(f"  {domain:<10} {len(exs)} ejemplos")

    # ── 2. Tokenizador compartido ─────────────────────────────────────────────
    tok = Tokenizer(VOCAB_SIZE).build(all_texts)
    print(f"\n  Vocab compartido: {len(tok._i2w)} palabras de {len(all_texts)} textos")

    # Pre-encode todos los ejemplos
    ids_by_domain: Dict[str, List[List[int]]] = {
        d: [tok.encode(q + " " + a) for q, a in exs]
        for d, exs in examples.items()
    }
    # Pool mezclado para Phase 1 / TF
    mixed_pool: List[Tuple[str, List[int]]] = []
    for domain, ids_list in ids_by_domain.items():
        for ids in ids_list:
            mixed_pool.append((domain, ids))
    random.shuffle(mixed_pool)

    # ── 3. Construir MoSE ────────────────────────────────────────────────────
    mose_cfg = make_mose_cfg()
    mose     = MoSEPipeline(mose_cfg)
    bd       = mose.parameter_breakdown()
    p_mose   = bd["total_unique"]
    print(f"\n  MoSE params: {p_mose:,}")
    print(f"    enc={bd['encoder']:,}  orch={bd['orchestrator']:,}  "
          f"dec={bd['decoder']:,}  unif={bd['unifier']:,}")
    for name in MOTOR_NAMES:
        print(f"    motor_{name}={bd[f'motor_{name}']:,}")

    # ── 4. Construir Transformer del mismo tamaño ─────────────────────────────
    tf      = build_matching_tf(p_mose)
    p_tf    = tf.count_parameters()
    n_layers_tf = len(tf.layers.layers)
    print(f"\n  TF params:   {p_tf:,}  "
          f"[hidden={HIDDEN_DIM} n_layers={n_layers_tf} n_heads=2 ffn_mult=4]")
    print(f"  Ratio: {p_mose/p_tf:.3f}x")

    # ── 5. Entrenar MoSE Phase 1 (backbone compartido) ───────────────────────
    print(f"\n{'─'*60}")
    print(f"  MoSE Phase 1: {STEPS_PHASE1} steps mezclados (backbone compartido)")
    print(f"{'─'*60}")
    opt_mose = torch.optim.AdamW(mose.parameters(), lr=LR, weight_decay=1e-2)
    mose.train()
    losses_p1: List[float] = []
    t0 = time.perf_counter()
    for step in range(1, STEPS_PHASE1 + 1):
        domain, ids = random.choice(mixed_pool)
        hint  = HINTS[domain]
        ids_t = tok.to_tensor(ids)
        l = step_mose(mose, opt_mose, ids_t, hint)
        if math.isfinite(l):
            losses_p1.append(l)
        if step % PRINT_EVERY == 0:
            valid = [x for x in losses_p1[-PRINT_EVERY:] if math.isfinite(x)]
            avg   = sum(valid) / len(valid) if valid else float("nan")
            print(f"  [Phase1] step {step:>4}  loss={avg:.4f}  "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
    p1_loss = sum(losses_p1[-50:]) / min(50, len(losses_p1)) if losses_p1 else float("nan")
    print(f"  Phase 1 done: {time.perf_counter()-t0:.0f}s  final_loss={p1_loss:.4f}")

    # ── 6. Entrenar MoSE Phase 2 (fine-tuning por motor) ─────────────────────
    print(f"\n  MoSE Phase 2: {STEPS_PHASE2} steps x {len(MOTOR_NAMES)} motores")
    p2_losses: Dict[str, float] = {}
    for domain in MOTOR_NAMES:
        hint      = HINTS[domain]
        ids_list  = ids_by_domain[domain]
        losses_d: List[float] = []
        t_d = time.perf_counter()
        for step in range(1, STEPS_PHASE2 + 1):
            ids_t = tok.to_tensor(random.choice(ids_list))
            l = step_mose(mose, opt_mose, ids_t, hint)
            if math.isfinite(l):
                losses_d.append(l)
            if step % PRINT_EVERY == 0:
                valid = [x for x in losses_d[-PRINT_EVERY:] if math.isfinite(x)]
                avg   = sum(valid) / len(valid) if valid else float("nan")
                print(f"  [{domain:<8}] step {step:>3}  loss={avg:.4f}  "
                      f"{time.perf_counter()-t_d:.0f}s", flush=True)
        d_loss = sum(losses_d[-20:]) / min(20, len(losses_d)) if losses_d else float("nan")
        p2_losses[domain] = d_loss
        print(f"  {domain} done: {time.perf_counter()-t_d:.0f}s  final_loss={d_loss:.4f}")

    t_mose_total = time.perf_counter() - t0
    print(f"\n  MoSE entrenamiento total: {t_mose_total:.0f}s")
    print(f"  MoSE losses finales Phase2: "
          + "  ".join(f"{d}={v:.3f}" for d, v in p2_losses.items()))

    # ── 7. Entrenar Transformer (3500 steps mezclados) ────────────────────────
    print(f"\n{'─'*60}")
    print(f"  TF: {total_steps} steps mezclados (todos los dominios)")
    print(f"{'─'*60}")
    opt_tf = torch.optim.AdamW(tf.parameters(), lr=LR, weight_decay=1e-2)
    tf.train()
    losses_tf: List[float] = []
    t_tf = time.perf_counter()
    for step in range(1, total_steps + 1):
        _, ids = random.choice(mixed_pool)
        ids_t  = tok.to_tensor(ids)
        l = step_tf(tf, opt_tf, ids_t)
        if math.isfinite(l):
            losses_tf.append(l)
        if step % PRINT_EVERY == 0:
            valid = [x for x in losses_tf[-PRINT_EVERY:] if math.isfinite(x)]
            avg   = sum(valid) / len(valid) if valid else float("nan")
            print(f"  [TF]  step {step:>4}  loss={avg:.4f}  "
                  f"{time.perf_counter()-t_tf:.0f}s", flush=True)
    tf_loss_final = sum(losses_tf[-50:]) / min(50, len(losses_tf)) if losses_tf else float("nan")
    t_tf_total = time.perf_counter() - t_tf
    print(f"  TF done: {t_tf_total:.0f}s  final_loss={tf_loss_final:.4f}")

    # ── 8. Losses finales lado a lado ─────────────────────────────────────────
    mose_loss_total = sum(p2_losses.values()) / len(p2_losses)
    print(f"\n  ── Losses finales (comparacion) ────────────────────────────")
    print(f"  MoSE Phase1={p1_loss:.4f}  Phase2 media={mose_loss_total:.4f}  "
          f"({t_mose_total:.0f}s total)")
    print(f"  TF   final={tf_loss_final:.4f}  ({t_tf_total:.0f}s total)")

    # ── 9. Evaluación en 100 queries nuevas ───────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  EVALUACIÓN: {N_EVAL} queries nuevas x {len(MOTOR_NAMES)} dominios")
    print(f"{'─'*60}")

    domain_f1_mose: Dict[str, List[float]] = defaultdict(list)
    domain_f1_tf:   Dict[str, List[float]] = defaultdict(list)

    for domain in MOTOR_NAMES:
        gen  = GENS[domain]
        hint = HINTS[domain]
        print(f"\n  [{domain.upper()}]")
        for i in range(N_EVAL):
            ex = gen.generate(level=1)
            q, a = ex.problem_text, ex.answer
            max_new = min(len(a.split()) * 2 + 4, 24)

            pm, motors = greedy_mose(mose, q, hint, tok, max_new)
            pt         = greedy_tf(tf, q, tok, max_new)

            f1m = word_f1(pm, a)
            f1t = word_f1(pt, a)
            domain_f1_mose[domain].append(f1m)
            domain_f1_tf[domain].append(f1t)

            if i < 3:  # 3 ejemplos cualitativos por dominio
                q_s  = q[:48].encode("ascii", "replace").decode("ascii")
                a_s  = a[:42].encode("ascii", "replace").decode("ascii")
                pm_s = pm[:42].encode("ascii", "replace").decode("ascii")
                pt_s = pt[:42].encode("ascii", "replace").decode("ascii")
                m_str = "+".join(motors) if motors else "?"
                print(f"    Q:    {q_s}")
                print(f"    Ref:  {a_s}")
                print(f"    MoSE [{m_str:<8}]: {pm_s}  F1={f1m:.2f}")
                print(f"    TF:                 {pt_s}  F1={f1t:.2f}")

        mean_m = sum(domain_f1_mose[domain]) / N_EVAL
        mean_t = sum(domain_f1_tf[domain]) / N_EVAL
        winner = "MoSE" if mean_m > mean_t else ("TF" if mean_t > mean_m else "=")
        print(f"    F1 MoSE={mean_m:.3f}  F1 TF={mean_t:.3f}  -> {winner}")

    # ── 10. Tabla resumen ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RESUMEN FINAL")
    print(SEP)
    print(f"  {'Dominio':<10}  {'F1 MoSE':>8}  {'F1 TF':>7}  {'Delta':>7}  {'Ganador'}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}")

    mose_wins = tf_wins = draws = 0
    all_m: List[float] = []
    all_t: List[float] = []
    for domain in MOTOR_NAMES:
        f1m = sum(domain_f1_mose[domain]) / N_EVAL
        f1t = sum(domain_f1_tf[domain]) / N_EVAL
        d   = f1m - f1t
        s   = "+" if d >= 0 else ""
        if f1m > f1t:
            w = "MoSE  <<<"
            mose_wins += 1
        elif f1t > f1m:
            w = "TF    <<<"
            tf_wins += 1
        else:
            w = "EMPATE"
            draws += 1
        all_m.append(f1m)
        all_t.append(f1t)
        print(f"  {domain:<10}  {f1m:>8.3f}  {f1t:>7.3f}  {s}{d:>6.3f}  {w}")

    mean_m = sum(all_m) / len(all_m)
    mean_t = sum(all_t) / len(all_t)
    mean_d = mean_m - mean_t
    s_m    = "+" if mean_d >= 0 else ""
    print(f"  {'TOTAL':<10}  {mean_m:>8.3f}  {mean_t:>7.3f}  {s_m}{mean_d:>6.3f}")
    print(SEP)
    print(f"  Params MoSE={p_mose:,}  TF={p_tf:,}  ratio={p_mose/p_tf:.3f}x")
    print(f"  Steps:  MoSE={total_steps} ({STEPS_PHASE1}+{STEPS_PHASE2}x5)  TF={total_steps}")
    print(f"  Tiempo: MoSE={t_mose_total:.0f}s  TF={t_tf_total:.0f}s")
    print(f"  Victorias MoSE={mose_wins}  TF={tf_wins}  Empates={draws}")
    print()

    if mose_wins > tf_wins:
        print(f"  >> MoSE GANA ({mose_wins}/{len(MOTOR_NAMES)} dominios, "
              f"F1 medio +{mean_m-mean_t:.3f})")
        print("     El routing especializado supera al generalista con igual presupuesto.")
    elif tf_wins > mose_wins:
        print(f"  >> Transformer GANA ({tf_wins}/{len(MOTOR_NAMES)} dominios, "
              f"F1 medio +{mean_t-mean_m:.3f})")
        print("     El generalista supera al especializado con 500 ejemplos/dominio.")
    else:
        print(f"  >> EMPATE ({mose_wins} vs {tf_wins}), delta={s_m}{mean_d:.3f}")

    print(f"  (nivel=1, {N_EXAMPLES} ejemplos/dominio — indicativo)")
    print(SEP)


if __name__ == "__main__":
    main()
