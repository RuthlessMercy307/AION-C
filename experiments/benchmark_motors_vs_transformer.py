"""
experiments/benchmark_motors_vs_transformer.py
===============================================
Comparación JUSTA: cada motor MoSE (1 motor solo, sin orchestrator ni unifier)
contra un Transformer vanilla con el MISMO número de parámetros.

Arquitectura del motor:
    StreamEncoder → GraphCrystallizer → CRE → StreamDecoder
    = CORAPipeline con hidden_dim=64, sin BudgetManager ni Validator

El Transformer tiene hidden_dim=64 y n_layers ajustado automáticamente para
igualar el conteo de parámetros del motor (~661K).

Config idéntica a benchmark_5motors_tiny.py:
    hidden_dim=64, vocab del dataset, nivel=1, 2000 steps, lr=3e-4, batch=1.

Uso:
    cd AION-C
    python -m experiments.benchmark_motors_vs_transformer
"""
from __future__ import annotations

import sys, os, math, random, re as _re, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from router.pipeline import CORAConfig, CORAPipeline
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
VOCAB_SIZE  = 256
MAX_LEN     = 64
HIDDEN_DIM  = 64
N_EXAMPLES  = 500
N_STEPS     = 2000
N_EVAL      = 100
LR          = 3e-4
PRINT_EVERY = 500

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


# ─── Config del motor (single-motor CORAPipeline) ─────────────────────────────

def make_motor_cfg() -> CORAConfig:
    """
    Motor único: Encoder → Crystallizer → CRE → Decoder.
    Mismas dimensiones que cada motor dentro del MoSE de benchmark_5motors_tiny.
    Sin BudgetManager ni Validator para comparación limpia.
    """
    return CORAConfig(
        hidden_dim            = HIDDEN_DIM,
        vocab_size            = VOCAB_SIZE,
        # Encoder (misma config tiny)
        enc_n_layers          = 2,
        enc_state_dim         = 4,
        enc_expand            = 2,
        enc_d_conv            = 4,
        enc_ffn_mult          = 2,
        # Crystallizer
        cryst_max_nodes       = 8,
        cryst_n_heads         = 2,
        cryst_node_threshold  = 0.3,
        cryst_edge_threshold  = 0.3,
        # CRE (1 capa de MP por iteración, igual que cada motor en MoSE)
        cre_edge_dim          = 16,
        cre_message_dim       = 32,
        cre_n_message_layers  = 1,
        cre_max_iterations    = 10,
        cre_use_convergence_gate = False,
        # Scratch pad
        pad_n_slots           = 8,
        pad_slot_dim          = 32,
        # Decoder (misma config tiny)
        dec_n_layers          = 2,
        dec_n_heads           = 2,
        dec_max_seq_len       = MAX_LEN,
        dec_state_dim         = 4,
        dec_expand            = 2,
        dec_d_conv            = 4,
        dec_ffn_mult          = 2,
        # Desactivar módulos opcionales
        use_budget_manager    = False,
        use_validator         = False,
    )


# ─── Tokenizador word-level ───────────────────────────────────────────────────

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


# ─── Word F1 ─────────────────────────────────────────────────────────────────

def word_f1(pred: str, ref: str) -> float:
    p, r = set(pred.lower().split()), set(ref.lower().split())
    if not p or not r:
        return 0.0
    tp = len(p & r)
    pr = tp / len(p)
    rc = tp / len(r)
    return round(2 * pr * rc / (pr + rc), 3) if (pr + rc) else 0.0


# ─── Vanilla Transformer (decoder-only causal) ────────────────────────────────

class VanillaTransformer(nn.Module):
    """
    Transformer decoder-only puro (sin cross-attention a memoria externa).
    Usa nn.TransformerEncoderLayer con causal mask para evitar overhead de
    la memoria dummy, y así n_layers controla los params de forma más limpia.
    """
    def __init__(self, vocab_size: int, hidden_dim: int = HIDDEN_DIM,
                 n_layers: int = 4, n_heads: int = 4,
                 max_len: int = MAX_LEN, ffn_mult: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len    = max_len
        self.token_emb  = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD)
        self.pos_emb    = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=0.0, batch_first=True,
        )
        self.layers  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device),
                          diagonal=1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, L = token_ids.shape
        pos  = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x    = self.token_emb(token_ids) + self.pos_emb(pos)
        mask = self._causal_mask(L, token_ids.device)
        x    = self.layers(x, mask=mask, is_causal=True)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        seen: set = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total


def build_matching_transformer(target_params: int,
                                vocab_size: int = VOCAB_SIZE,
                                hidden_dim: int = HIDDEN_DIM,
                                max_len: int = MAX_LEN,
                                n_heads: int = 2,
                                ffn_mult: int = 4) -> VanillaTransformer:
    """
    Crea un VanillaTransformer con el número de capas que minimiza
    |params - target_params|. Busca en el rango n_layers=1..30.
    """
    best_model  = None
    best_delta  = float("inf")
    for n_layers in range(1, 31):
        m = VanillaTransformer(vocab_size, hidden_dim, n_layers,
                                n_heads, max_len, ffn_mult)
        p = m.count_parameters()
        d = abs(p - target_params)
        if d < best_delta:
            best_delta = d
            best_model = m
        elif p > target_params + 100_000:
            break  # overshoot por mucho → parar
    return best_model


# ─── Greedy decode ────────────────────────────────────────────────────────────

def _min_new_before_eos(n_input_tokens: int) -> int:
    """Supresión dinámica de EOS: max(3, len(input) // 3).
    Nota: en dominios con respuestas cortas (CORA nivel=1), el fijo de 3
    funciona mejor que el dinámico porque no fuerza tokens extra en "Sí/No".
    """
    return 3  # fijo en 3 — mejor para respuestas cortas de nivel=1


def greedy_decode_motor(pipeline: CORAPipeline, prompt: str,
                        tok: Tokenizer, max_new: int = 20) -> str:
    pipeline.eval()
    ids = tok.encode(prompt)
    cur = tok.to_tensor(ids)
    min_new = _min_new_before_eos(len(ids))
    with torch.no_grad():
        for step in range(max_new):
            if cur.shape[1] >= MAX_LEN:
                break
            out    = pipeline(cur)
            logits = out.logits[0, -1].clone()
            if step < min_new:
                logits[EOS] = float("-inf")  # suprimir EOS prematuro
            nxt = logits.argmax().item()
            if nxt == EOS:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    pipeline.train()
    return tok.decode(cur[0, len(ids):].tolist())


def greedy_decode_tf(model: VanillaTransformer, prompt: str,
                     tok: Tokenizer, max_new: int = 20) -> str:
    model.eval()
    ids = tok.encode(prompt)
    cur = tok.to_tensor(ids)
    min_new = _min_new_before_eos(len(ids))
    with torch.no_grad():
        for step in range(max_new):
            if cur.shape[1] >= MAX_LEN:
                break
            logits = model(cur)[0, -1].clone()
            if step < min_new:
                logits[EOS] = float("-inf")  # suprimir EOS prematuro
            nxt = logits.argmax().item()
            if nxt == EOS:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    model.train()
    return tok.decode(cur[0, len(ids):].tolist())


# ─── Training loops ───────────────────────────────────────────────────────────

def train_model(model: nn.Module, ids_list: List[List[int]],
                tok: Tokenizer, n_steps: int, label: str,
                is_motor: bool = True) -> Tuple[List[float], float]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    model.train()
    losses: List[float] = []
    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        ids_t = tok.to_tensor(random.choice(ids_list))

        if is_motor:
            out   = model(ids_t)
            logits = out.logits
        else:
            logits = model(ids_t)

        loss = F.cross_entropy(logits[0, :-1], ids_t[0, 1:], ignore_index=PAD)

        if math.isfinite(loss.item()):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        if step % PRINT_EVERY == 0:
            valid = [x for x in losses[-PRINT_EVERY:] if math.isfinite(x)]
            avg = sum(valid) / len(valid) if valid else float("nan")
            elapsed = time.perf_counter() - t0
            print(f"    [{label}]  step {step:>4}  loss={avg:.4f}  {elapsed:.0f}s",
                  flush=True)

    elapsed    = time.perf_counter() - t0
    final_loss = (sum(losses[-50:]) / min(50, len(losses))) if losses else float("nan")
    return losses, elapsed, final_loss


# ─── Eval ────────────────────────────────────────────────────────────────────

def eval_motor(pipeline: CORAPipeline, gen, tok: Tokenizer,
               n_eval: int = N_EVAL) -> float:
    f1s = []
    for _ in range(n_eval):
        ex = gen.generate(level=1)
        q, a = ex.problem_text, ex.answer
        max_new = min(len(a.split()) * 2 + 4, 24)
        p = greedy_decode_motor(pipeline, q, tok, max_new)
        f1s.append(word_f1(p, a))
    return round(sum(f1s) / len(f1s), 3) if f1s else 0.0


def eval_tf(model: VanillaTransformer, gen, tok: Tokenizer,
            n_eval: int = N_EVAL) -> float:
    f1s = []
    for _ in range(n_eval):
        ex = gen.generate(level=1)
        q, a = ex.problem_text, ex.answer
        max_new = min(len(a.split()) * 2 + 4, 24)
        p = greedy_decode_tf(model, q, tok, max_new)
        f1s.append(word_f1(p, a))
    return round(sum(f1s) / len(f1s), 3) if f1s else 0.0


# ─── Benchmark de un motor ────────────────────────────────────────────────────

def run_motor(domain: str) -> Tuple[float, float, float, float, int, int]:
    """
    Retorna (f1_motor, f1_tf, elapsed_motor, elapsed_tf, params_motor, params_tf).
    """
    SEP = "-" * 60
    print(f"\n{SEP}")
    print(f"  Motor: {domain.upper()}")
    print(SEP)

    gen = GENS[domain]

    # ── 1. Generar ejemplos ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    examples: List[Tuple[str, str]] = []
    for _ in range(N_EXAMPLES):
        ex = gen.generate(level=1)
        examples.append((ex.problem_text, ex.answer))

    all_text = [q + " " + a for q, a in examples]
    tok = Tokenizer(VOCAB_SIZE).build(all_text)
    ids_list = [tok.encode(q + " " + a) for q, a in examples]
    print(f"  Vocab: {len(tok._i2w)} palabras | {N_EXAMPLES} ejemplos "
          f"({time.perf_counter()-t0:.1f}s)")

    # ── 2. Motor: CORAPipeline (Encoder -> Crystallizer -> CRE -> Decoder) ──
    motor_cfg = make_motor_cfg()
    motor     = CORAPipeline(motor_cfg)
    bd        = motor.parameter_breakdown()
    params_motor = bd["total_unique"]

    # ── 3. Transformer con el mismo numero de params ─────────────────────────
    tf = build_matching_transformer(
        target_params = params_motor,
        vocab_size    = VOCAB_SIZE,
        hidden_dim    = HIDDEN_DIM,
        max_len       = MAX_LEN,
        n_heads       = 2,
        ffn_mult      = 4,
    )
    params_tf = tf.count_parameters()

    print(f"  Motor (CORAPipeline): {params_motor:>9,} params  "
          f"[enc={bd['encoder']:,} cryst={bd['crystallizer']:,} "
          f"cre={bd['cre']:,} dec={bd['decoder']:,}]")
    print(f"  Transformer:          {params_tf:>9,} params  "
          f"[hidden={HIDDEN_DIM} n_layers={len(tf.layers.layers)} n_heads=2 ffn_mult=4]")
    print(f"  Ratio params:         {params_motor/params_tf:.2f}x")

    # ── 4. Entrenar motor ────────────────────────────────────────────────────
    print(f"  Entrenando Motor ({N_STEPS} steps)...")
    _, elapsed_motor, loss_motor_final = train_model(
        motor, ids_list, tok, N_STEPS, "Motor", is_motor=True)

    # ── 5. Entrenar Transformer ───────────────────────────────────────────────
    print(f"  Entrenando Transformer ({N_STEPS} steps)...")
    _, elapsed_tf, loss_tf_final = train_model(
        tf, ids_list, tok, N_STEPS, "TF   ", is_motor=False)

    print(f"  ── Losses finales (promedio ultimos 50 steps) ──────────────────")
    print(f"  Motor: loss={loss_motor_final:.4f}  ({elapsed_motor:.0f}s)")
    print(f"  TF:    loss={loss_tf_final:.4f}  ({elapsed_tf:.0f}s)")
    better = "Motor" if loss_motor_final < loss_tf_final else "TF"
    print(f"  Menor loss: {better}  "
          f"(delta={abs(loss_motor_final - loss_tf_final):.4f})")

    # ── 6. Eval en 100 nuevos ejemplos ───────────────────────────────────────
    print(f"  Evaluando {N_EVAL} ejemplos nuevos...")
    f1_motor = eval_motor(motor, gen, tok, N_EVAL)
    f1_tf    = eval_tf(tf, gen, tok, N_EVAL)

    # ── 7. Ejemplos cualitativos ─────────────────────────────────────────────
    print(f"\n  Ejemplos cualitativos (3):")
    for _ in range(3):
        ex = gen.generate(level=1)
        q, a = ex.problem_text, ex.answer
        max_new = min(len(a.split()) * 2 + 4, 24)
        pm = greedy_decode_motor(motor, q, tok, max_new)
        pt = greedy_decode_tf(tf, q, tok, max_new)
        q_s  = q[:50].encode("ascii", "replace").decode("ascii")
        a_s  = a[:45].encode("ascii", "replace").decode("ascii")
        pm_s = pm[:45].encode("ascii", "replace").decode("ascii")
        pt_s = pt[:45].encode("ascii", "replace").decode("ascii")
        print(f"    Q:     {q_s}")
        print(f"    Ref:   {a_s}")
        print(f"    Motor: {pm_s}  (F1={word_f1(pm,a):.2f})")
        print(f"    TF:    {pt_s}  (F1={word_f1(pt,a):.2f})")

    winner = "Motor" if f1_motor > f1_tf else ("TF" if f1_tf > f1_motor else "EMPATE")
    delta  = f1_motor - f1_tf
    sign   = "+" if delta >= 0 else ""
    print(f"\n  F1 Motor={f1_motor:.3f}  F1 TF={f1_tf:.3f}  "
          f"delta={sign}{delta:.3f}  Ganador: {winner}")

    return f1_motor, f1_tf, elapsed_motor, elapsed_tf, params_motor, params_tf


# ─── main ─────────────────────────────────────────────────────────────────────

def main(domains: Optional[List[str]] = None) -> None:
    """
    Args:
        domains: lista de motores a correr. None = los 5. Ej: ["cora"]
    """
    SEP = "=" * 65
    run_domains = domains or list(MOTOR_NAMES)

    print(SEP)
    print("  benchmark_motors_vs_transformer  [comparacion justa]")
    print(f"  hidden_dim={HIDDEN_DIM} | batch=1 | {N_STEPS} steps | "
          f"{N_EVAL} eval | nivel=1")
    print(f"  Motores: {', '.join(run_domains)}")
    print("  Motor: Encoder->Crystallizer->CRE->Decoder (1 motor, sin orch/unif)")
    print("  TF:    Decoder-only causal (mismo numero de params)")
    print("  Fix:   EOS suprimido en primeros 3 tokens generados (fijo)")
    print(SEP)

    results: Dict[str, Tuple[float, float, float, float, int, int]] = {}
    t_total = time.perf_counter()

    for domain in run_domains:
        results[domain] = run_motor(domain)

    total_elapsed = time.perf_counter() - t_total

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RESUMEN FINAL  (Motor = CORAPipeline, TF = Transformer equivalente)")
    print(SEP)
    print(f"  {'Motor':<10}  {'F1 Motor':>9}  {'F1 TF':>7}  "
          f"{'Delta':>7}  {'Ganador':<10}  {'P_motor':>9}  {'P_tf':>9}")
    print(f"  {'─'*10}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*9}  {'─'*9}")

    motor_wins = tf_wins = draws = 0
    all_f1m: List[float] = []
    all_f1t: List[float] = []

    for domain, (f1m, f1t, em, et, pm, pt) in results.items():
        delta = f1m - f1t
        sign  = "+" if delta >= 0 else ""
        if f1m > f1t:
            winner = "Motor  <<<"
            motor_wins += 1
        elif f1t > f1m:
            winner = "TF     <<<"
            tf_wins += 1
        else:
            winner = "EMPATE    "
            draws += 1
        all_f1m.append(f1m)
        all_f1t.append(f1t)
        print(f"  {domain:<10}  {f1m:>9.3f}  {f1t:>7.3f}  "
              f"{sign}{delta:>6.3f}  {winner:<10}  {pm:>9,}  {pt:>9,}")

    mean_m = sum(all_f1m) / len(all_f1m)
    mean_t = sum(all_f1t) / len(all_f1t)
    mean_d = mean_m - mean_t
    sign_m = "+" if mean_d >= 0 else ""
    print(f"  {'MEDIA':<10}  {mean_m:>9.3f}  {mean_t:>7.3f}  "
          f"{sign_m}{mean_d:>6.3f}")
    print(SEP)
    print(f"  Total: {total_elapsed:.0f}s  |  "
          f"Victorias Motor={motor_wins}  TF={tf_wins}  Empates={draws}")
    print()

    if motor_wins > tf_wins:
        adv = mean_m - mean_t
        print(f"  >> Motor MoSE GANA ({motor_wins}/{len(MOTOR_NAMES)} dominios, "
              f"F1 medio +{adv:.3f})")
        print("     El inductive bias del grafo causal ayuda en estas tareas.")
    elif tf_wins > motor_wins:
        adv = mean_t - mean_m
        print(f"  >> Transformer GANA ({tf_wins}/{len(MOTOR_NAMES)} dominios, "
              f"F1 medio +{adv:.3f})")
        print("     Con igual presupuesto de params, el TF converge mas rapido.")
    else:
        print(f"  >> EMPATE ({motor_wins} vs {tf_wins}), delta medio={sign_m}{mean_d:.3f}")

    print(f"  (2000 steps, 500 ejemplos sinteticos nivel=1 — indicativo)")
    print(SEP)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="*", default=None,
                        help="Motores a correr (ej: cora forge_c). Default: todos.")
    args = parser.parse_args()
    main(domains=args.domains)
