"""
experiments/benchmark_opus_cora.py
====================================
Mini-benchmark: datos Opus vs datos sintéticos para el motor CORA.

Compara la calidad del entrenamiento con:
  A) 1000 ejemplos del dataset Opus (generado por Claude Opus) — 500 train / 100 eval
  B) 500 ejemplos sintéticos (CausalGraphGenerator, nivel=1) — evaluación en 100 nuevos

Ambos entrenan CORAPipeline (Encoder → Crystallizer → CRE → Decoder)
con la misma config, mismo número de steps y training_utils.train_with_amp.

Métricas:
  - Word F1 en 100 ejemplos nuevos del mismo origen (Opus eval / sintético eval)
  - Loss final (promedio últimos 50 steps)
  - Tiempo de entrenamiento

Config:
  hidden_dim=64, vocab word-level del corpus, 500 steps, lr=3e-4,
  warmup=50 steps, batch=1, EOS suprimido en primeros 3 tokens.

Uso:
    cd AION-C
    python -m experiments.benchmark_opus_cora
    python -m experiments.benchmark_opus_cora --steps 200  # run rápido
    python -m experiments.benchmark_opus_cora --no-synthetic  # solo Opus
"""
from __future__ import annotations

import sys, os, math, random, re as _re, time, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from router.pipeline import CORAConfig, CORAPipeline
from synth.causal_graph_gen import CausalGraphGenerator
from experiments.opus_dataset import OpusDataset
from experiments.training_utils import train_with_amp

torch.set_num_threads(4)
torch.manual_seed(42)
random.seed(42)

# ─── Hiperparámetros ─────────────────────────────────────────────────────────

PAD, BOS, EOS = 0, 1, 2
VOCAB_SIZE     = 512    # más grande que en benchmark base — Opus tiene vocab rico
MAX_LEN        = 96
HIDDEN_DIM     = 64
N_TRAIN_OPUS   = 900    # de 1000 ejemplos Opus: 900 train, 100 eval
N_EVAL_OPUS    = 100
N_SYNTH        = 500    # ejemplos sintéticos para el baseline
N_STEPS        = 500
LR             = 3e-4
WARMUP_STEPS   = 50
PRINT_EVERY    = 100
MIN_NEW        = 3      # EOS suprimido en los primeros 3 tokens


# ─── Motor config ────────────────────────────────────────────────────────────

def make_motor_cfg(vocab_size: int) -> CORAConfig:
    return CORAConfig(
        hidden_dim            = HIDDEN_DIM,
        vocab_size            = vocab_size,
        enc_n_layers          = 2,
        enc_state_dim         = 4,
        enc_expand            = 2,
        enc_d_conv            = 4,
        enc_ffn_mult          = 2,
        cryst_max_nodes       = 8,
        cryst_n_heads         = 2,
        cryst_node_threshold  = 0.3,
        cryst_edge_threshold  = 0.3,
        cre_edge_dim          = 16,
        cre_message_dim       = 32,
        cre_n_message_layers  = 1,
        cre_max_iterations    = 10,
        cre_use_convergence_gate = False,
        pad_n_slots           = 8,
        pad_slot_dim          = 32,
        dec_n_layers          = 2,
        dec_n_heads           = 2,
        dec_max_seq_len       = MAX_LEN,
        dec_state_dim         = 4,
        dec_expand            = 2,
        dec_d_conv            = 4,
        dec_ffn_mult          = 2,
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
        ids  = [BOS] + [self._w2i.get(t, EOS) for t in toks] + [EOS]
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


# ─── Greedy decode ────────────────────────────────────────────────────────────

def greedy_decode(pipeline: CORAPipeline, prompt: str,
                  tok: Tokenizer, max_new: int = 24) -> str:
    pipeline.eval()
    ids = tok.encode(prompt)
    cur = tok.to_tensor(ids)
    with torch.no_grad():
        for step in range(max_new):
            if cur.shape[1] >= MAX_LEN:
                break
            out    = pipeline(cur)
            logits = out.logits[0, -1].clone()
            if step < MIN_NEW:
                logits[EOS] = float("-inf")
            nxt = logits.argmax().item()
            if nxt == EOS:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    pipeline.train()
    return tok.decode(cur[0, len(ids):].tolist())


# ─── Eval ────────────────────────────────────────────────────────────────────

def eval_f1(pipeline: CORAPipeline, eval_examples: List[Tuple[str, str]],
            tok: Tokenizer) -> Tuple[float, List[float]]:
    f1s = []
    for q, a in eval_examples:
        max_new = min(len(a.split()) * 2 + 4, 32)
        pred    = greedy_decode(pipeline, q, tok, max_new)
        f1s.append(word_f1(pred, a))
    mean = round(sum(f1s) / len(f1s), 3) if f1s else 0.0
    return mean, f1s


# ─── Run Opus ────────────────────────────────────────────────────────────────

def run_opus(n_steps: int, checkpoint_dir: Optional[str]) -> Tuple[float, float, float]:
    """
    Entrena CORAPipeline con datos Opus (CORA, 1000 ejemplos).
    Retorna (word_f1, final_loss, elapsed_s).
    """
    print("\n── Cargando dataset Opus (CORA, 1000 ejemplos) ──")
    opus_full = OpusDataset("cora", max_examples=1000, seed=42)
    train_ds, eval_ds = opus_full.train_eval_split(eval_size=N_EVAL_OPUS, seed=42)
    print(f"   Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # Construir corpus y tokenizador
    train_texts  = train_ds.get_all_texts()
    tok          = Tokenizer(VOCAB_SIZE).build(train_texts)
    print(f"   Vocab: {len(tok._i2w)} palabras")

    # Pre-tokenizar
    train_examples = [(ex["input"], ex["expected_output"])
                      for ex in train_ds._examples]
    ids_list = [tok.encode(q + " " + a) for q, a in train_examples]

    eval_examples = [(ex["input"], ex["expected_output"])
                     for ex in eval_ds._examples]

    # Mostrar 3 ejemplos de train
    print("   Ejemplos de train (3):")
    for q, a in train_examples[:3]:
        q_s = q[:70].encode("ascii", "replace").decode("ascii")
        a_s = a[:50].encode("ascii", "replace").decode("ascii")
        print(f"     Q: {q_s}")
        print(f"     A: {a_s}")

    # Construir y entrenar motor
    cfg      = make_motor_cfg(VOCAB_SIZE)
    pipeline = CORAPipeline(cfg)
    bd       = pipeline.parameter_breakdown()
    print(f"   Params: {bd['total_unique']:,} "
          f"[enc={bd['encoder']:,} cryst={bd['crystallizer']:,} "
          f"cre={bd['cre']:,} dec={bd['decoder']:,}]")

    ckpt_path = None
    if checkpoint_dir:
        ckpt_path = os.path.join(checkpoint_dir, "opus_cora.pt")

    print(f"\n── Entrenando Motor con datos Opus ({n_steps} steps) ──")
    _, elapsed, final_loss = train_with_amp(
        model            = pipeline,
        ids_list         = ids_list,
        tok              = tok,
        n_steps          = n_steps,
        label            = "Opus-CORA",
        is_motor         = True,
        lr               = LR,
        warmup_steps     = WARMUP_STEPS,
        checkpoint_every = 500,
        checkpoint_path  = ckpt_path,
        print_every      = PRINT_EVERY,
    )

    # Eval
    print(f"\n── Evaluando en {N_EVAL_OPUS} ejemplos Opus (nuevos, no vistos) ──")
    f1_mean, f1s = eval_f1(pipeline, eval_examples, tok)

    # Mostrar 5 ejemplos cualitativos
    print("   Ejemplos cualitativos (5):")
    for q, a in eval_examples[:5]:
        max_new = min(len(a.split()) * 2 + 4, 32)
        pred    = greedy_decode(pipeline, q, tok, max_new)
        f1      = word_f1(pred, a)
        q_s  = q[:55].encode("ascii", "replace").decode("ascii")
        a_s  = a[:40].encode("ascii", "replace").decode("ascii")
        p_s  = pred[:40].encode("ascii", "replace").decode("ascii")
        print(f"     Q:    {q_s}")
        print(f"     Ref:  {a_s}")
        print(f"     Pred: {p_s}  (F1={f1:.2f})")

    print(f"\n   RESULTADO OPUS: F1={f1_mean:.3f}  loss={final_loss:.4f}  {elapsed:.0f}s")
    return f1_mean, final_loss, elapsed


# ─── Run Sintético ────────────────────────────────────────────────────────────

def run_synthetic(n_steps: int, checkpoint_dir: Optional[str]) -> Tuple[float, float, float]:
    """
    Entrena CORAPipeline con datos sintéticos (CausalGraphGenerator, nivel=1).
    Retorna (word_f1, final_loss, elapsed_s).
    """
    print("\n── Generando datos sintéticos (CORA, 500 ejemplos, nivel=1) ──")
    gen      = CausalGraphGenerator()
    examples = []
    for _ in range(N_SYNTH):
        ex = gen.generate(level=1)
        examples.append((ex.problem_text, ex.answer))

    all_texts = [q + " " + a for q, a in examples]
    tok       = Tokenizer(VOCAB_SIZE).build(all_texts)
    ids_list  = [tok.encode(q + " " + a) for q, a in examples]
    print(f"   Vocab: {len(tok._i2w)} palabras  |  {N_SYNTH} ejemplos")

    # Construir y entrenar motor (mismo config que Opus)
    cfg      = make_motor_cfg(VOCAB_SIZE)
    pipeline = CORAPipeline(cfg)

    ckpt_path = None
    if checkpoint_dir:
        ckpt_path = os.path.join(checkpoint_dir, "synth_cora.pt")

    print(f"\n── Entrenando Motor con datos Sintéticos ({n_steps} steps) ──")
    _, elapsed, final_loss = train_with_amp(
        model            = pipeline,
        ids_list         = ids_list,
        tok              = tok,
        n_steps          = n_steps,
        label            = "Synth-CORA",
        is_motor         = True,
        lr               = LR,
        warmup_steps     = WARMUP_STEPS,
        checkpoint_every = 500,
        checkpoint_path  = ckpt_path,
        print_every      = PRINT_EVERY,
    )

    # Eval en 100 nuevos ejemplos sintéticos
    print(f"\n── Evaluando en 100 ejemplos sintéticos nuevos ──")
    eval_examples = []
    for _ in range(100):
        ex = gen.generate(level=1)
        eval_examples.append((ex.problem_text, ex.answer))

    f1_mean, _ = eval_f1(pipeline, eval_examples, tok)

    print(f"\n   RESULTADO SINTÉTICO: F1={f1_mean:.3f}  loss={final_loss:.4f}  {elapsed:.0f}s")
    return f1_mean, final_loss, elapsed


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mini-benchmark: datos Opus vs sintéticos para motor CORA"
    )
    parser.add_argument("--steps",        type=int,  default=500,
                        help="Número de steps de entrenamiento (default: 500)")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Solo corre el run con datos Opus")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directorio para guardar checkpoints (default: sin checkpoint)")
    args = parser.parse_args()

    SEP = "=" * 65
    print(SEP)
    print("  benchmark_opus_cora — Datos Opus vs Sintéticos")
    print(f"  hidden_dim={HIDDEN_DIM} | vocab={VOCAB_SIZE} | {args.steps} steps")
    print(f"  warmup={WARMUP_STEPS} | lr={LR} | EOS_min={MIN_NEW}")
    print(f"  Train Opus: {N_TRAIN_OPUS} ej | Eval Opus: {N_EVAL_OPUS} ej")
    print(f"  Train Synth: {N_SYNTH} ej | Eval Synth: 100 ej nuevos")
    print(SEP)

    t_total = time.perf_counter()

    # Run Opus
    f1_opus, loss_opus, t_opus = run_opus(args.steps, args.checkpoint_dir)

    # Run Sintético (opcional)
    f1_synth = loss_synth = t_synth = None
    if not args.no_synthetic:
        f1_synth, loss_synth, t_synth = run_synthetic(args.steps, args.checkpoint_dir)

    total = time.perf_counter() - t_total

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RESUMEN COMPARATIVO")
    print(SEP)
    print(f"  {'Fuente':<14}  {'F1 eval':>8}  {'Loss final':>10}  {'Tiempo':>8}")
    print(f"  {'─'*14}  {'─'*8}  {'─'*10}  {'─'*8}")
    print(f"  {'Opus (1000 ej)':<14}  {f1_opus:>8.3f}  {loss_opus:>10.4f}  {t_opus:>7.0f}s")

    if f1_synth is not None:
        print(f"  {'Sintético (500)':<14}  {f1_synth:>8.3f}  {loss_synth:>10.4f}  {t_synth:>7.0f}s")
        delta = f1_opus - f1_synth
        sign  = "+" if delta >= 0 else ""
        winner = "Opus" if delta > 0 else ("Sintético" if delta < 0 else "EMPATE")
        print(f"\n  Delta F1 (Opus - Sintético): {sign}{delta:.3f}  → Ganador: {winner}")
        if delta > 0.05:
            print("  >> La calidad Opus mejora el F1 de forma clara (>0.05).")
            print("     El razonamiento real de Opus enseña patrones más generalizables.")
        elif delta < -0.05:
            print("  >> Datos sintéticos ganan — Opus data puede ser más difícil (diff 2-5).")
            print("     Considera filtrar difficulty_range=(1,2) para nivel=1 comparable.")
        else:
            print("  >> Diferencia pequeña (<0.05) — calidad similar en este config.")

    print(f"\n  Tiempo total: {total:.0f}s")
    print(SEP)


if __name__ == "__main__":
    main()
