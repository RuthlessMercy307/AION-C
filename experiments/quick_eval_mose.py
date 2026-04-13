"""
experiments/quick_eval_mose.py
================================
Diagnóstico rápido: entrena 300 steps con 500 ejemplos por dominio
y evalúa Word F1 sobre los mismos ejemplos de entrenamiento.

Propósito: confirmar que la arquitectura puede aprender (F1 > 0).
Si F1 = 0 incluso en train data = problema en decoder/tokenizer.
Si F1 > 0.3 en train = overfitting esperado, arquitectura OK.

Uso:
    cd AION-C
    python -m experiments.quick_eval_mose
"""

from __future__ import annotations

import sys, os, math, random, re as _re, time
from collections import Counter as _Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from router.pipeline           import MoSEPipeline, MoSEConfig, MoSEOutput
from synth.causal_graph_gen    import CausalGraphGenerator
from synth.code_graph_gen      import CodeGraphGenerator
from synth.math_graph_gen      import MathGraphGenerator
from synth.narrative_graph_gen import NarrativeGraphGenerator
from synth.social_graph_gen    import SocialGraphGenerator
from orchestrator.model        import MOTOR_NAMES
from cre                       import PyGStyleBatcher

torch.set_num_threads(4)
torch.manual_seed(42)
random.seed(42)

DEVICE = torch.device("cpu")
BATCH  = 16
PAD, BOS, EOS, UNK = 0, 1, 2, 3

# ─── Generadores ─────────────────────────────────────────────────────────────

GENS: Dict[str, object] = {
    "cora":    CausalGraphGenerator(),
    "forge_c": CodeGraphGenerator(),
    "axiom":   MathGraphGenerator(),
    "muse":    NarrativeGraphGenerator(),
    "empathy": SocialGraphGenerator(),
}

MOTOR_HINTS: Dict[str, str] = {
    "cora":    "",
    "forge_c": "function ",
    "axiom":   "theorem ",
    "muse":    "historia ",
    "empathy": "siente ",
}

# ─── Tokenizador ─────────────────────────────────────────────────────────────

class SimpleTokenizer:
    PAD = 0; BOS = 1; EOS = 2; UNK = 3
    _OFFSET = 4

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self._word2id: Dict[str, int] = {}
        self._words:   List[str] = []

    def build_vocab(self, texts: List[str]) -> "SimpleTokenizer":
        freq: _Counter = _Counter()
        for t in texts:
            freq.update(_re.findall(r"\w+", t.lower()))
        max_w = self.vocab_size - self._OFFSET
        self._words   = [w for w, _ in freq.most_common(max_w)]
        self._word2id = {w: i + self._OFFSET for i, w in enumerate(self._words)}
        return self

    def encode(self, text: str, max_len: int) -> List[int]:
        toks = _re.findall(r"\w+", text.lower())
        ids  = [self.BOS] + [self._word2id.get(t, self.UNK) for t in toks] + [self.EOS]
        return ids[:max_len]

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i == self.EOS:
                break
            if i >= self._OFFSET:
                idx = i - self._OFFSET
                out.append(self._words[idx] if idx < len(self._words) else "<unk>")
        return " ".join(out)


# ─── Helpers ─────────────────────────────────────────────────────────────────

_batcher = PyGStyleBatcher()


def batched_forward(pipe, token_ids, hint=None, precomp=None):
    B = token_ids.shape[0]
    D = pipe.config.hidden_dim
    K = pipe.config.motor_max_nodes

    concepts = pipe.encoder(token_ids)
    orch_out = pipe.orchestrator(concepts, hint)

    motor_cryst = {}
    for act in orch_out.activations:
        if precomp is not None and act.motor_name == precomp["motor"]:
            continue
        with torch.no_grad():
            motor_cryst[act.motor_name] = pipe.motors[act.motor_name].build_graph(concepts)

    motor_cre_outs = {}
    for act in orch_out.activations:
        motor = pipe.motors[act.motor_name]
        graphs_b, node_feats_b, valid_b = [], [], []

        if precomp is not None and act.motor_name == precomp["motor"]:
            for b in range(B):
                if precomp["ncounts"][b] > 0:
                    graphs_b.append(precomp["graphs"][b])
                    node_feats_b.append(precomp["node_vecs"][b].detach().requires_grad_(True))
                    valid_b.append(b)
        else:
            co = motor_cryst[act.motor_name]
            for b in range(B):
                n = co.node_counts[b]
                if n > 0:
                    graphs_b.append(co.graphs[b])
                    node_feats_b.append(co.node_vectors[b, :n].detach().requires_grad_(True))
                    valid_b.append(b)

        if not graphs_b:
            motor_cre_outs[act.motor_name] = [None] * B
            continue

        batched  = _batcher.batch(graphs_b, node_feats_b)
        cre_outs = motor.cre.forward_batched(batched, n_iterations=act.n_iterations)
        cre_per_b = [None] * B
        for i, b in enumerate(valid_b):
            cre_per_b[b] = cre_outs[i]
        motor_cre_outs[act.motor_name] = cre_per_b

    all_graph_reprs = []
    last_unif_out = None
    for b in range(B):
        reprs = []
        for act in orch_out.activations:
            m  = pipe.motors[act.motor_name]
            co = motor_cre_outs[act.motor_name][b]
            reprs.append(
                m.get_graph_repr(co, k_nodes=K) if co is not None
                else torch.zeros(K, D)
            )
        last_unif_out = pipe.unifier(reprs)
        all_graph_reprs.append(last_unif_out.unified)

    graph_repr = torch.stack(all_graph_reprs, dim=0)
    dec_out    = pipe.decoder(token_ids, graph_repr, concepts)
    return dec_out


def word_f1(pred: str, ref: str) -> float:
    p, r = set(pred.lower().split()), set(ref.lower().split())
    if not p or not r:
        return 0.0
    tp = len(p & r)
    pr, rc = tp / len(p), tp / len(r)
    return round(2 * pr * rc / (pr + rc), 3) if (pr + rc) else 0.0


@torch.no_grad()
def greedy_decode(pipeline, prompt, hint, tok, cfg, max_new=24):
    pipeline.eval()
    ids = tok.encode(prompt, cfg.dec_max_seq_len - max_new)
    cur = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new):
        if cur.shape[1] >= cfg.dec_max_seq_len:
            break
        out = pipeline(cur, query_text=hint or None)
        nxt = out.logits[0, -1].argmax().item()
        if nxt == EOS:
            break
        cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    pipeline.train()
    return tok.decode(cur[0, len(ids):].tolist())


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 68
    N_TRAIN = 100   # pequeño para forzar overfitting rápido
    N_STEPS = 2000  # suficiente para bajar loss a <1.0 con 100 ejemplos
    N_EVAL  = 20    # ejemplos de eval por dominio (sobre train data)

    print(SEP)
    print("  quick_eval_mose — Diagnóstico F1")
    print(f"  train={N_TRAIN}/dom, steps={N_STEPS}, eval={N_EVAL}/dom en TRAIN data")
    print(SEP)

    # Config idéntica al script principal
    base = MoSEConfig.tiny()
    cfg  = MoSEConfig(
        hidden_dim=128, vocab_size=base.vocab_size,
        enc_n_layers=base.enc_n_layers, enc_state_dim=base.enc_state_dim,
        enc_expand=base.enc_expand, enc_d_conv=base.enc_d_conv,
        enc_ffn_mult=base.enc_ffn_mult, orch_mlp_hidden=64,
        orch_max_motors=base.orch_max_motors,
        orch_min_confidence=base.orch_min_confidence,
        motor_max_nodes=base.motor_max_nodes,
        motor_n_heads=base.motor_n_heads, motor_threshold=base.motor_threshold,
        unif_n_heads=base.unif_n_heads, dec_n_layers=base.dec_n_layers,
        dec_n_heads=base.dec_n_heads, dec_max_seq_len=base.dec_max_seq_len,
        dec_state_dim=base.dec_state_dim, dec_expand=base.dec_expand,
        dec_d_conv=base.dec_d_conv, dec_ffn_mult=base.dec_ffn_mult,
    )

    # Vocabulario
    print("Construyendo vocab...", end=" ", flush=True)
    vocab_texts = []
    for domain in MOTOR_NAMES:
        for _ in range(200):
            ex = GENS[domain].generate(level=random.choice([1, 2]))
            vocab_texts.append(ex.problem_text + " " + ex.answer)
    tok = SimpleTokenizer(cfg.vocab_size).build_vocab(vocab_texts)
    print(f"OK — {len(tok._words)} palabras (vocab_size={cfg.vocab_size})", flush=True)

    # Pipeline
    pipeline = MoSEPipeline(cfg).to(DEVICE)
    pipeline.encoder.enable_gradient_checkpointing()
    pipeline.decoder.enable_gradient_checkpointing()
    total_p = pipeline.parameter_breakdown()["total_unique"]
    print(f"Params: {total_p:,}", flush=True)

    # Generar datos de entrenamiento y almacenar las preguntas/respuestas originales
    print(f"Generando {N_TRAIN} ejemplos/dominio...", end=" ", flush=True)
    train_data: Dict[str, List[Tuple[str, str, List[int]]]] = {}
    for domain in MOTOR_NAMES:
        items = []
        for _ in range(N_TRAIN):
            ex   = GENS[domain].generate(level=random.choice([1, 2]))
            text = ex.problem_text + " " + ex.answer
            ids  = tok.encode(text, cfg.dec_max_seq_len)
            items.append((ex.problem_text, ex.answer, ids))
        train_data[domain] = items
    print("OK", flush=True)

    # Precomputar grafos
    print("Precomputando grafos...", end=" ", flush=True)
    t0 = time.perf_counter()
    precomp_store: Dict[str, dict] = {}
    pipeline.eval()
    with torch.no_grad():
        for domain in MOTOR_NAMES:
            motor  = pipeline.motors[domain]
            all_ids = [x[2] for x in train_data[domain]]
            graphs, nvecs, ncounts = [], [], []
            for start in range(0, len(all_ids), 32):
                chunk  = all_ids[start:start+32]
                maxl   = max(len(s) for s in chunk)
                padded = [s + [PAD]*(maxl-len(s)) for s in chunk]
                ids_t  = torch.tensor(padded, dtype=torch.long)
                conc   = pipeline.encoder(ids_t)
                cryst  = motor.build_graph(conc)
                for b in range(len(chunk)):
                    nk = cryst.node_counts[b]
                    graphs.append(cryst.graphs[b])
                    nvecs.append(cryst.node_vectors[b, :nk].cpu())
                    ncounts.append(nk)
            precomp_store[domain] = {
                "graphs": graphs, "node_vecs": nvecs, "ncounts": ncounts
            }
    pipeline.train()
    print(f"OK ({time.perf_counter()-t0:.1f}s)", flush=True)

    def get_batch(domain, bs):
        all_ids = [x[2] for x in train_data[domain]]
        pc = precomp_store[domain]
        idx = random.choices(range(len(all_ids)), k=bs)
        seqs = [all_ids[i] for i in idx]
        maxl = max(len(s) for s in seqs)
        padded = [s + [PAD]*(maxl-len(s)) for s in seqs]
        return torch.tensor(padded, dtype=torch.long), {
            "motor":     domain,
            "graphs":    [pc["graphs"][i]    for i in idx],
            "node_vecs": [pc["node_vecs"][i] for i in idx],
            "ncounts":   [pc["ncounts"][i]   for i in idx],
        }

    # ── Entrenamiento: 300 steps ──────────────────────────────────────────────
    print()
    print(f"Entrenando {N_STEPS} steps...", flush=True)
    opt = torch.optim.AdamW(pipeline.parameters(), lr=1e-4, weight_decay=0.01)
    pipeline.train()
    t_start = time.perf_counter()
    losses = []

    for step in range(1, N_STEPS + 1):
        domain = random.choice(MOTOR_NAMES)
        ids, precomp = get_batch(domain, BATCH)
        hint = MOTOR_HINTS[domain]
        dec_out = batched_forward(pipeline, ids, hint or None, precomp)
        loss = F.cross_entropy(
            dec_out.logits[:, :-1].reshape(-1, cfg.vocab_size),
            ids[:, 1:].reshape(-1),
            ignore_index=PAD,
        )
        if math.isfinite(loss.item()):
            loss.backward()
            nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
            opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item() if math.isfinite(loss.item()) else float("nan"))

        if step % 50 == 0:
            avg = sum(x for x in losses[-50:] if math.isfinite(x)) / max(1, sum(1 for x in losses[-50:] if math.isfinite(x)))
            print(f"  step {step:>4}  train_loss={avg:.4f}", flush=True)

        # Eval intermedio cada 500 steps para ver si F1 empieza a subir
        if step % 500 == 0:
            pipeline.eval()
            sample_domain = MOTOR_NAMES[0]
            hint = MOTOR_HINTS[sample_domain]
            items = random.sample(train_data[sample_domain], min(5, len(train_data[sample_domain])))
            spot_f1s = []
            for q, a, _ in items:
                p = greedy_decode(pipeline, q, hint, tok, cfg, max_new=min(len(a.split())*2+4, 24))
                spot_f1s.append(word_f1(p, a))
            spot = sum(spot_f1s) / len(spot_f1s) if spot_f1s else 0.0
            print(f"  [spot F1 @ step {step} on '{sample_domain}' train] = {spot:.3f}", flush=True)
            pipeline.train()

    t_el = time.perf_counter() - t_start
    final_loss = sum(x for x in losses[-20:] if math.isfinite(x)) / 20
    print(f"Entrenamiento terminado: {t_el:.0f}s — loss final={final_loss:.4f}", flush=True)

    # ── Eval Word F1 sobre TRAIN data ────────────────────────────────────────
    print()
    print(SEP)
    print(f"  Eval Word F1 -- {N_EVAL} ejemplos de TRAIN data por dominio")
    print(SEP)

    f1_per_domain: Dict[str, float] = {}
    pipeline.eval()

    for domain in MOTOR_NAMES:
        hint = MOTOR_HINTS[domain]
        items = random.sample(train_data[domain], min(N_EVAL, len(train_data[domain])))
        f1s = []
        print(f"\n  [{domain}]")
        for i, (q, a, _) in enumerate(items):
            max_new = min(len(a.split()) * 2 + 4, 32)
            p = greedy_decode(pipeline, q, hint, tok, cfg, max_new=max_new)
            f1 = word_f1(p, a)
            f1s.append(f1)
            if i < 5:  # mostrar primeros 5
                # Limpiar caracteres no-ASCII para evitar UnicodeEncodeError
                q_s = q[:55].encode('ascii', 'replace').decode('ascii')
                a_s = a[:55].encode('ascii', 'replace').decode('ascii')
                p_s = p[:55].encode('ascii', 'replace').decode('ascii')
                print(f"    Q: {q_s}")
                print(f"    A: {a_s}")
                print(f"    P: {p_s}  F1={f1:.2f}")
        mean = sum(f1s) / len(f1s) if f1s else 0.0
        f1_per_domain[domain] = mean
        print(f"  -> mean F1 [{domain}] = {mean:.3f}")

    pipeline.train()

    # ── Resumen ──────────────────────────────────────────────────────────────
    print()
    print(SEP)
    mean_f1 = sum(f1_per_domain.values()) / len(f1_per_domain)
    print(f"  RESUMEN (eval sobre TRAIN data, {N_STEPS} steps)")
    for d, f1 in f1_per_domain.items():
        bar = "#" * int(f1 * 20)
        print(f"    {d:<10}: {f1:.3f}  [{bar:<20}]")
    print(f"    {'MEDIA':<10}: {mean_f1:.3f}")
    print()
    if mean_f1 > 0.3:
        print("  >> ARQUITECTURA OK — F1 > 0.3 en datos de entrenamiento.")
        print("     El overfitting es esperable con 2000 ejemplos / 4.8M params.")
        print("     Solución: dataset mucho más grande (50k-100k ejemplos).")
    elif mean_f1 > 0.05:
        print("  >> F1 bajo pero > 0 — modelo está aprendiendo lentamente.")
        print("     Puede necesitar más steps o LR más alto.")
    else:
        print("  >> ALERTA: F1 ~ 0 incluso en train data.")
        print("     Problema en decoder, greedy_decode, o tokenizer.")
    print(SEP)


if __name__ == "__main__":
    main()
