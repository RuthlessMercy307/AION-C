"""
experiments/train_mose_cpu.py
==============================
Entrenamiento MoSE en CPU — word-level tokenizer, hidden_dim=128.

Fases:
  Phase 1 — Shared pretraining mezclado (5 dominios), hasta convergencia.
             Val cada 100 steps. Early stop si no mejora en 300 steps.
             ETA al target imprimida cada 200 steps. Max 20 000 steps.
  Phase 2 — 5 motores en paralelo (multiprocessing.Process), encoder/decoder
             congelados. Cada motor entrena solo sus pesos únicos.

Uso:
    cd AION-C
    python -m experiments.train_mose_cpu
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random
import re as _re
import time
import multiprocessing
import tempfile
from collections import Counter as _Counter
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

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

# ─────────────────────────────────────────────────────────────────────────────
# Hiperparámetros
# ─────────────────────────────────────────────────────────────────────────────

DEVICE     = torch.device("cpu")
BATCH      = 16
N_EXAMPLES = 2000      # ejemplos de entrenamiento por dominio
N_VAL      = 200       # ejemplos de validación por dominio

# Convergencia Phase 1
MAX_STEPS           = 50_000
EARLY_STOP_PATIENCE = 300    # steps sin mejora en val_loss
VAL_TARGET          = 0.5    # meta de val_loss
VAL_INTERVAL        = 500    # steps entre evaluaciones de val
ETA_INTERVAL        = 1000   # steps entre impresión de ETA
LOG_INTERVAL        = 30.0   # segundos entre logs de progreso

LR_PHASE1 = 1e-4
LR_MOTOR  = 1e-4
LR_SHARED = 2e-5

# Phase 2 por motor
P2_MAX_STEPS           = 10_000
P2_EARLY_STOP_PATIENCE = 1000
P2_VAL_INTERVAL        = 500
P2_ETA_INTERVAL        = 1000

torch.manual_seed(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizador word-level
# ─────────────────────────────────────────────────────────────────────────────

PAD, BOS, EOS, UNK = 0, 1, 2, 3


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

    @property
    def vocab_info(self) -> str:
        return f"{len(self._words)} palabras (vocab_size={self.vocab_size})"

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


TOKENIZER: Optional[SimpleTokenizer] = None


def encode(text: str, max_len: int) -> List[int]:
    return TOKENIZER.encode(text, max_len)


def decode(ids: List[int]) -> str:
    return TOKENIZER.decode(ids)


# ─────────────────────────────────────────────────────────────────────────────
# Generadores
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset pre-computado
# ─────────────────────────────────────────────────────────────────────────────

class PrecomputedDataset:
    def __init__(self, domain: str, n: int, tok: SimpleTokenizer,
                 max_len: int) -> None:
        gen  = GENS[domain]
        hint = MOTOR_HINTS[domain]
        ids_list: List[List[int]] = []
        for _ in range(n):
            lvl  = random.choice([1, 2])
            ex   = gen.generate(level=lvl)
            text = ex.problem_text + " " + ex.answer
            ids_list.append(tok.encode(text, max_len))

        self.domain  = domain
        self.hint    = hint
        self._ids    = ids_list
        self.n       = len(ids_list)
        self._graphs:      Optional[List] = None
        self._node_vecs:   Optional[List[torch.Tensor]] = None
        self._node_counts: Optional[List[int]] = None

    def precompute_graphs(self, pipeline: MoSEPipeline,
                          batch_size: int = 8) -> None:
        motor   = pipeline.motors[self.domain]
        graphs: List = []
        nvecs:  List[torch.Tensor] = []
        ncounts: List[int] = []

        pipeline.eval()
        with torch.no_grad():
            for start in range(0, self.n, batch_size):
                chunk  = self._ids[start:start + batch_size]
                maxl   = max(len(s) for s in chunk)
                padded = [s + [PAD] * (maxl - len(s)) for s in chunk]
                ids_t  = torch.tensor(padded, dtype=torch.long)
                conc   = pipeline.encoder(ids_t)
                cryst  = motor.build_graph(conc)
                for b in range(len(chunk)):
                    nk = cryst.node_counts[b]
                    graphs.append(cryst.graphs[b])
                    nvecs.append(cryst.node_vectors[b, :nk].cpu())
                    ncounts.append(nk)

        self._graphs      = graphs
        self._node_vecs   = nvecs
        self._node_counts = ncounts
        pipeline.train()

    def get_batch(self, bs: int) -> Tuple[torch.Tensor, Optional[dict]]:
        indices  = random.choices(range(self.n), k=bs)
        seqs     = [self._ids[i] for i in indices]
        maxl     = max(len(s) for s in seqs)
        padded   = [s + [PAD] * (maxl - len(s)) for s in seqs]
        token_ids = torch.tensor(padded, dtype=torch.long)

        if self._graphs is None:
            return token_ids, None

        precomp = {
            "motor":     self.domain,
            "graphs":    [self._graphs[i]    for i in indices],
            "node_vecs": [self._node_vecs[i] for i in indices],
            "ncounts":   [self._node_counts[i] for i in indices],
        }
        return token_ids, precomp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pbar(done: int, total: int, w: int = 20) -> str:
    f      = done / max(total, 1)
    filled = int(f * w)
    return f"[{'#'*filled}{'.'*(w-filled)}] {f*100:>5.1f}%"


def _fmt_eta(secs: float) -> str:
    secs = int(max(0, secs))
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


_batcher = PyGStyleBatcher()


# ─────────────────────────────────────────────────────────────────────────────
# batched_pipeline_forward
# ─────────────────────────────────────────────────────────────────────────────

def batched_pipeline_forward(
    pipe:      MoSEPipeline,
    token_ids: torch.Tensor,
    query_text: Optional[str] = None,
    precomp:   Optional[dict] = None,
) -> MoSEOutput:
    B      = token_ids.shape[0]
    D      = pipe.config.hidden_dim
    K      = pipe.config.motor_max_nodes
    device = token_ids.device
    dtype  = pipe.encoder.token_embedding.weight.dtype

    concepts = pipe.encoder(token_ids)
    orch_out = pipe.orchestrator(concepts, query_text)

    motor_cryst: dict = {}
    for act in orch_out.activations:
        if precomp is not None and act.motor_name == precomp["motor"]:
            continue
        with torch.no_grad():
            motor_cryst[act.motor_name] = \
                pipe.motors[act.motor_name].build_graph(concepts)

    motor_cre_outs: dict = {}
    for act in orch_out.activations:
        motor         = pipe.motors[act.motor_name]
        graphs_b:     List = []
        node_feats_b: List[torch.Tensor] = []
        valid_b:      List[int] = []

        if precomp is not None and act.motor_name == precomp["motor"]:
            for b in range(B):
                if precomp["ncounts"][b] > 0:
                    graphs_b.append(precomp["graphs"][b])
                    node_feats_b.append(
                        precomp["node_vecs"][b].detach().requires_grad_(True))
                    valid_b.append(b)
        else:
            co = motor_cryst[act.motor_name]
            for b in range(B):
                n = co.node_counts[b]
                if n > 0:
                    graphs_b.append(co.graphs[b])
                    node_feats_b.append(
                        co.node_vectors[b, :n].detach().requires_grad_(True))
                    valid_b.append(b)

        if not graphs_b:
            motor_cre_outs[act.motor_name] = [None] * B
            continue

        batched  = _batcher.batch(graphs_b, node_feats_b)
        cre_outs = motor.cre.forward_batched(batched, n_iterations=act.n_iterations)
        cre_per_b: List = [None] * B
        for i, b in enumerate(valid_b):
            cre_per_b[b] = cre_outs[i]
        motor_cre_outs[act.motor_name] = cre_per_b

    all_graph_reprs: List[torch.Tensor] = []
    last_unif_out = None
    for b in range(B):
        reprs: List[torch.Tensor] = []
        for act in orch_out.activations:
            m   = pipe.motors[act.motor_name]
            co  = motor_cre_outs[act.motor_name][b]
            reprs.append(
                m.get_graph_repr(co, k_nodes=K) if co is not None
                else torch.zeros(K, D, device=device, dtype=dtype)
            )
        last_unif_out = pipe.unifier(reprs)
        all_graph_reprs.append(last_unif_out.unified)

    graph_repr = torch.stack(all_graph_reprs, dim=0)
    dec_out    = pipe.decoder(token_ids, graph_repr, concepts)

    return MoSEOutput(
        logits        = dec_out.logits,
        anchor_logits = dec_out.anchor_logits,
        confidence    = dec_out.confidence,
        needs_clarif  = dec_out.needs_clarification,
        graph_repr    = graph_repr,
        orchestrator  = orch_out,
        unifier       = last_unif_out,
        active_motors = orch_out.motor_names,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validación
# ─────────────────────────────────────────────────────────────────────────────

def compute_val_loss(pipeline: MoSEPipeline,
                     val_datasets: Dict[str, PrecomputedDataset],
                     vocab_size: int,
                     samples_per_domain: int = 10) -> float:
    """Evalúa en `samples_per_domain` ejemplos por dominio y devuelve la media."""
    pipeline.eval()
    losses: List[float] = []
    with torch.no_grad():
        for domain, ds in val_datasets.items():
            for _ in range(0, samples_per_domain, BATCH):
                bs = min(BATCH, samples_per_domain)
                ids, precomp = ds.get_batch(bs)
                out  = batched_pipeline_forward(pipeline, ids, precomp=precomp)
                loss = F.cross_entropy(
                    out.logits[:, :-1].reshape(-1, vocab_size),
                    ids[:, 1:].reshape(-1),
                    ignore_index=PAD,
                )
                losses.append(loss.item())
    pipeline.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación greedy + Word F1
# ─────────────────────────────────────────────────────────────────────────────

def word_f1(pred: str, ref: str) -> float:
    p, r = set(pred.lower().split()), set(ref.lower().split())
    if not p or not r:
        return 0.0
    tp = len(p & r)
    pr, rc = tp / len(p), tp / len(r)
    return round(2 * pr * rc / (pr + rc), 3) if (pr + rc) else 0.0


def greedy_decode(pipeline: MoSEPipeline, prompt: str, hint: str,
                  tok: SimpleTokenizer, cfg: MoSEConfig,
                  max_new: int = 24) -> str:
    """Genera tokens usando batched_pipeline_forward — mismo camino que el training."""
    pipeline.eval()
    ids = tok.encode(prompt, cfg.dec_max_seq_len - max_new)
    cur = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_new):
            if cur.shape[1] >= cfg.dec_max_seq_len:
                break
            out = batched_pipeline_forward(pipeline, cur, hint or None, precomp=None)
            nxt = out.logits[0, -1].argmax().item()
            if nxt == EOS:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    pipeline.train()
    return tok.decode(cur[0, len(ids):].tolist())


def eval_motor(pipeline: MoSEPipeline, domain: str,
               tok: SimpleTokenizer, cfg: MoSEConfig,
               n: int = 10, prefix: str = "") -> float:
    gen  = GENS[domain]
    hint = MOTOR_HINTS[domain]
    f1s  = []
    print(f"{prefix}  Eval [{domain}] ({n} ejemplos):")
    for i in range(n):
        ex  = gen.generate(level=random.choice([1, 2]))
        q, a = ex.problem_text, ex.answer
        max_new = min(len(a.split()) * 2 + 4, 32)
        p   = greedy_decode(pipeline, q, hint, tok, cfg, max_new=max_new)
        f1  = word_f1(p, a)
        f1s.append(f1)
        print(f"{prefix}    [{i+1}] Q: {q[:50]}")
        print(f"{prefix}         A: {a[:50]}")
        print(f"{prefix}         P: {p[:50]}  F1={f1:.2f}")
    mean = sum(f1s) / len(f1s) if f1s else 0.0
    print(f"{prefix}         -> mean Word F1 = {mean:.3f}")
    return mean


# ─────────────────────────────────────────────────────────────────────────────
# Worker para Phase 2 paralelo
# ─────────────────────────────────────────────────────────────────────────────

def _motor_worker(domain: str,
                  state_dict_path: str,
                  cfg_kwargs: dict,
                  ids_train: List[List[int]],
                  ids_val:   List[List[int]],
                  tok_words:  List[str],
                  tok_w2id:   Dict[str, int],
                  vocab_size: int,
                  result_path: str) -> None:
    """
    Proceso hijo: entrena solo los pesos del motor `domain`.
    Estado del pipeline cargado desde state_dict_path.
    Imprime progreso con prefijo [DOMAIN].
    Resultado (motor state dict) guardado en result_path.
    """
    torch.set_num_threads(2)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

    PFX = f"[{domain.upper():<8}]"
    print(f"{PFX} Inicio proceso motor", flush=True)

    # ── Recrear tokenizer ─────────────────────────────────────────────────
    tok = SimpleTokenizer(vocab_size)
    tok._words   = tok_words
    tok._word2id = tok_w2id

    # ── Recrear config y pipeline ─────────────────────────────────────────
    cfg = MoSEConfig(**cfg_kwargs)
    pipeline = MoSEPipeline(cfg)
    state = torch.load(state_dict_path, map_location="cpu")
    pipeline.load_state_dict(state)

    # ── Congelar todo excepto el motor propio ─────────────────────────────
    for name, p in pipeline.named_parameters():
        if f"motors.{domain}" not in name:
            p.requires_grad_(False)

    trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"{PFX} Params entrenables: {trainable:,}", flush=True)

    # ── Datasets ──────────────────────────────────────────────────────────
    # Reconstruir datasets desde las listas de ids
    class _RawDS:
        def __init__(self, domain, ids_list, hint):
            self.domain = domain
            self.hint   = hint
            self._ids   = ids_list
            self.n      = len(ids_list)
            self._graphs = None
            self._node_vecs = None
            self._node_counts = None

        def precompute_graphs(self, pipeline, batch_size=8):
            motor   = pipeline.motors[self.domain]
            graphs, nvecs, ncounts = [], [], []
            pipeline.eval()
            with torch.no_grad():
                for start in range(0, self.n, batch_size):
                    chunk  = self._ids[start:start+batch_size]
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
            self._graphs, self._node_vecs, self._node_counts = graphs, nvecs, ncounts
            pipeline.train()

        def get_batch(self, bs):
            idx  = random.choices(range(self.n), k=bs)
            seqs = [self._ids[i] for i in idx]
            maxl = max(len(s) for s in seqs)
            pad  = [s + [PAD]*(maxl-len(s)) for s in seqs]
            tids = torch.tensor(pad, dtype=torch.long)
            if self._graphs is None:
                return tids, None
            return tids, {
                "motor":     self.domain,
                "graphs":    [self._graphs[i]    for i in idx],
                "node_vecs": [self._node_vecs[i] for i in idx],
                "ncounts":   [self._node_counts[i] for i in idx],
            }

    ds_train = _RawDS(domain, ids_train, MOTOR_HINTS[domain])
    ds_val   = _RawDS(domain, ids_val,   MOTOR_HINTS[domain])

    print(f"{PFX} Precomputando grafos (train={len(ids_train)}, val={len(ids_val)})...",
          flush=True)
    ds_train.precompute_graphs(pipeline)
    ds_val.precompute_graphs(pipeline)
    print(f"{PFX} Grafos listos.", flush=True)

    # ── Optimizer ─────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in pipeline.parameters() if p.requires_grad],
        lr=LR_MOTOR, weight_decay=0.01,
    )

    best_val   = float("inf")
    no_improve = 0
    val_hist:  List[Tuple[int, float]] = []
    train_losses: List[float] = []
    eta_ref_loss = None
    eta_ref_step = 0
    t_start = time.perf_counter()
    last_log = t_start

    pipeline.train()

    for step in range(1, P2_MAX_STEPS + 1):
        ids, precomp = ds_train.get_batch(BATCH)
        out  = batched_pipeline_forward(pipeline, ids, ds_train.hint or None, precomp)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, cfg.vocab_size),
            ids[:, 1:].reshape(-1),
            ignore_index=PAD,
        )
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            opt.zero_grad(set_to_none=True)
            train_losses.append(loss_val)
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in pipeline.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        train_losses.append(loss_val)

        # ── Validación ────────────────────────────────────────────────────
        if step % P2_VAL_INTERVAL == 0:
            pipeline.eval()
            v_losses: List[float] = []
            with torch.no_grad():
                for _ in range(0, min(32, ds_val.n), BATCH):
                    vi, vp = ds_val.get_batch(BATCH)
                    vo = batched_pipeline_forward(pipeline, vi, precomp=vp)
                    vl = F.cross_entropy(
                        vo.logits[:, :-1].reshape(-1, cfg.vocab_size),
                        vi[:, 1:].reshape(-1), ignore_index=PAD,
                    )
                    v_losses.append(vl.item())
            pipeline.train()
            val_loss = sum(v_losses) / len(v_losses)
            val_hist.append((step, val_loss))

            # Early stopping
            if val_loss < best_val - 1e-4:
                best_val   = val_loss
                no_improve = 0
            else:
                no_improve += P2_VAL_INTERVAL

            # ETA
            eta_str = ""
            if step % P2_ETA_INTERVAL == 0:
                if eta_ref_loss is not None and val_loss < eta_ref_loss:
                    rate = (eta_ref_loss - val_loss) / (step - eta_ref_step)
                    if rate > 0 and val_loss > VAL_TARGET:
                        sn  = (val_loss - VAL_TARGET) / rate
                        eta_str = f"  ETA_target={_fmt_eta(sn * (time.perf_counter()-t_start)/step)}"
                eta_ref_loss = val_loss
                eta_ref_step = step

            t_el  = time.perf_counter() - t_start
            t_avg = sum(train_losses[-50:]) / min(50, len(train_losses))
            print(f"{PFX} {_pbar(step, P2_MAX_STEPS)} step {step:>5}  "
                  f"train={t_avg:.4f}  val={val_loss:.4f}  "
                  f"best={best_val:.4f}  ni={no_improve}{eta_str}",
                  flush=True)

            if val_loss <= VAL_TARGET:
                print(f"{PFX} TARGET ALCANZADO val={val_loss:.4f} <= {VAL_TARGET}",
                      flush=True)
                break
            if no_improve >= P2_EARLY_STOP_PATIENCE:
                print(f"{PFX} Early stop ({no_improve} steps sin mejora)", flush=True)
                break
        elif time.perf_counter() - last_log >= LOG_INTERVAL:
            last_log = time.perf_counter()
            t_avg = sum(train_losses[-50:]) / min(50, len(train_losses))
            t_el  = time.perf_counter() - t_start
            print(f"{PFX} {_pbar(step, P2_MAX_STEPS)} step {step:>5}  "
                  f"train={t_avg:.4f}  {t_el:.0f}s",
                  flush=True)

    t_total = time.perf_counter() - t_start
    print(f"{PFX} Fin. best_val={best_val:.4f}  {t_total:.0f}s  "
          f"({t_total/60:.1f} min)", flush=True)

    # ── Guardar solo los pesos del motor ──────────────────────────────────
    motor_state = {
        k: v for k, v in pipeline.state_dict().items()
        if f"motors.{domain}" in k
    }
    torch.save({"domain": domain, "state": motor_state, "best_val": best_val},
               result_path)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global TOKENIZER

    # Limitar threads para evitar sobrecalentamiento durante entrenamiento largo
    torch.set_num_threads(4)

    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

    SEP = "=" * 68
    sep = "-" * 68

    print(SEP)
    print(f"  AION-C / MoSE -- Entrenamiento CPU  "
          f"[word-level | hidden_dim=128 | train={N_EXAMPLES}/dom | val={N_VAL}/dom]")
    print(SEP)

    # ── Config: tiny con hidden_dim=128 ──────────────────────────────────────
    base = MoSEConfig.tiny()
    cfg  = MoSEConfig(
        hidden_dim       = 128,
        vocab_size       = base.vocab_size,       # 512
        enc_n_layers     = base.enc_n_layers,     # 2
        enc_state_dim    = base.enc_state_dim,    # 4
        enc_expand       = base.enc_expand,       # 2
        enc_d_conv       = base.enc_d_conv,       # 4
        enc_ffn_mult     = base.enc_ffn_mult,     # 2
        orch_mlp_hidden  = 64,                    # 64 (proporcional a 128)
        orch_max_motors  = base.orch_max_motors,
        orch_min_confidence = base.orch_min_confidence,
        motor_max_nodes  = base.motor_max_nodes,  # 8
        motor_n_heads    = base.motor_n_heads,    # 4 → head_dim=32, ok
        motor_threshold  = base.motor_threshold,
        unif_n_heads     = base.unif_n_heads,     # 4 → head_dim=32, ok
        dec_n_layers     = base.dec_n_layers,     # 2
        dec_n_heads      = base.dec_n_heads,      # 4 → head_dim=32, ok
        dec_max_seq_len  = base.dec_max_seq_len,  # 128
        dec_state_dim    = base.dec_state_dim,    # 4
        dec_expand       = base.dec_expand,       # 2
        dec_d_conv       = base.dec_d_conv,       # 4
        dec_ffn_mult     = base.dec_ffn_mult,     # 2
    )
    print(f"Config: hidden_dim={cfg.hidden_dim}, vocab_size={cfg.vocab_size}, "
          f"max_seq_len={cfg.dec_max_seq_len}, K={cfg.motor_max_nodes}", flush=True)

    # ── Vocabulario (500 ej/dominio × 5 = 2500) ──────────────────────────────
    print("Construyendo vocabulario (500 ej/dominio x 5 dominios)...",
          end=" ", flush=True)
    t0 = time.perf_counter()
    vocab_texts: List[str] = []
    for domain in MOTOR_NAMES:
        gen = GENS[domain]
        for _ in range(500):
            ex = gen.generate(level=random.choice([1, 2]))
            vocab_texts.append(ex.problem_text + " " + ex.answer)
    TOKENIZER = SimpleTokenizer(cfg.vocab_size).build_vocab(vocab_texts)
    print(f"OK ({time.perf_counter()-t0:.1f}s) -- {TOKENIZER.vocab_info}", flush=True)

    # ── Pipeline ─────────────────────────────────────────────────────────────
    pipeline = MoSEPipeline(cfg).to(DEVICE)
    pipeline.encoder.enable_gradient_checkpointing()
    pipeline.decoder.enable_gradient_checkpointing()

    bd = pipeline.parameter_breakdown()
    total_p = bd["total_unique"]
    print(f"Parametros: {total_p:,} unicos "
          f"(enc={bd['encoder']:,} / dec={bd['decoder']:,} / "
          f"motores={sum(v for k,v in bd.items() if k.startswith('motor_')):,})",
          flush=True)

    # ── Datasets de entrenamiento y validación ────────────────────────────────
    print(f"Generando datasets (train={N_EXAMPLES}, val={N_VAL}) x {len(MOTOR_NAMES)} dominios...",
          end=" ", flush=True)
    t0 = time.perf_counter()
    TRAIN_DS: Dict[str, PrecomputedDataset] = {
        d: PrecomputedDataset(d, N_EXAMPLES, TOKENIZER, cfg.dec_max_seq_len)
        for d in MOTOR_NAMES
    }
    VAL_DS: Dict[str, PrecomputedDataset] = {
        d: PrecomputedDataset(d, N_VAL, TOKENIZER, cfg.dec_max_seq_len)
        for d in MOTOR_NAMES
    }
    print(f"OK ({time.perf_counter()-t0:.1f}s)", flush=True)

    print("Precomputando grafos (train + val)...", end=" ", flush=True)
    t0 = time.perf_counter()
    for d in MOTOR_NAMES:
        TRAIN_DS[d].precompute_graphs(pipeline, batch_size=32)
        VAL_DS[d].precompute_graphs(pipeline, batch_size=32)
    print(f"OK ({time.perf_counter()-t0:.1f}s)", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: Shared Pretraining — hasta convergencia
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print(SEP)
    print(f"  Phase 1 -- Shared Pretraining")
    print(f"  (max {MAX_STEPS:,} steps | val cada {VAL_INTERVAL} | "
          f"early_stop={EARLY_STOP_PATIENCE} | target val<{VAL_TARGET})")
    print(SEP)

    opt1   = torch.optim.AdamW(pipeline.parameters(), lr=LR_PHASE1, weight_decay=0.01)
    sched1 = torch.optim.lr_scheduler.OneCycleLR(
        opt1, LR_PHASE1, total_steps=MAX_STEPS, pct_start=0.05, anneal_strategy="cos"
    )

    train_losses_ph1: List[float] = []
    best_val_ph1  = float("inf")
    no_improve_ph1 = 0
    val_hist_ph1:  List[Tuple[int, float]] = []
    eta_ref_loss:  Optional[float] = None
    eta_ref_step   = 0
    best_state_ph1 = None  # state dict at best val

    pipeline.train()
    t_ph1_start = time.perf_counter()
    last_log    = t_ph1_start
    stop_reason = f"max steps ({MAX_STEPS:,})"

    for step in range(1, MAX_STEPS + 1):
        domain = random.choice(MOTOR_NAMES)
        ids, precomp = TRAIN_DS[domain].get_batch(BATCH)
        hint = TRAIN_DS[domain].hint

        out  = batched_pipeline_forward(pipeline, ids, hint or None, precomp)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, cfg.vocab_size),
            ids[:, 1:].reshape(-1),
            ignore_index=PAD,
        )
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            opt1.zero_grad(set_to_none=True)
            sched1.step()
            train_losses_ph1.append(loss_val)
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
        opt1.step()
        opt1.zero_grad(set_to_none=True)
        sched1.step()
        train_losses_ph1.append(loss_val)

        # ── Validación cada VAL_INTERVAL steps ────────────────────────────
        if step % VAL_INTERVAL == 0:
            val_loss = compute_val_loss(
                pipeline, VAL_DS, cfg.vocab_size, samples_per_domain=32)
            val_hist_ph1.append((step, val_loss))

            if math.isfinite(val_loss) and val_loss < best_val_ph1 - 1e-4:
                best_val_ph1  = val_loss
                no_improve_ph1 = 0
                import copy
                best_state_ph1 = copy.deepcopy(pipeline.state_dict())
            else:
                no_improve_ph1 += VAL_INTERVAL

            # ETA al target cada ETA_INTERVAL steps
            eta_str = ""
            if step % ETA_INTERVAL == 0:
                if eta_ref_loss is not None and val_loss < eta_ref_loss:
                    rate = (eta_ref_loss - val_loss) / (step - eta_ref_step)
                    if rate > 0 and val_loss > VAL_TARGET:
                        sn  = (val_loss - VAL_TARGET) / rate
                        eta_s = sn * (time.perf_counter() - t_ph1_start) / step
                        eta_str = (f"  ETA_target(val<{VAL_TARGET})="
                                   f"{_fmt_eta(eta_s)} ({sn:.0f} steps)")
                eta_ref_loss = val_loss
                eta_ref_step = step

            t_avg = sum(train_losses_ph1[-50:]) / min(50, len(train_losses_ph1))
            t_el  = time.perf_counter() - t_ph1_start
            thr   = step * BATCH / t_el
            lr    = sched1.get_last_lr()[0]
            print(f"  {_pbar(step, MAX_STEPS)} step {step:>6}  "
                  f"train={t_avg:.4f}  val={val_loss:.4f}  "
                  f"best={best_val_ph1:.4f}  ni={no_improve_ph1}  "
                  f"lr={lr:.1e}  {thr:.1f} s/s{eta_str}",
                  flush=True)

            if val_loss <= VAL_TARGET:
                stop_reason = f"target alcanzado (val={val_loss:.4f})"
                break
            if no_improve_ph1 >= EARLY_STOP_PATIENCE:
                stop_reason = f"early stop ({no_improve_ph1} steps sin mejora)"
                break

        # ── Checkpoint cada 1000 steps ────────────────────────────────────
        if step % 1000 == 0:
            ckpt_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_file = os.path.join(ckpt_dir, f"mose_step{step:06d}.pt")
            torch.save({
                "step":       step,
                "state_dict": pipeline.state_dict(),
                "best_val":   best_val_ph1,
                "val_hist":   val_hist_ph1,
            }, ckpt_file)
            print(f"  [ckpt] guardado -> {ckpt_file}", flush=True)

        elif time.perf_counter() - last_log >= LOG_INTERVAL:
            last_log = time.perf_counter()
            t_avg = sum(train_losses_ph1[-50:]) / min(50, len(train_losses_ph1))
            t_el  = time.perf_counter() - t_ph1_start
            thr   = step * BATCH / t_el
            lr    = sched1.get_last_lr()[0]
            eta_s = (MAX_STEPS - step) * (t_el / step)
            print(f"  {_pbar(step, MAX_STEPS)} step {step:>6}  "
                  f"train={t_avg:.4f}  lr={lr:.1e}  "
                  f"{thr:.1f} s/s  ETA_max={_fmt_eta(eta_s)}",
                  flush=True)

    t_ph1 = time.perf_counter() - t_ph1_start
    l0 = sum(train_losses_ph1[:10]) / 10
    l1 = sum(train_losses_ph1[-10:]) / 10
    print(f"\nPhase 1 done | {stop_reason}")
    print(f"  train {l0:.4f} -> {l1:.4f} | "
          f"best_val={best_val_ph1:.4f} | "
          f"{step} steps | {t_ph1:.0f}s ({t_ph1/60:.1f} min)", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: Motor fine-tuning paralelo
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print(SEP)
    print("  Phase 2 -- Motor Fine-Tuning SECUENCIAL")
    print(f"  (1 proceso a la vez, encoder+decoder congelados, "
          f"torch.set_num_threads(2) por proceso)")
    print(SEP)

    # Restaurar mejor estado antes de Phase 2 (evita guardar modelo con NaN)
    if best_state_ph1 is not None:
        pipeline.load_state_dict(best_state_ph1)
        print(f"Restaurado mejor estado Phase 1 (val={best_val_ph1:.4f})", flush=True)

    # Guardar pipeline en disco (workers lo cargan desde ahí)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save(pipeline.state_dict(), ckpt_path)
    print(f"Checkpoint Phase 1 -> {ckpt_path}", flush=True)

    # Preparar args para cada worker
    cfg_kwargs  = {
        "hidden_dim":      cfg.hidden_dim,
        "vocab_size":      cfg.vocab_size,
        "enc_n_layers":    cfg.enc_n_layers,
        "enc_state_dim":   cfg.enc_state_dim,
        "enc_expand":      cfg.enc_expand,
        "enc_d_conv":      cfg.enc_d_conv,
        "enc_ffn_mult":    cfg.enc_ffn_mult,
        "orch_mlp_hidden": cfg.orch_mlp_hidden,
        "orch_max_motors": cfg.orch_max_motors,
        "orch_min_confidence": cfg.orch_min_confidence,
        "motor_max_nodes": cfg.motor_max_nodes,
        "motor_n_heads":   cfg.motor_n_heads,
        "motor_threshold": cfg.motor_threshold,
        "unif_n_heads":    cfg.unif_n_heads,
        "dec_n_layers":    cfg.dec_n_layers,
        "dec_n_heads":     cfg.dec_n_heads,
        "dec_max_seq_len": cfg.dec_max_seq_len,
        "dec_state_dim":   cfg.dec_state_dim,
        "dec_expand":      cfg.dec_expand,
        "dec_d_conv":      cfg.dec_d_conv,
        "dec_ffn_mult":    cfg.dec_ffn_mult,
    }
    tok_words = TOKENIZER._words
    tok_w2id  = TOKENIZER._word2id

    # Archivos de resultado (uno por motor)
    result_paths = {d: tempfile.mktemp(suffix=f"_{d}.pt") for d in MOTOR_NAMES}

    t_ph2_start = time.perf_counter()
    for domain in MOTOR_NAMES:
        print(f"  Lanzando proceso [{domain}]...", flush=True)
        p = multiprocessing.Process(
            target=_motor_worker,
            args=(
                domain,
                ckpt_path,
                cfg_kwargs,
                TRAIN_DS[domain]._ids,
                VAL_DS[domain]._ids,
                tok_words,
                tok_w2id,
                cfg.vocab_size,
                result_paths[domain],
            ),
            daemon=True,
        )
        p.start()
        print(f"  Proceso [{domain}] lanzado (pid={p.pid})", flush=True)
        p.join()
        status = "OK" if p.exitcode == 0 else f"ERROR (code={p.exitcode})"
        print(f"  Proceso [{domain}] terminó: {status}", flush=True)

    t_ph2 = time.perf_counter() - t_ph2_start
    print(f"\nPhase 2 done | {t_ph2:.0f}s ({t_ph2/60:.1f} min)", flush=True)

    # Cargar resultados de cada motor en el pipeline principal
    print("Cargando pesos de motores en pipeline...", flush=True)
    main_state = pipeline.state_dict()
    for domain in MOTOR_NAMES:
        rpath = result_paths[domain]
        if os.path.exists(rpath):
            res = torch.load(rpath, map_location="cpu")
            main_state.update(res["state"])
            bv  = res.get("best_val", float("inf"))
            print(f"  [{domain}] best_val_ph2={bv:.4f}", flush=True)
        else:
            print(f"  [{domain}] WARNING: resultado no encontrado", flush=True)
    pipeline.load_state_dict(main_state)

    # Limpiar temporales
    for path in [ckpt_path] + list(result_paths.values()):
        try:
            os.unlink(path)
        except OSError:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # Eval final: Word F1 por motor (10 ejemplos cada uno)
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print(SEP)
    print("  Eval final -- Word F1 por motor (10 ejemplos)")
    print(SEP)

    f1_per_motor: Dict[str, float] = {}
    for domain in MOTOR_NAMES:
        f1 = eval_motor(pipeline, domain, TOKENIZER, cfg, n=10)
        f1_per_motor[domain] = f1

    # ── Resumen ──────────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_ph1_start
    print()
    print(SEP)
    print("  Resumen")
    print(SEP)
    print(f"  {'Tokenizer':<24}: word-level ({len(TOKENIZER._words)} words)")
    print(f"  {'Parametros':<24}: {total_p:,}")
    print(f"  {'hidden_dim':<24}: {cfg.hidden_dim}")
    print(f"  {'batch_size':<24}: {BATCH}")
    print(f"  {'Phase 1 stop':<24}: {stop_reason}")
    print(f"  {'Phase 1 best_val':<24}: {best_val_ph1:.4f}")
    print(f"  {'Phase 2 tiempo':<24}: {t_ph2:.0f}s ({t_ph2/60:.1f} min)")
    print(f"  {'Tiempo total':<24}: {_fmt_eta(t_total)} ({t_total/60:.1f} min)")
    print()
    print(f"  Word F1 por motor:")
    mean_f1 = sum(f1_per_motor.values()) / len(f1_per_motor)
    for d, f1 in f1_per_motor.items():
        bar = "#" * int(f1 * 20)
        print(f"    {d:<10}: {f1:.3f}  [{bar:<20}]")
    print(f"    {'MEDIA':<10}: {mean_f1:.3f}")
    print(SEP)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
