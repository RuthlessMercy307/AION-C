"""
experiments/train_mose_gpu.py
==============================
Entrenamiento de MoSE con torch-directml (DirectML / AMD / Intel / cualquier
GPU con driver DirectX 12). Fallback automático a CPU si DirectML no está
disponible o falla al inicializar.

Sin AMP ni GradScaler — DirectML no garantiza soporte completo de autocast.

Fases:
  Phase 1 — 500 steps mezclados (5 dominios)
  Phase 2 — 500 steps solo CORA

Diferencias vs train_mose_cpu.py:
  - device  = torch_directml.device()  (DML)  o  cpu  (fallback)
  - batch   = 8  (más VRAM disponible en GPU)
  - node_vecs de precompute se mueven a DML en get_batch
  - ManualGRUCell en CRE (reemplaza nn.GRUCell incompatible con DML)

Uso:
    cd AION-C
    python -m experiments.train_mose_gpu
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random
import time
import warnings
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
# Device — DirectML con fallback a CPU
# ─────────────────────────────────────────────────────────────────────────────

def _init_device() -> Tuple[torch.device, str]:
    """
    Intenta inicializar torch-directml. Si falla por cualquier razón,
    vuelve a CPU automáticamente.

    Returns (device, backend_name).
    """
    try:
        import torch_directml  # noqa: F401
        dml_dev = torch_directml.device()
        # Smoke test: un matmul pequeño para confirmar que la GPU responde
        _t = torch.randn(4, 4).to(dml_dev)
        _ = (_t @ _t).sum().item()
        n_gpus   = torch_directml.device_count()
        gpu_name = torch_directml.device_name(0) if n_gpus > 0 else "unknown"
        return dml_dev, f"DirectML ({gpu_name})"
    except Exception as e:
        warnings.warn(f"torch-directml no disponible ({e}), usando CPU.")
        return torch.device("cpu"), "CPU (fallback)"


DEVICE, BACKEND = _init_device()
USE_DML = str(DEVICE) != "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# DML compatibility patch — GELU → SiLU
# ─────────────────────────────────────────────────────────────────────────────
# DirectML (0.2.5.dev240914) does NOT implement aten::gelu.
# Any module that uses nn.GELU or F.gelu will hang on DML because the GPU
# submits the op but never gets a result (no shader compiled).
#
# Fix: intercept F.gelu at the torch.nn.functional level before any models
# are instantiated. Since nn.GELU.forward calls F.gelu at runtime (not at
# import time), patching the module attribute redirects all usages.
# SiLU (Swish) is functionally equivalent for our purposes.
#
# This patch is applied only when USE_DML=True; CPU/CUDA paths are unaffected.

if USE_DML:
    import torch.nn.functional as _F_dml
    _orig_gelu = _F_dml.gelu

    def _gelu_via_silu(input: torch.Tensor, approximate: str = "none") -> torch.Tensor:
        """GELU replacement for DML: SiLU (x·σ(x)) — same smoothness, no erf/tanh."""
        return _F_dml.silu(input)

    _F_dml.gelu = _gelu_via_silu
    # Also patch torch.nn.functional directly (same object, but defensive)
    import torch.nn.functional as _F2
    _F2.gelu = _gelu_via_silu

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

BATCH        = 8 if USE_DML else 4   # GPU tiene más memoria que CPU
N_EXAMPLES   = 64                    # ejemplos pre-generados por dominio
LOG_INTERVAL = 30.0                  # segundos entre logs

PHASE1_STEPS = 500
PHASE2_STEPS = 500

LR_PHASE1 = 1e-4
LR_MOTOR  = 1e-4
LR_SHARED = 2e-5

torch.manual_seed(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizador char-level
# ─────────────────────────────────────────────────────────────────────────────

PAD, BOS, EOS = 0, 1, 2


def encode(text: str, max_len: int) -> List[int]:
    ids = [BOS] + [(ord(c) % (cfg.vocab_size - 3)) + 3 for c in text] + [EOS]
    return ids[:max_len]


def decode(ids: List[int]) -> str:
    out = []
    for i in ids:
        if i == EOS:
            break
        if i > 2:
            out.append(chr((i - 3) % (cfg.vocab_size - 3)))
    return "".join(out)


def make_padded_batch(seqs: List[List[int]]) -> torch.Tensor:
    maxl   = max(len(s) for s in seqs)
    padded = [s + [PAD] * (maxl - len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long, device=DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Generadores de datos por dominio
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
    """
    Pre-genera, pre-tokeniza y pre-computa grafos del crystallizer para
    todos los ejemplos de un dominio.

    Estrategia DML:
      - Precompute corre en CPU (pipeline temporalmente en CPU) para evitar
        overhead de copias CPU<->DML durante la inicialización.
      - node_vecs se almacenan en CPU y se mueven a DEVICE en get_batch,
        justo antes del forward pass. Esto minimiza la memoria DML ocupada
        por tensores estáticos y permite que el backward fluya correctamente.
    """

    def __init__(self, domain: str, n: int) -> None:
        gen  = GENS[domain]
        hint = MOTOR_HINTS[domain]
        ids_list: List[List[int]] = []
        for _ in range(n):
            lvl  = random.choice([1, 2])
            ex   = gen.generate(level=lvl)
            text = ex.problem_text + " " + ex.answer
            ids_list.append(encode(text, cfg.dec_max_seq_len))

        self.domain    = domain
        self.hint      = hint
        self._ids      = ids_list
        self.n         = len(ids_list)

        self._graphs:      Optional[List] = None
        self._node_vecs:   Optional[List[torch.Tensor]] = None   # siempre CPU
        self._node_counts: Optional[List[int]] = None

    def precompute_graphs(self, pipeline: MoSEPipeline, batch_size: int = 8) -> None:
        """
        Ejecuta crystallizer UNA VEZ sobre todos los ejemplos en DEVICE.

        El pipeline NO se mueve entre dispositivos. Esto es crítico para DML:
        mover el pipeline CPU→DML resetea el cache de shaders compilados y
        causa cuelgues por recompilación al primer training step.
        Al correr en DEVICE, el precompute sirve como warmup natural de DML.
        """
        motor  = pipeline.motors[self.domain]
        graphs: List = []
        nvecs:  List[torch.Tensor] = []
        ncounts: List[int] = []

        pipeline.eval()
        with torch.no_grad():
            for start in range(0, self.n, batch_size):
                chunk   = self._ids[start : start + batch_size]
                maxl    = max(len(s) for s in chunk)
                padded  = [s + [PAD] * (maxl - len(s)) for s in chunk]
                ids_t   = torch.tensor(padded, dtype=torch.long, device=DEVICE)
                concepts = pipeline.encoder(ids_t)
                cryst    = motor.build_graph(concepts)
                for b in range(len(chunk)):
                    n = cryst.node_counts[b]
                    graphs.append(cryst.graphs[b])
                    nvecs.append(cryst.node_vectors[b, :n].cpu())   # guardar en CPU
                    ncounts.append(n)

        self._graphs      = graphs
        self._node_vecs   = nvecs   # CPU — se mueven a DEVICE en get_batch
        self._node_counts = ncounts
        pipeline.train()

    def get_batch(self, bs: int) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Devuelve (token_ids_en_DEVICE, precomp_dict | None).
        node_vecs se mueven de CPU a DEVICE aquí — justo a tiempo para el forward.
        """
        indices   = random.choices(range(self.n), k=bs)
        token_ids = make_padded_batch([self._ids[i] for i in indices])  # en DEVICE

        if self._graphs is None:
            return token_ids, None

        precomp = {
            "motor":     self.domain,
            "graphs":    [self._graphs[i]                    for i in indices],
            "node_vecs": [self._node_vecs[i].to(DEVICE)      for i in indices],  # CPU->DEVICE
            "ncounts":   [self._node_counts[i]               for i in indices],
        }
        return token_ids, precomp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de progreso
# ─────────────────────────────────────────────────────────────────────────────

def _pbar(done: int, total: int, w: int = 24) -> str:
    frac   = done / max(total, 1)
    filled = int(frac * w)
    bar    = "#" * filled + "." * (w - filled)
    return f"[{bar}] {frac*100:>5.1f}%"


def _fmt_eta(secs: float) -> str:
    secs = int(max(0, secs))
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# batched_pipeline_forward — sin autocast, sin GradScaler
# ─────────────────────────────────────────────────────────────────────────────

_batcher = PyGStyleBatcher()


def batched_pipeline_forward(
    pipe:       MoSEPipeline,
    token_ids:  torch.Tensor,
    query_text: Optional[str] = None,
    precomp:    Optional[dict] = None,
) -> MoSEOutput:
    """
    Forward del MoSEPipeline con CRE batching estilo PyG.
    Idéntico al de train_mose_cpu.py; funciona en DML porque todos los ops
    (matmul, scatter_add_, index_select, LayerNorm, Linear, ManualGRUCell)
    están soportados por DirectML.
    """
    B      = token_ids.shape[0]
    D      = pipe.config.hidden_dim
    K      = pipe.config.motor_max_nodes
    device = token_ids.device
    dtype  = pipe.encoder.token_embedding.weight.dtype

    # 1. Encoder
    concepts = pipe.encoder(token_ids)

    # 2. Orchestrator
    orch_out = pipe.orchestrator(concepts, query_text)

    # 3. Crystallizer (fallback no_grad si no hay precomp)
    motor_cryst: dict = {}
    for act in orch_out.activations:
        if precomp is not None and act.motor_name == precomp["motor"]:
            continue
        with torch.no_grad():
            motor_cryst[act.motor_name] = pipe.motors[act.motor_name].build_graph(concepts)

    # 4. CRE batching
    motor_cre_outs: dict = {}

    for act in orch_out.activations:
        motor        = pipe.motors[act.motor_name]
        graphs_b:    List = []
        node_feats_b: List[torch.Tensor] = []
        valid_b:     List[int] = []

        if precomp is not None and act.motor_name == precomp["motor"]:
            pg = precomp["graphs"]
            pv = precomp["node_vecs"]   # ya en DEVICE (movidos en get_batch)
            pn = precomp["ncounts"]
            for b in range(B):
                if pn[b] > 0:
                    graphs_b.append(pg[b])
                    node_feats_b.append(pv[b].detach().requires_grad_(True))
                    valid_b.append(b)
        else:
            cryst_out = motor_cryst[act.motor_name]
            for b in range(B):
                n = cryst_out.node_counts[b]
                if n > 0:
                    graphs_b.append(cryst_out.graphs[b])
                    node_feats_b.append(
                        cryst_out.node_vectors[b, :n].detach().requires_grad_(True))
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

    # 5. graph_repr [B, K, D]
    all_graph_reprs: List[torch.Tensor] = []
    last_unif_out = None

    for b in range(B):
        motor_reprs: List[torch.Tensor] = []
        for act in orch_out.activations:
            motor   = pipe.motors[act.motor_name]
            cre_out = motor_cre_outs[act.motor_name][b]
            if cre_out is None:
                motor_reprs.append(torch.zeros(K, D, device=device, dtype=dtype))
            else:
                motor_reprs.append(motor.get_graph_repr(cre_out, k_nodes=K))
        last_unif_out = pipe.unifier(motor_reprs)
        all_graph_reprs.append(last_unif_out.unified)

    graph_repr = torch.stack(all_graph_reprs, dim=0)

    # 6. Decoder
    dec_out = pipe.decoder(token_ids, graph_repr, concepts)

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
# Evaluación greedy
# ─────────────────────────────────────────────────────────────────────────────

def word_f1(pred: str, ref: str) -> float:
    p, r = set(pred.lower().split()), set(ref.lower().split())
    if not p or not r:
        return 0.0
    tp = len(p & r)
    pr = tp / len(p)
    rc = tp / len(r)
    return round(2 * pr * rc / (pr + rc), 3) if (pr + rc) else 0.0


@torch.no_grad()
def greedy_decode(pipeline: MoSEPipeline, prompt: str, hint: str,
                  max_new: int = 48) -> str:
    pipeline.eval()
    ids = encode(prompt, cfg.dec_max_seq_len - max_new)
    cur = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    for _ in range(max_new):
        if cur.shape[1] >= cfg.dec_max_seq_len:
            break
        out = pipeline(cur, query_text=hint or None)
        nxt = out.logits[0, -1].argmax().item()
        if nxt == EOS:
            break
        cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    pipeline.train()
    return decode(cur[0, len(ids):].tolist())


def eval_cora(pipeline: MoSEPipeline, n: int = 3) -> float:
    gen = GENS["cora"]
    f1s = []
    print("  Eval [cora]:")
    for i in range(n):
        ex = gen.generate(level=random.choice([1, 2]))
        q  = ex.problem_text
        a  = ex.answer
        p  = greedy_decode(pipeline, q, "", max_new=len(a) + 8)
        f1 = word_f1(p, a)
        f1s.append(f1)
        q_short = q[:56] + ("..." if len(q) > 56 else "")
        p_short = p[:56] + ("..." if len(p) > 56 else "")
        print(f"    [{i+1}] Q: {q_short}")
        print(f"         A: {a[:56]}")
        print(f"         P: {p_short}  F1={f1:.2f}")
    mean = sum(f1s) / len(f1s) if f1s else 0.0
    print(f"         -> mean Word F1 = {mean:.3f}")
    return mean


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global cfg

    sys.stdout.reconfigure(line_buffering=True)

    SEP = "-" * 68
    print(SEP)
    print(f"  AION-C / MoSE -- Entrenamiento GPU  [{BACKEND}]")
    print(SEP)

    global cfg
    cfg = MoSEConfig.tiny()
    print(f"Config: hidden_dim={cfg.hidden_dim}, vocab_size={cfg.vocab_size}, "
          f"max_seq_len={cfg.dec_max_seq_len}, K={cfg.motor_max_nodes}")

    # Pipeline creado directamente en DEVICE y nunca movido — crítico para DML
    # (mover entre CPU↔DML resetea el cache de shaders compilados).
    pipeline = MoSEPipeline(cfg).to(DEVICE)

    # Gradient checkpointing desactivado en DML: torch.utils.checkpoint llama
    # torch.cpu.amp.autocast internamente (use_reentrant=False), lo que puede
    # colgar con tensores en DirectML. En CPU se activa normalmente.
    if not USE_DML:
        pipeline.encoder.enable_gradient_checkpointing()
        pipeline.decoder.enable_gradient_checkpointing()
        ckpt_status = "encoder OK, decoder OK"
    else:
        ckpt_status = "desactivado (DML incompatible con torch.cpu.amp.autocast)"

    bd = pipeline.parameter_breakdown()
    print(f"Parametros: {bd['total_unique']:,} unicos "
          f"(enc={bd['encoder']:,} / dec={bd['decoder']:,} / "
          f"motores={sum(v for k,v in bd.items() if k.startswith('motor_')):,})")
    print(f"Gradient checkpointing: {ckpt_status}")
    print(f"batch_size={BATCH} | device={DEVICE} | backend={BACKEND}")

    # ── Pre-generar y pre-computar datasets ──────────────────────────────────
    print()
    print(f"Pre-generando {N_EXAMPLES} ejemplos x {len(MOTOR_NAMES)} dominios...",
          end=" ", flush=True)
    t0 = time.perf_counter()
    DATASETS: Dict[str, PrecomputedDataset] = {
        d: PrecomputedDataset(d, N_EXAMPLES) for d in MOTOR_NAMES
    }
    print(f"OK ({time.perf_counter()-t0:.1f}s)")

    # precompute_graphs mueve el pipeline a CPU, corre en CPU, lo devuelve a DEVICE
    print(f"Pre-computando grafos ({N_EXAMPLES} ej x {len(MOTOR_NAMES)} dominios)...",
          end=" ", flush=True)
    t0 = time.perf_counter()
    for d in MOTOR_NAMES:
        DATASETS[d].precompute_graphs(pipeline, batch_size=8)
    print(f"OK ({time.perf_counter()-t0:.1f}s) -- crystallizer eliminado del loop",
          flush=True)

    # ── DML Warmup: pre-compilar shaders para todos los componentes ─────────
    # DirectML compila shaders JIT en la primera llamada a cada operación.
    # Sin warmup, el primer training step incluye la compilación (~30s/op nueva).
    # Corremos un forward completo en eval+no_grad para triggear la compilación
    # antes del loop de entrenamiento, haciendo los primeros steps más predecibles.
    if USE_DML:
        print(f"DML warmup (compilar shaders)...", end=" ", flush=True)
        t0_warm = time.perf_counter()
        pipeline.eval()
        _warm_ids, _warm_precomp = DATASETS["cora"].get_batch(min(BATCH, 4))
        with torch.no_grad():
            _warm_out = batched_pipeline_forward(pipeline, _warm_ids, None, _warm_precomp)
            _ = _warm_out.logits[0, 0, 0].item()  # force DML sync
        pipeline.train()
        print(f"OK ({time.perf_counter()-t0_warm:.1f}s)", flush=True)

    # ── Phase 1: Shared Pretraining ──────────────────────────────────────────
    print()
    print(SEP)
    print(f"  Phase 1 -- Shared Pretraining  [{PHASE1_STEPS} steps | 5 dominios]")
    print(SEP)

    opt1   = torch.optim.AdamW(pipeline.parameters(), lr=LR_PHASE1, weight_decay=0.01)
    sched1 = torch.optim.lr_scheduler.OneCycleLR(
        opt1, LR_PHASE1, total_steps=PHASE1_STEPS, pct_start=0.1, anneal_strategy="cos"
    )

    losses_ph1: List[float] = []
    pipeline.train()
    t_ph1_start = time.perf_counter()
    last_log    = time.perf_counter()

    for step in range(1, PHASE1_STEPS + 1):
        domain = random.choice(MOTOR_NAMES)
        ids, precomp = DATASETS[domain].get_batch(BATCH)
        hint = DATASETS[domain].hint

        out  = batched_pipeline_forward(pipeline, ids, hint or None, precomp=precomp)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, cfg.vocab_size),
            ids[:, 1:].reshape(-1),
            ignore_index=PAD,
        )

        loss.backward()
        grad_norm1 = nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
        # DML: force backward to complete before next forward.
        # clip_grad_norm_ returns a DML scalar tensor; float() reads it to CPU,
        # which blocks until ALL pending DML ops (including backward) finish.
        if USE_DML:
            _ = float(grad_norm1)
        opt1.step()
        opt1.zero_grad(set_to_none=True)
        sched1.step()
        losses_ph1.append(loss.item())

        now = time.perf_counter()
        if now - last_log >= LOG_INTERVAL:
            last_log = now
            avg  = sum(losses_ph1[-50:]) / min(50, len(losses_ph1))
            lr   = sched1.get_last_lr()[0]
            t_el = now - t_ph1_start
            thr  = step * BATCH / t_el
            eta  = (PHASE1_STEPS - step) * (t_el / step)
            print(f"  {_pbar(step, PHASE1_STEPS)} step {step:>4}/{PHASE1_STEPS}  "
                  f"loss={avg:.4f}  lr={lr:.2e}  {thr:.1f} samp/s  ETA {_fmt_eta(eta)}",
                  flush=True)

    t_ph1  = time.perf_counter() - t_ph1_start
    l0_ph1 = sum(losses_ph1[:10]) / 10
    l1_ph1 = sum(losses_ph1[-10:]) / 10
    thr_ph1 = PHASE1_STEPS * BATCH / t_ph1
    print(f"\nPhase 1 done | loss {l0_ph1:.4f} -> {l1_ph1:.4f} | "
          f"{t_ph1:.0f}s ({t_ph1/60:.1f} min) | {thr_ph1:.1f} samp/s", flush=True)

    # ── Phase 2: CORA Fine-Tuning ────────────────────────────────────────────
    print()
    print(SEP)
    print(f"  Phase 2 -- CORA Fine-Tuning  [{PHASE2_STEPS} steps | solo CORA]")
    print(SEP)

    motor_params  = list(pipeline.motors["cora"].parameters())
    shared_params = (list(pipeline.encoder.parameters()) +
                     list(pipeline.decoder.parameters()) +
                     list(pipeline.orchestrator.parameters()) +
                     list(pipeline.unifier.parameters()))

    opt2 = torch.optim.AdamW(
        [{"params": motor_params,  "lr": LR_MOTOR},
         {"params": shared_params, "lr": LR_SHARED}],
        weight_decay=0.01,
    )

    warmup = min(50, PHASE2_STEPS // 10)
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, PHASE2_STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda)

    orig_conf = pipeline.orchestrator.config.min_confidence_to_activate
    pipeline.orchestrator.config.min_confidence_to_activate = 1.0

    losses_ph2: List[float] = []
    ds_cora = DATASETS["cora"]
    pipeline.train()
    t_ph2_start = time.perf_counter()
    last_log    = time.perf_counter()

    for step in range(1, PHASE2_STEPS + 1):
        ids, precomp = ds_cora.get_batch(BATCH)

        out  = batched_pipeline_forward(pipeline, ids, None, precomp=precomp)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, cfg.vocab_size),
            ids[:, 1:].reshape(-1),
            ignore_index=PAD,
        )

        loss.backward()
        grad_norm2 = nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
        if USE_DML:
            _ = float(grad_norm2)   # force DML backward sync (see Phase 1 comment)
        opt2.step()
        opt2.zero_grad(set_to_none=True)
        sched2.step()
        losses_ph2.append(loss.item())

        now = time.perf_counter()
        if now - last_log >= LOG_INTERVAL:
            last_log = now
            avg  = sum(losses_ph2[-50:]) / min(50, len(losses_ph2))
            t_el = now - t_ph2_start
            thr  = step * BATCH / t_el
            eta  = (PHASE2_STEPS - step) * (t_el / step)
            print(f"  {_pbar(step, PHASE2_STEPS)} step {step:>4}/{PHASE2_STEPS}  "
                  f"loss={avg:.4f}  {thr:.1f} samp/s  ETA {_fmt_eta(eta)}",
                  flush=True)

    pipeline.orchestrator.config.min_confidence_to_activate = orig_conf

    t_ph2  = time.perf_counter() - t_ph2_start
    l0_ph2 = sum(losses_ph2[:10]) / 10
    l1_ph2 = sum(losses_ph2[-10:]) / 10
    thr_ph2 = PHASE2_STEPS * BATCH / t_ph2
    print(f"\nPhase 2 done | loss {l0_ph2:.4f} -> {l1_ph2:.4f} | "
          f"{t_ph2:.0f}s ({t_ph2/60:.1f} min) | {thr_ph2:.1f} samp/s", flush=True)

    # ── Eval final ───────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  Eval final -- 3 ejemplos CORA (greedy decode)")
    print(SEP)
    mean_f1 = eval_cora(pipeline, n=3)

    # ── Resumen ──────────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_ph1_start
    print()
    print(SEP)
    print("  Resumen")
    print(SEP)
    print(f"  {'Backend':<22}: {BACKEND}")
    print(f"  {'Parametros':<22}: {bd['total_unique']:,}")
    print(f"  {'batch_size':<22}: {BATCH}")
    print(f"  {'Phase 1 (mixed)':<22}: {l0_ph1:.4f} -> {l1_ph1:.4f}  "
          f"({t_ph1:.0f}s, {thr_ph1:.1f} samp/s)")
    print(f"  {'Phase 2 (CORA)':<22}: {l0_ph2:.4f} -> {l1_ph2:.4f}  "
          f"({t_ph2:.0f}s, {thr_ph2:.1f} samp/s)")
    print(f"  {'Word F1 (CORA)':<22}: {mean_f1:.3f}")
    print(f"  {'Tiempo total':<22}: {_fmt_eta(t_total)} ({t_total/60:.1f} min)")
    print(SEP)


if __name__ == "__main__":
    main()
