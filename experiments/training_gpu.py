"""
experiments/training_gpu.py — Training Loop GPU de CORA
=========================================================

Versión GPU de first_training.py con:
  - hidden_dim=128 (3.24M params sin decoder)
  - 2000 steps, 1000 ejemplos, niveles 1-3
  - Detección automática ROCm (RX 6600) / CUDA / CPU fallback
  - Cosine LR decay: 1e-3 → 1e-5
  - Evaluación cada 200 steps con 3 ejemplos
  - Accuracy de nodos y relaciones por step
  - Reporte final en experiments/results/

Ejecutar:
    python -m experiments.training_gpu

Para forzar ROCm en RX 6600 desde terminal:
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python -m experiments.training_gpu
"""

from __future__ import annotations

# ── ROCm: debe estar ANTES de que torch inicialice CUDA ──────────────────────
# RX 6600 es gfx1032; ROCm no la soporta oficialmente → override a 10.3.0
# El script lo pone si no está ya en el entorno (puede sobreescribirse en shell)
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import io
import json
import math
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── UTF-8 stdout para Windows ────────────────────────────────────────────────
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.graph import CausalGraph
from synth.causal_graph_gen import CausalExample, CausalGraphGenerator
from encoder import StreamEncoder
from crystallizer import GraphCrystallizer
from cre import CausalReasoningEngine, DifferentiableScratchPad
from router.pipeline import CORAConfig


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN_DIM   = 128
VOCAB_SIZE   = 512
N_EXAMPLES   = 1000
N_STEPS      = 2000
LR_INIT      = 1e-3
LR_MIN       = 1e-5
WEIGHT_DECAY = 1e-4
PRINT_EVERY  = 50
EVAL_EVERY   = 200
N_EVAL_EX    = 3
CRE_ITERS    = 5
LAMBDA_CRE   = 0.1
GRAD_CLIP    = 1.0


def make_config() -> CORAConfig:
    """Config de entrenamiento GPU: hidden_dim=128, niveles 1-3."""
    return CORAConfig(
        hidden_dim   = HIDDEN_DIM,
        vocab_size   = VOCAB_SIZE,
        # Encoder — 3 capas, estado SSM=8
        enc_n_layers  = 3,
        enc_state_dim = 8,
        enc_expand    = 2,
        enc_d_conv    = 4,
        enc_ffn_mult  = 2,
        # Crystallizer — 16 slots, threshold bajo para que siempre haya nodos
        cryst_max_nodes      = 16,
        cryst_n_heads        = 4,
        cryst_node_threshold = 0.01,
        cryst_edge_threshold = 0.01,
        # CRE — 2 capas de message passing, 5 iteraciones
        cre_edge_dim         = 64,
        cre_message_dim      = 128,
        cre_n_message_layers = 2,
        cre_max_iterations   = CRE_ITERS,
        # ScratchPad
        pad_n_slots  = 16,
        pad_slot_dim = 64,
        # Decoder (no entrenado — solo para que CORAConfig sea válido)
        dec_n_layers    = 3,
        dec_n_heads     = 4,
        dec_max_seq_len = 128,
        dec_state_dim   = 8,
        dec_expand      = 2,
        dec_d_conv      = 4,
        dec_ffn_mult    = 2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN DE DISPOSITIVO Y VRAM
# ─────────────────────────────────────────────────────────────────────────────

def detect_device() -> Tuple[torch.device, str]:
    """
    Detecta CUDA / ROCm / CPU.
    Retorna (device, backend_name).
    HSA_OVERRIDE_GFX_VERSION=10.3.0 ya fue aplicado al inicio del módulo.
    """
    if torch.cuda.is_available():
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        backend = "ROCm/HIP" if is_rocm else "CUDA"
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"{backend} — {name}"
    return torch.device("cpu"), "CPU"


def print_vram_info(device: torch.device, n_params: int) -> None:
    """Imprime VRAM disponible y estimación de uso."""
    model_mb   = n_params * 4 / 1e6          # float32
    optim_mb   = n_params * 4 * 2 / 1e6      # AdamW: m + v por param
    activ_mb   = 80.0                          # buffer conservador para activaciones
    total_est  = model_mb + optim_mb + activ_mb

    print(f"\n[vram] Estimacion de uso:")
    print(f"       Modelo (float32)  : {model_mb:6.1f} MB")
    print(f"       Optimizer (AdamW) : {optim_mb:6.1f} MB")
    print(f"       Activaciones (est): {activ_mb:6.1f} MB")
    print(f"       TOTAL estimado    : {total_est:6.1f} MB")

    if device.type == "cuda":
        try:
            free_b, total_b = torch.cuda.mem_get_info(device)
            free_mb  = free_b  / 1e6
            total_mb = total_b / 1e6
            margin   = free_mb - total_est
            status   = "OK" if margin > 500 else ("AJUSTADO" if margin > 0 else "RIESGO OOM")
            print(f"       GPU disponible    : {free_mb:6.0f} MB / {total_mb:6.0f} MB total")
            print(f"       Margen libre      : {margin:6.0f} MB  [{status}]")
        except Exception:
            print(f"       (no se pudo consultar VRAM disponible)")
    else:
        ram_est = total_est
        print(f"       [CPU] RAM estimada : {ram_est:.0f} MB (sin GPU)")


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZACIÓN DETERMINISTA
# ─────────────────────────────────────────────────────────────────────────────

def _word_hash(word: str, vocab_size: int) -> int:
    h = 0
    for ch in word:
        h = (h * 31 + ord(ch)) % vocab_size
    return max(h, 1)


def simple_tokenize(
    text: str,
    vocab_size: int,
    max_len: int = 64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    words = text.lower().split()[:max_len]
    ids = [_word_hash(w, vocab_size) for w in words] or [1]
    return torch.tensor([ids], dtype=torch.long, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE MÓDULOS
# ─────────────────────────────────────────────────────────────────────────────

def build_modules(cfg: CORAConfig, device: torch.device):
    encoder     = StreamEncoder(cfg.encoder_config()).to(device)
    crystallizer = GraphCrystallizer(cfg.crystallizer_config()).to(device)
    cre         = CausalReasoningEngine(cfg.cre_config()).to(device)
    scratch_pad = DifferentiableScratchPad(cfg.scratch_pad_config()).to(device)
    return encoder, crystallizer, cre, scratch_pad


def count_unique_params(*modules) -> int:
    seen, total = set(), 0
    for mod in modules:
        for p in mod.parameters():
            if id(p) not in seen:
                seen.add(id(p)); total += p.numel()
    return total


# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH ADJACENCY
# ─────────────────────────────────────────────────────────────────────────────

def build_gt_adjacency(graph: CausalGraph, n: int) -> torch.Tensor:
    nodes    = graph.nodes[:n]
    node_idx = {nd.node_id: i for i, nd in enumerate(nodes)}
    adj      = torch.zeros(n, n)
    for edge in graph.edges:
        if edge.source_id in node_idx and edge.target_id in node_idx:
            adj[node_idx[edge.source_id], node_idx[edge.target_id]] = 1.0
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD PASS (sin decoder)
# ─────────────────────────────────────────────────────────────────────────────

def forward_enc_cryst_cre(
    encoder, crystallizer, cre, scratch_pad,
    token_ids: torch.Tensor,
    n_cre_iters: int,
):
    """encoder → crystallizer → CRE. Sin decoder."""
    concepts    = encoder(token_ids)                  # [B, L, D]
    crystal_out = crystallizer(concepts)              # CrystallizerOutput

    n_nodes = crystal_out.node_counts[0]
    if n_nodes == 0:
        D     = concepts.shape[-1]
        dummy = torch.zeros(1, D, device=concepts.device, dtype=concepts.dtype)
        return crystal_out, dummy, 0

    node_feats = crystal_out.node_vectors[0, :n_nodes, :]   # [n, D] — diferenciable
    graph      = crystal_out.graphs[0]
    cre_out    = cre(graph, node_feats, scratch_pad=scratch_pad, n_iterations=n_cre_iters)
    return crystal_out, cre_out.node_features, n_nodes


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE LOSS
# ─────────────────────────────────────────────────────────────────────────────

def node_count_loss(crystal_out, gt_n: int, max_nodes: int = 16) -> torch.Tensor:
    """
    MSE sobre el soft count de los top-K node scores.

    Usa topk().values (diferenciable) en lugar de sum sobre todas las L
    posiciones para evitar colapso: el gradiente fluye solo a los K scores
    más altos, no empuja todos hacia -inf simultáneamente.
    """
    node_scores = crystal_out.node_scores                              # [B, L]
    L = node_scores.shape[1]
    K = min(L, max_nodes)
    topk_logits = node_scores.topk(K, dim=1).values                   # [B, K]
    soft_count  = torch.sigmoid(topk_logits).sum(dim=1).clamp(min=0.5)  # [B]
    return F.mse_loss(soft_count, torch.full_like(soft_count, float(gt_n)))


def relation_loss(crystal_out, gt_adj: torch.Tensor, n: int) -> Optional[torch.Tensor]:
    K       = crystal_out.relation_logits.shape[1]
    n_align = min(n, K)
    if n_align < 2:
        return None
    max_logits = crystal_out.relation_logits[0].max(dim=-1).values  # [K, K]
    sub        = max_logits[:n_align, :n_align]
    gt_sub     = gt_adj[:n_align, :n_align].to(sub.device)
    mask       = ~torch.eye(n_align, dtype=torch.bool, device=sub.device)
    flat_l = sub[mask]; flat_t = gt_sub[mask]
    return F.binary_cross_entropy_with_logits(flat_l, flat_t) if flat_l.numel() > 0 else None


def cre_coherence_loss(cre_feats: torch.Tensor, gt_adj: torch.Tensor, n: int) -> Optional[torch.Tensor]:
    n_align = min(n, cre_feats.shape[0])
    if n_align < 2:
        return None
    D     = cre_feats.shape[1]
    feats = cre_feats[:n_align]
    gt    = gt_adj[:n_align, :n_align].to(feats.device)
    sim   = (feats @ feats.T) / math.sqrt(D)
    mask  = ~torch.eye(n_align, dtype=torch.bool, device=feats.device)
    flat_l = sim[mask]; flat_t = gt[mask]
    return F.binary_cross_entropy_with_logits(flat_l, flat_t) if flat_l.numel() > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS DE ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def node_accuracy(pred_n: int, gt_n: int) -> float:
    """1 - error normalizado; rango [0, 1]."""
    return max(0.0, 1.0 - abs(pred_n - gt_n) / max(gt_n, 1))


def relation_accuracy(crystal_out, gt_graph: CausalGraph, n_align: int) -> float:
    """
    Recall de edges GT: qué fracción de las relaciones ground truth el
    crystallizer detecta (max relation logit > 0 en la posición alineada).

    Retorna 1.0 si no hay GT edges (nada que detectar).
    """
    if n_align < 1:
        return 0.0

    gt_nodes = gt_graph.nodes[:n_align]
    node_idx = {nd.node_id: i for i, nd in enumerate(gt_nodes)}

    gt_pairs = set()
    for edge in gt_graph.edges:
        if edge.source_id in node_idx and edge.target_id in node_idx:
            gt_pairs.add((node_idx[edge.source_id], node_idx[edge.target_id]))

    if not gt_pairs:
        return 1.0

    rel_logits = crystal_out.relation_logits[0]   # [K, K, 16]
    detected   = sum(
        1 for (i, j) in gt_pairs
        if i < rel_logits.shape[0] and j < rel_logits.shape[1]
        and rel_logits[i, j].max().item() > 0.0
    )
    return detected / len(gt_pairs)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN PERIÓDICA
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    encoder, crystallizer, cre, scratch_pad,
    examples: List[CausalExample],
    cfg: CORAConfig,
    device: torch.device,
    step: int,
) -> List[Dict]:
    """
    Evalúa N_EVAL_EX ejemplos distribuidos. Imprime tabla con 3 ejemplos.
    Retorna lista de dicts con métricas por ejemplo.
    """
    encoder.eval(); crystallizer.eval(); cre.eval(); scratch_pad.eval()

    n = len(examples)
    indices = [n // (N_EVAL_EX + 1) * (k + 1) for k in range(N_EVAL_EX)]

    results = []
    with torch.no_grad():
        for idx in indices:
            ex       = examples[idx]
            tok      = simple_tokenize(ex.problem_text, cfg.vocab_size, device=device)
            crys_out, cre_feats, n_valid = forward_enc_cryst_cre(
                encoder, crystallizer, cre, scratch_pad, tok, CRE_ITERS
            )
            gt_n     = len(ex.graph.nodes)
            gt_e     = len(ex.graph.edges)
            pred_e   = len(crys_out.graphs[0].edges)
            n_align  = min(n_valid, gt_n, crys_out.relation_logits.shape[1])
            n_acc    = node_accuracy(n_valid, gt_n)
            r_acc    = relation_accuracy(crys_out, ex.graph, n_align)
            results.append({
                "idx": idx,
                "level": ex.complexity_level,
                "text": ex.problem_text[:70],
                "gt_nodes": gt_n, "pred_nodes": n_valid,
                "gt_edges": gt_e, "pred_edges": pred_e,
                "node_acc": round(n_acc, 3),
                "rel_acc":  round(r_acc, 3),
            })

    # ── Tabla de evaluación ──────────────────────────────────────────────────
    bar = "-" * 72
    print(f"\n  {bar}")
    print(f"  EVAL @ step {step:>4}   ({N_EVAL_EX} ejemplos)")
    print(f"  {bar}")
    print(f"  {'Lv':>2}  {'Nodos GT':>8}  {'Pred':>5}  {'NodeAcc':>7}  {'Edges GT':>8}  {'Pred':>5}  {'RelAcc':>6}")
    print(f"  {bar}")
    for r in results:
        print(
            f"  {r['level']:>2}  {r['gt_nodes']:>8}  {r['pred_nodes']:>5}  "
            f"{r['node_acc']:>7.1%}  {r['gt_edges']:>8}  {r['pred_edges']:>5}  "
            f"{r['rel_acc']:>6.1%}"
        )
        print(f"       \"{r['text']}\"")
    avg_nacc = sum(r["node_acc"] for r in results) / len(results)
    avg_racc = sum(r["rel_acc"]  for r in results) / len(results)
    print(f"  {bar}")
    print(f"  Promedio: node_acc={avg_nacc:.1%}  rel_acc={avg_racc:.1%}")

    encoder.train(); crystallizer.train(); cre.train(); scratch_pad.train()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ESTIMACIÓN DE VELOCIDAD
# ─────────────────────────────────────────────────────────────────────────────

def estimate_timing(
    encoder, crystallizer, cre, scratch_pad,
    example: CausalExample, cfg: CORAConfig,
    device: torch.device, all_params: list,
) -> float:
    """
    Corre un step de calentamiento y retorna ms/step estimados.
    No aplica el step al optimizer (descarta gradientes).
    """
    tok = simple_tokenize(example.problem_text, cfg.vocab_size, device=device)
    for mod in (encoder, crystallizer, cre, scratch_pad):
        mod.train()

    # Calentamiento GPU
    if device.type == "cuda":
        for _ in range(3):
            crys, feats, nv = forward_enc_cryst_cre(
                encoder, crystallizer, cre, scratch_pad, tok, CRE_ITERS
            )
            loss = node_count_loss(crys, len(example.graph.nodes), cfg.cryst_max_nodes)
            loss.backward()
            for p in all_params:
                if p.grad is not None:
                    p.grad.zero_()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    crys, feats, nv = forward_enc_cryst_cre(
        encoder, crystallizer, cre, scratch_pad, tok, CRE_ITERS
    )
    gt_n    = len(example.graph.nodes)
    n_align = min(nv, gt_n, cfg.cryst_max_nodes)
    gt_adj  = build_gt_adjacency(example.graph, n_align).to(device)

    l_nc = node_count_loss(crys, gt_n, cfg.cryst_max_nodes)
    l_rel = relation_loss(crys, gt_adj, n_align)
    l_coh = cre_coherence_loss(feats, gt_adj, n_align)
    total = l_nc
    if l_rel is not None: total = total + l_rel
    if l_coh is not None: total = total + LAMBDA_CRE * l_coh
    total.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000   # ms

    for p in all_params:
        if p.grad is not None:
            p.grad.zero_()
    return dt


# ─────────────────────────────────────────────────────────────────────────────
# REPORTE FINAL
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report(
    loss_history: List[Dict],
    node_acc_history: List[float],
    rel_acc_history: List[float],
    total_time: float,
    avg_ms: float,
    backend: str,
) -> Dict:
    print("\n" + "=" * 72)
    print("  REPORTE FINAL")
    print("=" * 72)

    mid = N_STEPS // 2
    l_first  = [r["total"] for r in loss_history[:mid]]
    l_second = [r["total"] for r in loss_history[mid:]]
    avg_l1   = sum(l_first)  / len(l_first)  if l_first  else float("nan")
    avg_l2   = sum(l_second) / len(l_second) if l_second else float("nan")
    l_improv = avg_l1 - avg_l2
    l_pct    = (l_improv / avg_l1 * 100) if avg_l1 > 0 else 0.0

    na_first  = node_acc_history[:mid]
    na_second = node_acc_history[mid:]
    avg_na1   = sum(na_first)  / len(na_first)  if na_first  else 0.0
    avg_na2   = sum(na_second) / len(na_second) if na_second else 0.0

    ra_first  = rel_acc_history[:mid]
    ra_second = rel_acc_history[mid:]
    avg_ra1   = sum(ra_first)  / len(ra_first)  if ra_first  else 0.0
    avg_ra2   = sum(ra_second) / len(ra_second) if ra_second else 0.0

    significant = l_pct > 5.0

    print(f"\n  Dispositivo : {backend}")
    print(f"  Duracion    : {total_time:.1f}s  ({avg_ms:.0f} ms/step)")
    print(f"\n  Loss:")
    print(f"    1a mitad (steps  1-{mid:>4})  : {avg_l1:.4f}")
    print(f"    2a mitad (steps {mid+1:>4}-{N_STEPS:>4}) : {avg_l2:.4f}")
    print(f"    Mejora   : {l_improv:+.4f}  ({l_pct:+.1f}%)")
    print(f"    Mejora significativa (>5%): {'SI' if significant else 'NO'}")

    print(f"\n  Node Accuracy (promedio movil 50 steps):")
    print(f"    1a mitad : {avg_na1:.1%}")
    print(f"    2a mitad : {avg_na2:.1%}")
    na_dir = "mejora" if avg_na2 > avg_na1 else "estable/baja"
    print(f"    Tendencia: {na_dir}")

    print(f"\n  Relation Accuracy (recall GT edges):")
    print(f"    1a mitad : {avg_ra1:.1%}")
    print(f"    2a mitad : {avg_ra2:.1%}")
    ra_dir = "mejora" if avg_ra2 > avg_ra1 else "estable/baja"
    print(f"    Tendencia: {ra_dir}")

    # Mejor y peor step
    best_step  = min(loss_history, key=lambda r: r["total"])
    worst_step = max(loss_history, key=lambda r: r["total"])
    print(f"\n  Mejor  step: {best_step['step']:>4}  loss={best_step['total']:.4f}")
    print(f"  Peor   step: {worst_step['step']:>4}  loss={worst_step['total']:.4f}")

    print("=" * 72)

    return {
        "backend": backend,
        "timing_seconds": round(total_time, 2),
        "avg_ms_per_step": round(avg_ms, 1),
        "loss_first_half": round(avg_l1, 6),
        "loss_second_half": round(avg_l2, 6),
        "loss_improvement": round(l_improv, 6),
        "loss_improvement_pct": round(l_pct, 2),
        "significant_improvement": significant,
        "node_acc_first_half": round(avg_na1, 4),
        "node_acc_second_half": round(avg_na2, 4),
        "rel_acc_first_half": round(avg_ra1, 4),
        "rel_acc_second_half": round(avg_ra2, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:
    # ── Setup ────────────────────────────────────────────────────────────────
    device, backend = detect_device()
    cfg = make_config()

    print("=" * 72)
    print("  CORA — Training GPU  |  hidden_dim=128  |  2000 steps")
    print("=" * 72)
    print(f"\n[device]  {backend}")
    print(f"[rocm]    HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'no definido')}")
    print(f"[config]  hidden_dim={cfg.hidden_dim}  vocab_size={cfg.vocab_size}")
    print(f"[config]  enc_layers={cfg.enc_n_layers}  cryst_max_nodes={cfg.cryst_max_nodes}")
    print(f"[config]  cre_iters={CRE_ITERS}  cre_layers={cfg.cre_n_message_layers}")
    print(f"[sched]   cosine decay  {LR_INIT:.0e} -> {LR_MIN:.0e}  over {N_STEPS} steps")

    # ── Módulos ──────────────────────────────────────────────────────────────
    encoder, crystallizer, cre, scratch_pad = build_modules(cfg, device)
    n_params = count_unique_params(encoder, crystallizer, cre, scratch_pad)
    print(f"[model]   {n_params:,} parametros entrenables (sin decoder)")

    print_vram_info(device, n_params)

    # ── Datos ────────────────────────────────────────────────────────────────
    print(f"\n[data]  Generando {N_EXAMPLES} ejemplos (niveles 1-3)...")
    t0_data = time.perf_counter()
    gen = CausalGraphGenerator(seed=42)
    examples: List[CausalExample] = gen.generate_batch(
        n=N_EXAMPLES,
        level_distribution={1: 0.34, 2: 0.33, 3: 0.33},
    )
    print(f"[data]  {len(examples)} ejemplos en {time.perf_counter()-t0_data:.2f}s  "
          f"(L1={sum(1 for e in examples if e.complexity_level==1)}  "
          f"L2={sum(1 for e in examples if e.complexity_level==2)}  "
          f"L3={sum(1 for e in examples if e.complexity_level==3)})")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    all_params = (
        list(encoder.parameters())
        + list(crystallizer.parameters())
        + list(cre.parameters())
        + list(scratch_pad.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_STEPS, eta_min=LR_MIN
    )

    # ── Estimación de tiempo ─────────────────────────────────────────────────
    print(f"\n[timing] Midiendo step de referencia...")
    ms_warmup = estimate_timing(
        encoder, crystallizer, cre, scratch_pad,
        examples[0], cfg, device, all_params,
    )
    est_total = ms_warmup * N_STEPS / 1000
    print(f"[timing] ~{ms_warmup:.0f} ms/step  →  {N_STEPS} steps ~ {est_total:.0f}s ({est_total/60:.1f} min)")
    if est_total > 300:
        print(f"[timing] ADVERTENCIA: estimado > 5 min.")

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  TRAINING: {N_STEPS} steps  |  {N_EXAMPLES} ejemplos  |  grad_clip={GRAD_CLIP}")
    print(f"{'─'*72}")

    loss_history:     List[Dict]  = []
    node_acc_history: List[float] = []
    rel_acc_history:  List[float] = []
    eval_snapshots:   List[Dict]  = []

    step_times: List[float] = []
    recent_loss = deque(maxlen=50)   # ventana para promedio móvil
    example_idx = 0

    for step in range(N_STEPS):
        ex = examples[example_idx % len(examples)]
        example_idx += 1

        tok    = simple_tokenize(ex.problem_text, cfg.vocab_size, device=device)
        gt_n   = len(ex.graph.nodes)
        gt_e   = len(ex.graph.edges)

        t_s = time.perf_counter()

        # ── Forward ──────────────────────────────────────────────────────────
        optimizer.zero_grad()
        crys_out, cre_feats, n_valid = forward_enc_cryst_cre(
            encoder, crystallizer, cre, scratch_pad, tok, CRE_ITERS
        )

        # ── Losses ───────────────────────────────────────────────────────────
        n_align  = min(n_valid, gt_n, cfg.cryst_max_nodes)
        gt_adj   = build_gt_adjacency(ex.graph, n_align).to(device)

        l_nc  = node_count_loss(crys_out, gt_n, cfg.cryst_max_nodes)
        l_rel = relation_loss(crys_out, gt_adj, n_align)
        l_coh = cre_coherence_loss(cre_feats, gt_adj, n_align)

        total = l_nc
        if l_rel is not None: total = total + l_rel
        if l_coh is not None: total = total + LAMBDA_CRE * l_coh

        # ── Backward + clip + step ────────────────────────────────────────────
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t_s
        step_times.append(dt)

        # ── Métricas por step ─────────────────────────────────────────────────
        n_acc   = node_accuracy(n_valid, gt_n)
        r_align = min(n_valid, gt_n, crys_out.relation_logits.shape[1])
        r_acc   = relation_accuracy(crys_out, ex.graph, r_align)

        node_acc_history.append(n_acc)
        rel_acc_history.append(r_acc)

        loss_val = float(total.item())
        recent_loss.append(loss_val)

        record = {
            "step": step + 1,
            "total":       round(loss_val, 6),
            "node_count":  round(float(l_nc.item()), 6),
            "relation":    round(float(l_rel.item()), 6) if l_rel is not None else None,
            "cre_coh":     round(float(l_coh.item()), 6) if l_coh is not None else None,
            "node_acc":    round(n_acc, 4),
            "rel_acc":     round(r_acc, 4),
            "lr":          round(scheduler.get_last_lr()[0], 8),
            "n_valid":     n_valid,
            "gt_n":        gt_n,
            "ms":          round(dt * 1000, 1),
        }
        loss_history.append(record)

        # ── Print periódico ───────────────────────────────────────────────────
        if (step + 1) % PRINT_EVERY == 0 or step == 0:
            elapsed   = sum(step_times)
            remaining = (N_STEPS - step - 1) * (elapsed / (step + 1))
            avg_l     = sum(recent_loss) / len(recent_loss)
            cur_lr    = scheduler.get_last_lr()[0]
            vram_str  = ""
            if device.type == "cuda":
                try:
                    alloc = torch.cuda.memory_allocated(device) / 1e6
                    vram_str = f"  VRAM={alloc:.0f}MB"
                except Exception:
                    pass
            print(
                f"  step {step+1:>4}/{N_STEPS}"
                f"  loss={avg_l:.4f}"
                f"  node={l_nc.item():.4f}"
                f"  nacc={n_acc:.1%}"
                f"  racc={r_acc:.1%}"
                f"  lr={cur_lr:.1e}"
                f"  ETA={remaining:.0f}s"
                f"{vram_str}"
            )

        # ── Evaluación periódica ──────────────────────────────────────────────
        if (step + 1) % EVAL_EVERY == 0:
            eval_results = evaluate(
                encoder, crystallizer, cre, scratch_pad,
                examples, cfg, device, step + 1,
            )
            eval_snapshots.append({
                "step": step + 1,
                "examples": eval_results,
                "avg_node_acc": round(sum(r["node_acc"] for r in eval_results) / len(eval_results), 4),
                "avg_rel_acc":  round(sum(r["rel_acc"]  for r in eval_results) / len(eval_results), 4),
            })

    # ── Reporte final ─────────────────────────────────────────────────────────
    total_time = sum(step_times)
    avg_ms     = sum(step_times) / len(step_times) * 1000
    summary    = print_final_report(
        loss_history, node_acc_history, rel_acc_history,
        total_time, avg_ms, backend,
    )

    # ── Guardar resultados ────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"training_gpu_{ts}.json")

    output = {
        "meta": {
            "timestamp": ts,
            "script": "training_gpu.py",
            "backend": backend,
            "hsa_override": os.environ.get("HSA_OVERRIDE_GFX_VERSION", ""),
        },
        "config": {
            "hidden_dim":    HIDDEN_DIM,
            "vocab_size":    VOCAB_SIZE,
            "n_examples":    N_EXAMPLES,
            "n_steps":       N_STEPS,
            "lr_init":       LR_INIT,
            "lr_min":        LR_MIN,
            "cre_iters":     CRE_ITERS,
            "lambda_cre":    LAMBDA_CRE,
            "grad_clip":     GRAD_CLIP,
            "n_params":      n_params,
        },
        "summary": summary,
        "eval_snapshots": eval_snapshots,
        "loss_curve":       loss_history,
        "node_acc_curve":   [round(x, 4) for x in node_acc_history],
        "rel_acc_curve":    [round(x, 4) for x in rel_acc_history],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[output] Resultados guardados en:\n         {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    train()
    print("\n[done] training_gpu.py completado.")


if __name__ == "__main__":
    main()
