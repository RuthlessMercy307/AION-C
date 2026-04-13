"""
experiments/first_training.py — Primer Training Loop Real de CORA
==================================================================

Objetivo: demostrar que CORA puede aprender a predecir relaciones causales
simples usando datos del CausalGraphGenerator.

Setup:
  - CORAPipeline con CORAConfig.tiny() (hidden_dim=64)
  - 500 ejemplos del CausalGraphGenerator niveles 1-2
  - Loss: node_count_loss (MSE) + relation_loss (BCE) + cre_coherence_loss
  - Solo encoder + crystallizer + CRE  (sin decoder)
  - AdamW, lr=1e-3, 200 steps, batch_size=1
  - Imprime loss cada 10 steps
  - Al final: 5 ejemplos predicho vs ground truth
  - Guarda curva de loss en JSON

Ejecutar:
    python -m experiments.first_training
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
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
from encoder.mamba_layer import StreamEncoderConfig
from crystallizer import GraphCrystallizer
from crystallizer.config import CrystallizerConfig
from cre import CausalReasoningEngine, DifferentiableScratchPad
from cre.config import CREConfig
from cre.scratch_pad import ScratchPadConfig
from router.pipeline import CORAConfig


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_device() -> torch.device:
    """Auto-detect CUDA / ROCm / CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"[device] CUDA disponible: {name}")
        return dev
    # ROCm expone una API CUDA — se detecta igual con cuda.is_available()
    print("[device] Sin GPU — usando CPU")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZACIÓN DETERMINISTA
# ─────────────────────────────────────────────────────────────────────────────

def _word_hash(word: str, vocab_size: int) -> int:
    """Hash determinista independiente del seed de Python."""
    h = 0
    for ch in word:
        h = (h * 31 + ord(ch)) % vocab_size
    return max(h, 1)  # evitar token 0 (padding)


def simple_tokenize(
    text: str,
    vocab_size: int,
    max_len: int = 64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Tokeniza un texto en token IDs via hash de palabras.
    Retorna [1, L] (batch-first).
    """
    words = text.lower().split()[:max_len]
    ids = [_word_hash(w, vocab_size) for w in words] or [1]
    return torch.tensor([ids], dtype=torch.long, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH ADJACENCY
# ─────────────────────────────────────────────────────────────────────────────

def build_gt_adjacency(graph: CausalGraph, n: int) -> torch.Tensor:
    """
    Matriz binaria [n x n] con las relaciones entre los primeros n nodos GT.

    Asumimos que los primeros n nodos del crystallizer se alinean con los
    primeros n nodos del grafo ground truth (por orden de aparición en el texto).
    """
    nodes = graph.nodes[:n]
    node_idx = {node.node_id: i for i, node in enumerate(nodes)}
    adj = torch.zeros(n, n)
    for edge in graph.edges:
        if edge.source_id in node_idx and edge.target_id in node_idx:
            i = node_idx[edge.source_id]
            j = node_idx[edge.target_id]
            adj[i, j] = 1.0
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_node_count_loss(
    crystal_out,
    gt_n_nodes: int,
    max_nodes: int = 8,
) -> torch.Tensor:
    """
    MSE entre el soft count de nodos (top-K) y el ground truth.

    IMPORTANTE: usa solo los top-K logits (diferenciable via topk.values).
    Si usáramos todas las L posiciones, el gradiente empujaría TODOS los
    logits hacia -inf simultáneamente → colapso a 0 nodos.
    Con top-K, el gradiente fluye únicamente a las K posiciones más activas,
    permitiendo al modelo aprender a seleccionar los nodos correctos.
    """
    node_scores = crystal_out.node_scores           # [B, L]
    L = node_scores.shape[1]
    K = min(L, max_nodes)

    # topk().values tiene grad_fn → backprop solo a las K posiciones top
    topk_logits = node_scores.topk(K, dim=1).values  # [B, K]
    soft_count  = torch.sigmoid(topk_logits).sum(dim=1)  # [B]

    # Floor: evitar colapso total (garantiza señal de gradiente siempre)
    soft_count  = soft_count.clamp(min=0.5)

    gt_tensor = torch.full_like(soft_count, float(gt_n_nodes))
    return F.mse_loss(soft_count, gt_tensor)


def compute_relation_loss(
    crystal_out,
    gt_adj: torch.Tensor,
    n: int,
) -> Optional[torch.Tensor]:
    """
    BCE entre los logits de relación del crystallizer y la adyacencia GT.

    crystal_out.relation_logits: [B, K, K, 16] — logits por tipo de relación
    gt_adj: [n, n] — binario

    Usamos max sobre los 16 tipos de relación para colapsar a [K, K],
    luego comparamos la submatriz [n, n] (off-diagonal) con gt_adj.
    """
    rel_logits = crystal_out.relation_logits  # [1, K, K, 16]
    K = rel_logits.shape[1]
    n_align = min(n, K)

    if n_align < 2:
        return None  # imposible calcular off-diagonal con < 2 nodos

    # Max sobre tipos → [1, K, K] → [K, K]
    max_logits = rel_logits[0].max(dim=-1).values  # [K, K]
    sub = max_logits[:n_align, :n_align]            # [n_align, n_align]

    gt_sub = gt_adj[:n_align, :n_align].to(sub.device)

    # Máscara off-diagonal
    off_diag = ~torch.eye(n_align, dtype=torch.bool, device=sub.device)

    logits_flat = sub[off_diag]
    target_flat = gt_sub[off_diag]

    if logits_flat.numel() == 0:
        return None

    return F.binary_cross_entropy_with_logits(logits_flat, target_flat)


def compute_cre_coherence_loss(
    cre_node_features: torch.Tensor,
    gt_adj: torch.Tensor,
    n: int,
) -> Optional[torch.Tensor]:
    """
    Pérdida de coherencia para dar gradiente al CRE.

    Compara la similitud entre pares de nodos CRE con la adyacencia GT:
    nodos causalmente conectados deberían tener mayor similitud.

    cre_node_features: [n_nodes, D]
    gt_adj: [n, n]
    """
    n_nodes = cre_node_features.shape[0]
    n_align = min(n, n_nodes)

    if n_align < 2:
        return None

    D = cre_node_features.shape[1]
    feats = cre_node_features[:n_align]            # [n_align, D]
    gt_sub = gt_adj[:n_align, :n_align].to(feats.device)

    # Similitud como producto escalar escalado → logits
    sim = (feats @ feats.T) / math.sqrt(D)         # [n_align, n_align]

    off_diag = ~torch.eye(n_align, dtype=torch.bool, device=feats.device)
    logits_flat = sim[off_diag]
    target_flat = gt_sub[off_diag]

    if logits_flat.numel() == 0:
        return None

    return F.binary_cross_entropy_with_logits(logits_flat, target_flat)


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD PASS (sin decoder)
# ─────────────────────────────────────────────────────────────────────────────

def forward_without_decoder(
    encoder: StreamEncoder,
    crystallizer: GraphCrystallizer,
    cre: CausalReasoningEngine,
    scratch_pad: DifferentiableScratchPad,
    token_ids: torch.Tensor,
    n_cre_iters: int = 3,
):
    """
    Forward pass de encoder → crystallizer → CRE.
    No incluye el decoder (no lo entrenamos todavía).

    Retorna:
        crystal_out: CrystallizerOutput
        cre_node_features: Tensor [n_nodes, D] — features refinadas por CRE
        n_valid_nodes: int
    """
    # Encoder: [B, L] → [B, L, D]
    concept_vectors = encoder(token_ids)

    # Crystallizer: [B, L, D] → CrystallizerOutput
    crystal_out = crystallizer(concept_vectors)

    # CRE: opera en el primer (único) item del batch
    n_nodes = crystal_out.node_counts[0]

    if n_nodes == 0:
        D = concept_vectors.shape[-1]
        dummy = torch.zeros(1, D, device=concept_vectors.device,
                            dtype=concept_vectors.dtype)
        return crystal_out, dummy, 0

    node_feats = crystal_out.node_vectors[0, :n_nodes, :]  # [n_nodes, D]
    graph = crystal_out.graphs[0]

    cre_out = cre(
        graph,
        node_feats,
        scratch_pad=scratch_pad,
        n_iterations=n_cre_iters,
    )

    return crystal_out, cre_out.node_features, n_nodes


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL MODELO (solo encoder + crystallizer + CRE)
# ─────────────────────────────────────────────────────────────────────────────

def build_trainable_modules(cfg: CORAConfig, device: torch.device):
    """
    Instancia encoder, crystallizer, CRE y scratch_pad.
    NO instancia el decoder.
    """
    encoder = StreamEncoder(cfg.encoder_config()).to(device)
    crystallizer = GraphCrystallizer(cfg.crystallizer_config()).to(device)
    cre = CausalReasoningEngine(cfg.cre_config()).to(device)
    scratch_pad = DifferentiableScratchPad(cfg.scratch_pad_config()).to(device)
    return encoder, crystallizer, cre, scratch_pad


def count_params(*modules) -> int:
    total = 0
    seen = set()
    for mod in modules:
        for p in mod.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


# ─────────────────────────────────────────────────────────────────────────────
# COMPARACIÓN PREDICHO VS GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

def compare_prediction_vs_gt(
    encoder: StreamEncoder,
    crystallizer: GraphCrystallizer,
    cre: CausalReasoningEngine,
    scratch_pad: DifferentiableScratchPad,
    example: CausalExample,
    vocab_size: int,
    device: torch.device,
    n_cre_iters: int = 3,
) -> Dict:
    """Corre forward en un ejemplo y retorna métricas para comparación."""
    token_ids = simple_tokenize(
        example.problem_text, vocab_size, device=device
    )

    with torch.no_grad():
        crystal_out, cre_feats, n_valid = forward_without_decoder(
            encoder, crystallizer, cre, scratch_pad,
            token_ids, n_cre_iters=n_cre_iters,
        )

    gt_graph = example.graph
    gt_n_nodes = len(gt_graph.nodes)
    gt_n_edges = len(gt_graph.edges)

    # Contar edges predichos: edges en el grafo discreto del crystallizer
    pred_graph = crystal_out.graphs[0]
    pred_n_nodes = crystal_out.node_counts[0]
    pred_n_edges = len(pred_graph.edges)

    # Tipos de relaciones en GT
    gt_rel_types = list({e.relation.value for e in gt_graph.edges})

    # Tipos de relaciones predichos
    pred_rel_types = list({e.relation.value for e in pred_graph.edges})

    return {
        "text_preview": example.problem_text[:80] + ("..." if len(example.problem_text) > 80 else ""),
        "level": example.complexity_level,
        "gt_nodes": gt_n_nodes,
        "pred_nodes": pred_n_nodes,
        "gt_edges": gt_n_edges,
        "pred_edges": pred_n_edges,
        "gt_relations": gt_rel_types,
        "pred_relations": pred_rel_types,
        "node_delta": pred_n_nodes - gt_n_nodes,
        "edge_delta": pred_n_edges - gt_n_edges,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FORMATEO DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(comparisons: List[Dict]) -> None:
    print()
    print("=" * 72)
    print(" COMPARACION PREDICHO vs GROUND TRUTH (5 ejemplos)")
    print("=" * 72)
    for i, c in enumerate(comparisons, 1):
        node_sign = "+" if c["node_delta"] >= 0 else ""
        edge_sign = "+" if c["edge_delta"] >= 0 else ""
        print(f"\n  [{i}] Nivel {c['level']}: {c['text_preview']}")
        print(f"      Nodos  — GT: {c['gt_nodes']:2d}  Pred: {c['pred_nodes']:2d}  "
              f"Delta: {node_sign}{c['node_delta']}")
        print(f"      Edges  — GT: {c['gt_edges']:2d}  Pred: {c['pred_edges']:2d}  "
              f"Delta: {edge_sign}{c['edge_delta']}")
        if c["gt_relations"]:
            print(f"      Rels GT   : {', '.join(c['gt_relations'][:4])}"
                  + ("..." if len(c["gt_relations"]) > 4 else ""))
        if c["pred_relations"]:
            print(f"      Rels Pred : {', '.join(c['pred_relations'][:4])}"
                  + ("..." if len(c["pred_relations"]) > 4 else ""))
        else:
            print(f"      Rels Pred : (ninguna — threshold no superado)")
    print()
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def train(
    n_examples: int = 500,
    n_steps: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    print_every: int = 10,
    cre_iters: int = 3,
    lambda_cre: float = 0.1,
    output_json: str = "loss_curve.json",
) -> None:
    # ── Setup ────────────────────────────────────────────────────────────────
    device = detect_device()

    cfg = CORAConfig.tiny()
    print(f"\n[config] hidden_dim={cfg.hidden_dim}, vocab_size={cfg.vocab_size}")
    print(f"[config] enc_layers={cfg.enc_n_layers}, cryst_max_nodes={cfg.cryst_max_nodes}")
    print(f"[config] cre_iters={cre_iters}, cre_max_iters={cfg.cre_max_iterations}")

    encoder, crystallizer, cre, scratch_pad = build_trainable_modules(cfg, device)

    n_params = count_params(encoder, crystallizer, cre, scratch_pad)
    print(f"[model] Parametros entrenables (sin decoder): {n_params:,}")

    # ── Datos ────────────────────────────────────────────────────────────────
    print(f"\n[data] Generando {n_examples} ejemplos (niveles 1-2)...")
    t0 = time.perf_counter()
    gen = CausalGraphGenerator(seed=42)
    examples: List[CausalExample] = gen.generate_batch(
        n=n_examples,
        level_distribution={1: 0.5, 2: 0.5},
    )
    t_gen = time.perf_counter() - t0
    print(f"[data] {len(examples)} ejemplos generados en {t_gen:.2f}s")

    # ── Optimizer ────────────────────────────────────────────────────────────
    all_params = (
        list(encoder.parameters())
        + list(crystallizer.parameters())
        + list(cre.parameters())
        + list(scratch_pad.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    # ── Estimación de tiempo ─────────────────────────────────────────────────
    print(f"\n[timing] Midiendo primer step para estimar duración total...")
    _ex = examples[0]
    _tok = simple_tokenize(_ex.problem_text, cfg.vocab_size, device=device)
    t_step_start = time.perf_counter()

    encoder.train(); crystallizer.train(); cre.train(); scratch_pad.train()
    optimizer.zero_grad()
    crystal_out, cre_feats, n_valid = forward_without_decoder(
        encoder, crystallizer, cre, scratch_pad, _tok, n_cre_iters=cre_iters
    )
    gt_n = len(_ex.graph.nodes)
    loss = compute_node_count_loss(crystal_out, gt_n, cfg.cryst_max_nodes)
    loss.backward()
    optimizer.zero_grad()

    t_step_1 = time.perf_counter() - t_step_start
    est_total = t_step_1 * n_steps
    print(f"[timing] 1 step ~ {t_step_1*1000:.0f}ms → {n_steps} steps ~ {est_total:.0f}s "
          f"({est_total/60:.1f} min)")
    if est_total > 600:
        print(f"[timing] ADVERTENCIA: estimado > 10 min en este hardware.")

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  TRAINING: {n_steps} steps | lr={lr} | batch_size=1")
    print(f"{'─'*60}")

    loss_history: List[Dict] = []
    step_times: List[float] = []
    example_cycle = 0

    for step in range(n_steps):
        ex = examples[example_cycle % len(examples)]
        example_cycle += 1

        token_ids = simple_tokenize(ex.problem_text, cfg.vocab_size, device=device)
        gt_graph = ex.graph
        gt_n_nodes = len(gt_graph.nodes)

        t_s = time.perf_counter()

        encoder.train(); crystallizer.train(); cre.train(); scratch_pad.train()
        optimizer.zero_grad()

        crystal_out, cre_feats, n_valid = forward_without_decoder(
            encoder, crystallizer, cre, scratch_pad,
            token_ids, n_cre_iters=cre_iters,
        )

        # ── Losses ──────────────────────────────────────────────────────────
        node_loss = compute_node_count_loss(crystal_out, gt_n_nodes, cfg.cryst_max_nodes)

        n_align = min(n_valid, gt_n_nodes, cfg.cryst_max_nodes)
        gt_adj = build_gt_adjacency(gt_graph, n_align).to(device)

        rel_loss = compute_relation_loss(crystal_out, gt_adj, n_align)
        coh_loss = compute_cre_coherence_loss(cre_feats, gt_adj, n_align)

        total_loss = node_loss
        if rel_loss is not None:
            total_loss = total_loss + rel_loss
        if coh_loss is not None:
            total_loss = total_loss + lambda_cre * coh_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        dt = time.perf_counter() - t_s
        step_times.append(dt)

        record = {
            "step": step + 1,
            "total": float(total_loss.item()),
            "node_count": float(node_loss.item()),
            "relation": float(rel_loss.item()) if rel_loss is not None else None,
            "cre_coherence": float(coh_loss.item()) if coh_loss is not None else None,
            "n_valid_nodes": n_valid,
            "gt_n_nodes": gt_n_nodes,
            "ms": round(dt * 1000, 1),
        }
        loss_history.append(record)

        if (step + 1) % print_every == 0 or step == 0:
            rel_str = f"{rel_loss.item():.4f}" if rel_loss is not None else "N/A"
            coh_str = f"{coh_loss.item():.4f}" if coh_loss is not None else "N/A"
            elapsed = sum(step_times)
            remaining = (n_steps - step - 1) * (elapsed / (step + 1))
            print(
                f"  step {step+1:>3}/{n_steps}"
                f"  loss={total_loss.item():.4f}"
                f"  node={node_loss.item():.4f}"
                f"  rel={rel_str}"
                f"  coh={coh_str}"
                f"  nodes={n_valid}/{gt_n_nodes}"
                f"  ETA={remaining:.0f}s"
            )

    # ── Resultados finales ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    avg_step = sum(step_times) / len(step_times) * 1000
    total_time = sum(step_times)
    print(f"  Training completo en {total_time:.1f}s  ({avg_step:.0f}ms/step)")

    # Curva de loss: primera y última mitad
    mid = n_steps // 2
    first_half = [r["total"] for r in loss_history[:mid]]
    second_half = [r["total"] for r in loss_history[mid:]]
    avg_first  = sum(first_half)  / len(first_half)  if first_half  else float("nan")
    avg_second = sum(second_half) / len(second_half) if second_half else float("nan")
    improvement = avg_first - avg_second
    pct = (improvement / avg_first * 100) if avg_first > 0 else 0.0

    print(f"  Loss promedio (1a mitad): {avg_first:.4f}")
    print(f"  Loss promedio (2a mitad): {avg_second:.4f}")
    if improvement > 0:
        print(f"  Mejora: -{improvement:.4f} ({pct:.1f}%) -- el modelo APRENDE")
    else:
        print(f"  Sin mejora clara aun en {n_steps} steps -- normal al inicio")

    # ── 5 comparaciones predicho vs GT ───────────────────────────────────────
    encoder.eval(); crystallizer.eval(); cre.eval(); scratch_pad.eval()

    # Elegir 5 ejemplos distribuidos (inicio, mitad, fin)
    n_ex = len(examples)
    comparison_indices = [0, n_ex // 4, n_ex // 2, 3 * n_ex // 4, n_ex - 1]
    comparisons = []
    for idx in comparison_indices:
        c = compare_prediction_vs_gt(
            encoder, crystallizer, cre, scratch_pad,
            examples[idx], cfg.vocab_size, device, n_cre_iters=cre_iters,
        )
        comparisons.append(c)

    print_comparison_table(comparisons)

    # ── Guardar JSON ─────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, output_json)
    output_data = {
        "config": {
            "hidden_dim": cfg.hidden_dim,
            "vocab_size": cfg.vocab_size,
            "n_steps": n_steps,
            "lr": lr,
            "cre_iters": cre_iters,
            "lambda_cre": lambda_cre,
            "n_examples": n_examples,
            "n_params": n_params,
        },
        "timing": {
            "total_seconds": round(total_time, 2),
            "avg_ms_per_step": round(avg_step, 1),
        },
        "loss_summary": {
            "avg_first_half": round(avg_first, 6),
            "avg_second_half": round(avg_second, 6),
            "improvement": round(improvement, 6),
            "improvement_pct": round(pct, 2),
        },
        "loss_curve": loss_history,
        "final_comparisons": comparisons,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n[output] Curva de loss guardada en: {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  CORA — Primer Training Loop")
    print("  Encoder + GraphCrystallizer + CRE")
    print("=" * 60)
    train(
        n_examples=500,
        n_steps=200,
        lr=1e-3,
        weight_decay=1e-4,
        print_every=10,
        cre_iters=3,
        lambda_cre=0.1,
        output_json="loss_curve.json",
    )
    print("\n[done] first_training.py completado.")


if __name__ == "__main__":
    main()
