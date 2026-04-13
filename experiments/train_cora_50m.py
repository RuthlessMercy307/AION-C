"""
experiments/train_cora_50m.py — CORA 37M: entrenamiento completo end-to-end
=============================================================================

Pipeline completo: encoder(Mamba,8L) → crystallizer → CRE(10 iter) → decoder(8L)

Arquitectura:
  hidden_dim=256, enc_n_layers=8, dec_n_layers=8, vocab_size=8000
  cryst_max_nodes=32, cre_message_layers=2, cre_iters=10
  ~37M parámetros, ~450MB VRAM (modelo+optimizer)

Training:
  5000 ejemplos CausalGraphGenerator L1-4, split 80/20
  AdamW lr=3e-4, cosine decay, grad_clip=1.0, 5000 steps
  Gradient accumulation × 4 (effective batch_size=4)
  Checkpoint cada 1000 steps → experiments/checkpoints/
  Eval cada 500 steps (3 ejemplos, greedy generation)
  Reporte final → experiments/results/

Dispositivo: ROCm (RX 6600) / CUDA / CPU fallback
  HSA_OVERRIDE_GFX_VERSION=10.3.0 aplicado antes de importar torch

Ejecutar:
  python -m experiments.train_cora_50m

Tiempo estimado en RX 6600: < 30 minutos
"""

from __future__ import annotations

# ── ROCm: configurar ANTES de que torch inicialice CUDA ─────────────────────
# RX 6600 es gfx1032 — ROCm no la soporta oficialmente sin override
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import io
import json
import math
import random
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── UTF-8 stdout ─────────────────────────────────────────────────────────────
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
from decoder import StreamDecoder
from router.pipeline import CORAConfig


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

N_DATASET      = 5000
TRAIN_FRAC     = 0.80
N_STEPS        = 5000
LR_INIT        = 3e-4
LR_MIN         = 1e-5
WEIGHT_DECAY   = 1e-2
GRAD_CLIP      = 1.0
ACCUM_STEPS    = 4          # effective batch_size = 4
PRINT_EVERY    = 100
EVAL_EVERY     = 500
CKPT_EVERY     = 1000
CRE_ITERS      = 10
LAMBDA_ND      = 2.0        # node detection BCE (supervisión posicional)
LAMBDA_REL     = 1.0
LAMBDA_COH     = 0.1
LAMBDA_LM      = 2.0        # LM loss pesa más — es la tarea principal
LAMBDA_LEX     = 0.5        # lexical grounding: ancla nodos a tokens del input
MAX_Q_LEN      = 80         # tokens de pregunta
MAX_A_LEN      = 48         # tokens de respuesta
VRAM_ABORT_GB  = 7.5        # abortar si el modelo necesita más de esto


# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARIO
# ─────────────────────────────────────────────────────────────────────────────

class SimpleVocab:
    """
    Vocabulario por frecuencia construido desde el dataset.
    Tokens especiales: PAD=0, BOS=1, EOS=2, UNK=3.
    Permite encode/decode — no usa hash (reversible).
    """
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    N_SPECIAL = 4

    def __init__(self, max_size: int = 8000):
        self.max_size = max_size
        self.word2id: Dict[str, int] = {
            "<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3,
        }
        self.id2word: Dict[int, str] = {v: k for k, v in self.word2id.items()}
        self._counts: Counter = Counter()

    def add_texts(self, texts: List[str]) -> None:
        for text in texts:
            self._counts.update(text.lower().split())

    def build(self) -> None:
        slots = self.max_size - self.N_SPECIAL
        for word, _ in self._counts.most_common(slots):
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx]  = word

    def encode(
        self,
        text: str,
        max_len: int = 128,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        words = text.lower().split()[:max_len]
        ids: List[int] = []
        if add_bos:
            ids.append(self.BOS_ID)
        ids.extend(self.word2id.get(w, self.UNK_ID) for w in words)
        if add_eos:
            ids.append(self.EOS_ID)
        return ids or [self.UNK_ID]

    def decode(self, ids: List[int]) -> str:
        skip = {self.PAD_ID, self.BOS_ID, self.EOS_ID}
        return " ".join(
            self.id2word.get(i, "<UNK>")
            for i in ids
            if i not in skip
        )

    def to_tensor(
        self,
        ids: List[int],
        device: torch.device,
        max_len: Optional[int] = None,
        pad_to: Optional[int] = None,
    ) -> torch.Tensor:
        if max_len:
            ids = ids[:max_len]
        if pad_to and len(ids) < pad_to:
            ids = ids + [self.PAD_ID] * (pad_to - len(ids))
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]

    def __len__(self) -> int:
        return len(self.word2id)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────

def make_50m_config(vocab_size: int = 8000) -> CORAConfig:
    """
    Configuración CORA 37M (nombrado 50M en el plan por el target original).
    hidden_dim=256 fluye por todo el pipeline.
    """
    return CORAConfig(
        hidden_dim   = 256,
        vocab_size   = vocab_size,
        # Encoder: Mamba SSM, 8 capas
        enc_n_layers  = 8,
        enc_state_dim = 16,
        enc_expand    = 2,
        enc_d_conv    = 4,
        enc_ffn_mult  = 4,
        # Crystallizer: 32 nodos máximo, 8 cabezas
        cryst_max_nodes      = 32,
        cryst_n_heads        = 8,
        cryst_node_threshold = 0.01,
        cryst_edge_threshold = 0.01,
        # CRE: 2 capas de message passing, 10 iteraciones
        cre_edge_dim         = 64,
        cre_message_dim      = 128,
        cre_n_message_layers = 2,
        cre_max_iterations   = CRE_ITERS,
        # ScratchPad
        pad_n_slots  = 32,
        pad_slot_dim = 128,
        # Decoder: Mamba + cross-attention, 8 capas
        dec_n_layers    = 8,
        dec_n_heads     = 8,
        dec_max_seq_len = 256,
        dec_state_dim   = 16,
        dec_expand      = 2,
        dec_d_conv      = 4,
        dec_ffn_mult    = 4,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN DE DISPOSITIVO Y VRAM
# ─────────────────────────────────────────────────────────────────────────────

def detect_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        tag     = "ROCm/HIP" if is_rocm else "CUDA"
        return torch.device("cuda"), f"{tag} — {torch.cuda.get_device_name(0)}"
    return torch.device("cpu"), "CPU (GPU no detectada)"


def check_vram(device: torch.device, n_params: int) -> None:
    """
    Estima uso de VRAM. Aborta si no hay suficiente margen.
    En CPU: solo informa de RAM estimada.
    """
    model_mb  = n_params * 4 / 1e6               # float32 pesos
    optim_mb  = n_params * 4 * 2 / 1e6           # AdamW m + v
    activ_mb  = 400.0                              # activaciones con batch_accum×4
    total_est = model_mb + optim_mb + activ_mb

    print(f"\n[vram] Estimacion VRAM/RAM:")
    print(f"       Modelo (f32)       : {model_mb:7.1f} MB")
    print(f"       Optimizer (AdamW)  : {optim_mb:7.1f} MB")
    print(f"       Activaciones (est) : {activ_mb:7.1f} MB")
    print(f"       TOTAL estimado     : {total_est:7.1f} MB  ({total_est/1024:.2f} GB)")

    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info(device)
        free_gb  = free_b  / 1e9
        total_gb = total_b / 1e9
        print(f"       GPU disponible     : {free_gb:.2f} GB / {total_gb:.2f} GB")
        if total_est / 1024 > VRAM_ABORT_GB:
            raise RuntimeError(
                f"VRAM insuficiente: el modelo necesita ~{total_est/1024:.1f}GB "
                f"pero el limite de seguridad es {VRAM_ABORT_GB}GB. "
                f"Reduce hidden_dim o n_layers."
            )
        margin = free_gb - total_est / 1024
        status = "OK" if margin > 1.0 else ("AJUSTADO" if margin > 0 else "RIESGO OOM")
        print(f"       Margen libre       : {margin:.2f} GB  [{status}]")
        if status == "RIESGO OOM":
            raise RuntimeError(
                f"Margen de VRAM insuficiente: {margin:.2f}GB libre. "
                f"Necesitas al menos 1GB de margen."
            )
    else:
        print(f"       [CPU] Sin limite de VRAM. RAM estimada: {total_est:.0f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(n: int, seed: int = 42) -> Tuple[List[CausalExample], List[CausalExample]]:
    """Genera n ejemplos L1-4 y hace split 80/20."""
    print(f"\n[data] Generando {n} ejemplos (niveles 1-4)...")
    t0 = time.perf_counter()
    gen = CausalGraphGenerator(seed=seed)
    examples = gen.generate_batch(
        n=n,
        level_distribution={1: 0.30, 2: 0.30, 3: 0.25, 4: 0.15},
    )
    rng = random.Random(seed)
    rng.shuffle(examples)
    split = int(n * TRAIN_FRAC)
    train_ex, eval_ex = examples[:split], examples[split:]
    dt = time.perf_counter() - t0
    lvl = {1: 0, 2: 0, 3: 0, 4: 0}
    for e in examples:
        lvl[e.complexity_level] = lvl.get(e.complexity_level, 0) + 1
    print(f"[data] {len(train_ex)} train / {len(eval_ex)} eval  ({dt:.2f}s)")
    print(f"[data] L1={lvl[1]} L2={lvl[2]} L3={lvl[3]} L4={lvl[4]}")
    return train_ex, eval_ex


def build_vocab(examples: List[CausalExample], max_size: int) -> SimpleVocab:
    """Construye vocabulario a partir de todas las preguntas y respuestas."""
    vocab = SimpleVocab(max_size=max_size)
    vocab.add_texts([e.problem_text for e in examples])
    vocab.add_texts([e.answer       for e in examples])
    vocab.build()
    print(f"[vocab] Vocabulario: {len(vocab):,} tokens (max={max_size})")
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH ADJACENCY
# ─────────────────────────────────────────────────────────────────────────────

def build_gt_adjacency(graph: CausalGraph, n: int) -> torch.Tensor:
    nodes    = graph.nodes[:n]
    node_idx = {nd.node_id: i for i, nd in enumerate(nodes)}
    adj = torch.zeros(n, n)
    for edge in graph.edges:
        if edge.source_id in node_idx and edge.target_id in node_idx:
            adj[node_idx[edge.source_id], node_idx[edge.target_id]] = 1.0
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────

def forward_full(
    encoder:      StreamEncoder,
    crystallizer: GraphCrystallizer,
    cre:          CausalReasoningEngine,
    scratch_pad:  DifferentiableScratchPad,
    decoder:      StreamDecoder,
    q_ids:        torch.Tensor,     # [1, L_q] — pregunta
    a_ids:        torch.Tensor,     # [1, L_a] — respuesta (para teacher forcing)
    cfg:          CORAConfig,
    n_cre_iters:  int,
    vocab:        SimpleVocab,
):
    """
    Forward pass completo:
      encoder(q) → crystallizer → CRE → graph_repr
      decoder_teacher_forced(a, graph_repr) → lm_logits

    Retorna (crystal_out, graph_repr, cre_feats, n_valid, lm_logits, concepts).
    concepts se incluye para que compute_all_losses pueda calcular lexical grounding.
    """
    device = q_ids.device
    D      = cfg.hidden_dim
    K      = cfg.cryst_max_nodes

    # ── Phase 1: Encode question ─────────────────────────────────────────────
    concepts    = encoder(q_ids)               # [1, L_q, D]
    crystal_out = crystallizer(concepts)       # CrystallizerOutput

    # ── Phase 2: CRE per graph ───────────────────────────────────────────────
    n_valid = crystal_out.node_counts[0]
    if n_valid == 0:
        cre_feats = torch.zeros(1, D, device=device, dtype=concepts.dtype)
    else:
        node_feats = crystal_out.node_vectors[0, :n_valid, :]
        cre_out    = cre(
            crystal_out.graphs[0],
            node_feats,
            scratch_pad  = scratch_pad,
            n_iterations = n_cre_iters,
        )
        cre_feats = cre_out.node_features  # [n_valid, D]

    # ── Phase 3: Build graph_repr [1, K, D] ──────────────────────────────────
    n = cre_feats.shape[0]
    if n >= K:
        padded = cre_feats[:K]
    elif n == 0:
        padded = torch.zeros(K, D, device=device, dtype=concepts.dtype)
    else:
        pad    = torch.zeros(K - n, D, device=device, dtype=concepts.dtype)
        padded = torch.cat([cre_feats, pad], dim=0)

    # Enriquecer cada slot del grafo con el contexto del encoder más relevante.
    # Operación sin parámetros: cada nodo atiende softly a los tokens del input
    # y suma su contexto, preservando la identidad léxica que el CRE puede perder.
    scale      = math.sqrt(D)
    attn_w     = torch.softmax(
        (padded.unsqueeze(0) @ concepts.transpose(1, 2)) / scale, dim=-1
    )                                          # [1, K, L_q]
    enc_ctx    = attn_w @ concepts             # [1, K, D]
    graph_repr = padded.unsqueeze(0) + enc_ctx  # [1, K, D]

    # ── Phase 4: Teacher-forced decode ───────────────────────────────────────
    # dec_input  = [BOS, a0, a1, ..., a_{T-2}]  shape [1, L_a]
    # dec_target =     [a0, a1, ..., a_{T-1}]  shape [1, L_a]
    bos       = torch.full((1, 1), vocab.BOS_ID, dtype=torch.long, device=device)
    dec_input = torch.cat([bos, a_ids[:, :-1]], dim=1)  # [1, L_a]
    dec_out   = decoder(dec_input, graph_repr, concepts)  # logits [1, L_a, V]

    return crystal_out, graph_repr, cre_feats, n_valid, dec_out.logits, concepts


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE LOSS
# ─────────────────────────────────────────────────────────────────────────────

def loss_node_detection(
    crystal_out,
    entity_spans: List[Tuple[int, int]],
    q_len: int,
) -> torch.Tensor:
    """
    BCE por posición sobre node_scores[0, :q_len].

    Target: tokens que pertenecen a una entidad del grafo GT → 1.0
            todos los demás tokens → 0.0

    Reemplaza la antigua loss de conteo (MSE sobre cuántos nodos hay).
    La supervisión posicional le dice al NodeDetector DÓNDE están los nodos,
    no solo cuántos — lo que causaba sobreactivación con la loss anterior.
    """
    scores = crystal_out.node_scores[0, :q_len]   # [q_len], raw logits
    target = torch.zeros_like(scores)
    for start, end in entity_spans:
        if start < 0:          # nodo no encontrado en el texto
            continue
        s = min(start, q_len)
        e = min(end, q_len)
        if e > s:
            target[s:e] = 1.0
    # bce_with_logits: AMP-safe, numerically stable, expects raw logits (no sigmoid)
    return F.binary_cross_entropy_with_logits(scores, target)


def loss_lexical_grounding(
    node_vectors: torch.Tensor,   # [1, K, D] — vectores iniciales del crystallizer
    encoder_out:  torch.Tensor,   # [1, L_q, D] — salida del encoder
    n_valid:      int,
) -> Optional[torch.Tensor]:
    """
    Fuerza cada nodo activo a mantenerse cerca de su token de encoder más cercano.
    Impide que el CRE abstraiga completamente la identidad léxica.
    Los targets se detienen del grafo computacional (solo arrastramos nodos).
    """
    if n_valid == 0:
        return None
    nodes = node_vectors[0, :n_valid]          # [n_valid, D]
    enc   = encoder_out[0]                     # [L_q, D]
    scale = math.sqrt(nodes.shape[-1])
    attn  = torch.softmax((nodes @ enc.T) / scale, dim=-1)  # [n_valid, L_q]
    enc_targets = (attn @ enc).detach()        # [n_valid, D] — no grad por targets
    return F.mse_loss(nodes, enc_targets)


def loss_relation(crystal_out, gt_adj: torch.Tensor, n: int) -> Optional[torch.Tensor]:
    """BCE sobre relation logits (max por tipo) vs adyacencia GT."""
    K       = crystal_out.relation_logits.shape[1]
    n_align = min(n, K)
    if n_align < 2:
        return None
    max_l  = crystal_out.relation_logits[0].max(dim=-1).values  # [K, K]
    sub    = max_l[:n_align, :n_align]
    gt_sub = gt_adj[:n_align, :n_align].to(sub.device)
    mask   = ~torch.eye(n_align, dtype=torch.bool, device=sub.device)
    fl, ft = sub[mask], gt_sub[mask]
    return F.binary_cross_entropy_with_logits(fl, ft) if fl.numel() > 0 else None


def loss_cre_coherence(cre_feats: torch.Tensor, gt_adj: torch.Tensor, n: int) -> Optional[torch.Tensor]:
    """BCE entre similitud coseno de pares CRE y adyacencia GT."""
    n_align = min(n, cre_feats.shape[0])
    if n_align < 2:
        return None
    D      = cre_feats.shape[1]
    feats  = cre_feats[:n_align]
    gt     = gt_adj[:n_align, :n_align].to(feats.device)
    sim    = (feats @ feats.T) / math.sqrt(D)
    mask   = ~torch.eye(n_align, dtype=torch.bool, device=feats.device)
    fl, ft = sim[mask], gt[mask]
    return F.binary_cross_entropy_with_logits(fl, ft) if fl.numel() > 0 else None


def loss_lm(lm_logits: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Cross-entropy teacher-forced sobre la respuesta."""
    # logits: [1, L_a, V],  target: [1, L_a]
    V = lm_logits.shape[-1]
    return F.cross_entropy(
        lm_logits.reshape(-1, V),
        target_ids.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=0.1,
    )


def compute_all_losses(
    crystal_out, cre_feats, n_valid, lm_logits,
    gt_graph: CausalGraph, a_ids: torch.Tensor,
    cfg: CORAConfig, vocab: SimpleVocab, device: torch.device,
    encoder_out: torch.Tensor,                  # [1, L_q, D] — salida del encoder
    entity_spans: List[Tuple[int, int]],        # spans de entidades en q_ids
    q_len: int,                                  # tokens reales (sin padding)
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combina las cinco pérdidas y retorna total + desglose."""
    gt_n    = len(gt_graph.nodes)
    n_align = min(n_valid, gt_n, cfg.cryst_max_nodes)
    gt_adj  = build_gt_adjacency(gt_graph, n_align).to(device)

    # Relation loss: usa gt_n directo (no n_valid) para que gradientes fluyan
    # incluso cuando NodeDetector no ha aprendido aún.
    K           = crystal_out.relation_logits.shape[1]
    n_align_rel = min(gt_n, K)
    gt_adj_rel  = build_gt_adjacency(gt_graph, n_align_rel).to(device)

    l_nd  = loss_node_detection(crystal_out, entity_spans, q_len)
    l_rel = loss_relation(crystal_out, gt_adj_rel, n_align_rel)
    l_coh = loss_cre_coherence(cre_feats, gt_adj, n_align)
    l_lm  = loss_lm(lm_logits, a_ids, vocab.PAD_ID)
    l_lex = loss_lexical_grounding(crystal_out.node_vectors, encoder_out, n_valid)

    total = LAMBDA_ND * l_nd
    if l_rel is not None: total = total + LAMBDA_REL * l_rel
    if l_coh is not None: total = total + LAMBDA_COH * l_coh
    total = total + LAMBDA_LM * l_lm
    if l_lex is not None: total = total + LAMBDA_LEX * l_lex

    breakdown = {
        "total":   float(total.item()),
        "nd":      float(l_nd.item()),
        "rel":     float(l_rel.item()) if l_rel is not None else None,
        "coh":     float(l_coh.item()) if l_coh is not None else None,
        "lm":      float(l_lm.item()),
        "lex":     float(l_lex.item()) if l_lex is not None else None,
        "n_valid": n_valid,
        "gt_n":    gt_n,
    }
    return total, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def node_span_acc(
    node_scores_1d: torch.Tensor,          # [L], raw logits (pre-sigmoid)
    entity_spans: List[Tuple[int, int]],
    threshold: float = 0.0,               # 0.0 ≡ sigmoid 0.5 decision boundary
) -> float:
    """
    Recall de detección de entidades: fracción de spans GT con al menos
    un token cuyo node_score supera el umbral.
    Métrica directa de si el NodeDetector aprendió a activar las posiciones
    correctas del input, en vez de comparar conteos abstractos.
    """
    valid = [(s, e) for s, e in entity_spans if s >= 0]
    if not valid:
        return 0.0
    L = len(node_scores_1d)
    detected = sum(
        1 for s, e in valid
        if s < L and node_scores_1d[s:min(e, L)].max().item() > threshold
    )
    return detected / len(valid)


def relation_recall(crystal_out, gt_graph: CausalGraph, n_align: int) -> float:
    """Fracción de GT edges detectados (max relation_logit > 0 en posición alineada)."""
    if n_align < 1:
        return 0.0
    gt_nodes = gt_graph.nodes[:n_align]
    idx = {nd.node_id: i for i, nd in enumerate(gt_nodes)}
    pairs = {(idx[e.source_id], idx[e.target_id])
             for e in gt_graph.edges
             if e.source_id in idx and e.target_id in idx}
    if not pairs:
        return 1.0
    rl = crystal_out.relation_logits[0]  # [K, K, 16]
    detected = sum(
        1 for (i, j) in pairs
        if i < rl.shape[0] and j < rl.shape[1] and rl[i, j].max().item() > 0
    )
    return detected / len(pairs)


def word_overlap(pred: str, target: str) -> float:
    """F1 de solapamiento de palabras (token-level)."""
    pred_w   = set(pred.lower().split())
    target_w = set(target.lower().split())
    if not target_w:
        return 1.0
    precision = len(pred_w & target_w) / max(len(pred_w), 1)
    recall    = len(pred_w & target_w) / len(target_w)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN GREEDY
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_sampled(
    decoder:     StreamDecoder,
    graph_repr:  torch.Tensor,      # [1, K, D]
    vocab:       SimpleVocab,
    max_len:     int = MAX_A_LEN,
    device:      Optional[torch.device] = None,
    temperature: float = 0.8,
) -> str:
    """Genera respuesta token a token con temperature sampling (evita mode collapse)."""
    if device is None:
        device = graph_repr.device
    ids = torch.full((1, 1), vocab.BOS_ID, dtype=torch.long, device=device)
    generated: List[int] = []

    for _ in range(max_len):
        out     = decoder(ids, graph_repr)
        logits  = out.logits[0, -1, :]          # [V]
        probs   = torch.softmax(logits / temperature, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        if next_id == vocab.EOS_ID:
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    return vocab.decode(generated)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    encoder: StreamEncoder,
    crystallizer: GraphCrystallizer,
    cre: CausalReasoningEngine,
    scratch_pad: DifferentiableScratchPad,
    decoder: StreamDecoder,
    eval_examples: List[CausalExample],
    cfg: CORAConfig,
    vocab: SimpleVocab,
    device: torch.device,
    step: int,
    n_show: int = 3,
) -> Dict:
    """Evalúa n_show ejemplos. Imprime tabla. Retorna métricas."""
    encoder.eval(); crystallizer.eval(); cre.eval()
    scratch_pad.eval(); decoder.eval()

    n_ex = len(eval_examples)
    indices = sorted(random.sample(range(n_ex), min(n_show, n_ex)))

    span_accs, rel_recalls, overlaps = [], [], []
    shown: List[Dict] = []

    for idx in indices:
        ex    = eval_examples[idx]
        q_len = min(len(ex.problem_text.lower().split()), MAX_Q_LEN)
        q_ids = vocab.to_tensor(
            vocab.encode(ex.problem_text, MAX_Q_LEN), device, max_len=MAX_Q_LEN
        )
        a_ids = vocab.to_tensor(
            vocab.encode(ex.answer, MAX_A_LEN, add_eos=True), device
        )

        crystal_out, graph_repr, cre_feats, n_valid, lm_logits, _enc_out = forward_full(
            encoder, crystallizer, cre, scratch_pad, decoder,
            q_ids, a_ids, cfg, CRE_ITERS, vocab,
        )

        gt_n    = len(ex.graph.nodes)
        n_align = min(n_valid, gt_n, cfg.cryst_max_nodes)
        # Span-based node accuracy: recall de spans de entidades detectados
        sa      = node_span_acc(crystal_out.node_scores[0, :q_len], ex.entity_spans)
        rr      = relation_recall(crystal_out, ex.graph, n_align)
        gen_text = generate_sampled(decoder, graph_repr, vocab, device=device)
        ov       = word_overlap(gen_text, ex.answer)

        n_spans_found = sum(
            1 for s, e in ex.entity_spans
            if s >= 0 and s < q_len
            and crystal_out.node_scores[0, s:min(e, q_len)].max().item() > 0.0
        )

        span_accs.append(sa); rel_recalls.append(rr); overlaps.append(ov)
        shown.append({
            "level":        ex.complexity_level,
            "gt_n":         gt_n,
            "n_spans":      len([sp for sp in ex.entity_spans if sp[0] >= 0]),
            "spans_found":  n_spans_found,
            "gt_edges":     len(ex.graph.edges),
            "span_acc":     round(sa, 3),
            "rel_recall":   round(rr, 3),
            "word_overlap": round(ov, 3),
            "q_preview":    ex.problem_text[:60],
            "gt_answer":    ex.answer,
            "gen_answer":   gen_text,
        })

    # ── Print ────────────────────────────────────────────────────────────────
    sep = "-" * 76
    print(f"\n  {sep}")
    print(f"  EVAL @ step {step:>5}   ({n_show} ejemplos aleatorios de eval set)")
    print(f"  {sep}")
    for r in shown:
        print(f"\n  [L{r['level']}] {r['q_preview']}...")
        print(f"    Entidades GT:{r['gt_n']:>2}  Detectadas:{r['spans_found']:>2}/{r['n_spans']:>2}"
              f"  SpanAcc:{r['span_acc']:.0%}"
              f"  Edges GT:{r['gt_edges']:>2}  RelRecall:{r['rel_recall']:.0%}")
        print(f"    GT  : {r['gt_answer'][:72]}")
        print(f"    Gen : {r['gen_answer'][:72] or '(vacio)'}")
        print(f"    WordF1: {r['word_overlap']:.0%}")
    print(f"\n  {sep}")
    avg_sa = sum(span_accs)  / len(span_accs)
    avg_rr = sum(rel_recalls) / len(rel_recalls)
    avg_ov = sum(overlaps)   / len(overlaps)
    print(f"  Promedio: span_acc={avg_sa:.1%}  rel_recall={avg_rr:.1%}  word_F1={avg_ov:.1%}")

    encoder.train(); crystallizer.train(); cre.train()
    scratch_pad.train(); decoder.train()

    return {
        "step":            step,
        "avg_span_acc":    round(avg_sa, 4),
        "avg_rel_recall":  round(avg_rr, 4),
        "avg_word_f1":     round(avg_ov, 4),
        "examples":        shown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    step: int,
    encoder: StreamEncoder,
    crystallizer: GraphCrystallizer,
    cre: CausalReasoningEngine,
    scratch_pad: DifferentiableScratchPad,
    decoder: StreamDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_history: List[Dict],
    ckpt_dir: str,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"cora_50m_step{step:05d}.pt")
    torch.save({
        "step":          step,
        "encoder":       encoder.state_dict(),
        "crystallizer":  crystallizer.state_dict(),
        "cre":           cre.state_dict(),
        "scratch_pad":   scratch_pad.state_dict(),
        "decoder":       decoder.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "loss_history":  loss_history[-200:],   # últimos 200 records
    }, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# REPORTE FINAL
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report(
    loss_history: List[Dict],
    eval_snapshots: List[Dict],
    total_time: float,
    avg_ms: float,
    backend: str,
    n_params: int,
) -> Dict:
    print("\n" + "=" * 76)
    print("  REPORTE FINAL — CORA 50M")
    print("=" * 76)

    mid = N_STEPS // 2

    def avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    l_first  = [r["total"] for r in loss_history[:mid]]
    l_second = [r["total"] for r in loss_history[mid:]]
    lm_first = [r["lm"]    for r in loss_history[:mid] if r.get("lm")]
    lm_second= [r["lm"]    for r in loss_history[mid:] if r.get("lm")]

    l1, l2   = avg(l_first), avg(l_second)
    lm1, lm2 = avg(lm_first), avg(lm_second)
    l_imp    = (l1 - l2) / l1 * 100 if l1 > 0 else 0
    lm_imp   = (lm1 - lm2) / lm1 * 100 if lm1 > 0 else 0

    # Span accuracy trend from eval snapshots
    na_vals = [s["avg_span_acc"] for s in eval_snapshots]
    rr_vals = [s["avg_rel_recall"] for s in eval_snapshots]
    wf_vals = [s["avg_word_f1"]   for s in eval_snapshots]

    print(f"\n  Dispositivo : {backend}")
    print(f"  Parametros  : {n_params:,}  (~{n_params/1e6:.0f}M)")
    print(f"  Duracion    : {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"  Velocidad   : {avg_ms:.0f} ms/paso")

    print(f"\n  Loss total  (1a mitad → 2a mitad): {l1:.4f} → {l2:.4f}  ({l_imp:+.1f}%)")
    print(f"  LM loss     (1a mitad → 2a mitad): {lm1:.4f} → {lm2:.4f}  ({lm_imp:+.1f}%)")

    sig = l_imp > 5.0
    print(f"  Mejora significativa (>5%): {'SI' if sig else 'NO  (normal en primeros pasos)'}")

    if eval_snapshots:
        print(f"\n  Progreso durante evaluaciones:")
        print(f"  {'Step':>6}  {'SpanAcc':>7}  {'RelRecall':>9}  {'WordF1':>7}")
        for s in eval_snapshots:
            print(f"  {s['step']:>6}  {s['avg_span_acc']:>7.1%}  {s['avg_rel_recall']:>9.1%}  {s['avg_word_f1']:>7.1%}")

    if len(na_vals) >= 2:
        na_trend = "mejora" if na_vals[-1] > na_vals[0] else "estable"
        wf_trend = "mejora" if wf_vals[-1] > wf_vals[0] else "estable"
        print(f"\n  Span accuracy: {na_trend} ({na_vals[0]:.1%} → {na_vals[-1]:.1%})")
        print(f"  Word F1:       {wf_trend} ({wf_vals[0]:.1%} → {wf_vals[-1]:.1%})")

    print("=" * 76)

    return {
        "backend": backend,
        "n_params": n_params,
        "total_seconds": round(total_time, 1),
        "avg_ms_per_step": round(avg_ms, 1),
        "loss_total_first_half": round(l1, 6),
        "loss_total_second_half": round(l2, 6),
        "loss_total_improvement_pct": round(l_imp, 2),
        "lm_loss_first_half": round(lm1, 6),
        "lm_loss_second_half": round(lm2, 6),
        "lm_loss_improvement_pct": round(lm_imp, 2),
        "significant_improvement": sig,
        "eval_span_acc_trend": na_vals,
        "eval_rel_recall_trend": rr_vals,
        "eval_word_f1_trend": wf_vals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:
    ts_start = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Directorios de salida ─────────────────────────────────────────────────
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir  = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ── Header ───────────────────────────────────────────────────────────────
    print("=" * 76)
    print("  CORA 50M — Entrenamiento End-to-End")
    print("  Encoder(8L) + Crystallizer + CRE(10iter) + Decoder(8L)")
    print("=" * 76)

    device, backend = detect_device()
    print(f"\n[device] {backend}")
    print(f"[rocm]   HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION','')}")

    # ── Dataset + Vocab ───────────────────────────────────────────────────────
    train_ex, eval_ex = build_dataset(N_DATASET)
    cfg   = make_50m_config(vocab_size=8000)
    vocab = build_vocab(train_ex + eval_ex, max_size=cfg.vocab_size)
    # Actualizar config con vocab real (puede ser < 8000 si el dataset es pequeño)
    actual_vocab = len(vocab)
    cfg   = make_50m_config(vocab_size=actual_vocab)

    print(f"\n[config] hidden_dim={cfg.hidden_dim}  enc_layers={cfg.enc_n_layers}  "
          f"dec_layers={cfg.dec_n_layers}")
    print(f"[config] cryst_max_nodes={cfg.cryst_max_nodes}  "
          f"cre_iters={CRE_ITERS}  vocab={actual_vocab}")
    print(f"[train]  {N_STEPS} steps  accum×{ACCUM_STEPS}  "
          f"lr {LR_INIT:.0e}→{LR_MIN:.0e}")

    # ── Construir módulos ─────────────────────────────────────────────────────
    encoder      = StreamEncoder(cfg.encoder_config()).to(device)
    crystallizer = GraphCrystallizer(cfg.crystallizer_config()).to(device)
    cre_engine   = CausalReasoningEngine(cfg.cre_config()).to(device)
    scratch_pad  = DifferentiableScratchPad(cfg.scratch_pad_config()).to(device)
    decoder      = StreamDecoder(cfg.decoder_config()).to(device)

    seen, n_params = set(), 0
    for mod in [encoder, crystallizer, cre_engine, scratch_pad, decoder]:
        for p in mod.parameters():
            if id(p) not in seen:
                seen.add(id(p)); n_params += p.numel()

    print(f"[model]  {n_params:,} parametros ({n_params/1e6:.1f}M)")

    # ── VRAM check ────────────────────────────────────────────────────────────
    check_vram(device, n_params)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    all_params = (
        list(encoder.parameters())
        + list(crystallizer.parameters())
        + list(cre_engine.parameters())
        + list(scratch_pad.parameters())
        + list(decoder.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_STEPS, eta_min=LR_MIN
    )

    # ── Estimación de tiempo ──────────────────────────────────────────────────
    print(f"\n[timing] Midiendo throughput (1 step con accum×{ACCUM_STEPS})...")
    _ex   = train_ex[0]
    _qlen = min(len(_ex.problem_text.lower().split()), MAX_Q_LEN)
    _q    = vocab.to_tensor(vocab.encode(_ex.problem_text, MAX_Q_LEN), device, MAX_Q_LEN)
    _a    = vocab.to_tensor(vocab.encode(_ex.answer, MAX_A_LEN, add_eos=True), device)

    if device.type == "cuda":
        for _ in range(2):  # calentamiento
            crys, gr, cf, nv, lml, enc_out = forward_full(
                encoder, crystallizer, cre_engine, scratch_pad, decoder,
                _q, _a, cfg, CRE_ITERS, vocab,
            )
            loss_dummy, _ = compute_all_losses(
                crys, cf, nv, lml, _ex.graph, _a, cfg, vocab, device,
                enc_out, _ex.entity_spans, _qlen,
            )
            (loss_dummy / ACCUM_STEPS).backward()
        for p in all_params:
            if p.grad is not None: p.grad.zero_()
        torch.cuda.synchronize()

    t_ref = time.perf_counter()
    for _ in range(ACCUM_STEPS):
        crys, gr, cf, nv, lml, enc_out = forward_full(
            encoder, crystallizer, cre_engine, scratch_pad, decoder,
            _q, _a, cfg, CRE_ITERS, vocab,
        )
        loss_ref, _ = compute_all_losses(
            crys, cf, nv, lml, _ex.graph, _a, cfg, vocab, device,
            enc_out, _ex.entity_spans, _qlen,
        )
        (loss_ref / ACCUM_STEPS).backward()
    for p in all_params:
        if p.grad is not None: p.grad.zero_()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_step = (time.perf_counter() - t_ref) * 1000
    est_s   = ms_step * N_STEPS / 1000
    print(f"[timing] {ms_step:.0f} ms/step → {N_STEPS} steps ≈ {est_s:.0f}s ({est_s/60:.1f} min)")
    if est_s > 1800:
        print(f"[timing] ADVERTENCIA: estimado > 30 min en este hardware.")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'─'*76}")
    print(f"  ENTRENANDO: {N_STEPS} steps  |  accum={ACCUM_STEPS}  |  grad_clip={GRAD_CLIP}")
    print(f"{'─'*76}")

    loss_history:   List[Dict]  = []
    eval_snapshots: List[Dict]  = []
    step_times:     List[float] = []
    recent_loss     = deque(maxlen=50)
    ex_idx          = 0

    for step in range(N_STEPS):
        t_s = time.perf_counter()

        # ── Gradient accumulation × ACCUM_STEPS ───────────────────────────────
        optimizer.zero_grad()
        accum_breakdown: Dict[str, list] = {k: [] for k in ["total","nd","rel","coh","lm","lex","n_valid","gt_n"]}

        for acc in range(ACCUM_STEPS):
            # Shuffle training examples at epoch boundaries (mode-collapse fix)
            if ex_idx > 0 and ex_idx % len(train_ex) == 0:
                random.shuffle(train_ex)
            ex    = train_ex[ex_idx % len(train_ex)]
            ex_idx += 1
            q_len = min(len(ex.problem_text.lower().split()), MAX_Q_LEN)
            q_ids = vocab.to_tensor(vocab.encode(ex.problem_text, MAX_Q_LEN), device, MAX_Q_LEN)
            a_ids = vocab.to_tensor(vocab.encode(ex.answer, MAX_A_LEN, add_eos=True), device)

            crys_out, graph_repr, cre_feats, n_valid, lm_logits, enc_out = forward_full(
                encoder, crystallizer, cre_engine, scratch_pad, decoder,
                q_ids, a_ids, cfg, CRE_ITERS, vocab,
            )
            total, bd = compute_all_losses(
                crys_out, cre_feats, n_valid, lm_logits,
                ex.graph, a_ids, cfg, vocab, device, enc_out,
                ex.entity_spans, q_len,
            )
            (total / ACCUM_STEPS).backward()

            for k, v in bd.items():
                if v is not None:
                    accum_breakdown[k].append(v)

        # ── Optimizer step ─────────────────────────────────────────────────────
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t_s
        step_times.append(dt)

        # ── Registro ───────────────────────────────────────────────────────────
        def mean_or_none(lst): return round(sum(lst)/len(lst), 6) if lst else None

        record = {
            "step":    step + 1,
            "total":   mean_or_none(accum_breakdown["total"]),
            "nd":      mean_or_none(accum_breakdown["nd"]),
            "rel":     mean_or_none(accum_breakdown["rel"]),
            "coh":     mean_or_none(accum_breakdown["coh"]),
            "lm":      mean_or_none(accum_breakdown["lm"]),
            "lex":     mean_or_none(accum_breakdown["lex"]),
            "n_valid": mean_or_none(accum_breakdown["n_valid"]),
            "gt_n":    mean_or_none(accum_breakdown["gt_n"]),
            "lr":      round(scheduler.get_last_lr()[0], 8),
            "ms":      round(dt * 1000, 1),
        }
        loss_history.append(record)
        recent_loss.append(record["total"])

        # ── Print periódico ────────────────────────────────────────────────────
        if (step + 1) % PRINT_EVERY == 0 or step == 0:
            elapsed   = sum(step_times)
            remaining = (N_STEPS - step - 1) * elapsed / (step + 1)
            avg_l     = sum(recent_loss) / len(recent_loss)
            cur_lr    = scheduler.get_last_lr()[0]
            lm_str    = f"{record['lm']:.4f}" if record["lm"] else "N/A"
            rel_str   = f"{record['rel']:.4f}" if record["rel"] else "N/A"
            lex_str   = f"{record['lex']:.4f}" if record.get("lex") is not None else "N/A"
            vram_str  = ""
            if device.type == "cuda":
                try:
                    alloc = torch.cuda.memory_allocated(device) / 1e6
                    vram_str = f"  vram={alloc:.0f}MB"
                except Exception:
                    pass
            print(
                f"  step {step+1:>5}/{N_STEPS}"
                f"  loss={avg_l:.4f}"
                f"  lm={lm_str}"
                f"  rel={rel_str}"
                f"  lex={lex_str}"
                f"  lr={cur_lr:.1e}"
                f"  ETA={remaining:.0f}s"
                f"{vram_str}"
            )

            # ── Diagnóstico NodeDetector (cada PRINT_EVERY steps) ──────────────
            try:
                _diag_ex = train_ex[ex_idx % len(train_ex)]
                _spans   = _diag_ex.entity_spans
                _n_spans = sum(1 for s in _spans if s[0] >= 0)
                _n_total = len(_spans)
                with torch.no_grad():
                    _q_len = min(len(_diag_ex.problem_text.lower().split()), MAX_Q_LEN)
                    _q_ids = vocab.to_tensor(vocab.encode(_diag_ex.problem_text, MAX_Q_LEN), device, MAX_Q_LEN)
                    _a_ids = vocab.to_tensor(vocab.encode(_diag_ex.answer, MAX_A_LEN, add_eos=True), device)
                    _cout, *_ = forward_full(
                        encoder, crystallizer, cre_engine, scratch_pad, decoder,
                        _q_ids, _a_ids, cfg, CRE_ITERS, vocab,
                    )
                    _scores = _cout.node_scores[0, :_q_len].cpu()
                    _in_mask  = torch.zeros(_q_len, dtype=torch.bool)
                    for _s, _e in _spans:
                        if _s >= 0:
                            _in_mask[min(_s,_q_len):min(_e,_q_len)] = True
                    _out_mask = ~_in_mask
                    _in_mean  = _scores[_in_mask].mean().item()  if _in_mask.any()  else float('nan')
                    _out_mean = _scores[_out_mask].mean().item() if _out_mask.any() else float('nan')
                    _nd_val   = record.get("nd")
                print(
                    f"  [diag] spans={_n_spans}/{_n_total}"
                    f"  score_in={_in_mean:.3f}"
                    f"  score_out={_out_mean:.3f}"
                    f"  nd_loss={_nd_val:.4f}" if _nd_val is not None else f"  nd_loss=N/A"
                )
            except Exception as _diag_e:
                print(f"  [diag] error: {_diag_e}")

        # ── Evaluación periódica ───────────────────────────────────────────────
        if (step + 1) % EVAL_EVERY == 0:
            snap = evaluate(
                encoder, crystallizer, cre_engine, scratch_pad, decoder,
                eval_ex, cfg, vocab, device, step + 1,
            )
            eval_snapshots.append(snap)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (step + 1) % CKPT_EVERY == 0:
            ckpt_path = save_checkpoint(
                step + 1, encoder, crystallizer, cre_engine,
                scratch_pad, decoder, optimizer, scheduler,
                loss_history, ckpt_dir,
            )
            print(f"\n  [ckpt] Guardado: {ckpt_path}")

    # ── Reporte final ─────────────────────────────────────────────────────────
    total_time = sum(step_times)
    avg_ms     = total_time / len(step_times) * 1000
    summary    = print_final_report(
        loss_history, eval_snapshots,
        total_time, avg_ms, backend, n_params,
    )

    # ── Guardar JSON ──────────────────────────────────────────────────────────
    out_path = os.path.join(results_dir, f"train_cora_50m_{ts_start}.json")
    output = {
        "meta": {
            "timestamp": ts_start,
            "backend":   backend,
            "hsa_override": os.environ.get("HSA_OVERRIDE_GFX_VERSION", ""),
        },
        "config": {
            "hidden_dim":   cfg.hidden_dim,
            "vocab_size":   actual_vocab,
            "n_params":     n_params,
            "n_steps":      N_STEPS,
            "accum_steps":  ACCUM_STEPS,
            "lr_init":      LR_INIT,
            "lr_min":       LR_MIN,
            "cre_iters":    CRE_ITERS,
            "lambda_lm":    LAMBDA_LM,
            "lambda_rel":   LAMBDA_REL,
            "lambda_coh":   LAMBDA_COH,
            "n_train":      len(train_ex),
            "n_eval":       len(eval_ex),
        },
        "summary":        summary,
        "eval_snapshots": eval_snapshots,
        "loss_curve":     loss_history,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[output] Resultados en: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    train()
    print("\n[done] train_cora_50m.py completado.")


if __name__ == "__main__":
    main()
