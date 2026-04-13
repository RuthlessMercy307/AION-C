"""
crystallizer/model.py — GraphCrystallizer: concept vectors → CausalGraph
=========================================================================

El GraphCrystallizer es el segundo módulo del pipeline AION-C CEN:

  concept_vectors [B, L, D]     ← del StreamEncoder
          ↓
  NodeDetector
    node_scores    [B, L]        — sigmoid: ¿es nodo esta posición?
    type_logits    [B, L, 7]     — clasificación de tipo de nodo
    confidence     [B, L]        — certeza del detector
          ↓
  Selección top-K (K = min(L, max_nodes))
    topk_indices   [B, K]        — posiciones con mayor node_score
    node_mask      [B, K]        — ≥ node_threshold
          ↓
  CrossAttentionPooler
    node_vectors   [B, K, D]     — vectores enriquecidos con contexto
          ↓
  AsymmetricRelationScorer
    relation_logits [B, K, K, 16] — scores de relación por par dirigido
          ↓
  Construcción de CausalGraph
    grafo discreto con nodos válidos y aristas que superan edge_threshold
          ↓
  CrystallizerOutput
    .graphs           List[CausalGraph]  — estructura discreta (no diferenciable)
    .node_scores      Tensor [B, L]      — diferenciable ← para loss de detección
    .node_type_logits Tensor [B, L, 7]  — diferenciable ← para loss de clasificación
    .node_confidence  Tensor [B, L]      — diferenciable
    .node_vectors     Tensor [B, K, D]   — diferenciable ← para loss de embedding
    .relation_logits  Tensor [B, K, K, 16] — diferenciable ← para loss de relación
    .node_counts      List[int]          — nodos válidos por batch item

DISEÑO PARA DIFERENCIABILIDAD:
    El grafo es una estructura discreta → no hay gradientes a través de él.
    Los tensores suaves (node_scores, relation_logits, etc.) SÍ son diferenciables.
    El entrenamiento usa estos tensores para calcular la pérdida.
    La inferencia usa el grafo discreto para el razonamiento simbólico.

    Analogía: igual que Gumbel-Softmax en VAEs — muestra discreta en forward,
    gradiente continuo en backward.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from core.graph import (
    CAUSAL_RELATIONS,
    NODE_TYPES,
    CausalEdge,
    CausalGraph,
    CausalNode,
    CausalRelation,
    NodeType,
)
from .config import CrystallizerConfig
from .node_detector import NodeDetector
from .pooler import CrossAttentionPooler
from .relation_scorer import AsymmetricRelationScorer


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrystallizerOutput:
    """
    Resultado del GraphCrystallizer para un batch.

    Dos "vistas" del mismo resultado:
      1. Discreta  → graphs:          estructura de datos para el CEC
      2. Continua  → tensores suaves: para calcular pérdida en entrenamiento

    Invariantes:
        len(graphs) == len(node_counts) == B
        node_counts[b] <= max_nodes para todo b
        node_scores.shape == [B, L]
        node_vectors.shape == [B, K, D]  donde K = min(L, max_nodes)
        relation_logits.shape == [B, K, K, n_relation_types]
    """
    graphs:           List[CausalGraph]    # Un grafo por batch item
    node_scores:      torch.Tensor         # [B, L]     — diferenciable
    node_type_logits: torch.Tensor         # [B, L, n_types] — diferenciable
    node_confidence:  torch.Tensor         # [B, L]     — diferenciable
    node_vectors:     torch.Tensor         # [B, K, D]  — diferenciable
    relation_logits:  torch.Tensor         # [B, K, K, R] — diferenciable
    node_counts:      List[int]            # nodos reales (sin padding) por batch


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CRYSTALLIZER
# ─────────────────────────────────────────────────────────────────────────────

class GraphCrystallizer(nn.Module):
    """
    Convierte concept vectors [B, L, D] en CausalGraphs + tensores de entrenamiento.

    Configuración tiny (testing):
        config = CrystallizerConfig(hidden_dim=256, max_nodes=32)

    Uso:
        config = CrystallizerConfig()
        gc     = GraphCrystallizer(config)
        vecs   = torch.randn(2, 64, 256)   # [B=2, L=64, D=256]
        out    = gc(vecs)

        graph_0 = out.graphs[0]            # CausalGraph para el primer ejemplo
        print(graph_0)                     # CausalGraph(id=..., nodes=N, edges=E)

        # Para entrenamiento:
        loss = supervised_node_loss(out.node_scores, target_node_mask)
               + supervised_relation_loss(out.relation_logits, target_relations)
        loss.backward()                    # gradientes fluyen a través de vecs
    """

    def __init__(self, config: CrystallizerConfig) -> None:
        super().__init__()
        self.config = config

        self.node_detector   = NodeDetector(config)
        self.pooler          = CrossAttentionPooler(config)
        self.relation_scorer = AsymmetricRelationScorer(config)

    # ── Forward principal ─────────────────────────────────────────────────────

    def forward(self, concepts: torch.Tensor) -> CrystallizerOutput:
        """
        Args:
            concepts: [B, L, D] — concept vectors del StreamEncoder

        Returns:
            CrystallizerOutput con grafos discretos + tensores diferenciables
        """
        B, L, D = concepts.shape
        cfg = self.config

        # ── 1. Detección de nodos ────────────────────────────────────────────
        node_scores, type_logits, confidence = self.node_detector(concepts)
        # node_scores:  [B, L]
        # type_logits:  [B, L, n_types]
        # confidence:   [B, L]

        # ── 2. Selección de candidatos top-K ────────────────────────────────
        # Tomamos los K = min(L, max_nodes) con mayor score.
        # Luego aplicamos el umbral para decidir cuáles son válidos.
        K = min(L, cfg.max_nodes)
        topk_scores, topk_indices = torch.topk(node_scores, K, dim=1)  # [B, K]
        node_mask = topk_scores >= cfg.node_threshold                   # [B, K] bool

        # Contar nodos válidos por batch item
        node_counts: List[int] = node_mask.sum(dim=1).tolist()

        # ── 3. Pooler — enriquecer con contexto ─────────────────────────────
        # Recoger concept vectors de las K posiciones seleccionadas
        # topk_indices: [B, K] → usamos como índices en dim=1
        gather_idx = topk_indices.unsqueeze(-1).expand(B, K, D)  # [B, K, D]
        node_queries = torch.gather(concepts, 1, gather_idx)      # [B, K, D]

        # Cross-attention: cada nodo consulta la secuencia completa
        node_vectors = self.pooler(node_queries, concepts)         # [B, K, D]

        # ── 4. Puntuación de relaciones ──────────────────────────────────────
        # Batched: [B, K, D] × [B, K, D] → [B, K, K, R]
        relation_logits = self.relation_scorer(node_vectors, node_vectors)
        # [B, K, K, R]

        # ── 5. Construcción de grafos ────────────────────────────────────────
        graphs = self._build_graphs(
            B, K,
            topk_indices,
            node_mask,
            node_counts,
            type_logits,
            confidence,
            node_scores,
            relation_logits,
        )

        return CrystallizerOutput(
            graphs=graphs,
            node_scores=node_scores,
            node_type_logits=type_logits,
            node_confidence=confidence,
            node_vectors=node_vectors,
            relation_logits=relation_logits,
            node_counts=node_counts,
        )

    # ── Construcción del grafo discreto ──────────────────────────────────────

    def _build_graphs(
        self,
        B: int,
        K: int,
        topk_indices:   torch.Tensor,   # [B, K] — posición en la secuencia original
        node_mask:      torch.Tensor,   # [B, K] bool — qué candidatos son válidos
        node_counts:    List[int],
        type_logits:    torch.Tensor,   # [B, L, n_types]
        confidence:     torch.Tensor,   # [B, L]
        node_scores:    torch.Tensor,   # [B, L]
        relation_logits: torch.Tensor,  # [B, K, K, R]
    ) -> List[CausalGraph]:
        """
        Convierte los tensores suaves en estructuras CausalGraph discretas.
        Este método NO contribuye al grafo computacional de autograd.
        """
        graphs: List[CausalGraph] = []

        with torch.no_grad():
            cfg = self.config

            for b in range(B):
                graph = CausalGraph()
                n_valid = node_counts[b]

                if n_valid == 0:
                    graphs.append(graph)
                    continue

                # ── Agregar nodos válidos ─────────────────────────────────
                for i in range(n_valid):
                    seq_pos = topk_indices[b, i].item()  # posición original

                    # Tipo de nodo
                    type_idx = type_logits[b, seq_pos].argmax().item()
                    node_type = NodeType(NODE_TYPES[type_idx])

                    # Confianza en [0, 1]
                    conf_val = float(confidence[b, seq_pos].item())
                    conf_val = max(0.0, min(1.0, conf_val))

                    node = CausalNode(
                        node_id=f"n{i}",
                        label=f"concept_pos{seq_pos}",
                        node_type=node_type,
                        confidence=conf_val,
                        metadata={"seq_pos": int(seq_pos)},
                    )
                    graph.add_node(node)

                # ── Agregar aristas ───────────────────────────────────────
                for i in range(n_valid):
                    for j in range(n_valid):
                        if i == j:
                            continue

                        logits_ij = relation_logits[b, i, j]          # [R]
                        best_rel_idx = logits_ij.argmax().item()
                        edge_strength = torch.sigmoid(
                            logits_ij[best_rel_idx]
                        ).item()

                        if edge_strength < cfg.edge_threshold:
                            continue

                        # Confianza = mínimo de las confianzas de los nodos
                        pos_i = topk_indices[b, i].item()
                        pos_j = topk_indices[b, j].item()
                        conf_i = float(confidence[b, pos_i].item())
                        conf_j = float(confidence[b, pos_j].item())
                        edge_conf = min(conf_i, conf_j) * edge_strength
                        edge_conf = max(0.0, min(1.0, edge_conf))
                        edge_str  = max(0.0, min(1.0, edge_strength))

                        edge = CausalEdge(
                            source_id=f"n{i}",
                            target_id=f"n{j}",
                            relation=CausalRelation(
                                CAUSAL_RELATIONS[best_rel_idx]
                            ),
                            strength=edge_str,
                            confidence=edge_conf,
                        )
                        graph.add_edge(edge)

                graphs.append(graph)

        return graphs

    # ── Utilidades ────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        """Desglose de parámetros por sub-módulo."""
        return {
            "node_detector":   sum(
                p.numel() for p in self.node_detector.parameters()
            ),
            "pooler":          sum(
                p.numel() for p in self.pooler.parameters()
            ),
            "relation_scorer": sum(
                p.numel() for p in self.relation_scorer.parameters()
            ),
            "total":           self.count_parameters(),
        }
