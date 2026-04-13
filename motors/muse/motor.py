"""
motors/muse/motor.py — CreativeMotor: implementación BaseMotor para MUSE
=========================================================================

MUSE razona sobre GRAFOS NARRATIVOS.
No razona sobre lógica sino sobre TENSIÓN y ARCO.

La diferencia clave con CORA, FORGE-C y AXIOM:
  - Vocabulario propio: 8 tipos de nodo + 10 relaciones narrativas
  - CRE inicializado con NARRATIVE_RELATIONS → funciones de mensaje para arcos
  - NarrativeCrystallizer: detecta nodos y relaciones narrativas
  - build_graph produce grafos con NarrativeNodeType y NarrativeRelation

PIPELINE:
    concepts [B, L, D]
        ↓  build_graph()
    CrystallizerOutput          (NarrativeCrystallizer)
        .graphs: List[CausalGraph]  — nodos NarrativeNodeType, aristas NarrativeRelation
        .node_vectors: [B, K, D]
        ↓  reason()
    CREOutput                   (CausalReasoningEngine + NARRATIVE_RELATIONS)
        .node_features: [N, D]     — representaciones refinadas del estado narrativo
        .edge_features: [E, edge_dim]
        ↓  get_graph_repr()
    [k_nodes, D]                — para el decoder

SEMÁNTICA DEL MESSAGE PASSING NARRATIVO:
    MOTIVATES:      propaga "qué impulsa a qué" hacia adelante
    CONFLICTS_WITH: señal de tensión activa entre nodos
    DEVELOPS_INTO:  propaga transformación narrativa
    RESOLVES:       propaga cierre de tensión hacia CONFLICT
    INTENSIFIES:    amplifica la señal de tensión
    SUBVERTS:       inversión de expectativa (para giros narrativos)
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from core.graph import CausalGraph
from crystallizer.config import CrystallizerConfig
from crystallizer.node_detector import NodeDetector
from crystallizer.pooler import CrossAttentionPooler
from crystallizer.relation_scorer import AsymmetricRelationScorer
from crystallizer.model import CrystallizerOutput
from cre.config import CREConfig
from cre.engine import CausalReasoningEngine, CREOutput
from motors.base_motor import BaseMotor
from motors.muse.relations import (
    NARRATIVE_NODE_TYPES,
    NARRATIVE_RELATIONS,
    NarrativeEdge,
    NarrativeNode,
    NarrativeNodeType,
    NarrativeRelation,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CreativeMotorConfig:
    """
    Configuración conjunta para CreativeMotor.

    Restricciones:
        crystallizer.hidden_dim == cre.node_dim
        crystallizer.n_node_types == 8   (len(NARRATIVE_NODE_TYPES))
        crystallizer.n_relation_types == 10 (len(NARRATIVE_RELATIONS))
    """
    crystallizer: CrystallizerConfig = field(
        default_factory=lambda: CrystallizerConfig(
            n_node_types=8,
            n_relation_types=10,
        )
    )
    cre: CREConfig = field(
        default_factory=lambda: CREConfig(n_relation_types=10)
    )

    def __post_init__(self) -> None:
        if self.crystallizer.hidden_dim != self.cre.node_dim:
            raise ValueError(
                f"crystallizer.hidden_dim ({self.crystallizer.hidden_dim}) "
                f"must equal cre.node_dim ({self.cre.node_dim})"
            )
        if self.crystallizer.n_node_types != len(NARRATIVE_NODE_TYPES):
            raise ValueError(
                f"crystallizer.n_node_types must be {len(NARRATIVE_NODE_TYPES)}, "
                f"got {self.crystallizer.n_node_types}"
            )
        if self.crystallizer.n_relation_types != len(NARRATIVE_RELATIONS):
            raise ValueError(
                f"crystallizer.n_relation_types must be {len(NARRATIVE_RELATIONS)}, "
                f"got {self.crystallizer.n_relation_types}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVE CRYSTALLIZER
# ─────────────────────────────────────────────────────────────────────────────

class NarrativeCrystallizer(nn.Module):
    """
    Crystallizer especializado para grafos narrativos.

    Misma arquitectura que CodeCrystallizer y MathCrystallizer:
    NodeDetector (8 tipos) + CrossAttentionPooler + AsymmetricRelationScorer (10 rels).
    _build_graphs crea NarrativeNode/NarrativeEdge con NarrativeNodeType/NarrativeRelation.
    """

    def __init__(self, config: CrystallizerConfig) -> None:
        super().__init__()
        self.config          = config
        self.node_detector   = NodeDetector(config)
        self.pooler          = CrossAttentionPooler(config)
        self.relation_scorer = AsymmetricRelationScorer(config)

    def forward(self, concepts: torch.Tensor) -> CrystallizerOutput:
        B, L, D = concepts.shape
        cfg = self.config

        node_scores, type_logits, confidence = self.node_detector(concepts)

        K = min(L, cfg.max_nodes)
        topk_scores, topk_indices = torch.topk(node_scores, K, dim=1)
        node_mask    = topk_scores >= cfg.node_threshold
        node_counts: List[int] = node_mask.sum(dim=1).tolist()

        gather_idx   = topk_indices.unsqueeze(-1).expand(B, K, D)
        node_queries = torch.gather(concepts, 1, gather_idx)
        node_vectors = self.pooler(node_queries, concepts)
        relation_logits = self.relation_scorer(node_vectors, node_vectors)

        graphs = self._build_graphs(
            B, K, topk_indices, node_mask, node_counts,
            type_logits, confidence, node_scores, relation_logits,
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

    def _build_graphs(
        self,
        B, K, topk_indices, node_mask, node_counts,
        type_logits, confidence, node_scores, relation_logits,
    ) -> List[CausalGraph]:
        graphs: List[CausalGraph] = []
        with torch.no_grad():
            cfg = self.config
            for b in range(B):
                graph   = CausalGraph()
                n_valid = node_counts[b]
                if n_valid == 0:
                    graphs.append(graph)
                    continue

                for i in range(n_valid):
                    seq_pos   = topk_indices[b, i].item()
                    type_idx  = type_logits[b, seq_pos].argmax().item()
                    narr_type = NarrativeNodeType(NARRATIVE_NODE_TYPES[type_idx])
                    conf_val  = float(max(0.0, min(1.0, confidence[b, seq_pos].item())))
                    node = NarrativeNode(
                        node_id=f"n{i}",
                        label=f"narr_pos{seq_pos}",
                        node_type=narr_type,
                        confidence=conf_val,
                        metadata={"seq_pos": int(seq_pos)},
                    )
                    graph.add_node(node)

                for i in range(n_valid):
                    for j in range(n_valid):
                        if i == j:
                            continue
                        logits_ij    = relation_logits[b, i, j]
                        best_rel_idx = logits_ij.argmax().item()
                        edge_strength = float(torch.sigmoid(logits_ij[best_rel_idx]).item())
                        if edge_strength < cfg.edge_threshold:
                            continue
                        pos_i = topk_indices[b, i].item()
                        pos_j = topk_indices[b, j].item()
                        conf_i = float(confidence[b, pos_i].item())
                        conf_j = float(confidence[b, pos_j].item())
                        edge_conf = max(0.0, min(1.0, min(conf_i, conf_j) * edge_strength))
                        narr_rel  = NarrativeRelation(NARRATIVE_RELATIONS[best_rel_idx])
                        edge = NarrativeEdge(
                            source_id=f"n{i}",
                            target_id=f"n{j}",
                            relation=narr_rel,
                            strength=max(0.0, min(1.0, edge_strength)),
                            confidence=edge_conf,
                        )
                        graph.add_edge(edge)

                graphs.append(graph)
        return graphs


# ─────────────────────────────────────────────────────────────────────────────
# CREATIVE MOTOR
# ─────────────────────────────────────────────────────────────────────────────

class CreativeMotor(BaseMotor):
    """
    Motor de razonamiento narrativo que implementa la interfaz BaseMotor.

    Envuelve NarrativeCrystallizer y CausalReasoningEngine configurado con
    NARRATIVE_RELATIONS (10 funciones de mensaje para relaciones narrativas).

    MOTIVATES aprende a propagar motivación hacia adelante,
    CONFLICTS_WITH aprende a detectar y mantener tensión,
    RESOLVES aprende a propagar el cierre de conflictos,
    SUBVERTS aprende a señalar giros narrativos inesperados, etc.
    """

    def __init__(self, config: CreativeMotorConfig) -> None:
        super().__init__()
        self.config       = config
        self.crystallizer = NarrativeCrystallizer(config.crystallizer)
        self.cre          = CausalReasoningEngine(config.cre, relation_keys=NARRATIVE_RELATIONS)

    def define_node_types(self) -> List[str]:
        """Retorna los 8 tipos de nodo del grafo narrativo."""
        return list(NARRATIVE_NODE_TYPES)

    def define_relations(self) -> List[str]:
        """Retorna las 10 relaciones narrativas de MUSE."""
        return list(NARRATIVE_RELATIONS)

    def build_graph(self, concepts: torch.Tensor) -> CrystallizerOutput:
        """
        Convierte concept vectors en grafos narrativos.

        Args:
            concepts: [B, L, D] — concept vectors del encoder

        Returns:
            CrystallizerOutput con narrative-typed graphs y tensores diferenciables
        """
        return self.crystallizer(concepts)

    def reason(
        self,
        graph: CausalGraph,
        node_features: torch.Tensor,
        n_iterations: int = 3,
    ) -> CREOutput:
        """
        Refina el estado narrativo con message passing iterativo.

        Cada iteración propaga motivaciones (MOTIVATES, DEVELOPS_INTO),
        mantiene tensiones abiertas (CONFLICTS_WITH, INTENSIFIES) y
        propaga resoluciones (RESOLVES).

        Args:
            graph:         CausalGraph con NarrativeRelation en aristas
            node_features: [N, D] — features iniciales (estado narrativo inicial)
            n_iterations:  int    — iteraciones de "razonamiento narrativo"

        Returns:
            CREOutput con node_features refinadas (estado narrativo final)
        """
        return self.cre.forward(graph, node_features, n_iterations=n_iterations)

    def get_graph_repr(
        self,
        cre_output: CREOutput,
        k_nodes: int,
    ) -> torch.Tensor:
        """
        Extrae representación de tamaño fijo del estado narrativo para el decoder.

        Los nodos con mayor norma L2 son los "más activos" narrativamente
        (los elementos del arco más relevantes tras el razonamiento).

        Args:
            cre_output: CREOutput — salida de reason()
            k_nodes:    int — número fijo de vectores

        Returns:
            [k_nodes, D]
        """
        h = cre_output.node_features
        N, D = h.shape
        device = h.device

        if N == 0:
            return torch.zeros(k_nodes, D, device=device, dtype=h.dtype)
        if N >= k_nodes:
            top_idx = h.norm(dim=-1).topk(k_nodes).indices.sort().values
            return h[top_idx]
        else:
            pad = torch.zeros(k_nodes - N, D, device=device, dtype=h.dtype)
            return torch.cat([h, pad], dim=0)
