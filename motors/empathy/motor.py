"""
motors/empathy/motor.py — SocialMotor: implementación BaseMotor para EMPATHY
=============================================================================

EMPATHY razona sobre GRAFOS SOCIALES.
Modela intenciones, creencias, normas y emociones de personas.

La diferencia clave con los otros motores:
  - Vocabulario propio: 8 tipos de nodo + 10 relaciones sociales
  - CRE inicializado con SOCIAL_RELATIONS → funciones de mensaje para interacción social
  - SocialCrystallizer: detecta nodos y relaciones sociales
  - build_graph produce grafos con SocialNodeType y SocialRelation

PIPELINE:
    concepts [B, L, D]
        ↓  build_graph()
    CrystallizerOutput          (SocialCrystallizer)
        .graphs: List[CausalGraph]  — nodos SocialNodeType, aristas SocialRelation
        .node_vectors: [B, K, D]
        ↓  reason()
    CREOutput                   (CausalReasoningEngine + SOCIAL_RELATIONS)
        .node_features: [N, D]     — representaciones refinadas del estado social
        .edge_features: [E, edge_dim]
        ↓  get_graph_repr()
    [k_nodes, D]                — para el decoder

SEMÁNTICA DEL MESSAGE PASSING SOCIAL:
    WANTS:         propaga "qué quiere cada persona" → detecta conflictos de intención
    BELIEVES:      propaga "qué cree cada persona" → detecta creencias falsas
    MISUNDERSTANDS: señal de malentendido → para detección de conflicto implícito
    VIOLATES_NORM: señal de transgresión → modula respuesta empática
    EMPATHIZES:    propaga comprensión emocional bidireccional
    TRUSTS:        propaga confianza (o su ausencia) entre nodos PERSON
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
from motors.empathy.relations import (
    SOCIAL_NODE_TYPES,
    SOCIAL_RELATIONS,
    SocialEdge,
    SocialNode,
    SocialNodeType,
    SocialRelation,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SocialMotorConfig:
    """
    Configuración conjunta para SocialMotor.

    Restricciones:
        crystallizer.hidden_dim == cre.node_dim
        crystallizer.n_node_types == 8   (len(SOCIAL_NODE_TYPES))
        crystallizer.n_relation_types == 10 (len(SOCIAL_RELATIONS))
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
        if self.crystallizer.n_node_types != len(SOCIAL_NODE_TYPES):
            raise ValueError(
                f"crystallizer.n_node_types must be {len(SOCIAL_NODE_TYPES)}, "
                f"got {self.crystallizer.n_node_types}"
            )
        if self.crystallizer.n_relation_types != len(SOCIAL_RELATIONS):
            raise ValueError(
                f"crystallizer.n_relation_types must be {len(SOCIAL_RELATIONS)}, "
                f"got {self.crystallizer.n_relation_types}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# SOCIAL CRYSTALLIZER
# ─────────────────────────────────────────────────────────────────────────────

class SocialCrystallizer(nn.Module):
    """
    Crystallizer especializado para grafos sociales.

    Misma arquitectura que los demás crystallizers:
    NodeDetector (8 tipos) + CrossAttentionPooler + AsymmetricRelationScorer (10 rels).
    _build_graphs crea SocialNode/SocialEdge con SocialNodeType/SocialRelation.
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
                    seq_pos    = topk_indices[b, i].item()
                    type_idx   = type_logits[b, seq_pos].argmax().item()
                    social_type = SocialNodeType(SOCIAL_NODE_TYPES[type_idx])
                    conf_val   = float(max(0.0, min(1.0, confidence[b, seq_pos].item())))
                    node = SocialNode(
                        node_id=f"n{i}",
                        label=f"social_pos{seq_pos}",
                        node_type=social_type,
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
                        social_rel  = SocialRelation(SOCIAL_RELATIONS[best_rel_idx])
                        edge = SocialEdge(
                            source_id=f"n{i}",
                            target_id=f"n{j}",
                            relation=social_rel,
                            strength=max(0.0, min(1.0, edge_strength)),
                            confidence=edge_conf,
                        )
                        graph.add_edge(edge)

                graphs.append(graph)
        return graphs


# ─────────────────────────────────────────────────────────────────────────────
# SOCIAL MOTOR
# ─────────────────────────────────────────────────────────────────────────────

class SocialMotor(BaseMotor):
    """
    Motor de razonamiento social que implementa la interfaz BaseMotor.

    Envuelve SocialCrystallizer y CausalReasoningEngine configurado con
    SOCIAL_RELATIONS (10 funciones de mensaje para relaciones sociales).

    WANTS aprende a propagar deseos y detectar conflictos de intención,
    MISUNDERSTANDS aprende a detectar asimetrías en el modelo mental,
    VIOLATES_NORM aprende a señalar transgresiones sociales,
    EMPATHIZES aprende a propagar comprensión emocional, etc.
    """

    def __init__(self, config: SocialMotorConfig) -> None:
        super().__init__()
        self.config       = config
        self.crystallizer = SocialCrystallizer(config.crystallizer)
        self.cre          = CausalReasoningEngine(config.cre, relation_keys=SOCIAL_RELATIONS)

    def define_node_types(self) -> List[str]:
        """Retorna los 8 tipos de nodo del grafo social."""
        return list(SOCIAL_NODE_TYPES)

    def define_relations(self) -> List[str]:
        """Retorna las 10 relaciones sociales de EMPATHY."""
        return list(SOCIAL_RELATIONS)

    def build_graph(self, concepts: torch.Tensor) -> CrystallizerOutput:
        """
        Convierte concept vectors en grafos sociales.

        Args:
            concepts: [B, L, D] — concept vectors del encoder

        Returns:
            CrystallizerOutput con social-typed graphs y tensores diferenciables
        """
        return self.crystallizer(concepts)

    def reason(
        self,
        graph: CausalGraph,
        node_features: torch.Tensor,
        n_iterations: int = 3,
    ) -> CREOutput:
        """
        Refina el estado social con message passing iterativo.

        Cada iteración propaga intenciones (WANTS, EXPECTS), actualiza
        modelos mentales (BELIEVES, MISUNDERSTANDS) y señala transgresiones
        (VIOLATES_NORM) y respuestas empáticas (EMPATHIZES, RECIPROCATES).

        Args:
            graph:         CausalGraph con SocialRelation en aristas
            node_features: [N, D] — features iniciales (estado social inicial)
            n_iterations:  int    — iteraciones de "razonamiento social"

        Returns:
            CREOutput con node_features refinadas (modelo mental actualizado)
        """
        return self.cre.forward(graph, node_features, n_iterations=n_iterations)

    def get_graph_repr(
        self,
        cre_output: CREOutput,
        k_nodes: int,
    ) -> torch.Tensor:
        """
        Extrae representación de tamaño fijo del estado social para el decoder.

        Los nodos con mayor norma L2 son los "más activos" socialmente
        (los participantes y estados más relevantes tras el razonamiento).

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
