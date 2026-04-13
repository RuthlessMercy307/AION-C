"""
motors/forge_c/motor.py — CodeMotor: implementación BaseMotor para FORGE-C
===========================================================================

FORGE-C razona sobre GRAFOS DE CÓDIGO en lugar de grafos causales abstractos.

La diferencia clave con CORAMotor:
  - Vocabulario propio: 8 tipos de nodo + 12 relaciones de código
  - CRE inicializado con CODE_RELATIONS → funciones de mensaje especializadas
  - CodeCrystallizer: detecta nodos y relaciones con vocabulario de código
  - build_graph produce grafos con CodeNodeType y CodeRelation

PIPELINE:
    concepts [B, L, D]
        ↓  build_graph()
    CodeCrystallizerOutput      (CodeCrystallizer)
        .graphs: List[CausalGraph]  — nodos con CodeNodeType, aristas con CodeRelation
        .node_vectors: [B, K, D]
        ↓  reason()
    CREOutput                   (CausalReasoningEngine + CODE_RELATIONS)
        .node_features: [N, D]
        .edge_features: [E, edge_dim]
        ↓  get_graph_repr()
    [k_nodes, D]                — para el decoder

El CRE es REUTILIZADO: misma arquitectura (message passing + GRU + LayerNorm)
pero con message_fns inicializadas para las 12 relaciones de código, no las 16
relaciones causales de CORA. Los pesos son distintos porque los conceptos son
distintos: CALLS aprende transformaciones de grafo de llamadas, mientras que
CAUSES aprendería transformaciones causales.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from core.graph import CausalGraph, CausalNode
from crystallizer.config import CrystallizerConfig
from crystallizer.node_detector import NodeDetector
from crystallizer.pooler import CrossAttentionPooler
from crystallizer.relation_scorer import AsymmetricRelationScorer
from crystallizer.model import CrystallizerOutput
from cre.config import CREConfig
from cre.engine import CausalReasoningEngine, CREOutput
from motors.base_motor import BaseMotor
from motors.forge_c.relations import (
    CODE_NODE_TYPES,
    CODE_RELATIONS,
    CodeEdge,
    CodeNode,
    CodeNodeType,
    CodeRelation,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CodeMotorConfig:
    """
    Configuración conjunta para CodeMotor.

    crystallizer: controla el CodeCrystallizer (detección de nodos + relaciones)
    cre:          controla el CausalReasoningEngine con CODE_RELATIONS

    Restricciones:
        crystallizer.hidden_dim == cre.node_dim
        crystallizer.n_node_types == 8   (len(CODE_NODE_TYPES))
        crystallizer.n_relation_types == 12 (len(CODE_RELATIONS))
    """
    crystallizer: CrystallizerConfig = field(
        default_factory=lambda: CrystallizerConfig(
            n_node_types=8,       # len(CODE_NODE_TYPES)
            n_relation_types=12,  # len(CODE_RELATIONS)
        )
    )
    cre: CREConfig = field(default_factory=CREConfig)

    def __post_init__(self) -> None:
        if self.crystallizer.hidden_dim != self.cre.node_dim:
            raise ValueError(
                f"crystallizer.hidden_dim ({self.crystallizer.hidden_dim}) "
                f"must equal cre.node_dim ({self.cre.node_dim})"
            )
        if self.crystallizer.n_node_types != len(CODE_NODE_TYPES):
            raise ValueError(
                f"crystallizer.n_node_types must be {len(CODE_NODE_TYPES)} "
                f"(len(CODE_NODE_TYPES)), got {self.crystallizer.n_node_types}"
            )
        if self.crystallizer.n_relation_types != len(CODE_RELATIONS):
            raise ValueError(
                f"crystallizer.n_relation_types must be {len(CODE_RELATIONS)} "
                f"(len(CODE_RELATIONS)), got {self.crystallizer.n_relation_types}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CODE CRYSTALLIZER
# ─────────────────────────────────────────────────────────────────────────────

class CodeCrystallizer(nn.Module):
    """
    Crystallizer especializado para grafos de código.

    Misma arquitectura que GraphCrystallizer (NodeDetector + CrossAttentionPooler +
    AsymmetricRelationScorer) pero:
    - NodeDetector clasifica en 8 tipos de código (CodeNodeType)
    - RelationScorer puntúa 12 relaciones de código (CodeRelation)
    - _build_graphs construye nodos con CodeNodeType y aristas con CodeRelation

    Reutiliza los módulos ya implementados en crystallizer/ — solo cambian
    los hiperparámetros (n_node_types=8, n_relation_types=12) y la asignación
    de tipos en _build_graphs.
    """

    def __init__(self, config: CrystallizerConfig) -> None:
        super().__init__()
        self.config = config
        self.node_detector   = NodeDetector(config)
        self.pooler          = CrossAttentionPooler(config)
        self.relation_scorer = AsymmetricRelationScorer(config)

    def forward(self, concepts: torch.Tensor) -> CrystallizerOutput:
        """
        Args:
            concepts: [B, L, D] — concept vectors del encoder

        Returns:
            CrystallizerOutput con code-typed graphs y tensores diferenciables
        """
        B, L, D = concepts.shape
        cfg = self.config

        node_scores, type_logits, confidence = self.node_detector(concepts)

        K = min(L, cfg.max_nodes)
        topk_scores, topk_indices = torch.topk(node_scores, K, dim=1)
        node_mask   = topk_scores >= cfg.node_threshold
        node_counts: List[int] = node_mask.sum(dim=1).tolist()

        gather_idx  = topk_indices.unsqueeze(-1).expand(B, K, D)
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
        B: int,
        K: int,
        topk_indices:   torch.Tensor,
        node_mask:      torch.Tensor,
        node_counts:    List[int],
        type_logits:    torch.Tensor,
        confidence:     torch.Tensor,
        node_scores:    torch.Tensor,
        relation_logits: torch.Tensor,
    ) -> List[CausalGraph]:
        graphs: List[CausalGraph] = []
        with torch.no_grad():
            cfg = self.config
            for b in range(B):
                graph    = CausalGraph()
                n_valid  = node_counts[b]
                if n_valid == 0:
                    graphs.append(graph)
                    continue

                # ── Nodos con CodeNodeType ────────────────────────────────────
                for i in range(n_valid):
                    seq_pos  = topk_indices[b, i].item()
                    type_idx = type_logits[b, seq_pos].argmax().item()
                    code_type = CodeNodeType(CODE_NODE_TYPES[type_idx])
                    conf_val  = float(max(0.0, min(1.0, confidence[b, seq_pos].item())))
                    node = CodeNode(
                        node_id=f"n{i}",
                        label=f"code_pos{seq_pos}",
                        node_type=code_type,   # CodeNodeType, not NodeType
                        confidence=conf_val,
                        metadata={"seq_pos": int(seq_pos)},
                    )
                    graph.add_node(node)

                # ── Aristas con CodeRelation ──────────────────────────────────
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
                        edge_str  = max(0.0, min(1.0, edge_strength))

                        code_rel = CodeRelation(CODE_RELATIONS[best_rel_idx])
                        edge = CodeEdge(
                            source_id=f"n{i}",
                            target_id=f"n{j}",
                            relation=code_rel,   # CodeRelation, not CausalRelation
                            strength=edge_str,
                            confidence=edge_conf,
                        )
                        graph.add_edge(edge)

                graphs.append(graph)
        return graphs


# ─────────────────────────────────────────────────────────────────────────────
# CODE MOTOR
# ─────────────────────────────────────────────────────────────────────────────

class CodeMotor(BaseMotor):
    """
    Motor de razonamiento sobre código que implementa la interfaz BaseMotor.

    Envuelve CodeCrystallizer (concept vectors → code graph) y
    CausalReasoningEngine configurado con CODE_RELATIONS (message passing
    con funciones especializadas para relaciones de código).

    La arquitectura CRE es idéntica a la de CORA: message passing tipado +
    GRU cell + LayerNorm. Solo cambian las claves del ModuleDict de message_fns
    (12 relaciones de código en lugar de 16 causales).

    Parámetros:
        crystallizer: CodeCrystallizer — construye el grafo de código
        cre:          CausalReasoningEngine — refina con code-specific message fns
    """

    def __init__(self, config: CodeMotorConfig) -> None:
        super().__init__()
        self.config       = config
        self.crystallizer = CodeCrystallizer(config.crystallizer)
        # CRE con vocabulario de relaciones de código
        self.cre          = CausalReasoningEngine(config.cre, relation_keys=CODE_RELATIONS)

    # ── Métodos de introspección ──────────────────────────────────────────────

    def define_node_types(self) -> List[str]:
        """Retorna los 8 tipos de nodo del grafo de código."""
        return list(CODE_NODE_TYPES)

    def define_relations(self) -> List[str]:
        """Retorna las 12 relaciones de código de FORGE-C."""
        return list(CODE_RELATIONS)

    # ── Métodos de procesamiento ──────────────────────────────────────────────

    def build_graph(self, concepts: torch.Tensor) -> CrystallizerOutput:
        """
        Convierte concept vectors en grafos de código.

        Args:
            concepts: [B, L, D] — vectores del encoder (D == crystallizer.hidden_dim)

        Returns:
            CrystallizerOutput con code-typed graphs y tensores diferenciables
        """
        return self.crystallizer(concepts)

    def reason(
        self,
        graph: CausalGraph,
        node_features: torch.Tensor,
        n_iterations: int = 3,
    ) -> CREOutput:
        """
        Refina el grafo de código con message passing iterativo.

        Usa las funciones de mensaje especializadas para CODE_RELATIONS:
        CALLS aprende a propagar señales de "impacto de cambio",
        DATA_FLOWS_TO propaga datos entre nodos, etc.

        Args:
            graph:         CausalGraph con CodeRelation en sus aristas
            node_features: [N, D] — features iniciales
            n_iterations:  int — iteraciones (default: 3)

        Returns:
            CREOutput con node_features refinadas
        """
        return self.cre.forward(graph, node_features, n_iterations=n_iterations)

    def get_graph_repr(
        self,
        cre_output: CREOutput,
        k_nodes: int,
    ) -> torch.Tensor:
        """
        Extrae representación de tamaño fijo del grafo de código para el decoder.

        Selecciona los k_nodes más relevantes por norma L2 (los más "activos"
        semánticamente después del reasoning). Completa con ceros si N < k_nodes.

        Args:
            cre_output: CREOutput — salida de reason()
            k_nodes:    int — número fijo de vectores a devolver

        Returns:
            [k_nodes, D] — representación de tamaño fijo
        """
        h = cre_output.node_features   # [N, D]
        N, D = h.shape
        device = h.device

        if N == 0:
            return torch.zeros(k_nodes, D, device=device, dtype=h.dtype)

        if N >= k_nodes:
            norms   = h.norm(dim=-1)
            top_idx = norms.topk(k_nodes).indices.sort().values
            return h[top_idx]
        else:
            pad = torch.zeros(k_nodes - N, D, device=device, dtype=h.dtype)
            return torch.cat([h, pad], dim=0)
