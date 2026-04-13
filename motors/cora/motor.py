"""
motors/cora/motor.py — CORAMotor: implementación BaseMotor para razonamiento causal
====================================================================================

CORAMotor (Causal Online Reasoning Architecture Motor) envuelve los módulos
existentes GraphCrystallizer y CausalReasoningEngine en la interfaz BaseMotor.

NO reimplementa ninguna lógica — solo delega en los módulos ya existentes.

PIPELINE:
    concepts [B, L, D]
        ↓  build_graph()
    CrystallizerOutput          (GraphCrystallizer)
        .graphs: List[CausalGraph]
        .node_vectors: [B, K, D]
        ↓  reason()
    CREOutput                   (CausalReasoningEngine)
        .node_features: [N, D] — refinadas
        .edge_features: [E, edge_dim]
        ↓  get_graph_repr()
    [k_nodes, D]                — para el decoder

USO:
    from motors.cora import CORAMotor
    from motors.cora.motor import CORAMotorConfig

    config = CORAMotorConfig()
    motor  = CORAMotor(config)

    out     = motor.build_graph(concepts)   # [B, L, D] → CrystallizerOutput
    cre_out = motor.reason(out.graphs[0], out.node_vectors[0, :out.node_counts[0]])
    repr_   = motor.get_graph_repr(cre_out, k_nodes=16)  # [16, D]
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from core.graph import CAUSAL_RELATIONS, NODE_TYPES, CausalGraph
from crystallizer.config import CrystallizerConfig
from crystallizer.model import GraphCrystallizer, CrystallizerOutput
from cre.config import CREConfig
from cre.engine import CausalReasoningEngine, CREOutput
from motors.base_motor import BaseMotor


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CORAMotorConfig:
    """
    Configuración conjunta para CORAMotor.

    Combina CrystallizerConfig y CREConfig en un único objeto
    para facilitar la construcción del motor.

    Las dimensiones deben ser compatibles:
        crystallizer.hidden_dim == cre.node_dim
    """
    crystallizer: CrystallizerConfig = field(default_factory=CrystallizerConfig)
    cre:          CREConfig          = field(default_factory=CREConfig)

    def __post_init__(self) -> None:
        if self.crystallizer.hidden_dim != self.cre.node_dim:
            raise ValueError(
                f"crystallizer.hidden_dim ({self.crystallizer.hidden_dim}) "
                f"must equal cre.node_dim ({self.cre.node_dim})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CORA MOTOR
# ─────────────────────────────────────────────────────────────────────────────

class CORAMotor(BaseMotor):
    """
    Motor de razonamiento causal que implementa la interfaz BaseMotor.

    Envuelve GraphCrystallizer (concept vectors → CausalGraph) y
    CausalReasoningEngine (message passing iterativo) sin reimplementar
    ninguna de su lógica.

    Parámetros:
        crystallizer: GraphCrystallizer — construye el grafo desde concept vectors
        cre:          CausalReasoningEngine — refina el grafo por iteración
    """

    def __init__(self, config: CORAMotorConfig) -> None:
        super().__init__()
        self.config       = config
        self.crystallizer = GraphCrystallizer(config.crystallizer)
        self.cre          = CausalReasoningEngine(config.cre)

    # ── Métodos de introspección ──────────────────────────────────────────────

    def define_node_types(self) -> List[str]:
        """Retorna los 7 tipos de nodo del sistema causal AION-C."""
        return list(NODE_TYPES)

    def define_relations(self) -> List[str]:
        """Retorna las 16 relaciones causales del sistema AION-C."""
        return list(CAUSAL_RELATIONS)

    # ── Métodos de procesamiento ──────────────────────────────────────────────

    def build_graph(self, concepts: torch.Tensor) -> CrystallizerOutput:
        """
        Convierte concept vectors en CausalGraphs.

        Args:
            concepts: [B, L, D] — vectores de concepto (D == crystallizer.hidden_dim)

        Returns:
            CrystallizerOutput:
                .graphs       List[CausalGraph]       — un grafo por batch item
                .node_vectors [B, K, D]               — features de nodos seleccionados
                .node_counts  List[int]                — nodos válidos por batch item
                (+ tensores diferenciables para entrenamiento)
        """
        return self.crystallizer(concepts)

    def reason(
        self,
        graph: CausalGraph,
        node_features: torch.Tensor,
        n_iterations: int = 3,
    ) -> CREOutput:
        """
        Refina el grafo con message passing iterativo.

        Args:
            graph:         CausalGraph — estructura construida por build_graph
            node_features: [N, D]      — features iniciales (típicamente node_vectors[b])
            n_iterations:  int         — iteraciones de refinamiento (default: 3)

        Returns:
            CREOutput:
                .node_features [N, D]   — features refinadas
                .edge_features [E, edge_dim]
                .iterations_run int
        """
        return self.cre.forward(graph, node_features, n_iterations=n_iterations)

    def get_graph_repr(
        self,
        cre_output: CREOutput,
        k_nodes: int,
    ) -> torch.Tensor:
        """
        Extrae representación de tamaño fijo para el decoder.

        Selecciona los k_nodes más relevantes por norma L2 de sus features.
        Si el grafo tiene menos de k_nodes, rellena con ceros.

        Args:
            cre_output: CREOutput — salida de reason()
            k_nodes:    int       — tamaño fijo del output

        Returns:
            [k_nodes, D] — los k nodos más relevantes, ordenados por relevancia
        """
        h = cre_output.node_features    # [N, D]
        N, D = h.shape
        device = h.device

        if N == 0:
            return torch.zeros(k_nodes, D, device=device, dtype=h.dtype)

        if N >= k_nodes:
            # Seleccionar los k_nodes nodos con mayor norma L2
            norms    = h.norm(dim=-1)           # [N]
            top_idx  = norms.topk(k_nodes).indices  # [k_nodes]
            # Ordenar por índice para reproducibilidad determinista
            top_idx  = top_idx.sort().values
            return h[top_idx]                   # [k_nodes, D]
        else:
            # Rellenar con ceros hasta k_nodes
            pad = torch.zeros(k_nodes - N, D, device=device, dtype=h.dtype)
            return torch.cat([h, pad], dim=0)   # [k_nodes, D]
