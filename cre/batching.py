"""
cre/batching.py — PyG-style graph batching para AION-C
=======================================================

APPROACH: concatenar grafos en un super-grafo desconectado con offsets
en edge_index. NO se usan nodos dummy / padding.

Por qué funciona sin perder calidad:
    En message passing, cada nodo recibe mensajes solo de sus VECINOS.
    Si Grafo A y Grafo B no tienen edges conectándolos (grafos desconectados),
    los nodos de A NUNCA reciben mensajes de B.
    → El resultado es IDÉNTICO a procesarlos por separado.
    → Es el método estándar de PyTorch Geometric desde 2019.

Diferencia con forward_batch (padding):
    forward_batch: padea hasta max_nodes con nodos dummy que contaminan MP.
    forward_batched: concatena sin padding — grafos desconectados = sin contaminación.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from core.graph import CausalGraph


# ─────────────────────────────────────────────────────────────────────────────
# CONTENEDOR DEL SUPER-GRAFO CONCATENADO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchedGraph:
    """
    Super-grafo concatenado resultado del batching estilo PyG.

    node_features:   [N_total, D]   — todos los nodos concatenados
    edge_index:      [2, E_total]   — [src, tgt] con offsets aplicados
    edge_rel_vals:   List[str]      — strings de tipo de relación (para message_fns
                                      y para que el CRE construya rel_indices según
                                      su propio vocabulario de relaciones)
    edge_strengths:  [E_total]      — strength de cada arista
    edge_confidences:[E_total]      — confidence de cada arista
    batch:           [N_total]      — ID de grafo al que pertenece cada nodo (0-based)
    n_graphs:        int            — número de grafos en el batch
    nodes_per_graph: List[int]      — número de nodos por grafo
    edges_per_graph: List[int]      — número de aristas por grafo

    Nota: NO se almacena edge_type (índices numéricos) porque ese mapeo depende
    del vocabulario de cada motor (CAUSAL_RELATIONS, CODE_RELATIONS, MATH_RELATIONS…).
    El CRE construye los índices internamente en forward_batched usando
    self.relation_keys, que es la fuente de verdad para cada motor.
    """
    node_features:    torch.Tensor           # [N_total, D]
    edge_index:       torch.Tensor           # [2, E_total] long
    edge_rel_vals:    List[str]              # E_total strings
    edge_strengths:   torch.Tensor           # [E_total]
    edge_confidences: torch.Tensor           # [E_total]
    batch:            torch.Tensor           # [N_total] long
    n_graphs:         int
    nodes_per_graph:  List[int]
    edges_per_graph:  List[int]

    @property
    def n_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1] if self.edge_index.numel() > 0 else 0

    @property
    def device(self) -> torch.device:
        return self.node_features.device


# ─────────────────────────────────────────────────────────────────────────────
# BATCHER ESTILO PyG
# ─────────────────────────────────────────────────────────────────────────────

class PyGStyleBatcher:
    """
    Convierte una lista de CausalGraphs en un super-grafo concatenado
    para procesamiento batch en GPU.

    Método de PyTorch Geometric adaptado para AION-C / CORA.
    NO usa padding con nodos dummy — ese approach causó F1 Word de 0.905→0.661.

    Uso:
        batcher = PyGStyleBatcher()
        batched = batcher.batch(graphs, node_features_list)
        # batched.node_features: [N_total, D]
        # batched.edge_index:    [2, E_total] con offsets
        # batched.batch:         [N_total] → ID de grafo por nodo

        results = batcher.unbatch(batched, refined_features)
        # results: List[Tensor] — [n_i, D] por grafo original
    """

    def batch(
        self,
        graphs:              List[CausalGraph],
        node_features_list:  List[torch.Tensor],   # List of [N_i, D]
    ) -> BatchedGraph:
        """
        Construye el super-grafo concatenado con offsets en edge_index.

        El offset asegura que las aristas del Grafo B no apunten a nodos del Grafo A:
            Grafo A (3 nodos): edges [(0,1), (1,2)]         → sin cambio
            Grafo B (5 nodos): edges [(0,1),(1,2),(2,3)]    → +3: [(3,4),(4,5),(5,6)]

        Args:
            graphs:             B CausalGraphs con source_idx/target_idx asignados
            node_features_list: B tensores [N_i, D]

        Returns:
            BatchedGraph con el super-grafo concatenado
        """
        if len(graphs) == 0:
            raise ValueError("graphs list cannot be empty")
        if len(graphs) != len(node_features_list):
            raise ValueError(
                f"len(graphs)={len(graphs)} != len(node_features_list)={len(node_features_list)}"
            )

        device = node_features_list[0].device
        dtype  = node_features_list[0].dtype

        # ── Concatenar node features ──────────────────────────────────────────
        all_features = torch.cat(node_features_list, dim=0)  # [N_total, D]

        # ── Construir edge_index con offsets y metadatos de aristas ───────────
        # rel_vals se almacena como strings — el CRE construye los índices numéricos
        # usando su propio self.relation_keys en forward_batched. Esto permite que el
        # batcher sea agnóstico al vocabulario del motor (CORA, FORGE-C, AXIOM, etc.).
        all_src:       List[int]   = []
        all_tgt:       List[int]   = []
        all_rel_vals:  List[str]   = []
        all_strengths: List[float] = []
        all_confs:     List[float] = []
        batch_ids:     List[int]   = []
        nodes_per_graph: List[int] = []
        edges_per_graph: List[int] = []

        offset = 0
        for graph_idx, (graph, feats) in enumerate(zip(graphs, node_features_list)):
            n_nodes = feats.shape[0]
            n_edges = len(graph.edges)

            nodes_per_graph.append(n_nodes)
            edges_per_graph.append(n_edges)

            # Nodos: registrar batch ID
            batch_ids.extend([graph_idx] * n_nodes)

            # Aristas: aplicar offset a source_idx y target_idx
            for edge in graph.edges:
                all_src.append(edge.source_idx + offset)
                all_tgt.append(edge.target_idx + offset)
                all_rel_vals.append(edge.relation.value)
                all_strengths.append(edge.strength)
                all_confs.append(edge.confidence)

            offset += n_nodes

        # ── Construir tensores ────────────────────────────────────────────────
        E_total = len(all_src)

        if E_total > 0:
            src_t = torch.tensor(all_src, dtype=torch.long, device=device)
            tgt_t = torch.tensor(all_tgt, dtype=torch.long, device=device)
            edge_index = torch.stack([src_t, tgt_t], dim=0)   # [2, E_total]
            strengths  = torch.tensor(all_strengths, dtype=dtype, device=device)
            confs      = torch.tensor(all_confs,     dtype=dtype, device=device)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            strengths  = torch.zeros(0, dtype=dtype, device=device)
            confs      = torch.zeros(0, dtype=dtype, device=device)

        batch = torch.tensor(batch_ids, dtype=torch.long, device=device)

        return BatchedGraph(
            node_features    = all_features,
            edge_index       = edge_index,
            edge_rel_vals    = all_rel_vals,
            edge_strengths   = strengths,
            edge_confidences = confs,
            batch            = batch,
            n_graphs         = len(graphs),
            nodes_per_graph  = nodes_per_graph,
            edges_per_graph  = edges_per_graph,
        )

    def unbatch(
        self,
        batched:          BatchedGraph,
        refined_features: torch.Tensor,   # [N_total, D]
    ) -> List[torch.Tensor]:
        """
        Separa features refinados por grafo original usando el batch vector.

        Args:
            batched:          BatchedGraph original (usa nodes_per_graph)
            refined_features: [N_total, D] — output del forward batched

        Returns:
            List of [N_i, D] tensors, uno por grafo
        """
        results: List[torch.Tensor] = []
        offset = 0
        for n_nodes in batched.nodes_per_graph:
            results.append(refined_features[offset : offset + n_nodes])
            offset += n_nodes
        return results
