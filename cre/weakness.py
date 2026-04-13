"""
cre/weakness.py — WeaknessDetector
====================================

Examina el grafo causal y sus representaciones neurales para detectar
debilidades que indican que se necesitan más iteraciones de refinamiento.

POR QUÉ DETECTAR DEBILIDADES:
    El CRE itera el mismo grafo con WEIGHT SHARING. Sin guía, hace N iteraciones
    fijas sin importar si el grafo ya convergió o si tiene problemas irresueltos.

    El WeaknessDetector actúa como un "auditor" después de cada iteración:
    - Si encuentra debilidades → genera focus_mask para concentrar el refinamiento
    - Si no encuentra nada → la ConvergenceGate puede parar antes

    Esto replica lo que hace un razonador humano: no repasar lo que ya está claro,
    sino enfocarse en lo que sigue siendo incierto o contradictorio.

CINCO TIPOS DE DEBILIDADES:

    low_confidence:
        Un nodo no "sabe" qué es. Sus features tienen baja magnitud o el
        confidence_scorer (MLP aprendido) le da un score bajo.
        → Necesita más mensajes de sus vecinos para reforzarse.

    missing_cause:
        Un nodo no-FACT, no-QUESTION no tiene ninguna arista causal entrante.
        Es un "efecto sin causa": el grafo no explica POR QUÉ existe este nodo.
        → Puede necesitar más iteraciones para conectarse con causas latentes.

    unresolved_contradiction:
        Dos nodos conectados por CONTRADICTS y ambos tienen alta confianza.
        Si ambos "creen" ser verdaderos a la vez, la contradicción no se resolvió.
        → Necesita más iteraciones para que uno "ceda" (baje su confianza).

    circular_reasoning:
        Ciclo en el grafo dirigido causal (A→B→C→A).
        Cada nodo en el ciclo "explica" a otro que lo explica a él.
        → No es necesariamente un error pero señala razonamiento circular.

    weak_evidence:
        Un nodo HYPOTHESIS no tiene ninguna arista SUPPORTS entrante.
        Hipótesis sin evidencia a su favor — pura conjetura.
        → Necesita evidencia o debe bajar su confianza.

ARQUITECTURA NEURAL:
    confidence_scorer: Linear(D, C) → GELU → Linear(C, 1)
        Mapea features de nodo → logit de confianza.
        sigmoid(logit) ∈ (0,1): cuán seguro está el nodo de sí mismo.
        Inicializado cerca de 0.5 (inicio incierto, mejora con entrenamiento).

    Los chequeos estructurales (missing_cause, contradiction, cycles, evidence)
    son puramente algorítmicos — sin parámetros adicionales.

FOCUS MASK:
    focus_mask[i] = True  →  nodo i tiene al menos una debilidad
    Usado por el engine para dar más peso al update de nodos débiles.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, field
from typing import Dict, List, Set

import torch
import torch.nn as nn

from core.graph import CausalGraph, CausalRelation, NodeType


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES: qué relaciones cuentan como "causa entrante"
# ─────────────────────────────────────────────────────────────────────────────

# Relaciones que explican la existencia de un nodo (causalidad directa o derivada)
_CAUSAL_IN: Set[str] = {
    CausalRelation.CAUSES.value,
    CausalRelation.ENABLES.value,
    CausalRelation.LEADS_TO.value,
    CausalRelation.FOLLOWS_FROM.value,
    CausalRelation.IMPLIES.value,
    CausalRelation.REQUIRES.value,
    CausalRelation.SUPPORTS.value,
}

# Relaciones de dirección causal fuerte (para detección de ciclos)
_CAUSAL_DIRECTED: Set[str] = {
    CausalRelation.CAUSES.value,
    CausalRelation.ENABLES.value,
    CausalRelation.LEADS_TO.value,
    CausalRelation.IMPLIES.value,
    CausalRelation.FOLLOWS_FROM.value,
    CausalRelation.PRECEDES.value,
}

# Tipos de weakness
LOW_CONFIDENCE            = "low_confidence"
MISSING_CAUSE             = "missing_cause"
UNRESOLVED_CONTRADICTION  = "unresolved_contradiction"
CIRCULAR_REASONING        = "circular_reasoning"
WEAK_EVIDENCE             = "weak_evidence"

WEAKNESS_TYPES: List[str] = [
    LOW_CONFIDENCE,
    MISSING_CAUSE,
    UNRESOLVED_CONTRADICTION,
    CIRCULAR_REASONING,
    WEAK_EVIDENCE,
]


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES DE SALIDA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Weakness:
    """
    Una debilidad detectada en un nodo del grafo.

    Campos:
        node_idx:  índice entero en el tensor de features [0, N)
        node_id:   ID del nodo en el CausalGraph (string)
        type:      una de las constantes WEAKNESS_TYPES
        severity:  [0, 1] — qué tan grave es la debilidad
                   0.0 = apenas detectada, 1.0 = crítica
    """
    node_idx: int
    node_id:  str
    type:     str
    severity: float


@dataclass
class WeaknessReport:
    """
    Reporte completo de debilidades del grafo después de una iteración del CRE.

    Campos:
        weaknesses:    lista de Weakness individuales (puede estar vacía si el grafo es fuerte)
        focus_mask:    [N] bool  — True = nodo tiene debilidades, necesita más refinamiento
        node_severity: [N] float — severidad máxima por nodo (0 si sin debilidades)
        confidence:    [N] float — scores de confianza neural ∈ (0, 1) para todos los nodos
        n_weaknesses:  número total de debilidades detectadas
        mean_severity: severidad media sobre todos los nodos
    """
    weaknesses:    List[Weakness]
    focus_mask:    torch.Tensor    # [N] bool
    node_severity: torch.Tensor    # [N] float
    confidence:    torch.Tensor    # [N] float   ← confianza neural, para ConvergenceGate
    n_weaknesses:  int
    mean_severity: float


# ─────────────────────────────────────────────────────────────────────────────
# WEAKNESS DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class WeaknessDetector(nn.Module):
    """
    Detecta debilidades en el grafo causal combinando análisis estructural
    y puntuación neural de confianza.

    Uso:
        cfg      = CREConfig(node_dim=256)
        detector = WeaknessDetector(cfg.node_dim, cfg.weakness_conf_threshold,
                                    cfg.weakness_conf_hidden)
        report   = detector(graph, node_features, edge_features)
        # report.focus_mask: [N] bool
        # report.n_weaknesses: int
        # report.confidence: [N] float

    La detección es más útil a partir de la 2ª iteración del CRE,
    cuando los features ya contienen señal causal (no solo la inicialización aleatoria).
    """

    def __init__(
        self,
        node_dim:             int   = 256,
        confidence_threshold: float = 0.35,   # sigmoid < threshold → low_confidence
        confidence_hidden:    int   = 64,
        norm_eps:             float = 1e-6,
    ) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.norm_eps = norm_eps

        # ── Neural confidence scorer ──────────────────────────────────────────
        # Mapea features de nodo → logit de confianza.
        # Inicializado con bias pequeño para empezar en incertidumbre media.
        self.confidence_scorer = nn.Sequential(
            nn.Linear(node_dim, confidence_hidden),
            nn.GELU(),
            nn.Linear(confidence_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── API pública ───────────────────────────────────────────────────────────

    def forward(
        self,
        graph:         CausalGraph,
        node_features: torch.Tensor,   # [N, D]
        edge_features: torch.Tensor,   # [E, edge_dim]
    ) -> WeaknessReport:
        """
        Detecta debilidades en el grafo y sus representaciones neurales.

        Args:
            graph:         CausalGraph con nodos y aristas
            node_features: [N, D] — features actuales de los nodos
            edge_features: [E, edge_dim] — features actuales de las aristas

        Returns:
            WeaknessReport con lista de debilidades, focus_mask y severidades
        """
        N = node_features.shape[0]
        device = node_features.device
        nodes = graph.nodes   # List[CausalNode], ordenada por índice de inserción
        edges = graph.edges   # List[CausalEdge]

        # ── 1. Confidence neural scoring ──────────────────────────────────────
        conf_logits = self.confidence_scorer(node_features).squeeze(-1)   # [N]
        confidence  = torch.sigmoid(conf_logits)                           # [N] ∈ (0,1)

        # ── 2. Análisis estructural ───────────────────────────────────────────
        # Índices de nodes para acceso rápido
        node_id_to_idx: Dict[str, int] = {n.node_id: i for i, n in enumerate(nodes)}

        # Construir sets de nodos con debilidades estructurales
        low_conf_set     = self._find_low_confidence(confidence, nodes, node_id_to_idx)
        missing_set      = self._find_missing_cause(nodes, edges, node_id_to_idx)
        contradiction_set = self._find_unresolved_contradictions(
            nodes, edges, node_id_to_idx, confidence
        )
        cycle_set        = self._find_circular_reasoning(nodes, edges, node_id_to_idx)
        evidence_set     = self._find_weak_evidence(nodes, edges, node_id_to_idx)

        # ── 3. Construir lista de Weakness ───────────────────────────────────
        weaknesses: List[Weakness] = []

        # Agrupa por nodo para calcular severidad
        node_severity = torch.zeros(N, device=device, dtype=node_features.dtype)

        for idx, node in enumerate(nodes):
            nid = node.node_id
            sevs = []

            if idx in low_conf_set:
                sev = float(1.0 - confidence[idx].item())
                weaknesses.append(Weakness(idx, nid, LOW_CONFIDENCE, sev))
                sevs.append(sev)

            if idx in missing_set:
                sev = 0.7
                weaknesses.append(Weakness(idx, nid, MISSING_CAUSE, sev))
                sevs.append(sev)

            if idx in contradiction_set:
                sev = float(confidence[idx].item())   # más severo cuanto más alta la conf
                weaknesses.append(Weakness(idx, nid, UNRESOLVED_CONTRADICTION, sev))
                sevs.append(sev)

            if idx in cycle_set:
                sev = 0.6
                weaknesses.append(Weakness(idx, nid, CIRCULAR_REASONING, sev))
                sevs.append(sev)

            if idx in evidence_set:
                sev = 0.5
                weaknesses.append(Weakness(idx, nid, WEAK_EVIDENCE, sev))
                sevs.append(sev)

            if sevs:
                node_severity[idx] = max(sevs)

        # ── 4. focus_mask: nodos con alguna debilidad ─────────────────────────
        focus_mask = node_severity > 0.0   # [N] bool

        # ── 5. Métricas de resumen ────────────────────────────────────────────
        n_weak       = len([w for w in weaknesses if True])  # todas
        mean_sev     = float(node_severity.mean().item()) if N > 0 else 0.0

        return WeaknessReport(
            weaknesses    = weaknesses,
            focus_mask    = focus_mask,
            node_severity = node_severity,
            confidence    = confidence,
            n_weaknesses  = n_weak,
            mean_severity = mean_sev,
        )

    # ── Análisis estructurales ────────────────────────────────────────────────

    def _find_low_confidence(
        self,
        confidence:     torch.Tensor,
        nodes:          list,
        node_id_to_idx: Dict[str, int],
    ) -> Set[int]:
        """Nodos cuyo confidence_scorer devuelve un score bajo."""
        result: Set[int] = set()
        for i in range(len(nodes)):
            if float(confidence[i].item()) < self.confidence_threshold:
                result.add(i)
        return result

    def _find_missing_cause(
        self,
        nodes:          list,
        edges:          list,
        node_id_to_idx: Dict[str, int],
    ) -> Set[int]:
        """
        Nodos que no tienen ninguna arista causal entrante.
        Excluye FACT (ya grounded) y QUESTION (no necesita causa).
        """
        # Nodos que tienen al menos una causa entrante
        has_cause: Set[str] = set()
        for edge in edges:
            if edge.relation.value in _CAUSAL_IN:
                has_cause.add(edge.target_id)

        result: Set[int] = set()
        excluded_types = {NodeType.FACT, NodeType.QUESTION}
        for i, node in enumerate(nodes):
            if node.node_type in excluded_types:
                continue
            if node.node_id not in has_cause:
                result.add(i)
        return result

    def _find_unresolved_contradictions(
        self,
        nodes:          list,
        edges:          list,
        node_id_to_idx: Dict[str, int],
        confidence:     torch.Tensor,
    ) -> Set[int]:
        """
        Nodos conectados por CONTRADICTS donde ambos tienen alta confianza.
        Una contradicción "resuelta" tiene un lado con baja confianza.
        """
        high_threshold = 1.0 - self.confidence_threshold
        result: Set[int] = set()

        for edge in edges:
            if edge.relation != CausalRelation.CONTRADICTS:
                continue
            src_idx = node_id_to_idx.get(edge.source_id, -1)
            tgt_idx = node_id_to_idx.get(edge.target_id, -1)
            if src_idx < 0 or tgt_idx < 0:
                continue
            src_conf = float(confidence[src_idx].item())
            tgt_conf = float(confidence[tgt_idx].item())
            # Ambos con alta confianza → contradicción no resuelta
            if src_conf > high_threshold and tgt_conf > high_threshold:
                result.add(src_idx)
                result.add(tgt_idx)
        return result

    def _find_circular_reasoning(
        self,
        nodes:          list,
        edges:          list,
        node_id_to_idx: Dict[str, int],
    ) -> Set[int]:
        """
        Detecta ciclos en el grafo dirigido causal usando DFS con coloreo.
        Retorna los índices de todos los nodos que participan en algún ciclo.

        Solo considera aristas de _CAUSAL_DIRECTED (ignora CONTRADICTS, CORRELATES, etc.)
        """
        # Construir lista de adyacencia (solo aristas causales dirigidas)
        adj: Dict[str, List[str]] = {n.node_id: [] for n in nodes}
        for edge in edges:
            if edge.relation.value in _CAUSAL_DIRECTED:
                if edge.source_id in adj:
                    adj[edge.source_id].append(edge.target_id)

        # DFS con colores: 0=white, 1=gray(en pila), 2=black(terminado)
        color: Dict[str, int] = {n.node_id: 0 for n in nodes}
        cycle_nodes: Set[str] = set()
        path: List[str] = []

        def dfs(node_id: str) -> bool:
            """Retorna True si encuentra un ciclo desde node_id."""
            color[node_id] = 1
            path.append(node_id)
            for neighbor in adj.get(node_id, []):
                if neighbor not in color:
                    continue
                if color[neighbor] == 1:
                    # Ciclo encontrado: todos los nodos desde neighbor hasta node_id
                    if neighbor not in path:
                        continue
                    cycle_start = path.index(neighbor)
                    for cnode in path[cycle_start:]:
                        cycle_nodes.add(cnode)
                    return True
                if color[neighbor] == 0:
                    if dfs(neighbor):
                        # Si encontramos ciclo más arriba, marcamos el camino actual también
                        pass
            path.pop()
            color[node_id] = 2
            return False

        for node in nodes:
            if color[node.node_id] == 0:
                dfs(node.node_id)

        # Convertir IDs a índices
        result: Set[int] = set()
        for nid in cycle_nodes:
            if nid in node_id_to_idx:
                result.add(node_id_to_idx[nid])
        return result

    def _find_weak_evidence(
        self,
        nodes:          list,
        edges:          list,
        node_id_to_idx: Dict[str, int],
    ) -> Set[int]:
        """
        Nodos HYPOTHESIS sin ninguna arista SUPPORTS entrante.
        Una hipótesis sin evidencia es una debilidad.
        """
        supported: Set[str] = set()
        for edge in edges:
            if edge.relation == CausalRelation.SUPPORTS:
                supported.add(edge.target_id)

        result: Set[int] = set()
        for i, node in enumerate(nodes):
            if node.node_type == NodeType.HYPOTHESIS:
                if node.node_id not in supported:
                    result.add(i)
        return result
