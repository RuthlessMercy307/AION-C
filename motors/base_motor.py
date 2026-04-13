"""
motors/base_motor.py — BaseMotor: interfaz abstracta para todos los motores AION-C
====================================================================================

En la arquitectura MoSE (Mixture of Specialized Engines), cada motor especializado
implementa esta interfaz. El Orchestrator puede invocar cualquier motor de forma
uniforme sin conocer sus detalles internos.

INTERFAZ:
    define_node_types()            — tipos de nodo que maneja este motor
    define_relations()             — relaciones causales que usa este motor
    build_graph(concepts)          — concept vectors → CausalGraph
    reason(graph, node_features,   — refina el grafo con el CRE del motor
           n_iterations)
    get_graph_repr(cre_output,     — extrae tensor [k_nodes, D] para el decoder
                   k_nodes)

TODOS los motores (CORA, TemporalMotor, CounterfactualMotor, etc.)
heredan de BaseMotor y deben implementar estos cinco métodos.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn


class BaseMotor(nn.Module, ABC):
    """
    Clase base abstracta para todos los motores especializados de AION-C.

    Hereda de nn.Module para que los motores sean módulos PyTorch entrenables,
    y de ABC para forzar la implementación de los métodos de interfaz.

    Subclases deben implementar los cinco métodos abstractos. Cualquier motor
    que no implemente todos ellos lanzará TypeError al instanciarse.

    Uso:
        class MyMotor(BaseMotor):
            def define_node_types(self): return ["entity", "event"]
            def define_relations(self): return ["causes", "prevents"]
            def build_graph(self, concepts): ...
            def reason(self, graph, node_features, n_iterations): ...
            def get_graph_repr(self, cre_output, k_nodes): ...
    """

    # ── Métodos de introspección ──────────────────────────────────────────────

    def get_relation_keys(self) -> List[str]:
        """
        Retorna la lista de relaciones usada por el CRE de este motor.

        Delegado a self.cre.relation_keys — la fuente de verdad para el
        vocabulario de relaciones de este motor (CAUSAL_RELATIONS, CODE_RELATIONS,
        MATH_RELATIONS, etc.). Usado por PyGStyleBatcher para construir el mapa
        rel→idx correcto al batching.

        Subclases sin self.cre deben sobreescribir este método.
        """
        return list(self.cre.relation_keys)  # type: ignore[attr-defined]

    @abstractmethod
    def define_node_types(self) -> List[str]:
        """
        Retorna la lista de tipos de nodo que maneja este motor.

        Returns:
            List[str] — por ejemplo ["entity", "event", "state", ...]
        """
        ...

    @abstractmethod
    def define_relations(self) -> List[str]:
        """
        Retorna la lista de tipos de relación causal que usa este motor.

        Returns:
            List[str] — por ejemplo ["causes", "enables", "prevents", ...]
        """
        ...

    # ── Métodos de procesamiento ──────────────────────────────────────────────

    @abstractmethod
    def build_graph(self, concepts: torch.Tensor):
        """
        Convierte concept vectors en un CausalGraph estructurado.

        Args:
            concepts: [B, L, D] — vectores de concepto del StreamEncoder

        Returns:
            Objeto con .graphs (List[CausalGraph]) y tensores diferenciables
            para entrenamiento (node_scores, relation_logits, etc.)
        """
        ...

    @abstractmethod
    def reason(
        self,
        graph,
        node_features: torch.Tensor,
        n_iterations: int = 3,
    ):
        """
        Refina el grafo con el CRE del motor mediante message passing iterativo.

        Args:
            graph:         CausalGraph — estructura discreta construida por build_graph
            node_features: [N, D]      — features iniciales de los nodos
            n_iterations:  int         — iteraciones de refinamiento

        Returns:
            CREOutput con node_features refinadas, edge_features, iterations_run, etc.
        """
        ...

    @abstractmethod
    def get_graph_repr(
        self,
        cre_output,
        k_nodes: int,
    ) -> torch.Tensor:
        """
        Extrae una representación de tamaño fijo del grafo razonado para el decoder.

        Dado que distintos grafos tienen distinto número de nodos, este método
        normaliza la salida a exactamente k_nodes vectores — los nodos más
        relevantes (por norma) completados con ceros si hay menos de k_nodes.

        Args:
            cre_output: CREOutput — salida del motor después de reason()
            k_nodes:    int       — número de nodos a devolver

        Returns:
            [k_nodes, D] — representación de tamaño fijo para el decoder
        """
        ...
