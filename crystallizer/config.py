"""
crystallizer/config.py — Configuración del GraphCrystallizer
=============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CrystallizerConfig:
    """
    Hiperparámetros del GraphCrystallizer.

    Diseñado para coincidir con el StreamEncoder upstream:
        hidden_dim = StreamEncoderConfig.concept_dim = 128  (o 256 en tiny)
        n_relation_types = len(CAUSAL_RELATIONS) = 16
        n_node_types     = len(NodeType)         = 7

    Configuración tiny para testing:
        hidden_dim=256, max_nodes=32

    Umbrales:
        node_threshold  — sigmoid score mínimo para considerar un nodo
        edge_threshold  — sigmoid score mínimo para agregar una arista
        Valores bajos (0.3) garantizan grafos no vacíos en tests con pesos random.
    """

    # Dimensiones
    hidden_dim: int   = 256   # = concept_dim del StreamEncoder
    max_nodes: int    = 32    # máximo de nodos por grafo output

    # Vocabularios (deben coincidir con core/graph.py)
    n_relation_types: int = 16   # len(CAUSAL_RELATIONS)
    n_node_types:     int = 7    # len(NodeType)

    # Umbrales de detección
    node_threshold: float = 0.3  # probabilidad mínima para ser nodo
    edge_threshold: float = 0.3  # sigmoid score mínimo para ser arista

    # Arquitectura del pooler
    pooler_heads: int = 4        # cabezas en CrossAttentionPooler

    # Arquitectura del relation scorer
    relation_hidden_dim: int = 64  # dimensión por relación en AsymmetricRelationScorer

    # Arquitectura interna del NodeDetector
    node_confidence_hidden_dim: int = 64

    def __post_init__(self) -> None:
        if self.hidden_dim % self.pooler_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"pooler_heads ({self.pooler_heads})"
            )
        if not 0.0 < self.node_threshold < 1.0:
            raise ValueError(f"node_threshold must be in (0,1), got {self.node_threshold}")
        if not 0.0 < self.edge_threshold < 1.0:
            raise ValueError(f"edge_threshold must be in (0,1), got {self.edge_threshold}")
