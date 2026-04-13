"""
AION-C crystallizer — GraphCrystallizer: concept vectors → CausalGraph.

Convierte concept vectors [B, L, D] (del StreamEncoder) en grafos causales
estructurados, usando detección de nodos, pooling por cross-attention
y puntuación de relaciones asimétricas.
"""

from .config import CrystallizerConfig
from .model import CrystallizerOutput, GraphCrystallizer
from .node_detector import NodeDetector
from .pooler import CrossAttentionPooler
from .relation_scorer import AsymmetricRelationScorer

__all__ = [
    "AsymmetricRelationScorer",
    "CrossAttentionPooler",
    "CrystallizerConfig",
    "CrystallizerOutput",
    "GraphCrystallizer",
    "NodeDetector",
]
