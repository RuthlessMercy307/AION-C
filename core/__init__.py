"""
AION-C core — tipos y dataclasses compartidos por todos los módulos de CORA.
"""

from .graph import (
    CausalEdge,
    CausalGraph,
    CausalNode,
    CausalRelation,
    NodeType,
    CAUSAL_RELATIONS,
    CAUSAL_RELATIONS_LIST,
    CONTRADICTION_PAIRS,
    INHIBITORY_RELATIONS,
    NODE_TYPES,
    POSITIVE_RELATIONS,
    STRUCTURAL_RELATIONS,
    SYMMETRIC_RELATIONS,
    TEMPORAL_RELATIONS,
)

__all__ = [
    "CausalEdge",
    "CausalGraph",
    "CausalNode",
    "CausalRelation",
    "NodeType",
    "CAUSAL_RELATIONS",
    "CAUSAL_RELATIONS_LIST",
    "CONTRADICTION_PAIRS",
    "INHIBITORY_RELATIONS",
    "NODE_TYPES",
    "POSITIVE_RELATIONS",
    "STRUCTURAL_RELATIONS",
    "SYMMETRIC_RELATIONS",
    "TEMPORAL_RELATIONS",
]
