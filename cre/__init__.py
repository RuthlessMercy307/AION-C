"""
AION-C cre — Causal Reasoning Engine.

Motor de razonamiento iterativo basado en typed message passing con weight sharing.
CausalGraph + node features → node features refinados.

Componentes adicionales (parada adaptativa):
    WeaknessDetector  — detecta debilidades en el grafo para guiar refinamiento
    ConvergenceGate   — decide cuándo parar de iterar (activo con use_convergence_gate=True)

Componentes opcionales (capacidad):
    SparseMoE / ExpertGroup — especialización post-MP (activo con use_moe=True)
"""

from .aggregator import AttentiveAggregator
from .auto_scale import AutoScaler, AutoScaleResult
from .batching import BatchedGraph, PyGStyleBatcher
from .config import CREConfig
from .convergence import ConvergenceDecision, ConvergenceGate
from .engine import CREOutput, CausalReasoningEngine
from .message_passing import CausalMessagePassingLayer
from .moe import ExpertGroup, MoEOutput, SparseMoE
from .scratch_pad import DifferentiableScratchPad, ScratchPadConfig
from .weakness import (
    WEAKNESS_TYPES,
    Weakness,
    WeaknessDetector,
    WeaknessReport,
)

__all__ = [
    "AttentiveAggregator",
    "AutoScaleResult",
    "AutoScaler",
    "BatchedGraph",
    "CREConfig",
    "CREOutput",
    "CausalMessagePassingLayer",
    "CausalReasoningEngine",
    "ConvergenceDecision",
    "ConvergenceGate",
    "DifferentiableScratchPad",
    "ExpertGroup",
    "MoEOutput",
    "PyGStyleBatcher",
    "ScratchPadConfig",
    "SparseMoE",
    "WEAKNESS_TYPES",
    "Weakness",
    "WeaknessDetector",
    "WeaknessReport",
]
