"""
sparse/ — Activación esparsa (Parte 27 del MEGA-PROMPT).

Conditional computation dentro de cada motor. Pequeñas gating networks
(~1% del motor) producen máscaras de activación por capa para que sólo
10-20% de los pesos estén activos por inferencia.

Compatible con los adapters LoRA de Parte 22: la máscara se aplica
DESPUÉS de sumar el delta del adapter.
"""

from sparse.gating import (
    SparseConfig,
    GateNetwork,
    SparseLinear,
    SparsityTracker,
    attach_sparse_gates,
    detach_sparse_gates,
    sparsity_loss,
)

__all__ = [
    "SparseConfig",
    "GateNetwork",
    "SparseLinear",
    "SparsityTracker",
    "attach_sparse_gates",
    "detach_sparse_gates",
    "sparsity_loss",
]
