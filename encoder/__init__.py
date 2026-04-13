"""
AION-C encoder — StreamEncoder basado en Mamba-style SSM.

Convierte token IDs en concept vectors con complejidad O(L),
evitando la O(L²) de attention. Paso previo al GraphConstructor.
"""

from .mamba_layer import (
    GatedFFN,
    MambaLayer,
    RMSNorm,
    SelectiveSSM,
    StreamEncoderConfig,
)
from .model import StreamEncoder

__all__ = [
    "GatedFFN",
    "MambaLayer",
    "RMSNorm",
    "SelectiveSSM",
    "StreamEncoder",
    "StreamEncoderConfig",
]
