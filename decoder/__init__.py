"""
AION-C decoder — StreamDecoder.

Decodificador autoregresivo condicionado en el grafo causal del CRE.
CausalGraph → node_features + token_ids → token logits.
"""

from .config import StreamDecoderConfig
from .hybrid_layer import HybridDecoderLayer
from .meta_head import MetaOutput, OutputMetaHead
from .model import DecoderOutput, StreamDecoder

__all__ = [
    "DecoderOutput",
    "HybridDecoderLayer",
    "MetaOutput",
    "OutputMetaHead",
    "StreamDecoder",
    "StreamDecoderConfig",
]
