"""
AION-C router — CORAPipeline end-to-end.

Conecta SE → GC → CRE → SD en un módulo único diferenciable.
"""

from .pipeline import CORAConfig, CORAPipeline, PipelineOutput

__all__ = [
    "CORAConfig",
    "CORAPipeline",
    "PipelineOutput",
]
