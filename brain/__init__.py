"""
brain/ — Versionado del cerebro de AION-C
==========================================

BrainVersionManager mantiene snapshots del modelo en versiones discretas
(brain/v1, brain/v2, ...) para soportar rollback y comparación entre
versiones después de aprendizajes.
"""

from .version_manager import BrainVersionManager, BrainVersion

__all__ = ["BrainVersionManager", "BrainVersion"]
