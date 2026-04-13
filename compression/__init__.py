"""
compression/ — Compresión jerárquica en 3 niveles (Parte 26 del MEGA-PROMPT).

Niveles:
    EPISODIC  — interacciones crudas, alta fidelidad, TTL corto.
    STABLE    — clusters de episodios relacionados, resumen + anclas.
    NUCLEAR   — conceptos abstractos estables, candidatos a consolidación
                en pesos.

Flujo:
    crudo(L1) ──cluster──▶ estable(L2) ──abstracción──▶ nuclear(L3)

Nunca se pierde toda la evidencia cruda: cada nivel superior conserva
ejemplos ancla, y un nivel superior puede descomprimirse para re-examinar
la evidencia original.
"""

from compression.hierarchy import (
    MemoryLevel,
    StoredItem,
    Cluster,
    HierarchicalStore,
    Clusterer,
    HierarchicalCompressor,
    jaccard_similarity,
    sleep_compress_hook,
)

__all__ = [
    "MemoryLevel",
    "StoredItem",
    "Cluster",
    "HierarchicalStore",
    "Clusterer",
    "HierarchicalCompressor",
    "jaccard_similarity",
    "sleep_compress_hook",
]
