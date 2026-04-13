"""
pruning/ — Pruning con 4 señales (Parte 24 del MEGA-PROMPT).

Señales:
    S1. FRECUENCIA       — número de accesos (suavizado exponencial).
    S2. RECENCIA         — curva de olvido a partir del último acceso.
    S3. UTILIDAD         — reward acumulado del item (Parte 25).
    S4. COSTO_RECUPERAC. — cuán caro fue reconstruir el item (protegido).

Acciones:
    KEEP       — se mantiene como está
    PROMOTE    — sube a la capa rápida (cache calentito)
    COMPRESS   — degrada a nivel superior (Parte 26) antes de borrar
    DELETE     — se descarta de la memoria

El TTL dinámico se recalcula por item en cada ciclo de sueño según
retain_score. Alto score → TTL largo; bajo → TTL corto o borrado.
"""

from pruning.pruner import (
    PruneSignals,
    PruneConfig,
    PruneAction,
    PruneDecision,
    PruneReport,
    MemoryPruner,
    sleep_prune_hook,
)

__all__ = [
    "PruneSignals",
    "PruneConfig",
    "PruneAction",
    "PruneDecision",
    "PruneReport",
    "MemoryPruner",
    "sleep_prune_hook",
]
