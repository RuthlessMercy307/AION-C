"""
growth/ — Crecimiento dinámico del cerebro AION-C (Parte 22 del MEGA-PROMPT).

Tres mecanismos de crecimiento:
    22.1 Adapters (LoRA-style)     → growth.adapters
    22.2 Expansión de motor        → growth.expansion    (futuro)
    22.3 Sub-motores               → growth.sub_motor    (futuro)
    22.4 Reglas de decisión        → growth.policy

Registro persistente de adapters por motor:
    brain/adapters/<motor>/<concept>/adapter.pt  + meta.json
"""

from growth.adapters import (
    LoRAConfig,
    LoRALinear,
    AdapterPack,
    attach_adapter_pack,
    detach_adapter_pack,
    build_adapter_pack,
    auto_target_paths,
    freeze_base_parameters,
)
from growth.registry import AdapterMeta, AdapterRegistry
from growth.policy import GrowthDecision, decide_growth, GrowthPolicy

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "AdapterPack",
    "attach_adapter_pack",
    "detach_adapter_pack",
    "build_adapter_pack",
    "auto_target_paths",
    "freeze_base_parameters",
    "AdapterMeta",
    "AdapterRegistry",
    "GrowthDecision",
    "decide_growth",
    "GrowthPolicy",
]
