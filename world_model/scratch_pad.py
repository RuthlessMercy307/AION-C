"""
world_model/scratch_pad.py — ScratchPad y schemas por motor (Parte 19.1)
=========================================================================

El scratch pad tiene 16 slots. Cada motor define qué significa cada slot
mediante un ScratchPadSchema con SlotSpec por slot ocupado.

Ejemplo (FORGE-C):
  slot 0: variables actuales {"x": 5, "y": null}
  slot 1: stack de llamadas ["main", "process"]
  slot 2: output esperado "hola mundo"
  slot 3: error detectado "null pointer en línea 3"

Los slots no usados quedan vacíos (None). El verifier exige que los
slots marcados `required=True` estén presentes y con el tipo correcto.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


SLOT_COUNT = 16


@dataclass
class SlotSpec:
    """Definición semántica de un slot dentro de un schema."""
    index:    int
    name:     str
    expected_type: type             # int, str, float, list, dict, bool
    required: bool = False
    description: str = ""


@dataclass
class ScratchPadSchema:
    """Esquema de slots para un motor concreto."""
    motor:    str
    slots:    List[SlotSpec]

    def slot_by_index(self, index: int) -> Optional[SlotSpec]:
        for s in self.slots:
            if s.index == index:
                return s
        return None

    def slot_by_name(self, name: str) -> Optional[SlotSpec]:
        for s in self.slots:
            if s.name == name:
                return s
        return None

    def required_indices(self) -> List[int]:
        return [s.index for s in self.slots if s.required]


# ─────────────────────────────────────────────────────────────────────────────
# Schemas por motor (los que define la Parte 19.1)
# ─────────────────────────────────────────────────────────────────────────────


FORGE_C_SCHEMA = ScratchPadSchema(
    motor="forge_c",
    slots=[
        SlotSpec(0, "variables",   dict, required=True,  description="estado de variables"),
        SlotSpec(1, "call_stack",  list, required=True,  description="stack de llamadas"),
        SlotSpec(2, "expected_output", str,  required=False, description="output esperado"),
        SlotSpec(3, "error",       str,  required=False, description="error detectado"),
    ],
)

AXIOM_SCHEMA = ScratchPadSchema(
    motor="axiom",
    slots=[
        SlotSpec(0, "proven",      list, required=True,  description="proposiciones demostradas"),
        SlotSpec(1, "to_prove",    list, required=True,  description="lo que falta por demostrar"),
        SlotSpec(2, "hypotheses",  list, required=False, description="hipótesis activas"),
        SlotSpec(3, "contradiction", str, required=False, description="contradicción encontrada"),
    ],
)

CORA_SCHEMA = ScratchPadSchema(
    motor="cora",
    slots=[
        SlotSpec(0, "causes",      list, required=True,  description="causas iniciales"),
        SlotSpec(1, "direct_effects",   list, required=True,  description="efectos directos"),
        SlotSpec(2, "indirect_effects", list, required=False, description="efectos indirectos"),
        SlotSpec(3, "prediction",  str,  required=True,  description="predicción final"),
    ],
)

MUSE_SCHEMA = ScratchPadSchema(
    motor="muse",
    slots=[
        SlotSpec(0, "tension",     float, required=True,  description="tensión actual [0,1]"),
        SlotSpec(1, "conflicts",   list,  required=True,  description="conflictos abiertos"),
        SlotSpec(2, "expectation", str,   required=False, description="expectativa del lector"),
        SlotSpec(3, "subversion",  str,   required=False, description="cómo se subvierte la expectativa"),
    ],
)

EMPATHY_SCHEMA = ScratchPadSchema(
    motor="empathy",
    slots=[
        SlotSpec(0, "emotion",        str, required=True,  description="emoción detectada"),
        SlotSpec(1, "probable_cause", str, required=True,  description="causa probable"),
        SlotSpec(2, "need",           str, required=True,  description="necesidad del usuario"),
        SlotSpec(3, "response_strategy", str, required=True, description="estrategia de respuesta"),
    ],
)


SCHEMAS_BY_MOTOR: Dict[str, ScratchPadSchema] = {
    "forge_c": FORGE_C_SCHEMA,
    "axiom":   AXIOM_SCHEMA,
    "cora":    CORA_SCHEMA,
    "muse":    MUSE_SCHEMA,
    "empathy": EMPATHY_SCHEMA,
}


# ─────────────────────────────────────────────────────────────────────────────
# ScratchPad
# ─────────────────────────────────────────────────────────────────────────────


class ScratchPad:
    """
    16 slots de estado estructurado.

    Args:
        schema: schema del motor (define qué significa cada slot ocupado)
        size:   número de slots totales (default 16)
    """

    def __init__(self, schema: Optional[ScratchPadSchema] = None, size: int = SLOT_COUNT) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self.schema = schema
        self._slots: List[Any] = [None] * size

    # ── acceso a slots ─────────────────────────────────────────────────

    def get(self, index: int) -> Any:
        if not 0 <= index < self.size:
            raise IndexError(f"slot index out of range: {index}")
        return self._slots[index]

    def set(self, index: int, value: Any) -> None:
        if not 0 <= index < self.size:
            raise IndexError(f"slot index out of range: {index}")
        self._slots[index] = value

    def set_by_name(self, name: str, value: Any) -> None:
        if self.schema is None:
            raise ValueError("no schema attached; use set(index, ...) instead")
        spec = self.schema.slot_by_name(name)
        if spec is None:
            raise KeyError(f"slot '{name}' not in schema for motor {self.schema.motor}")
        self.set(spec.index, value)

    def get_by_name(self, name: str) -> Any:
        if self.schema is None:
            raise ValueError("no schema attached; use get(index) instead")
        spec = self.schema.slot_by_name(name)
        if spec is None:
            raise KeyError(f"slot '{name}' not in schema for motor {self.schema.motor}")
        return self.get(spec.index)

    def clear(self) -> None:
        self._slots = [None] * self.size

    def is_empty(self) -> bool:
        return all(v is None for v in self._slots)

    def filled_indices(self) -> List[int]:
        return [i for i, v in enumerate(self._slots) if v is not None]

    # ── serialización ──────────────────────────────────────────────────

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "motor":  self.schema.motor if self.schema else None,
            "size":   self.size,
            "slots":  {},
        }
        for i, v in enumerate(self._slots):
            if v is None:
                continue
            if self.schema is not None:
                spec = self.schema.slot_by_index(i)
                if spec is not None:
                    out["slots"][spec.name] = v
                    continue
            out["slots"][f"slot_{i}"] = v
        return out

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], schema: Optional[ScratchPadSchema] = None) -> "ScratchPad":
        size = data.get("size", SLOT_COUNT)
        sp = cls(schema=schema, size=size)
        slot_dict = data.get("slots", {})
        if schema is not None:
            for name, value in slot_dict.items():
                spec = schema.slot_by_name(name)
                if spec is not None:
                    sp.set(spec.index, value)
                elif name.startswith("slot_"):
                    try:
                        idx = int(name.split("_", 1)[1])
                        sp.set(idx, value)
                    except (ValueError, IndexError):
                        pass
        else:
            for name, value in slot_dict.items():
                if name.startswith("slot_"):
                    try:
                        idx = int(name.split("_", 1)[1])
                        sp.set(idx, value)
                    except (ValueError, IndexError):
                        pass
        return sp

    def copy(self) -> "ScratchPad":
        sp = ScratchPad(schema=self.schema, size=self.size)
        sp._slots = list(self._slots)
        return sp

    def __len__(self) -> int:
        return sum(1 for v in self._slots if v is not None)

    def __repr__(self) -> str:
        motor = self.schema.motor if self.schema else "?"
        return f"ScratchPad(motor={motor}, filled={len(self)}/{self.size})"


__all__ = [
    "SLOT_COUNT",
    "SlotSpec",
    "ScratchPadSchema",
    "ScratchPad",
    "FORGE_C_SCHEMA",
    "AXIOM_SCHEMA",
    "CORA_SCHEMA",
    "MUSE_SCHEMA",
    "EMPATHY_SCHEMA",
    "SCHEMAS_BY_MOTOR",
]
