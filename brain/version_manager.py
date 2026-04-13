"""
brain/version_manager.py — Versionado del cerebro (Parte 9.5 del MEGA-PROMPT)
==============================================================================

Cada consolidación de aprendizaje crea una versión del modelo:
  brain/v1/   — estado inicial post-training
  brain/v2/   — después de aprender JavaScript
  brain/v3/   — después de aprender SQLite
  ...

API:
    bvm = BrainVersionManager(root_dir="brain/")
    v1 = bvm.save_version(state_dict, metadata={"learned": "javascript", "score": 0.94})
    versions = bvm.list_versions()       # [BrainVersion(...), ...]
    state = bvm.load_version(v1.id)
    diff = bvm.compare("v1", "v2")
    bvm.rollback("v1")                   # devuelve el state_dict de v1

Cada versión vive en `<root>/<id>/` con:
    weights.pt    — torch.save del state_dict
    metadata.json — info de la versión (timestamp, parent, scores, notes)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


WEIGHTS_FILENAME  = "weights.pt"
METADATA_FILENAME = "metadata.json"


@dataclass
class BrainVersion:
    """Una versión del cerebro."""
    id:          str                              # "v1", "v2", ...
    parent_id:   Optional[str] = None             # versión de la que se derivó
    timestamp:   float = field(default_factory=time.time)
    notes:       str = ""
    metrics:     Dict[str, float] = field(default_factory=dict)
    metadata:    Dict[str, Any]   = field(default_factory=dict)
    path:        Optional[str]    = None          # ruta al directorio en disco

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":        self.id,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp,
            "notes":     self.notes,
            "metrics":   dict(self.metrics),
            "metadata":  dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrainVersion":
        return cls(
            id=data["id"],
            parent_id=data.get("parent_id"),
            timestamp=data.get("timestamp", time.time()),
            notes=data.get("notes", ""),
            metrics=dict(data.get("metrics", {})),
            metadata=dict(data.get("metadata", {})),
        )


class BrainVersionManager:
    """
    Gestor de versiones del modelo. Cada `save_version` crea una nueva
    carpeta con id incremental (v1, v2, ...).

    Args:
        root_dir: directorio raíz donde viven todas las versiones
    """

    def __init__(self, root_dir) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── id incremental ─────────────────────────────────────────────────

    def _next_id(self) -> str:
        existing = self._all_ids()
        nums = []
        for vid in existing:
            if vid.startswith("v"):
                try:
                    nums.append(int(vid[1:]))
                except ValueError:
                    continue
        next_n = (max(nums) + 1) if nums else 1
        return f"v{next_n}"

    def _all_ids(self) -> List[str]:
        if not self.root.exists():
            return []
        out = []
        for p in self.root.iterdir():
            if p.is_dir() and (p / METADATA_FILENAME).exists():
                out.append(p.name)
        return sorted(out, key=lambda x: int(x[1:]) if x.startswith("v") and x[1:].isdigit() else 0)

    # ── persistencia ───────────────────────────────────────────────────

    def save_version(
        self,
        state_dict: Dict[str, torch.Tensor],
        notes: str = "",
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> BrainVersion:
        """
        Guarda un nuevo snapshot. Devuelve el BrainVersion creado.
        Si parent_id es None y existe versión anterior, se enlaza con la última.
        """
        if version_id is None:
            version_id = self._next_id()
        if parent_id is None:
            existing = self._all_ids()
            parent_id = existing[-1] if existing else None

        vdir = self.root / version_id
        vdir.mkdir(parents=True, exist_ok=True)

        torch.save(state_dict, str(vdir / WEIGHTS_FILENAME))

        version = BrainVersion(
            id=version_id,
            parent_id=parent_id,
            timestamp=time.time(),
            notes=notes,
            metrics=dict(metrics or {}),
            metadata=dict(metadata or {}),
            path=str(vdir),
        )
        with open(vdir / METADATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump(version.to_dict(), f, ensure_ascii=False, indent=2)
        return version

    def load_version(self, version_id: str) -> Dict[str, torch.Tensor]:
        """Devuelve el state_dict de la versión."""
        vdir = self.root / version_id
        wpath = vdir / WEIGHTS_FILENAME
        if not wpath.exists():
            raise FileNotFoundError(f"version not found: {version_id}")
        return torch.load(str(wpath), map_location="cpu", weights_only=False)

    def get_metadata(self, version_id: str) -> BrainVersion:
        vdir = self.root / version_id
        mpath = vdir / METADATA_FILENAME
        if not mpath.exists():
            raise FileNotFoundError(f"version metadata not found: {version_id}")
        with open(mpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = BrainVersion.from_dict(data)
        v.path = str(vdir)
        return v

    def list_versions(self) -> List[BrainVersion]:
        """Devuelve la lista de versiones (ordenadas por id ascendente)."""
        return [self.get_metadata(vid) for vid in self._all_ids()]

    def latest(self) -> Optional[BrainVersion]:
        ids = self._all_ids()
        return self.get_metadata(ids[-1]) if ids else None

    def exists(self, version_id: str) -> bool:
        return version_id in self._all_ids()

    # ── comparación / rollback ─────────────────────────────────────────

    def compare(self, id_a: str, id_b: str) -> Dict[str, Any]:
        """
        Compara metadata y métricas entre dos versiones.
        Devuelve un dict con las diferencias.
        """
        a = self.get_metadata(id_a)
        b = self.get_metadata(id_b)
        all_keys = set(a.metrics) | set(b.metrics)
        metric_diff = {}
        for k in all_keys:
            va = a.metrics.get(k)
            vb = b.metrics.get(k)
            metric_diff[k] = {
                "a":     va,
                "b":     vb,
                "delta": (vb - va) if (va is not None and vb is not None) else None,
            }
        return {
            "id_a":         id_a,
            "id_b":         id_b,
            "parent_a":     a.parent_id,
            "parent_b":     b.parent_id,
            "delta_seconds": b.timestamp - a.timestamp,
            "metric_diff":  metric_diff,
        }

    def rollback(self, version_id: str) -> Dict[str, torch.Tensor]:
        """
        Carga el state_dict de `version_id` para que el caller lo aplique
        al modelo actual. (Esta clase no toca un modelo vivo — devuelve los
        pesos para que el caller llame model.load_state_dict.)
        """
        return self.load_version(version_id)

    def delete_version(self, version_id: str) -> bool:
        """Borra una versión del disco. Devuelve True si existía."""
        vdir = self.root / version_id
        if not vdir.exists():
            return False
        for f in vdir.iterdir():
            try:
                f.unlink()
            except OSError:
                pass
        try:
            vdir.rmdir()
        except OSError:
            return False
        return True


__all__ = [
    "BrainVersion",
    "BrainVersionManager",
    "WEIGHTS_FILENAME",
    "METADATA_FILENAME",
]
