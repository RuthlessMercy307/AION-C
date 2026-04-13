"""
growth/registry.py — Registro persistente de adapters AION-C (Parte 22.1).

Layout en disco:
    <root>/adapters/<motor>/<concept>/adapter.pt     ← adapter_state_dict (lora_A/B)
    <root>/adapters/<motor>/<concept>/meta.json      ← metadata

Donde root típicamente es "brain/" (alineado con BrainVersionManager).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from growth.adapters import AdapterPack, LoRAConfig


# ════════════════════════════════════════════════════════════════════════════
# Metadata
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AdapterMeta:
    concept_name: str
    motor_name: str
    rank: int
    alpha: float
    dropout: float
    target_paths: List[str]
    num_params: int
    size_bytes: int
    created_at: float                # unix ts
    parent_brain_version: Optional[str] = None
    reward_score: float = 0.0        # actualizado por Parte 25
    usage_count: int = 0             # accesos acumulados (Parte 24 señal 1)
    last_used_at: float = 0.0        # última vez invocado (Parte 24 señal 2)
    retrieval_cost: float = 0.0      # costo de cargarlo (Parte 24 señal 4)
    tags: List[str] = field(default_factory=list)

    def to_json(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Dict) -> "AdapterMeta":
        return cls(**d)


# ════════════════════════════════════════════════════════════════════════════
# AdapterRegistry
# ════════════════════════════════════════════════════════════════════════════

class AdapterRegistry:
    """Gestor del directorio de adapters persistidos.

    Operaciones O(archivos_en_disco) — diseñado para cientos/miles de adapters,
    no millones. Si crece demasiado, la Parte 24 (pruning) se encarga.
    """

    ADAPTER_FILE = "adapter.pt"
    META_FILE = "meta.json"

    def __init__(self, root_dir: str | Path) -> None:
        self.root = Path(root_dir)
        self.adapters_root = self.root / "adapters"
        self.adapters_root.mkdir(parents=True, exist_ok=True)

    # ── Save ──────────────────────────────────────────────────────────────
    def save(
        self,
        pack: AdapterPack,
        parent_brain_version: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> AdapterMeta:
        concept_dir = self._concept_dir(pack.motor_name, pack.concept_name)
        concept_dir.mkdir(parents=True, exist_ok=True)

        # Weights
        state = pack.adapter_state_dict()
        torch.save(state, concept_dir / self.ADAPTER_FILE)

        # Meta
        meta = AdapterMeta(
            concept_name=pack.concept_name,
            motor_name=pack.motor_name,
            rank=pack.config.rank,
            alpha=pack.config.alpha,
            dropout=pack.config.dropout,
            target_paths=pack.target_paths(),
            num_params=pack.num_adapter_parameters(),
            size_bytes=pack.size_bytes(),
            created_at=time.time(),
            parent_brain_version=parent_brain_version,
            tags=list(tags or []),
        )
        (concept_dir / self.META_FILE).write_text(
            json.dumps(meta.to_json(), indent=2), encoding="utf-8"
        )
        return meta

    # ── Load ──────────────────────────────────────────────────────────────
    def load_into(self, pack: AdapterPack) -> AdapterMeta:
        """Carga los pesos persistidos dentro de un AdapterPack ya construido.

        El pack debe haberse construido con los mismos target_paths que están
        almacenados. Se verifica esto antes de copiar.
        """
        concept_dir = self._concept_dir(pack.motor_name, pack.concept_name)
        if not concept_dir.exists():
            raise FileNotFoundError(f"No adapter at {concept_dir}")

        meta = self.get_meta(pack.motor_name, pack.concept_name)
        if set(meta.target_paths) != set(pack.target_paths()):
            raise ValueError(
                f"Pack target_paths {pack.target_paths()} do not match "
                f"stored {meta.target_paths} for adapter "
                f"{pack.motor_name}/{pack.concept_name}"
            )

        state = torch.load(
            concept_dir / self.ADAPTER_FILE, map_location="cpu", weights_only=True
        )
        pack.load_adapter_state_dict(state)
        return meta

    def get_meta(self, motor_name: str, concept_name: str) -> AdapterMeta:
        concept_dir = self._concept_dir(motor_name, concept_name)
        data = json.loads((concept_dir / self.META_FILE).read_text(encoding="utf-8"))
        return AdapterMeta.from_json(data)

    def update_meta(self, meta: AdapterMeta) -> None:
        concept_dir = self._concept_dir(meta.motor_name, meta.concept_name)
        (concept_dir / self.META_FILE).write_text(
            json.dumps(meta.to_json(), indent=2), encoding="utf-8"
        )

    # ── List / search ─────────────────────────────────────────────────────
    def list(self, motor_name: Optional[str] = None) -> List[AdapterMeta]:
        """Lista todos los adapters (opcionalmente filtrados por motor).

        Ordenados por created_at asc para mostrar en UI.
        """
        metas: List[AdapterMeta] = []
        if motor_name is None:
            motors = [p.name for p in self.adapters_root.iterdir() if p.is_dir()]
        else:
            motors = [motor_name]

        for m in motors:
            motor_dir = self.adapters_root / m
            if not motor_dir.exists():
                continue
            for concept_dir in sorted(motor_dir.iterdir()):
                if not concept_dir.is_dir():
                    continue
                meta_path = concept_dir / self.META_FILE
                if not meta_path.exists():
                    continue
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                    metas.append(AdapterMeta.from_json(data))
                except Exception:
                    continue
        metas.sort(key=lambda m: m.created_at)
        return metas

    def exists(self, motor_name: str, concept_name: str) -> bool:
        return (self._concept_dir(motor_name, concept_name) / self.META_FILE).exists()

    def delete(self, motor_name: str, concept_name: str) -> bool:
        concept_dir = self._concept_dir(motor_name, concept_name)
        if not concept_dir.exists():
            return False
        for f in concept_dir.iterdir():
            f.unlink()
        concept_dir.rmdir()
        return True

    def total_size_bytes(self, motor_name: Optional[str] = None) -> int:
        return sum(m.size_bytes for m in self.list(motor_name=motor_name))

    def count(self, motor_name: Optional[str] = None) -> int:
        return len(self.list(motor_name=motor_name))

    # ── Routing por keyword ───────────────────────────────────────────────
    def route_by_query(
        self,
        query: str,
        motor_name: Optional[str] = None,
        max_hits: int = 3,
    ) -> List[AdapterMeta]:
        """Búsqueda simple por substring/tag. Placeholder hasta routing neural.

        Match si el concept_name aparece en la query (case-insensitive) o si
        algún tag del adapter aparece en la query.
        """
        q = query.lower()
        hits: List[AdapterMeta] = []
        for meta in self.list(motor_name=motor_name):
            name_hit = meta.concept_name.lower() in q
            tag_hit = any(t.lower() in q for t in meta.tags)
            if name_hit or tag_hit:
                hits.append(meta)
        hits.sort(key=lambda m: (m.reward_score, m.usage_count), reverse=True)
        return hits[:max_hits]

    # ── Internals ─────────────────────────────────────────────────────────
    def _concept_dir(self, motor_name: str, concept_name: str) -> Path:
        safe_motor = _safe(motor_name)
        safe_concept = _safe(concept_name)
        return self.adapters_root / safe_motor / safe_concept


def _safe(name: str) -> str:
    # No permitimos separadores de path ni caracteres problemáticos.
    bad = '/\\:*?"<>|'
    out = "".join(("_" if c in bad else c) for c in name).strip()
    if not out:
        raise ValueError("name cannot be empty after sanitization")
    return out
