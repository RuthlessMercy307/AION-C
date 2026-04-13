"""
synth/dataset_unifier.py — Convierte datasets viejos+nuevos al formato canónico
=================================================================================

Pipeline:
  1. Lee dataset_50k.jsonl (legacy) → canonicaliza con canonicalize_legacy()
  2. Genera los 12.5K nuevos (5K conv + 3K tools + 2K skill + 2K mem + 500 id)
  3. Mezcla, garantiza EOS en 100%, aplica balance 50/50 SKILL/MEM vs sin
  4. Escribe dataset_canonical.jsonl con un record por línea
  5. Devuelve estadísticas de diversidad

Reglas de balance (Parte 15):
  - 50% de ejemplos con SKILL o MEM inyectado, 50% sin

El unifier es testeable: las funciones individuales toman iterables
en memoria para que los tests no necesiten escribir archivos grandes.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .canonical_format import (
    CanonicalRecord, canonicalize_legacy, has_eos, build_record,
)


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DiversityStats:
    """Estadísticas de diversidad de un dataset combinado."""
    total:           int = 0
    by_type:         Dict[str, int] = field(default_factory=dict)
    by_domain:       Dict[str, int] = field(default_factory=dict)
    by_language:     Dict[str, int] = field(default_factory=dict)
    with_skill:      int = 0
    with_mem:        int = 0
    with_tool:       int = 0
    multi_turn:      int = 0
    eos_count:       int = 0
    eos_missing:     int = 0
    skill_or_mem_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total":            self.total,
            "by_type":          dict(self.by_type),
            "by_domain":        dict(self.by_domain),
            "by_language":      dict(self.by_language),
            "with_skill":       self.with_skill,
            "with_mem":         self.with_mem,
            "with_tool":        self.with_tool,
            "multi_turn":       self.multi_turn,
            "eos_count":        self.eos_count,
            "eos_missing":      self.eos_missing,
            "skill_or_mem_pct": round(self.skill_or_mem_pct, 4),
        }


def compute_diversity(records: Iterable[CanonicalRecord]) -> DiversityStats:
    """Recorre los records una vez y produce estadísticas completas."""
    stats = DiversityStats()
    for r in records:
        stats.total += 1
        stats.by_type[r.type] = stats.by_type.get(r.type, 0) + 1
        stats.by_domain[r.domain] = stats.by_domain.get(r.domain, 0) + 1
        stats.by_language[r.language] = stats.by_language.get(r.language, 0) + 1
        if r.has_skill:
            stats.with_skill += 1
        if r.has_mem:
            stats.with_mem += 1
        if r.has_tool:
            stats.with_tool += 1
        if r.is_multi_turn:
            stats.multi_turn += 1
        if has_eos(r.text):
            stats.eos_count += 1
        else:
            stats.eos_missing += 1
    if stats.total > 0:
        with_skill_or_mem = stats.with_skill + stats.with_mem - max(0, stats.with_skill + stats.with_mem - stats.total)
        # Approximation: count records that have AT LEAST one of the two
        # (a record can have both, but we count it once). To be exact we'd
        # need to store the OR per record — recompute below if needed.
        stats.skill_or_mem_pct = with_skill_or_mem / stats.total
    return stats


def compute_diversity_exact(records: List[CanonicalRecord]) -> DiversityStats:
    """Variante que calcula skill_or_mem_pct exacto (necesita una lista)."""
    stats = compute_diversity(records)
    if stats.total > 0:
        with_either = sum(1 for r in records if r.has_skill or r.has_mem)
        stats.skill_or_mem_pct = with_either / stats.total
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────


def read_legacy_jsonl(path) -> Iterator[Dict[str, Any]]:
    """Lee un .jsonl legacy y yield-ea cada record dict."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_canonical_jsonl(records: Iterable[CanonicalRecord], path) -> int:
    """Escribe records al disco como .jsonl. Devuelve el conteo escrito."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_canonical_jsonl(path) -> Iterator[CanonicalRecord]:
    """Lee un .jsonl canónico y yield-ea CanonicalRecords."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield CanonicalRecord.from_dict(json.loads(line))
            except json.JSONDecodeError:
                continue


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────


def canonicalize_legacy_dataset(legacy_records: Iterable[Dict[str, Any]]) -> Iterator[CanonicalRecord]:
    """Convierte un iterable de records legacy a CanonicalRecord."""
    for rec in legacy_records:
        yield canonicalize_legacy(rec)


def merge_and_shuffle(
    iterables: List[Iterable[CanonicalRecord]],
    seed: int = 42,
) -> List[CanonicalRecord]:
    """Acumula todos los iterables, mezcla con seed y devuelve la lista."""
    rng = random.Random(seed)
    all_records: List[CanonicalRecord] = []
    for it in iterables:
        all_records.extend(it)
    rng.shuffle(all_records)
    return all_records


def verify_eos_all(records: Iterable[CanonicalRecord]) -> int:
    """Devuelve el número de records que NO tienen EOS. 0 = todo OK."""
    return sum(1 for r in records if not has_eos(r.text))


def fix_eos(records: Iterable[CanonicalRecord]) -> Iterator[CanonicalRecord]:
    """Garantiza EOS al final de cada record. Yield-ea records arreglados."""
    from .canonical_format import EOS_MARKER
    for r in records:
        if not has_eos(r.text):
            r.text = r.text.rstrip() + "\n" + EOS_MARKER
        yield r


__all__ = [
    "DiversityStats",
    "compute_diversity",
    "compute_diversity_exact",
    "read_legacy_jsonl",
    "write_canonical_jsonl",
    "read_canonical_jsonl",
    "canonicalize_legacy_dataset",
    "merge_and_shuffle",
    "verify_eos_all",
    "fix_eos",
]
