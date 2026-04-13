"""
synth/canonical_dataloader.py — DataLoader helpers for canonical 70K dataset
==============================================================================

Lee `dataset_canonical_70k.jsonl` y produce muestras tokenizadas con:
  - tokens: List[int] (con BPE EOS=2 anexado al final)
  - domain_id: int (mapeado a motor index)
  - has_skill / has_mem / has_tool: flags

Incluye:
  - load_canonical_records(path)         — lee el .jsonl en memoria
  - balanced_indices(records, ratio)     — produce índices con balance 50/50
                                            entre SKILL/MEM y sin SKILL/MEM
  - weighted_sampler_indices(...)        — wrapper para muestreo continuo
                                            (in-memory, sin torch.utils.data)

Las funciones son testeables sin torch ni modelo.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .canonical_format import CanonicalRecord


# Mapeo dominio → índice de motor (consistente con train_4090.py)
DOMAIN_TO_MOTOR_IDX: Dict[str, int] = {
    "cora":    0,
    "forge_c": 1,
    "muse":    2,
    "axiom":   3,
    "empathy": 4,
    "general": 0,  # general → cora por default
}

MOTOR_NAMES = ["cora", "forge_c", "muse", "axiom", "empathy"]
EOS_TOKEN_ID = 2  # BPE EOS del tokenizer aion_32k


def load_canonical_records(path) -> List[CanonicalRecord]:
    """Lee el .jsonl canónico completo a memoria."""
    p = Path(path)
    out: List[CanonicalRecord] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(CanonicalRecord.from_dict(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return out


def domain_to_motor_idx(domain: str) -> int:
    """Mapea un nombre de dominio a un índice de motor (0-4)."""
    return DOMAIN_TO_MOTOR_IDX.get(domain, 0)


def encode_record(tok, record: CanonicalRecord, max_len: int = 256) -> List[int]:
    """
    Tokeniza el texto canónico de un record y append-ea BPE EOS al final.
    El texto canónico ya termina con la cadena literal '[EOS]', pero el
    BPE EOS token id 2 sirve como señal robusta de stop para el decoder.
    """
    text = record.text
    try:
        ids = tok.encode(text, max_len - 1)
    except TypeError:
        ids = tok.encode(text)[:max_len - 1]
    if not ids or ids[-1] != EOS_TOKEN_ID:
        ids.append(EOS_TOKEN_ID)
    return ids


def balanced_indices(
    records: List[CanonicalRecord],
    target_ratio: float = 0.5,
    seed: int = 42,
) -> List[int]:
    """
    Produce una lista de índices de records que aproxima `target_ratio`
    de records con SKILL o MEM inyectado.

    Si la ratio actual es menor que target, REPITE índices de records
    con skill/mem hasta alcanzarla. Si es mayor, REPITE índices sin.

    El resultado tiene el mismo tamaño que `records`.
    """
    if not records:
        return []
    if not 0.0 <= target_ratio <= 1.0:
        raise ValueError("target_ratio must be in [0, 1]")
    rng = random.Random(seed)
    with_idx    = [i for i, r in enumerate(records) if r.has_skill or r.has_mem]
    without_idx = [i for i, r in enumerate(records) if not (r.has_skill or r.has_mem)]
    n_total = len(records)
    target_with = int(round(n_total * target_ratio))
    target_without = n_total - target_with

    def _take(pool: List[int], n: int) -> List[int]:
        if not pool or n <= 0:
            return []
        if n <= len(pool):
            return rng.sample(pool, n)
        # repetir con choices
        return [rng.choice(pool) for _ in range(n)]

    out = _take(with_idx, target_with) + _take(without_idx, target_without)
    rng.shuffle(out)
    return out


def weighted_sampler_indices(
    records: List[CanonicalRecord],
    n_steps: int,
    target_ratio: float = 0.5,
    seed: int = 42,
) -> List[int]:
    """
    Produce `n_steps` índices, manteniendo el balance 50/50 entre records
    con SKILL/MEM y sin SKILL/MEM independientemente del tamaño.
    """
    if not records:
        return []
    rng = random.Random(seed)
    with_idx    = [i for i, r in enumerate(records) if r.has_skill or r.has_mem]
    without_idx = [i for i, r in enumerate(records) if not (r.has_skill or r.has_mem)]
    out: List[int] = []
    for step in range(n_steps):
        if rng.random() < target_ratio and with_idx:
            out.append(rng.choice(with_idx))
        elif without_idx:
            out.append(rng.choice(without_idx))
        elif with_idx:
            out.append(rng.choice(with_idx))
    return out


@dataclass
class DatasetStats:
    total:        int
    n_with:       int
    n_without:    int
    by_domain:    Dict[str, int] = field(default_factory=dict)
    by_type:      Dict[str, int] = field(default_factory=dict)


def quick_stats(records: List[CanonicalRecord]) -> DatasetStats:
    s = DatasetStats(
        total=len(records),
        n_with=sum(1 for r in records if r.has_skill or r.has_mem),
        n_without=sum(1 for r in records if not (r.has_skill or r.has_mem)),
    )
    for r in records:
        s.by_domain[r.domain] = s.by_domain.get(r.domain, 0) + 1
        s.by_type[r.type] = s.by_type.get(r.type, 0) + 1
    return s


__all__ = [
    "DOMAIN_TO_MOTOR_IDX",
    "MOTOR_NAMES",
    "EOS_TOKEN_ID",
    "load_canonical_records",
    "domain_to_motor_idx",
    "encode_record",
    "balanced_indices",
    "weighted_sampler_indices",
    "quick_stats",
    "DatasetStats",
]
