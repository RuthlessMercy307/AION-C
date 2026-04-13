#!/usr/bin/env python3
"""
synth/build_canonical_dataset.py — Pipeline completo de Fase B
================================================================

Lee dataset_50k.jsonl (legacy 57.5K) + genera 12.5K nuevos
+ unifica al formato canónico + verifica EOS + verifica diversidad
+ escribe datasets/dataset_canonical_70k.jsonl.

Uso:
    python -m synth.build_canonical_dataset
    python -m synth.build_canonical_dataset --out datasets/foo.jsonl

Salida:
    datasets/dataset_canonical_70k.jsonl   — record canónico por línea
    datasets/dataset_canonical_70k.stats.json — diversidad
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

# Permite ejecutar como script o como módulo
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from synth.canonical_format import CanonicalRecord
    from synth.conversational_gen import generate_conversational
    from synth.tool_gen import generate_tool_calls
    from synth.skill_injected_gen import generate_skill_injected
    from synth.mem_injected_gen import generate_mem_injected
    from synth.identity_gen import generate_identity
    from synth.dataset_unifier import (
        canonicalize_legacy_dataset, read_legacy_jsonl,
        write_canonical_jsonl, merge_and_shuffle,
        verify_eos_all, fix_eos, compute_diversity_exact,
    )
else:
    from .canonical_format import CanonicalRecord
    from .conversational_gen import generate_conversational
    from .tool_gen import generate_tool_calls
    from .skill_injected_gen import generate_skill_injected
    from .mem_injected_gen import generate_mem_injected
    from .identity_gen import generate_identity
    from .dataset_unifier import (
        canonicalize_legacy_dataset, read_legacy_jsonl,
        write_canonical_jsonl, merge_and_shuffle,
        verify_eos_all, fix_eos, compute_diversity_exact,
    )


REPO = Path(__file__).resolve().parent.parent
DEFAULT_LEGACY = REPO / "datasets" / "dataset_50k.jsonl"
DEFAULT_OUT    = REPO / "datasets" / "dataset_canonical_70k.jsonl"
DEFAULT_STATS  = REPO / "datasets" / "dataset_canonical_70k.stats.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_dataset(
    legacy_path: Path = DEFAULT_LEGACY,
    out_path:    Path = DEFAULT_OUT,
    stats_path:  Path = DEFAULT_STATS,
    seed:        int  = 42,
    n_conv:      int  = 5000,
    n_tool:      int  = 3000,
    n_skill:     int  = 2000,
    n_mem:       int  = 2000,
    n_identity:  int  = 500,
) -> dict:
    """
    Construye el dataset canónico unificado y devuelve el dict de stats.
    """
    log("=" * 60)
    log("FASE B — Construcción del dataset canónico")
    log("=" * 60)

    # 1) Legacy → canonical
    log(f"[1/6] Leyendo legacy {legacy_path.name}...")
    if not legacy_path.exists():
        log(f"  AVISO: {legacy_path} no existe — generando solo los 12.5K nuevos")
        legacy_canonical: List[CanonicalRecord] = []
    else:
        legacy_canonical = list(canonicalize_legacy_dataset(read_legacy_jsonl(legacy_path)))
        log(f"  Legacy canonicalizado: {len(legacy_canonical):,} ejemplos")

    # 2) Generar los 5 batches nuevos
    log(f"[2/6] Generando {n_conv:,} ejemplos conversacionales multi-turn...")
    conv_records = list(generate_conversational(n=n_conv, seed=seed))
    log(f"  Conversacionales: {len(conv_records):,}")

    log(f"[3/6] Generando {n_tool:,} ejemplos con tool calls...")
    tool_records = list(generate_tool_calls(n=n_tool, seed=seed))
    log(f"  Tool calls: {len(tool_records):,}")

    log(f"[4/6] Generando {n_skill:,} ejemplos con skills inyectados...")
    skill_records = list(generate_skill_injected(n=n_skill, seed=seed))
    log(f"  Skills inyectados: {len(skill_records):,}")

    log(f"[5/6] Generando {n_mem:,} ejemplos con MEM inyectada + {n_identity} de identidad...")
    mem_records = list(generate_mem_injected(n=n_mem, seed=seed))
    identity_records = list(generate_identity(n=n_identity, seed=seed))
    log(f"  MEM inyectados: {len(mem_records):,}, Identidad: {len(identity_records):,}")

    # 3) Mezclar
    all_records = merge_and_shuffle(
        [legacy_canonical, conv_records, tool_records, skill_records, mem_records, identity_records],
        seed=seed,
    )
    log(f"  Total combinado: {len(all_records):,}")

    # 4) Garantizar EOS en 100%
    log(f"[6/6] Verificando EOS y escribiendo...")
    missing_before = verify_eos_all(all_records)
    if missing_before > 0:
        log(f"  {missing_before} ejemplos sin EOS — aplicando fix_eos")
        all_records = list(fix_eos(all_records))
    missing_after = verify_eos_all(all_records)
    assert missing_after == 0, f"EOS check failed: {missing_after} still missing"
    log(f"  EOS coverage: 100% ({len(all_records):,} ejemplos)")

    # 5) Diversidad
    stats = compute_diversity_exact(all_records)
    stats_dict = stats.to_dict()
    log(f"  by_type: {stats_dict['by_type']}")
    log(f"  by_domain: {stats_dict['by_domain']}")
    log(f"  by_language: {stats_dict['by_language']}")
    log(f"  with_skill: {stats_dict['with_skill']:,}")
    log(f"  with_mem: {stats_dict['with_mem']:,}")
    log(f"  with_tool: {stats_dict['with_tool']:,}")
    log(f"  multi_turn: {stats_dict['multi_turn']:,}")
    log(f"  skill_or_mem_pct: {stats_dict['skill_or_mem_pct']:.2%}")

    # 6) Escribir
    n_written = write_canonical_jsonl(all_records, out_path)
    log(f"  Escrito: {n_written:,} ejemplos en {out_path}")

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=2)
    log(f"  Stats: {stats_path}")
    log("Done.")
    return stats_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    ap.add_argument("--out",    type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stats",  type=Path, default=DEFAULT_STATS)
    ap.add_argument("--seed",   type=int,  default=42)
    ap.add_argument("--n-conv",     type=int, default=5000)
    ap.add_argument("--n-tool",     type=int, default=3000)
    ap.add_argument("--n-skill",    type=int, default=2000)
    ap.add_argument("--n-mem",      type=int, default=2000)
    ap.add_argument("--n-identity", type=int, default=500)
    args = ap.parse_args()
    build_dataset(
        legacy_path=args.legacy,
        out_path=args.out,
        stats_path=args.stats,
        seed=args.seed,
        n_conv=args.n_conv, n_tool=args.n_tool, n_skill=args.n_skill,
        n_mem=args.n_mem, n_identity=args.n_identity,
    )


if __name__ == "__main__":
    main()
