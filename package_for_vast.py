#!/usr/bin/env python3
"""
package_for_vast.py — Empaqueta TODO para Vast.ai (RTX 4090) en un único zip
==============================================================================

Uso local:
    python package_for_vast.py
    # → produce aion_c_vast.zip (~50-60 MB sin checkpoints)

En Vast (RTX 4090):
    scp -P <port> aion_c_vast.zip root@<host>:/root/
    ssh -p <port> root@<host>
    cd /root && unzip aion_c_vast.zip && cd AION-C
    pip install -r requirements_vast.txt
    python train_1b_canonical.py --config 1b --steps 15000

Lo que incluye (TODO lo nuevo de Fases A-D + wiring + Fase E):
  agent/        — tools, planner, skills, self_check, lifecycle, goals,
                  reasoning_levels, tool_executor
  brain/        — version_manager
  evaluation/   — eval_prompts, metrics
  memory/       — semantic_store, user_model, response_cache, conversation_history
  skills/       — los 11 .md
  soma/         — interface
  symbolic/     — engine + 11 reglas (axiom/forge_c/cora)
  synth/        — generators canónicos + dataset_unifier + dataloader +
                  build_canonical_dataset
  training/     — anti_forgetting
  world_model/  — scratch_pad + simulators + verifier
  tests/        — TODOS los tests (para correr en Vast antes de training)
  router/, encoder/, decoder/, motors/, orchestrator/, crystallizer/,
  cre/, core/, tokenizer/, experiments/, validation/, inference/,
  budget/, unifier/

Más:
  - datasets/dataset_canonical_70k.jsonl  (el dataset 70K canónico, ~36 MB)
  - train_1b_canonical.py                 (el script de training)
  - train_tiny_canonical.py               (script tiny para sanity check)
  - verify_tiny_e2e.py                    (verifier E2E)
  - requirements_vast.txt                 (deps mínimas: torch + nada más)
  - VAST_README.md                        (guía paso a paso)

NO incluye:
  - checkpoints/ (los pesos viejos no sirven, se entrena from scratch)
  - .git, .pytest_cache, __pycache__
  - app.py viejo (Flask reemplazado por backend/)
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Set

# Force UTF-8 stdout (Windows cp1252 chokes on ✓ ⚠)
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


_ROOT = Path(__file__).resolve().parent
_OUT_NAME = "aion_c_vast.zip"


# ─────────────────────────────────────────────────────────────────────────────
# Lo que incluimos
# ─────────────────────────────────────────────────────────────────────────────

# Directorios completos a incluir (recursivo)
DIRS_TO_PACKAGE: List[str] = [
    # Modelo / pipeline core
    "router",
    "encoder",
    "decoder",
    "motors",
    "orchestrator",
    "crystallizer",
    "cre",
    "core",
    "tokenizer",

    # Agent / Fase A
    "agent",
    "memory",
    "training",

    # Fase A extras
    "world_model",
    "symbolic",
    "brain",
    "soma",

    # Fase B
    "synth",
    "skills",

    # Fase E
    "evaluation",

    # Fase F — bloque cognitivo completo (Partes 22-27)
    "growth",
    "composition",
    "sleep",
    "pruning",
    "reward",
    "compression",
    "sparse",

    # Backend (por si querés correr la UI en Vast también)
    "backend",

    # Tests para correr antes de training
    "tests",

    # Auxiliares del repo original
    "experiments",
    "validation",
    "inference",
    "budget",
    "unifier",
    "tools",
    "visualization",
]

# Archivos top-level a incluir
FILES_TO_PACKAGE: List[str] = [
    # Scripts de training
    "train_1b_canonical.py",
    "train_tiny_canonical.py",
    "train_1b_sequential.py",         # Motor-Sequential Training
    "eval_final.py",                   # Eval post-training

    # Verificación E2E
    "verify_tiny_e2e.py",
    "auto_learn_demo.py",

    # Spec del training (lección OOD del identity skill)
    "TRAINING_SPEC.md",

    # Datasets críticos — 86K (72.5K base + real_knowledge + search + search+learn)
    "datasets/dataset_canonical_86k.jsonl",
    "datasets/dataset_canonical_86k.stats.json",
    "datasets/dataset_canonical_72_5k.jsonl",
    "datasets/dataset_canonical_70k.stats.json",
    "datasets/metacognitive_2500.jsonl",
    "datasets/real_knowledge.jsonl",
    "datasets/real_knowledge.stats.json",
    "datasets/search_web_3k.jsonl",
    "datasets/search_and_learn_2k.jsonl",
]

# Patterns to exclude (no copiar)
EXCLUDE_PATTERNS: Set[str] = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".gitignore",
    ".venv",
    ".idea",
    ".vscode",
    "node_modules",
    ".DS_Store",
}

EXCLUDE_SUFFIXES: Set[str] = {
    ".pyc", ".pyo", ".pyd",
}

# Files inside excluded that we still want
EXCLUDE_FILE_PATTERNS: Set[str] = {
    "checkpoints",  # los pesos viejos no se incluyen — from scratch
}


def should_exclude(path: Path) -> bool:
    parts = path.parts
    for ex in EXCLUDE_PATTERNS:
        if ex in parts:
            return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Generación del requirements.txt para Vast
# ─────────────────────────────────────────────────────────────────────────────


REQUIREMENTS_VAST = """\
# Mínimo necesario para entrenar AION-C 1.1B en RTX 4090
# El resto del código (agent, world_model, symbolic, etc.) es stdlib only.

# PyTorch con CUDA 12.4 — verificar versión disponible en Vast
# (las imágenes de Vast suelen traer torch ya instalado, pero forzamos la versión)
torch>=2.4.0
torchvision

# Tokenizer (sentencepiece backend)
sentencepiece>=0.2.0

# Dependencies del backend FastAPI (opcional, solo si querés correr la UI en Vast)
# fastapi>=0.110
# uvicorn[standard]>=0.30
# python-multipart
# httpx
"""


VAST_README = """\
# AION-C 1.1B Training en Vast.ai (RTX 4090)

## Setup inicial (5 min)

```bash
# 1) Subir el zip
scp -P <PORT> aion_c_vast.zip root@<HOST>:/root/

# 2) SSH al instance
ssh -p <PORT> root@<HOST>

# 3) Descomprimir
cd /root
unzip aion_c_vast.zip
cd AION-C

# 4) Instalar deps mínimas (la imagen de Vast suele traer torch ya)
pip install -r requirements_vast.txt

# 5) Verificar GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## Pre-flight: correr tests críticos (2 min)

```bash
python -m pytest tests/test_evaluation.py tests/test_canonical_dataloader.py tests/test_brain_version.py -q
```

Esperado: ~110 tests pass, 0 fail.

## Sanity check con tiny (5 min)

Antes de tirar las 17 horas de GPU al 1.1B, verifica que el script corre OK:

```bash
python train_tiny_canonical.py --steps 100
python verify_tiny_e2e.py
```

Esperado: tiny entrenado, 7/7 E2E checks PASS.

## Training del 1.1B (estimado 15-17h con early stopping)

```bash
python train_1b_canonical.py --config 1b --steps 15000 2>&1 | tee train_1b.log
```

Parámetros activos:
  - fp16 mixed precision (autocast + GradScaler)
  - cosine LR con warmup=300, lr base 1e-4
  - grad_accum=16 (effective batch 16)
  - routing_w=1.0, balance_w=0.5
  - eval cada 200 steps con 50 prompts canónicos
  - save best por gen_quality.combined (no val_loss)
  - early stopping patience=500
  - BrainVersionManager → brain/v1/
  - context 1024 (dec_max_seq_len)

Logs cada 50 steps: lm/route/balance loss, routing acc%, lr, sps, ETA, VRAM.
Eval cada 200 steps: exact_match, BLEU, routing_accuracy, combined.

## Checkpoint final + descarga

El checkpoint best-by-combined queda en:
  - `checkpoints/aion_1b_canonical.pt`  (state_dict + history JSON)
  - `brain/v1/weights.pt`               (BrainVersionManager)
  - `brain/v1/metadata.json`            (metrics + parent)
  - `checkpoints/aion_1b_canonical.metrics.json`  (history completo)

Para descargar a tu PC:
```bash
scp -P <PORT> root@<HOST>:/root/AION-C/checkpoints/aion_1b_canonical.pt ./
scp -P <PORT> -r root@<HOST>:/root/AION-C/brain/ ./
```

## Recovery: si Vast se cae a mitad

```bash
# Resume desde el último best
python train_1b_canonical.py --config 1b --steps 15000 \\
    --resume checkpoints/aion_1b_canonical.pt
```

## Budget esperado

- $0.229/hr × 17h ≈ $3.90
- 15K steps × ~3.5 sps ≈ 75 min training puro
- Eval cada 200 steps × 75 evals × ~3 min ≈ 225 min eval
- **Total ~5 horas** si converge rápido, hasta 17h si llega a max steps

## Si el modelo sale mediocre

1. Mira `checkpoints/aion_1b_canonical.metrics.json` — ¿el combined estaba subiendo o flat?
2. Si está flat → la loss del LM no baja → puede ser dataset o config problem
3. Si combined sube pero exact_match no → el modelo aprende formato pero no contenido
4. Considera más steps o diferente recipe (subir lr, bajar warmup, distinct dataset)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def add_dir_to_zip(zf: zipfile.ZipFile, dir_path: Path, arc_root: str) -> int:
    """Agrega un directorio recursivamente al zip. Devuelve count de archivos."""
    if not dir_path.exists():
        return 0
    n = 0
    for p in dir_path.rglob("*"):
        if p.is_file():
            if should_exclude(p):
                continue
            arcname = f"{arc_root}/{p.relative_to(dir_path.parent)}"
            zf.write(str(p), arcname)
            n += 1
    return n


def add_file_to_zip(zf: zipfile.ZipFile, file_path: Path, arc_root: str) -> bool:
    if not file_path.exists():
        return False
    arcname = f"{arc_root}/{file_path.relative_to(file_path.parent.parent)}"
    zf.write(str(file_path), arcname)
    return True


def write_aux_files(repo_root: Path) -> None:
    """Escribe requirements_vast.txt y VAST_README.md en el repo (no en /tmp)."""
    (repo_root / "requirements_vast.txt").write_text(REQUIREMENTS_VAST, encoding="utf-8")
    (repo_root / "VAST_README.md").write_text(VAST_README, encoding="utf-8")


def main() -> None:
    print("=" * 64)
    print("  AION-C Vast.ai Packager")
    print("=" * 64)

    if not (_ROOT / "datasets" / "dataset_canonical_86k.jsonl").exists():
        print("ERROR: datasets/dataset_canonical_86k.jsonl no existe")
        print("       Correr primero: synth.metacognitive_gen + real_knowledge_gen + search_web_gen + search_and_learn_gen, luego merge")
        sys.exit(1)

    print(f"\nWriting requirements_vast.txt + VAST_README.md...")
    write_aux_files(_ROOT)

    out_path = _ROOT / _OUT_NAME
    print(f"\nBuilding {_OUT_NAME}...")

    arc_root = "AION-C"
    total_files = 0

    with zipfile.ZipFile(str(out_path), "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Directorios
        print("\n[dirs]")
        for d in DIRS_TO_PACKAGE:
            dpath = _ROOT / d
            if not dpath.exists():
                print(f"  ⚠ skip {d:20s} (not found)")
                continue
            n = add_dir_to_zip(zf, dpath, arc_root)
            print(f"  ✓ {d:20s} {n:>5} files")
            total_files += n

        # Archivos top-level
        print("\n[top-level files]")
        for fname in FILES_TO_PACKAGE:
            fpath = _ROOT / fname
            if not fpath.exists():
                print(f"  ⚠ skip {fname:50s} (not found)")
                continue
            arcname = f"{arc_root}/{fname}"
            zf.write(str(fpath), arcname)
            print(f"  ✓ {fname}")
            total_files += 1

        # requirements_vast.txt + VAST_README.md (recién escritos)
        for aux in ("requirements_vast.txt", "VAST_README.md"):
            ap = _ROOT / aux
            if ap.exists():
                zf.write(str(ap), f"{arc_root}/{aux}")
                print(f"  ✓ {aux}")
                total_files += 1

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print()
    print("=" * 64)
    print(f"  Done. {total_files} files → {_OUT_NAME} ({size_mb:.1f} MB)")
    print("=" * 64)
    print(f"\nPath: {out_path}")
    print()
    print("Next steps:")
    print(f"  scp -P <PORT> {_OUT_NAME} root@<HOST>:/root/")
    print(f"  ssh -p <PORT> root@<HOST>")
    print(f"  cd /root && unzip {_OUT_NAME} && cd AION-C")
    print(f"  pip install -r requirements_vast.txt")
    print(f"  python train_1b_canonical.py --config 1b --steps 15000")
    print()


if __name__ == "__main__":
    main()
