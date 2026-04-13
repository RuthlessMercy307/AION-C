#!/usr/bin/env python3
"""
package_for_h200.py -- Empaqueta TODO para H200 en un solo zip
==============================================================

Uso:
    cd AION-C
    python package_for_h200.py

Produce: aion_c_h200.zip que contiene:
  - AION-C/ completo (codigo, tests, configs)
  - AION-C/datasets/opus/ (los .jsonl de Opus copiados directamente)
  - AION-C/datasets/instruction_tuning.jsonl (28.5K ejemplos)
  - AION-C/train_h200.py (script maestro en la raiz)

En la H200:
    unzip aion_c_h200.zip
    cd AION-C
    python train_h200.py
"""

import os
import shutil
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent  # AION-C/
_IAS  = _ROOT.parent                    # ias/


def main():
    print("=" * 60)
    print("  AION-C H200 Packager")
    print("=" * 60)

    # -- 1. Ensure instruction tuning dataset exists --
    it_path = _ROOT / "datasets" / "instruction_tuning.jsonl"
    if not it_path.exists():
        print("\n[1/4] Generating instruction tuning dataset...")
        from synth.instruction_gen import InstructionGenerator, write_jsonl
        gen = InstructionGenerator(seed=42)
        examples = gen.generate_all()
        write_jsonl(examples, it_path)
    else:
        n = sum(1 for _ in open(str(it_path), encoding="utf-8"))
        print(f"\n[1/4] Instruction tuning dataset exists: {n:,} examples")

    # -- 2. Copy Opus datasets into AION-C/datasets/opus/ --
    print("\n[2/4] Copying Opus datasets...")
    opus_dest = _ROOT / "datasets" / "opus"
    opus_dest.mkdir(parents=True, exist_ok=True)

    opus_src = _IAS / "DataSet-Generator-Claude-Opus" / "mose_distillation_datasets" / "datasets"
    if not opus_src.exists():
        print(f"  WARNING: Opus source not found at {opus_src}")
        print("  The zip will work but training will use synthetic fallback.")
    else:
        count = 0
        for jsonl in opus_src.glob("mose_*.jsonl"):
            dest = opus_dest / jsonl.name
            if not dest.exists() or dest.stat().st_size != jsonl.stat().st_size:
                print(f"  Copying {jsonl.name} ({jsonl.stat().st_size / 1e6:.1f} MB)...")
                shutil.copy2(str(jsonl), str(dest))
                count += 1
            else:
                print(f"  {jsonl.name} already up-to-date")
        print(f"  {count} files copied to {opus_dest}")

    # -- 3. Create zip --
    print("\n[3/4] Creating aion_c_h200.zip...")
    zip_path = _ROOT / "aion_c_h200.zip"

    exclude_dirs = {"__pycache__", ".pytest_cache", "runs", ".git", "output",
                    "checkpoints"}
    exclude_files = {"aion_c.zip", "aion_c_h200.zip"}

    file_count = 0
    total_size = 0

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(str(_ROOT)):
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            for fname in filenames:
                if fname in exclude_files:
                    continue
                if fname.startswith("C:Temp"):
                    continue
                # Skip old checkpoint .pt files (NOT in datasets/)
                if fname.endswith(".pt") and "datasets" not in dirpath:
                    continue
                # Skip tokenizer binary (large, not needed for 32K)
                if fname.endswith(".model") and "tokenizer" in dirpath:
                    continue

                full = os.path.join(dirpath, fname)
                # arcname relative to parent of AION-C so it unpacks as AION-C/
                arcname = os.path.relpath(full, str(_ROOT.parent))
                zf.write(full, arcname)
                fsize = os.path.getsize(full)
                total_size += fsize
                file_count += 1

    zip_size = zip_path.stat().st_size
    print(f"  Files:         {file_count}")
    print(f"  Uncompressed:  {total_size / 1e6:.1f} MB")
    print(f"  Compressed:    {zip_size / 1e6:.1f} MB")
    print(f"  Saved to:      {zip_path}")

    # -- 4. Verify --
    print("\n[4/4] Verifying contents...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        names = zf.namelist()

    must_have = [
        "train_h200.py",
        "train_production.py",
        "instruction_gen.py",
        "instruction_tuning.jsonl",
        "quantize.py",
        "run_local.py",
        "graph_viewer.py",
        "smoke_test_full.py",
        "opus_dataset.py",
        "pipeline.py",
    ]
    for f in must_have:
        found = any(f in n for n in names)
        status = "OK" if found else "MISSING"
        print(f"  [{status}] {f}")

    # Check opus datasets
    opus_count = sum(1 for n in names if "opus/mose_" in n.replace("\\", "/"))
    print(f"  Opus datasets in zip: {opus_count} files")

    print()
    print("=" * 60)
    print("  DONE!")
    print()
    print("  On H200:")
    print("    unzip aion_c_h200.zip")
    print("    cd AION-C")
    print("    python train_h200.py")
    print("    # ~2 hours later...")
    print("    python -m inference.run_local")
    print("=" * 60)


if __name__ == "__main__":
    main()
