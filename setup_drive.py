"""
setup_drive.py — Empaqueta AION-C en un ZIP para subir a Google Drive.

Uso:
    python setup_drive.py              # genera aion_c.zip en la raíz del repo
    python setup_drive.py --out /ruta  # ruta personalizada

El ZIP incluye todos los módulos Python del proyecto (sin tests, experiments,
__pycache__, ni archivos de configuración de entorno).
"""

import argparse
import os
import zipfile
from pathlib import Path

# ── Módulos a incluir ────────────────────────────────────────────────────────
INCLUDE_PACKAGES = [
    "core",
    "encoder",
    "crystallizer",
    "cre",
    "decoder",
    "router",
    "motors",
    "orchestrator",
    "unifier",
    "validation",
    "budget",
    "synth",
    "tools",
    "experiments",   # benchmarks y scripts de entrenamiento
    "tests",         # suite de tests
]

# ── Patrones a excluir ───────────────────────────────────────────────────────
EXCLUDE_DIRS  = {"__pycache__", ".git", ".pytest_cache", "node_modules", ".venv", "venv"}
EXCLUDE_FILES = {".DS_Store", "Thumbs.db"}
INCLUDE_EXTS  = {".py", ".md"}


def should_include(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return False
    if path.name in EXCLUDE_FILES:
        return False
    return path.suffix in INCLUDE_EXTS


def build_zip(repo_root: Path, out_path: Path) -> None:
    added = 0
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Archivos en la raíz del repo (.py, .md)
        for root_file in sorted(repo_root.glob("*")):
            if root_file.is_file() and root_file.suffix in INCLUDE_EXTS:
                if root_file.name not in EXCLUDE_FILES:
                    zf.write(root_file, root_file.relative_to(repo_root))
                    added += 1

        # Paquetes
        for pkg in INCLUDE_PACKAGES:
            pkg_dir = repo_root / pkg
            if not pkg_dir.exists():
                print(f"  [SKIP] {pkg}/ — not found")
                continue
            for f in sorted(pkg_dir.rglob("*")):
                if f.is_file() and should_include(f):
                    arcname = f.relative_to(repo_root)
                    zf.write(f, arcname)
                    added += 1

    size_kb = out_path.stat().st_size / 1024
    print(f"Written : {out_path}")
    print(f"Files   : {added}")
    print(f"Size    : {size_kb:.1f} KB")
    print()
    print("Contents:")
    with zipfile.ZipFile(out_path) as zf:
        by_pkg: dict = {}
        for name in zf.namelist():
            pkg = name.split("/")[0]
            by_pkg.setdefault(pkg, 0)
            by_pkg[pkg] += 1
        for pkg, n in sorted(by_pkg.items()):
            print(f"  {pkg:<20} {n:>3} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package AION-C for Google Drive")
    parser.add_argument("--out", default=None, help="Output ZIP path (default: <repo>/aion_c.zip)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.resolve()
    out_path  = Path(args.out) if args.out else repo_root / "aion_c.zip"

    print(f"Repo    : {repo_root}")
    print(f"Output  : {out_path}")
    print(f"Packing : {', '.join(INCLUDE_PACKAGES)}")
    print()

    build_zip(repo_root, out_path)


if __name__ == "__main__":
    main()
