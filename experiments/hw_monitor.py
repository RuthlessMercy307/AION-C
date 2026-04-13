"""
experiments/hw_monitor.py — Snapshot ligero de hardware para benchmarks.

Mide:
    - CPU: %uso global, frecuencia, temp (si disponible)
    - RAM: usada total, %
    - GPU: VRAM usada (si se provee un callback del framework), temp (via
      herramienta externa si disponible: nvidia-smi, rocm-smi, Wmic)
    - Proceso actual: RSS y CPU%

Todo es best-effort. Si un dato no está disponible, se devuelve None y
se incluye en el campo "unavailable" del snapshot.

Uso:
    from experiments.hw_monitor import HWMonitor
    mon = HWMonitor()
    snap = mon.snapshot()
    # dict con keys cpu_percent, ram_used_gb, ram_total_gb, ...

No requiere dependencias fuera de stdlib+psutil. Si psutil no está
disponible, devuelve todos los campos en None.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ════════════════════════════════════════════════════════════════════════════
# psutil optional import
# ════════════════════════════════════════════════════════════════════════════

try:
    import psutil  # type: ignore
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ════════════════════════════════════════════════════════════════════════════
# Snapshot model
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class HWSnapshot:
    timestamp: float = field(default_factory=time.time)
    cpu_percent: Optional[float] = None
    cpu_freq_mhz: Optional[float] = None
    cpu_temp_c: Optional[float] = None
    ram_used_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    ram_percent: Optional[float] = None
    proc_rss_gb: Optional[float] = None
    proc_cpu_percent: Optional[float] = None
    gpu_vram_used_gb: Optional[float] = None
    gpu_vram_total_gb: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    gpu_util_percent: Optional[float] = None
    gpu_backend: Optional[str] = None  # "nvidia-smi" / "rocm-smi" / "cuda" / "n/a"
    unavailable: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ════════════════════════════════════════════════════════════════════════════
# CPU / RAM
# ════════════════════════════════════════════════════════════════════════════

def _cpu_snapshot(snap: HWSnapshot) -> None:
    if not HAS_PSUTIL:
        snap.unavailable.append("cpu_psutil_missing")
        return
    try:
        snap.cpu_percent = psutil.cpu_percent(interval=None)
    except Exception:
        snap.unavailable.append("cpu_percent")
    try:
        freq = psutil.cpu_freq()
        if freq:
            snap.cpu_freq_mhz = float(freq.current)
    except Exception:
        snap.unavailable.append("cpu_freq")
    # CPU temperature — psutil.sensors_temperatures() works on Linux;
    # on Windows it usually returns empty.
    try:
        sensors = getattr(psutil, "sensors_temperatures", lambda: {})()
        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if key in sensors and sensors[key]:
                snap.cpu_temp_c = float(sensors[key][0].current)
                break
    except Exception:
        snap.unavailable.append("cpu_temp_sensor")


def _ram_snapshot(snap: HWSnapshot) -> None:
    if not HAS_PSUTIL:
        snap.unavailable.append("ram_psutil_missing")
        return
    try:
        mem = psutil.virtual_memory()
        snap.ram_used_gb = round((mem.total - mem.available) / (1024 ** 3), 2)
        snap.ram_total_gb = round(mem.total / (1024 ** 3), 2)
        snap.ram_percent = float(mem.percent)
    except Exception:
        snap.unavailable.append("ram")

    try:
        proc = psutil.Process(os.getpid())
        snap.proc_rss_gb = round(proc.memory_info().rss / (1024 ** 3), 2)
        snap.proc_cpu_percent = proc.cpu_percent(interval=None)
    except Exception:
        snap.unavailable.append("proc_stats")


# ════════════════════════════════════════════════════════════════════════════
# GPU detection + stats
# ════════════════════════════════════════════════════════════════════════════

def _try_nvidia_smi() -> Optional[Dict[str, float]]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
        ).decode("utf-8", errors="replace").strip().splitlines()
        if not out:
            return None
        used, total, temp, util = [x.strip() for x in out[0].split(",")]
        return {
            "vram_used_gb": round(float(used) / 1024, 2),
            "vram_total_gb": round(float(total) / 1024, 2),
            "temp_c": float(temp),
            "util_percent": float(util),
            "backend": "nvidia-smi",
        }
    except Exception:
        return None


def _try_rocm_smi() -> Optional[Dict[str, float]]:
    if shutil.which("rocm-smi") is None:
        return None
    try:
        # rocm-smi returns JSON with --json; parse memory and temp
        out = subprocess.check_output(
            ["rocm-smi", "--showmemuse", "--showtemp", "--json"],
            timeout=3,
        ).decode("utf-8", errors="replace")
        import json
        data = json.loads(out)
        first_gpu_key = next(iter(data))
        info = data[first_gpu_key]
        result: Dict[str, float] = {"backend": "rocm-smi"}  # type: ignore
        # Keys differ by version; try common ones
        for k, v in info.items():
            kl = k.lower()
            if "memory used" in kl or "memory use" in kl:
                try:
                    result["vram_used_gb"] = round(float(v) / 1024, 2)  # type: ignore
                except Exception:
                    pass
            if "temperature" in kl and "edge" in kl:
                try:
                    result["temp_c"] = float(v)  # type: ignore
                except Exception:
                    pass
        return result if len(result) > 1 else None
    except Exception:
        return None


def _try_wmic_gpu() -> Optional[Dict[str, Any]]:
    """Windows WMIC fallback for AMD GPU info (no temp, just names)."""
    if os.name != "nt":
        return None
    try:
        out = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get",
             "Name,AdapterRAM", "/format:csv"],
            timeout=3,
        ).decode("utf-8", errors="replace")
        lines = [l for l in out.strip().splitlines() if l and "Node" not in l]
        if len(lines) < 2:
            return None
        # Parse first non-empty data row
        header = lines[0].split(",")
        for row in lines[1:]:
            cols = row.split(",")
            if len(cols) == len(header):
                d = dict(zip(header, cols))
                ram_raw = d.get("AdapterRAM", "0")
                try:
                    total_gb = round(int(ram_raw) / (1024 ** 3), 2)
                except Exception:
                    total_gb = None
                return {
                    "backend": "wmic",
                    "vram_total_gb": total_gb,
                    "vram_used_gb": None,
                    "temp_c": None,
                    "gpu_name": d.get("Name", "").strip(),
                }
    except Exception:
        return None
    return None


def _gpu_snapshot(snap: HWSnapshot) -> None:
    for detector in (_try_nvidia_smi, _try_rocm_smi, _try_wmic_gpu):
        info = detector()
        if info is not None:
            snap.gpu_backend = str(info.get("backend", "unknown"))
            snap.gpu_vram_used_gb = info.get("vram_used_gb")
            snap.gpu_vram_total_gb = info.get("vram_total_gb")
            snap.gpu_temp_c = info.get("temp_c")
            snap.gpu_util_percent = info.get("util_percent")
            return
    snap.gpu_backend = "n/a"
    snap.unavailable.append("gpu_backend")


# ════════════════════════════════════════════════════════════════════════════
# HWMonitor (aggregator)
# ════════════════════════════════════════════════════════════════════════════

class HWMonitor:
    """Snapshot no bloqueante de hardware.

    Uso:
        mon = HWMonitor()
        mon.warmup()          # inicializa counters de psutil
        snap_before = mon.snapshot()
        ... work ...
        snap_after = mon.snapshot()
    """

    def __init__(self) -> None:
        self._warmed = False

    def warmup(self) -> None:
        if not HAS_PSUTIL:
            return
        try:
            psutil.cpu_percent(interval=None)
            psutil.Process(os.getpid()).cpu_percent(interval=None)
        except Exception:
            pass
        self._warmed = True

    def snapshot(self) -> HWSnapshot:
        if not self._warmed:
            self.warmup()
            # Small pause so CPU percent is meaningful
            time.sleep(0.1)
        snap = HWSnapshot()
        _cpu_snapshot(snap)
        _ram_snapshot(snap)
        _gpu_snapshot(snap)
        return snap

    def summary(self, snap: HWSnapshot) -> str:
        parts = []
        if snap.cpu_percent is not None:
            parts.append(f"cpu {snap.cpu_percent:.0f}%")
        if snap.ram_used_gb is not None and snap.ram_total_gb is not None:
            parts.append(f"ram {snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f}GB ({snap.ram_percent:.0f}%)")
        if snap.proc_rss_gb is not None:
            parts.append(f"proc_rss {snap.proc_rss_gb:.2f}GB")
        if snap.gpu_vram_used_gb is not None:
            vt = snap.gpu_vram_total_gb or 0
            parts.append(f"vram {snap.gpu_vram_used_gb:.1f}/{vt:.1f}GB")
        elif snap.gpu_vram_total_gb is not None:
            parts.append(f"vram ?/{snap.gpu_vram_total_gb:.1f}GB")
        if snap.cpu_temp_c is not None:
            parts.append(f"cpu_temp {snap.cpu_temp_c:.0f}C")
        if snap.gpu_temp_c is not None:
            parts.append(f"gpu_temp {snap.gpu_temp_c:.0f}C")
        return " · ".join(parts) if parts else "(no stats available)"


# ════════════════════════════════════════════════════════════════════════════
# CLI rápido
# ════════════════════════════════════════════════════════════════════════════

def _cli() -> None:  # pragma: no cover
    mon = HWMonitor()
    mon.warmup()
    time.sleep(0.3)
    snap = mon.snapshot()
    print("HW snapshot:")
    for k, v in snap.to_dict().items():
        print(f"  {k}: {v}")
    print()
    print("Summary:", mon.summary(snap))


if __name__ == "__main__":
    _cli()
