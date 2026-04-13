"""
training/monitoring/logger.py — Append-only JSONL metrics logger.

Cada línea del archivo es un dict autoserializable con métricas del step.
Formato estándar:

    {
        "step": 42,
        "phase": "phase_2_motor:cora",
        "loss": 3.142,
        "routing_acc": 87.5,
        "lr": 0.0001,
        "sps": 0.654,
        "ram_gb": 5.2,
        "ram_pct": 32.4,
        "cpu_temp": 68.5,          # nullable
        "gpu_temp": null,
        "proc_rss_gb": 4.8,
        "eta_min": 38.1,
        "timestamp": 1775928900.5
    }

El archivo es append-only: cada write es atómico (una línea JSON + newline)
así que el reader puede leerlo mientras el writer escribe sin corrupción.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


@dataclass
class MetricEntry:
    """Una fila del metrics.jsonl."""
    step: int
    phase: str
    loss: float
    sps: float
    lr: float
    ram_gb: Optional[float] = None
    ram_pct: Optional[float] = None
    proc_rss_gb: Optional[float] = None
    cpu_temp: Optional[float] = None
    gpu_temp: Optional[float] = None
    routing_acc: Optional[float] = None
    eta_min: Optional[float] = None
    elapsed_sec: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsLogger:
    """Append-only JSONL writer con flush inmediato para que readers vean
    las líneas nuevas apenas se escriben.

    Thread-safe: una Lock protege las escrituras.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        # Create (or truncate if restart scenario) — we actually append
        # so multi-phase runs keep history
        if not self.path.exists():
            self.path.touch()

    def log(self, entry: MetricEntry) -> None:
        """Append one JSON line and flush immediately."""
        line = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except (OSError, AttributeError):
                    pass

    def log_dict(self, **kwargs: Any) -> None:
        """Convenience: build MetricEntry from kwargs and log."""
        entry = MetricEntry(**kwargs)
        self.log(entry)

    def read_all(self) -> List[Dict[str, Any]]:
        """Read every logged entry. For the health checker and status."""
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def read_last_n(self, n: int) -> List[Dict[str, Any]]:
        """Read only the last N entries (for light status queries)."""
        rows = self.read_all()
        return rows[-n:]

    def count(self) -> int:
        return len(self.read_all())


def find_latest_log_dir(base_dir: Path) -> Optional[Path]:
    """Find the most recent sequential_* log directory under base_dir."""
    if not base_dir.exists():
        return None
    candidates = [p for p in base_dir.iterdir()
                  if p.is_dir() and p.name.startswith("sequential_")]
    if not candidates:
        return None
    # Sort by name (timestamps in names are sortable lexically)
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]
