"""
training/monitoring/context.py — MonitoringContext agrupa logger + control + watchdog.

El trainer recibe un MonitoringContext opcional y lo usa en su training
loop para escribir métricas, leer el control file, y manejar acciones.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from training.monitoring.logger import MetricsLogger, MetricEntry
from training.monitoring.control import ControlFile, ControlAction
from training.monitoring.watchdog import Watchdog, WatchdogConfig


@dataclass
class MonitoringContext:
    """Estado y handles del sistema de monitoring.

    Se pasa al trainer. El trainer lo usa para:
        - ctx.log_step(...) cada 25 steps
        - ctx.poll_control() cada 10 steps
        - ctx.close() al final
    """
    log_dir: Path
    logger: MetricsLogger
    control: ControlFile
    watchdog: Optional[Watchdog]
    log_every: int = 25
    poll_every: int = 10
    started_at: float = field(default_factory=time.time)

    # ── Logging ─────────────────────────────────────────────────────────
    def log_step(
        self,
        step: int,
        phase: str,
        loss: float,
        sps: float,
        lr: float,
        **extras: Any,
    ) -> None:
        """Append a metric entry. Snapshot hardware state inline."""
        ram_gb = None
        ram_pct = None
        proc_rss_gb = None
        cpu_temp = None
        try:
            import psutil
            vm = psutil.virtual_memory()
            ram_gb = round((vm.total - vm.available) / (1024 ** 3), 2)
            ram_pct = float(vm.percent)
            proc = psutil.Process(os.getpid())
            proc_rss_gb = round(proc.memory_info().rss / (1024 ** 3), 2)
            try:
                sensors = psutil.sensors_temperatures()
                for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
                    if key in sensors and sensors[key]:
                        cpu_temp = float(sensors[key][0].current)
                        break
            except Exception:
                pass
        except ImportError:
            pass

        entry = MetricEntry(
            step=step,
            phase=phase,
            loss=round(loss, 4),
            sps=round(sps, 4),
            lr=lr,
            ram_gb=ram_gb,
            ram_pct=ram_pct,
            proc_rss_gb=proc_rss_gb,
            cpu_temp=cpu_temp,
            gpu_temp=None,
            routing_acc=extras.get("routing_acc"),
            eta_min=extras.get("eta_min"),
            elapsed_sec=extras.get("elapsed_sec"),
        )
        self.logger.log(entry)

    # ── Control ─────────────────────────────────────────────────────────
    def poll_control(self) -> Optional[Dict[str, Any]]:
        """Read the control file and consume any pending action."""
        return self.control.consume()

    def write_note(self, message: str, author: str = "training") -> None:
        """Append a note to the events log."""
        events_path = self.log_dir / "events.log"
        try:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[{time.strftime('%H:%M:%S')}] [{author}] {message}\n")
        except OSError:
            pass

    # ── Lifecycle ───────────────────────────────────────────────────────
    def close(self) -> None:
        if self.watchdog is not None:
            self.watchdog.stop()


def create_monitoring_context(
    log_dir: Optional[Path] = None,
    enable_watchdog: bool = True,
    watchdog_config: Optional[WatchdogConfig] = None,
    log_every: int = 25,
    poll_every: int = 10,
) -> MonitoringContext:
    """Build a ready-to-use MonitoringContext.

    If log_dir is None, creates a fresh directory under training/logs/
    with a timestamp name.
    """
    if log_dir is None:
        from pathlib import Path as _Path
        base = _Path(__file__).resolve().parent.parent / "logs"
        stamp = time.strftime("sequential_%Y%m%d_%H%M%S")
        log_dir = base / stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = MetricsLogger(log_dir / "metrics.jsonl")
    control = ControlFile(log_dir / "control.json")
    watchdog = None
    if enable_watchdog:
        watchdog = Watchdog(
            control_file=control,
            config=watchdog_config or WatchdogConfig(),
            log_path=log_dir / "watchdog.log",
        )
        watchdog.start()

    # Write a marker file that points to THIS directory as the "active" one
    # so external tools can find it without knowing the timestamp.
    try:
        base = log_dir.parent
        marker = base / "latest.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(log_dir.name, encoding="utf-8")
    except OSError:
        pass

    ctx = MonitoringContext(
        log_dir=log_dir,
        logger=logger,
        control=control,
        watchdog=watchdog,
        log_every=log_every,
        poll_every=poll_every,
    )
    ctx.write_note(f"monitoring started (log_dir={log_dir.name})", author="monitoring")
    return ctx
