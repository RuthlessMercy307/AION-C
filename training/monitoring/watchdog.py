"""
training/monitoring/watchdog.py — Background thread que vigila el sistema.

Cada `interval_sec` segundos (default 60) lee RAM, temps, y escribe al
ControlFile si detecta una condición CRÍTICA que requiera pause
inmediato.

Condiciones críticas por default (ajustables):
    - RAM sistema > 15 GB usado      → pause (16 GB total = swap inminente)
    - CPU temp > 90°C                 → pause (thermal emergency)
    - GPU temp > 95°C                 → pause
    - proceso RSS > 14 GB             → pause

El watchdog NO lee métricas de training — eso es trabajo del health
checker. El watchdog sólo reacciona a estado del HARDWARE/OS.

Corre en un thread daemon: muere automáticamente cuando el proceso
principal termina.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from training.monitoring.control import ControlFile, ControlAction


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class WatchdogConfig:
    interval_sec: float = 60.0
    ram_critical_gb: float = 15.0       # RAM total usado
    proc_rss_critical_gb: float = 14.0  # RSS del proceso de training
    cpu_temp_critical_c: float = 90.0
    gpu_temp_critical_c: float = 95.0
    log_every_check: bool = False       # si True, log cada check aunque esté OK
    # Cooldown: cuántos segundos esperar entre writes al control file para
    # evitar spam si la condición persiste.
    cooldown_sec: float = 120.0


# ════════════════════════════════════════════════════════════════════════════
# Watchdog
# ════════════════════════════════════════════════════════════════════════════

class Watchdog:
    """Background system-health watcher.

    Uso:
        watch = Watchdog(control_file, WatchdogConfig())
        watch.start()
        ... training loop ...
        watch.stop()
    """

    def __init__(
        self,
        control_file: ControlFile,
        config: Optional[WatchdogConfig] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self.config = config or WatchdogConfig()
        self.control = control_file
        self.log_path = log_path
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_write_ts = 0.0
        self._checks_run = 0
        self._alerts_raised = 0

    # ── Lifecycle ───────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="AION-C-Watchdog",
            daemon=True,
        )
        self._thread.start()
        self._log("watchdog started")

    def stop(self, join_timeout: float = 3.0) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=join_timeout)
        self._thread = None
        self._log(f"watchdog stopped after {self._checks_run} checks, {self._alerts_raised} alerts")

    # ── Main loop ───────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._check_once()
            except Exception as exc:
                self._log(f"watchdog check error: {exc}")
            # Sleep in small chunks so stop() is responsive
            for _ in range(int(self.config.interval_sec * 10)):
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)

    def _check_once(self) -> None:
        self._checks_run += 1
        reasons: List[str] = []

        # System RAM
        try:
            import psutil
            vm = psutil.virtual_memory()
            ram_used_gb = (vm.total - vm.available) / (1024 ** 3)
            if ram_used_gb > self.config.ram_critical_gb:
                reasons.append(f"RAM used {ram_used_gb:.1f}GB > {self.config.ram_critical_gb}GB")

            # Proc RSS
            proc = psutil.Process(os.getpid())
            rss_gb = proc.memory_info().rss / (1024 ** 3)
            if rss_gb > self.config.proc_rss_critical_gb:
                reasons.append(f"proc RSS {rss_gb:.1f}GB > {self.config.proc_rss_critical_gb}GB")

            # CPU temp (best-effort; often unavailable on Windows)
            cpu_temp = None
            try:
                sensors = psutil.sensors_temperatures()
                for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
                    if key in sensors and sensors[key]:
                        cpu_temp = float(sensors[key][0].current)
                        break
            except Exception:
                cpu_temp = None
            if cpu_temp is not None and cpu_temp > self.config.cpu_temp_critical_c:
                reasons.append(f"CPU temp {cpu_temp:.0f}C > {self.config.cpu_temp_critical_c}C")
        except ImportError:
            # psutil not available — watchdog is a no-op
            if self._checks_run == 1:
                self._log("psutil not available, watchdog disabled")
            return
        except Exception as exc:
            self._log(f"psutil read error: {exc}")
            return

        if self.config.log_every_check:
            self._log(f"check {self._checks_run}: ram={ram_used_gb:.1f}GB rss={rss_gb:.1f}GB reasons={reasons}")

        # If any critical condition, escalate via control file
        if reasons:
            now = time.time()
            if now - self._last_write_ts >= self.config.cooldown_sec:
                self._alerts_raised += 1
                self._last_write_ts = now
                msg = "; ".join(reasons)
                self.control.write(
                    action=ControlAction.PAUSE.value,
                    author="watchdog",
                    reasons=reasons,
                    message=f"CRITICAL: {msg}",
                )
                self._log(f"ALERT {self._alerts_raised}: {msg} -> wrote pause to control file")

    # ── Logging ─────────────────────────────────────────────────────────
    def _log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] [watchdog] {msg}"
        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass
        print(line, flush=True)

    # ── Stats ───────────────────────────────────────────────────────────
    @property
    def checks_run(self) -> int:
        return self._checks_run

    @property
    def alerts_raised(self) -> int:
        return self._alerts_raised
