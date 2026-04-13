"""
training/monitoring/control.py — Bidirectional control file.

El training script lee `control.json` cada 10 steps. Claude (u otro
agente externo) escribe a ese archivo para pedir acciones.

Acciones soportadas:

    {"action": "none"}                   # no-op
    {"action": "pause"}                   # pausa limpia (guarda checkpoint)
    {"action": "resume"}                  # reanuda desde pausa
    {"action": "adjust_lr", "value": 5e-5}   # cambia LR
    {"action": "stop"}                    # para todo, guarda y exit
    {"action": "note", "message": "..."}  # nota que el training loggea

Campos adicionales:
    "timestamp": float — cuándo se escribió la action
    "author":    str   — quién la pidió ("claude", "watchdog", "user", ...)

Cada vez que el training consume una action, reescribe el archivo con
`{"action": "none"}` para evitar reejecutarla. Esto implementa el
patrón "one-shot" — cada orden se ejecuta exactamente una vez.

El archivo es JSON simple, no JSONL. Writes son atómicos via temp+rename.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


class ControlAction(str, Enum):
    NONE = "none"
    PAUSE = "pause"
    RESUME = "resume"
    ADJUST_LR = "adjust_lr"
    STOP = "stop"
    NOTE = "note"


class ControlFile:
    """JSON control channel shared with external agents.

    Thread-safe via internal lock. Atomic writes via rename.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        if not self.path.exists():
            self._write({"action": ControlAction.NONE.value,
                         "timestamp": time.time(),
                         "author": "init"})

    # ── Read ────────────────────────────────────────────────────────────
    def read(self) -> Dict[str, Any]:
        """Read the current control state. Returns a dict with at least
        'action' and 'timestamp'. On any error, returns a NONE no-op."""
        try:
            with self._lock:
                if not self.path.exists():
                    return {"action": ControlAction.NONE.value,
                            "timestamp": time.time()}
                with self.path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if "action" not in data:
                    data["action"] = ControlAction.NONE.value
                return data
        except (OSError, json.JSONDecodeError):
            return {"action": ControlAction.NONE.value,
                    "timestamp": time.time()}

    # ── Write ───────────────────────────────────────────────────────────
    def write(
        self,
        action: str,
        author: str = "user",
        **extras: Any,
    ) -> None:
        """Write a new action atomically. Existing content is replaced."""
        data = {
            "action": action,
            "author": author,
            "timestamp": time.time(),
            **extras,
        }
        with self._lock:
            self._write(data)

    def clear(self, author: str = "training") -> None:
        """Reset to NONE no-op. Called by training after consuming an action."""
        self.write(action=ControlAction.NONE.value, author=author)

    def _write(self, data: Dict[str, Any]) -> None:
        """Atomic write via temp file + rename."""
        # Create temp file in the same directory for atomic rename
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=self.path.name + ".",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except (OSError, AttributeError):
                    pass
            # Atomic rename
            os.replace(tmp_path, str(self.path))
        except Exception:
            # Cleanup temp if something failed
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ── Helpers for consumers ───────────────────────────────────────────
    def consume(self) -> Optional[Dict[str, Any]]:
        """Read + immediately clear. Returns the action dict or None if NONE.

        Used inside the training loop every N steps: if there's a pending
        action, this reads it AND resets the file to NONE so the same
        action doesn't fire twice.
        """
        data = self.read()
        action = str(data.get("action", ControlAction.NONE.value))
        if action == ControlAction.NONE.value:
            return None
        # Reset so the action doesn't repeat
        self.clear(author="training_consumed")
        return data
