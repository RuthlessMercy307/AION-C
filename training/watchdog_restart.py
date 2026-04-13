"""
training/watchdog_restart.py — Auto-recovery loop para training crasheado.

Corre el comando de training como subprocess. Si muere con exit code !=0,
detecta el último checkpoint, arma el comando de resume y relanza.
Máximo 3 reintentos; después del tercer fail, graba el evento y sale.

Uso:
    # Arranca con el comando de training inicial
    python training/watchdog_restart.py -- \\
        python train_1b_sequential.py --config 1b --phase-1-optimizer sgd \\
            --phase-1-steps 1500 --monitoring on

    # Ajustar retries
    python training/watchdog_restart.py --max-retries 5 -- \\
        python train_1b_sequential.py --config 1b ...

El script NO modifica el comando original en el primer intento. A
partir del segundo intento, añade `--resume` para que el training
reanude desde el último checkpoint.

Logs:
    training/logs/latest/ (symlink vía latest.txt)/restart.log
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def get_latest_log_dir() -> Optional[Path]:
    base = ROOT / "training" / "logs"
    marker = base / "latest.txt"
    if marker.exists():
        try:
            name = marker.read_text(encoding="utf-8").strip()
            d = base / name
            if d.exists():
                return d
        except OSError:
            pass
    # Fallback: find most recent sequential_* dir
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("sequential_")] if base.exists() else []
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def append_log(log_dir: Optional[Path], msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] [restart] {msg}"
    print(line, flush=True)
    if log_dir is None:
        return
    try:
        path = log_dir / "restart.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def run_with_retries(cmd: List[str], max_retries: int = 3,
                     cooldown_sec: float = 10.0) -> int:
    """Run `cmd` as subprocess, retrying on non-zero exit code.

    Returns the final exit code. Max `max_retries` retries after the
    first attempt. Between retries, sleeps cooldown_sec.

    From retry 2 onwards, adds `--resume` to the command IF it's a
    train_1b_sequential.py call (best effort).
    """
    attempt = 0
    final_exit_code = 0
    log_dir = get_latest_log_dir()

    while attempt <= max_retries:
        if attempt > 0:
            # Add --resume if the command is train_1b_sequential.py
            if any("train_1b_sequential" in c for c in cmd) and "--resume" not in cmd:
                cmd = cmd + ["--resume"]
            append_log(log_dir, f"retry {attempt}/{max_retries} in {cooldown_sec:.0f}s: {' '.join(cmd)}")
            time.sleep(cooldown_sec)
        else:
            append_log(log_dir, f"initial run: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(cmd, cwd=str(ROOT))
            proc.wait()
            rc = proc.returncode
            append_log(log_dir, f"process exited with code {rc}")
            log_dir = get_latest_log_dir()  # Refresh in case training created a new one
            if rc == 0:
                append_log(log_dir, "SUCCESS")
                return 0
            final_exit_code = rc
        except KeyboardInterrupt:
            append_log(log_dir, "interrupted by user (Ctrl-C)")
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            return 130
        except Exception as exc:
            append_log(log_dir, f"subprocess error: {exc}")
            final_exit_code = 1

        attempt += 1

    append_log(log_dir, f"all {max_retries + 1} attempts failed (last exit={final_exit_code})")
    return final_exit_code


def main() -> None:
    p = argparse.ArgumentParser(
        description="Auto-restart wrapper for training scripts",
        usage="python training/watchdog_restart.py [--max-retries N] -- <command>",
    )
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max retries after first attempt (default 3)")
    p.add_argument("--cooldown", type=float, default=10.0,
                   help="Seconds to wait between retries (default 10)")
    # Everything after `--` is the command to run
    if "--" not in sys.argv:
        p.print_help()
        sys.exit(1)
    sep = sys.argv.index("--")
    own_args = sys.argv[1:sep]
    cmd = sys.argv[sep + 1:]
    args = p.parse_args(own_args)
    if not cmd:
        print("ERROR: no command provided after `--`", file=sys.stderr)
        sys.exit(1)

    rc = run_with_retries(cmd, max_retries=args.max_retries, cooldown_sec=args.cooldown)
    sys.exit(rc)


if __name__ == "__main__":
    main()
