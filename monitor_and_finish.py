#!/usr/bin/env python3
"""
monitor_and_finish.py — Autonomous monitor for H200 training
=============================================================

Runs on the H200 machine. Checks training progress every 10 minutes.
When training completes:
  1. Checks if any motor has F1 < 0.10
  2. Retrains weak motors with lr=1e-5, patience=3000
  3. Quantizes the final model
  4. Writes a READY flag file

Usage (on H200):
    nohup python3 monitor_and_finish.py > /root/monitor.log 2>&1 &
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/AION-C")
OUTPUT = ROOT / "output" / "production"
TRAIN_LOG = Path("/root/train_stdout.log")
READY_FLAG = ROOT / "output" / "TRAINING_COMPLETE"
CHECK_INTERVAL = 600  # 10 minutes


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def is_training_running():
    """Check if train_h200.py is still running."""
    r = subprocess.run(["pgrep", "-f", "train_h200"], capture_output=True, text=True)
    return r.returncode == 0


def get_last_phase_info():
    """Parse the training log to find current phase and progress."""
    if not TRAIN_LOG.exists():
        return "unknown", {}

    lines = TRAIN_LOG.read_text(encoding="utf-8", errors="replace").split("\n")

    last_phase = "unknown"
    last_monitor = {}
    phases_done = []

    for line in lines:
        if "PHASE" in line and "done:" in line.lower():
            phases_done.append(line.strip())
        if "Phase" in line and "done:" in line:
            phases_done.append(line.strip())
        if "[Monitor] step=" in line:
            parts = line.split()
            for p in parts:
                if p.startswith("step="):
                    last_monitor["step"] = p.split("=")[1].rstrip(",")
                if p.startswith("val_loss="):
                    last_monitor["val_loss"] = p.split("=")[1].rstrip(",")
                if p.startswith("val_F1="):
                    last_monitor["val_f1"] = p.split("=")[1].rstrip(",")
        if "PHASE 0" in line and "done" not in line.lower():
            last_phase = "Phase 0"
        if "PHASE 1" in line and "done" not in line.lower():
            last_phase = "Phase 1"
        if "PHASE 2" in line and "done" not in line.lower():
            last_phase = "Phase 2"
        if "PHASE 3" in line and "done" not in line.lower():
            last_phase = "Phase 3"
        if "PHASE 4" in line and "done" not in line.lower():
            last_phase = "Phase 4"
        if "Phase2-" in line and "steps_max" not in line:
            for motor in ["CORA", "FORGE_C", "MUSE", "AXIOM", "EMPATHY"]:
                if motor in line:
                    last_phase = f"Phase 2 ({motor})"

    return last_phase, {
        "last_monitor": last_monitor,
        "phases_done": phases_done[-5:] if phases_done else [],
    }


def get_motor_f1_scores():
    """Extract per-motor F1 from Phase 2 monitor logs."""
    f1_scores = {}
    for motor in ["cora", "forge_c", "muse", "axiom", "empathy"]:
        log_path = OUTPUT / f"phase2_{motor}_monitor.jsonl"
        if not log_path.exists():
            continue
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        if lines:
            try:
                last = json.loads(lines[-1])
                f1_scores[motor] = last.get("val_f1", 0.0)
            except json.JSONDecodeError:
                pass
    return f1_scores


def retrain_weak_motors(weak_motors):
    """Retrain motors with F1 < 0.10."""
    log(f"Retraining weak motors: {weak_motors}")

    sys.path.insert(0, str(ROOT))
    os.chdir(str(ROOT))

    import torch
    from experiments.train_production import (
        build_pipeline_and_tok, load_all_datasets,
        run_phase2_motor, _TrainHparams, unfreeze,
    )
    from experiments.training_utils import save_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Phase 3 or Phase 4 checkpoint (best available)
    for ckpt_name in ["phase4_instruction.pt", "phase3_final.pt", "phase1.pt"]:
        ckpt_path = OUTPUT / ckpt_name
        if ckpt_path.exists():
            break

    log(f"Loading checkpoint: {ckpt_path}")
    pipeline, tok, mose_cfg = build_pipeline_and_tok("production", device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    pipeline.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    pipeline.to(device)

    # Load datasets
    ds_root = ROOT / "datasets" / "opus"
    datasets = load_all_datasets(max_examples=20000, eval_size=300, dataset_root=ds_root)

    # Hparams for retraining: lower lr, more patience
    hparams = _TrainHparams(
        ph2_steps=999999,
        ph2_lr=1e-5,  # very low lr for fine-tuning
        ph2_warmup=100,
        ph2_eval=200,
        ph2_patience=3000,
        ph2_conv_win=1500,
        conv_delta=0.0003,
        train_batch_size=4,
        grad_accum_steps=1,
        n_train=20000,
        n_val=300,
        max_seq=128,
        ckpt_every=500,
    )

    retrain_dir = OUTPUT / "retrain"
    retrain_dir.mkdir(exist_ok=True)

    for motor_name in weak_motors:
        if motor_name not in datasets:
            log(f"  Skipping {motor_name} — no dataset")
            continue
        train_ds, val_ds = datasets[motor_name]
        log(f"  Retraining {motor_name} (lr=1e-5, patience=3000)...")
        try:
            r = run_phase2_motor(
                pipeline=pipeline, cfg=mose_cfg, tok=tok,
                motor_name=motor_name, train_ds=train_ds, val_ds=val_ds,
                hparams=hparams, device=device, checkpoint_dir=retrain_dir,
            )
            log(f"  {motor_name} done: steps={r.steps_run}, loss={r.final_loss:.4f}, "
                f"val_loss={r.best_val_loss:.4f}, F1={r.best_val_f1:.3f}")
        except Exception as e:
            log(f"  {motor_name} FAILED: {e}")

    # Save final checkpoint
    final_path = OUTPUT / "phase_retrain_final.pt"
    save_checkpoint(final_path, pipeline,
                    torch.optim.AdamW(pipeline.parameters(), lr=1e-5),
                    None, 0, 0.0)
    log(f"Retrained checkpoint saved: {final_path}")

    del pipeline
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def quantize_final():
    """Quantize the best checkpoint."""
    log("Quantizing final model to INT4...")
    sys.path.insert(0, str(ROOT))
    os.chdir(str(ROOT))

    import torch
    from inference.quantize import quantize_state_dict

    # Find best checkpoint
    for name in ["phase_retrain_final.pt", "phase4_instruction.pt", "phase3_final.pt"]:
        p = OUTPUT / name
        if p.exists():
            log(f"  Quantizing {p.name}")
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state", ckpt)
            quantized, stats = quantize_state_dict(sd, group_size=128)
            out = OUTPUT / "production_int4.pt"
            torch.save({"quantized_state": quantized, "config_name": "production"}, str(out))
            size_mb = out.stat().st_size / 1e6
            log(f"  Saved: {out} ({size_mb:.1f} MB)")
            return
    log("  No checkpoint found for quantization")


def write_ready_flag():
    """Write flag file indicating training is complete."""
    info = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(OUTPUT),
    }
    # List all files in output
    files = []
    for f in sorted(OUTPUT.rglob("*")):
        if f.is_file():
            files.append({"path": str(f.relative_to(OUTPUT)),
                         "size_mb": f.stat().st_size / 1e6})
    info["files"] = files

    READY_FLAG.write_text(json.dumps(info, indent=2))
    log(f"READY flag written: {READY_FLAG}")


def main():
    log("=" * 60)
    log("  AION-C Autonomous Monitor")
    log(f"  Checking every {CHECK_INTERVAL // 60} minutes")
    log("=" * 60)

    while True:
        running = is_training_running()
        phase, info = get_last_phase_info()

        if running:
            monitor = info.get("last_monitor", {})
            step = monitor.get("step", "?")
            vl = monitor.get("val_loss", "?")
            f1 = monitor.get("val_f1", "?")
            log(f"Training running | {phase} | step={step} | val_loss={vl} | F1={f1}")

            # Show phases done
            for pd in info.get("phases_done", [])[-3:]:
                log(f"  {pd[:120]}")

            time.sleep(CHECK_INTERVAL)
            continue

        # Training finished!
        log("=" * 60)
        log("  TRAINING COMPLETED!")
        log("=" * 60)

        # Show final phases
        for pd in info.get("phases_done", []):
            log(f"  {pd[:120]}")

        # Check motor F1 scores
        f1_scores = get_motor_f1_scores()
        log(f"Motor F1 scores: {f1_scores}")

        weak = [m for m, f1 in f1_scores.items() if f1 < 0.10]
        if weak:
            log(f"Weak motors (F1 < 0.10): {weak}")
            try:
                retrain_weak_motors(weak)
            except Exception as e:
                log(f"Retrain failed: {e}")
                import traceback; traceback.print_exc()
        else:
            log("All motors F1 >= 0.10 — no retraining needed")

        # Quantize
        try:
            quantize_final()
        except Exception as e:
            log(f"Quantization failed: {e}")

        # Write ready flag
        write_ready_flag()

        # List output files
        log("\nOutput files:")
        total_size = 0
        for f in sorted(OUTPUT.rglob("*")):
            if f.is_file():
                s = f.stat().st_size / 1e6
                total_size += s
                log(f"  {str(f.relative_to(OUTPUT)):<45s} {s:>8.1f} MB")
        log(f"  {'TOTAL':<45s} {total_size:>8.1f} MB")

        log("\n  DONE — Training complete, output ready for download")
        log("  The TRAINING_COMPLETE flag has been written")
        break


if __name__ == "__main__":
    main()
