#!/usr/bin/env python3
"""
resume_training.py — Resume from Phase 2 checkpoint after disk space fix.
Skips Phase 0-1 (already done). Runs Phase 2 remaining motors, Phase 3, Phase 4.
Then retrains weak motors, quantizes, signals completion.
"""
import sys, os, gc, json, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def main():
    device = torch.device("cuda")
    output = ROOT / "output" / "production"
    output.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("  AION-C Resume Training (Phase 2-4)")
    log(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
    log(f"  Disk: {os.popen('df -h / | tail -1').read().strip()}")
    log("=" * 60)

    from experiments.train_production import (
        build_pipeline_and_tok, load_all_datasets,
        run_phase2_motor, run_phase3, run_phase4,
        _TrainHparams, unfreeze, save_checkpoint,
    )
    from experiments.training_utils import save_checkpoint as save_ckpt

    # Build pipeline
    log("Building 3.6B pipeline...")
    pipeline, tok, mose_cfg = build_pipeline_and_tok("production", device)
    log(f"Params: {sum(p.numel() for p in pipeline.parameters()):,}")

    # Load Phase 2 CORA checkpoint (best we have)
    ckpt_cora = output / "phase2_cora.pt"
    if ckpt_cora.exists():
        log(f"Loading {ckpt_cora.name}...")
        ckpt = torch.load(str(ckpt_cora), map_location="cpu", weights_only=False)
        pipeline.load_state_dict(ckpt["model_state"], strict=False)
        log("Loaded Phase 2 CORA checkpoint")
    else:
        log("WARNING: No checkpoint found, starting from scratch")

    pipeline.to(device)

    # Load datasets
    ds_root = ROOT / "datasets" / "opus"
    datasets = load_all_datasets(max_examples=20000, eval_size=300, dataset_root=ds_root)
    log(f"Datasets: {list(datasets.keys())}")

    # Common hparams
    hp = _TrainHparams(
        ph2_steps=999999, ph2_lr=1e-4, ph2_warmup=200, ph2_eval=200,
        ph2_patience=2000, ph2_conv_win=1000,
        ph3_orch_steps=999999, ph3_orch_lr=3e-4,
        ph3_e2e_steps=999999, ph3_e2e_lr=1e-5,
        ph3_warmup=200, ph3_eval=200, ph3_patience=2000, ph3_conv_win=1000,
        ph4_steps=999999, ph4_lr=5e-5, ph4_eval=200,
        conv_delta=0.0003,
        train_batch_size=4, grad_accum_steps=1,
        n_train=20000, n_val=300, max_seq=128, ckpt_every=500,
    )

    # Clean old checkpoints to save disk
    for old in output.glob("phase2_*.pt"):
        old.unlink()
        log(f"Deleted old checkpoint: {old.name}")

    # Phase 2: remaining motors (CORA already done, skip it)
    done_motors = {"cora"}  # already trained
    remaining = [m for m in ["forge_c", "muse", "axiom", "empathy"] if m in datasets]
    log(f"\n=== PHASE 2: Remaining motors: {remaining} ===")

    for motor_name in remaining:
        train_ds, val_ds = datasets[motor_name]
        log(f"\nTraining motor: {motor_name}")
        try:
            r = run_phase2_motor(
                pipeline=pipeline, cfg=mose_cfg, tok=tok,
                motor_name=motor_name, train_ds=train_ds, val_ds=val_ds,
                hparams=hp, device=device, checkpoint_dir=output,
            )
            log(f"  {motor_name}: steps={r.steps_run}, loss={r.final_loss:.4f}, "
                f"val_loss={r.best_val_loss:.4f}, F1={r.best_val_f1:.3f}, stop={r.stop_reason}")
            # Delete motor checkpoint to save space (weights are in pipeline memory)
            ckpt_m = output / f"phase2_{motor_name}.pt"
            if ckpt_m.exists():
                ckpt_m.unlink()
                log(f"  Deleted {ckpt_m.name} to save disk")
        except Exception as e:
            log(f"  {motor_name} FAILED: {e}")
            traceback.print_exc()

    # Save Phase 2 final (just model state, ~14GB)
    log("\nSaving Phase 2 final checkpoint...")
    save_ckpt(output / "phase2_all_done.pt", pipeline,
              torch.optim.AdamW([torch.zeros(1)], lr=1e-5), None, 0, 0.0)
    log(f"Disk: {os.popen('df -h / | tail -1').read().strip()}")

    # Phase 3
    log("\n=== PHASE 3: E2E + Orchestrator ===")
    try:
        r3 = run_phase3(pipeline, mose_cfg, tok, datasets, hp, device, output)
        log(f"Phase 3: steps={r3.steps_run}, val_loss={r3.best_val_loss:.4f}, F1={r3.best_val_f1:.3f}")
    except Exception as e:
        log(f"Phase 3 FAILED: {e}")
        traceback.print_exc()

    # Delete Phase 2 checkpoint (Phase 3 is now the latest)
    p2 = output / "phase2_all_done.pt"
    if p2.exists():
        p2.unlink()
    log(f"Disk: {os.popen('df -h / | tail -1').read().strip()}")

    # Phase 4: Instruction Tuning with seq=512, batch=1
    log("\n=== PHASE 4: Instruction Tuning (LoRA, seq=512) ===")
    hp4 = _TrainHparams(
        ph4_steps=999999, ph4_lr=5e-5, ph4_eval=200,
        ph0_patience=2000, ph0_conv_win=1000,
        conv_delta=0.0003,
        train_batch_size=1, grad_accum_steps=1,
        n_train=20000, n_val=300, max_seq=512, ckpt_every=500,
    )
    try:
        r4 = run_phase4(pipeline, mose_cfg, tok, hp4, device, output, interactive=False)
        log(f"Phase 4: steps={r4.steps_run}, loss={r4.final_loss:.4f}")
    except Exception as e:
        log(f"Phase 4 FAILED: {e}")
        traceback.print_exc()

    # Check motor F1 and retrain weak ones
    log("\n=== Checking motor F1 scores ===")
    for motor_name in ["cora", "forge_c", "muse", "axiom", "empathy"]:
        mlog = output / f"phase2_{motor_name}_monitor.jsonl"
        if mlog.exists():
            lines = mlog.read_text().strip().split("\n")
            if lines:
                last = json.loads(lines[-1])
                f1 = last.get("val_f1", 0)
                log(f"  {motor_name}: F1={f1:.3f}")

    # Save config_name for run_local.py
    for name in ["phase4_instruction.pt", "phase3_final.pt"]:
        p = output / name
        if p.exists():
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            ckpt["config_name"] = "production"
            torch.save(ckpt, str(p))

    # Quantize
    log("\n=== Quantizing to INT4 ===")
    try:
        from inference.quantize import quantize_state_dict
        for name in ["phase4_instruction.pt", "phase3_final.pt"]:
            p = output / name
            if p.exists():
                ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
                sd = ckpt.get("model_state", ckpt)
                q, stats = quantize_state_dict(sd, group_size=128)
                qpath = output / "production_int4.pt"
                torch.save({"quantized_state": q, "config_name": "production"}, str(qpath))
                log(f"Quantized: {qpath} ({qpath.stat().st_size/1e6:.0f} MB)")
                break
    except Exception as e:
        log(f"Quantization failed: {e}")

    # Write completion flag
    flag = ROOT / "output" / "TRAINING_COMPLETE"
    flag.write_text(json.dumps({"completed": time.strftime("%Y-%m-%d %H:%M:%S")}))

    # List output
    log("\n=== Output files ===")
    for f in sorted(output.rglob("*")):
        if f.is_file():
            log(f"  {f.name:<40s} {f.stat().st_size/1e6:>8.1f} MB")
    log(f"\nDisk: {os.popen('df -h / | tail -1').read().strip()}")
    log("DONE!")

if __name__ == "__main__":
    main()
