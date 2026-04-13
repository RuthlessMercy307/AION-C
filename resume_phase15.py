#!/usr/bin/env python3
"""
resume_phase15.py — Phase 1.5 (mixed retrain) + Phase 3 + Phase 4
==================================================================

Phase 1.5: Unfreeze last 6 decoder layers + weak motors (CORA, FORGE_C, AXIOM).
           Train with MIXED data from all 5 domains to avoid catastrophic forgetting.
Phase 3:   Orchestrator routing + E2E fine-tune.
Phase 4:   Instruction Tuning with LoRA, seq=256.
"""
import sys, os, gc, json, time, math, random, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn as nn
import torch.nn.functional as F


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    device = torch.device("cuda")
    output = ROOT / "output" / "production"
    output.mkdir(parents=True, exist_ok=True)

    log("=" * 65)
    log("  AION-C Phase 1.5 + Phase 3 + Phase 4")
    log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
    log(f"  Disk: {os.popen('df -h / | tail -1').read().strip()}")
    log("=" * 65)

    from experiments.train_production import (
        build_pipeline_and_tok, load_all_datasets,
        run_phase3, run_phase4,
        _TrainHparams, _phase_loop, _encode,
        unfreeze, freeze, count_trainable,
        make_mixed_ids, make_ids_list, get_fixed_examples,
        PAD,
    )
    from experiments.training_utils import (
        TrainingMonitor, make_cosine_scheduler, save_checkpoint,
    )

    # Build pipeline
    log("Building 3.6B pipeline...")
    pipeline, tok, mose_cfg = build_pipeline_and_tok("production", device)
    total_params = sum(p.numel() for p in pipeline.parameters())
    log(f"Params: {total_params:,}")

    # Load best available checkpoint
    for ckpt_name in ["phase3_final.pt", "phase2_axiom.pt", "phase2_empathy.pt"]:
        ckpt_path = output / ckpt_name
        if ckpt_path.exists():
            log(f"Loading {ckpt_name}...")
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            pipeline.load_state_dict(ckpt["model_state"], strict=False)
            log(f"Loaded {ckpt_name}")
            break
    else:
        log("WARNING: No checkpoint found!")

    pipeline.to(device)

    # Load datasets
    ds_root = ROOT / "datasets" / "opus"
    datasets = load_all_datasets(max_examples=20000, eval_size=300, dataset_root=ds_root)
    log(f"Datasets: {list(datasets.keys())}")

    # ================================================================
    # PHASE 1.5: Mixed retrain with unfrozen decoder tail
    # ================================================================
    log("")
    log("=" * 65)
    log("  PHASE 1.5: Mixed retrain — last 6 decoder layers + weak motors")
    log("=" * 65)

    # Freeze everything first
    freeze(pipeline)

    # Unfreeze weak motors
    weak_motors = ["cora", "forge_c", "axiom"]
    for name, param in pipeline.named_parameters():
        for motor in weak_motors:
            if f"motors.{motor}" in name:
                param.requires_grad_(True)

    # Unfreeze last 6 decoder layers (layers 22-27 of 28)
    decoder_layers = list(pipeline.decoder.layers)
    n_layers = len(decoder_layers)
    unfreeze_from = max(0, n_layers - 6)
    for i in range(unfreeze_from, n_layers):
        for p in decoder_layers[i].parameters():
            p.requires_grad_(True)
    # Also unfreeze decoder final norm + lm_head for output adaptation
    for p in pipeline.decoder.final_norm.parameters():
        p.requires_grad_(True)
    # lm_head shares weights with embedding, unfreeze it
    pipeline.decoder.lm_head.weight.requires_grad_(True)

    trainable = count_trainable(pipeline)
    log(f"Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")
    log(f"Decoder layers unfrozen: {unfreeze_from}-{n_layers-1} (last 6 of {n_layers})")
    log(f"Weak motors unfrozen: {weak_motors}")

    # Mixed data: all 5 domains, weighted toward weak motors
    # 50% weak (CORA/FORGE_C/AXIOM), 50% strong (MUSE/EMPATHY)
    hp15 = _TrainHparams(
        ph1_steps=999999, ph1_lr=3e-4, ph1_warmup=200,
        ph1_eval=200, ph1_patience=1500, ph1_conv_win=800,
        conv_delta=0.0003,
        train_batch_size=4, grad_accum_steps=1,
        n_train=20000, n_val=300, max_seq=128, ckpt_every=99999,
    )

    # Build mixed data with weighting
    ids_weak = []
    ids_strong = []
    for motor in ["cora", "forge_c", "axiom"]:
        if motor in datasets:
            train_ds, _ = datasets[motor]
            ids_weak.extend(make_ids_list(train_ds, tok, n=5000, max_len=128))
    for motor in ["muse", "empathy"]:
        if motor in datasets:
            train_ds, _ = datasets[motor]
            ids_strong.extend(make_ids_list(train_ds, tok, n=2500, max_len=128))

    all_ids = ids_weak + ids_strong
    random.Random(42).shuffle(all_ids)
    log(f"Mixed training data: {len(ids_weak)} weak + {len(ids_strong)} strong = {len(all_ids)} total")

    # Val data
    val_ids = make_mixed_ids(datasets, tok, n=300, max_len=128)
    fixed_examples = get_fixed_examples(datasets)

    # Monitor
    monitor = TrainingMonitor(
        model=pipeline, tok=tok,
        val_ids_list=val_ids[:50],
        fixed_examples=fixed_examples,
        cfg=mose_cfg,
        eval_every=hp15.ph1_eval,
        patience=hp15.ph1_patience,
        convergence_delta=hp15.conv_delta,
        convergence_window=hp15.ph1_conv_win,
        log_path=output / "phase15_monitor.jsonl",
        device=device, is_motor=True,
    )

    # Optimizer: only trainable params
    trainable_params = [p for p in pipeline.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=hp15.ph1_lr, weight_decay=1e-2)
    sched = make_cosine_scheduler(opt, hp15.ph1_warmup, hp15.ph1_steps)

    use_amp = True
    amp_dtype = torch.bfloat16

    rng = random.Random(42)
    def get_ids():
        return rng.choice(all_ids)

    log(f"LR={hp15.ph1_lr:.1e}  patience={hp15.ph1_patience}  eval_every={hp15.ph1_eval}")
    log("Starting Phase 1.5...")

    losses, elapsed, final_loss, stop_reason = _phase_loop(
        model=pipeline,
        get_ids=get_ids,
        optimizer=opt,
        scheduler=sched,
        n_steps=hp15.ph1_steps,
        monitor=monitor,
        label="Phase1.5-Mixed",
        device=device,
        checkpoint_path=None,  # no intermediate checkpoints
        ckpt_every=99999,
        print_every=50,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        grad_accum_steps=hp15.grad_accum_steps,
        batch_size=hp15.train_batch_size,
    )

    log(f"Phase 1.5 done: steps={len(losses)}, loss={final_loss:.4f}, stop={stop_reason}")

    # Save Phase 1.5 checkpoint
    save_checkpoint(output / "phase15_final.pt", pipeline, opt, sched, len(losses), final_loss)
    log(f"Saved phase15_final.pt")
    log(f"Disk: {os.popen('df -h / | tail -1').read().strip()}")

    # Unfreeze everything for Phase 3
    unfreeze(pipeline)

    # ================================================================
    # PHASE 3
    # ================================================================
    log("")
    log("=" * 65)
    log("  PHASE 3: Orchestrator routing + E2E fine-tune")
    log("=" * 65)

    hp3 = _TrainHparams(
        ph3_orch_steps=999999, ph3_orch_lr=3e-4,
        ph3_e2e_steps=999999, ph3_e2e_lr=1e-5,
        ph3_warmup=200, ph3_eval=200, ph3_patience=1500, ph3_conv_win=800,
        conv_delta=0.0003,
        train_batch_size=4, grad_accum_steps=1,
        n_train=20000, n_val=300, max_seq=128, ckpt_every=99999,
    )

    try:
        r3 = run_phase3(pipeline, mose_cfg, tok, datasets, hp3, device, output)
        log(f"Phase 3: steps={r3.steps_run}, val_loss={r3.best_val_loss:.4f}, F1={r3.best_val_f1:.3f}")
    except Exception as e:
        log(f"Phase 3 FAILED: {e}")
        traceback.print_exc()

    # Delete old checkpoints to save disk
    for old in ["phase15_final.pt", "phase2_axiom.pt", "phase2_empathy.pt"]:
        p = output / old
        if p.exists():
            p.unlink()
            log(f"Deleted {old}")
    log(f"Disk: {os.popen('df -h / | tail -1').read().strip()}")

    # ================================================================
    # PHASE 4: Instruction Tuning with LoRA, seq=256
    # ================================================================
    log("")
    log("=" * 65)
    log("  PHASE 4: Instruction Tuning (LoRA rank=16, seq=256)")
    log("=" * 65)

    hp4 = _TrainHparams(
        ph4_steps=999999, ph4_lr=5e-5, ph4_eval=200,
        ph0_patience=1500, ph0_conv_win=800,
        conv_delta=0.0003,
        train_batch_size=1, grad_accum_steps=1,
        n_train=20000, n_val=300, max_seq=256, ckpt_every=99999,
    )

    try:
        r4 = run_phase4(pipeline, mose_cfg, tok, hp4, device, output, interactive=False)
        log(f"Phase 4: steps={r4.steps_run}, loss={r4.final_loss:.4f}")
    except Exception as e:
        log(f"Phase 4 FAILED: {e}")
        traceback.print_exc()

    # Save config_name for run_local.py
    for name in ["phase4_instruction.pt", "phase3_final.pt"]:
        p = output / name
        if p.exists():
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            ckpt["config_name"] = "production"
            torch.save(ckpt, str(p))

    # Quantize
    log("")
    log("Quantizing to INT4...")
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

    # Completion flag
    flag = ROOT / "output" / "TRAINING_COMPLETE"
    flag.write_text(json.dumps({"completed": time.strftime("%Y-%m-%d %H:%M:%S")}))

    # List output
    log("")
    log("=== Output files ===")
    for f in sorted(output.rglob("*")):
        if f.is_file():
            log(f"  {f.name:<45s} {f.stat().st_size/1e6:>8.1f} MB")
    log(f"Disk: {os.popen('df -h / | tail -1').read().strip()}")
    log("DONE!")


if __name__ == "__main__":
    main()
