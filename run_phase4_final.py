#!/usr/bin/env python3
"""Phase 4 only: Instruction Tuning with LoRA. Then quantize. Then done."""
import sys, os, gc, json, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONIOENCODING"] = "utf-8"
import torch

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    device = torch.device("cuda")
    output = ROOT / "output" / "production"

    from experiments.train_production import (
        build_pipeline_and_tok, run_phase4, _TrainHparams,
    )
    from experiments.training_utils import save_checkpoint

    log("Building 3.6B pipeline...")
    pipeline, tok, cfg = build_pipeline_and_tok("production", device)

    ckpt_path = output / "phase2_axiom.pt"
    log(f"Loading {ckpt_path.name}...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    pipeline.load_state_dict(ckpt["model_state"], strict=False)
    pipeline.to(device)
    del ckpt; gc.collect(); torch.cuda.empty_cache()
    log(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    log("\n=== PHASE 4: Instruction Tuning (LoRA rank=16, seq=256) ===")
    hp4 = _TrainHparams(
        ph4_steps=999999, ph4_lr=5e-5, ph4_eval=200,
        ph0_patience=1500, ph0_conv_win=800,
        conv_delta=0.0003,
        train_batch_size=1, grad_accum_steps=1,
        n_train=20000, n_val=300, max_seq=256, ckpt_every=500,
    )
    r4 = run_phase4(pipeline, cfg, tok, hp4, device, output, interactive=False)
    log(f"Phase 4: steps={r4.steps_run} loss={r4.final_loss:.4f} stop={r4.stop_reason}")

    # Tag checkpoint
    for name in ["phase4_instruction.pt"]:
        p = output / name
        if p.exists():
            c = torch.load(str(p), map_location="cpu", weights_only=False)
            c["config_name"] = "production"
            torch.save(c, str(p))
            log(f"Tagged {name}")

    # Delete old checkpoint
    old = output / "phase2_axiom.pt"
    if old.exists(): old.unlink(); log("Deleted phase2_axiom.pt")

    # Quantize
    log("\nQuantizing to INT4...")
    from inference.quantize import quantize_state_dict
    p4 = output / "phase4_instruction.pt"
    if p4.exists():
        c = torch.load(str(p4), map_location="cpu", weights_only=False)
        q, stats = quantize_state_dict(c["model_state"], group_size=128)
        qp = output / "production_int4.pt"
        torch.save({"quantized_state": q, "config_name": "production"}, str(qp))
        log(f"INT4: {qp.name} ({qp.stat().st_size/1e6:.0f} MB)")

    (ROOT / "output" / "TRAINING_COMPLETE").write_text(
        json.dumps({"completed": time.strftime("%Y-%m-%d %H:%M:%S")}))

    log("\nOutput files:")
    for f in sorted(output.rglob("*")):
        if f.is_file():
            log(f"  {f.name:<40s} {f.stat().st_size/1e6:>8.1f} MB")
    log(f"Disk: {os.popen('df -h / | tail -1').read().strip()}")
    log("ALL DONE!")

if __name__ == "__main__":
    main()
