#!/usr/bin/env python3
"""Phase 3 + Phase 4 only. Fast. No checkpoints except final."""
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
    output.mkdir(parents=True, exist_ok=True)

    from experiments.train_production import (
        build_pipeline_and_tok, load_all_datasets,
        run_phase3, run_phase4, _TrainHparams,
    )
    from experiments.training_utils import save_checkpoint

    log("Building 3.6B pipeline...")
    pipeline, tok, cfg = build_pipeline_and_tok("production", device)

    # Load best checkpoint
    ckpt_path = output / "phase2_axiom.pt"
    log(f"Loading {ckpt_path.name}...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    pipeline.load_state_dict(ckpt["model_state"], strict=False)
    pipeline.to(device)
    del ckpt; gc.collect(); torch.cuda.empty_cache()

    ds_root = ROOT / "datasets" / "opus"
    datasets = load_all_datasets(max_examples=20000, eval_size=300, dataset_root=ds_root)
    log(f"Datasets: {list(datasets.keys())}")

    # Phase 3: seq=128 batch=4, fast
    log("\n=== PHASE 3 ===")
    hp3 = _TrainHparams(
        ph3_orch_steps=999999, ph3_orch_lr=3e-4,
        ph3_e2e_steps=999999, ph3_e2e_lr=1e-5,
        ph3_warmup=200, ph3_eval=200, ph3_patience=1500, ph3_conv_win=800,
        conv_delta=0.0003,
        train_batch_size=4, grad_accum_steps=1,
        n_train=20000, n_val=300, max_seq=128, ckpt_every=99999,
    )
    r3 = run_phase3(pipeline, cfg, tok, datasets, hp3, device, output)
    log(f"Phase 3: steps={r3.steps_run} val_loss={r3.best_val_loss:.4f} F1={r3.best_val_f1:.3f} stop={r3.stop_reason}")

    # Delete old checkpoint
    old = output / "phase2_axiom.pt"
    if old.exists(): old.unlink(); log("Deleted phase2_axiom.pt")
    log(f"Disk: {os.popen('df -h / | tail -1').read().strip()}")

    # Phase 4: seq=256 batch=1
    log("\n=== PHASE 4 ===")
    hp4 = _TrainHparams(
        ph4_steps=999999, ph4_lr=5e-5, ph4_eval=200,
        ph0_patience=1500, ph0_conv_win=800,
        conv_delta=0.0003,
        train_batch_size=1, grad_accum_steps=1,
        n_train=20000, n_val=300, max_seq=256, ckpt_every=500,
    )
    r4 = run_phase4(pipeline, cfg, tok, hp4, device, output, interactive=False)
    log(f"Phase 4: steps={r4.steps_run} loss={r4.final_loss:.4f} stop={r4.stop_reason}")

    # Tag checkpoints
    for name in ["phase4_instruction.pt", "phase3_final.pt"]:
        p = output / name
        if p.exists():
            c = torch.load(str(p), map_location="cpu", weights_only=False)
            c["config_name"] = "production"
            torch.save(c, str(p))

    # Quantize
    log("\nQuantizing...")
    from inference.quantize import quantize_state_dict
    for name in ["phase4_instruction.pt", "phase3_final.pt"]:
        p = output / name
        if p.exists():
            c = torch.load(str(p), map_location="cpu", weights_only=False)
            q, _ = quantize_state_dict(c["model_state"], group_size=128)
            qp = output / "production_int4.pt"
            torch.save({"quantized_state": q, "config_name": "production"}, str(qp))
            log(f"Saved {qp.name} ({qp.stat().st_size/1e6:.0f} MB)")
            break

    (ROOT / "output" / "TRAINING_COMPLETE").write_text(
        json.dumps({"completed": time.strftime("%Y-%m-%d %H:%M:%S")}))

    log("\nOutput:")
    for f in sorted(output.rglob("*")):
        if f.is_file():
            log(f"  {f.name:<40s} {f.stat().st_size/1e6:>8.1f} MB")
    log("DONE!")

if __name__ == "__main__":
    main()
