#!/usr/bin/env python3
"""Quick verify: load aion_1b_direct.pt and run the 6 eval prompts cleanly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONIOENCODING"] = "utf-8"
import torch

from router.pipeline import MoSEPipeline, MoSEConfig
from experiments.train_production import build_tokenizer

PROMPTS = [
    "Hola, quien eres?",
    "If rain causes floods, what happens?",
    "Write a Python function to add two numbers",
    "What is 25% of 200?",
    "Mi amigo esta triste porque perdio su trabajo",
    "Write a short poem about the moon",
]

tok = build_tokenizer(32_000)
cfg = MoSEConfig(
    hidden_dim=1024, vocab_size=tok.vocab_size,
    enc_n_layers=12, enc_state_dim=16, enc_expand=2, enc_d_conv=4, enc_ffn_mult=4,
    orch_mlp_hidden=512, orch_max_motors=3, orch_min_confidence=0.3,
    motor_max_nodes=8, motor_n_heads=8, motor_threshold=0.01, unif_n_heads=8,
    dec_n_layers=16, dec_n_heads=8, dec_max_seq_len=512,
    dec_state_dim=16, dec_expand=2, dec_d_conv=4, dec_ffn_mult=4,
)
pipe = MoSEPipeline(cfg)
ck = torch.load("checkpoints/aion_1b_direct.pt", map_location="cpu", weights_only=False)
pipe.load_state_dict(ck["model_state"], strict=False)
print(f"Loaded step={ck['step']}  val_loss={ck['val_loss']:.4f}  config={ck.get('config_name')}")
pipe.eval()

EOS = 2
for p in PROMPTS:
    try: enc = tok.encode(p, 96)
    except TypeError: enc = tok.encode(p)[:96]
    cur = torch.tensor([enc], dtype=torch.long)
    pl = len(enc)
    with torch.no_grad():
        for _ in range(60):
            o = pipe(cur)
            nxt = int(o.logits[0, -1].argmax().item())
            if nxt in (0, EOS): break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
            if cur.shape[1] >= 160: break
    pred = tok.decode(cur[0, pl:].tolist()) if cur.shape[1] > pl else "(empty)"
    routed = o.active_motors[0] if hasattr(o, "active_motors") and o.active_motors else "?"
    print(f"  [{routed:>8}] {p[:38]:<38} -> {pred[:70]}")
