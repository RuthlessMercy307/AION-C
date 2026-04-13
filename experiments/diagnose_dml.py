"""
experiments/diagnose_dml.py — Diagnostico de cuelgues en DirectML.
Prueba secuencialmente. El ultimo print visible indica donde cuelga.

Uso:
    cd AION-C
    python -m experiments.diagnose_dml 2>&1 | tee diagnose_dml.log
    (Ctrl-C si cuelga; el log muestra el ultimo print que salio)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Device ───────────────────────────────────────────────────────────────────

try:
    import torch_directml
    DML = torch_directml.device()
    _t = torch.randn(4, 4).to(DML)
    _ = (_t @ _t).sum().item()
    print(f"[DML OK] device={DML}  name={torch_directml.device_name(0)}", flush=True)
except Exception as e:
    print(f"[DML FAIL] {e} -- usando CPU", flush=True)
    DML = torch.device("cpu")

ON_DML = str(DML) != "cpu"

# Patch F.gelu -> F.silu for DML compatibility
if ON_DML:
    def _gelu_via_silu(input, approximate="none"):
        return F.silu(input)
    F.gelu = _gelu_via_silu
    print("[PATCH] F.gelu -> F.silu", flush=True)

# ── Utils ─────────────────────────────────────────────────────────────────────

SEP = "-" * 60

def run(label, fn, *args, **kwargs):
    print(f"  {label}...", end=" ", flush=True)
    t0 = time.perf_counter()
    r = fn(*args, **kwargs)
    print(f"OK ({(time.perf_counter()-t0)*1000:.0f}ms)", flush=True)
    return r

def sync(tensor, label=""):
    """Read scalar from DML tensor to CPU, forcing GPU sync."""
    val = tensor.flatten()[0].item()
    if label:
        print(f"  [sync {label}] val={val:.6f}", flush=True)
    return val

# ── Pipeline ─────────────────────────────────────────────────────────────────

from router.pipeline import MoSEPipeline, MoSEConfig

cfg = MoSEConfig.tiny()
print(f"Config: hidden={cfg.hidden_dim} enc_layers={cfg.enc_n_layers} "
      f"dec_layers={cfg.dec_n_layers} seq_len={cfg.dec_max_seq_len}", flush=True)

pipeline = MoSEPipeline(cfg).to(DML)

B = 4

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0: DML WARMUP
# Run simple ops to trigger DML shader compilation for basic patterns.
# DML compiles shaders JIT on first use -- warmup prevents surprises in tests.
# ═══════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("0. DML WARMUP -- compilar shaders para patrones basicos")
print(SEP)

# Warmup: run a simple matmul + silu + layernorm to compile those shaders
_w_x = torch.randn(B, 64, device=DML)
_w_lin = nn.Linear(64, 64).to(DML)
_w_ln = nn.LayerNorm(64).to(DML)
_w_y = _w_ln(F.silu(_w_lin(_w_x)))
sync(_w_y, "warmup Linear+SiLU+LN")

# Warmup: SSM-style ops (the scan loop creates many small element-wise ops)
for L_warm in [4, 8, 16, 32]:
    _ids = torch.randint(0, cfg.vocab_size, (B, L_warm), dtype=torch.long, device=DML)
    pipeline.eval()
    with torch.no_grad():
        _c = pipeline.encoder(_ids)
    sync(_c, f"encoder warmup L={L_warm}")
    print(f"  encoder warmup L={L_warm} OK", flush=True)

print("Warmup completo", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("0b. MICRO-TESTS: ops especificas del orchestrator")
print(SEP)

# Test mean(dim=1) on [B, L, D] — what the orchestrator does with concepts
_x_BLD = torch.randn(B, 32, 64, device=DML)
_pooled = _x_BLD.mean(dim=1)  # [B, D]
sync(_pooled, "mean(dim=1) [4,32,64]->[4,64]")
print(f"  mean(dim=1): {_pooled.shape} OK", flush=True)

_query = _pooled.mean(dim=0)  # [D]
sync(_query, "mean(dim=0) [4,64]->[64]")
print(f"  mean(dim=0): {_query.shape} OK", flush=True)

# Test softmax on [5]
_logits_test = torch.randn(5, device=DML)
_scores_test = F.softmax(_logits_test, dim=-1)
sync(_scores_test, "softmax [5]")
_scores_list = _scores_test.tolist()
print(f"  softmax+tolist [5]: {[f'{x:.3f}' for x in _scores_list]} OK", flush=True)

# NOTE: nn.Linear with 1D [D] input hangs on DML (no shader for rank-1 matmul).
# The orchestrator was fixed to use keepdim=True (always 2D [1, D] input).

# Test: MLP on 2D tensor [1, D] (keepdim=True version — the fixed orchestrator path)
_orch_cls_2d = nn.Sequential(
    nn.Linear(64, 32), nn.SiLU(), nn.LayerNorm(32),
    nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 5),
).to(DML)
_q2d = torch.randn(1, 64, device=DML)  # 2D tensor — rank 2
sync(_q2d, "fresh 2D [1,64]")
_lo2 = _orch_cls_2d(_q2d)
sync(_lo2, "MLP(2D) [1,5]")
print(f"  MLP on 2D [1,64]: {_lo2.shape} OK", flush=True)

# Test: use mean(dim=0, keepdim=True) to keep 2D shape
_qk = _x_BLD.mean(dim=1).mean(dim=0, keepdim=True)  # [1, 64] — 2D
sync(_qk, "mean keepdim [1,64]")
_lok = _orch_cls_2d(_qk)
sync(_lok, "MLP(mean_keepdim) [1,5]")
print(f"  MLP on mean(keepdim) [1,64]: {_lok.shape} OK", flush=True)

print("Micro-tests OK", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("1. ENCODER + SYNC + ORCHESTRATOR (sin backward previo)")
print(SEP)

L = 32
ids_1 = torch.randint(0, cfg.vocab_size, (B, L), dtype=torch.long, device=DML)
pipeline.eval()
with torch.no_grad():
    concepts_1 = run("encoder eval+no_grad", pipeline.encoder, ids_1)
    sync(concepts_1, "encoder output")  # sync antes de orchestrator
    orch_1 = run("orchestrator eval+no_grad", pipeline.orchestrator, concepts_1)
    print(f"  activations={[x.motor_name for x in orch_1.activations]}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("2. ENCODER + ORCHESTRATOR (sin sync entre ellos)")
print(SEP)

ids_2 = torch.randint(0, cfg.vocab_size, (B, L), dtype=torch.long, device=DML)
pipeline.eval()
with torch.no_grad():
    concepts_2 = run("encoder eval+no_grad", pipeline.encoder, ids_2)
    # NO sync here -- test if shaders are cached after warmup
    orch_2 = run("orchestrator eval+no_grad (no sync)", pipeline.orchestrator, concepts_2)
    print(f"  activations={[x.motor_name for x in orch_2.activations]}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("3. FULL PIPELINE eval+no_grad")
print(SEP)

ids_3 = torch.randint(0, cfg.vocab_size, (B, L), dtype=torch.long, device=DML)
pipeline.eval()
with torch.no_grad():
    out_3 = run("pipeline eval+no_grad", pipeline, ids_3)
    print(f"  logits={out_3.logits.shape}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("4. FULL PIPELINE train+grad + backward + sync + step")
print(SEP)

pipeline.train()
opt = torch.optim.AdamW(pipeline.parameters(), lr=1e-4)

for step_i in range(1, 4):
    ids_s = torch.randint(0, cfg.vocab_size, (B, L), dtype=torch.long, device=DML)
    out_s = run(f"  step {step_i} forward", pipeline, ids_s)
    loss_s = F.cross_entropy(
        out_s.logits[:, :-1].reshape(-1, cfg.vocab_size),
        ids_s[:, 1:].reshape(-1),
        ignore_index=0,
    )
    run(f"  step {step_i} backward", loss_s.backward)
    # DML sync: read grad norm to CPU so backward completes before next forward
    grad_norm = nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
    _ = float(grad_norm)   # <-- forces DML sync (reads scalar to CPU)
    print(f"  step {step_i} grad_norm={float(grad_norm):.4f}", flush=True)
    run(f"  step {step_i} opt.step", opt.step)
    opt.zero_grad(set_to_none=True)
    print(f"  loss={loss_s.item():.4f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("DIAGNOSTICO COMPLETO -- todos los tests pasaron")
print(SEP)
