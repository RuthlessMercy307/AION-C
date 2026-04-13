"""
experiments/diagnose_backward.py
=================================
Diagnóstico del backward del MoSE pipeline en CPU.

Identifica el origen de SliceBackward0/SelectBackward0 en el CRE trazando:
  - Cuántas veces se ejecuta el crystallizer
  - Stats del BatchedGraph (nodos, aristas)
  - Top 30 ops del profiler (forward + backward)
  - Conteo de slice/select/index/gather ops

Uso:
    cd AION-C
    python -m experiments.diagnose_backward
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.autograd.profiler import profile, record_function

from router.pipeline import MoSEPipeline, MoSEConfig
from synth.causal_graph_gen import CausalGraphGenerator
from cre import PyGStyleBatcher

# -----------------------------------------------------------------------------
# 1. Pipeline en CPU
# -----------------------------------------------------------------------------

device = torch.device("cpu")
cfg    = MoSEConfig.tiny()
print(f"Config: hidden_dim={cfg.hidden_dim}, vocab_size={cfg.vocab_size}, "
      f"max_seq_len={cfg.dec_max_seq_len}, K={cfg.motor_max_nodes}")

pipeline = MoSEPipeline(cfg).to(device)
print(f"Pipeline params: {pipeline.count_parameters():,}")

# -----------------------------------------------------------------------------
# 2. Generar 8 ejemplos CORA
# -----------------------------------------------------------------------------

PAD, BOS, EOS = 0, 1, 2

def encode(text: str, max_len: int) -> list:
    ids = [BOS] + [(ord(c) % (cfg.vocab_size - 3)) + 3 for c in text] + [EOS]
    return ids[:max_len]

gen   = CausalGraphGenerator()
seqs  = []
for _ in range(8):
    ex   = gen.generate(level=1)
    text = ex.problem_text + " " + ex.answer
    seqs.append(encode(text, cfg.dec_max_seq_len))

maxl      = max(len(s) for s in seqs)
padded    = [s + [PAD] * (maxl - len(s)) for s in seqs]
token_ids = torch.tensor(padded, dtype=torch.long, device=device)
print(f"\nBatch: B={token_ids.shape[0]}, L={token_ids.shape[1]}")

# -----------------------------------------------------------------------------
# 6. Instrumentar crystallizer con contador de llamadas
# -----------------------------------------------------------------------------

crystallizer_calls: dict[str, int] = {}

for motor_name, motor in pipeline.motors.items():
    orig_fn = motor.build_graph
    def _patched(concepts, _name=motor_name, _orig=orig_fn):
        crystallizer_calls[_name] = crystallizer_calls.get(_name, 0) + 1
        print(f"  [CRYSTALLIZER] motor={_name} llamado "
              f"(total calls este motor: {crystallizer_calls[_name]})")
        return _orig(concepts)
    motor.build_graph = _patched  # type: ignore[method-assign]

# -----------------------------------------------------------------------------
# 3 + 4. Forward local con captura de BatchedGraph
# -----------------------------------------------------------------------------

_batcher         = PyGStyleBatcher()
captured_batched: list = []   # guarda cada BatchedGraph creado

def local_batched_forward(token_ids: torch.Tensor, query_text: str | None = None):
    B = token_ids.shape[0]
    D = cfg.hidden_dim
    K = cfg.motor_max_nodes
    dtype = pipeline.encoder.token_embedding.weight.dtype

    # -- Encoder --------------------------------------------------------------
    with record_function("ENCODER_forward"):
        concepts = pipeline.encoder(token_ids)          # [B, L, D]

    # -- Orchestrator ---------------------------------------------------------
    orch_out = pipeline.orchestrator(concepts, query_text)
    print(f"  Orchestrator -> active motors: {orch_out.motor_names}")

    # -- Crystallizer (no_grad — refleja el notebook actual) -------------------
    motor_cryst: dict = {}
    for act in orch_out.activations:
        with torch.no_grad():
            motor_cryst[act.motor_name] = \
                pipeline.motors[act.motor_name].build_graph(concepts)

    # -- CRE batching ----------------------------------------------------------
    motor_cre_outs: dict = {}
    for act in orch_out.activations:
        motor     = pipeline.motors[act.motor_name]
        cryst_out = motor_cryst[act.motor_name]

        graphs_b: list = []
        node_feats_b: list = []
        valid_b: list = []

        for b in range(B):
            n = cryst_out.node_counts[b]
            if n > 0:
                graphs_b.append(cryst_out.graphs[b])
                node_feats_b.append(
                    cryst_out.node_vectors[b, :n].detach().requires_grad_(True))
                valid_b.append(b)

        if not graphs_b:
            motor_cre_outs[act.motor_name] = [None] * B
            continue

        batched = _batcher.batch(graphs_b, node_feats_b)
        captured_batched.append((act.motor_name, batched))

        # -- 4. Imprimir stats del BatchedGraph -----------------------------
        print(f"  BatchedGraph [{act.motor_name}]: "
              f"n_graphs={batched.n_graphs}, "
              f"n_nodes={batched.n_nodes}, "
              f"n_edges={batched.n_edges}, "
              f"edges/graph={batched.n_edges/batched.n_graphs:.1f}")

        with record_function("CRE_forward"):
            cre_outs = motor.cre.forward_batched(
                batched, n_iterations=act.n_iterations)

        cre_per_b = [None] * B
        for i, b in enumerate(valid_b):
            cre_per_b[b] = cre_outs[i]
        motor_cre_outs[act.motor_name] = cre_per_b

    # -- graph_repr + unifier -------------------------------------------------
    all_graph_reprs = []
    for b in range(B):
        motor_reprs = []
        for act in orch_out.activations:
            motor   = pipeline.motors[act.motor_name]
            cre_out = motor_cre_outs[act.motor_name][b]
            if cre_out is None:
                motor_reprs.append(
                    torch.zeros(K, D, device=device, dtype=dtype))
            else:
                motor_reprs.append(motor.get_graph_repr(cre_out, k_nodes=K))
        all_graph_reprs.append(pipeline.unifier(motor_reprs).unified)

    graph_repr = torch.stack(all_graph_reprs, dim=0)   # [B, K, D]

    # -- Decoder ---------------------------------------------------------------
    with record_function("DECODER_forward"):
        dec_out = pipeline.decoder(token_ids, graph_repr, concepts)

    return dec_out

# -----------------------------------------------------------------------------
# 5. Ejecutar con profiler (forward + backward)
# -----------------------------------------------------------------------------

print("\n" + "=" * 68)
print("  PROFILER — forward + backward (1 step, CPU)")
print("=" * 68)

pipeline.train()

with profile(use_cuda=False, with_stack=False, record_shapes=False) as prof:
    out = local_batched_forward(token_ids, "cora")

    with record_function("LOSS"):
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, cfg.vocab_size),
            token_ids[:, 1:].reshape(-1),
            ignore_index=PAD,
        )

    loss.backward()

# -----------------------------------------------------------------------------
# 6. Crystallizer check
# -----------------------------------------------------------------------------

print("\n" + "-" * 68)
print("CRYSTALLIZER CALLS:")
if crystallizer_calls:
    for name, count in crystallizer_calls.items():
        print(f"  motor={name}: {count} call(s)  "
              f"{'<- crystallizer activo (no precompute)' if count > 0 else ''}")
else:
    print("  0 calls — crystallizer NO ejecutado (precompute activo)")

# -----------------------------------------------------------------------------
# 5. Top 30 ops
# -----------------------------------------------------------------------------

print("\n" + "-" * 68)
print("TOP 30 OPS por cpu_time_total (forward + backward):")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

# -----------------------------------------------------------------------------
# 7. Conteo de slice / select / index / gather
# -----------------------------------------------------------------------------

TARGET_KEYWORDS = ["slice", "select", "index", "gather"]

print("-" * 68)
print("OPS tipo slice/select/index/gather:")
print(f"  {'Operación':<45} {'Llamadas':>9} {'CPU total ms':>14}")
print(f"  {'-'*70}")

found: list[tuple[str, int, float]] = []
for evt in prof.key_averages():
    name_l = evt.key.lower()
    if any(kw in name_l for kw in TARGET_KEYWORDS):
        found.append((evt.key, evt.count, evt.cpu_time_total / 1_000))

found.sort(key=lambda x: -x[2])

for op, count, ms in found:
    print(f"  {op:<45} {count:>9,} {ms:>13.1f}ms")

if not found:
    print("  (ninguna op encontrada)")

print(f"\n  TOTAL ops de este tipo : {sum(c for _, c, _ in found):,}")
print(f"  TOTAL tiempo (ms)      : {sum(t for _, _, t in found):.1f}ms")

# -----------------------------------------------------------------------------
# Resumen final
# -----------------------------------------------------------------------------

print("\n" + "=" * 68)
if captured_batched:
    for motor_name, bg in captured_batched:
        print(f"  [{motor_name}] BatchedGraph final: "
              f"{bg.n_nodes} nodos, {bg.n_edges} aristas, "
              f"{bg.n_graphs} grafos")
print(f"  Loss: {loss.item():.4f}")
print("=" * 68)
