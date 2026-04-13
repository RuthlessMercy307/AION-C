"""
experiments/_patch_nb3.py
=========================
Regenera benchmark_cora_vs_transformer.ipynb con:
  - Bundle base64 actualizado (incluye cre/weakness.py, cre/convergence.py,
    y todos los archivos modificados desde _patch_nb2)
  - Cell 5 (CORA): use_convergence_gate=True + tracking de iteraciones CRE
  - Cell 6 (comparison): fila de iteraciones promedio del CRE en tabla
  - Cell 0 (title): mención de ConvergenceGate
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import json, base64, re, pathlib, textwrap

NB_PATH   = pathlib.Path('benchmark_cora_vs_transformer.ipynb')
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent   # AION-C/

# ── Archivos a actualizar en el bundle ────────────────────────────────────────
# (todos los que cambiaron desde la última regeneración + los nuevos)
FILES_TO_UPDATE = [
    # Nuevos (no existían en el bundle)
    'cre/weakness.py',
    'cre/convergence.py',
    # Modificados
    'cre/config.py',
    'cre/engine.py',
    'cre/__init__.py',
    'crystallizer/node_detector.py',
    'decoder/hybrid_layer.py',
    'decoder/model.py',
    'router/pipeline.py',
]

# ── Helper: re-encode one file ────────────────────────────────────────────────
def encode_file(file_path: pathlib.Path) -> str:
    """Returns the multi-line base64 block string (without outer parens)."""
    raw   = file_path.read_bytes()
    b64   = base64.b64encode(raw).decode('ascii')
    lines = textwrap.wrap(b64, 76)
    return "(\n        '" + "'\n        '".join(lines) + "'\n    )"


def update_existing_key(cell_src: str, bundle_key: str, file_path: pathlib.Path) -> str:
    """Replace the base64 block for an existing key in the bundle."""
    escaped = re.escape(bundle_key)
    pattern = rf"('{escaped}':\s*)\([^)]*?\)"
    replacement = r'\g<1>' + encode_file(file_path)
    new_src, n = re.subn(pattern, replacement, cell_src, count=1, flags=re.DOTALL)
    if n == 0:
        raise ValueError(f"Key '{bundle_key}' not found in bundle for update")
    return new_src


def insert_new_key(cell_src: str, after_key: str, new_key: str, file_path: pathlib.Path) -> str:
    """Insert a new key/block into the bundle dict right after `after_key`."""
    escaped_after = re.escape(after_key)
    # Find the end of the after_key block: last '),' after the key
    m = re.search(rf"'{escaped_after}':\s*\(.*?\),", cell_src, re.DOTALL)
    if not m:
        raise ValueError(f"Anchor key '{after_key}' not found for insertion")
    insert_pos = m.end()
    new_entry = f"\n    '{new_key}': {encode_file(file_path)},"
    return cell_src[:insert_pos] + new_entry + cell_src[insert_pos:]


# ── Load notebook ─────────────────────────────────────────────────────────────
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell2 = ''.join(nb['cells'][2]['source'])

# Detect which keys are new vs existing
existing_keys = set(re.findall(r"'([a-zA-Z_/]+\.py)':\s*\(", cell2))
print(f'Bundle has {len(existing_keys)} existing keys')

# ── Apply bundle updates ───────────────────────────────────────────────────────
for bundle_key in FILES_TO_UPDATE:
    file_path = REPO_ROOT / bundle_key
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    if bundle_key in existing_keys:
        cell2 = update_existing_key(cell2, bundle_key, file_path)
        print(f'  updated : {bundle_key}')
    else:
        # Insert new keys in the right position
        if bundle_key == 'cre/weakness.py':
            cell2 = insert_new_key(cell2, 'cre/scratch_pad.py', bundle_key, file_path)
        elif bundle_key == 'cre/convergence.py':
            cell2 = insert_new_key(cell2, 'cre/weakness.py', bundle_key, file_path)
        else:
            raise ValueError(f"Don't know where to insert new key: {bundle_key}")
        print(f'  inserted: {bundle_key}')

# ── Verify round-trips ────────────────────────────────────────────────────────
print('\nVerifying round-trips...')
for bundle_key in FILES_TO_UPDATE:
    escaped = re.escape(bundle_key)
    m = re.search(rf"'{escaped}':\s*\((.*?)\)", cell2, re.DOTALL)
    if not m:
        raise ValueError(f"Key '{bundle_key}' missing after update")
    b64_raw = re.sub(r"[\s']", '', m.group(1))
    decoded = base64.b64decode(b64_raw).decode('utf-8')
    disk    = (REPO_ROOT / bundle_key).read_text(encoding='utf-8')
    if decoded != disk:
        raise ValueError(f"Round-trip mismatch for {bundle_key}")
    print(f'  OK: {bundle_key}')

nb['cells'][2]['source'] = cell2

# ── Cell 0: title ─────────────────────────────────────────────────────────────
nb['cells'][0]['source'] = (
    '# CORA 5M vs Transformer Baseline \u2014 Benchmark mismos datos (T4)\n\n'
    'Dos arquitecturas, **exactamente los mismos 2000 ejemplos**, batch=1 sin batching.\n\n'
    '**Fix de grounding léxico**: cada `HybridDecoderLayer` tiene doble cross-attention:\n'
    '1. al grafo causal (estructura) 2. a los concept vectors del encoder (identidad léxica).\n\n'
    '**ConvergenceGate activo en CORA**: el CRE para de iterar cuando el grafo converge.\n'
    'Se reporta el promedio de iteraciones usadas — queries simples deben converger en 1-2 iters.\n\n'
    '| Modelo | Arquitectura | Batch | Strategy |\n'
    '|--------|-------------|-------|----------|\n'
    '| **Transformer** | 4enc+4dec est\u00e1ndar | 1 (igual que CORA) | Secuencial FP16 |\n'
    '| **CORA 5M** | Mamba+Crystallizer+CRE(gate)+Decoder(dual cross-attn) | 1 | Secuencial FP16 |\n\n'
    'Pregunta: dados los **mismos datos**, **\u00bfqui\u00e9n aprende mejor?**\n'
    'Bonus: \u00bfcu\u00e1ntas iteraciones necesita CORA de media?\n'
)

# ── Cell 5: CORA with use_convergence_gate=True + iteration tracking ───────────
nb['cells'][5]['source'] = """\
import math, time, random, torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

torch.cuda.empty_cache()
_free0,_=torch.cuda.mem_get_info()
print(f'[CORA] VRAM libre: {_free0/1e9:.2f} GB')

# \u2500\u2500 Imports del pipeline AION-C \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
from encoder          import StreamEncoder
from crystallizer     import GraphCrystallizer
from cre             import CausalReasoningEngine
from decoder         import StreamDecoder
from router.pipeline import CORAConfig

# \u2500\u2500 Construir CORA con CORAConfig (use_convergence_gate=True) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# cre_max_iterations=5: cap conservador; el gate para antes si converge.
# cre_min_iterations=1: siempre al menos 1 iteraci\u00f3n completa.
CORA_CFG = CORAConfig(
    hidden_dim   = 256,
    vocab_size   = ACTUAL_VOCAB,
    # Encoder: Mamba SSM, 4 capas
    enc_n_layers  = 4, enc_state_dim = 16, enc_expand = 2,
    enc_d_conv    = 4, enc_ffn_mult  = 4,
    # Crystallizer
    cryst_max_nodes      = 32,  cryst_n_heads        = 8,
    cryst_node_threshold = 0.01, cryst_edge_threshold = 0.01,
    # CRE: ConvergenceGate activo
    cre_edge_dim          = 64,  cre_message_dim      = 128,
    cre_n_message_layers  = 2,   cre_max_iterations   = 5,
    cre_use_convergence_gate = True,  # parada adaptativa
    cre_min_iterations       = 1,     # safety floor
    # ScratchPad
    pad_n_slots  = 16, pad_slot_dim = 128,
    # Decoder: 2 capas
    dec_n_layers    = 2, dec_n_heads     = 8,
    dec_max_seq_len = 256, dec_state_dim = 16,
    dec_expand      = 2,  dec_d_conv     = 4, dec_ffn_mult = 4,
)

enc  = StreamEncoder(CORA_CFG.encoder_config()).to(DEVICE)
crys = GraphCrystallizer(CORA_CFG.crystallizer_config()).to(DEVICE)
cre  = CausalReasoningEngine(CORA_CFG.cre_config()).to(DEVICE)
dec  = StreamDecoder(CORA_CFG.decoder_config()).to(DEVICE)

all_params = (list(enc.parameters()) + list(crys.parameters()) +
              list(cre.parameters())  + list(dec.parameters()))
n_params   = sum(p.numel() for p in all_params)
cora_opt   = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=1e-2)
cora_scaler= GradScaler(enabled=USE_AMP)
print(f'[CORA] {n_params:,} params')
print(f'[CORA] ConvergenceGate: max_iter={CORA_CFG.cre_max_iterations}, '
      f'min_iter={CORA_CFG.cre_min_iterations}')

K_NODES = CORA_CFG.cryst_max_nodes
D_MODEL = CORA_CFG.hidden_dim

# \u2500\u2500 Forward pass completo: q \u2192 encoder \u2192 crystallizer \u2192 CRE \u2192 decoder \u2500\u2500\u2500\u2500\u2500\u2500
def cora_forward(src_ids, tgt_ids):
    \"\"\"
    Pipeline seq2seq con ConvergenceGate activo.
    Devuelve (logits [1, L-1, V], loss, n_iters) o (None, None, 0).
    \"\"\"
    src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    tgt = torch.tensor(tgt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    # 1. Encode pregunta \u2192 concept vectors [1, L_q, D]
    concepts = enc(src)

    # 2. Crystallizer: concept vectors \u2192 CrystallizerOutput
    crystal = crys(concepts)
    n_valid = crystal.node_counts[0]   # int

    # 3. CRE: refinar node features (ConvergenceGate decide cu\u00e1ndo parar)
    if n_valid == 0:
        cre_feats  = torch.zeros(1, D_MODEL, device=DEVICE, dtype=concepts.dtype)
        n_iters    = 0
    else:
        node_feats = crystal.node_vectors[0, :n_valid, :]   # [n_valid, D]
        cre_out    = cre(crystal.graphs[0], node_feats)     # CREOutput
        cre_feats  = cre_out.node_features                  # [n_valid, D]
        n_iters    = cre_out.iterations_run                 # int

    # 4. Construir graph_repr [1, K, D] con padding
    n = cre_feats.shape[0]
    if n == 0:
        padded = torch.zeros(K_NODES, D_MODEL, device=DEVICE, dtype=concepts.dtype)
    elif n >= K_NODES:
        padded = cre_feats[:K_NODES]
    else:
        pad    = torch.zeros(K_NODES - n, D_MODEL, device=DEVICE, dtype=concepts.dtype)
        padded = torch.cat([cre_feats, pad], dim=0)
    graph_repr = padded.unsqueeze(0)                        # [1, K, D]

    # 5. Decoder con teacher forcing + grounding léxico
    dec_input = tgt[:, :-1]                                 # [1, L-1]
    dec_out   = dec(dec_input, graph_repr, concepts)        # DecoderOutput
    logits    = dec_out.logits                              # [1, L-1, V]

    loss = F.cross_entropy(
        logits.reshape(-1, ACTUAL_VOCAB),
        tgt[:, 1:].reshape(-1),
        ignore_index=0,
    )
    return logits, loss, n_iters

# \u2500\u2500 Evaluaci\u00f3n \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def evaluate_cora():
    enc.eval(); crys.eval(); cre.eval(); dec.eval()
    losses=[]; wf1s=[]; iters=[]
    with torch.no_grad():
        for ex in EVAL_EX[:100]:
            src_ids=SHARED_VOCAB.encode(ex.problem_text,max_len=_MAX_Q,add_bos=True,add_eos=True)
            tgt_ids=SHARED_VOCAB.encode(ex.answer,max_len=_MAX_A,add_bos=True,add_eos=True)
            with autocast(enabled=USE_AMP):
                logits, loss, n_i = cora_forward(src_ids, tgt_ids)
            if loss is None:
                losses.append(9.9); wf1s.append(0.0); continue
            losses.append(loss.item())
            pred = logits.argmax(-1).squeeze(0).tolist()
            wf1s.append(word_f1(pred, tgt_ids[1:]))
            if n_i > 0: iters.append(n_i)
    enc.train(); crys.train(); cre.train(); dec.train()
    avg_iters = sum(iters)/len(iters) if iters else 0.0
    return sum(losses)/len(losses), sum(wf1s)/len(wf1s), avg_iters

# \u2500\u2500 Training: exactamente TOTAL_STEPS=2000 ejemplos, batch=1 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
CORA_EVALS  = []   # [(step, val_loss, val_wf1, avg_iters)]
CORA_LOSSES = []
CORA_ITERS  = []   # iteraciones CRE por step de training
train_cycle = TRAIN_EX.copy()
random.shuffle(train_cycle)
train_idx = 0

t_start = time.perf_counter()
for step in range(1, TOTAL_STEPS + 1):
    ex = train_cycle[train_idx % len(train_cycle)]
    train_idx += 1

    src_ids = SHARED_VOCAB.encode(ex.problem_text, max_len=_MAX_Q, add_bos=True, add_eos=True)
    tgt_ids = SHARED_VOCAB.encode(ex.answer,       max_len=_MAX_A, add_bos=True, add_eos=True)

    cora_opt.zero_grad()
    with autocast(enabled=USE_AMP):
        _, loss, n_iters = cora_forward(src_ids, tgt_ids)

    if loss is None:
        CORA_LOSSES.append(0.0)
        continue

    cora_scaler.scale(loss).backward()
    cora_scaler.unscale_(cora_opt)
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    cora_scaler.step(cora_opt); cora_scaler.update()
    CORA_LOSSES.append(loss.item())
    if n_iters > 0: CORA_ITERS.append(n_iters)

    if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:
        elapsed = time.perf_counter() - t_start
        val_loss, val_wf1, avg_i = evaluate_cora()
        CORA_EVALS.append((step, val_loss, val_wf1, avg_i))
        avg_loss = sum(CORA_LOSSES[-EVAL_EVERY:]) / max(1, len(CORA_LOSSES[-EVAL_EVERY:]))
        avg_iters_train = (sum(CORA_ITERS[-EVAL_EVERY:]) /
                           max(1, len(CORA_ITERS[-EVAL_EVERY:])))
        print(f'[CORA] step={step:>4}/{TOTAL_STEPS}  t={elapsed:5.1f}s  '
              f'train={avg_loss:.4f}  val={val_loss:.4f}  '
              f'wf1={val_wf1:.3f}  iters={avg_iters_train:.2f}')

CORA_TOTAL_TIME = time.perf_counter() - t_start
CORA_FINAL_LOSS = CORA_EVALS[-1][1]
CORA_FINAL_WF1  = CORA_EVALS[-1][2]
CORA_AVG_ITERS  = sum(CORA_ITERS) / max(1, len(CORA_ITERS))
CORA_EX_PER_SEC = TOTAL_STEPS / CORA_TOTAL_TIME
print(f'\\n[CORA] COMPLETADO')
print(f'  tiempo      : {CORA_TOTAL_TIME:.1f}s  ({CORA_EX_PER_SEC:.1f} ex/s)')
print(f'  loss        : {CORA_FINAL_LOSS:.4f}   wf1: {CORA_FINAL_WF1:.3f}')
print(f'  iters CRE   : {CORA_AVG_ITERS:.2f} promedio  (max={CORA_CFG.cre_max_iterations})')
"""

# ── Cell 6: comparison with CRE iterations row ────────────────────────────────
nb['cells'][6]['source'] = """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# \u2500\u2500 Tabla resumen \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
speed_ratio = TF_EX_PER_SEC / max(CORA_EX_PER_SEC, 0.01)
wf1_winner  = 'Transformer' if TF_FINAL_WF1  >= CORA_FINAL_WF1  else 'CORA'
loss_winner = 'Transformer' if TF_FINAL_LOSS <= CORA_FINAL_LOSS else 'CORA'

print('=' * 68)
print(f'  BENCHMARK: {TOTAL_STEPS} ejemplos identicos, batch=1, sin batching')
print('=' * 68)
print(f'  {"Metric":<28} {"Transformer":>16} {"CORA 5M":>16}')
print(f'  {"-"*28} {"-"*16} {"-"*16}')
print(f'  {"Ejemplos vistos":<28} {TOTAL_STEPS:>16,} {TOTAL_STEPS:>16,}')
print(f'  {"Tiempo total (s)":<28} {TF_TOTAL_TIME:>16.1f} {CORA_TOTAL_TIME:>16.1f}')
print(f'  {"Throughput (ex/s)":<28} {TF_EX_PER_SEC:>16.1f} {CORA_EX_PER_SEC:>16.1f}')
print(f'  {"Loss final (val)":<28} {TF_FINAL_LOSS:>16.4f} {CORA_FINAL_LOSS:>16.4f}')
print(f'  {"Word F1 final":<28} {TF_FINAL_WF1:>16.3f} {CORA_FINAL_WF1:>16.3f}')
print(f'  {"Iters CRE promedio":<28} {"N/A":>16} {CORA_AVG_ITERS:>16.2f}')
print('=' * 68)
print(f'  Transformer es {speed_ratio:.1f}x mas rapido que CORA')
print(f'  Mejor calidad (loss)    : {loss_winner}')
print(f'  Mejor calidad (Word F1) : {wf1_winner}')
iters_pct = (1.0 - CORA_AVG_ITERS / CORA_CFG.cre_max_iterations) * 100
print(f'  Gate ahorro compute CRE : {iters_pct:.1f}% menos iters vs max={CORA_CFG.cre_max_iterations}')
print('=' * 68)

# \u2500\u2500 Gr\u00e1ficas: loss y Word F1 por step \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

tf_steps   = [e[0] for e in TF_EVALS]
tf_losses  = [e[1] for e in TF_EVALS]
tf_wf1s    = [e[2] for e in TF_EVALS]
cora_steps = [e[0] for e in CORA_EVALS]
cora_losses= [e[1] for e in CORA_EVALS]
cora_wf1s  = [e[2] for e in CORA_EVALS]
cora_iters_by_step = [e[3] for e in CORA_EVALS]

ax = axes[0]
ax.plot(tf_steps,   tf_losses,   'b-o', label='Transformer', linewidth=2)
ax.plot(cora_steps, cora_losses, 'r-s', label='CORA 5M',     linewidth=2)
ax.set_xlabel('Step (= ejemplos vistos)'); ax.set_ylabel('Val Loss')
ax.set_title(f'Loss vs Step (mismos {TOTAL_STEPS} ej.)')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(tf_steps,   tf_wf1s,   'b-o', label='Transformer', linewidth=2)
ax.plot(cora_steps, cora_wf1s, 'r-s', label='CORA 5M',     linewidth=2)
ax.set_xlabel('Step (= ejemplos vistos)'); ax.set_ylabel('Word F1')
ax.set_title('Word F1 vs Step')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('benchmark_vs_steps.png', dpi=120, bbox_inches='tight')
plt.show(); print('[fig] benchmark_vs_steps.png guardado')

# \u2500\u2500 Gr\u00e1fica extra: iteraciones CRE por checkpoint \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig3, ax3 = plt.subplots(figsize=(8, 3))
ax3.plot(cora_steps, cora_iters_by_step, 'g-^', linewidth=2, label='Avg CRE iters (val)')
ax3.axhline(y=CORA_CFG.cre_max_iterations, color='r', linestyle='--',
            alpha=0.5, label=f'max_iterations={CORA_CFG.cre_max_iterations}')
ax3.axhline(y=CORA_CFG.cre_min_iterations, color='b', linestyle='--',
            alpha=0.5, label=f'min_iterations={CORA_CFG.cre_min_iterations}')
ax3.set_xlabel('Step'); ax3.set_ylabel('Avg iterations')
ax3.set_title('CRE ConvergenceGate: iteraciones usadas (menos = queries simples)')
ax3.legend(); ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, CORA_CFG.cre_max_iterations + 0.5)
plt.tight_layout()
plt.savefig('benchmark_cre_iters.png', dpi=120, bbox_inches='tight')
plt.show(); print('[fig] benchmark_cre_iters.png guardado')

# \u2500\u2500 Gr\u00e1ficas vs tiempo real \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
tf_times   = [e[0] / TF_EX_PER_SEC   for e in TF_EVALS]
cora_times = [e[0] / CORA_EX_PER_SEC for e in CORA_EVALS]

ax = axes2[0]
ax.plot(tf_times,   tf_losses,   'b-o', label=f'Transformer ({TF_EX_PER_SEC:.0f} ex/s)')
ax.plot(cora_times, cora_losses, 'r-s', label=f'CORA ({CORA_EX_PER_SEC:.1f} ex/s)')
ax.set_xlabel('Tiempo real (s)'); ax.set_ylabel('Val Loss')
ax.set_title('Loss vs Tiempo real')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes2[1]
ax.plot(tf_times,   tf_wf1s,   'b-o', label=f'Transformer ({TF_EX_PER_SEC:.0f} ex/s)')
ax.plot(cora_times, cora_wf1s, 'r-s', label=f'CORA ({CORA_EX_PER_SEC:.1f} ex/s)')
ax.set_xlabel('Tiempo real (s)'); ax.set_ylabel('Word F1')
ax.set_title('Word F1 vs Tiempo real')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('benchmark_vs_time.png', dpi=120, bbox_inches='tight')
plt.show(); print('[fig] benchmark_vs_time.png guardado')

# \u2500\u2500 5 ejemplos comparativos \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print('\\n' + '='*68)
print('  5 EJEMPLOS COMPARATIVOS')
print('='*68)

tf_model.eval(); enc.eval(); crys.eval(); cre.eval(); dec.eval()
with torch.no_grad():
    for i, ex in enumerate(EVAL_EX[:5]):
        src_ids = SHARED_VOCAB.encode(ex.problem_text, max_len=_MAX_Q, add_bos=True, add_eos=True)
        tgt_ids = SHARED_VOCAB.encode(ex.answer,       max_len=_MAX_A, add_bos=True, add_eos=True)
        gold    = SHARED_VOCAB.decode(tgt_ids[1:])

        # Transformer greedy
        src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        tgt = torch.tensor(tgt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        with autocast(enabled=USE_AMP):
            logits_tf = tf_model(src, tgt[:,:-1], src_key_padding_mask=(src==0))
        tf_pred_ids = logits_tf.argmax(-1).squeeze(0).tolist()
        tf_pred     = SHARED_VOCAB.decode(tf_pred_ids)
        tf_f1       = word_f1(tf_pred_ids, tgt_ids[1:])

        # CORA greedy (teacher forcing para comparacion justa)
        with autocast(enabled=USE_AMP):
            logits_cr, _, n_i = cora_forward(src_ids, tgt_ids)
        cora_pred_ids = logits_cr.argmax(-1).squeeze(0).tolist() if logits_cr is not None else []
        cora_pred     = SHARED_VOCAB.decode(cora_pred_ids)
        cora_f1       = word_f1(cora_pred_ids, tgt_ids[1:])

        print(f'\\nEj {i+1}: {ex.problem_text[:80]}')
        print(f'  Gold       : {gold}')
        print(f'  Transformer: {tf_pred}  [WF1={tf_f1:.2f}]')
        print(f'  CORA       : {cora_pred}  [WF1={cora_f1:.2f}]  [iters={n_i}]')
print('='*68)
"""

# ── Verification ──────────────────────────────────────────────────────────────
c2 = nb['cells'][2]['source']
c5 = nb['cells'][5]['source']
c6 = nb['cells'][6]['source']

checks = [
    # Bundle: new files present
    ('Bundle has cre/weakness.py',      "'cre/weakness.py':"    in c2),
    ('Bundle has cre/convergence.py',   "'cre/convergence.py':" in c2),
    ('Bundle has cre/engine.py',        "'cre/engine.py':"      in c2),
    ('Bundle has cre/config.py',        "'cre/config.py':"      in c2),
    ('Bundle has router/pipeline.py',   "'router/pipeline.py':" in c2),
    # Cell 5: convergence gate
    ('Cell5 use_convergence_gate=True', 'cre_use_convergence_gate = True'      in c5),
    ('Cell5 cre_min_iterations',        'cre_min_iterations'                   in c5),
    ('Cell5 cre_max_iterations=5',      'cre_max_iterations   = 5'             in c5),
    ('Cell5 iterations_run tracked',    'cre_out.iterations_run'               in c5),
    ('Cell5 CORA_ITERS list',           'CORA_ITERS'                           in c5),
    ('Cell5 CORA_AVG_ITERS',            'CORA_AVG_ITERS'                       in c5),
    ('Cell5 returns n_iters',           'return logits, loss, n_iters'         in c5),
    ('Cell5 grounding: concepts to dec','dec(dec_input, graph_repr, concepts)' in c5),
    # Cell 6: iterations row in table
    ('Cell6 Iters row in table',        'Iters CRE promedio'                   in c6),
    ('Cell6 CORA_AVG_ITERS in table',   'CORA_AVG_ITERS'                       in c6),
    ('Cell6 gate savings printed',      'Gate ahorro compute CRE'              in c6),
    ('Cell6 CRE iters plot',            'benchmark_cre_iters.png'              in c6),
    ('Cell6 5 examples iters shown',    '[iters={n_i}]'                        in c6),
]

print('\nVerification:')
all_ok = True
for label, ok in checks:
    mark = 'OK  ' if ok else 'FAIL'
    print(f'  {mark} {label}')
    if not ok: all_ok = False

if all_ok:
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('\nALL OK — notebook guardado')
else:
    print('\nFAILED — notebook NO guardado')
    raise SystemExit(1)
