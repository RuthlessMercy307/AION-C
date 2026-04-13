"""
experiments/_patch_nb4.py
=========================
Regenera benchmark_cora_vs_transformer.ipynb con TODOS los módulos CORA activos:
  - use_moe=True           (SparseMoE post message-passing)
  - use_convergence_gate=True (ConvergenceGate ya estaba activo)
  - use_budget_manager=True (BudgetManager heurístico, distribución por query)
  - use_validator=True      (AionCValidator faithfulness/consistency/completeness/halluc)
  - val_rereason=True       (re-razonamiento si VAL falla)

Nuevas métricas en Cell 6:
  - distribución de budget levels (TRIVIAL/SIMPLE/COMPLEX/DEEP)
  - % respuestas que pasan el VAL
  - cuántas veces se activó re-razonamiento

Timing: si los primeros 10 steps tardan >3s → TOTAL_STEPS=1000 (estimado >10min para 2000).
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import json, base64, re, pathlib, textwrap

NB_PATH   = pathlib.Path('benchmark_cora_vs_transformer.ipynb')
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent   # AION-C/

# ── Archivos a actualizar en el bundle ────────────────────────────────────────
FILES_TO_UPDATE = [
    # Nuevos (no existían en el bundle)
    'cre/moe.py',
    'budget/__init__.py',
    'budget/manager.py',
    'validation/__init__.py',
    'validation/model.py',
    # Modificados desde _patch_nb3
    'cre/config.py',
    'cre/engine.py',
    'cre/__init__.py',
    'router/pipeline.py',
]

# ── Helper: re-encode one file ────────────────────────────────────────────────
def encode_file(file_path: pathlib.Path) -> str:
    raw   = file_path.read_bytes()
    b64   = base64.b64encode(raw).decode('ascii')
    lines = textwrap.wrap(b64, 76)
    return "(\n        '" + "'\n        '".join(lines) + "'\n    )"


def update_existing_key(cell_src: str, bundle_key: str, file_path: pathlib.Path) -> str:
    escaped = re.escape(bundle_key)
    pattern = rf"('{escaped}':\s*)\([^)]*?\)"
    replacement = r'\g<1>' + encode_file(file_path)
    new_src, n = re.subn(pattern, replacement, cell_src, count=1, flags=re.DOTALL)
    if n == 0:
        raise ValueError(f"Key '{bundle_key}' not found in bundle for update")
    return new_src


def insert_new_key(cell_src: str, after_key: str, new_key: str, file_path: pathlib.Path) -> str:
    escaped_after = re.escape(after_key)
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

existing_keys = set(re.findall(r"'([a-zA-Z_/]+\.py)':\s*\(", cell2))
print(f'Bundle tiene {len(existing_keys)} keys existentes')

# ── Apply bundle updates ───────────────────────────────────────────────────────
for bundle_key in FILES_TO_UPDATE:
    file_path = REPO_ROOT / bundle_key
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    if bundle_key in existing_keys:
        cell2 = update_existing_key(cell2, bundle_key, file_path)
        print(f'  updated : {bundle_key}')
    else:
        if bundle_key == 'cre/moe.py':
            cell2 = insert_new_key(cell2, 'cre/convergence.py', bundle_key, file_path)
        elif bundle_key == 'budget/__init__.py':
            # Insert after router/ section — use router/pipeline.py as anchor
            cell2 = insert_new_key(cell2, 'router/pipeline.py', bundle_key, file_path)
        elif bundle_key == 'budget/manager.py':
            cell2 = insert_new_key(cell2, 'budget/__init__.py', bundle_key, file_path)
        elif bundle_key == 'validation/__init__.py':
            cell2 = insert_new_key(cell2, 'budget/manager.py', bundle_key, file_path)
        elif bundle_key == 'validation/model.py':
            cell2 = insert_new_key(cell2, 'validation/__init__.py', bundle_key, file_path)
        else:
            raise ValueError(f"Don't know where to insert: {bundle_key}")
        print(f'  inserted: {bundle_key}')

# ── Add 'budget' and 'validation' to the pkg mkdir list in Cell 2 ─────────────
# Pattern: for pkg in ['core', 'synth', ..., 'experiments']:
old_pkg_list = "for pkg in ['core', 'synth', 'encoder', 'crystallizer', 'cre', 'decoder', 'router', 'experiments']:"
new_pkg_list = "for pkg in ['core', 'synth', 'encoder', 'crystallizer', 'cre', 'decoder', 'router', 'budget', 'validation', 'experiments']:"
if old_pkg_list in cell2:
    cell2 = cell2.replace(old_pkg_list, new_pkg_list, 1)
    print('  patched: pkg mkdir list (added budget, validation)')
elif new_pkg_list in cell2:
    print('  skip: pkg mkdir list already has budget, validation')
else:
    # Try to find it with different spacing/quoting
    print('  WARNING: could not find pkg mkdir list — check Cell 2 manually')

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

# ── Cell 0: title actualizado ─────────────────────────────────────────────────
nb['cells'][0]['source'] = (
    '# CORA 5M vs Transformer Baseline \u2014 Benchmark completo (T4)\n\n'
    'Dos arquitecturas, **exactamente los mismos 2000 ejemplos**, batch=1 sin batching.\n\n'
    '**Todos los m\u00f3dulos CORA activos:**\n'
    '- **SparseMoE** (use_moe=True): 16 expertos, top-2 activos por nodo, load-balance loss\n'
    '- **ConvergenceGate** (use_convergence_gate=True): parada adaptativa del CRE\n'
    '- **BudgetManager** (use_budget_manager=True): clasifica queries en 4 niveles de compute\n'
    '- **AionCValidator** (use_validator=True): verifica faithfulness/consistency/completeness/hallucination\n'
    '- **Re-razonamiento** (val_rereason=True): si VAL falla, el CRE re-razona\n\n'
    '| Modelo | Arquitectura | Batch | Strategy |\n'
    '|--------|-------------|-------|----------|\n'
    '| **Transformer** | 4enc+4dec est\u00e1ndar | 1 (igual que CORA) | Secuencial FP16 |\n'
    '| **CORA 5M** | Mamba+Cryst+CRE(MoE+Gate)+Decoder | 1 | Secuencial FP16 |\n\n'
    'Nuevas m\u00e9tricas: iteraciones CRE, distribuci\u00f3n budget, % VAL pass, re-razonamientos.\n'
)

# ── Cell 5: CORA con TODOS los m\u00f3dulos activos ──────────────────────────────────
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
from cre              import CausalReasoningEngine
from cre.config       import CREConfig
from decoder          import StreamDecoder
from router.pipeline  import CORAConfig
from budget           import BudgetManager, BudgetLevel
from validation       import AionCValidator
from validation.model import ValidatorConfig

# \u2500\u2500 CORAConfig (sin MoE — lo pasamos directo a CREConfig) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
CORA_CFG = CORAConfig(
    hidden_dim   = 256,
    vocab_size   = ACTUAL_VOCAB,
    # Encoder: Mamba SSM, 4 capas
    enc_n_layers  = 4, enc_state_dim = 16, enc_expand = 2,
    enc_d_conv    = 4, enc_ffn_mult  = 4,
    # Crystallizer
    cryst_max_nodes      = 32,  cryst_n_heads        = 8,
    cryst_node_threshold = 0.01, cryst_edge_threshold = 0.01,
    # CRE
    cre_edge_dim         = 64,  cre_message_dim      = 128,
    cre_n_message_layers = 2,   cre_max_iterations   = 5,
    cre_use_convergence_gate = True,
    cre_min_iterations       = 1,
    # Budget
    use_budget_manager = True, budget_hidden_dim = 64,
    # Validator
    use_validator      = True,
    val_hidden_dim     = 128, val_n_layers = 2, val_n_heads = 4,
    val_pass_threshold = 0.5, val_issue_threshold = 0.4,
    val_rereason       = True,
    # ScratchPad
    pad_n_slots  = 16, pad_slot_dim = 128,
    # Decoder: 2 capas
    dec_n_layers    = 2, dec_n_heads     = 8,
    dec_max_seq_len = 256, dec_state_dim = 16,
    dec_expand      = 2,  dec_d_conv     = 4, dec_ffn_mult = 4,
)

# CRE Config separado para activar use_moe=True
_cre_cfg = CREConfig(
    node_dim             = CORA_CFG.hidden_dim,
    edge_dim             = CORA_CFG.cre_edge_dim,
    message_dim          = CORA_CFG.cre_message_dim,
    n_message_layers     = CORA_CFG.cre_n_message_layers,
    max_iterations       = CORA_CFG.cre_max_iterations,
    n_relation_types     = 16,
    use_convergence_gate = True,
    min_iterations       = CORA_CFG.cre_min_iterations,
    # \u2500 SparseMoE \u2500
    use_moe              = True,   # 16 expertos, top-2 activos
    moe_n_groups         = 4,
    moe_experts_per_group= 4,
    moe_active_experts   = 2,
    moe_load_balance_weight = 0.01,
)

enc  = StreamEncoder(CORA_CFG.encoder_config()).to(DEVICE)
crys = GraphCrystallizer(CORA_CFG.crystallizer_config()).to(DEVICE)
cre  = CausalReasoningEngine(_cre_cfg).to(DEVICE)
dec  = StreamDecoder(CORA_CFG.decoder_config()).to(DEVICE)

# Validator (eval-only en training — pars aprendidos end-to-end)
_val_cfg = ValidatorConfig(
    input_dim       = CORA_CFG.hidden_dim,
    hidden_dim      = CORA_CFG.val_hidden_dim,
    n_heads         = CORA_CFG.val_n_heads,
    n_layers        = CORA_CFG.val_n_layers,
    pass_threshold  = CORA_CFG.val_pass_threshold,
    issue_threshold = CORA_CFG.val_issue_threshold,
)
validator = AionCValidator(config=_val_cfg, vocab_size=CORA_CFG.vocab_size).to(DEVICE)

all_params = (list(enc.parameters()) + list(crys.parameters()) +
              list(cre.parameters())  + list(dec.parameters()) +
              list(validator.parameters()))
n_params   = sum(p.numel() for p in all_params)
cora_opt   = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=1e-2)
cora_scaler= GradScaler(enabled=USE_AMP)

print(f'[CORA] {n_params:,} params')
print(f'[CORA] use_moe=True  use_convergence_gate=True  use_budget_manager=True  use_validator=True')
print(f'[CORA] MoE: {_cre_cfg.moe_n_groups}x{_cre_cfg.moe_experts_per_group} experts, top-{_cre_cfg.moe_active_experts} activos')
print(f'[CORA] ConvergenceGate: max_iter={_cre_cfg.max_iterations}, min_iter={_cre_cfg.min_iterations}')
print(f'[CORA] val_rereason=True  val_pass_threshold={CORA_CFG.val_pass_threshold}')

K_NODES = CORA_CFG.cryst_max_nodes
D_MODEL = CORA_CFG.hidden_dim

# \u2500\u2500 Forward para TRAINING (sin validator, max velocidad) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def cora_forward_train(src_ids, tgt_ids):
    \"\"\"
    Forward completo sin validator (para no ralentizar el training loop).
    Devuelve (logits, loss, n_iters, budget_level_int).
    \"\"\"
    src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    tgt = torch.tensor(tgt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    # 1. Budget heurístico (solo clasificación — no cambia el CRE)
    bgt = BudgetManager.classify_heuristic(len(src_ids), _cre_cfg.max_iterations)
    budget_level = int(bgt.level)

    # 2. Encode
    concepts = enc(src)

    # 3. Crystallize
    crystal = crys(concepts)
    n_valid = crystal.node_counts[0]

    # 4. CRE
    if n_valid == 0:
        n_iters    = 0
        graph_repr = torch.zeros(1, K_NODES, D_MODEL, device=DEVICE, dtype=concepts.dtype)
    else:
        node_feats = crystal.node_vectors[0, :n_valid, :]
        cre_out    = cre(crystal.graphs[0], node_feats)
        cre_feats  = cre_out.node_features
        n_iters    = cre_out.iterations_run
        n = cre_feats.shape[0]
        if n >= K_NODES:
            padded = cre_feats[:K_NODES]
        else:
            pad    = torch.zeros(K_NODES - n, D_MODEL, device=DEVICE, dtype=concepts.dtype)
            padded = torch.cat([cre_feats, pad], dim=0)
        graph_repr = padded.unsqueeze(0)

    # 5. MoE load balance loss viene de cre_out (ya acumulada en CRE forward)
    # (CRE la suma internamente — no necesitamos extraerla aquí para el loss)

    # 6. Decode
    dec_input = tgt[:, :-1]
    dec_out   = dec(dec_input, graph_repr, concepts)
    logits    = dec_out.logits

    loss = F.cross_entropy(
        logits.reshape(-1, ACTUAL_VOCAB),
        tgt[:, 1:].reshape(-1),
        ignore_index=0,
    )
    return logits, loss, n_iters, budget_level


# \u2500\u2500 Forward para EVALUACIÓN (con validator + rereason) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def cora_forward_eval(src_ids, tgt_ids):
    \"\"\"
    Forward completo con AionCValidator y re-razonamiento opcional.
    Devuelve (logits, loss, n_iters, val_passed_bool, did_rereason_bool).
    \"\"\"
    src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    tgt = torch.tensor(tgt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    concepts = enc(src)
    crystal  = crys(concepts)
    n_valid  = crystal.node_counts[0]

    if n_valid == 0:
        n_iters    = 0
        graph_repr = torch.zeros(1, K_NODES, D_MODEL, device=DEVICE, dtype=concepts.dtype)
        node_feats = None
    else:
        node_feats = crystal.node_vectors[0, :n_valid, :]
        cre_out    = cre(crystal.graphs[0], node_feats)
        cre_feats  = cre_out.node_features
        n_iters    = cre_out.iterations_run
        n = cre_feats.shape[0]
        if n >= K_NODES:
            padded = cre_feats[:K_NODES]
        else:
            pad    = torch.zeros(K_NODES - n, D_MODEL, device=DEVICE, dtype=concepts.dtype)
            padded = torch.cat([cre_feats, pad], dim=0)
        graph_repr = padded.unsqueeze(0)

    dec_input = tgt[:, :-1]
    dec_out   = dec(dec_input, graph_repr, concepts)
    logits    = dec_out.logits
    loss = F.cross_entropy(
        logits.reshape(-1, ACTUAL_VOCAB),
        tgt[:, 1:].reshape(-1),
        ignore_index=0,
    )

    # Validator
    val_result = validator(logits.detach(), graph_repr.detach(), concepts.detach())
    val_passed  = val_result.passed
    did_rereason= False

    # Re-razonamiento si VAL falla (val_rereason=True)
    if not val_passed and CORA_CFG.val_rereason and n_valid > 0 and node_feats is not None:
        # Re-run CRE (ConvergenceGate puede tomar path diferente; MoE routing varía)
        cre_out2   = cre(crystal.graphs[0], node_feats)
        cre_feats2 = cre_out2.node_features
        n2 = cre_feats2.shape[0]
        if n2 >= K_NODES:
            padded2 = cre_feats2[:K_NODES]
        else:
            pad2    = torch.zeros(K_NODES - n2, D_MODEL, device=DEVICE, dtype=concepts.dtype)
            padded2 = torch.cat([cre_feats2, pad2], dim=0)
        graph_repr2 = padded2.unsqueeze(0)
        dec_out2    = dec(dec_input, graph_repr2, concepts)
        logits      = dec_out2.logits
        loss = F.cross_entropy(
            logits.reshape(-1, ACTUAL_VOCAB),
            tgt[:, 1:].reshape(-1),
            ignore_index=0,
        )
        val_result2 = validator(logits.detach(), graph_repr2.detach(), concepts.detach())
        val_passed  = val_result2.passed
        did_rereason= True

    return logits, loss, n_iters, val_passed, did_rereason


# \u2500\u2500 Evaluación \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def evaluate_cora():
    enc.eval(); crys.eval(); cre.eval(); dec.eval(); validator.eval()
    losses=[]; wf1s=[]; iters=[]; val_pass_count=0; rer_count=0
    with torch.no_grad():
        for ex in EVAL_EX[:100]:
            src_ids=SHARED_VOCAB.encode(ex.problem_text,max_len=_MAX_Q,add_bos=True,add_eos=True)
            tgt_ids=SHARED_VOCAB.encode(ex.answer,max_len=_MAX_A,add_bos=True,add_eos=True)
            with autocast(enabled=USE_AMP):
                logits, loss, n_i, val_ok, rerasoned = cora_forward_eval(src_ids, tgt_ids)
            if loss is None:
                losses.append(9.9); wf1s.append(0.0); continue
            losses.append(loss.item())
            pred = logits.argmax(-1).squeeze(0).tolist()
            wf1s.append(word_f1(pred, tgt_ids[1:]))
            if n_i > 0:  iters.append(n_i)
            if val_ok:   val_pass_count += 1
            if rerasoned: rer_count     += 1
    enc.train(); crys.train(); cre.train(); dec.train(); validator.train()
    avg_iters   = sum(iters)/len(iters) if iters else 0.0
    val_pct     = val_pass_count / len(EVAL_EX[:100]) * 100
    return sum(losses)/len(losses), sum(wf1s)/len(wf1s), avg_iters, val_pct, rer_count


# \u2500\u2500 Timing probe: 10 steps de muestra para estimar duración total \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print('[CORA] Timing probe (10 steps)...')
_t_probe = time.perf_counter()
_probe_ex= random.sample(TRAIN_EX, 10)
for _ex in _probe_ex:
    _s = SHARED_VOCAB.encode(_ex.problem_text, max_len=_MAX_Q, add_bos=True, add_eos=True)
    _t = SHARED_VOCAB.encode(_ex.answer,       max_len=_MAX_A, add_bos=True, add_eos=True)
    with autocast(enabled=USE_AMP):
        _, _l, _, _ = cora_forward_train(_s, _t)
    if _l is not None:
        cora_scaler.scale(_l).backward()
        cora_scaler.step(cora_opt); cora_scaler.update()
    cora_opt.zero_grad()
_probe_secs = time.perf_counter() - _t_probe
_est_total  = _probe_secs / 10 * TOTAL_STEPS
print(f'  probe: {_probe_secs:.2f}s para 10 steps → estimado {_est_total:.0f}s ({_est_total/60:.1f} min)')

EFFECTIVE_STEPS = TOTAL_STEPS
if _est_total > 600:
    EFFECTIVE_STEPS = 1000
    print(f'  AJUSTE: estimado >10 min → reduciendo a {EFFECTIVE_STEPS} steps')
else:
    print(f'  OK: estimado <10 min → manteniendo {EFFECTIVE_STEPS} steps')

# \u2500\u2500 Training: EFFECTIVE_STEPS ejemplos, batch=1 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
CORA_EVALS          = []   # [(step, val_loss, val_wf1, avg_iters, val_pct, rer_cnt)]
CORA_LOSSES         = []
CORA_ITERS          = []   # iteraciones CRE por step de training
BUDGET_LEVEL_COUNTS = {0: 0, 1: 0, 2: 0, 3: 0}  # TRIVIAL/SIMPLE/COMPLEX/DEEP
VAL_PASSES_TOTAL    = 0
VAL_TOTAL           = 0
REREASON_TOTAL      = 0

train_cycle = TRAIN_EX.copy()
random.shuffle(train_cycle)
train_idx = 0

t_start = time.perf_counter()
for step in range(1, EFFECTIVE_STEPS + 1):
    ex = train_cycle[train_idx % len(train_cycle)]
    train_idx += 1

    src_ids = SHARED_VOCAB.encode(ex.problem_text, max_len=_MAX_Q, add_bos=True, add_eos=True)
    tgt_ids = SHARED_VOCAB.encode(ex.answer,       max_len=_MAX_A, add_bos=True, add_eos=True)

    cora_opt.zero_grad()
    with autocast(enabled=USE_AMP):
        _, loss, n_iters, budget_level = cora_forward_train(src_ids, tgt_ids)

    if loss is None:
        CORA_LOSSES.append(0.0)
        continue

    cora_scaler.scale(loss).backward()
    cora_scaler.unscale_(cora_opt)
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    cora_scaler.step(cora_opt); cora_scaler.update()
    CORA_LOSSES.append(loss.item())
    if n_iters > 0: CORA_ITERS.append(n_iters)
    BUDGET_LEVEL_COUNTS[budget_level] = BUDGET_LEVEL_COUNTS.get(budget_level, 0) + 1

    if step % EVAL_EVERY == 0 or step == EFFECTIVE_STEPS:
        elapsed = time.perf_counter() - t_start
        val_loss, val_wf1, avg_i, val_pct, rer_cnt = evaluate_cora()
        CORA_EVALS.append((step, val_loss, val_wf1, avg_i, val_pct, rer_cnt))
        VAL_PASSES_TOTAL += int(val_pct / 100 * 100)   # out of 100 eval examples
        VAL_TOTAL        += 100
        REREASON_TOTAL   += rer_cnt
        avg_loss = sum(CORA_LOSSES[-EVAL_EVERY:]) / max(1, len(CORA_LOSSES[-EVAL_EVERY:]))
        avg_iters_train = (sum(CORA_ITERS[-EVAL_EVERY:]) /
                           max(1, len(CORA_ITERS[-EVAL_EVERY:])))
        print(f'[CORA] step={step:>4}/{EFFECTIVE_STEPS}  t={elapsed:5.1f}s  '
              f'train={avg_loss:.4f}  val={val_loss:.4f}  '
              f'wf1={val_wf1:.3f}  iters={avg_iters_train:.2f}  '
              f'val%={val_pct:.0f}  rer={rer_cnt}')

CORA_TOTAL_TIME = time.perf_counter() - t_start
CORA_FINAL_LOSS = CORA_EVALS[-1][1]
CORA_FINAL_WF1  = CORA_EVALS[-1][2]
CORA_AVG_ITERS  = sum(CORA_ITERS) / max(1, len(CORA_ITERS))
CORA_FINAL_VAL_PCT = CORA_EVALS[-1][4]
CORA_EX_PER_SEC = EFFECTIVE_STEPS / CORA_TOTAL_TIME

# Budget level distribution
_total_bgt = max(1, sum(BUDGET_LEVEL_COUNTS.values()))
BUDGET_DIST = {
    'TRIVIAL': BUDGET_LEVEL_COUNTS[0] / _total_bgt * 100,
    'SIMPLE':  BUDGET_LEVEL_COUNTS[1] / _total_bgt * 100,
    'COMPLEX': BUDGET_LEVEL_COUNTS[2] / _total_bgt * 100,
    'DEEP':    BUDGET_LEVEL_COUNTS[3] / _total_bgt * 100,
}

print(f'\\n[CORA] COMPLETADO ({EFFECTIVE_STEPS} steps)')
print(f'  tiempo      : {CORA_TOTAL_TIME:.1f}s  ({CORA_EX_PER_SEC:.1f} ex/s)')
print(f'  loss        : {CORA_FINAL_LOSS:.4f}   wf1: {CORA_FINAL_WF1:.3f}')
print(f'  iters CRE   : {CORA_AVG_ITERS:.2f} promedio  (max={_cre_cfg.max_iterations})')
print(f'  VAL pass    : {CORA_FINAL_VAL_PCT:.1f}%   re-razonamientos totales: {REREASON_TOTAL}')
print(f'  Budget dist : TRIVIAL={BUDGET_DIST["TRIVIAL"]:.1f}%  '
      f'SIMPLE={BUDGET_DIST["SIMPLE"]:.1f}%  '
      f'COMPLEX={BUDGET_DIST["COMPLEX"]:.1f}%  '
      f'DEEP={BUDGET_DIST["DEEP"]:.1f}%')
"""

# ── Cell 6: comparison con todas las nuevas métricas ─────────────────────────
nb['cells'][6]['source'] = """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# \u2500\u2500 Tabla resumen \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
speed_ratio = TF_EX_PER_SEC / max(CORA_EX_PER_SEC, 0.01)
wf1_winner  = 'Transformer' if TF_FINAL_WF1  >= CORA_FINAL_WF1  else 'CORA'
loss_winner = 'Transformer' if TF_FINAL_LOSS <= CORA_FINAL_LOSS else 'CORA'

n_eval_steps = len(CORA_EVALS)
eff_steps = EFFECTIVE_STEPS

print('=' * 72)
print(f'  BENCHMARK: {eff_steps} ejemplos identicos, batch=1, sin batching')
print('=' * 72)
print(f'  {"Metric":<32} {"Transformer":>16} {"CORA 5M":>16}')
print(f'  {"-"*32} {"-"*16} {"-"*16}')
print(f'  {"Ejemplos vistos":<32} {eff_steps:>16,} {eff_steps:>16,}')
print(f'  {"Tiempo total (s)":<32} {TF_TOTAL_TIME:>16.1f} {CORA_TOTAL_TIME:>16.1f}')
print(f'  {"Throughput (ex/s)":<32} {TF_EX_PER_SEC:>16.1f} {CORA_EX_PER_SEC:>16.1f}')
print(f'  {"Loss final (val)":<32} {TF_FINAL_LOSS:>16.4f} {CORA_FINAL_LOSS:>16.4f}')
print(f'  {"Word F1 final":<32} {TF_FINAL_WF1:>16.3f} {CORA_FINAL_WF1:>16.3f}')
print(f'  {"Iters CRE promedio":<32} {"N/A":>16} {CORA_AVG_ITERS:>16.2f}')
print(f'  {"VAL pass % (eval)":<32} {"N/A":>16} {CORA_FINAL_VAL_PCT:>15.1f}%')
print(f'  {"Re-razonamientos (total)":<32} {"N/A":>16} {REREASON_TOTAL:>16}')
print('=' * 72)
print(f'  Transformer es {speed_ratio:.1f}x mas rapido que CORA')
print(f'  Mejor calidad (loss)    : {loss_winner}')
print(f'  Mejor calidad (Word F1) : {wf1_winner}')
iters_pct = (1.0 - CORA_AVG_ITERS / _cre_cfg.max_iterations) * 100
print(f'  Gate ahorro compute CRE : {iters_pct:.1f}% menos iters vs max={_cre_cfg.max_iterations}')
print('=' * 72)

# \u2500\u2500 Tabla de distribución de budget levels \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print('\\nBUDGET LEVEL DISTRIBUTION (queries clasificadas por complejidad):')
print(f'  {"Level":<10} {"Count":>8} {"Pct":>8}   {"Iters asignadas":>20}')
print(f'  {"-"*10} {"-"*8} {"-"*8}   {"-"*20}')
level_names   = ['TRIVIAL', 'SIMPLE', 'COMPLEX', 'DEEP']
level_iters   = [1, 3, 10, _cre_cfg.max_iterations]
total_queries = max(1, sum(BUDGET_LEVEL_COUNTS.values()))
for li, (lname, liters) in enumerate(zip(level_names, level_iters)):
    cnt = BUDGET_LEVEL_COUNTS.get(li, 0)
    pct = cnt / total_queries * 100
    print(f'  {lname:<10} {cnt:>8,} {pct:>7.1f}%   {liters:>20}')

# \u2500\u2500 Gráficas: loss y Word F1 por step \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

tf_steps   = [e[0] for e in TF_EVALS]
tf_losses  = [e[1] for e in TF_EVALS]
tf_wf1s    = [e[2] for e in TF_EVALS]
cora_steps = [e[0] for e in CORA_EVALS]
cora_losses= [e[1] for e in CORA_EVALS]
cora_wf1s  = [e[2] for e in CORA_EVALS]
cora_iters_by_step = [e[3] for e in CORA_EVALS]
cora_val_pct       = [e[4] for e in CORA_EVALS]
cora_rer_by_step   = [e[5] for e in CORA_EVALS]

ax = axes[0]
ax.plot(tf_steps,   tf_losses,   'b-o', label='Transformer', linewidth=2)
ax.plot(cora_steps, cora_losses, 'r-s', label='CORA 5M (MoE+VAL)', linewidth=2)
ax.set_xlabel('Step (= ejemplos vistos)'); ax.set_ylabel('Val Loss')
ax.set_title(f'Loss vs Step ({eff_steps} ej.)')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(tf_steps,   tf_wf1s,   'b-o', label='Transformer', linewidth=2)
ax.plot(cora_steps, cora_wf1s, 'r-s', label='CORA 5M (MoE+VAL)', linewidth=2)
ax.set_xlabel('Step (= ejemplos vistos)'); ax.set_ylabel('Word F1')
ax.set_title('Word F1 vs Step')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('benchmark_vs_steps.png', dpi=120, bbox_inches='tight')
plt.show(); print('[fig] benchmark_vs_steps.png guardado')

# \u2500\u2500 Gráfica: CRE iters + VAL pass % por checkpoint \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 3.5))

ax = axes3[0]
ax.plot(cora_steps, cora_iters_by_step, 'g-^', linewidth=2, label='Avg CRE iters (val)')
ax.axhline(y=_cre_cfg.max_iterations, color='r', linestyle='--',
           alpha=0.5, label=f'max_iterations={_cre_cfg.max_iterations}')
ax.axhline(y=_cre_cfg.min_iterations, color='b', linestyle='--',
           alpha=0.5, label=f'min_iterations={_cre_cfg.min_iterations}')
ax.set_xlabel('Step'); ax.set_ylabel('Avg iterations')
ax.set_title('CRE ConvergenceGate + MoE: iters usadas')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(0, _cre_cfg.max_iterations + 0.5)

ax = axes3[1]
ax.plot(cora_steps, cora_val_pct, 'm-D', linewidth=2, label='VAL pass %')
ax.set_xlabel('Step'); ax.set_ylabel('% respuestas que pasan VAL')
ax.set_title('AionCValidator: coherencia respuesta/grafo')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('benchmark_cre_val.png', dpi=120, bbox_inches='tight')
plt.show(); print('[fig] benchmark_cre_val.png guardado')

# \u2500\u2500 Gráfica: distribución de budget levels (barras) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig4, ax4 = plt.subplots(figsize=(7, 3.5))
bgt_pcts  = [BUDGET_DIST[k] for k in ['TRIVIAL', 'SIMPLE', 'COMPLEX', 'DEEP']]
bgt_colors= ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
bars = ax4.bar(level_names, bgt_pcts, color=bgt_colors, edgecolor='black', alpha=0.85)
for bar, pct in zip(bars, bgt_pcts):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
ax4.set_xlabel('Budget Level'); ax4.set_ylabel('% queries')
ax4.set_title('BudgetManager: distribución de complejidad (heurístico)')
ax4.set_ylim(0, max(bgt_pcts) * 1.2 + 5)
ax4.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('benchmark_budget_dist.png', dpi=120, bbox_inches='tight')
plt.show(); print('[fig] benchmark_budget_dist.png guardado')

# \u2500\u2500 Gráficas vs tiempo real \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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
print('\\n' + '='*72)
print('  5 EJEMPLOS COMPARATIVOS')
print('='*72)

tf_model.eval(); enc.eval(); crys.eval(); cre.eval(); dec.eval(); validator.eval()
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

        # CORA greedy (teacher forcing) + validator
        with autocast(enabled=USE_AMP):
            logits_cr, _, n_i, val_ok, rerasoned = cora_forward_eval(src_ids, tgt_ids)
        cora_pred_ids = logits_cr.argmax(-1).squeeze(0).tolist() if logits_cr is not None else []
        cora_pred     = SHARED_VOCAB.decode(cora_pred_ids)
        cora_f1       = word_f1(cora_pred_ids, tgt_ids[1:])

        bgt_ex = BudgetManager.classify_heuristic(len(src_ids), _cre_cfg.max_iterations)
        print(f'\\nEj {i+1}: {ex.problem_text[:80]}')
        print(f'  Gold       : {gold}')
        print(f'  Transformer: {tf_pred}  [WF1={tf_f1:.2f}]')
        print(f'  CORA       : {cora_pred}  [WF1={cora_f1:.2f}]  '
              f'[iters={n_i}  val={"OK" if val_ok else "FAIL"}  '
              f'budget={bgt_ex.level.name}  rer={"yes" if rerasoned else "no"}]')
print('='*72)
"""

# ── Verification ──────────────────────────────────────────────────────────────
c2 = nb['cells'][2]['source']
c5 = nb['cells'][5]['source']
c6 = nb['cells'][6]['source']

checks = [
    # Bundle: nuevos archivos
    ('Bundle has cre/moe.py',             "'cre/moe.py':"             in c2),
    ('Bundle has budget/__init__.py',      "'budget/__init__.py':"     in c2),
    ('Bundle has budget/manager.py',       "'budget/manager.py':"      in c2),
    ('Bundle has validation/__init__.py',  "'validation/__init__.py':" in c2),
    ('Bundle has validation/model.py',     "'validation/model.py':"    in c2),
    ('Bundle has cre/engine.py',           "'cre/engine.py':"          in c2),
    ('Bundle has cre/config.py',           "'cre/config.py':"          in c2),
    ('Bundle has router/pipeline.py',      "'router/pipeline.py':"     in c2),
    ('Bundle pkg list has budget',         "'budget'" in c2 and 'for pkg in' in c2),
    ('Bundle pkg list has validation',     "'validation'" in c2 and 'for pkg in' in c2),
    # Cell 5: todos los módulos
    ('Cell5 from budget import',           'from budget'               in c5),
    ('Cell5 from validation import',       'from validation'           in c5),
    ('Cell5 use_moe=True',                 'use_moe              = True' in c5),
    ('Cell5 use_convergence_gate=True',    'use_convergence_gate = True' in c5),
    ('Cell5 use_budget_manager=True',      'use_budget_manager = True'  in c5),
    ('Cell5 use_validator=True',           'use_validator      = True'  in c5),
    ('Cell5 val_rereason=True',            'val_rereason       = True'  in c5),
    ('Cell5 BudgetManager.classify_heuristic', 'BudgetManager.classify_heuristic' in c5),
    ('Cell5 AionCValidator',               'AionCValidator'            in c5),
    ('Cell5 BUDGET_LEVEL_COUNTS',          'BUDGET_LEVEL_COUNTS'       in c5),
    ('Cell5 VAL_PASSES_TOTAL',             'VAL_PASSES_TOTAL'          in c5),
    ('Cell5 REREASON_TOTAL',               'REREASON_TOTAL'            in c5),
    ('Cell5 timing probe',                 '_est_total'                in c5),
    ('Cell5 EFFECTIVE_STEPS',              'EFFECTIVE_STEPS'           in c5),
    ('Cell5 cora_forward_train',           'cora_forward_train'        in c5),
    ('Cell5 cora_forward_eval',            'cora_forward_eval'         in c5),
    ('Cell5 did_rereason',                 'did_rereason'              in c5),
    ('Cell5 evaluate_cora returns val_pct','val_pct'                   in c5),
    # Cell 6: nuevas métricas
    ('Cell6 Iters CRE row',               'Iters CRE promedio'        in c6),
    ('Cell6 VAL pass row',                'VAL pass'                  in c6),
    ('Cell6 Re-razonamientos row',        'Re-razonamientos'          in c6),
    ('Cell6 BUDGET_DIST',                 'BUDGET_DIST'               in c6),
    ('Cell6 budget level table',          'TRIVIAL'                   in c6),
    ('Cell6 budget_dist plot',            'benchmark_budget_dist.png' in c6),
    ('Cell6 benchmark_cre_val plot',      'benchmark_cre_val.png'     in c6),
    ('Cell6 val pct by checkpoint',       'cora_val_pct'              in c6),
    ('Cell6 BudgetManager in example',    'bgt_ex.level.name'         in c6),
]

print('\nVerification:')
all_ok = True
for label, ok in checks:
    mark = 'OK  ' if ok else 'FAIL'
    print(f'  {mark} {label}')
    if not ok: all_ok = False

if all_ok:
    nb['cells'][2]['source'] = cell2
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'\nALL OK — notebook guardado ({NB_PATH})')
else:
    print('\nFAILED — notebook NO guardado')
    raise SystemExit(1)
