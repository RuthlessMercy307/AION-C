"""
_gen_notebook_fast.py
Genera colab_train_cora_50m_fast.ipynb reutilizando la cell-3 base64
del notebook lento y reemplazando las celdas de setup/training.
"""

import json
import os
import uuid
import pathlib

BASE = pathlib.Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Reutilizar cell 3 del notebook anterior (28 archivos en base64)
# ─────────────────────────────────────────────────────────────────────────────
src_nb_path = BASE / "colab_train_cora_50m.ipynb"
with open(src_nb_path, "r", encoding="utf-8") as f:
    src_nb = json.load(f)

# Cell 3 (index 2) = el bloque base64 con todos los archivos
cell3_source = src_nb["cells"][2]["source"]   # lista de strings


# ─────────────────────────────────────────────────────────────────────────────
# Helper: convierte string multilínea en lista-de-líneas para nbformat
# ─────────────────────────────────────────────────────────────────────────────
def to_source(code: str):
    lines = code.split("\n")
    result = []
    for i, line in enumerate(lines):
        result.append(line + ("\n" if i < len(lines) - 1 else ""))
    return result


def mk_code(code: str, cell_id: str = None) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id or uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source(code),
    }


def mk_md(text: str, cell_id: str = None) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id or uuid.uuid4().hex[:8],
        "metadata": {},
        "source": to_source(text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 1 — Título
# ─────────────────────────────────────────────────────────────────────────────
CELL1_MD = """\
# CORA 50M — Fast Training (T4, 15-20 min)

Pipeline completo: Encoder(Mamba) → Crystallizer → CRE → Decoder
- Config primaria: **hidden_dim=256, 8 layers, CRE=3 iter** (~37M params)
- Auto-fallback:   **hidden_dim=128, 4 layers, CRE=3 iter** ( ~5M params) si >25 min
- 2000 steps · batch=1 · cosine LR · checkpoints en Google Drive"""

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 2 — Mount Drive
# ─────────────────────────────────────────────────────────────────────────────
CELL2_CODE = """\
from google.colab import drive
drive.mount('/content/drive')
import os

DRIVE_DIR = '/content/drive/MyDrive/cora_50m_fast'
os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/results',     exist_ok=True)
print(f'Drive montado. Directorio: {DRIVE_DIR}')"""

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 4 — GPU check (idéntica a la versión lenta)
# ─────────────────────────────────────────────────────────────────────────────
CELL4_CODE = """\
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    free_b, total_b = torch.cuda.mem_get_info()
    print(f'GPU     : {name}')
    print(f'VRAM    : {free_b/1e9:.1f} GB libre / {total_b/1e9:.1f} GB total')
    for n_params, desc in [(37_053_274, '256D-8L (37M)'), (5_100_000, '128D-4L (5M)')]:
        model_mb  = n_params * 4 / 1e6
        optim_mb  = n_params * 4 * 2 / 1e6
        total_est = model_mb + optim_mb + 300
        ok = 'OK' if total_est/1024 < free_b/1e9 else 'RIESGO'
        print(f'  {desc}: ~{total_est:.0f} MB estimado [{ok}]')
else:
    print('Sin GPU — ve a Runtime -> Change runtime type -> T4 GPU')"""

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 5 — Training con timing automático y auto-fallback
# ─────────────────────────────────────────────────────────────────────────────
CELL5_CODE = """\
import sys, os, time, torch, shutil, glob
sys.path.insert(0, '/content/aion_c')

import experiments.train_cora_50m as trainer
from router.pipeline import CORAConfig

# ── Constantes fast ──────────────────────────────────────────────────────────
N_STEPS      = 2000
CRE_ITERS    = 3        # 3 en vez de 10 — principal aceleración
ACCUM_STEPS  = 1        # sin gradient accumulation
EVAL_EVERY   = 400
CKPT_EVERY   = 1000
PRINT_EVERY  = 50
TARGET_MIN   = 25.0     # abortar config si estimado > 25 min
DRIVE_DIR    = '/content/drive/MyDrive/cora_50m_fast'

# Aplicar constantes al módulo
trainer.N_DATASET   = 5000
trainer.N_STEPS     = N_STEPS
trainer.CRE_ITERS   = CRE_ITERS
trainer.ACCUM_STEPS = ACCUM_STEPS
trainer.EVAL_EVERY  = EVAL_EVERY
trainer.CKPT_EVERY  = CKPT_EVERY
trainer.PRINT_EVERY = PRINT_EVERY
trainer.TRAIN_FRAC  = 0.80

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── Dataset + vocab (una sola vez, se reutilizan) ────────────────────────────
print('\\n[setup] Generando dataset...')
train_ex, eval_ex = trainer.build_dataset(5000)
vocab = trainer.build_vocab(train_ex + eval_ex, max_size=8000)
actual_vocab = len(vocab)
print(f'[setup] vocab={actual_vocab}  train={len(train_ex)}  eval={len(eval_ex)}')

# Cachear para que train() no regenere
_t, _e, _v = train_ex, eval_ex, vocab
trainer.build_dataset = lambda n, seed=42: (_t, _e)
trainer.build_vocab   = lambda examples, max_size: _v


# ── Factory de configs candidatos ────────────────────────────────────────────
def make_cfg(hidden_dim, n_layers, vocab_size):
    n_heads = 8 if hidden_dim >= 256 else 4
    return CORAConfig(
        hidden_dim   = hidden_dim,
        vocab_size   = vocab_size,
        enc_n_layers  = n_layers,
        enc_state_dim = 16,
        enc_expand    = 2,
        enc_d_conv    = 4,
        enc_ffn_mult  = 4,
        cryst_max_nodes      = 32,
        cryst_n_heads        = n_heads,
        cryst_node_threshold = 0.01,
        cryst_edge_threshold = 0.01,
        cre_edge_dim         = 64,
        cre_message_dim      = min(128, hidden_dim // 2),
        cre_n_message_layers = 2,
        cre_max_iterations   = CRE_ITERS,
        pad_n_slots  = 32,
        pad_slot_dim = min(128, hidden_dim // 2),
        dec_n_layers    = n_layers,
        dec_n_heads     = n_heads,
        dec_max_seq_len = 256,
        dec_state_dim   = 16,
        dec_expand      = 2,
        dec_d_conv      = 4,
        dec_ffn_mult    = 4,
    )


# ── Benchmark de 1 step ───────────────────────────────────────────────────────
def benchmark_ms(cfg, device, vocab, train_ex, n_cre, n_warmup=3):
    from encoder import StreamEncoder
    from crystallizer import GraphCrystallizer
    from cre import CausalReasoningEngine, DifferentiableScratchPad
    from decoder import StreamDecoder

    enc  = StreamEncoder(cfg.encoder_config()).to(device)
    crys = GraphCrystallizer(cfg.crystallizer_config()).to(device)
    cre  = CausalReasoningEngine(cfg.cre_config()).to(device)
    pad  = DifferentiableScratchPad(cfg.scratch_pad_config()).to(device)
    dec  = StreamDecoder(cfg.decoder_config()).to(device)
    all_p = (list(enc.parameters()) + list(crys.parameters()) +
             list(cre.parameters()) + list(pad.parameters()) + list(dec.parameters()))
    n_p = sum(p.numel() for p in all_p if id(p))
    opt = torch.optim.AdamW(all_p, lr=3e-4)

    ex    = train_ex[0]
    q_len = min(len(ex.problem_text.lower().split()), 80)
    q_ids = vocab.to_tensor(vocab.encode(ex.problem_text, 80), device, 80)
    a_ids = vocab.to_tensor(vocab.encode(ex.answer, 48, add_eos=True), device)

    # Warmup
    for _ in range(n_warmup):
        opt.zero_grad()
        co, gr, cf, nv, lm, enc_out = trainer.forward_full(enc,crys,cre,pad,dec,q_ids,a_ids,cfg,n_cre,vocab)
        tot, _ = trainer.compute_all_losses(co,cf,nv,lm,ex.graph,a_ids,cfg,vocab,device,enc_out,ex.entity_spans,q_len)
        tot.backward()
    if device.type == 'cuda': torch.cuda.synchronize()

    opt.zero_grad()
    t0 = time.perf_counter()
    co, gr, cf, nv, lm, enc_out = trainer.forward_full(enc,crys,cre,pad,dec,q_ids,a_ids,cfg,n_cre,vocab)
    tot, _ = trainer.compute_all_losses(co,cf,nv,lm,ex.graph,a_ids,cfg,vocab,device,enc_out,ex.entity_spans,q_len)
    tot.backward()
    if device.type == 'cuda': torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000

    del enc,crys,cre,pad,dec,opt,all_p
    if device.type=='cuda': torch.cuda.empty_cache()
    return ms, n_p


# ── Auto-selección de config ──────────────────────────────────────────────────
CANDIDATES = [
    (256, 8, 'hidden_dim=256  layers=8  (~37M params)'),
    (128, 4, 'hidden_dim=128  layers=4  (~ 5M params)'),
]

selected_cfg  = None
selected_desc = None
selected_ms   = None

print('\\n[timing] Midiendo throughput por configuracion...')
print('-' * 60)

for hidden_dim, n_layers, desc in CANDIDATES:
    cfg_try = make_cfg(hidden_dim, n_layers, actual_vocab)
    ms, n_p = benchmark_ms(cfg_try, device, vocab, train_ex, CRE_ITERS)
    est_min = ms * N_STEPS / 60_000
    ok = est_min <= TARGET_MIN
    status = 'OK' if ok else 'lento'
    print(f'  {desc}')
    print(f'    {ms:6.0f} ms/step  ->  {N_STEPS} steps = {est_min:.1f} min  [{status}]')

    if ok:
        selected_cfg  = cfg_try
        selected_desc = desc
        selected_ms   = ms
        print(f'    -> SELECCIONADA')
        break
    else:
        print(f'    -> probando siguiente...')

if selected_cfg is None:
    selected_cfg  = make_cfg(128, 4, actual_vocab)
    selected_desc = 'hidden_dim=128  layers=4  (fallback minimo)'
    ms, _ = benchmark_ms(selected_cfg, device, vocab, train_ex, CRE_ITERS, n_warmup=1)
    selected_ms = ms

print('-' * 60)
print(f'\\nCONFIG FINAL : {selected_desc}')
print(f'TIEMPO ESTIM : {selected_ms * N_STEPS / 60_000:.1f} min ({selected_ms:.0f} ms/step x {N_STEPS} steps)')
print()

# ── Patch make_50m_config para devolver la config seleccionada ───────────────
_frozen = selected_cfg
trainer.make_50m_config = lambda vocab_size=None: _frozen

# ── Patch save_checkpoint para guardar en Drive ──────────────────────────────
_orig_save = trainer.save_checkpoint
def _drive_save(step, encoder, crystallizer, cre, scratch_pad, decoder,
                optimizer, scheduler, loss_history, ckpt_dir):
    path = _orig_save(step, encoder, crystallizer, cre, scratch_pad, decoder,
                      optimizer, scheduler, loss_history,
                      f'{DRIVE_DIR}/checkpoints')
    print(f'  [Drive] checkpoint guardado: {os.path.basename(path)}')
    return path
trainer.save_checkpoint = _drive_save

# ── Lanzar entrenamiento ──────────────────────────────────────────────────────
trainer.train()

# ── Copiar resultados a Drive ─────────────────────────────────────────────────
result_files = sorted(glob.glob('/content/aion_c/experiments/results/train_cora_50m_*.json'))
for src_f in result_files:
    dst = f'{DRIVE_DIR}/results/{os.path.basename(src_f)}'
    shutil.copy(src_f, dst)
    print(f'[Drive] resultados: {dst}')"""

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 6 — Curva de loss + ejemplos eval (igual a la versión lenta)
# ─────────────────────────────────────────────────────────────────────────────
CELL6_CODE = """\
import json, glob, os
import matplotlib.pyplot as plt

DRIVE_DIR = '/content/drive/MyDrive/cora_50m_fast'

# Buscar resultados: primero local, luego en Drive
result_files = (sorted(glob.glob('/content/aion_c/experiments/results/train_cora_50m_*.json')) +
                sorted(glob.glob(f'{DRIVE_DIR}/results/train_cora_50m_*.json')))

if not result_files:
    print('No se encontraron archivos de resultados.')
    print('Asegurate de haber ejecutado la celda de entrenamiento.')
else:
    with open(result_files[-1], encoding='utf-8') as f:
        data = json.load(f)

    cfg_used = data.get('config', {})
    summary  = data.get('summary', {})
    print(f'Modelo   : hidden_dim={cfg_used.get(\"hidden_dim\")}  params={cfg_used.get(\"n_params\",0):,}')
    print(f'Steps    : {cfg_used.get(\"n_steps\")}  LM mejora: {summary.get(\"lm_loss_improvement_pct\",0):.1f}%')
    print(f'Tiempo   : {summary.get(\"total_seconds\",0)/60:.1f} min  ({summary.get(\"avg_ms_per_step\",0):.0f} ms/step)')
    print()

    curve  = data['loss_curve']
    steps  = [r['step']  for r in curve]
    total  = [r['total'] for r in curve]
    lm_pts = [(r['step'], r['lm']) for r in curve if r.get('lm')]

    # Rolling average helper
    def smooth(vals, w=30):
        out = []
        for i in range(len(vals)):
            sl = vals[max(0, i-w):i+1]
            out.append(sum(sl)/len(sl))
        return out

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, total, alpha=0.25, color='steelblue', linewidth=0.7)
    axes[0].plot(steps, smooth(total), color='steelblue', linewidth=2, label='MA-30')
    axes[0].set_title('Loss Total (NC + Rel + Coh + LM)')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    if lm_pts:
        lm_s, lm_v = zip(*lm_pts)
        axes[1].plot(lm_s, lm_v, alpha=0.25, color='coral', linewidth=0.7)
        axes[1].plot(lm_s, smooth(list(lm_v)), color='coral', linewidth=2)
        axes[1].set_title('LM Loss (generacion de respuestas)')
        axes[1].set_xlabel('Step'); axes[1].grid(alpha=0.3)

    plt.suptitle('CORA 50M Fast — Curva de entrenamiento', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_png = f'{DRIVE_DIR}/loss_curve_fast.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Grafico guardado: {out_png}')

    # Eval snapshots
    snaps = data.get('eval_snapshots', [])
    if snaps:
        print()
        print('=' * 72)
        print('EVALUACIONES DURANTE ENTRENAMIENTO')
        print('=' * 72)
        print(f'  {"Step":>6}  {"NodeAcc":>8}  {"RelRecall":>10}  {"WordF1":>8}')
        print('  ' + '-' * 40)
        for s in snaps:
            print(f'  {s["step"]:>6}  {s["avg_node_acc"]:>8.1%}  '
                  f'{s["avg_rel_recall"]:>10.1%}  {s["avg_word_f1"]:>8.1%}')

        # Ultimos 5 ejemplos del ultimo snapshot
        last = snaps[-1]
        print()
        print('=' * 72)
        print(f'ULTIMOS 5 EJEMPLOS EVAL (step {last["step"]})')
        print('=' * 72)
        for i, ex in enumerate(last.get('examples', [])[:5], 1):
            print(f'\\n[{i}] Nivel {ex["level"]}  '
                  f'Nodos GT:{ex["gt_n"]} Pred:{ex["pred_n"]}  '
                  f'NodeAcc:{ex["node_acc"]:.0%}  RelRecall:{ex["rel_recall"]:.0%}')
            print(f'  Input: {ex["q_preview"]}...')
            print(f'  GT   : {ex["gt_answer"][:80]}')
            gen = ex.get("gen_answer", "") or "(vacio)"
            print(f'  Gen  : {gen[:80]}')
            print(f'  WordF1: {ex["word_overlap"]:.0%}')"""


# ─────────────────────────────────────────────────────────────────────────────
# Ensamblar notebook
# ─────────────────────────────────────────────────────────────────────────────

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "accelerator": "GPU",
        "colab": {
            "name": "colab_train_cora_50m_fast.ipynb",
            "provenance": [],
        },
    },
    "cells": [
        mk_md(CELL1_MD, "cell-001"),
        mk_code(CELL2_CODE, "cell-002"),
        # Cell 3 — reuse verbatim from existing notebook
        {
            "cell_type": "code",
            "id": "cell-003",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": cell3_source,
        },
        mk_code(CELL4_CODE, "cell-004"),
        mk_code(CELL5_CODE, "cell-005"),
        mk_code(CELL6_CODE, "cell-006"),
    ],
}

out_path = BASE / "colab_train_cora_50m_fast.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

size = os.path.getsize(out_path)
print(f"Notebook generado: {out_path}")
print(f"Tamanio: {size:,} bytes  ({size/1024:.0f} KB)")
print(f"Celdas: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells'], 1):
    src = cell['source']
    n = len(src) if isinstance(src, list) else src.count('\n')
    print(f"  Cell {i} [{cell['cell_type']:8}]  {n:5} lines")
