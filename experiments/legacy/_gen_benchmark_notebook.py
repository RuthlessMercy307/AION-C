"""
_gen_benchmark_notebook.py
Genera benchmark_cora_vs_transformer.ipynb

Notebook autocontenido de Colab que:
  1. Extrae los 28 archivos fuente de CORA (base64, celda 3 reutilizada)
  2. Genera dataset compartido + vocab (mismos 5000 ejemplos, seed=42)
  3. Entrena Transformer Baseline ~2M  (hidden=128, 4+4 layers, solo LM loss)
  4. Entrena CORA 5M             (hidden=128, 4L, CRE=3, loss completa)
  5. Tabla comparativa + plots lado a lado

Tiempo estimado en T4: < 15 min total.
"""

import json
import os
import uuid
import pathlib

BASE = pathlib.Path(__file__).parent

# ─── Reutilizar celda 3 del notebook lento (28 archivos en base64) ────────────
src_nb_path = BASE / "colab_train_cora_50m.ipynb"
with open(src_nb_path, "r", encoding="utf-8") as f:
    src_nb = json.load(f)
cell3_source = src_nb["cells"][2]["source"]  # lista de strings ya en formato nbformat


# ─── Helpers ──────────────────────────────────────────────────────────────────

def to_source(code: str):
    """Convierte bloque de texto a lista de líneas nbformat."""
    raw = code.split("\n")
    out = []
    for i, line in enumerate(raw):
        out.append(line + ("\n" if i < len(raw) - 1 else ""))
    return out


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


# ═════════════════════════════════════════════════════════════════════════════
# CELDA 1 — Título
# ═════════════════════════════════════════════════════════════════════════════

CELL1_MD = """\
# CORA 5M vs Transformer Baseline — Benchmark en T4

Dos arquitecturas, mismos datos, mismos hiperparámetros de entrenamiento.

| Modelo | Arquitectura | Loss |
|--------|-------------|------|
| **CORA 5M** | Encoder(Mamba,4L) + Crystallizer + CRE(3iter) + Decoder(4L) | ND + Rel + Coh + LM |
| **Transformer** | Embedding + 4 enc + 4 dec (Transformer estándar) | LM only |

Ambos: 2000 steps · AdamW lr=3e-4 · cosine decay · 5000 ejemplos L1-4 · vocab compartido · **GPU obligatorio**"""


# ═════════════════════════════════════════════════════════════════════════════
# CELDA 2 — Drive mount
# ═════════════════════════════════════════════════════════════════════════════

CELL2_CODE = """\
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE_DIR = '/content/drive/MyDrive/cora_benchmark'
os.makedirs(DRIVE_DIR, exist_ok=True)
print(f'Drive montado. Resultados en: {DRIVE_DIR}')"""


# ═════════════════════════════════════════════════════════════════════════════
# CELDA 4 — GPU check + dataset + vocab compartido
# ═════════════════════════════════════════════════════════════════════════════

CELL4_CODE = """\
import sys, os, time, random, torch
from collections import Counter
sys.path.insert(0, '/content/aion_c')

# ── Verificación GPU obligatoria ──────────────────────────────────────────────
print(f'PyTorch : {torch.__version__}')
if not torch.cuda.is_available():
    raise RuntimeError(
        '\\n\\u26a0 No se detecta GPU.\\n'
        'Ve a Runtime \\u2192 Change runtime type \\u2192 T4 GPU y vuelve a ejecutar.'
    )

_name = torch.cuda.get_device_name(0)
_free, _total = torch.cuda.mem_get_info()
print(f'GPU     : {_name}')
print(f'VRAM    : {_free/1e9:.1f} GB libre / {_total/1e9:.1f} GB total')
DEVICE = torch.device('cuda')
print(f'device  : {DEVICE}')

from synth.causal_graph_gen import CausalGraphGenerator


# ── SimpleVocab compartido (interfaz identica a la del trainer) ───────────────
class SimpleVocab:
    PAD_ID = 0; BOS_ID = 1; EOS_ID = 2; UNK_ID = 3; N_SPECIAL = 4

    def __init__(self, max_size=8000):
        self.max_size = max_size
        self.word2id  = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.id2word  = {v: k for k, v in self.word2id.items()}
        self._counts  = Counter()

    def add_texts(self, texts):
        for t in texts:
            self._counts.update(t.lower().split())

    def build(self):
        for word, _ in self._counts.most_common(self.max_size - self.N_SPECIAL):
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx]  = word

    def encode(self, text, max_len=128, add_bos=False, add_eos=False):
        words = text.lower().split()[:max_len]
        ids   = []
        if add_bos: ids.append(self.BOS_ID)
        ids.extend(self.word2id.get(w, self.UNK_ID) for w in words)
        if add_eos: ids.append(self.EOS_ID)
        return ids or [self.UNK_ID]

    def decode(self, ids):
        skip = {self.PAD_ID, self.BOS_ID, self.EOS_ID}
        return ' '.join(self.id2word.get(i, '<UNK>') for i in ids if i not in skip)

    def to_tensor(self, ids, device, max_len=None, pad_to=None):
        if max_len: ids = ids[:max_len]
        if pad_to and len(ids) < pad_to:
            ids = ids + [self.PAD_ID] * (pad_to - len(ids))
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    def __len__(self): return len(self.word2id)


# ── Generar dataset compartido (seed=42 → ambos modelos ven los mismos datos) ─
print('\\n[data] Generando 5000 ejemplos L1-4 (seed=42)...')
_t0  = time.perf_counter()
_gen = CausalGraphGenerator(seed=42)
_all = _gen.generate_batch(n=5000, level_distribution={1: .30, 2: .30, 3: .25, 4: .15})
_rng = random.Random(42)
_rng.shuffle(_all)
TRAIN_EX = _all[:4000]
EVAL_EX  = _all[4000:]
print(f'[data] {len(TRAIN_EX)} train / {len(EVAL_EX)} eval  ({time.perf_counter()-_t0:.1f}s)')

# ── Vocab compartido ──────────────────────────────────────────────────────────
SHARED_VOCAB = SimpleVocab(max_size=8000)
SHARED_VOCAB.add_texts([e.problem_text for e in _all])
SHARED_VOCAB.add_texts([e.answer       for e in _all])
SHARED_VOCAB.build()
ACTUAL_VOCAB = len(SHARED_VOCAB)
print(f'[vocab] {ACTUAL_VOCAB} tokens')
print(f'\\n[check] entity_spans en primeros 3 ejemplos:')
for _ex in TRAIN_EX[:3]:
    _ok = sum(1 for s in _ex.entity_spans if s[0] >= 0)
    print(f'  L{_ex.complexity_level}  spans={_ok}/{len(_ex.entity_spans)}  {_ex.problem_text[:50]}...')"""


# ═════════════════════════════════════════════════════════════════════════════
# CELDA 5 — Transformer Baseline
# ═════════════════════════════════════════════════════════════════════════════

CELL5_CODE = """\
import math, time, random, torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# GPU explícito — falla con error claro si no hay GPU
assert torch.cuda.is_available(), '\\u26a0 Se requiere GPU T4. Ve a Runtime -> Change runtime type.'
_DEVICE = torch.device('cuda')
torch.cuda.empty_cache()
_free0, _ = torch.cuda.mem_get_info()
print(f'[Transformer] VRAM libre al inicio: {_free0/1e9:.2f} GB  device={_DEVICE}')

_MAX_Q  = 80
_MAX_A  = 48


# ── Modelo ────────────────────────────────────────────────────────────────────
class TransformerBaseline(nn.Module):
    \"\"\"
    Transformer seq2seq estándar.
    Encoder: embedding + positional + N TransformerEncoderLayer
    Decoder: embedding + positional + N TransformerDecoderLayer (con cross-attention)
    Weight tying tgt_embed <-> lm_head.
    \"\"\"
    def __init__(self, vocab_size, D=128, n_heads=4,
                 n_enc=4, n_dec=4, ffn_mult=4, max_len=256, dropout=0.1):
        super().__init__()
        self.D    = D
        ffn_dim   = D * ffn_mult
        self.src_emb = nn.Embedding(vocab_size, D, padding_idx=0)
        self.tgt_emb = nn.Embedding(vocab_size, D, padding_idx=0)
        self.pos     = nn.Embedding(max_len, D)
        enc_l = nn.TransformerEncoderLayer(
            D, n_heads, ffn_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_l, n_enc)
        dec_l = nn.TransformerDecoderLayer(
            D, n_heads, ffn_dim, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_l, n_dec)
        self.norm    = nn.LayerNorm(D)
        self.head    = nn.Linear(D, vocab_size, bias=False)
        self.head.weight = self.tgt_emb.weight          # weight tying
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def _pos(self, x):
        return self.pos(torch.arange(x.shape[1], device=x.device))

    def encode(self, src):
        pad_mask = (src == 0)
        x = self.src_emb(src) * math.sqrt(self.D) + self._pos(src)
        return self.encoder(x, src_key_padding_mask=pad_mask), pad_mask

    def forward(self, src, tgt):
        memory, mem_mask = self.encode(src)
        x      = self.tgt_emb(tgt) * math.sqrt(self.D) + self._pos(tgt)
        L      = tgt.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=src.device)
        out    = self.decoder(x, memory, tgt_mask=causal,
                              memory_key_padding_mask=mem_mask)
        return self.head(self.norm(out))                 # [B, L_a, V]


# ── Helpers ────────────────────────────────────────────────────────────────────
def word_f1(pred, target):
    pw = set(pred.lower().split()); tw = set(target.lower().split())
    if not tw: return 1.0
    p = len(pw & tw) / max(len(pw), 1)
    r = len(pw & tw) / len(tw)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


@torch.no_grad()
def tf_generate(model, q_ids, vocab, device, temperature=0.8, max_len=_MAX_A):
    model.eval()
    q_ids  = q_ids.to(device)
    memory, mem_mask = model.encode(q_ids)
    ids = torch.full((1, 1), vocab.BOS_ID, dtype=torch.long, device=device)
    out = []
    for _ in range(max_len):
        x      = model.tgt_emb(ids) * math.sqrt(model.D) + model._pos(ids)
        causal = nn.Transformer.generate_square_subsequent_mask(
            ids.shape[1], device=device)
        dec    = model.decoder(x, memory, tgt_mask=causal,
                               memory_key_padding_mask=mem_mask)
        probs  = torch.softmax(model.head(model.norm(dec))[0, -1, :] / temperature, -1)
        nxt    = int(torch.multinomial(probs, 1).item())
        if nxt == vocab.EOS_ID: break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    model.train()
    return vocab.decode(out)


@torch.no_grad()
def tf_eval(model, eval_ex, vocab, device, step, n_show=3):
    model.eval()
    idxs = sorted(random.sample(range(len(eval_ex)), min(n_show, len(eval_ex))))
    rows = []
    for i in idxs:
        ex  = eval_ex[i]
        q   = vocab.to_tensor(vocab.encode(ex.problem_text, _MAX_Q), device, _MAX_Q)
        gen = tf_generate(model, q, vocab, device)
        rows.append({
            'level': ex.complexity_level,
            'q':     ex.problem_text[:60],
            'gt':    ex.answer,
            'gen':   gen,
            'f1':    round(word_f1(gen, ex.answer), 3),
        })
    avg = sum(r['f1'] for r in rows) / len(rows)
    print(f'\\n  [TF eval @ step {step}]  avg Word F1 = {avg:.1%}')
    for r in rows:
        print(f'    [L{r["level"]}] {r["q"]}...')
        print(f'      GT : {r["gt"][:70]}')
        print(f'      Gen: {r["gen"][:70] or "(vacio)"}   F1={r["f1"]:.0%}')
    model.train()
    return {'step': step, 'avg_wf1': round(avg, 4), 'examples': rows}


# ── Construir modelo y mover explícitamente a GPU ─────────────────────────────
TF_MODEL  = TransformerBaseline(ACTUAL_VOCAB, D=128, n_heads=4, n_enc=4, n_dec=4).cuda()
TF_PARAMS = sum(p.numel() for p in TF_MODEL.parameters() if p.requires_grad)
_free1, _ = torch.cuda.mem_get_info()
print(f'[Transformer]  {TF_PARAMS:,} parametros  ({TF_PARAMS/1e6:.2f}M)')
print(f'               D=128  heads=4  4 enc + 4 dec layers  FFN=512')
print(f'               VRAM tras cargar: {_free1/1e9:.2f} GB libre')

# ── Training ───────────────────────────────────────────────────────────────────
_TF_STEPS  = 2000
_TF_EVAL   = 400
_TF_PRINT  = 100
_LR_INIT   = 3e-4
_LR_MIN    = 1e-5

tf_opt   = torch.optim.AdamW(TF_MODEL.parameters(), lr=_LR_INIT, weight_decay=1e-2)
tf_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    tf_opt, T_max=_TF_STEPS, eta_min=_LR_MIN)

TF_HISTORY = []; TF_EVALS = []
_recent    = deque(maxlen=50)
_train_ex  = list(TRAIN_EX); _ex_idx = 0

print(f'\\n[Transformer]  Entrenando {_TF_STEPS} steps  device={_DEVICE}')
TF_MODEL.train()
_t0 = time.perf_counter()

for _step in range(_TF_STEPS):
    if _ex_idx > 0 and _ex_idx % len(_train_ex) == 0:
        random.shuffle(_train_ex)
    ex      = _train_ex[_ex_idx % len(_train_ex)]; _ex_idx += 1
    q_ids   = SHARED_VOCAB.to_tensor(
        SHARED_VOCAB.encode(ex.problem_text, _MAX_Q), _DEVICE, _MAX_Q)
    a_ids   = SHARED_VOCAB.to_tensor(
        SHARED_VOCAB.encode(ex.answer, _MAX_A, add_eos=True), _DEVICE)
    bos     = torch.full((1, 1), SHARED_VOCAB.BOS_ID, dtype=torch.long, device=_DEVICE)
    dec_in  = torch.cat([bos, a_ids[:, :-1]], dim=1)

    tf_opt.zero_grad()
    logits = TF_MODEL(q_ids, dec_in)
    loss   = F.cross_entropy(
        logits.reshape(-1, ACTUAL_VOCAB), a_ids.reshape(-1),
        ignore_index=SHARED_VOCAB.PAD_ID, label_smoothing=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(TF_MODEL.parameters(), 1.0)
    tf_opt.step(); tf_sched.step()

    lv = float(loss.item()); _recent.append(lv)
    TF_HISTORY.append({'step': _step + 1, 'lm': round(lv, 6)})

    if (_step + 1) % _TF_PRINT == 0 or _step == 0:
        _ela = time.perf_counter() - _t0
        _eta = (_TF_STEPS - _step - 1) * _ela / (_step + 1)
        print(f'  step {_step+1:>5}/{_TF_STEPS}'
              f'  loss={sum(_recent)/len(_recent):.4f}'
              f'  lr={tf_sched.get_last_lr()[0]:.1e}'
              f'  ETA={_eta:.0f}s')

    if (_step + 1) % _TF_EVAL == 0:
        TF_EVALS.append(tf_eval(TF_MODEL, EVAL_EX, SHARED_VOCAB, _DEVICE, _step + 1))

_total_t = time.perf_counter() - _t0
print(f'\\n[Transformer] DONE  {_total_t/60:.1f} min'
      f'  {_total_t/_TF_STEPS*1000:.0f} ms/step'
      f'  final_loss={TF_HISTORY[-1]["lm"]:.4f}')"""


# ═════════════════════════════════════════════════════════════════════════════
# CELDA 6 — CORA 5M (128D-4L, CRE=3)
# ═════════════════════════════════════════════════════════════════════════════

CELL6_CODE = """\
import sys, os, glob, json, torch
sys.path.insert(0, '/content/aion_c')

# GPU explícito — falla con error claro si no hay GPU
assert torch.cuda.is_available(), '\\u26a0 Se requiere GPU T4. Ve a Runtime -> Change runtime type.'
torch.cuda.empty_cache()
_free0, _ = torch.cuda.mem_get_info()
print(f'[CORA] VRAM libre al inicio: {_free0/1e9:.2f} GB  device=cuda')

import experiments.train_cora_50m as trainer
from router.pipeline import CORAConfig

# ── Config fija 128D-4L (para que ambos modelos sean comparables) ─────────────
def _cfg_128(vocab_size=None):
    vs = vocab_size or ACTUAL_VOCAB
    return CORAConfig(
        hidden_dim=128, vocab_size=vs,
        enc_n_layers=4, enc_state_dim=16, enc_expand=2, enc_d_conv=4, enc_ffn_mult=4,
        cryst_max_nodes=32, cryst_n_heads=4,
        cryst_node_threshold=0.01, cryst_edge_threshold=0.01,
        cre_edge_dim=64, cre_message_dim=64, cre_n_message_layers=2, cre_max_iterations=3,
        pad_n_slots=32, pad_slot_dim=64,
        dec_n_layers=4, dec_n_heads=4, dec_max_seq_len=256,
        dec_state_dim=16, dec_expand=2, dec_d_conv=4, dec_ffn_mult=4,
    )

CORA_CFG = _cfg_128(ACTUAL_VOCAB)

# ── Patch constantes del trainer ──────────────────────────────────────────────
trainer.N_STEPS          = 2000
trainer.N_DATASET        = 5000
trainer.CRE_ITERS        = 3
trainer.ACCUM_STEPS      = 1
trainer.EVAL_EVERY       = 400
trainer.CKPT_EVERY       = 2000     # solo al final
trainer.PRINT_EVERY      = 100
trainer.make_50m_config  = _cfg_128

# ── Reutilizar dataset + vocab generados en celda 4 ──────────────────────────
_sv = SHARED_VOCAB
trainer.build_dataset = lambda n, seed=42: (list(TRAIN_EX), list(EVAL_EX))
trainer.build_vocab   = lambda examples, max_size=8000: _sv

# ── Capturar referencias a los modelos en el checkpoint final ─────────────────
CORA_ENCODER = CORA_CRYSTALLIZER = CORA_CRE = CORA_SCRATCH_PAD = CORA_DECODER = None

_orig_ckpt = trainer.save_checkpoint
def _capture(step, encoder, crystallizer, cre, scratch_pad, decoder,
             opt, sched, hist, ckpt_dir):
    global CORA_ENCODER, CORA_CRYSTALLIZER, CORA_CRE, CORA_SCRATCH_PAD, CORA_DECODER
    CORA_ENCODER      = encoder
    CORA_CRYSTALLIZER = crystallizer
    CORA_CRE          = cre
    CORA_SCRATCH_PAD  = scratch_pad
    CORA_DECODER      = decoder
    return _orig_ckpt(step, encoder, crystallizer, cre, scratch_pad, decoder,
                      opt, sched, hist,
                      '/content/aion_c/experiments/checkpoints')
trainer.save_checkpoint = _capture

# ── Entrenar (detect_device() dentro del trainer detecta CUDA automáticamente) ─
print(f'[CORA] Iniciando entrenamiento...')
trainer.train()

# ── Confirmar que los modelos están en GPU ────────────────────────────────────
if CORA_ENCODER is not None:
    _p = next(CORA_ENCODER.parameters())
    print(f'[CORA] Modelos en device: {_p.device}')
    assert str(_p.device).startswith('cuda'), f'\\u26a0 CORA no está en GPU: {_p.device}'
_free1, _ = torch.cuda.mem_get_info()
print(f'[CORA] VRAM libre tras entrenamiento: {_free1/1e9:.2f} GB')

# ── Cargar resultados del JSON ────────────────────────────────────────────────
_files = sorted(
    glob.glob('/content/aion_c/experiments/results/train_cora_50m_*.json'),
    key=os.path.getmtime)
if _files:
    with open(_files[-1], encoding='utf-8') as _f:
        CORA_DATA    = json.load(_f)
    CORA_EVALS   = CORA_DATA.get('eval_snapshots', [])
    CORA_HISTORY = CORA_DATA.get('loss_curve', [])
    print(f'\\n[CORA] {len(CORA_EVALS)} eval snapshots cargados')
else:
    CORA_DATA = {}; CORA_EVALS = []; CORA_HISTORY = []
    print('[CORA] ADVERTENCIA: no se encontraron resultados JSON')"""


# ═════════════════════════════════════════════════════════════════════════════
# CELDA 7 — Tabla comparativa + plots
# ═════════════════════════════════════════════════════════════════════════════

CELL7_CODE = """\
import os, sys, torch, random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, '/content/aion_c')
import experiments.train_cora_50m as trainer

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MAX_Q  = 80
_MAX_A  = 48

# Acceso seguro a variables que pueden no existir si se omitio una celda
_cora_evals   = globals().get('CORA_EVALS',   [])
_cora_history = globals().get('CORA_HISTORY', [])
_tf_evals     = globals().get('TF_EVALS',     [])
_tf_history   = globals().get('TF_HISTORY',   [])
_cora_ok      = all(globals().get(x) is not None
                    for x in ['CORA_ENCODER', 'CORA_DECODER', 'CORA_CFG'])

# ── Tabla Word F1 por step ─────────────────────────────────────────────────────
print('=' * 72)
print('  BENCHMARK: CORA 5M  vs  Transformer Baseline')
print('=' * 72)

_cora_by_step = {s['step']: s['avg_word_f1'] for s in _cora_evals}
_tf_by_step   = {s['step']: s['avg_wf1']     for s in _tf_evals}
_all_steps    = sorted(set(list(_cora_by_step) + list(_tf_by_step)))

if _all_steps:
    print(f'\\n  {"Step":>6}  {"CORA Word F1":>14}  {"TF Word F1":>12}  {"Delta (C-TF)":>14}')
    print('  ' + '-' * 52)
    for _s in _all_steps:
        _c  = _cora_by_step.get(_s); _t = _tf_by_step.get(_s)
        _cs = f'{_c:.1%}' if _c is not None else '     N/A'
        _ts = f'{_t:.1%}' if _t is not None else '     N/A'
        _ds = f'{_c-_t:+.1%}' if (_c is not None and _t is not None) else '     ---'
        print(f'  {_s:>6}  {_cs:>14}  {_ts:>12}  {_ds:>14}')

# ── Final LM loss ─────────────────────────────────────────────────────────────
print()
if _tf_history:
    print(f'  LM loss final  Transformer : {_tf_history[-1]["lm"]:.4f}')
if _cora_history:
    _lm_vals = [r['lm'] for r in _cora_history if r.get('lm') is not None]
    if _lm_vals:
        print(f'  LM loss final  CORA 5M     : {_lm_vals[-1]:.4f}')

# ── 5 ejemplos comparativos con re-evaluacion ─────────────────────────────────
print('\\n' + '=' * 72)
print('  5 EJEMPLOS COMPARATIVOS  (seed=0, re-evaluacion en modelos finales)')
print('=' * 72)

_rng  = random.Random(0)
_idxs = sorted(_rng.sample(range(len(EVAL_EX)), min(5, len(EVAL_EX))))

for _rank, _idx in enumerate(_idxs, 1):
    _ex    = EVAL_EX[_idx]
    _q_ids = SHARED_VOCAB.to_tensor(
        SHARED_VOCAB.encode(_ex.problem_text, _MAX_Q), _DEVICE, _MAX_Q)

    # Transformer
    _tf_gen = tf_generate(TF_MODEL, _q_ids, SHARED_VOCAB, _DEVICE)
    _tf_f1  = word_f1(_tf_gen, _ex.answer)

    # CORA
    if _cora_ok:
        _dummy = torch.zeros(1, 1, dtype=torch.long, device=_DEVICE)
        with torch.no_grad():
            _co, _gr, _cf, _nv, _, _enc = trainer.forward_full(
                CORA_ENCODER, CORA_CRYSTALLIZER, CORA_CRE,
                CORA_SCRATCH_PAD, CORA_DECODER,
                _q_ids, _dummy, CORA_CFG, 3, SHARED_VOCAB,
            )
            _cora_gen = trainer.generate_sampled(
                CORA_DECODER, _gr, SHARED_VOCAB, device=_DEVICE)
        _cora_f1 = word_f1(_cora_gen, _ex.answer)
    else:
        _cora_gen = '(ejecuta celda CORA primero)'; _cora_f1 = 0.0

    print(f'\\n  [{_rank}] L{_ex.complexity_level}  {_ex.problem_text[:65]}...')
    print(f'       GT   : {_ex.answer[:72]}')
    print(f'       CORA : {_cora_gen[:72] or "(vacio)"}  F1={_cora_f1:.0%}')
    print(f'       TF   : {_tf_gen[:72]   or "(vacio)"}  F1={_tf_f1:.0%}')

# ── Plots ──────────────────────────────────────────────────────────────────────
_fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(14, 5))

def _smooth(vals, w=30):
    return [sum(vals[max(0, i-w):i+1]) / len(vals[max(0, i-w):i+1])
            for i in range(len(vals))]

# LM loss curves
if _tf_history:
    _ts = [r['step'] for r in _tf_history]
    _tv = [r['lm']   for r in _tf_history]
    _ax0.plot(_ts, _tv, alpha=0.15, color='steelblue', lw=0.8)
    _ax0.plot(_ts, _smooth(_tv), color='steelblue', lw=2,
              label=f'Transformer ({globals().get("TF_PARAMS", 0)/1e6:.1f}M)')

if _cora_history:
    _cs2 = [r['step'] for r in _cora_history if r.get('lm') is not None]
    _cv2 = [r['lm']   for r in _cora_history if r.get('lm') is not None]
    if _cs2:
        _ax0.plot(_cs2, _cv2, alpha=0.15, color='coral', lw=0.8)
        _ax0.plot(_cs2, _smooth(_cv2), color='coral', lw=2, label='CORA 5M')

_ax0.set_title('LM Loss (MA-30)')
_ax0.set_xlabel('Step'); _ax0.set_ylabel('Loss')
_ax0.legend(); _ax0.grid(alpha=0.3)

# Word F1 comparison
if _all_steps:
    if _cora_by_step:
        _ax1.plot(list(_cora_by_step), list(_cora_by_step.values()),
                  'o-', color='coral', lw=2, ms=7, label='CORA 5M')
    if _tf_by_step:
        _ax1.plot(list(_tf_by_step), list(_tf_by_step.values()),
                  's-', color='steelblue', lw=2, ms=7, label='Transformer')
    _ax1.set_title('Word F1 en eval')
    _ax1.set_xlabel('Step')
    _ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    _ax1.legend(); _ax1.grid(alpha=0.3)

_fig.suptitle('CORA 5M vs Transformer Baseline', fontsize=14, fontweight='bold')
_fig.tight_layout()

try:
    _out_png = os.path.join(DRIVE_DIR, 'comparison.png')
    _fig.savefig(_out_png, dpi=150, bbox_inches='tight')
    print(f'\\nGrafico guardado: {_out_png}')
except Exception:
    pass
plt.show()"""


# ═════════════════════════════════════════════════════════════════════════════
# Ensamblar notebook
# ═════════════════════════════════════════════════════════════════════════════

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
            "name": "benchmark_cora_vs_transformer.ipynb",
            "provenance": [],
        },
    },
    "cells": [
        mk_md(CELL1_MD,   "cell-b01"),
        mk_code(CELL2_CODE, "cell-b02"),
        # Cell 3 — 28 archivos en base64, reutilizado del notebook lento
        {
            "cell_type": "code",
            "id": "cell-b03",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": cell3_source,
        },
        mk_code(CELL4_CODE, "cell-b04"),
        mk_code(CELL5_CODE, "cell-b05"),
        mk_code(CELL6_CODE, "cell-b06"),
        mk_code(CELL7_CODE, "cell-b07"),
    ],
}

out_path = BASE / "benchmark_cora_vs_transformer.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

size   = os.path.getsize(out_path)
n_cells = len(nb["cells"])
print(f"Notebook generado: {out_path}")
print(f"Tamanio: {size:,} bytes  ({size/1024:.0f} KB)")
print(f"Celdas: {n_cells}")
for i, cell in enumerate(nb["cells"], 1):
    src = cell["source"]
    n   = len(src) if isinstance(src, list) else src.count("\n")
    print(f"  Cell {i} [{cell['cell_type']:8}]  {n:5} lines")
