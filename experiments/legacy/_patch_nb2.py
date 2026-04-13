import sys; sys.stdout.reconfigure(encoding='utf-8')
import json

with open('benchmark_cora_vs_transformer.ipynb','r',encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 0: title ─────────────────────────────────────────────────────────────
nb['cells'][0]['source'] = (
    '# CORA 5M vs Transformer Baseline \u2014 Benchmark mismos datos (T4)\n\n'
    'Dos arquitecturas, **exactamente los mismos 2000 ejemplos**, batch=1 sin batching.\n\n'
    '**Fix de grounding léxico**: cada `HybridDecoderLayer` ahora tiene doble cross-attention:\n'
    '1. al grafo causal (estructura) 2. a los concept vectors del encoder (identidad léxica).\n\n'
    '| Modelo | Arquitectura | Batch | Strategy |\n'
    '|--------|-------------|-------|----------|\n'
    '| **Transformer** | 4enc+4dec est\u00e1ndar | 1 (igual que CORA) | Secuencial FP16 |\n'
    '| **CORA 5M** | Mamba+Crystallizer+CRE+Decoder(dual cross-attn) | 1 | Secuencial FP16 |\n\n'
    'Pregunta: dados los **mismos datos**, **\u00bfqui\u00e9n aprende mejor?**\n'
    'Bonus: \u00bfcu\u00e1nto m\u00e1s r\u00e1pido es el Transformer?\n'
)

# ── Cell 3: setup ─────────────────────────────────────────────────────────────
nb['cells'][3]['source'] = """\
import sys, os, time, random, math, torch
from collections import Counter

# \u2500\u2500 Device \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
try:
    import torch_xla.core.xla_model as xm
    DEVICE  = xm.xla_device()
    BACKEND = 'TPU/XLA'
    USE_AMP = False
    IS_CUDA = False
    print('[device] TPU/XLA detectado')
except ImportError:
    if not torch.cuda.is_available():
        raise RuntimeError(
            '\\n\u26a0 No GPU detectada.\\nRuntime -> Change runtime type -> T4 GPU')
    DEVICE  = torch.device('cuda')
    BACKEND = torch.cuda.get_device_name(0)
    USE_AMP = True
    IS_CUDA = True
    _free, _total = torch.cuda.mem_get_info()
    print(f'[device] {BACKEND}')
    print(f'[vram]   {_free/1e9:.1f} GB libres / {_total/1e9:.1f} GB total')

sys.path.insert(0, '/content/aion_c')
from synth.causal_graph_gen import CausalGraphGenerator

# \u2500\u2500 Vocabulario compartido \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
class SimpleVocab:
    PAD_ID=0; BOS_ID=1; EOS_ID=2; UNK_ID=3; N_SPECIAL=4
    def __init__(self, max_size=8000):
        self.max_size=max_size
        self.word2id={'<PAD>':0,'<BOS>':1,'<EOS>':2,'<UNK>':3}
        self.id2word={v:k for k,v in self.word2id.items()}
        self._counts=Counter()
    def add_texts(self,texts):
        for t in texts: self._counts.update(t.lower().split())
    def build(self):
        for word,_ in self._counts.most_common(self.max_size-self.N_SPECIAL):
            if word not in self.word2id:
                idx=len(self.word2id); self.word2id[word]=idx; self.id2word[idx]=word
    def encode(self,text,max_len=128,add_bos=False,add_eos=False):
        ids=[]
        if add_bos: ids.append(self.BOS_ID)
        ids.extend(self.word2id.get(w,self.UNK_ID) for w in text.lower().split()[:max_len])
        if add_eos: ids.append(self.EOS_ID)
        return ids or [self.UNK_ID]
    def decode(self,ids):
        skip={self.PAD_ID,self.BOS_ID,self.EOS_ID}
        return ' '.join(self.id2word.get(i,'<UNK>') for i in ids if i not in skip)
    def to_tensor(self,ids,device,max_len=None,pad_to=None):
        if max_len: ids=ids[:max_len]
        if pad_to and len(ids)<pad_to: ids=ids+[self.PAD_ID]*(pad_to-len(ids))
        return torch.tensor(ids,dtype=torch.long,device=device).unsqueeze(0)
    def __len__(self): return len(self.word2id)

# \u2500\u2500 Dataset: 2000 de train, 500 de eval (seed=42, reproducible) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print('[data] Generando 5000 ejemplos L1-4...')
_t0=time.perf_counter()
_gen=CausalGraphGenerator(seed=42)
_all=_gen.generate_batch(n=5000,level_distribution={1:.30,2:.30,3:.25,4:.15})
_rng=random.Random(42); _rng.shuffle(_all)
TRAIN_EX = _all[:2000]   # mismos 2000 ejemplos para ambos modelos
EVAL_EX  = _all[2000:2500]
print(f'[data] {len(TRAIN_EX)} train / {len(EVAL_EX)} eval  ({time.perf_counter()-_t0:.1f}s)')

SHARED_VOCAB=SimpleVocab(max_size=8000)
SHARED_VOCAB.add_texts([e.problem_text for e in _all])
SHARED_VOCAB.add_texts([e.answer       for e in _all])
SHARED_VOCAB.build()
ACTUAL_VOCAB=len(SHARED_VOCAB)
print(f'[vocab] {ACTUAL_VOCAB} tokens')

# \u2500\u2500 Benchmark config \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
TOTAL_STEPS  = 2000     # ambos modelos ven exactamente estos ejemplos
EVAL_EVERY   = 400      # eval cada N steps
_MAX_Q=80; _MAX_A=48
print(f'[bench] TOTAL_STEPS={TOTAL_STEPS}  EVAL_EVERY={EVAL_EVERY}')
print('[bench] Ambos modelos: batch=1, sin gradient accumulation')
"""

# ── Cell 4: Transformer ────────────────────────────────────────────────────────
nb['cells'][4]['source'] = """\
import math, time, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from collections import Counter

torch.cuda.empty_cache()
_free0,_=torch.cuda.mem_get_info()
print(f'[Transformer] VRAM libre: {_free0/1e9:.2f} GB')

# \u2500\u2500 Modelo \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, n_enc=4, n_dec=4,
                 dim_ff=1024, max_len=256, dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.src_emb=nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tgt_emb=nn.Embedding(vocab_size, d_model, padding_idx=0)
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
        enc_layer=nn.TransformerEncoderLayer(d_model,nhead,dim_ff,dropout,batch_first=True,norm_first=True)
        dec_layer=nn.TransformerDecoderLayer(d_model,nhead,dim_ff,dropout,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(enc_layer,n_enc)
        self.decoder=nn.TransformerDecoder(dec_layer,n_dec)
        self.proj=nn.Linear(d_model,vocab_size)
        self._init()
    def _init(self):
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)
    def _pe(self,x): return x+self.pe[:,:x.size(1)]
    def forward(self,src,tgt,src_key_padding_mask=None):
        S=tgt.size(1)
        tgt_mask=nn.Transformer.generate_square_subsequent_mask(S,device=src.device)
        se=self._pe(self.src_emb(src)*math.sqrt(self.d_model))
        te=self._pe(self.tgt_emb(tgt)*math.sqrt(self.d_model))
        mem=self.encoder(se,src_key_padding_mask=src_key_padding_mask)
        out=self.decoder(te,mem,tgt_mask=tgt_mask,
                         memory_key_padding_mask=src_key_padding_mask)
        return self.proj(out)

def word_f1(pred_ids, gold_ids, skip={0,1,2}):
    p=[i for i in pred_ids if i not in skip]
    g=[i for i in gold_ids  if i not in skip]
    if not p and not g: return 1.0
    if not p or  not g: return 0.0
    pc=Counter(p); gc=Counter(g)
    overlap=sum((pc&gc).values())
    prec=overlap/len(p); rec=overlap/len(g)
    if prec+rec==0: return 0.0
    return 2*prec*rec/(prec+rec)

def evaluate_tf(model):
    model.eval()
    wf1s=[]; losses=[]
    with torch.no_grad():
        for ex in EVAL_EX[:200]:
            src_ids=SHARED_VOCAB.encode(ex.problem_text,max_len=_MAX_Q,add_bos=True,add_eos=True)
            tgt_ids=SHARED_VOCAB.encode(ex.answer,max_len=_MAX_A,add_bos=True,add_eos=True)
            src=torch.tensor(src_ids,dtype=torch.long,device=DEVICE).unsqueeze(0)
            tgt=torch.tensor(tgt_ids,dtype=torch.long,device=DEVICE).unsqueeze(0)
            src_pad=(src==0)
            with autocast(enabled=USE_AMP):
                logits=model(src,tgt[:,:-1],src_key_padding_mask=src_pad)
                loss=F.cross_entropy(logits.reshape(-1,ACTUAL_VOCAB),tgt[:,1:].reshape(-1),ignore_index=0)
            losses.append(loss.item())
            pred=logits.argmax(-1).squeeze(0).tolist()
            wf1s.append(word_f1(pred,tgt_ids[1:]))
    model.train()
    return sum(losses)/len(losses), sum(wf1s)/len(wf1s)

# \u2500\u2500 Instanciar \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
tf_model = TransformerBaseline(ACTUAL_VOCAB).to(DEVICE)
tf_opt   = torch.optim.AdamW(tf_model.parameters(), lr=3e-4, weight_decay=1e-2)
scaler   = GradScaler(enabled=USE_AMP)
n_params = sum(p.numel() for p in tf_model.parameters())
print(f'[Transformer] {n_params:,} params')

# \u2500\u2500 Training: exactamente TOTAL_STEPS=2000 ejemplos, batch=1 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
TF_EVALS  = []   # [(step, val_loss, val_wf1)]
TF_LOSSES = []   # train loss por step
train_cycle = TRAIN_EX.copy()
random.shuffle(train_cycle)
train_idx = 0

t_start = time.perf_counter()
for step in range(1, TOTAL_STEPS + 1):
    ex = train_cycle[train_idx % len(train_cycle)]
    train_idx += 1

    src_ids = SHARED_VOCAB.encode(ex.problem_text, max_len=_MAX_Q, add_bos=True, add_eos=True)
    tgt_ids = SHARED_VOCAB.encode(ex.answer,       max_len=_MAX_A, add_bos=True, add_eos=True)
    src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    tgt = torch.tensor(tgt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    src_pad = (src == 0)

    tf_opt.zero_grad()
    with autocast(enabled=USE_AMP):
        logits = tf_model(src, tgt[:,:-1], src_key_padding_mask=src_pad)
        loss   = F.cross_entropy(logits.reshape(-1, ACTUAL_VOCAB),
                                 tgt[:,1:].reshape(-1), ignore_index=0)
    scaler.scale(loss).backward()
    scaler.unscale_(tf_opt)
    torch.nn.utils.clip_grad_norm_(tf_model.parameters(), 1.0)
    scaler.step(tf_opt); scaler.update()
    TF_LOSSES.append(loss.item())

    if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:
        elapsed = time.perf_counter() - t_start
        val_loss, val_wf1 = evaluate_tf(tf_model)
        TF_EVALS.append((step, val_loss, val_wf1))
        avg = sum(TF_LOSSES[-EVAL_EVERY:]) / len(TF_LOSSES[-EVAL_EVERY:])
        print(f'[TF] step={step:>4}/{TOTAL_STEPS}  t={elapsed:5.1f}s  '
              f'train={avg:.4f}  val={val_loss:.4f}  wf1={val_wf1:.3f}')

TF_TOTAL_TIME = time.perf_counter() - t_start
TF_FINAL_LOSS = TF_EVALS[-1][1]
TF_FINAL_WF1  = TF_EVALS[-1][2]
TF_EX_PER_SEC = TOTAL_STEPS / TF_TOTAL_TIME
print(f'\\n[Transformer] COMPLETADO')
print(f'  tiempo : {TF_TOTAL_TIME:.1f}s  ({TF_EX_PER_SEC:.1f} ex/s)')
print(f'  loss   : {TF_FINAL_LOSS:.4f}   wf1: {TF_FINAL_WF1:.3f}')
"""

# ── Cell 5: CORA ──────────────────────────────────────────────────────────────
nb['cells'][5]['source'] = """\
import math, time, random, torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

torch.cuda.empty_cache()
_free0,_=torch.cuda.mem_get_info()
print(f'[CORA] VRAM libre: {_free0/1e9:.2f} GB')

# \u2500\u2500 Imports del pipeline AION-C \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Carpetas reales: encoder/, crystallizer/, cre/, decoder/, router/
from encoder          import StreamEncoder
from crystallizer     import GraphCrystallizer
from cre             import CausalReasoningEngine
from decoder         import StreamDecoder
from router.pipeline import CORAConfig

# \u2500\u2500 Construir CORA con CORAConfig \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# CORAConfig garantiza que hidden_dim fluye consistentemente por todos los
# submódulos (encoder.concept_dim == crystallizer.hidden_dim == cre.node_dim
# == decoder.node_dim == decoder.hidden_dim).
CORA_CFG = CORAConfig(
    hidden_dim   = 256,
    vocab_size   = ACTUAL_VOCAB,
    # Encoder: Mamba SSM, 4 capas
    enc_n_layers  = 4, enc_state_dim = 16, enc_expand = 2,
    enc_d_conv    = 4, enc_ffn_mult  = 4,
    # Crystallizer
    cryst_max_nodes      = 32,  cryst_n_heads        = 8,
    cryst_node_threshold = 0.01, cryst_edge_threshold = 0.01,
    # CRE: 2 capas de MP, 3 iteraciones (suficiente para benchmark)
    cre_edge_dim         = 64,  cre_message_dim      = 128,
    cre_n_message_layers = 2,   cre_max_iterations   = 3,
    # ScratchPad
    pad_n_slots  = 16, pad_slot_dim = 128,
    # Decoder: 2 capas (más ligero para benchmark)
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

K_NODES = CORA_CFG.cryst_max_nodes
D_MODEL = CORA_CFG.hidden_dim

# \u2500\u2500 Forward pass completo: q\u2192 encoder \u2192 crystallizer \u2192 CRE \u2192 decoder \u2500\u2500\u2500\u2500\u2500\u2500
def cora_forward(src_ids, tgt_ids):
    \"\"\"
    Pipeline seq2seq:
        encoder(q) \u2192 crystallizer \u2192 CRE \u2192 graph_repr \u2192 decoder(a)
    Devuelve (logits [1, L-1, V], loss escalar) o (None, None) si grafo vac\u00edo.
    \"\"\"
    src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    tgt = torch.tensor(tgt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    # 1. Encode pregunta \u2192 concept vectors [1, L_q, D]
    concepts = enc(src)

    # 2. Crystallizer: concept vectors \u2192 CrystallizerOutput
    crystal = crys(concepts)
    n_valid = crystal.node_counts[0]   # int

    # 3. CRE: refinar node features
    if n_valid == 0:
        # Sin nodos: representaci\u00f3n de ceros (gradiente fluye desde decoder)
        cre_feats = torch.zeros(1, D_MODEL, device=DEVICE, dtype=concepts.dtype)
    else:
        node_feats = crystal.node_vectors[0, :n_valid, :]   # [n_valid, D]
        cre_out    = cre(crystal.graphs[0], node_feats)     # CREOutput
        cre_feats  = cre_out.node_features                  # [n_valid, D]

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
    #    dec_input  = [BOS, a0, ..., a_{T-2}]  = tgt[:, :-1]
    #    dec_target = [a0,  a1, ..., a_{T-1}]  = tgt[:, 1:]
    #    concepts pasados para que el decoder preserve identidad léxica
    dec_input = tgt[:, :-1]                                 # [1, L-1]
    dec_out   = dec(dec_input, graph_repr, concepts)        # DecoderOutput
    logits    = dec_out.logits                              # [1, L-1, V]

    loss = F.cross_entropy(
        logits.reshape(-1, ACTUAL_VOCAB),
        tgt[:, 1:].reshape(-1),
        ignore_index=0,
    )
    return logits, loss

# \u2500\u2500 Evaluaci\u00f3n \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def evaluate_cora():
    enc.eval(); crys.eval(); cre.eval(); dec.eval()
    losses=[]; wf1s=[]
    with torch.no_grad():
        for ex in EVAL_EX[:100]:
            src_ids=SHARED_VOCAB.encode(ex.problem_text,max_len=_MAX_Q,add_bos=True,add_eos=True)
            tgt_ids=SHARED_VOCAB.encode(ex.answer,max_len=_MAX_A,add_bos=True,add_eos=True)
            with autocast(enabled=USE_AMP):
                logits, loss = cora_forward(src_ids, tgt_ids)
            if loss is None:
                losses.append(9.9); wf1s.append(0.0); continue
            losses.append(loss.item())
            pred = logits.argmax(-1).squeeze(0).tolist()
            wf1s.append(word_f1(pred, tgt_ids[1:]))
    enc.train(); crys.train(); cre.train(); dec.train()
    return sum(losses)/len(losses), sum(wf1s)/len(wf1s)

# \u2500\u2500 Training: exactamente TOTAL_STEPS=2000 ejemplos, batch=1 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
CORA_EVALS  = []
CORA_LOSSES = []
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
        _, loss = cora_forward(src_ids, tgt_ids)

    if loss is None:
        CORA_LOSSES.append(0.0)
        continue

    cora_scaler.scale(loss).backward()
    cora_scaler.unscale_(cora_opt)
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    cora_scaler.step(cora_opt); cora_scaler.update()
    CORA_LOSSES.append(loss.item())

    if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:
        elapsed = time.perf_counter() - t_start
        val_loss, val_wf1 = evaluate_cora()
        CORA_EVALS.append((step, val_loss, val_wf1))
        avg = sum(CORA_LOSSES[-EVAL_EVERY:]) / max(1, len(CORA_LOSSES[-EVAL_EVERY:]))
        print(f'[CORA] step={step:>4}/{TOTAL_STEPS}  t={elapsed:5.1f}s  '
              f'train={avg:.4f}  val={val_loss:.4f}  wf1={val_wf1:.3f}')

CORA_TOTAL_TIME = time.perf_counter() - t_start
CORA_FINAL_LOSS = CORA_EVALS[-1][1]
CORA_FINAL_WF1  = CORA_EVALS[-1][2]
CORA_EX_PER_SEC = TOTAL_STEPS / CORA_TOTAL_TIME
print(f'\\n[CORA] COMPLETADO')
print(f'  tiempo : {CORA_TOTAL_TIME:.1f}s  ({CORA_EX_PER_SEC:.1f} ex/s)')
print(f'  loss   : {CORA_FINAL_LOSS:.4f}   wf1: {CORA_FINAL_WF1:.3f}')
"""

# ── Cell 6: comparison ────────────────────────────────────────────────────────
nb['cells'][6]['source'] = """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# \u2500\u2500 Tabla resumen \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
speed_ratio = TF_EX_PER_SEC / max(CORA_EX_PER_SEC, 0.01)
wf1_winner  = 'Transformer' if TF_FINAL_WF1  >= CORA_FINAL_WF1  else 'CORA'
loss_winner = 'Transformer' if TF_FINAL_LOSS <= CORA_FINAL_LOSS else 'CORA'

print('=' * 64)
print(f'  BENCHMARK: {TOTAL_STEPS} ejemplos identicos, batch=1, sin batching')
print('=' * 64)
print(f'  {"Metric":<24} {"Transformer":>16} {"CORA 5M":>16}')
print(f'  {"-"*24} {"-"*16} {"-"*16}')
print(f'  {"Ejemplos vistos":<24} {TOTAL_STEPS:>16,} {TOTAL_STEPS:>16,}')
print(f'  {"Tiempo total (s)":<24} {TF_TOTAL_TIME:>16.1f} {CORA_TOTAL_TIME:>16.1f}')
print(f'  {"Throughput (ex/s)":<24} {TF_EX_PER_SEC:>16.1f} {CORA_EX_PER_SEC:>16.1f}')
print(f'  {"Loss final (val)":<24} {TF_FINAL_LOSS:>16.4f} {CORA_FINAL_LOSS:>16.4f}')
print(f'  {"Word F1 final":<24} {TF_FINAL_WF1:>16.3f} {CORA_FINAL_WF1:>16.3f}')
print('=' * 64)
print(f'  Transformer es {speed_ratio:.1f}x mas rapido que CORA')
print(f'  Mejor calidad (loss)    : {loss_winner}')
print(f'  Mejor calidad (Word F1) : {wf1_winner}')
print('=' * 64)

# \u2500\u2500 Gr\u00e1ficas: loss y Word F1 por step \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

tf_steps   = [e[0] for e in TF_EVALS]
tf_losses  = [e[1] for e in TF_EVALS]
tf_wf1s    = [e[2] for e in TF_EVALS]
cora_steps = [e[0] for e in CORA_EVALS]
cora_losses= [e[1] for e in CORA_EVALS]
cora_wf1s  = [e[2] for e in CORA_EVALS]

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

# \u2500\u2500 Gr\u00e1ficas vs tiempo real (bonus: muestra diferencia de velocidad) \u2500\u2500\u2500\u2500\u2500\u2500
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
print('\\n' + '='*64)
print('  5 EJEMPLOS COMPARATIVOS')
print('='*64)

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

        # CORA greedy (re-usa cora_forward con teacher forcing para eval)
        with autocast(enabled=USE_AMP):
            logits_cr, _ = cora_forward(src_ids, tgt_ids)
        cora_pred_ids = logits_cr.argmax(-1).squeeze(0).tolist() if logits_cr is not None else []
        cora_pred     = SHARED_VOCAB.decode(cora_pred_ids)
        cora_f1       = word_f1(cora_pred_ids, tgt_ids[1:])

        print(f'\\nEj {i+1}: {ex.problem_text[:80]}')
        print(f'  Gold       : {gold}')
        print(f'  Transformer: {tf_pred}  [WF1={tf_f1:.2f}]')
        print(f'  CORA       : {cora_pred}  [WF1={cora_f1:.2f}]')
print('='*64)
"""

# ── Verification ──────────────────────────────────────────────────────────────
c3 = nb['cells'][3]['source']
c4 = nb['cells'][4]['source']
c5 = nb['cells'][5]['source']
c6 = nb['cells'][6]['source']

checks = [
    # Cell 3
    ('Cell3 TOTAL_STEPS=2000',           'TOTAL_STEPS  = 2000'                  in c3),
    ('Cell3 EVAL_EVERY=400',             'EVAL_EVERY   = 400'                   in c3),
    ('Cell3 no time budget',             'TRAIN_BUDGET_SECS'         not in c3),
    ('Cell3 TRAIN_EX=2000',              'TRAIN_EX = _all[:2000]'               in c3),
    # Cell 4
    ('Cell4 for loop steps',             'for step in range(1, TOTAL_STEPS + 1)' in c4),
    ('Cell4 TF_TOTAL_TIME',              'TF_TOTAL_TIME'                        in c4),
    ('Cell4 word_f1 defined',            'def word_f1'                          in c4),
    # Cell 5 — imports corrected
    ('Cell5 from encoder import',        'from encoder          import StreamEncoder' in c5),
    ('Cell5 from crystallizer import',   'from crystallizer     import GraphCrystallizer' in c5),
    ('Cell5 from cre import',            'from cre             import CausalReasoningEngine' in c5),
    ('Cell5 from decoder import',        'from decoder         import StreamDecoder' in c5),
    ('Cell5 from router.pipeline',       'from router.pipeline import CORAConfig'    in c5),
    ('Cell5 no stream_encoder',          'stream_encoder'            not in c5),
    ('Cell5 no stream_decoder',          'stream_decoder'            not in c5),
    # Cell 5 — correct API
    ('Cell5 CORAConfig used',            'CORA_CFG = CORAConfig('                in c5),
    ('Cell5 crystal.node_counts',        'crystal.node_counts[0]'               in c5),
    ('Cell5 crystal.node_vectors',       'crystal.node_vectors[0'               in c5),
    ('Cell5 cre(crystal.graphs[0]',      'cre(crystal.graphs[0], node_feats)'   in c5),
    ('Cell5 cre_out.node_features',      'cre_out.node_features'                in c5),
    ('Cell5 dec_out.logits',             'dec_out.logits'                       in c5),
    ('Cell5 teacher forcing tgt[:,:-1]', 'tgt[:, :-1]'                         in c5),
    ('Cell5 concepts passed to dec',     'dec(dec_input, graph_repr, concepts)' in c5),
    ('Cell5 CORA_TOTAL_TIME',            'CORA_TOTAL_TIME'                      in c5),
    # Cell 6
    ('Cell6 speed_ratio',                'speed_ratio'                          in c6),
    ('Cell6 step plot',                  'Loss vs Step'                         in c6),
    ('Cell6 time plot',                  'Loss vs Tiempo real'                  in c6),
    ('Cell6 5 examples',                 'EJEMPLOS COMPARATIVOS'                in c6),
]

print('Verification:')
all_ok = True
for label, ok in checks:
    mark = 'OK  ' if ok else 'FAIL'
    print(f'  {mark} {label}')
    if not ok: all_ok = False

if all_ok:
    with open('benchmark_cora_vs_transformer.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('\nALL OK — notebook guardado')
else:
    print('\nFAILED — notebook NO guardado')
    raise SystemExit(1)
