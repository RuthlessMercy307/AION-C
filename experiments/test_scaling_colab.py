"""
experiments/test_scaling_colab.py
==================================
Verificación de escalado de MoSE: parámetros reales, VRAM y throughput.

Uso:
  Copiar este archivo a un notebook de Google Colab y ejecutar celda por celda,
  o ejecutar directamente:

      cd AION-C
      python -m experiments.test_scaling_colab

Funciona en:
  - CPU  (tu PC): verifica parámetros y forward pass; sin timing de GPU
  - CUDA (Colab): verifica todo incluyendo VRAM real y throughput en GPU

Secuencia:
  1. Tiny:  MoSEScaleConfig.tiny()  -> MoSEPipeline(MoSEConfig.tiny())
  2. Medium: MoSEScaleConfig.medium() -> MoSEPipeline en bf16 (reduce si OOM)
  3. Forward + backward con 3 ejemplos de Opus -> throughput
  4. Comparación VRAM real vs estimada
  5. Extrapolación a production (3.5B) -> H200 fit analysis
  6. Tabla resumen final

Nota: MoSEConfig (usado para instanciar MoSEPipeline) tiene un único
hidden_dim para todos los componentes, mientras que MoSEScaleConfig tiene
dimensiones asimétricas (enc_dim ≠ dec_dim en producción). Los conteos
de parámetros difieren; se reportan ambos.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS — todos los necesarios para Colab (autocontenido)
# ─────────────────────────────────────────────────────────────────────────────

import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Asegurar que el root de AION-C está en el path (para Colab)
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from router.pipeline    import MoSEPipeline, MoSEConfig
from router.config_3_5b import MoSEScaleConfig

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

H200_VRAM_GB   = 141.0   # NVIDIA H200 SXM5
T4_VRAM_GB     =  15.0   # Colab T4 gratuito
PAD            = 0
SEQ_LEN        = 64      # longitud de secuencia para benchmarks
BATCH_SIZE     = 3       # batch para el forward pass de timing

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_CUDA = DEVICE.type == "cuda"

print(f"Device : {DEVICE}")
if HAS_CUDA:
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM   : {total_vram:.1f} GB total")
    # bf16 está disponible en todas las GPU Ampere+ (A100, T4, etc.)
    USE_BF16 = torch.cuda.is_bf16_supported()
    AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
    print(f"AMP    : {AMP_DTYPE}")
else:
    USE_BF16 = False
    AMP_DTYPE = torch.float32
    total_vram = None
    print("Modo CPU — sin timing de GPU, VRAM estimada")

print()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class _SimpleTokenizer:
    """
    Tokenizador word-level mínimo para el benchmark.
    Cargado solo si AIONTokenizer no está disponible.
    """

    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = vocab_size
        self._w2i: Dict[str, int] = {}
        self._i2w: Dict[int, str] = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>"}

    def build_vocab(self, texts: List[str]) -> "_SimpleTokenizer":
        words = {w for t in texts for w in t.lower().split()}
        for i, w in enumerate(sorted(words), start=4):
            if i >= self.vocab_size:
                break
            self._w2i[w] = i
            self._i2w[i] = w
        return self

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        toks = [self.BOS]
        for w in text.lower().split():
            toks.append(self._w2i.get(w, self.UNK))
        toks.append(self.EOS)
        return toks[:max_len]

    def decode(self, ids: List[int]) -> str:
        return " ".join(
            self._i2w.get(i, "<unk>")
            for i in ids
            if i not in (self.PAD, self.BOS, self.EOS)
        )

    def to_tensor(self, ids: List[int]) -> torch.Tensor:
        return torch.tensor([ids], dtype=torch.long)


def _load_tokenizer(vocab_size: int = 512):
    """
    Carga AIONTokenizer si vocab_size=32000 y el modelo existe;
    sino usa _SimpleTokenizer con el vocab_size especificado.
    Retorna (tok, source).
    """
    model_path = _ROOT / "tokenizer" / "aion_32k.model"
    if vocab_size == 32_000 and model_path.exists():
        try:
            from tokenizer import AIONTokenizer

            class _WrappedAION:
                """Wrapper para compatibilidad con la API to_tensor/encode(text, max_len)."""
                def __init__(self, inner):
                    self._tok = inner
                    self.vocab_size = inner.vocab_size
                    self.PAD = inner.pad_id
                    self.EOS = inner.eos_id
                    self.BOS = inner.bos_id

                def encode(self, text: str, max_len: int = 512) -> List[int]:
                    ids = self._tok.encode(text)
                    return ids[:max_len]

                def decode(self, ids: List[int]) -> str:
                    return self._tok.decode(ids)

                def to_tensor(self, ids: List[int]) -> torch.Tensor:
                    return torch.tensor([ids], dtype=torch.long)

            return _WrappedAION(AIONTokenizer(model_path)), "AIONTokenizer(32K)"
        except Exception:
            pass

    # Fallback: tokenizador word-level mínimo con vocab = vocab_size
    tok = _SimpleTokenizer(vocab_size=vocab_size)
    # Seed de texto para construir vocab
    seed = "the quick brown fox causal graph system model encoder decoder motor"
    tok.build_vocab([seed * 50])
    return tok, f"SimpleTokenizer({vocab_size})"


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _model_bytes(model: nn.Module) -> int:
    """Bytes de los pesos del modelo (basado en dtype real)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _vram_allocated_gb() -> float:
    """VRAM actualmente asignada en GPU (GB). 0 en CPU."""
    if HAS_CUDA:
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def _vram_peak_gb() -> float:
    """VRAM pico desde la última llamada a _reset_vram_stats()."""
    if HAS_CUDA:
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def _reset_vram_stats() -> None:
    if HAS_CUDA:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _cpu_vram_estimate_gb(model: nn.Module, training: bool = False) -> float:
    """
    Estima VRAM en CPU usando tamaño de pesos del modelo.
    En CPU no hay GPU, así que esto es una aproximación del tamaño en memoria RAM.
    """
    bytes_model = _model_bytes(model)
    if training:
        # Gradientes (~1×) + estados optimizer AdamW (~2× fp32) + activaciones (~3×)
        return bytes_model * 6 / 1e9
    return bytes_model / 1e9


def _load_opus_batch(
    tok,
    batch_size: int = 3,
    seq_len: int = SEQ_LEN,
    motors: Tuple[str, ...] = ("cora", "axiom", "empathy"),
) -> Optional[torch.Tensor]:
    """
    Carga batch_size ejemplos de OpusDataset, tokeniza y devuelve tensor [B, seq_len].
    Retorna None si el dataset no está disponible.
    """
    try:
        from experiments.opus_dataset import OpusDataset
        ids_batch = []
        for motor in motors[:batch_size]:
            ds = OpusDataset(motor=motor, max_examples=20)
            ex = ds.generate()
            text = ex.problem_text + " " + ex.answer
            ids = tok.encode(text, seq_len)
            # Pad hasta seq_len
            ids = ids[:seq_len] + [PAD] * max(0, seq_len - len(ids))
            ids_batch.append(ids)
        return torch.tensor(ids_batch, dtype=torch.long)
    except Exception as e:
        print(f"  [WARNING] OpusDataset no disponible ({e}). Usando batch sintético.")
        return None


def _synthetic_batch(vocab_size: int, batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """Batch sintético para cuando el dataset no está disponible."""
    return torch.randint(1, max(2, vocab_size - 1), (batch_size, seq_len))


def _make_batch(
    tok,
    vocab_size: int,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
) -> Tuple[torch.Tensor, str]:
    """Retorna (tensor [B, L], source_label)."""
    batch = _load_opus_batch(tok, batch_size=batch_size, seq_len=seq_len)
    if batch is not None:
        return batch, "OpusDataset"
    return _synthetic_batch(vocab_size, batch_size, seq_len), "synthetic"


def _time_forward(
    model: nn.Module,
    ids: torch.Tensor,
    n_warmup: int = 2,
    n_runs: int = 5,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> Tuple[float, float]:
    """
    Mide tiempo de forward pass.
    Retorna (tiempo_medio_ms, throughput_ejemplos_por_segundo).
    """
    device = next(model.parameters()).device
    ids = ids.to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _ = model(ids)
        if HAS_CUDA:
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _ = model(ids)
            if HAS_CUDA:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    model.train()
    mean_ms = (sum(times) / len(times)) * 1000
    throughput = ids.shape[0] / (mean_ms / 1000)
    return mean_ms, throughput


def _time_backward(
    model: nn.Module,
    ids: torch.Tensor,
    n_warmup: int = 1,
    n_runs: int = 3,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> Tuple[float, float]:
    """
    Mide tiempo de forward + backward pass.
    Retorna (tiempo_medio_ms, throughput_ejemplos_por_segundo).
    """
    device = next(model.parameters()).device
    ids = ids.to(device)
    model.train()

    def _step():
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(ids)
            logits = out.logits  # [B, L, V]
            # loss en primer elemento del batch (ids[0])
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                ids[:, 1:].reshape(-1),
                ignore_index=PAD,
            )
        loss.backward()
        return loss.item()

    # Warmup
    for _ in range(n_warmup):
        model.zero_grad(set_to_none=True)
        _step()
    if HAS_CUDA:
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        model.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        _step()
        if HAS_CUDA:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_ms = (sum(times) / len(times)) * 1000
    throughput = ids.shape[0] / (mean_ms / 1000)
    return mean_ms, throughput


def _to_bf16(model: nn.Module) -> nn.Module:
    """Convierte el modelo a bf16 si hay CUDA con soporte."""
    if HAS_CUDA and USE_BF16:
        return model.to(dtype=torch.bfloat16)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADOS ACUMULADOS
# ─────────────────────────────────────────────────────────────────────────────

results: List[Dict[str, Any]] = []


# ─────────────────────────────────────────────────────────────────────────────
# 1. TINY CONFIG
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("1. TINY CONFIG")
print("=" * 65)

scale_tiny = MoSEScaleConfig.tiny()
mose_tiny_cfg = MoSEConfig.tiny()

# Parámetros analíticos (MoSEScaleConfig)
params_analytical_tiny = scale_tiny.count_params()
vram_est_infer_tiny = scale_tiny.estimate_vram_bf16(training=False)
vram_est_train_tiny = scale_tiny.estimate_vram_bf16(training=True)

print(f"  MoSEScaleConfig.tiny() — analítico")
print(f"    params estimados : {params_analytical_tiny:>14,}")
print(f"    VRAM infer bf16  : {vram_est_infer_tiny:.3f} GB")
print(f"    VRAM train bf16  : {vram_est_train_tiny:.3f} GB")

# Parámetros reales (MoSEPipeline)
_reset_vram_stats()
pipeline_tiny = MoSEPipeline(mose_tiny_cfg).to(DEVICE)
params_real_tiny = _count_params(pipeline_tiny)
vram_real_infer_tiny = _vram_peak_gb() if HAS_CUDA else _cpu_vram_estimate_gb(pipeline_tiny)

print(f"\n  MoSEPipeline(MoSEConfig.tiny()) — real")
print(f"    params reales    : {params_real_tiny:>14,}")
print(f"    VRAM real        : {vram_real_infer_tiny:.3f} GB ({DEVICE})")
ratio_tiny = params_real_tiny / max(1, params_analytical_tiny)
print(f"    ratio real/anal  : {ratio_tiny:.4f}")

# Tokenizador para batch
tok_tiny, tok_name = _load_tokenizer(vocab_size=mose_tiny_cfg.vocab_size)
print(f"\n  Tokenizador: {tok_name}")

# Batch de datos
batch_ids, batch_src = _make_batch(tok_tiny, mose_tiny_cfg.vocab_size, BATCH_SIZE, SEQ_LEN)
print(f"  Batch: {batch_ids.shape} ({batch_src})")

# Forward pass timing
fwd_ms, fwd_thr = _time_forward(pipeline_tiny, batch_ids, use_amp=HAS_CUDA)
print(f"\n  Forward  : {fwd_ms:.1f} ms / batch  ->  {fwd_thr:.1f} ej/s")

# Backward pass timing (solo en CPU o si cabe en memoria)
bwd_ms, bwd_thr = _time_backward(
    pipeline_tiny, batch_ids, use_amp=HAS_CUDA, amp_dtype=AMP_DTYPE
)
print(f"  Backward : {bwd_ms:.1f} ms / batch  ->  {bwd_thr:.1f} ej/s")

# VRAM real durante training
_reset_vram_stats()
pipeline_tiny.train()
with torch.amp.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=HAS_CUDA):
    out_tiny = pipeline_tiny(batch_ids.to(DEVICE))
    loss = F.cross_entropy(
        out_tiny.logits[:, :-1].reshape(-1, out_tiny.logits.shape[-1]),
        batch_ids[:, 1:].to(DEVICE).reshape(-1), ignore_index=PAD
    )
loss.backward()
vram_real_train_tiny = _vram_peak_gb() if HAS_CUDA else _cpu_vram_estimate_gb(pipeline_tiny, training=True)
pipeline_tiny.zero_grad(set_to_none=True)

print(f"  VRAM train real  : {vram_real_train_tiny:.3f} GB")

results.append({
    "config"          : "tiny",
    "params_analytical": params_analytical_tiny,
    "params_real"     : params_real_tiny,
    "vram_est_infer"  : vram_est_infer_tiny,
    "vram_est_train"  : vram_est_train_tiny,
    "vram_real_infer" : vram_real_infer_tiny,
    "vram_real_train" : vram_real_train_tiny,
    "fwd_ms"          : fwd_ms,
    "fwd_thr"         : fwd_thr,
    "bwd_ms"          : bwd_ms,
    "bwd_thr"         : bwd_thr,
    "fits_h200"       : True,
})

# Liberar memoria
del pipeline_tiny
gc.collect()
if HAS_CUDA:
    torch.cuda.empty_cache()

print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. MEDIUM CONFIG
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("2. MEDIUM CONFIG")
print("=" * 65)

scale_medium = MoSEScaleConfig.medium()
params_analytical_med = scale_medium.count_params()
vram_est_infer_med    = scale_medium.estimate_vram_bf16(training=False)
vram_est_train_med    = scale_medium.estimate_vram_bf16(training=True)

print(f"  MoSEScaleConfig.medium() — analítico")
print(f"    params estimados : {params_analytical_med:>14,}  ({params_analytical_med/1e6:.1f}M)")
print(f"    VRAM infer bf16  : {vram_est_infer_med:.3f} GB")
print(f"    VRAM train bf16  : {vram_est_train_med:.3f} GB")

# Intentar instanciar con dims progresivamente menores hasta que quepa
# (Nota: MoSEConfig tiene hidden_dim único vs MoSEScaleConfig que tiene enc/dec separados)

MEDIUM_CANDIDATES = [
    # (label, hidden_dim, vocab_size, enc_layers, dec_layers, seq_len)
    ("medium_768_l6",  768, 32000, 6, 6, 512),
    ("medium_512_l6",  512, 32000, 6, 6, 512),
    ("medium_384_l6",  384, 32000, 6, 6, 256),
    ("medium_256_l6",  256, 32000, 6, 6, 256),
    ("medium_256_l4",  256, 32000, 4, 4, 256),
]

pipeline_medium    = None
mose_medium_cfg    = None
medium_label       = None
medium_seq_len     = SEQ_LEN
params_real_med    = 0
vram_real_infer_med = 0.0
vram_real_train_med = 0.0

for label, hidden_dim, vocab_size, enc_l, dec_l, seq in MEDIUM_CANDIDATES:
    try:
        cfg_try = MoSEConfig(
            hidden_dim    = hidden_dim,
            vocab_size    = vocab_size,
            enc_n_layers  = enc_l,
            dec_n_layers  = dec_l,
            enc_state_dim = 16,
            dec_state_dim = 16,
            dec_max_seq_len = seq,
        )
        _reset_vram_stats()
        p = MoSEPipeline(cfg_try)
        p = _to_bf16(p).to(DEVICE)
        params_real_med = _count_params(p)
        vram_real_infer_med = _vram_peak_gb() if HAS_CUDA else _cpu_vram_estimate_gb(p)

        # Probar que cabe un forward pass
        _test_ids = _synthetic_batch(vocab_size, batch_size=BATCH_SIZE, seq_len=min(seq, SEQ_LEN))
        with torch.no_grad():
            _out = p(_test_ids.to(DEVICE))
        del _test_ids, _out

        pipeline_medium = p
        mose_medium_cfg = cfg_try
        medium_label    = label
        medium_seq_len  = seq
        print(f"\n  Instanciado: {label}")
        print(f"    hidden_dim  : {hidden_dim}")
        print(f"    enc/dec L   : {enc_l}/{dec_l}")
        print(f"    seq_len     : {seq}")
        print(f"    params reales: {params_real_med:>14,}  ({params_real_med/1e6:.1f}M)")
        print(f"    VRAM asignada: {vram_real_infer_med:.3f} GB ({DEVICE})")
        break

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
            print(f"  {label}: OOM — probando config más pequeña...")
            gc.collect()
            if HAS_CUDA:
                torch.cuda.empty_cache()
        else:
            raise

if pipeline_medium is None:
    print("  ERROR: No se pudo instanciar ninguna variante de medium.")
    results.append({
        "config": "medium",
        "params_analytical": params_analytical_med,
        "params_real": None,
        "vram_est_infer": vram_est_infer_med,
        "vram_est_train": vram_est_train_med,
        "vram_real_infer": None,
        "vram_real_train": None,
        "fwd_ms": None, "fwd_thr": None,
        "bwd_ms": None, "bwd_thr": None,
        "fits_h200": True,
    })
else:
    tok_med, tok_name_med = _load_tokenizer(vocab_size=mose_medium_cfg.vocab_size)
    print(f"  Tokenizador: {tok_name_med}")

    batch_ids_med, batch_src_med = _make_batch(
        tok_med, mose_medium_cfg.vocab_size, BATCH_SIZE,
        min(medium_seq_len, SEQ_LEN)
    )
    print(f"  Batch: {batch_ids_med.shape} ({batch_src_med})")

    # ── Forward pass timing ───────────────────────────────────────────────────
    fwd_ms_med, fwd_thr_med = _time_forward(
        pipeline_medium, batch_ids_med,
        use_amp=HAS_CUDA, amp_dtype=AMP_DTYPE
    )
    print(f"\n  Forward  : {fwd_ms_med:.1f} ms / batch  ->  {fwd_thr_med:.1f} ej/s")

    # ── Backward pass timing ──────────────────────────────────────────────────
    bwd_ms_med, bwd_thr_med = _time_backward(
        pipeline_medium, batch_ids_med,
        use_amp=HAS_CUDA, amp_dtype=AMP_DTYPE
    )
    print(f"  Backward : {bwd_ms_med:.1f} ms / batch  ->  {bwd_thr_med:.1f} ej/s")

    # ── VRAM real durante training ────────────────────────────────────────────
    _reset_vram_stats()
    pipeline_medium.train()
    with torch.amp.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=HAS_CUDA):
        out_med = pipeline_medium(batch_ids_med.to(DEVICE))
        loss_med = F.cross_entropy(
            out_med.logits[:, :-1].reshape(-1, out_med.logits.shape[-1]),
            batch_ids_med[:, 1:].to(DEVICE).reshape(-1), ignore_index=PAD
        )
    loss_med.backward()
    vram_real_train_med = _vram_peak_gb() if HAS_CUDA else _cpu_vram_estimate_gb(pipeline_medium, training=True)
    pipeline_medium.zero_grad(set_to_none=True)
    print(f"  VRAM train real  : {vram_real_train_med:.3f} GB")

    results.append({
        "config"           : f"medium ({medium_label})",
        "params_analytical": params_analytical_med,
        "params_real"      : params_real_med,
        "vram_est_infer"   : vram_est_infer_med,
        "vram_est_train"   : vram_est_train_med,
        "vram_real_infer"  : vram_real_infer_med,
        "vram_real_train"  : vram_real_train_med,
        "fwd_ms"           : fwd_ms_med,
        "fwd_thr"          : fwd_thr_med,
        "bwd_ms"           : bwd_ms_med,
        "bwd_thr"          : bwd_thr_med,
        "fits_h200"        : True,
    })

    del pipeline_medium
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()

print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. VRAM REAL VS ESTIMADA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("3. COMPARACIÓN VRAM REAL vs ESTIMADA")
print("=" * 65)

VRAM_WARN_THRESHOLD = 0.30  # >30% de error -> warning

for r in results:
    if r.get("vram_real_infer") is None:
        continue
    est  = r["vram_est_infer"]
    real = r["vram_real_infer"]
    if real == 0.0 or est == 0.0:
        continue
    err = abs(est - real) / max(real, 1e-9)
    flag = " *** WARNING >30%" if err > VRAM_WARN_THRESHOLD else ""
    print(f"  {r['config']:30s}  est={est:.3f}GB  real={real:.3f}GB  err={err:.1%}{flag}")

    r["vram_err_pct"] = err * 100


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXTRAPOLACIÓN — PRODUCTION (3.5B) y H200
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("4. EXTRAPOLACIÓN -> PRODUCTION (3.437B) + H200")
print("=" * 65)

scale_prod = MoSEScaleConfig.production()
params_prod = scale_prod.count_params()
vram_est_infer_prod = scale_prod.estimate_vram_bf16(training=False)
vram_est_train_prod = scale_prod.estimate_vram_bf16(training=True)

print(f"  MoSEScaleConfig.production() — analítico")
print(f"    params           : {params_prod:>16,}  ({params_prod/1e9:.3f}B)")
print(f"    VRAM infer bf16  : {vram_est_infer_prod:.2f} GB")
print(f"    VRAM train bf16  : {vram_est_train_prod:.2f} GB")
print()

# Extrapolación empírica basada en tiny y medium
# ratio_real_vs_analytical: cuánto difiere el real del estimado
valid_rows = [r for r in results if r.get("vram_real_infer") and r.get("vram_est_infer")]
if len(valid_rows) >= 1:
    # Media de los ratios (real/estimated) observados
    ratios = [r["vram_real_infer"] / r["vram_est_infer"] for r in valid_rows if r["vram_est_infer"] > 0]
    mean_ratio = sum(ratios) / len(ratios)
    vram_extrap_infer = vram_est_infer_prod * mean_ratio
    vram_extrap_train = vram_est_train_prod * mean_ratio
    print(f"  Extrapolación empírica (ratio promedio = {mean_ratio:.3f}):")
    print(f"    VRAM infer extrap: {vram_extrap_infer:.2f} GB")
    print(f"    VRAM train extrap: {vram_extrap_train:.2f} GB")
else:
    # Sin datos reales: usar estimación analítica directamente
    mean_ratio = 1.0
    vram_extrap_infer = vram_est_infer_prod
    vram_extrap_train = vram_est_train_prod
    print(f"  Sin datos empíricos suficientes — usando estimación analítica.")

print()

# ── Análisis de fit en H200 ───────────────────────────────────────────────────
fits_infer = vram_extrap_infer <= H200_VRAM_GB
fits_train = vram_extrap_train <= H200_VRAM_GB
margin_infer = H200_VRAM_GB - vram_extrap_infer
margin_train = H200_VRAM_GB - vram_extrap_train

print(f"  H200 ({H200_VRAM_GB:.0f} GB):")
print(f"    Inferencia : {vram_extrap_infer:.2f} GB  -> {'CABE OK' if fits_infer else 'NO CABE NO'}  (margin {margin_infer:+.1f} GB)")
print(f"    Training   : {vram_extrap_train:.2f} GB  -> {'CABE OK' if fits_train else 'NO CABE NO'}  (margin {margin_train:+.1f} GB)")

# Cuántos H200 necesitaría para training si no cabe en uno
n_h200_needed = math.ceil(vram_extrap_train / H200_VRAM_GB)
if n_h200_needed > 1:
    print(f"    Training requiere {n_h200_needed}× H200 (tensor parallelism)")

results.append({
    "config"           : "production (3.437B)",
    "params_analytical": params_prod,
    "params_real"      : None,
    "vram_est_infer"   : vram_est_infer_prod,
    "vram_est_train"   : vram_est_train_prod,
    "vram_real_infer"  : None,
    "vram_real_train"  : None,
    "vram_extrap_infer": vram_extrap_infer,
    "vram_extrap_train": vram_extrap_train,
    "fwd_ms"           : None,
    "fwd_thr"          : None,
    "bwd_ms"           : None,
    "bwd_thr"          : None,
    "fits_h200"        : fits_infer,
    "fits_h200_train"  : fits_train,
    "n_h200_needed"    : n_h200_needed,
})

print()


# ─────────────────────────────────────────────────────────────────────────────
# 5. TABLA RESUMEN
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_params(p: Optional[int]) -> str:
    if p is None:
        return "N/A"
    if p >= 1e9:
        return f"{p/1e9:.3f}B"
    if p >= 1e6:
        return f"{p/1e6:.1f}M"
    return f"{p/1e3:.1f}K"


def _fmt_vram(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.3f}GB"


def _fmt_thr(t: Optional[float]) -> str:
    if t is None:
        return "N/A"
    return f"{t:.1f} ej/s"


print()
print("=" * 65)
print("TABLA RESUMEN")
print("=" * 65)

# Header
hdr = f"{'Config':28s} {'Params':>10s} {'VRAM est':>10s} {'VRAM real':>10s} {'Throughput':>12s} {'H200?':>8s}"
print(hdr)
print("-" * 65)

for r in results:
    cfg_label  = r["config"][:28]
    params_str = _fmt_params(r["params_analytical"])
    vram_est   = r.get("vram_extrap_infer") or r["vram_est_infer"]
    vram_real  = r.get("vram_real_infer")
    thr        = r.get("fwd_thr")
    h200       = "Si" if r.get("fits_h200") else "No"

    print(
        f"{cfg_label:28s} {params_str:>10s} {_fmt_vram(vram_est):>10s} "
        f"{_fmt_vram(vram_real):>10s} {_fmt_thr(thr):>12s} {h200:>8s}"
    )

print("=" * 65)
print()
print("Notas:")
print(f"  - VRAM estimada  : formula bf16 = params × 2B / 1e9 (MoSEScaleConfig.estimate_vram_bf16)")
print(f"  - VRAM real      : torch.cuda.memory_allocated() (GPU) o estimacion RAM (CPU)")
print(f"  - Throughput     : forward pass, B={BATCH_SIZE} ej, seq_len={SEQ_LEN}")
print(f"  - Device actual  : {DEVICE}")
print(f"  - Production     : params reales no medidos (demasiado grande para instanciar)")

# ── Guardar resultados a JSON ─────────────────────────────────────────────────
_out_path = _HERE / "scaling_results.json"
try:
    with open(_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResultados guardados en: {_out_path}")
except Exception as e:
    print(f"\n[WARNING] No se pudo guardar scaling_results.json: {e}")

print("\nDone.")
