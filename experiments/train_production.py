"""
experiments/train_production.py
=================================
Script MAESTRO de entrenamiento de AION-C.

Ejecuta 4 fases de entrenamiento en secuencia (o individualmente):

  Phase 0 — Aprender a hablar
    Encoder + Decoder solamente (no CRE, no motores).
    graph_repr = zeros para bypasear motores.
    Objetivo: que el decoder genere texto coherente.

  Phase 1 — Backbone compartido
    Pipeline completo: encoder + decoder + motores + orchestrator.
    Carga checkpoint de Phase 0. Entrena todos los componentes con
    datos mezclados de los 5 dominios.

  Phase 2 — Motores especializados
    Para cada motor (CORA, FORGE-C, AXIOM, MUSE, EMPATHY):
    congela backbone compartido, entrena SOLO los layers del motor.
    Carga checkpoint de Phase 1. Usa datos del dominio del motor.
    Para al convergir o early stopping (sin steps fijos).

  Phase 3 — Fine-tune E2E + Orchestrator
    Primero entrena el orchestrator con el dataset de routing.
    Luego fine-tune end-to-end con lr bajo.
    Carga todos los checkpoints anteriores.

Uso:
    cd AION-C
    # Smoke test completo en CPU (<30 min):
    python -m experiments.train_production --config tiny --phase all

    # Solo Phase 0:
    python -m experiments.train_production --config tiny --phase 0

    # Reanudar desde checkpoint:
    python -m experiments.train_production --config medium --phase all --resume

    # Produccion en H200:
    python -m experiments.train_production --config production --phase all --device cuda

Argumentos:
    --config   : tiny / medium / production  (default: tiny)
    --phase    : 0 / 1 / 2 / 3 / all        (default: all)
    --resume   : continuar desde checkpoint si existe
    --device   : cpu / cuda / auto           (default: auto)
    --run-dir  : directorio de checkpoints y logs (default: runs/aion_<config>)
    --max-steps: override del maximo de steps por fase (debug)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── paths ───────────────────────────────────────────────────────────────────
import sys, os
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from router.pipeline    import MoSEPipeline, MoSEConfig
from orchestrator.model import MOTOR_NAMES
from experiments.training_utils import (
    TrainingMonitor,
    make_cosine_scheduler,
    save_checkpoint,
    load_checkpoint,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

PAD            = 0
EOS            = 2
MOTOR_TO_IDX   = {m: i for i, m in enumerate(MOTOR_NAMES)}
# expected_output del orchestrator dataset puede estar en mayusculas
MOTOR_ALIASES  = {
    "cora": "cora", "forge_c": "forge_c", "forge-c": "forge_c",
    "muse": "muse", "axiom": "axiom", "empathy": "empathy",
    "CORA": "cora", "FORGE_C": "forge_c", "FORGE-C": "forge_c",
    "MUSE": "muse", "AXIOM": "axiom", "EMPATHY": "empathy",
}


# ─────────────────────────────────────────────────────────────────────────────
# PRESETS DE CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _TrainHparams:
    """Hiperparámetros de entrenamiento indexados por fase."""
    # Phase 0
    ph0_steps:    int   = 300
    ph0_lr:       float = 3e-4
    ph0_warmup:   int   = 50
    ph0_eval:     int   = 100   # eval_every
    ph0_patience: int   = 300
    ph0_conv_win: int   = 200

    # Phase 1
    ph1_steps:    int   = 300
    ph1_lr:       float = 2e-4
    ph1_warmup:   int   = 50
    ph1_eval:     int   = 100
    ph1_patience: int   = 300
    ph1_conv_win: int   = 200

    # Phase 2 (por motor)
    ph2_steps:    int   = 200      # maximo por motor (se para antes si converge)
    ph2_lr:       float = 3e-4
    ph2_warmup:   int   = 20
    ph2_eval:     int   = 50
    ph2_patience: int   = 200
    ph2_conv_win: int   = 100

    # Phase 3 (orchestrator routing)
    ph3_orch_steps:  int   = 100
    ph3_orch_lr:     float = 1e-3
    ph3_e2e_steps:   int   = 150
    ph3_e2e_lr:      float = 5e-5
    ph3_warmup:      int   = 20
    ph3_eval:        int   = 50
    ph3_patience:    int   = 200
    ph3_conv_win:    int   = 100

    # Datos
    n_train:    int   = 500     # ejemplos de entrenamiento por motor
    n_val:      int   = 50      # ejemplos de validacion
    max_seq:    int   = 128     # longitud maxima de secuencia
    batch_size: int   = 1       # batch size (1 = secuencial)

    # Phase 4 (instruction tuning)
    ph4_steps:    int   = 200
    ph4_lr:       float = 1e-4
    ph4_eval:     int   = 50

    # Convergence delta (shared across phases)
    conv_delta: float = 0.001

    # Batching: real GPU-parallel batch + gradient accumulation
    # effective_batch = batch_size * grad_accum_steps
    train_batch_size: int = 1
    grad_accum_steps: int = 1

    # Checkpoints
    ckpt_every: int   = 100


_PRESETS: Dict[str, _TrainHparams] = {
    "tiny": _TrainHparams(),  # valores por defecto (smoke test, ~3 min CPU)

    "medium": _TrainHparams(
        ph0_steps=3000,    ph0_warmup=200,  ph0_eval=500,
        ph0_patience=2000, ph0_conv_win=1000,
        ph1_steps=3000,    ph1_warmup=200,  ph1_eval=500,
        ph1_patience=2000, ph1_conv_win=1000,
        ph2_steps=2000,    ph2_warmup=100,  ph2_eval=500,
        ph2_patience=1000, ph2_conv_win=500,
        ph3_orch_steps=1000, ph3_orch_lr=5e-4,
        ph3_e2e_steps=2000,  ph3_e2e_lr=2e-5,
        ph3_warmup=100,   ph3_eval=200,
        ph3_patience=800, ph3_conv_win=400,
        n_train=3000,     n_val=100,
        max_seq=512,      batch_size=1,
        ckpt_every=500,
    ),

    "production": _TrainHparams(
        ph0_steps=20000,   ph0_warmup=1000, ph0_eval=1000,
        ph0_patience=5000, ph0_conv_win=3000,
        ph1_steps=30000,   ph1_warmup=2000, ph1_eval=1000,
        ph1_patience=5000, ph1_conv_win=3000,
        ph2_steps=20000,   ph2_warmup=500,  ph2_eval=1000,
        ph2_patience=3000, ph2_conv_win=2000,
        ph3_orch_steps=10000, ph3_orch_lr=3e-4,
        ph3_e2e_steps=20000,  ph3_e2e_lr=1e-5,
        ph3_warmup=500,   ph3_eval=500,
        ph3_patience=2000, ph3_conv_win=1000,
        n_train=20000,    n_val=200,
        max_seq=2048,     batch_size=1,
        ckpt_every=2000,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO DE FASE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    phase:        int
    label:        str
    steps_run:    int
    final_loss:   float
    best_val_loss: float
    best_val_f1:  float
    elapsed_s:    float
    stop_reason:  str          # "converged", "early_stop", "max_steps"
    checkpoint:   Optional[str] = None
    extra:        Dict[str, Any] = field(default_factory=dict)

    def summary_line(self) -> str:
        mins = self.elapsed_s / 60
        return (
            f"  Phase {self.phase} [{self.label:20s}]: "
            f"steps={self.steps_run:>5}  "
            f"loss={self.final_loss:.4f}  "
            f"val_loss={self.best_val_loss:.4f}  "
            f"F1={self.best_val_f1:.3f}  "
            f"stop={self.stop_reason}  "
            f"time={mins:.1f}min"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZADOR
# ─────────────────────────────────────────────────────────────────────────────

class _SimpleTokenizer:
    """Tokenizador word-level para tiny config (vocab_size <= 1024)."""
    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = vocab_size
        self._w2i: Dict[str, int] = {}
        self._i2w: Dict[int, str] = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>"}

    def build_vocab(self, texts: List[str]) -> "_SimpleTokenizer":
        freq: Dict[str, int] = {}
        for t in texts:
            for w in t.lower().split():
                freq[w] = freq.get(w, 0) + 1
        for i, (w, _) in enumerate(
            sorted(freq.items(), key=lambda x: -x[1]), start=4
        ):
            if i >= self.vocab_size:
                break
            self._w2i[w] = i
            self._i2w[i] = w
        return self

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        toks = [self.BOS]
        for w in text.lower().split():
            toks.append(self._w2i.get(w, self.UNK))
            if len(toks) >= max_len - 1:
                break
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


class _AIONTokenizerWrapper:
    """Wrapper de AIONTokenizer con API compatible con _SimpleTokenizer."""

    def __init__(self, inner) -> None:
        self._tok      = inner
        self.vocab_size = inner.vocab_size
        self.PAD       = inner.pad_id
        self.BOS       = inner.bos_id
        self.EOS       = inner.eos_id
        self.UNK       = inner.unk_id

    def encode(self, text: str, max_len: int = 512) -> List[int]:
        return self._tok.encode(text)[:max_len]

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def to_tensor(self, ids: List[int]) -> torch.Tensor:
        return torch.tensor([ids], dtype=torch.long)


def build_tokenizer(vocab_size: int, corpus_texts: Optional[List[str]] = None):
    """
    Construye el tokenizador correcto según el vocab_size.
    - vocab_size == 32000 y AIONTokenizer disponible -> AIONTokenizerWrapper
    - sino -> _SimpleTokenizer construido desde corpus_texts
    """
    if vocab_size == 32_000:
        model_path = _ROOT / "tokenizer" / "aion_32k.model"
        if model_path.exists():
            try:
                from tokenizer import AIONTokenizer
                return _AIONTokenizerWrapper(AIONTokenizer(model_path))
            except Exception:
                pass

    # Fallback: tokenizador word-level
    tok = _SimpleTokenizer(vocab_size=vocab_size)
    if corpus_texts:
        tok.build_vocab(corpus_texts)
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE FREEZE / UNFREEZE
# ─────────────────────────────────────────────────────────────────────────────

def freeze(module: nn.Module) -> None:
    """Congela todos los parametros de un modulo."""
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze(module: nn.Module) -> None:
    """Descongela todos los parametros de un modulo."""
    for p in module.parameters():
        p.requires_grad_(True)


def freeze_all_except(pipeline: MoSEPipeline, keep_prefix: str) -> None:
    """
    Congela todos los parametros excepto los cuyo nombre contiene `keep_prefix`.

    Ejemplo:
        freeze_all_except(pipeline, "motors.cora")
        # solo entrena los parametros de pipeline.motors["cora"]
    """
    for name, p in pipeline.named_parameters():
        if keep_prefix in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


def freeze_all_except_list(pipeline: MoSEPipeline, keep_prefixes: List[str]) -> None:
    """Congela todo excepto los prefijos listados."""
    for name, p in pipeline.named_parameters():
        if any(pfx in name for pfx in keep_prefixes):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_datasets(
    motors:       Tuple[str, ...] = tuple(MOTOR_NAMES),
    max_examples: int             = 20000,
    eval_size:    int             = 100,
    seed:         int             = 42,
    dataset_root: Optional[Path]  = None,
) -> Dict[str, Tuple[Any, Any]]:
    """
    Carga los datasets de Opus para cada motor.

    Args:
        dataset_root: ruta explícita a la carpeta con los .jsonl.
                      Si None, usa la ruta por defecto de OpusDataset
                      (DataSet-Generator-Claude-Opus junto a AION-C).

    Returns:
        Dict[motor_name -> (train_dataset, eval_dataset)]
        Devuelve dict vacío si todos los motores fallan (ver WARNING en stdout).
    """
    from experiments.opus_dataset import OpusDataset
    result: Dict[str, Tuple[Any, Any]] = {}
    for motor in motors:
        try:
            ds = OpusDataset(
                motor        = motor,
                max_examples = max_examples,
                seed         = seed,
                dataset_root = dataset_root,
            )
            train_ds, val_ds = ds.train_eval_split(eval_size=eval_size, seed=seed)
            result[motor] = (train_ds, val_ds)
            print(f"  [OK] {motor}: {len(ds)} ejemplos", flush=True)
        except FileNotFoundError as e:
            print(f"  [WARNING] {motor}: archivo no encontrado — {e}", flush=True)
        except Exception as e:
            print(f"  [WARNING] {motor}: {type(e).__name__}: {e}", flush=True)
    if not result:
        print(
            "  [WARNING] Todos los datasets fallaron. "
            "Verifica que DataSet-Generator-Claude-Opus este junto a AION-C "
            "o pasa dataset_root explicitamente.",
            flush=True,
        )
    return result


def load_orchestrator_dataset(
    max_examples: int            = 5000,
    eval_size:    int            = 100,
    seed:         int            = 42,
    dataset_root: Optional[Path] = None,
) -> Tuple[Any, Any]:
    """Carga el dataset del orchestrator para Phase 3."""
    from experiments.opus_dataset import OpusDataset
    ds = OpusDataset(
        motor        = "orchestrator",
        max_examples = max_examples,
        seed         = seed,
        dataset_root = dataset_root,
    )
    return ds.train_eval_split(eval_size=eval_size, seed=seed)


def _encode(tok, text: str, max_len: int) -> List[int]:
    """Encode unificado para ambos tipos de tokenizador."""
    try:
        return tok.encode(text, max_len)
    except TypeError:
        return tok.encode(text)[:max_len]


def _make_synthetic_ids(tok, n: int, max_len: int) -> List[List[int]]:
    """
    Fallback: genera n secuencias sintéticas con CausalGraphGenerator.
    Se usa cuando los datasets Opus no están disponibles.
    """
    try:
        from synth import CausalGraphGenerator
        gen = CausalGraphGenerator(seed=99)
        rng = random.Random(99)
        ids_list = []
        for _ in range(n):
            ex   = gen.generate(level=1)
            text = ex.problem_text + " " + ex.answer
            ids  = _encode(tok, text, max_len)
            ids_list.append(ids)
        return ids_list
    except Exception:
        # Ultra-fallback: secuencias de tokens aleatorias válidas
        rng    = random.Random(99)
        vocab  = max(getattr(tok, "vocab_size", 512), 10)
        eos    = getattr(tok, "EOS", 2)
        ids_list = []
        for _ in range(n):
            length = rng.randint(4, min(max_len - 1, 20))
            ids    = [rng.randint(4, vocab - 1) for _ in range(length)] + [eos]
            ids_list.append(ids[:max_len])
        return ids_list


def make_ids_list(
    dataset,
    tok,
    n:       int = 500,
    max_len: int = 128,
) -> List[List[int]]:
    """Genera n secuencias pre-tokenizadas desde un dataset.
    Si dataset es None usa fallback sintético.
    """
    if dataset is None:
        print("  [WARN] make_ids_list: dataset=None, usando fallback sintetico",
              flush=True)
        return _make_synthetic_ids(tok, n, max_len)
    ids_list = []
    for _ in range(n):
        ex   = dataset.generate()
        text = ex.problem_text + " " + ex.answer
        ids  = _encode(tok, text, max_len)
        ids_list.append(ids)
    return ids_list


def make_mixed_ids(
    datasets: Dict[str, Tuple[Any, Any]],
    tok,
    n:        int = 500,
    max_len:  int = 128,
) -> List[List[int]]:
    """Genera n secuencias mezclando todos los dominios.
    Si datasets está vacío usa fallback sintético con CausalGraphGenerator.
    """
    if not datasets:
        print(
            "  [WARN] make_mixed_ids: datasets vacio, usando fallback sintetico "
            "(CausalGraphGenerator). Para datos reales verifica dataset_root.",
            flush=True,
        )
        return _make_synthetic_ids(tok, n, max_len)
    motors   = list(datasets.keys())
    ids_list = []
    for i in range(n):
        motor = motors[i % len(motors)]
        train_ds, _ = datasets[motor]
        ex   = train_ds.generate()
        text = ex.problem_text + " " + ex.answer
        ids  = _encode(tok, text, max_len)
        ids_list.append(ids)
    return ids_list


def get_fixed_examples(
    datasets: Dict[str, Tuple[Any, Any]],
    seed:     int = 42,
    n_per_motor: int = 1,
) -> List[Tuple[str, str, str]]:
    """
    Retorna ejemplos fijos (uno por motor) para evaluacion con greedy decode.
    """
    rng      = random.Random(seed)
    examples = []
    for motor in MOTOR_NAMES:
        if motor not in datasets:
            continue
        _, val_ds = datasets[motor]
        pool = [val_ds.generate() for _ in range(min(10, n_per_motor * 5))]
        rng.shuffle(pool)
        for ex in pool[:n_per_motor]:
            examples.append((ex.problem_text, ex.answer, motor))
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# MODELO PHASE 0 (encoder + decoder, sin motores)
# ─────────────────────────────────────────────────────────────────────────────

class _Phase0Model(nn.Module):
    """
    Wrapper para Phase 0: encoder + decoder sin motores.

    El decoder recibe un graph_repr de ceros (B, K, D) para bypasear
    los motores. Solo entrena encoder y decoder.
    """

    def __init__(self, pipeline: MoSEPipeline, cfg: MoSEConfig) -> None:
        super().__init__()
        self.pipeline = pipeline
        self._K       = cfg.motor_max_nodes
        self._hidden  = cfg.hidden_dim

    def forward(self, token_ids: torch.Tensor):
        B = token_ids.shape[0]
        concepts  = self.pipeline.encoder(token_ids)           # [B, L, D]
        D         = concepts.shape[-1]
        graph_rep = torch.zeros(
            B, self._K, D,
            device=token_ids.device, dtype=concepts.dtype
        )
        dec_out = self.pipeline.decoder(
            token_ids, graph_rep, encoder_concepts=concepts
        )
        return dec_out   # has .logits [B, L, V]


# ─────────────────────────────────────────────────────────────────────────────
# BUCLE DE ENTRENAMIENTO CENTRAL
# ─────────────────────────────────────────────────────────────────────────────

def _phase_loop(
    model:          nn.Module,
    get_ids:        Callable[[], List[int]],  # retorna lista de ids aleatoria
    optimizer:      torch.optim.Optimizer,
    scheduler,
    n_steps:        int,
    monitor:        Optional[TrainingMonitor],
    label:          str,
    device:         torch.device,
    checkpoint_path: Optional[Path],
    ckpt_every:     int  = 500,
    print_every:    int  = 100,
    use_amp:        bool = False,
    amp_dtype:      torch.dtype = torch.float32,
    extra_loss_fn:  Optional[Callable[[], float]] = None,
    max_grad_norm:  float = 1.0,
    start_step:     int = 0,
    grad_accum_steps: int = 1,
    batch_size:     int = 1,
) -> Tuple[List[float], float, float, str]:
    """
    Bucle de entrenamiento generico para una fase.

    Args:
        model:      Modelo (debe retornar objeto con .logits o tensor).
        get_ids:    Callable que devuelve una secuencia de token IDs.
        extra_loss_fn: Si se provee, su resultado se suma al CE loss.
        start_step: Paso desde el cual reanudar (para resume).
        grad_accum_steps: acumula gradientes de N micro-batches antes de
                          hacer optimizer.step(). Effective batch = N.

    Returns:
        (losses, elapsed_s, final_loss_avg50, stop_reason)
    """
    model.train()
    losses:     List[float] = []
    t0          = time.perf_counter()
    stop_reason = "max_steps"
    accum       = max(1, grad_accum_steps)
    batch_size  = max(1, batch_size)

    for step in range(start_step + 1, n_steps + 1):
        # --- Gradient accumulation loop ---
        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for _micro in range(accum):
            # Build a real batch of batch_size examples
            batch_ids = [get_ids() for _ in range(batch_size)]
            max_len   = max(len(s) for s in batch_ids)
            # Pad to same length
            padded = [s + [PAD] * (max_len - len(s)) for s in batch_ids]
            ids_t  = torch.tensor(padded, dtype=torch.long, device=device)  # [B, L]

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                out    = model(ids_t)
                logits = out.logits if hasattr(out, "logits") else out  # [B, L, V]
                # CE loss averaged over batch
                ce     = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    ids_t[:, 1:].reshape(-1),
                    ignore_index=PAD,
                )
                loss   = ce / accum
                if extra_loss_fn is not None:
                    loss = loss + extra_loss_fn() / accum

            if math.isfinite(loss.item()):
                loss.backward()
                accum_loss += loss.item() * accum

        if accum_loss == 0.0:
            continue

        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        losses.append(accum_loss / accum)

        if step % print_every == 0:
            recent = [x for x in losses[-print_every:] if math.isfinite(x)]
            avg    = sum(recent) / len(recent) if recent else float("nan")
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t0
            steps_done = step - start_step
            sps = steps_done / max(0.1, elapsed)
            steps_left = n_steps - step
            eta_s = steps_left / max(0.01, sps)
            eta_m = eta_s / 60
            print(
                f"    [{label}]  step {step:>6}/{n_steps}  "
                f"loss={avg:.4f}  lr={lr_now:.2e}  "
                f"{sps:.1f} steps/s  {elapsed:.0f}s elapsed  "
                f"ETA {eta_m:.0f}m",
                flush=True,
            )

        if checkpoint_path is not None and step % ckpt_every == 0:
            avg50 = (
                sum(losses[-50:]) / min(50, len(losses)) if losses else float("nan")
            )
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, step, avg50)

        if monitor is not None and math.isfinite(loss.item()):
            sig = monitor.step(step, loss.item())
            if sig is not None:
                stop_reason = sig
                break

    elapsed    = time.perf_counter() - t0
    final_loss = (
        sum(losses[-50:]) / min(50, len(losses)) if losses else float("nan")
    )
    return losses, elapsed, final_loss, stop_reason


# ─────────────────────────────────────────────────────────────────────────────
# BANNERS Y RESUMEN
# ─────────────────────────────────────────────────────────────────────────────

def _banner(phase: int, title: str, lines: List[str]) -> None:
    bar = "=" * 65
    print(f"\n{bar}", flush=True)
    print(f"  PHASE {phase} — {title}", flush=True)
    print(bar, flush=True)
    for l in lines:
        print(f"  {l}", flush=True)
    print(flush=True)


def _summary(results: List[PhaseResult]) -> None:
    print("\n" + "=" * 65, flush=True)
    print("  RESUMEN FINAL", flush=True)
    print("=" * 65, flush=True)
    total = 0.0
    for r in results:
        print(r.summary_line(), flush=True)
        total += r.elapsed_s
    print(f"\n  Tiempo total: {total/60:.1f} min", flush=True)
    print("=" * 65, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — ENCODER + DECODER SOLO
# ─────────────────────────────────────────────────────────────────────────────

def run_phase0(
    pipeline:      MoSEPipeline,
    cfg:           MoSEConfig,
    tok,
    datasets:      Dict[str, Tuple],
    hparams:       _TrainHparams,
    device:        torch.device,
    checkpoint_dir: Path,
    resume:        bool = False,
    max_steps_override: Optional[int] = None,
) -> PhaseResult:
    """
    Phase 0: entrena encoder y decoder solamente (sin motores/CRE).
    """
    label   = "Phase0-EncDec"
    ckpt_p  = checkpoint_dir / "phase0.pt"
    n_steps = max_steps_override or hparams.ph0_steps

    _banner(0, "APRENDER A HABLAR (encoder + decoder)", [
        f"Modelo: Phase0Model (encoder+decoder, no motors)",
        f"Steps max    : {n_steps}",
        f"LR           : {hparams.ph0_lr:.1e}",
        f"eval_every   : {hparams.ph0_eval}",
        f"patience     : {hparams.ph0_patience}",
        f"Checkpoint   : {ckpt_p}",
    ])

    # Congelar: motores, orchestrator, unifier
    freeze(pipeline.orchestrator)
    freeze(pipeline.unifier)
    for motor in pipeline.motors.values():
        freeze(motor)
    unfreeze(pipeline.encoder)
    unfreeze(pipeline.decoder)

    trainable = count_trainable(pipeline)
    print(f"  Parametros entrenables: {trainable:,}", flush=True)

    # Modelo wrapper
    model = _Phase0Model(pipeline, cfg).to(device)

    # Datos
    ids_list = make_mixed_ids(datasets, tok, n=hparams.n_train, max_len=hparams.max_seq)
    val_list = make_mixed_ids(datasets, tok, n=hparams.n_val,   max_len=hparams.max_seq)
    fixed    = get_fixed_examples(datasets)

    # Tokenizar val para monitor
    monitor_val = val_list[:50]

    # Monitor
    monitor = TrainingMonitor(
        model          = model,
        tok            = tok,
        val_ids_list   = monitor_val,
        fixed_examples = fixed,
        cfg            = cfg,
        eval_every     = hparams.ph0_eval,
        patience       = hparams.ph0_patience,
        convergence_delta  = hparams.conv_delta,
        convergence_window = hparams.ph0_conv_win,
        log_path       = checkpoint_dir / "phase0_monitor.jsonl",
        device         = device,
        is_motor       = False,  # _Phase0Model retorna DecoderOutput no MoSEOutput
    )

    # Optimizer
    opt  = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=hparams.ph0_lr, weight_decay=1e-2,
    )
    sched = make_cosine_scheduler(opt, hparams.ph0_warmup, n_steps)

    # Resume
    start_step = 0
    if resume and ckpt_p.exists():
        start_step, _ = load_checkpoint(ckpt_p, model, opt, sched)
        print(f"  Reanudando desde step {start_step}", flush=True)

    use_amp   = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float32

    rng = random.Random(42)
    def get_ids():
        return rng.choice(ids_list)

    t_start = time.perf_counter()
    losses, elapsed, final_loss, stop_reason = _phase_loop(
        model           = model,
        get_ids         = get_ids,
        optimizer       = opt,
        scheduler       = sched,
        n_steps         = n_steps,
        monitor         = monitor,
        label           = label,
        device          = device,
        checkpoint_path = ckpt_p,
        ckpt_every      = hparams.ckpt_every,
        print_every     = max(1, hparams.ph0_eval // 4),
        use_amp         = use_amp,
        amp_dtype       = amp_dtype,
        start_step      = start_step,
        grad_accum_steps = hparams.grad_accum_steps,
        batch_size       = hparams.train_batch_size,
    )

    # Checkpoint final
    save_checkpoint(ckpt_p, model, opt, sched, len(losses) + start_step, final_loss)
    print(f"\n  [Phase0] Checkpoint -> {ckpt_p}", flush=True)

    # Restaurar para Phase 1
    unfreeze(pipeline.orchestrator)
    unfreeze(pipeline.unifier)
    for motor in pipeline.motors.values():
        unfreeze(motor)

    best_record = (
        min(monitor.eval_history, key=lambda r: r["val_loss"])
        if monitor.eval_history else {}
    )

    return PhaseResult(
        phase         = 0,
        label         = label,
        steps_run     = len(losses),
        final_loss    = final_loss,
        best_val_loss = best_record.get("val_loss", float("nan")),
        best_val_f1   = best_record.get("val_f1",   0.0),
        elapsed_s     = elapsed,
        stop_reason   = stop_reason,
        checkpoint    = str(ckpt_p),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — BACKBONE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1(
    pipeline:      MoSEPipeline,
    cfg:           MoSEConfig,
    tok,
    datasets:      Dict[str, Tuple],
    hparams:       _TrainHparams,
    device:        torch.device,
    checkpoint_dir: Path,
    resume:        bool = False,
    max_steps_override: Optional[int] = None,
) -> PhaseResult:
    """
    Phase 1: entrena el pipeline completo con datos mezclados.
    Carga checkpoint de Phase 0 si existe.
    """
    label  = "Phase1-Backbone"
    ckpt_p = checkpoint_dir / "phase1.pt"
    ckpt0  = checkpoint_dir / "phase0.pt"
    n_steps = max_steps_override or hparams.ph1_steps

    _banner(1, "BACKBONE COMPARTIDO (pipeline completo)", [
        f"Steps max    : {n_steps}",
        f"LR           : {hparams.ph1_lr:.1e}",
        f"eval_every   : {hparams.ph1_eval}",
        f"patience     : {hparams.ph1_patience}",
        f"Checkpoint   : {ckpt_p}",
    ])

    # Cargar Phase 0 encoder/decoder si existe
    if ckpt0.exists():
        ckpt = torch.load(ckpt0, map_location="cpu", weights_only=False)
        # Extraer solo encoder y decoder del modelo Phase0
        enc_prefix = "pipeline.encoder."
        dec_prefix = "pipeline.decoder."
        sd = pipeline.state_dict()
        for k, v in ckpt["model_state"].items():
            if k.startswith(enc_prefix) or k.startswith(dec_prefix):
                if k in sd:
                    sd[k] = v
        pipeline.load_state_dict(sd)
        print(f"  Encoder/decoder cargados desde Phase0: {ckpt0}", flush=True)
    else:
        print(f"  [INFO] Phase0 checkpoint no encontrado, iniciando desde cero", flush=True)

    # Descongelar todo para Phase 1
    unfreeze(pipeline)
    trainable = count_trainable(pipeline)
    print(f"  Parametros entrenables: {trainable:,}", flush=True)

    # Datos
    ids_list = make_mixed_ids(datasets, tok, n=hparams.n_train, max_len=hparams.max_seq)
    val_list = make_mixed_ids(datasets, tok, n=hparams.n_val,   max_len=hparams.max_seq)
    fixed    = get_fixed_examples(datasets)

    # Monitor
    pipeline.to(device)
    monitor = TrainingMonitor(
        model          = pipeline,
        tok            = tok,
        val_ids_list   = val_list[:50],
        fixed_examples = fixed,
        cfg            = cfg,
        eval_every     = hparams.ph1_eval,
        patience       = hparams.ph1_patience,
        convergence_delta  = hparams.conv_delta,
        convergence_window = hparams.ph1_conv_win,
        log_path       = checkpoint_dir / "phase1_monitor.jsonl",
        device         = device,
        is_motor       = True,
    )

    opt   = torch.optim.AdamW(pipeline.parameters(), lr=hparams.ph1_lr, weight_decay=1e-2)
    sched = make_cosine_scheduler(opt, hparams.ph1_warmup, n_steps)

    start_step = 0
    if resume and ckpt_p.exists():
        start_step, _ = load_checkpoint(ckpt_p, pipeline, opt, sched)
        print(f"  Reanudando desde step {start_step}", flush=True)

    use_amp   = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float32

    rng = random.Random(43)
    def get_ids():
        return rng.choice(ids_list)

    losses, elapsed, final_loss, stop_reason = _phase_loop(
        model           = pipeline,
        get_ids         = get_ids,
        optimizer       = opt,
        scheduler       = sched,
        n_steps         = n_steps,
        monitor         = monitor,
        label           = label,
        device          = device,
        checkpoint_path = ckpt_p,
        ckpt_every      = hparams.ckpt_every,
        print_every     = max(1, hparams.ph1_eval // 4),
        use_amp         = use_amp,
        amp_dtype       = amp_dtype,
        start_step      = start_step,
        grad_accum_steps = hparams.grad_accum_steps,
        batch_size       = hparams.train_batch_size,
    )

    save_checkpoint(ckpt_p, pipeline, opt, sched, len(losses) + start_step, final_loss)
    print(f"\n  [Phase1] Checkpoint -> {ckpt_p}", flush=True)

    best_record = (
        min(monitor.eval_history, key=lambda r: r["val_loss"])
        if monitor.eval_history else {}
    )

    return PhaseResult(
        phase         = 1,
        label         = label,
        steps_run     = len(losses),
        final_loss    = final_loss,
        best_val_loss = best_record.get("val_loss", float("nan")),
        best_val_f1   = best_record.get("val_f1",   0.0),
        elapsed_s     = elapsed,
        stop_reason   = stop_reason,
        checkpoint    = str(ckpt_p),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — MOTORES ESPECIALIZADOS
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2_motor(
    pipeline:      MoSEPipeline,
    cfg:           MoSEConfig,
    tok,
    motor_name:    str,
    train_ds,
    val_ds,
    hparams:       _TrainHparams,
    device:        torch.device,
    checkpoint_dir: Path,
    resume:        bool = False,
    max_steps_override: Optional[int] = None,
) -> PhaseResult:
    """Entrena un solo motor con su dataset de dominio."""
    label  = f"Phase2-{motor_name.upper()}"
    ckpt_p = checkpoint_dir / f"phase2_{motor_name}.pt"
    n_steps = max_steps_override or hparams.ph2_steps

    print(f"\n  [{label}] motor={motor_name}  steps_max={n_steps}", flush=True)

    # IMPORTANTE: No congelar el encoder/decoder via requires_grad — eso rompe
    # la cadena de gradientes (concepts.requires_grad=False → motor no recibe grad).
    # En su lugar: dejar todo unfrozen para que el grad fluya, pero solo poner
    # los parámetros del motor en el optimizer. Así el encoder/decoder no se
    # actualizan pero sí permiten el flujo de gradientes.
    unfreeze(pipeline)
    motor_prefix = f"motors.{motor_name}"
    motor_params = [
        p for name, p in pipeline.named_parameters()
        if motor_prefix in name
    ]
    n_motor  = sum(p.numel() for p in motor_params)
    n_frozen = sum(p.numel() for p in pipeline.parameters()) - n_motor
    print(f"  [{label}] Motor params (optimized): {n_motor:,}", flush=True)
    print(f"  [{label}] Other params (grad flows but not optimized): {n_frozen:,}", flush=True)

    # Datos del dominio especifico
    ids_list = make_ids_list(train_ds, tok, n=hparams.n_train, max_len=hparams.max_seq)
    val_list = make_ids_list(val_ds,   tok, n=hparams.n_val,   max_len=hparams.max_seq)
    # 5 ejemplos fijos del mismo dominio
    fixed = [(val_ds.generate().problem_text, val_ds.generate().answer, motor_name)
             for _ in range(5)]

    # Monitor por motor
    monitor = TrainingMonitor(
        model          = pipeline,
        tok            = tok,
        val_ids_list   = val_list[:50],
        fixed_examples = fixed,
        cfg            = cfg,
        eval_every     = hparams.ph2_eval,
        patience       = hparams.ph2_patience,
        convergence_delta  = hparams.conv_delta,
        convergence_window = hparams.ph2_conv_win,
        log_path       = checkpoint_dir / f"phase2_{motor_name}_monitor.jsonl",
        device         = device,
        is_motor       = True,
        motor_hints    = {motor_name: motor_name + " "},
    )

    opt   = torch.optim.AdamW(motor_params, lr=hparams.ph2_lr, weight_decay=1e-2)
    sched = make_cosine_scheduler(opt, hparams.ph2_warmup, n_steps)

    start_step = 0
    if resume and ckpt_p.exists():
        start_step, _ = load_checkpoint(ckpt_p, pipeline, opt, sched)

    use_amp   = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float32

    rng = random.Random(hash(motor_name) % (2**31))
    def get_ids():
        return rng.choice(ids_list)

    losses, elapsed, final_loss, stop_reason = _phase_loop(
        model           = pipeline,
        get_ids         = get_ids,
        optimizer       = opt,
        scheduler       = sched,
        n_steps         = n_steps,
        monitor         = monitor,
        label           = label,
        device          = device,
        checkpoint_path = ckpt_p,
        ckpt_every      = hparams.ckpt_every,
        print_every     = max(1, hparams.ph2_eval // 4),
        use_amp         = use_amp,
        amp_dtype       = amp_dtype,
        start_step      = start_step,
        grad_accum_steps = hparams.grad_accum_steps,
        batch_size       = hparams.train_batch_size,
    )

    save_checkpoint(ckpt_p, pipeline, opt, sched, len(losses) + start_step, final_loss)
    print(f"  [{label}] Checkpoint -> {ckpt_p}", flush=True)

    # Restaurar gradientes para siguiente motor
    unfreeze(pipeline)

    best_record = (
        min(monitor.eval_history, key=lambda r: r["val_loss"])
        if monitor.eval_history else {}
    )

    return PhaseResult(
        phase         = 2,
        label         = label,
        steps_run     = len(losses),
        final_loss    = final_loss,
        best_val_loss = best_record.get("val_loss", float("nan")),
        best_val_f1   = best_record.get("val_f1",   0.0),
        elapsed_s     = elapsed,
        stop_reason   = stop_reason,
        checkpoint    = str(ckpt_p),
        extra         = {"motor": motor_name},
    )


def run_phase2(
    pipeline:      MoSEPipeline,
    cfg:           MoSEConfig,
    tok,
    datasets:      Dict[str, Tuple],
    hparams:       _TrainHparams,
    device:        torch.device,
    checkpoint_dir: Path,
    resume:        bool = False,
    max_steps_override: Optional[int] = None,
    motors:        Optional[List[str]] = None,
) -> List[PhaseResult]:
    """
    Phase 2: entrena cada motor con su dataset de dominio.
    Carga checkpoint de Phase 1 antes de empezar.
    """
    motors_to_train = motors or MOTOR_NAMES

    _banner(2, "MOTORES ESPECIALIZADOS", [
        f"Motores      : {motors_to_train}",
        f"Steps/motor  : {max_steps_override or hparams.ph2_steps}",
        f"LR           : {hparams.ph2_lr:.1e}",
        f"patience     : {hparams.ph2_patience}",
        f"Datos        : {hparams.n_train} ej/motor",
    ])

    # Cargar Phase 1 backbone
    ckpt1 = checkpoint_dir / "phase1.pt"
    if ckpt1.exists():
        ckpt = torch.load(ckpt1, map_location="cpu", weights_only=False)
        pipeline.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  Backbone cargado desde Phase1: {ckpt1}", flush=True)
    else:
        print(f"  [INFO] Phase1 checkpoint no encontrado, usando pesos actuales", flush=True)

    pipeline.to(device)
    results: List[PhaseResult] = []

    for motor_name in motors_to_train:
        if motor_name not in datasets:
            print(f"  [SKIP] {motor_name} — dataset no disponible", flush=True)
            continue
        train_ds, val_ds = datasets[motor_name]
        r = run_phase2_motor(
            pipeline         = pipeline,
            cfg              = cfg,
            tok              = tok,
            motor_name       = motor_name,
            train_ds         = train_ds,
            val_ds           = val_ds,
            hparams          = hparams,
            device           = device,
            checkpoint_dir   = checkpoint_dir,
            resume           = resume,
            max_steps_override = max_steps_override,
        )
        results.append(r)

    # Cargar todos los checkpoints de motors para Phase 3
    for motor_name in motors_to_train:
        ckpt_m = checkpoint_dir / f"phase2_{motor_name}.pt"
        if ckpt_m.exists():
            ckpt = torch.load(ckpt_m, map_location="cpu", weights_only=False)
            pipeline.load_state_dict(ckpt["model_state"], strict=False)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — FINE-TUNE E2E + ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_phase3(
    pipeline:       MoSEPipeline,
    cfg:            MoSEConfig,
    tok,
    datasets:       Dict[str, Tuple],
    hparams:        _TrainHparams,
    device:         torch.device,
    checkpoint_dir: Path,
    resume:         bool = False,
    max_steps_override: Optional[int] = None,
) -> PhaseResult:
    """
    Phase 3: entrena orchestrator routing + fine-tune E2E.
    """
    ckpt_p  = checkpoint_dir / "phase3_final.pt"
    n_orch  = max_steps_override or hparams.ph3_orch_steps
    n_e2e   = max_steps_override or hparams.ph3_e2e_steps

    _banner(3, "FINE-TUNE E2E + ORCHESTRATOR", [
        f"Sub-fase 3a  : Orchestrator routing ({n_orch} steps, lr={hparams.ph3_orch_lr:.1e})",
        f"Sub-fase 3b  : Fine-tune E2E ({n_e2e} steps, lr={hparams.ph3_e2e_lr:.1e})",
        f"eval_every   : {hparams.ph3_eval}",
        f"patience     : {hparams.ph3_patience}",
        f"Checkpoint   : {ckpt_p}",
    ])

    pipeline.to(device)
    use_amp   = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float32

    # ── Sub-fase 3a: Orchestrator routing ────────────────────────────────────
    print("\n  Sub-fase 3a: Orchestrator routing", flush=True)

    # Cargar dataset orchestrator
    try:
        orch_train_ds, orch_val_ds = load_orchestrator_dataset(
            max_examples=5000, eval_size=min(hparams.n_val, 100)
        )
        has_orch_data = True
    except Exception as e:
        print(f"  [WARNING] Dataset orchestrator no disponible: {e}", flush=True)
        has_orch_data = False

    routing_acc_history: List[float] = []

    if has_orch_data:
        # Dejar todo unfrozen para que el grad fluya, pero solo optimizar orchestrator
        unfreeze(pipeline)
        orch_params = [
            p for name, p in pipeline.named_parameters()
            if "orchestrator" in name
        ]
        n_orch_p = sum(p.numel() for p in orch_params)
        print(f"  Parametros orchestrator (optimized): {n_orch_p:,}", flush=True)

        # Pre-tokenizar orchestrator dataset
        orch_ids_list = make_ids_list(orch_train_ds, tok, n=min(hparams.n_train, 1000), max_len=hparams.max_seq)
        orch_labels   = []  # routing labels (motor index)

        rng_orch = random.Random(44)
        # Generar pares (ids, motor_label) para el training de routing
        orch_pairs: List[Tuple[List[int], int]] = []
        for _ in range(len(orch_ids_list)):
            ex = orch_train_ds.generate()
            motor_raw = ex.answer.strip()  # e.g., "EMPATHY"
            motor_canonical = MOTOR_ALIASES.get(motor_raw, motor_raw.lower())
            if motor_canonical not in MOTOR_TO_IDX:
                continue
            ids   = _encode(tok, ex.problem_text, hparams.max_seq)
            label = MOTOR_TO_IDX[motor_canonical]
            orch_pairs.append((ids, label))

        if orch_pairs:
            opt_orch  = torch.optim.AdamW(
                orch_params,
                lr=hparams.ph3_orch_lr, weight_decay=1e-2
            )
            sched_orch = make_cosine_scheduler(opt_orch, hparams.ph3_warmup, n_orch)

            # Convergence tracking for orchestrator routing
            orch_eval_every = hparams.ph3_eval
            orch_best_acc = 0.0
            orch_no_improve = 0
            orch_patience = hparams.ph3_patience

            pipeline.train()
            orch_losses: List[float] = []
            for step in range(1, n_orch + 1):
                ids, tgt_idx = rng_orch.choice(orch_pairs)
                ids_t = torch.tensor([ids], dtype=torch.long, device=device)
                tgt_t = torch.tensor([tgt_idx], dtype=torch.long, device=device)

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    concepts   = pipeline.encoder(ids_t)          # [1, L, D]
                    pooled     = concepts.mean(dim=1).mean(dim=0, keepdim=True)  # [1, D]
                    orch_logits = pipeline.orchestrator.classifier(pooled)       # [1, 5]
                    route_loss = F.cross_entropy(orch_logits, tgt_t)             # scalar

                if not math.isfinite(route_loss.item()):
                    continue

                opt_orch.zero_grad(set_to_none=True)
                route_loss.backward()
                nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
                opt_orch.step()
                sched_orch.step()
                orch_losses.append(route_loss.item())

                if step % orch_eval_every == 0:
                    pipeline.eval()
                    correct = 0
                    total   = 0
                    with torch.no_grad():
                        for _ in range(min(50, len(orch_pairs))):
                            v_ids, v_tgt = rng_orch.choice(orch_pairs)
                            v_ids_t = torch.tensor([v_ids], dtype=torch.long, device=device)
                            v_out   = pipeline.orchestrator(pipeline.encoder(v_ids_t))
                            pred    = v_out.logits.argmax().item()
                            correct += int(pred == v_tgt)
                            total   += 1
                    acc = correct / total if total > 0 else 0.0
                    routing_acc_history.append(acc)
                    avg_loss = sum(orch_losses[-20:]) / len(orch_losses[-20:])
                    print(
                        f"    [Phase3-Orch] step {step:>6}  "
                        f"loss={avg_loss:.4f}  routing_acc={acc:.3f}",
                        flush=True,
                    )
                    pipeline.train()

                    # Early stopping on routing accuracy
                    if acc > orch_best_acc + 0.01:
                        orch_best_acc = acc
                        orch_no_improve = 0
                    else:
                        orch_no_improve += orch_eval_every
                    if orch_no_improve >= orch_patience:
                        print(f"  [Phase3a] Early stop: routing_acc={acc:.3f} "
                              f"(best={orch_best_acc:.3f})", flush=True)
                        break
                    # Stop if accuracy is good enough
                    if acc >= 0.8:
                        print(f"  [Phase3a] Accuracy threshold reached: {acc:.3f}",
                              flush=True)
                        break

            print(f"  [Phase3a] Routing loss final: "
                  f"{sum(orch_losses[-20:])/len(orch_losses[-20:]) if orch_losses else float('nan'):.4f}"
                  f"  steps={len(orch_losses)}  best_acc={orch_best_acc:.3f}",
                  flush=True)

    # ── Sub-fase 3b: Fine-tune E2E ────────────────────────────────────────────
    print("\n  Sub-fase 3b: Fine-tune E2E (lr bajo)", flush=True)

    unfreeze(pipeline)
    trainable = count_trainable(pipeline)
    print(f"  Parametros entrenables (E2E): {trainable:,}", flush=True)

    # Datos mezclados para E2E
    ids_list = make_mixed_ids(datasets, tok, n=hparams.n_train, max_len=hparams.max_seq)
    val_list = make_mixed_ids(datasets, tok, n=hparams.n_val,   max_len=hparams.max_seq)
    fixed    = get_fixed_examples(datasets)

    monitor = TrainingMonitor(
        model          = pipeline,
        tok            = tok,
        val_ids_list   = val_list[:50],
        fixed_examples = fixed,
        cfg            = cfg,
        eval_every     = hparams.ph3_eval,
        patience       = hparams.ph3_patience,
        convergence_delta  = hparams.conv_delta,
        convergence_window = hparams.ph3_conv_win,
        log_path       = checkpoint_dir / "phase3_e2e_monitor.jsonl",
        device         = device,
        is_motor       = True,
    )

    opt_e2e  = torch.optim.AdamW(pipeline.parameters(), lr=hparams.ph3_e2e_lr, weight_decay=1e-2)
    sched_e2e = make_cosine_scheduler(opt_e2e, hparams.ph3_warmup, n_e2e)

    rng_e2e = random.Random(45)
    def get_ids_e2e():
        return rng_e2e.choice(ids_list)

    losses_e2e, elapsed_e2e, final_loss, stop_reason = _phase_loop(
        model           = pipeline,
        get_ids         = get_ids_e2e,
        optimizer       = opt_e2e,
        scheduler       = sched_e2e,
        n_steps         = n_e2e,
        monitor         = monitor,
        label           = "Phase3-E2E",
        device          = device,
        checkpoint_path = ckpt_p,
        ckpt_every      = hparams.ckpt_every,
        print_every     = max(1, hparams.ph3_eval // 4),
        use_amp         = use_amp,
        amp_dtype       = amp_dtype,
        grad_accum_steps = hparams.grad_accum_steps,
        batch_size       = hparams.train_batch_size,
    )

    save_checkpoint(ckpt_p, pipeline, opt_e2e, sched_e2e,
                    len(losses_e2e), final_loss)
    print(f"\n  [Phase3] Checkpoint final -> {ckpt_p}", flush=True)

    best_record = (
        min(monitor.eval_history, key=lambda r: r["val_loss"])
        if monitor.eval_history else {}
    )
    final_routing_acc = routing_acc_history[-1] if routing_acc_history else float("nan")

    return PhaseResult(
        phase         = 3,
        label         = "Phase3-E2E+Orch",
        steps_run     = len(losses_e2e),
        final_loss    = final_loss,
        best_val_loss = best_record.get("val_loss", float("nan")),
        best_val_f1   = best_record.get("val_f1",   0.0),
        elapsed_s     = elapsed_e2e,
        stop_reason   = stop_reason,
        checkpoint    = str(ckpt_p),
        extra         = {"routing_acc": final_routing_acc},
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE PIPELINE Y TOKENIZADOR
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline_and_tok(
    config_name: str,
    device:      torch.device,
) -> Tuple[MoSEPipeline, Any, MoSEConfig]:
    """
    Construye el pipeline y el tokenizador para el config dado.

    Returns:
        (pipeline, tok, mose_cfg)
    """
    if config_name == "tiny":
        mose_cfg = MoSEConfig.tiny()
    elif config_name == "medium":
        # Medium ~340M params: hidden_dim=768, 6 enc layers, 12 dec layers
        mose_cfg = MoSEConfig(
            hidden_dim    = 768,
            vocab_size    = 32_000,
            enc_n_layers  = 6,
            dec_n_layers  = 12,
            enc_state_dim = 16,
            dec_state_dim = 16,
            dec_n_heads   = 8,
            motor_n_heads = 8,
            unif_n_heads  = 8,
            dec_max_seq_len = 512,
            orch_mlp_hidden = 512,
        )
    elif config_name == "production":
        # Production 3.6B: hidden_dim=1536 uniforme, 14 enc layers, 28 dec layers
        # Verificado en H200: 40 GB peak VRAM con bf16, cabe holgadamente en 143 GB
        # MoSEScaleConfig.production() usa dims asimetricos (dec=1536, motors=768)
        # pero MoSEConfig requiere hidden_dim uniforme. Con hd=1536, dl=28 llegamos
        # a 3.6B params que es cercano al target de 3.44B.
        mose_cfg = MoSEConfig(
            hidden_dim    = 1536,
            vocab_size    = 32_000,
            enc_n_layers  = 14,
            dec_n_layers  = 28,
            enc_state_dim = 16,
            dec_state_dim = 16,
            dec_n_heads   = 16,
            motor_n_heads = 8,
            unif_n_heads  = 8,
            dec_max_seq_len = 2048,
            orch_mlp_hidden = 2048,
            motor_max_nodes = 32,
        )
    else:
        raise ValueError(f"Config desconocido: {config_name!r}. Usa tiny/medium/production.")

    print(f"  Creando MoSEPipeline (config={config_name})...", flush=True)
    pipeline = MoSEPipeline(mose_cfg).to(device)
    params   = sum(p.numel() for p in pipeline.parameters())
    print(f"  Parametros totales: {params:,} ({params/1e6:.1f}M)", flush=True)

    # Tokenizador
    corpus_texts: List[str] = []
    if mose_cfg.vocab_size < 32_000:
        # Construir vocab desde corpus Opus
        try:
            from experiments.opus_dataset import OpusDataset
            for motor in MOTOR_NAMES[:3]:
                ds = OpusDataset(motor=motor, max_examples=500)
                corpus_texts.extend(ds.get_all_texts())
        except Exception:
            corpus_texts = ["the quick brown fox"] * 200

    tok = build_tokenizer(mose_cfg.vocab_size, corpus_texts)
    print(f"  Tokenizador: vocab_size={tok.vocab_size}", flush=True)

    return pipeline, tok, mose_cfg


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE EVAL
# ─────────────────────────────────────────────────────────────────────────────

def interactive_eval(
    pipeline:  MoSEPipeline,
    tok,
    device:    torch.device,
    max_len:   int = 128,
    questions: Optional[List[str]] = None,
) -> bool:
    """
    Abre un prompt interactivo para probar el modelo.

    Returns:
        True si el usuario quiere continuar con interactive mode,
        False si escribió 'quit'.
    """
    pipeline.eval()

    if questions is not None:
        # Automated mode (for smoke tests)
        for q in questions:
            print(f"\n  [Interactive] Q: {q}", flush=True)
            ids = _encode(tok, q, max_len)
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = pipeline(ids_t)
            pred_ids = out.logits[0].argmax(dim=-1).tolist()
            try:
                resp = tok.decode(pred_ids)
            except Exception:
                resp = str(pred_ids[:20])
            print(f"  [Interactive] A: {resp[:200]}", flush=True)
        return True

    # Manual interactive mode
    print("\n" + "-" * 50, flush=True)
    print("  INTERACTIVE EVAL (Enter vacio = continuar, 'quit' = desactivar)", flush=True)
    print("-" * 50, flush=True)

    while True:
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("", flush=True)
            return True

        if user_input == "":
            return True  # continue training with interactive
        if user_input.lower() == "quit":
            return False  # disable interactive

        ids = _encode(tok, user_input, max_len)
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = pipeline(ids_t)
        pred_ids = out.logits[0].argmax(dim=-1).tolist()
        try:
            resp = tok.decode(pred_ids)
        except Exception:
            resp = str(pred_ids[:20])
        print(f"  AION-C: {resp[:500]}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — INSTRUCTION TUNING CON LoRA
# ─────────────────────────────────────────────────────────────────────────────

class _LoRALinear(nn.Module):
    """
    LoRA adapter: W_frozen + alpha/rank * B @ A.

    Solo A y B son entrenables. W_frozen queda congelado.

    Expone .weight y .bias como properties para compatibilidad con
    módulos de PyTorch que acceden directamente a estos atributos
    (e.g. MultiheadAttention accede a self.out_proj.weight).
    """
    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank     = rank
        self.alpha    = alpha
        self.in_features  = original.in_features
        self.out_features = original.out_features
        in_f  = original.in_features
        out_f = original.out_features
        # A: down-project, B: up-project
        self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        # Freeze original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        """Expone el peso base para compatibilidad con PyTorch internals."""
        return self.original.weight

    @property
    def bias(self):
        """Expone el bias base para compatibilidad con PyTorch internals."""
        return self.original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * (self.alpha / self.rank)
        return base + lora


def apply_lora(module: nn.Module, rank: int = 16, alpha: float = 16.0, prefix: str = "") -> int:
    """
    Aplica LoRA a todos los nn.Linear de un módulo.
    Retorna el número de adaptadores aplicados.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, _LoRALinear(child, rank, alpha))
            count += 1
        else:
            count += apply_lora(child, rank, alpha, prefix=f"{prefix}.{name}")
    return count


def run_phase4(
    pipeline:       MoSEPipeline,
    cfg:            MoSEConfig,
    tok,
    hparams:        _TrainHparams,
    device:         torch.device,
    checkpoint_dir: Path,
    resume:         bool = False,
    max_steps_override: Optional[int] = None,
    interactive:    bool = False,
) -> PhaseResult:
    """
    Phase 4: Instruction tuning con LoRA rank=16 en decoder.

    Carga el dataset de instruction tuning generado por synth/instruction_gen.py.
    Aplica LoRA al decoder (encoder congelado, motores congelados).
    """
    label   = "Phase4-InstructionTuning"
    ckpt_p  = checkpoint_dir / "phase4_instruction.pt"
    n_steps = max_steps_override or getattr(hparams, 'ph4_steps', hparams.ph3_e2e_steps)

    _banner(4, "INSTRUCTION TUNING (LoRA decoder)", [
        f"Steps max    : {n_steps}",
        f"LoRA rank    : 16",
        f"Interactive  : {interactive}",
        f"Checkpoint   : {ckpt_p}",
    ])

    # Cargar Phase 3 checkpoint si existe
    ckpt3 = checkpoint_dir / "phase3_final.pt"
    if ckpt3.exists():
        ckpt = torch.load(ckpt3, map_location="cpu", weights_only=False)
        pipeline.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  Cargado Phase3: {ckpt3}", flush=True)

    # Congelar todo
    freeze(pipeline)

    # Aplicar LoRA al decoder
    n_lora = apply_lora(pipeline.decoder, rank=16, alpha=16.0)
    print(f"  LoRA adaptadores aplicados al decoder: {n_lora}", flush=True)

    trainable = count_trainable(pipeline)
    print(f"  Parametros entrenables (LoRA): {trainable:,}", flush=True)

    pipeline.to(device)

    # Cargar instruction tuning dataset
    try:
        from synth.instruction_gen import load_instruction_dataset
        it_data = load_instruction_dataset()
        print(f"  Instruction tuning dataset: {len(it_data)} ejemplos", flush=True)
    except (FileNotFoundError, ImportError) as e:
        print(f"  [WARNING] IT dataset no disponible: {e}", flush=True)
        print(f"  Generando dataset on-the-fly...", flush=True)
        try:
            from synth.instruction_gen import InstructionGenerator, write_jsonl
            gen = InstructionGenerator(seed=42)
            it_data = gen.generate_all()
            out_path = _ROOT / "datasets" / "instruction_tuning.jsonl"
            write_jsonl(it_data, out_path)
        except Exception as e2:
            print(f"  [ERROR] No se puede generar IT dataset: {e2}", flush=True)
            it_data = []

    # Pre-tokenizar
    max_seq = hparams.max_seq
    if it_data:
        rng_it = random.Random(46)
        ids_list = []
        for ex in it_data[:hparams.n_train]:
            text = ex.get("instruction", "") + " " + ex.get("response", "")
            if ex.get("system_prompt"):
                text = ex["system_prompt"] + "\n" + text
            ids = _encode(tok, text, max_seq)
            ids_list.append(ids)
    else:
        ids_list = _make_synthetic_ids(tok, hparams.n_train, max_seq)

    if not ids_list:
        return PhaseResult(
            phase=4, label=label, steps_run=0, final_loss=float("nan"),
            best_val_loss=float("nan"), best_val_f1=0.0, elapsed_s=0.0,
            stop_reason="no_data", checkpoint=str(ckpt_p),
        )

    # Optimizer (solo LoRA params)
    lora_params = [p for p in pipeline.parameters() if p.requires_grad]
    lr = getattr(hparams, 'ph4_lr', 1e-4)
    opt   = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-2)
    sched = make_cosine_scheduler(opt, min(200, max(1, n_steps // 10)), n_steps)

    eval_every = max(1, getattr(hparams, 'ph4_eval', hparams.ph3_eval))

    use_amp   = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float32

    rng = random.Random(47)
    interactive_active = interactive

    # Convergence tracking for Phase 4
    ph4_best_loss = float("inf")
    ph4_no_improve = 0
    ph4_patience = getattr(hparams, 'ph0_patience', 500)  # reuse patience
    ph4_conv_window = getattr(hparams, 'ph0_conv_win', 300)
    ph4_conv_delta = getattr(hparams, 'conv_delta', 0.001)
    ph4_prev_avg = float("inf")
    ph4_stable_steps = 0
    print_every = max(1, eval_every // 4)
    stop_reason = "max_steps"

    pipeline.train()
    losses: List[float] = []
    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        ids   = rng.choice(ids_list)
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            out    = pipeline(ids_t)
            logits = out.logits
            ce     = F.cross_entropy(logits[0, :-1], ids_t[0, 1:], ignore_index=PAD)

        if not math.isfinite(ce.item()):
            continue

        opt.zero_grad(set_to_none=True)
        ce.backward()
        nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        sched.step()
        losses.append(ce.item())

        if step % print_every == 0:
            recent = losses[-50:]
            avg = sum(recent) / len(recent) if recent else float("nan")
            elapsed_now = time.perf_counter() - t0
            sps = step / max(0.1, elapsed_now)
            print(f"    [Phase4-IT] step {step:>6}/{n_steps}  loss={avg:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  {sps:.1f} steps/s",
                  flush=True)

        # Interactive eval
        if interactive_active and step % eval_every == 0:
            interactive_active = interactive_eval(pipeline, tok, device, max_seq)
            pipeline.train()

        # Convergence check every eval_every
        if step % eval_every == 0 and len(losses) >= 50:
            avg_loss = sum(losses[-50:]) / 50
            # Early stopping
            if avg_loss < ph4_best_loss - 1e-6:
                ph4_best_loss = avg_loss
                ph4_no_improve = 0
            else:
                ph4_no_improve += eval_every
            if ph4_no_improve >= ph4_patience:
                print(f"  [Phase4] Early stop: loss={avg_loss:.4f} "
                      f"(best={ph4_best_loss:.4f})", flush=True)
                stop_reason = "early_stop"
                break
            # Convergence
            delta = abs(ph4_prev_avg - avg_loss)
            if delta < ph4_conv_delta:
                ph4_stable_steps += eval_every
            else:
                ph4_stable_steps = 0
            ph4_prev_avg = avg_loss
            if ph4_stable_steps >= ph4_conv_window:
                print(f"  [Phase4] Converged: loss={avg_loss:.4f} "
                      f"(stable for {ph4_stable_steps} steps)", flush=True)
                stop_reason = "converged"
                break

        # Checkpoint
        if step % hparams.ckpt_every == 0:
            save_checkpoint(ckpt_p, pipeline, opt, sched, step,
                           sum(losses[-50:]) / min(50, len(losses)))

    elapsed = time.perf_counter() - t0
    final_loss = sum(losses[-50:]) / min(50, len(losses)) if losses else float("nan")

    # Save
    save_checkpoint(ckpt_p, pipeline, opt, sched, len(losses), final_loss)
    print(f"\n  [Phase4] Checkpoint -> {ckpt_p}", flush=True)

    return PhaseResult(
        phase=4, label=label, steps_run=len(losses), final_loss=final_loss,
        best_val_loss=ph4_best_loss, best_val_f1=0.0, elapsed_s=elapsed,
        stop_reason=stop_reason, checkpoint=str(ckpt_p),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AION-C Master Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",    default="tiny",
                   choices=["tiny", "medium", "production"],
                   help="Configuracion del modelo")
    p.add_argument("--phase",     default="all",
                   choices=["0", "1", "2", "3", "4", "all"],
                   help="Fase(s) a ejecutar")
    p.add_argument("--resume",    action="store_true",
                   help="Reanudar desde checkpoint si existe")
    p.add_argument("--device",    default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="Device de entrenamiento")
    p.add_argument("--run-dir",   default=None,
                   help="Directorio de checkpoints y logs (default: runs/aion_<config>)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override del maximo de steps por fase (debug)")
    p.add_argument("--motors",    nargs="*", default=None,
                   help="Motores a entrenar en Phase2 (default: todos)")
    p.add_argument("--interactive", action="store_true",
                   help="Modo interactivo: cada eval_every steps abre prompt para probar el modelo")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir = Path(args.run_dir) if args.run_dir else _ROOT / "runs" / f"aion_{args.config}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "#" * 65, flush=True)
    print(f"  AION-C Master Training Script", flush=True)
    print(f"  config={args.config}  phase={args.phase}  device={device}", flush=True)
    print(f"  run_dir={run_dir}", flush=True)
    print("#" * 65 + "\n", flush=True)

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline, tok, mose_cfg = build_pipeline_and_tok(args.config, device)
    hparams = copy.copy(_PRESETS[args.config])   # shallow copy — campos son primitivos
    if args.max_steps is not None:
        # Override todos los step counts (sin mutar el preset global)
        hparams.ph0_steps = hparams.ph1_steps = hparams.ph2_steps = args.max_steps
        hparams.ph3_orch_steps = hparams.ph3_e2e_steps = max(1, args.max_steps // 2)
        hparams.ph4_steps = args.max_steps
        hparams.ph0_eval = hparams.ph1_eval = hparams.ph2_eval = hparams.ph3_eval = max(
            5, args.max_steps // 4
        )
        hparams.ph4_eval = max(5, args.max_steps // 4)

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("  Cargando datasets Opus...", flush=True)
    datasets = load_all_datasets(
        max_examples = hparams.n_train + hparams.n_val + 100,
        eval_size    = hparams.n_val,
    )
    available = list(datasets.keys())
    print(f"  Dominios disponibles: {available}", flush=True)

    # ── Execute phases ────────────────────────────────────────────────────────
    phases_to_run = (
        ["0", "1", "2", "3", "4"] if args.phase == "all" else [args.phase]
    )

    all_results: List[PhaseResult] = []
    t_global = time.perf_counter()

    if "0" in phases_to_run:
        r = run_phase0(
            pipeline        = pipeline,
            cfg             = mose_cfg,
            tok             = tok,
            datasets        = datasets,
            hparams         = hparams,
            device          = device,
            checkpoint_dir  = run_dir,
            resume          = args.resume,
            max_steps_override = args.max_steps,
        )
        all_results.append(r)

    if "1" in phases_to_run:
        r = run_phase1(
            pipeline        = pipeline,
            cfg             = mose_cfg,
            tok             = tok,
            datasets        = datasets,
            hparams         = hparams,
            device          = device,
            checkpoint_dir  = run_dir,
            resume          = args.resume,
            max_steps_override = args.max_steps,
        )
        all_results.append(r)

    if "2" in phases_to_run:
        phase2_results = run_phase2(
            pipeline        = pipeline,
            cfg             = mose_cfg,
            tok             = tok,
            datasets        = datasets,
            hparams         = hparams,
            device          = device,
            checkpoint_dir  = run_dir,
            resume          = args.resume,
            max_steps_override = args.max_steps,
            motors          = args.motors,
        )
        all_results.extend(phase2_results)

    if "3" in phases_to_run:
        r = run_phase3(
            pipeline        = pipeline,
            cfg             = mose_cfg,
            tok             = tok,
            datasets        = datasets,
            hparams         = hparams,
            device          = device,
            checkpoint_dir  = run_dir,
            resume          = args.resume,
            max_steps_override = args.max_steps,
        )
        all_results.append(r)

    if "4" in phases_to_run:
        r = run_phase4(
            pipeline        = pipeline,
            cfg             = mose_cfg,
            tok             = tok,
            hparams         = hparams,
            device          = device,
            checkpoint_dir  = run_dir,
            resume          = args.resume,
            max_steps_override = args.max_steps,
            interactive     = getattr(args, 'interactive', False),
        )
        all_results.append(r)

    # ── Resumen ───────────────────────────────────────────────────────────────
    total_s = time.perf_counter() - t_global
    _summary(all_results)
    print(f"\n  Tiempo global: {total_s/60:.1f} min", flush=True)

    # Guardar resumen JSON
    summary_path = run_dir / "training_summary.json"
    try:
        summary_data = [
            {
                "phase":        r.phase,
                "label":        r.label,
                "steps_run":    r.steps_run,
                "final_loss":   r.final_loss,
                "best_val_loss": r.best_val_loss,
                "best_val_f1":  r.best_val_f1,
                "elapsed_s":    r.elapsed_s,
                "stop_reason":  r.stop_reason,
                "checkpoint":   r.checkpoint,
                **r.extra,
            }
            for r in all_results
        ]
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, default=str)
        print(f"  Resumen guardado en: {summary_path}", flush=True)
    except Exception:
        pass

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
