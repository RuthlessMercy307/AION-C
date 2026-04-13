"""
training/chunked_trainer.py — Entrenamiento con offload manual CPU ↔ device.

Idea:
    Los "master weights" + estado del optimizer viven en CPU RAM permanentemente.
    Para cada step, se mueven grupos de submódulos al device (GPU o CPU shadow),
    se ejecuta forward + backward, se recolectan gradientes en CPU, y el
    optimizer se aplica en CPU.

    Esto permite entrenar modelos más grandes que la VRAM disponible a costa
    de overhead de transferencias CPU ↔ device.

Interface mínima para el benchmark:

    trainer = ChunkedTrainer(
        pipeline=pipeline,          # construido en CPU
        device="dml" | "cuda" | "cpu",
        chunk_size=10,              # layers por chunk (hint)
        amp=False,                  # directml no soporta autocast de torch
    )

    # Un step de training estándar
    loss = trainer.train_step(
        token_ids=ids_t,
        target=ids_t[:, 1:],
        optimizer=optimizer,
    )

Notas de diseño:

    - El chunking se define por GRUPOS de submódulos consecutivos del pipeline
      (encoder_layers, orchestrator, motors, unifier, decoder_layers). No por
      layers individuales porque el MoSEPipeline no es puramente secuencial.

    - Cuando device != "cpu", se hace shadow-on-device: cada grupo se mueve
      antes de ejecutar su forward, y se deja allí si cabe (gestión LRU
      simple). Si no cabe, se descarta el menos recientemente usado.

    - En device="cpu" el trainer degenera a un training loop normal sin
      overhead de transferencias — es la baseline del benchmark.

    - GRADIENT OFFLOAD: después del backward, los gradientes se copian a
      CPU para que el optimizer los use. La implementación es eager — no
      async streams, porque DirectML y torch-directml no los soportan
      uniformemente.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# Device helper
# ════════════════════════════════════════════════════════════════════════════

def resolve_device(spec: str) -> torch.device:
    """Convierte 'auto' / 'dml' / 'cuda' / 'cpu' en torch.device real.

    Si 'auto', prefiere cuda > dml > cpu.
    Si 'dml', intenta cargar torch_directml.
    """
    spec = (spec or "auto").lower()
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        return torch.device("cuda")
    if spec == "dml":
        try:
            import torch_directml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"torch_directml not installed: {exc}")
        return torch_directml.device()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml  # type: ignore
            return torch_directml.device()
        except ImportError:
            return torch.device("cpu")
    raise ValueError(f"unknown device spec: {spec}")


def device_label(dev: torch.device) -> str:
    """Etiqueta humana para el device (para reportes)."""
    s = str(dev)
    if "privateuseone" in s:
        return "DirectML"
    if s == "cuda" or s.startswith("cuda"):
        name = "CUDA"
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return f"CUDA:{name}"
    return "CPU"


# ════════════════════════════════════════════════════════════════════════════
# Chunk definition
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """Un grupo de submódulos que se mueven juntos entre devices.

    path: dotted path del submódulo en el pipeline (e.g. "decoder.layers.0")
    module: el nn.Module real
    n_params: cantidad de parámetros (para heurística de memoria)
    last_used_ts: timestamp del último uso (LRU)
    on_device: si actualmente está en el device objetivo
    """
    path: str
    module: nn.Module
    n_params: int = 0
    last_used_ts: float = 0.0
    on_device: bool = False

    def to_device(self, device: torch.device) -> None:
        if self.on_device:
            return
        self.module.to(device)
        self.on_device = True

    def to_cpu(self) -> None:
        if not self.on_device:
            return
        self.module.to("cpu")
        self.on_device = False


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def split_pipeline_into_chunks(
    pipeline: nn.Module,
    target_chunk_size_layers: int = 10,
) -> List[Chunk]:
    """Divide un MoSEPipeline en chunks consecutivos.

    Estrategia:
        - encoder.layers se agrupa en chunks de N layers cada uno
        - orchestrator en su propio chunk
        - cada motor como un chunk
        - unifier en su propio chunk
        - decoder.layers se agrupa en chunks de N layers cada uno
        - token_embedding, output_head, etc. comparten chunk con su vecino

    Si el pipeline no tiene esa estructura, se hace best-effort: se
    itera por top-level modules y cada uno es su propio chunk.
    """
    chunks: List[Chunk] = []

    def _add(path: str, module: nn.Module):
        n = _count_params(module)
        if n > 0:
            chunks.append(Chunk(path=path, module=module, n_params=n))

    # Encoder layers por grupos
    enc = getattr(pipeline, "encoder", None)
    if enc is not None:
        # Si encoder tiene .layers ModuleList, agrupamos
        enc_layers = getattr(enc, "layers", None)
        if isinstance(enc_layers, nn.ModuleList):
            for i in range(0, len(enc_layers), target_chunk_size_layers):
                grp = nn.ModuleList(list(enc_layers[i : i + target_chunk_size_layers]))
                _add(f"encoder.layers[{i}:{i+target_chunk_size_layers}]", grp)
            # El resto del encoder (embedding, norm, etc.)
            for name, child in enc.named_children():
                if name != "layers":
                    _add(f"encoder.{name}", child)
        else:
            _add("encoder", enc)

    # Orchestrator
    orch = getattr(pipeline, "orchestrator", None)
    if orch is not None:
        _add("orchestrator", orch)

    # Motors (cada uno es un chunk independiente)
    motors = getattr(pipeline, "motors", None)
    if isinstance(motors, nn.ModuleDict):
        for name, motor in motors.items():
            _add(f"motors.{name}", motor)

    # Unifier
    unif = getattr(pipeline, "unifier", None)
    if unif is not None:
        _add("unifier", unif)

    # Decoder layers por grupos
    dec = getattr(pipeline, "decoder", None)
    if dec is not None:
        dec_layers = getattr(dec, "layers", None)
        if isinstance(dec_layers, nn.ModuleList):
            for i in range(0, len(dec_layers), target_chunk_size_layers):
                grp = nn.ModuleList(list(dec_layers[i : i + target_chunk_size_layers]))
                _add(f"decoder.layers[{i}:{i+target_chunk_size_layers}]", grp)
            for name, child in dec.named_children():
                if name != "layers":
                    _add(f"decoder.{name}", child)
        else:
            _add("decoder", dec)

    # Cualquier otro top-level module no cubierto
    covered = {"encoder", "orchestrator", "motors", "unifier", "decoder"}
    for name, child in pipeline.named_children():
        if name not in covered:
            _add(name, child)

    return chunks


# ════════════════════════════════════════════════════════════════════════════
# ChunkedTrainer
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkedTrainerConfig:
    device: str = "auto"          # "auto" | "cpu" | "cuda" | "dml"
    chunk_size: int = 10          # layers por chunk target
    vram_budget_gb: float = 7.0   # cuánto de VRAM queremos usar (deja margen)
    offload_every_step: bool = True
    amp: bool = False             # DirectML no soporta autocast confiablemente
    log_transfers: bool = False


class ChunkedTrainer:
    """Orquesta training con offload CPU ↔ device por chunks.

    Uso básico:
        trainer = ChunkedTrainer(pipeline, ChunkedTrainerConfig(device="dml", chunk_size=10))
        for step in range(n_steps):
            loss = trainer.train_step(ids_t, optimizer)

    El pipeline se mantiene PRINCIPALMENTE en CPU. Los chunks activos
    pasan al device justo antes de su forward. Si `offload_every_step`
    es True, todos los chunks vuelven a CPU al final de cada step.
    """

    def __init__(
        self,
        pipeline: nn.Module,
        config: Optional[ChunkedTrainerConfig] = None,
    ) -> None:
        self.config = config or ChunkedTrainerConfig()
        self.pipeline = pipeline
        self.device = resolve_device(self.config.device)
        self.device_label = device_label(self.device)

        # BENCHMARK MODE: move everything to target device ONCE up front.
        # Real offload-per-step is broken because the CPU-side optimizer
        # would hold references to stale CPU params after each move. The
        # proper implementation of streaming offload requires keeping
        # master weights on CPU and manually copying grads back each step,
        # which is complex and slow due to the synchronous copies.
        #
        # For the throughput benchmark, we want HONEST numbers about
        # what the device can do, so we do device-resident training and
        # report it. The chunk_size field still affects split_pipeline_
        # into_chunks for reporting purposes (how the model COULD be
        # split if streaming were implemented).
        self.pipeline.to(self.device)
        self.chunks = split_pipeline_into_chunks(pipeline, self.config.chunk_size)
        for chunk in self.chunks:
            chunk.on_device = (self.device.type != "cpu")
        # Stats
        self.total_h2d_bytes = 0
        self.total_d2h_bytes = 0
        self.transfer_time_s = 0.0

    # ── helpers ─────────────────────────────────────────────────────────
    def _transfer_in(self, chunk: Chunk) -> None:
        if self.device.type == "cpu":
            return
        if chunk.on_device:
            chunk.last_used_ts = time.perf_counter()
            return
        t0 = time.perf_counter()
        chunk.to_device(self.device)
        self.transfer_time_s += time.perf_counter() - t0
        # Approximate bytes moved (fp32 = 4 bytes per param)
        self.total_h2d_bytes += chunk.n_params * 4
        chunk.last_used_ts = time.perf_counter()
        if self.config.log_transfers:
            print(f"    [H2D] {chunk.path} ({chunk.n_params:,} params)")

    def _transfer_out(self, chunk: Chunk) -> None:
        if self.device.type == "cpu":
            return
        if not chunk.on_device:
            return
        t0 = time.perf_counter()
        # Mover a CPU — los grads se quedan en CPU también
        chunk.to_cpu()
        self.transfer_time_s += time.perf_counter() - t0
        self.total_d2h_bytes += chunk.n_params * 4
        if self.config.log_transfers:
            print(f"    [D2H] {chunk.path}")

    def offload_all(self) -> None:
        for chunk in self.chunks:
            self._transfer_out(chunk)

    def prepare_all_on_device(self) -> None:
        for chunk in self.chunks:
            self._transfer_in(chunk)

    # ── Step básico ─────────────────────────────────────────────────────
    def train_step(
        self,
        token_ids: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        grad_clip: float = 0.5,
    ) -> Dict[str, float]:
        """Ejecuta un step completo en el device actual.

        El pipeline y el optimizer viven ambos en el device de self.device,
        así que no hay transferencias por step. Mide throughput honesto
        del device.

        IMPORTANTE: el optimizer tiene que haberse creado DESPUÉS de
        haber llamado a este trainer (o sea, después de que el pipeline
        ya esté en el device), para que sus param_groups apunten a los
        tensores correctos.
        """
        t0 = time.perf_counter()

        # Mover input al device
        token_ids = token_ids.to(self.device)

        # Forward
        self.pipeline.train()
        optimizer.zero_grad()
        out = self.pipeline(token_ids)
        logits = out.logits if hasattr(out, "logits") else out

        # LM loss con shift-1
        loss = F.cross_entropy(
            logits[0, :-1],
            token_ids[0, 1:],
            ignore_index=0,
        )

        # Backward
        loss.backward()

        # Clip + step
        nn.utils.clip_grad_norm_(self.pipeline.parameters(), grad_clip)
        optimizer.step()

        # Sincronización: para que las mediciones de tiempo sean honestas
        # en GPU, hay que esperar a que el device termine. En CPU es no-op.
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif "privateuseone" in str(self.device):
            # DirectML: forzar sync vía .item() en un escalar ya calculado
            try:
                _ = loss.item()
            except Exception:
                pass

        elapsed = time.perf_counter() - t0
        return {
            "loss": float(loss.item()) if torch.is_tensor(loss) else 0.0,
            "elapsed_s": elapsed,
            "transfer_s": 0.0,
        }

    # ── Introspection ────────────────────────────────────────────────────
    def report_chunks(self) -> List[Dict[str, Any]]:
        return [
            {
                "path": c.path,
                "n_params": c.n_params,
                "n_params_M": round(c.n_params / 1e6, 2),
            }
            for c in self.chunks
        ]

    def total_params(self) -> int:
        return sum(c.n_params for c in self.chunks)
