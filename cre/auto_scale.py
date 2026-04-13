"""
cre/auto_scale.py — Auto-detección del batch_size óptimo por throughput
========================================================================

AutoScaler determina el batch_size que MAXIMIZA samples/s, no el máximo
que cabe en memoria. Un batch muy grande puede ser más lento que uno mediano
porque la GPU se satura y el overhead de concatenar/desempaquetar domina.

Estrategia:
    GPU:
        1. Filtro VRAM: descartar candidatos que OOMean o dejan <20% libre.
        2. Timing probe: 3 pasos cronometrados por candidato (con GPU sync).
        3. Devolver el candidato con mayor samples/s.
    CPU:
        Heurística RAM: 1 batch por 2GB disponibles, máximo 8.

Candidatos probados (potencias de 2, empezando en 8 — por debajo no vale):
    [8, 16, 32, 64, 128, 256, 512]

Ejemplo de salida en T4 15GB con modelo tiny:
    batch=  8:   87 samp/s  (step 92ms)
    batch= 16:  163 samp/s  (step 98ms)
    batch= 32:  301 samp/s  (step 106ms)  ← elegido
    batch= 64:  268 samp/s  (step 239ms)
    batch=128:  192 samp/s  (step 665ms)
    → Selected: batch_size=32 @ 301 samp/s
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from core.graph import CausalGraph, CausalEdge, CausalNode, CausalRelation, NodeType

if TYPE_CHECKING:
    from .batching import PyGStyleBatcher

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO DEL PROBE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AutoScaleResult:
    """
    Resultado del AutoScaler.

    batch_size:      int    — batch_size elegido (el que maximiza samples/s)
    samples_per_sec: float  — throughput medido en el probe (0.0 si CPU/no medido)
    probe_log:       list   — [(batch_size, samp_s, step_ms)] para cada candidato
                              probado. Vacío en CPU.
    """
    batch_size:      int
    samples_per_sec: float
    probe_log:       List[Tuple[int, float, float]] = field(default_factory=list)

    def print_summary(self) -> None:
        """Imprime el resumen del probe en formato legible."""
        if not self.probe_log:
            print(f"AutoScaler [CPU]: batch_size={self.batch_size}")
            return
        print("AutoScaler probe results:")
        for bs, samp_s, step_ms in self.probe_log:
            marker = " ← selected" if bs == self.batch_size else ""
            print(f"  batch={bs:>4}:  {samp_s:>6.0f} samp/s  (step {step_ms:>5.0f}ms){marker}")
        print(f"Selected: batch_size={self.batch_size}  @ {self.samples_per_sec:.0f} samp/s")


# ─────────────────────────────────────────────────────────────────────────────
# AUTO SCALER
# ─────────────────────────────────────────────────────────────────────────────

class AutoScaler:
    """
    Selecciona el batch_size que maximiza samples/s para el hardware actual.

    No elige simplemente el mayor batch que cabe — hace un timing probe real
    porque batch muy grande puede ser más lento (overhead de Python / CPU↔GPU).

    Uso:
        scaler = AutoScaler()
        result = scaler.find_optimal_batch(engine, sample_graph, device, node_dim=64)
        result.print_summary()
        # → Selected: batch_size=32 @ 301 samp/s
        BATCH = result.batch_size
    """

    # Candidatos a probar — empezamos en 8 (por debajo el overhead de Python domina)
    GPU_CANDIDATES    = [8, 16, 32, 64, 128, 256, 512]
    # Pasos de timing por candidato (GPU sync entre cada uno)
    N_PROBE_STEPS     = 3
    # VRAM mínima libre para considerar el candidato seguro
    MIN_FREE_FRACTION = 0.20

    def find_optimal_batch(
        self,
        model:        nn.Module,
        sample_graph: CausalGraph,
        device:       torch.device,
        node_dim:     Optional[int] = None,
        n_iterations: int = 3,
    ) -> AutoScaleResult:
        """
        Encuentra el batch_size que maximiza samples/s en el hardware actual.

        Pasos:
            1. Para cada candidato: verificar VRAM (descartar si OOM o <20% libre).
            2. Para cada candidato que pasó VRAM: medir N_PROBE_STEPS × step_time.
            3. Calcular samples/s = batch_size / avg_step_time.
            4. Devolver el candidato con mayor samples/s.

        Args:
            model:        CausalReasoningEngine con forward_batched
            sample_graph: grafo de muestra para construir el batch de prueba
            device:       dispositivo objetivo
            node_dim:     dim de node features (infiere de config si es None)
            n_iterations: iteraciones CRE por paso de prueba

        Returns:
            AutoScaleResult con batch_size, samples_per_sec y probe_log
        """
        if device.type == "cpu":
            bs = self._estimate_cpu_optimal()
            return AutoScaleResult(batch_size=bs, samples_per_sec=0.0)

        if not torch.cuda.is_available():
            bs = self._estimate_cpu_optimal()
            return AutoScaleResult(batch_size=bs, samples_per_sec=0.0)

        if node_dim is None:
            node_dim = self._infer_node_dim(model)

        sample_graph = _ensure_indices(sample_graph)

        from .batching import PyGStyleBatcher
        batcher = PyGStyleBatcher()

        n_nodes     = len(sample_graph)
        best_bs     = self.GPU_CANDIDATES[0]
        best_samp_s = 0.0
        probe_log: List[Tuple[int, float, float]] = []

        for batch_size in self.GPU_CANDIDATES:
            # ── 1. VRAM check ─────────────────────────────────────────────────
            try:
                torch.cuda.empty_cache()
                feats   = [torch.randn(n_nodes, node_dim, device=device)
                           for _ in range(batch_size)]
                batched = batcher.batch([sample_graph] * batch_size, feats)

                with torch.no_grad():
                    _ = model.forward_batched(batched, n_iterations=n_iterations)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                free_b, total_b = torch.cuda.mem_get_info(device)
                if free_b / total_b < self.MIN_FREE_FRACTION:
                    logger.debug(
                        f"AutoScaler: batch={batch_size} exceeds VRAM margin "
                        f"(free={free_b/total_b:.1%}), stopping"
                    )
                    break

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.debug(f"AutoScaler: batch={batch_size} → OOM, stopping")
                break
            except Exception as exc:
                logger.warning(f"AutoScaler: batch={batch_size} failed ({exc}), stopping")
                break

            # ── 2. Timing probe ───────────────────────────────────────────────
            step_times: List[float] = []
            try:
                for _ in range(self.N_PROBE_STEPS):
                    feats   = [torch.randn(n_nodes, node_dim, device=device)
                               for _ in range(batch_size)]
                    batched = batcher.batch([sample_graph] * batch_size, feats)

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    with torch.no_grad():
                        _ = model.forward_batched(batched, n_iterations=n_iterations)

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    step_times.append(time.perf_counter() - t0)

            except Exception as exc:
                logger.warning(f"AutoScaler: timing probe batch={batch_size} failed ({exc})")
                continue

            avg_step_s = sum(step_times) / len(step_times)
            samp_s     = batch_size / avg_step_s
            step_ms    = avg_step_s * 1000

            probe_log.append((batch_size, samp_s, step_ms))
            logger.debug(
                f"AutoScaler: batch={batch_size}  {samp_s:.0f} samp/s  "
                f"(step {step_ms:.0f}ms)"
            )

            if samp_s > best_samp_s:
                best_samp_s = samp_s
                best_bs     = batch_size

        return AutoScaleResult(
            batch_size      = best_bs,
            samples_per_sec = best_samp_s,
            probe_log       = probe_log,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _estimate_cpu_optimal(self) -> int:
        """Heurística RAM: 1 batch por 2GB disponibles, máximo 8."""
        try:
            import psutil
            ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            return min(8, max(1, int(ram_gb / 2)))
        except ImportError:
            return 4

    def _infer_node_dim(self, model: nn.Module) -> int:
        """Infiere node_dim del config del modelo."""
        if hasattr(model, "config") and hasattr(model.config, "node_dim"):
            return model.config.node_dim
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                return module.in_features
        return 256


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_indices(graph: CausalGraph) -> CausalGraph:
    """
    Verifica que source_idx y target_idx estén asignados en todas las aristas.
    Si no, reconstruye el grafo para que add_edge los asigne.
    """
    for edge in graph.edges:
        if edge.source_idx == -1 or edge.target_idx == -1:
            new_graph = CausalGraph()
            for node in graph.nodes:
                new_graph.add_node(node)
            for e in graph.edges:
                new_graph.add_edge(CausalEdge(
                    source_id=e.source_id,
                    target_id=e.target_id,
                    relation=e.relation,
                    strength=e.strength,
                    confidence=e.confidence,
                ))
            return new_graph
    return graph
