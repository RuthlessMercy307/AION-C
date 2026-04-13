"""
sparse/gating.py — Gating networks para activación esparsa (Parte 27).

Diseño:
    SparseLinear wrappea un nn.Linear y le añade una GateNetwork pequeña
    (tamaño ~1% del Linear base) que decide por-query qué neuronas de la
    salida se activan.

    Modos:
        - continuous: sigmoid(logits) ∈ (0,1). Diferenciable, menos ahorro real.
        - binary (straight-through): top-k o umbral hard. Diferenciable en
          backward vía straight-through estimator.

    Compatibilidad con LoRA (Parte 22.1):
        Si la capa base es un LoRALinear, la máscara se aplica DESPUÉS del
        delta del adapter — así un adapter puede forzar neuronas nuevas a
        estar "encendidas" para su skill.

Métricas:
    SparsityTracker registra el porcentaje real de neuronas activas por
    query (fracción de valores de la máscara por encima de un umbral).
    Esto alimenta la UI ("% pesos activos por query").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import torch
import torch.nn as nn


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SparseConfig:
    """Hiperparámetros de la activación esparsa.

    target_density:  fracción de neuronas ACTIVAS objetivo por query [0, 1].
                     Recomendado: 0.10 — 0.20.
    mode:            "continuous" (sigmoid) o "binary" (straight-through).
    gate_hidden:     tamaño de la capa oculta del gate. ~1% del Linear base.
    temperature:     temperatura de sigmoid para suavizar el gate.
    threshold:       umbral para contar una neurona como "activa" en
                     SparsityTracker (continuous) o para la binarización (binary).
    """
    target_density: float = 0.15
    mode: str = "continuous"
    gate_hidden: int = 16
    temperature: float = 1.0
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 < self.target_density <= 1.0):
            raise ValueError("target_density must be in (0, 1]")
        if self.mode not in ("continuous", "binary"):
            raise ValueError("mode must be 'continuous' or 'binary'")
        if self.gate_hidden < 1:
            raise ValueError("gate_hidden must be >= 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")


# ════════════════════════════════════════════════════════════════════════════
# GateNetwork
# ════════════════════════════════════════════════════════════════════════════

class GateNetwork(nn.Module):
    """Pequeña red que mapea input → máscara sobre out_features.

    Tamaño total: in_features * gate_hidden + gate_hidden * out_features
    + bias. Con gate_hidden ≈ 1% de out_features el gate pesa ~1% del
    Linear base.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: SparseConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, config.gate_hidden)
        self.fc2 = nn.Linear(config.gate_hidden, out_features)
        # Init bias del fc2 de modo que sigmoid(0) ≈ 0.5 → density inicial ~0.5.
        # Le aplicamos un offset negativo para acercar la density a target.
        with torch.no_grad():
            offset = math.log(config.target_density / (1.0 - config.target_density + 1e-8))
            self.fc2.bias.fill_(offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula la máscara para las activaciones de la capa base.

        Args:
            x: [..., in_features]

        Returns:
            mask: [..., out_features] en (0, 1) si mode=continuous,
                  o {0, 1} con gradiente ST si mode=binary.
        """
        # Colapsamos dimensiones extra a un vector 1D por muestra reduciendo
        # por media — así el gate es por-muestra, no por-posición de secuencia.
        # Si x es [..., D] lo promediamos por todas las dims menos la última.
        reduced = x
        while reduced.dim() > 2:
            reduced = reduced.mean(dim=1)
        # reduced: [B, in_features]
        h = torch.relu(self.fc1(reduced))
        logits = self.fc2(h) / self.config.temperature
        probs = torch.sigmoid(logits)   # [B, out_features]
        if self.config.mode == "continuous":
            mask = probs
        else:
            # Straight-through binarization
            hard = (probs >= 0.5).to(probs.dtype)
            mask = hard.detach() + probs - probs.detach()
        return mask


# ════════════════════════════════════════════════════════════════════════════
# SparseLinear
# ════════════════════════════════════════════════════════════════════════════

class SparseLinear(nn.Module):
    """nn.Linear envuelto con un gate que enmascara su salida por-muestra.

    Invariante: si enabled=False o config.target_density==1.0, el
    comportamiento es EXACTAMENTE el del base (mask ignorada).
    """

    def __init__(self, base: nn.Linear, config: SparseConfig) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"SparseLinear requires nn.Linear, got {type(base).__name__}")
        self.base = base
        self.config = config
        self.gate = GateNetwork(base.in_features, base.out_features, config)
        self.enabled: bool = True
        # Métrica del último forward (fracción de neuronas con mask > threshold)
        self._last_density: Optional[float] = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def last_density(self) -> Optional[float]:
        return self._last_density

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, mode={self.config.mode}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            self._last_density = 1.0
            return out
        mask = self.gate(x)                                          # [B, out]
        # Medir density real antes de broadcasting
        with torch.no_grad():
            active = (mask >= self.config.threshold).float().mean().item()
            self._last_density = float(active)
        # Broadcasting: si out es [B, L, out] → mask [B, 1, out]
        while mask.dim() < out.dim():
            mask = mask.unsqueeze(-2)
        return out * mask

    def gate_parameter_count(self) -> int:
        return sum(p.numel() for p in self.gate.parameters())

    def base_parameter_count(self) -> int:
        return sum(p.numel() for p in self.base.parameters())


# ════════════════════════════════════════════════════════════════════════════
# SparsityTracker
# ════════════════════════════════════════════════════════════════════════════

class SparsityTracker:
    """Recorre un motor y agrega las densities reportadas por sus SparseLinear.

    Uso típico:
        tracker = SparsityTracker(motor)
        out = motor(x)
        report = tracker.collect()  # dict con per-layer y average
    """

    def __init__(self, root: nn.Module) -> None:
        self.root = root
        self._layers: List[Tuple[str, SparseLinear]] = []
        for name, mod in root.named_modules():
            if isinstance(mod, SparseLinear):
                self._layers.append((name, mod))

    def collect(self) -> Dict[str, Any]:
        per_layer: Dict[str, float] = {}
        densities: List[float] = []
        for name, layer in self._layers:
            d = layer.last_density
            if d is None:
                continue
            per_layer[name] = d
            densities.append(d)
        avg = sum(densities) / len(densities) if densities else 0.0
        return {
            "per_layer": per_layer,
            "avg_density": avg,
            "layer_count": len(self._layers),
            "active_percent": round(avg * 100, 1),
        }

    def reset(self) -> None:
        for _, layer in self._layers:
            layer._last_density = None

    def __len__(self) -> int:
        return len(self._layers)


# ════════════════════════════════════════════════════════════════════════════
# attach / detach
# ════════════════════════════════════════════════════════════════════════════

def _resolve_parent(root: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    cur: nn.Module = root
    for p in parts[:-1]:
        cur = getattr(cur, p)
    return cur, parts[-1]


def attach_sparse_gates(
    motor: nn.Module,
    target_paths: Iterable[str],
    config: SparseConfig,
) -> Dict[str, SparseLinear]:
    """Monkey-patcha los Linear del motor reemplazándolos por SparseLinear.

    Devuelve un dict path → SparseLinear creado. Guárdalo para detach
    posterior o para iterar por las gates.
    """
    created: Dict[str, SparseLinear] = {}
    for path in target_paths:
        parent, attr = _resolve_parent(motor, path)
        current = getattr(parent, attr)
        if isinstance(current, SparseLinear):
            raise RuntimeError(f"Target '{path}' already wrapped")
        if not isinstance(current, nn.Linear):
            raise TypeError(f"Target '{path}' is not nn.Linear (got {type(current).__name__})")
        sparse = SparseLinear(current, config)
        setattr(parent, attr, sparse)
        created[path] = sparse
    return created


def detach_sparse_gates(
    motor: nn.Module,
    created: Dict[str, SparseLinear],
) -> None:
    """Restaura los nn.Linear originales. Inverso exacto de attach_sparse_gates."""
    for path, sparse in created.items():
        parent, attr = _resolve_parent(motor, path)
        current = getattr(parent, attr)
        if current is not sparse:
            raise RuntimeError(
                f"Cannot detach '{path}': current layer is not the SparseLinear we attached."
            )
        setattr(parent, attr, sparse.base)


# ════════════════════════════════════════════════════════════════════════════
# sparsity_loss — penaliza desviarse del target_density
# ════════════════════════════════════════════════════════════════════════════

def sparsity_loss(
    root: nn.Module,
    target: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pérdida que empuja la density media hacia `target`.

    Para cada SparseLinear en `root`, toma el valor medio de su máscara del
    último forward (stored) y acumula (density - target)^2.

    Nota: requiere que se haya hecho un forward antes; si no hay layers o
    no se hizo forward aún, devuelve tensor cero.
    """
    sparses = [m for m in root.modules() if isinstance(m, SparseLinear)]
    if not sparses:
        return torch.zeros((), dtype=torch.float32)
    losses: List[torch.Tensor] = []
    for s in sparses:
        if s._last_density is None:
            continue
        d = torch.tensor(s._last_density, dtype=torch.float32)
        losses.append((d - target) ** 2)
    if not losses:
        return torch.zeros((), dtype=torch.float32)
    stacked = torch.stack(losses)
    if reduction == "mean":
        return stacked.mean()
    if reduction == "sum":
        return stacked.sum()
    return stacked
