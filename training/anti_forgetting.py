"""
training/anti_forgetting.py — Anti-catastrophic-forgetting (Parte 9.3)
========================================================================

Las 5 capas de defensa contra forgetting catastrófico, según el plan:

  Capa 1 — Hechos en MEM         (no tocan pesos → 0% forgetting)
                                 → cubierta por memory/semantic_store.py
                                 → MemFactsLayer es un wrapper conceptual.

  Capa 2 — Motores aislados      (entrenar FORGE-C no afecta EMPATHY)
                                 → MotorIsolation: máscara de qué motores
                                   tienen requires_grad=True para una sesión
                                   de entrenamiento dada.

  Capa 3 — Importancia bayesiana (running variance por peso; pesos estables
                                 = protegidos)
                                 → WeightImportanceTracker: mantiene
                                   running mean/var de cada parámetro y
                                   produce una máscara de "importance".

  Capa 4 — Exam antes de consolidar (50 preguntas, rollback si baja >2%)
                                 → ExamRunner + RollbackManager.

  Capa 5 — Replay quirúrgico     (solo ejemplos que dependían de los pesos
                                 que cambiaron, identificados por gradient
                                 magnitude)
                                 → SelectiveReplay: dado un set de pesos
                                   "modificados" (delta_norm), filtra los
                                   ejemplos cuyo gradient toca esos pesos.

Diseño:
  - Todas las capas son INDEPENDIENTES y testeables sin un modelo real.
    Reciben módulos torch.nn.Module o iterables de parámetros.
  - Las máscaras se almacenan como dicts {param_name: tensor} para que las
    capas se puedan combinar (intersección, unión).
  - Sin dependencias adicionales: solo torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 2 — Aislamiento de motores
# ─────────────────────────────────────────────────────────────────────────────


class MotorIsolation:
    """
    Congela todos los motores excepto los que están en `train_motors`.

    Asume que el modelo expone un atributo dict-like `motors` con los motores
    indexados por nombre (cora, forge_c, axiom, muse, empathy). Si el modelo
    no tiene `motors`, no hace nada.

    Uso:
        isolation = MotorIsolation(model, train_motors=["forge_c"])
        with isolation:
            ... train solo forge_c ...
        # al salir del context, restaura requires_grad original
    """

    def __init__(self, model: nn.Module, train_motors: Sequence[str]) -> None:
        self.model = model
        self.train_motors = set(train_motors)
        self._original: Dict[str, bool] = {}

    def _iter_motor_params(self) -> Iterable[Tuple[str, str, nn.Parameter]]:
        motors = getattr(self.model, "motors", None)
        if motors is None:
            return
        # motors puede ser dict, ModuleDict, o lista de tuplas
        if hasattr(motors, "items"):
            iterator = motors.items()
        else:
            iterator = enumerate(motors)
        for name, motor in iterator:
            for pname, p in motor.named_parameters():
                yield str(name), pname, p

    def apply(self) -> int:
        """
        Aplica la isolation: requires_grad=False para motores no en train_motors.
        Devuelve el número de parámetros congelados.
        """
        n_frozen = 0
        for motor_name, pname, p in self._iter_motor_params():
            full = f"{motor_name}.{pname}"
            self._original[full] = p.requires_grad
            if motor_name not in self.train_motors:
                p.requires_grad = False
                n_frozen += 1
        return n_frozen

    def restore(self) -> None:
        """Restaura requires_grad al estado original."""
        for motor_name, pname, p in self._iter_motor_params():
            full = f"{motor_name}.{pname}"
            if full in self._original:
                p.requires_grad = self._original[full]
        self._original.clear()

    def __enter__(self) -> "MotorIsolation":
        self.apply()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 3 — Importancia bayesiana por peso
# ─────────────────────────────────────────────────────────────────────────────


class WeightImportanceTracker:
    """
    Mantiene running mean / variance de cada parámetro a lo largo del
    entrenamiento. Pesos con varianza baja se consideran "estables" y deben
    protegerse de cambios grandes.

    Uso:
        tracker = WeightImportanceTracker(model)
        for step in training:
            ... loss.backward() ...
            tracker.update()  # antes o después de optimizer.step()
            optimizer.step()
        importance = tracker.importance_mask(threshold=0.01)
        # importance[name] = 1.0 si estable, 0.0 si cambiante
        protection = tracker.protection_factor(min_factor=0.1)
        # multiplica gradientes por protection antes de optimizer.step()
    """

    def __init__(self, model: nn.Module, momentum: float = 0.99) -> None:
        if not 0.0 < momentum < 1.0:
            raise ValueError("momentum must be in (0, 1)")
        self.model = model
        self.momentum = momentum
        self._mean: Dict[str, torch.Tensor] = {}
        self._sq:   Dict[str, torch.Tensor] = {}
        self._n_updates: int = 0
        # Inicializar con los valores actuales
        for name, p in model.named_parameters():
            self._mean[name] = p.detach().clone()
            self._sq[name]   = p.detach().clone() ** 2

    def update(self) -> None:
        """Actualiza running mean/var de los parámetros."""
        m = self.momentum
        for name, p in self.model.named_parameters():
            if name not in self._mean:
                continue
            x = p.detach()
            self._mean[name].mul_(m).add_(x, alpha=1 - m)
            self._sq[name].mul_(m).add_(x ** 2, alpha=1 - m)
        self._n_updates += 1

    def variance(self, name: str) -> Optional[torch.Tensor]:
        if name not in self._mean:
            return None
        v = self._sq[name] - self._mean[name] ** 2
        return v.clamp_min(0.0)

    def importance_mask(self, threshold: float = 1e-4) -> Dict[str, torch.Tensor]:
        """
        Devuelve una máscara binaria por parámetro: 1.0 donde varianza < threshold
        (peso estable → protegido), 0.0 donde varianza >= threshold.
        """
        out: Dict[str, torch.Tensor] = {}
        for name in self._mean:
            v = self.variance(name)
            out[name] = (v < threshold).float()
        return out

    def protection_factor(self, threshold: float = 1e-4, min_factor: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Factor multiplicativo para los gradientes:
          1.0 si el peso es libre (varianza alta)
          min_factor si el peso es protegido (varianza baja)
        """
        out: Dict[str, torch.Tensor] = {}
        for name in self._mean:
            v = self.variance(name)
            mask = (v < threshold).float()
            out[name] = 1.0 - mask * (1.0 - min_factor)
        return out

    @property
    def n_updates(self) -> int:
        return self._n_updates


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 4 — Exam + Rollback
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExamItem:
    """Una pregunta del exam."""
    query:    str
    expected: str            # respuesta o substring esperada
    domain:   str = "general"


@dataclass
class ExamResult:
    """Resultado de correr el exam completo."""
    score:        float            # 0.0 - 1.0
    correct:      int
    total:        int
    per_item:     List[Tuple[ExamItem, bool]] = field(default_factory=list)


class ExamRunner:
    """
    Corre un exam de N preguntas y devuelve un score.

    El generator es inyectable: callable(query, item) → str.
    Las respuestas se validan con un matcher (substring por defecto).
    """

    def __init__(
        self,
        items: List[ExamItem],
        generator_fn: Callable[[str, ExamItem], str],
        matcher_fn: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        self.items = list(items)
        self.generator_fn = generator_fn
        self.matcher_fn = matcher_fn or self._default_matcher

    @staticmethod
    def _default_matcher(generated: str, expected: str) -> bool:
        g = (generated or "").lower().strip()
        e = (expected or "").lower().strip()
        if not g or not e:
            return False
        return e in g or g in e

    def run(self) -> ExamResult:
        per_item: List[Tuple[ExamItem, bool]] = []
        correct = 0
        for item in self.items:
            try:
                resp = self.generator_fn(item.query, item)
            except Exception:
                resp = ""
            ok = self.matcher_fn(resp, item.expected)
            per_item.append((item, ok))
            if ok:
                correct += 1
        total = max(1, len(self.items))
        return ExamResult(
            score=correct / total,
            correct=correct,
            total=len(self.items),
            per_item=per_item,
        )


class RollbackManager:
    """
    Snapshot + rollback de state_dict de un modelo.

    Uso:
        rb = RollbackManager(model)
        rb.snapshot()  # antes de entrenar
        ... train ...
        if exam_score_drop > 0.02:
            rb.rollback()  # restaura los pesos del snapshot
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._snapshot: Optional[Dict[str, torch.Tensor]] = None

    def snapshot(self) -> None:
        self._snapshot = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def has_snapshot(self) -> bool:
        return self._snapshot is not None

    def rollback(self) -> None:
        if self._snapshot is None:
            raise RuntimeError("no snapshot to rollback to")
        self.model.load_state_dict(self._snapshot)

    def discard(self) -> None:
        self._snapshot = None


def should_rollback(before_score: float, after_score: float, max_drop: float = 0.02) -> bool:
    """True si la caída del score es mayor que `max_drop`."""
    return (before_score - after_score) > max_drop


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 5 — Replay quirúrgico
# ─────────────────────────────────────────────────────────────────────────────


class SelectiveReplay:
    """
    Selecciona ejemplos de replay basándose en cuáles pesos cambiaron mucho.

    Estrategia simple: dada una "delta map" (delta_norm por parámetro),
    se calcula un score por ejemplo a partir de su gradient registrado
    durante un forward+backward previo. Los ejemplos cuyo gradient tiene
    overlap más alto con los pesos cambiados son los candidatos a replay.

    Uso:
        replay = SelectiveReplay()
        replay.register_example(ex_id, gradients_dict)
        ...
        delta = compute_weight_delta(before_state, after_state)
        candidates = replay.select(delta, top_k=10)
        # train sobre los candidates para "anclar" los conceptos viejos
    """

    def __init__(self) -> None:
        self._examples: Dict[Any, Dict[str, float]] = {}  # id → {param_name: grad_norm}

    def register_example(self, example_id: Any, gradients: Dict[str, torch.Tensor]) -> None:
        """Registra los gradients de un ejemplo (acepta tensors o floats)."""
        norms: Dict[str, float] = {}
        for name, g in gradients.items():
            if isinstance(g, torch.Tensor):
                norms[name] = float(g.detach().norm().item())
            else:
                norms[name] = float(g)
        self._examples[example_id] = norms

    def has_example(self, example_id: Any) -> bool:
        return example_id in self._examples

    def select(self, delta: Dict[str, float], top_k: int = 10) -> List[Any]:
        """
        Devuelve los `top_k` example_ids cuyo gradient histórico tiene
        mayor overlap con los pesos cambiados (delta).
        """
        if not self._examples:
            return []
        scores: List[Tuple[Any, float]] = []
        for ex_id, grad_norms in self._examples.items():
            score = 0.0
            for name, gn in grad_norms.items():
                d = delta.get(name, 0.0)
                score += float(gn) * float(d)
            scores.append((ex_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [ex_id for ex_id, s in scores[:top_k] if s > 0.0]

    def clear(self) -> None:
        self._examples.clear()

    def __len__(self) -> int:
        return len(self._examples)


def compute_weight_delta(
    before: Dict[str, torch.Tensor],
    after:  Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Calcula la norma del cambio por parámetro entre dos state_dicts.
    Devuelve {param_name: float}.
    """
    out: Dict[str, float] = {}
    for name, t_after in after.items():
        if name not in before:
            continue
        diff = (t_after - before[name]).detach()
        out[name] = float(diff.norm().item())
    return out


__all__ = [
    "MotorIsolation",
    "WeightImportanceTracker",
    "ExamItem",
    "ExamResult",
    "ExamRunner",
    "RollbackManager",
    "should_rollback",
    "SelectiveReplay",
    "compute_weight_delta",
]
