"""
cre/convergence.py — ConvergenceGate
======================================

Decide cuándo el CRE debe parar de iterar basado en múltiples señales.

POR QUÉ UN GATE Y NO ITERACIONES FIJAS:
    Con iteraciones fijas, el CRE aplica siempre el mismo compute sin importar
    si el grafo ya convergió (desperdicio) o si todavía tiene problemas (falta).

    El ConvergenceGate implementa "parada adaptativa":
        - Query simple  → converge en 1-3 iteraciones (ahorra cómputo)
        - Query compleja → usa 5-10+ iteraciones (invierte donde hace falta)

    Esto replica la intuición de un razonador humano: un problema fácil no
    requiere "dormir sobre ello". Uno difícil puede necesitar repasar varias veces.

CUATRO SEÑALES DE CONVERGENCIA:

    delta_norm (principal):
        ||h_t - h_{t-1}|| / (||h_{t-1}|| + ε)
        Cuando los features dejan de cambiar significativamente, el sistema
        ha convergido a un punto fijo (o está muy cerca).
        → Señal más robusta: directamente observable, sin sesgos de red.

    global_confidence (secundaria):
        Media de confidence scores del WeaknessDetector sobre todos los nodos.
        Cuando los nodos están "seguros de sí mismos" colectivamente, el grafo
        es consistente y el razonamiento está anclado.
        → Señal aprendida: mejora con el entrenamiento del WeaknessDetector.

    weakness_ratio (secundaria):
        n_weak_current / max(1, n_weak_initial)
        ¿Cuántas debilidades quedan relativas al inicio?
        Si resolvimos la mayoría de las debilidades, tiene sentido parar.
        → Señal de progreso: captura si realmente estamos mejorando.

    input_coverage (secundaria):
        ||h_t - h_0|| / (||h_0|| + ε)
        ¿Cuánto se alejaron los features de su inicialización?
        Si el movimiento total es muy pequeño, el grafo no está aprendiendo nada.
        → Señal de "actividad mínima": previene convergencia prematura en primeras iters.

CRITERIO DE PARADA (en orden de prioridad):
    1. NUNCA parar antes de min_iterations (safety floor)
    2. SIEMPRE parar en max_iterations (safety cap)
    3. Parar si delta_norm < delta_threshold   (features estabilizados)
    4. Parar si global_conf > conf_threshold
       AND weakness_ratio < weak_threshold      (grafo resuelto)

CONVERGENCE SCORE:
    Un score suave ∈ [0, 1] que combina las señales para análisis.
    No se usa directamente para la decisión (la decisión usa thresholds),
    pero es útil para logging y para futura adaptación del threshold.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass

import torch

from .weakness import WeaknessReport


# ───────────────────────────────��─────────────────────────────────────────────
# DATACLASSES
# ───────────────────────────────────────────��────────────────────────��────────

@dataclass
class ConvergenceDecision:
    """
    Decisión del ConvergenceGate para una iteración dada.

    Campos:
        should_stop:       True si el CRE debe parar ahora
        reason:            por qué se detuvo (o por qué continuó)
        delta_norm:        ||h_new - h_prev|| / ||h_prev||
        global_confidence: media de confidence scores [0, 1]
        weakness_ratio:    n_weak_current / n_weak_initial [0, 1+]
        coverage_ratio:    ||h_current - h_initial|| / ||h_initial||
        convergence_score: señal suave combinada ∈ [0, 1] (1 = convergido)
    """
    should_stop:        bool
    reason:             str
    delta_norm:         float
    global_confidence:  float
    weakness_ratio:     float
    coverage_ratio:     float
    convergence_score:  float


# Razones de parada / continuación
_REASON_MIN_ITER     = "min_iter_not_reached"
_REASON_DELTA        = "delta_stable"
_REASON_CONFIDENCE   = "high_confidence_low_weakness"
_REASON_CONTINUING   = "continuing"


# ────────────────────────────────────────────────────────────────────────────���
# CONVERGENCE GATE
# ─────────────────────────────────���───────────────────────────��───────────────

class ConvergenceGate:
    """
    Decide cuándo parar el loop de iteraciones del CRE.

    No es un nn.Module porque no tiene parámetros propios: combina señales
    observables con thresholds configurables. Los parámetros entrenables
    (confidence scorer) viven en el WeaknessDetector.

    Uso:
        gate = ConvergenceGate(
            delta_threshold=0.01,
            conf_threshold=0.75,
            weakness_threshold=0.25,
            min_iterations=1,
        )

        # En cada iteración del CRE (pasado iteration >= 1):
        decision = gate.check(
            h_current    = h,         # [N, D] features tras la iteración
            h_prev       = h_prev,    # [N, D] features antes de la iteración
            h_initial    = h_0,       # [N, D] features iniciales (iter=0)
            report       = wr,        # WeaknessReport de esta iteración
            n_weak_init  = n0,        # debilidades en la primera iteración
            iteration    = it,        # 0-indexed (primera iteración = 0)
        )
        if decision.should_stop:
            break
    """

    def __init__(
        self,
        delta_threshold:   float = 0.01,   # ||Δh||/||h|| < esto → features estables
        conf_threshold:    float = 0.75,   # conf media > esto → nodos seguros
        weakness_threshold: float = 0.25,  # ratio < esto → mayoría de debilidades resueltas
        min_iterations:    int   = 1,      # safety floor
        norm_eps:          float = 1e-8,
    ) -> None:
        self.delta_threshold    = delta_threshold
        self.conf_threshold     = conf_threshold
        self.weakness_threshold = weakness_threshold
        self.min_iterations     = min_iterations
        self.norm_eps           = norm_eps

    def check(
        self,
        h_current:   torch.Tensor,      # [N, D] — después de la iteración actual
        h_prev:      torch.Tensor,      # [N, D] — antes de la iteración actual
        h_initial:   torch.Tensor,      # [N, D] — features al inicio del forward()
        report:      WeaknessReport,
        n_weak_init: int,               # n_weaknesses de la primera iteración
        iteration:   int,               # 0-indexed
    ) -> ConvergenceDecision:
        """
        Evalúa si el CRE debe parar en esta iteración.

        Args:
            h_current:   features actuales [N, D]
            h_prev:      features de la iteración anterior [N, D]
            h_initial:   features al inicio del forward [N, D]
            report:      WeaknessReport de la iteración actual
            n_weak_init: número de debilidades en la primera iteración (o 1)
            iteration:   0-indexed (0 = primera iteración completada)

        Returns:
            ConvergenceDecision con la decisión y los valores de las señales
        """
        # ── Señal 1: delta_norm ─────────────────────────────���─────────────────
        with torch.no_grad():
            delta        = (h_current - h_prev).norm()
            prev_norm    = h_prev.norm() + self.norm_eps
            delta_norm   = float((delta / prev_norm).item())

            # ── Señal 2: global_confidence ────────────────────────────────────
            global_conf  = float(report.confidence.mean().item())

            # ── Señal 3: weakness_ratio ───────────────────────────────��───────
            n_weak_now    = max(report.n_weaknesses, 0)
            weakness_ratio = n_weak_now / max(n_weak_init, 1)

            # ── Señal 4: coverage_ratio ────────────────���──────────────────────
            # Qué tan lejos están los features actuales del inicio.
            # Si es muy pequeño en iteraciones tempranas, puede ser que el grafo
            # no está recibiendo suficiente señal — no converger todavía.
            coverage_delta = (h_current - h_initial).norm()
            init_norm      = h_initial.norm() + self.norm_eps
            coverage_ratio = float((coverage_delta / init_norm).item())

        # ── Convergence score: señal suave combinada ───────────────��──────────
        # Alta cuando: delta pequeño, conf alta, pocas debilidades
        # Normaliza delta en [0,1] con una función sigmoidea centrada en threshold
        delta_score = max(0.0, 1.0 - delta_norm / (self.delta_threshold + self.norm_eps))
        conf_score  = global_conf                               # ya en [0,1]
        weak_score  = max(0.0, 1.0 - weakness_ratio)           # 1 = sin debilidades
        convergence_score = (delta_score + conf_score + weak_score) / 3.0

        # ── Safety floor: nunca parar antes de min_iterations ────────────────
        if iteration < self.min_iterations - 1:   # iteration es 0-indexed
            return ConvergenceDecision(
                should_stop       = False,
                reason            = _REASON_MIN_ITER,
                delta_norm        = delta_norm,
                global_confidence = global_conf,
                weakness_ratio    = weakness_ratio,
                coverage_ratio    = coverage_ratio,
                convergence_score = convergence_score,
            )

        # ── Criterio 1: delta estable (señal principal) ───────────────────────
        if delta_norm < self.delta_threshold:
            return ConvergenceDecision(
                should_stop       = True,
                reason            = _REASON_DELTA,
                delta_norm        = delta_norm,
                global_confidence = global_conf,
                weakness_ratio    = weakness_ratio,
                coverage_ratio    = coverage_ratio,
                convergence_score = convergence_score,
            )

        # ── Criterio 2: alta confianza + pocas debilidades ────────────────────
        if (global_conf > self.conf_threshold
                and weakness_ratio < self.weakness_threshold):
            return ConvergenceDecision(
                should_stop       = True,
                reason            = _REASON_CONFIDENCE,
                delta_norm        = delta_norm,
                global_confidence = global_conf,
                weakness_ratio    = weakness_ratio,
                coverage_ratio    = coverage_ratio,
                convergence_score = convergence_score,
            )

        # ── Continuar iterando ──────────────────────────────��─────────────────
        return ConvergenceDecision(
            should_stop       = False,
            reason            = _REASON_CONTINUING,
            delta_norm        = delta_norm,
            global_confidence = global_conf,
            weakness_ratio    = weakness_ratio,
            coverage_ratio    = coverage_ratio,
            convergence_score = convergence_score,
        )
