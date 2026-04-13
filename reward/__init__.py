"""
reward/ — Reward probabilístico (Parte 25 del MEGA-PROMPT).

Fórmula:
    reward = α·R_explicit + β·R_implicit + γ·R_intrinsic

    R_explicit: thumbs up/down, correcciones directas del usuario. Alto α.
    R_implicit: continuación sin corrección, "gracias", re-pregunta,
                código copiado, abandono. β moderado.
    R_intrinsic: confianza del modelo (entropía), consistencia simbólica.
                γ bajo (regularizador).

El reward nunca es booleano: es probabilidad en [0, 1] con varianza
estimada. Las decisiones conservadoras usan media − k·std.
"""

from reward.reward import (
    ExplicitSignal,
    ImplicitSignals,
    IntrinsicSignals,
    RewardSignals,
    RewardConfig,
    RewardEstimator,
    RewardEstimate,
    ImplicitDetector,
    RewardLedger,
    sleep_reward_hook,
)

__all__ = [
    "ExplicitSignal",
    "ImplicitSignals",
    "IntrinsicSignals",
    "RewardSignals",
    "RewardConfig",
    "RewardEstimator",
    "RewardEstimate",
    "ImplicitDetector",
    "RewardLedger",
    "sleep_reward_hook",
]
