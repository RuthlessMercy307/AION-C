"""
sleep/ — Sleep Cycle ritual de AION-C (Parte 23 del MEGA-PROMPT).

El sistema entra periódicamente en un estado de "sueño" donde no atiende
queries y procesa todo lo aprendido durante la vigilia. Es un ritual
determinista de SEIS preguntas resueltas en orden estricto.

    1. ¿Qué viví desde el último sueño?      (recolección de episodios)
    2. ¿Qué fue útil y qué no?               (reward — Parte 25)
    3. ¿Qué debo olvidar?                    (pruning — Parte 24)
    4. ¿Qué debo comprimir?                  (compresión — Parte 26)
    5. ¿Qué debo consolidar en los pesos?    (auto-learn — Parte 9)
    6. ¿Qué debo preguntarme mañana?         (follow-ups → goals)

Las Partes 24, 25 y 26 son sub-fases de este ritual. Hasta que estén
implementadas, el ciclo usa stubs que mantienen la estructura y dejan la
interfaz lista para inyectarlas.
"""

from sleep.cycle import (
    Episode,
    EpisodicBuffer,
    PhaseResult,
    SleepCycleLog,
    SleepCycle,
    SLEEP_QUESTIONS,
)
from sleep.daemon import SleepDaemon, SleepTrigger

__all__ = [
    "Episode",
    "EpisodicBuffer",
    "PhaseResult",
    "SleepCycleLog",
    "SleepCycle",
    "SLEEP_QUESTIONS",
    "SleepDaemon",
    "SleepTrigger",
]
