"""
world_model/ — World Model interno (Parte 19 del MEGA-PROMPT)
==============================================================

Simulador de consecuencias en el scratch pad. Cada motor usa los 16 slots
de manera distinta para SIMULAR antes de generar la respuesta final.

Componentes:
  ScratchPad           — 16 slots de estado estructurado por motor
  ScratchPadSchema     — esquema semántico de slots por motor
  WorldSimulator       — base ABC para simuladores
  ForgeCSimulator      — estado de ejecución (variables, stack, output)
  AxiomSimulator       — estado de la prueba (paso a paso)
  CoraSimulator        — propagación causal
  MuseSimulator        — estado narrativo (tensión, conflictos)
  EmpathySimulator     — modelo mental del usuario
  ScratchPadVerifier   — verifica coherencia interna del scratch pad
  SimulationLoop       — simulate → verify → re-simulate hasta coherencia
"""

from .scratch_pad import (
    ScratchPad, ScratchPadSchema, SlotSpec,
    SCHEMAS_BY_MOTOR,
    FORGE_C_SCHEMA, AXIOM_SCHEMA, CORA_SCHEMA, MUSE_SCHEMA, EMPATHY_SCHEMA,
)
from .simulator import (
    WorldSimulator,
    ForgeCSimulator, AxiomSimulator, CoraSimulator,
    MuseSimulator, EmpathySimulator,
    build_default_simulators,
)
from .verifier import (
    VerificationResult, ScratchPadVerifier, SimulationLoop,
)

__all__ = [
    "ScratchPad",
    "ScratchPadSchema",
    "SlotSpec",
    "SCHEMAS_BY_MOTOR",
    "FORGE_C_SCHEMA",
    "AXIOM_SCHEMA",
    "CORA_SCHEMA",
    "MUSE_SCHEMA",
    "EMPATHY_SCHEMA",
    "WorldSimulator",
    "ForgeCSimulator",
    "AxiomSimulator",
    "CoraSimulator",
    "MuseSimulator",
    "EmpathySimulator",
    "build_default_simulators",
    "VerificationResult",
    "ScratchPadVerifier",
    "SimulationLoop",
]
