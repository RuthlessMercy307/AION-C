"""
composition/ — Trayectorias compuestas (Parte 22.5 del MEGA-PROMPT).

El orchestrator del MoSEPipeline selecciona 1 motor para una query. Este
paquete lo eleva a DIRECTOR DE FLUJO: descompone la query en una secuencia
de motores, ejecuta cada uno con sub-objetivo y contexto previo, y fusiona
los outputs en una salida coherente.

    query → TrajectoryPlanner → Trajectory
          → CompositeOrchestrator.execute(generate_fn)
          → [motor_1 out, motor_2 out, ...]
          → TrajectoryUnifier.fuse()
          → salida final + trace

El MoSEPipeline existente NO se modifica. Este paquete opera a nivel de
agente: cada TrajectoryStep dispara una llamada de generación independiente
al pipeline con un prompt enriquecido por los outputs anteriores.
"""

from composition.trajectories import (
    TrajectoryStep,
    Trajectory,
    TrajectoryPlanner,
    CompositeOrchestrator,
    TrajectoryUnifier,
    StepResult,
    TrajectoryResult,
    MAX_TRAJECTORY_DEPTH,
)

__all__ = [
    "TrajectoryStep",
    "Trajectory",
    "TrajectoryPlanner",
    "CompositeOrchestrator",
    "TrajectoryUnifier",
    "StepResult",
    "TrajectoryResult",
    "MAX_TRAJECTORY_DEPTH",
]
