"""
composition/trajectories.py — Trayectorias compuestas de motores (Parte 22.5).

Modelo:
    Trajectory = [TrajectoryStep, ...]       secuencia ordenada de motores
    TrajectoryStep = (motor_name, sub_goal, depends_on, max_tokens)

Ejecución:
    Para cada step en orden:
        prompt = build_prompt(query, sub_goal, [outputs previos referenciados])
        out_i  = generate_fn(motor_name, prompt)
    Luego el TrajectoryUnifier fusiona todos los outputs en texto final.

Diseño consciente:
    - generate_fn es INYECTABLE. En producción es una función que corre
      MoSEPipeline con el motor forzado. En tests es un stub determinista.
    - El MoSEPipeline NO se modifica — esto es orquestación a nivel de agente.
    - Límite de profundidad duro (MAX_TRAJECTORY_DEPTH) para evitar bucles.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

# Los 5 motores válidos del MoSE. Cualquier step que referencia algo fuera
# de aquí es rechazado por el planner.
VALID_MOTORS = frozenset({"cora", "forge_c", "muse", "axiom", "empathy"})

# Tope duro de la longitud de una trayectoria para evitar loops.
MAX_TRAJECTORY_DEPTH = 6


# ════════════════════════════════════════════════════════════════════════════
# Modelos
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TrajectoryStep:
    """Un paso de una trayectoria compuesta.

    motor_name:     nombre del motor del MoSE a invocar.
    sub_goal:       objetivo específico para este motor (texto libre, va al prompt).
    depends_on:     índices de steps previos cuyos outputs deben incluirse en el
                    prompt. [] = ver la query cruda sin contexto previo.
    max_tokens:     tope de generación para este step.
    """
    motor_name: str
    sub_goal: str
    depends_on: List[int] = field(default_factory=list)
    max_tokens: int = 80

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    """Plan de ejecución ordenado.

    query:     query original del usuario.
    steps:     lista de TrajectoryStep en orden de ejecución.
    rationale: explicación del planner (para logging/UI/interpretabilidad).
    """
    query: str
    steps: List[TrajectoryStep]
    rationale: str = ""

    def __post_init__(self) -> None:
        if len(self.steps) == 0:
            raise ValueError("Trajectory must have at least one step")
        if len(self.steps) > MAX_TRAJECTORY_DEPTH:
            raise ValueError(
                f"Trajectory depth {len(self.steps)} exceeds "
                f"MAX_TRAJECTORY_DEPTH={MAX_TRAJECTORY_DEPTH}"
            )
        for i, step in enumerate(self.steps):
            if step.motor_name not in VALID_MOTORS:
                raise ValueError(
                    f"Step {i}: unknown motor '{step.motor_name}'. "
                    f"Valid: {sorted(VALID_MOTORS)}"
                )
            for dep in step.depends_on:
                if dep < 0 or dep >= i:
                    raise ValueError(
                        f"Step {i} depends_on {dep}: must be in [0, {i-1}]"
                    )

    def motor_sequence(self) -> List[str]:
        return [s.motor_name for s in self.steps]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "rationale": self.rationale,
            "steps": [s.to_dict() for s in self.steps],
            "motor_sequence": self.motor_sequence(),
        }


@dataclass
class StepResult:
    """Resultado de ejecutar un step."""
    step_index: int
    motor_name: str
    sub_goal: str
    prompt: str
    output: str
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryResult:
    """Resultado completo de una ejecución de trayectoria."""
    trajectory: Trajectory
    step_results: List[StepResult]
    fused_output: str
    total_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory": self.trajectory.to_dict(),
            "steps": [r.to_dict() for r in self.step_results],
            "fused_output": self.fused_output,
            "total_ms": self.total_ms,
        }


# ════════════════════════════════════════════════════════════════════════════
# TrajectoryPlanner — reglas heurísticas
# ════════════════════════════════════════════════════════════════════════════

# Patrones que sugieren cada motor, ordenados por tipo de señal.
_FORGE_C_HINTS = (
    "código", "codigo", "function", "función", "funcion",
    "python", "javascript", "typescript", "rust", "java ", "go ",
    "sql", "bash", "c++", "algoritmo", "class ", "def ", "variable",
    "loop", "bucle", "bug", "error en el código", "compil",
    "program", "debug", "refactor",
)
_MUSE_HINTS = (
    "cuento", "historia", "story", "poema", "poem", "metáfora", "metafora",
    "describe como si", "creativo", "narrat", "imagina", "escena",
    "como si fuera", "as a story", "as if",
)
_EMPATHY_HINTS = (
    "triste", "feliz", "siento", "sentirme", "feel ", "lonely", "solo",
    "frustrad", "ansios", "angust", "deprim", "me duele", "no puedo más",
    "emoción", "emocion",
)
_AXIOM_HINTS = (
    "demuestra", "teorema", "prove", "proof", "%", "ecuación", "ecuacion",
    "calcula", "resuelve", "álgebra", "algebra", "derivad", "integral",
    "límite", "limite", "matem",
)
_CORA_HINTS = (
    "por qué", "porque", "por que", "causa", "efecto", "why", "because",
    "consecuencia", "explica por qué", "provoca", "conduce a", "implica",
    "razón", "razon",
)
_COMPOSITIONAL_HINTS = (
    " y ", " and ", " vs ", " versus ", "diferencia entre", "comparar",
    "compare", "qué es y", "what is and", "explica ... como",
    "explain ... as",
)
_TRANSFORM_AS_HINTS = (
    "como cuento", "como historia", "como poema", "as a story",
    "as a poem", "in the form of", "en forma de",
)


class TrajectoryPlanner:
    """Descompone una query en una secuencia de motores.

    El planner es heurístico (substring-based) en esta fase. Se puede
    sustituir por uno neural después, manteniendo la misma interfaz.
    """

    def __init__(self, max_depth: int = MAX_TRAJECTORY_DEPTH) -> None:
        if max_depth < 1 or max_depth > MAX_TRAJECTORY_DEPTH:
            raise ValueError(
                f"max_depth must be in [1, {MAX_TRAJECTORY_DEPTH}], got {max_depth}"
            )
        self.max_depth = max_depth

    # ── API pública ───────────────────────────────────────────────────────
    def plan(self, query: str) -> Trajectory:
        q = (query or "").strip()
        if not q:
            raise ValueError("query cannot be empty")
        low = q.lower()

        detected = self._detect_motors(low)
        # Señal de "transforma A como B": fuerza secuencia A → B.
        transform_target = self._detect_transform_target(low)

        if transform_target is not None:
            # motor técnico (no-muse) primero, muse al final.
            base = self._primary_technical(detected, low)
            steps = [
                TrajectoryStep(
                    motor_name=base,
                    sub_goal=f"analiza el contenido técnico de: {q}",
                    depends_on=[],
                ),
                TrajectoryStep(
                    motor_name=transform_target,
                    sub_goal=f"reexpresa el análisis previo {self._muse_style(low)}",
                    depends_on=[0],
                ),
            ]
            return Trajectory(
                query=q,
                steps=steps,
                rationale=(
                    f"transform-as detected → {base} → {transform_target}"
                ),
            )

        # Pregunta compuesta con "y" / "and" / comparación
        if self._is_compositional(low) and len(detected) >= 2:
            uniq: List[str] = []
            for m in detected:
                if m not in uniq:
                    uniq.append(m)
            # Si cora no está al final, garantizamos un step unificador de cora
            if uniq[-1] != "cora":
                uniq = uniq[: self.max_depth - 1] + ["cora"]
            else:
                uniq = uniq[: self.max_depth]
            steps: List[TrajectoryStep] = []
            for i, m in enumerate(uniq):
                is_last = i == len(uniq) - 1
                if is_last and len(uniq) > 1:
                    # El último step unifica — depende de todos los anteriores.
                    sub_goal = f"compara/relaciona los outputs previos para: {q}"
                    deps = list(range(i))
                else:
                    sub_goal = f"responde la parte de {m} de: {q}"
                    deps = []
                steps.append(
                    TrajectoryStep(
                        motor_name=m, sub_goal=sub_goal, depends_on=deps
                    )
                )
            return Trajectory(
                query=q,
                steps=steps,
                rationale=f"compositional → {[s.motor_name for s in steps]}",
            )

        # Caso simple: 1 o 2 motores
        if not detected:
            detected = ["cora"]
        uniq: List[str] = []
        for m in detected:
            if m not in uniq:
                uniq.append(m)
        uniq = uniq[: self.max_depth]
        steps = [
            TrajectoryStep(
                motor_name=m,
                sub_goal=f"responde: {q}",
                depends_on=list(range(i)) if i > 0 else [],
            )
            for i, m in enumerate(uniq)
        ]
        return Trajectory(
            query=q,
            steps=steps,
            rationale=f"default → {uniq}",
        )

    # ── Detectores ────────────────────────────────────────────────────────
    def _detect_motors(self, low: str) -> List[str]:
        found: List[str] = []
        if any(h in low for h in _FORGE_C_HINTS):
            found.append("forge_c")
        if any(h in low for h in _MUSE_HINTS):
            found.append("muse")
        if any(h in low for h in _EMPATHY_HINTS):
            found.append("empathy")
        if any(h in low for h in _AXIOM_HINTS):
            found.append("axiom")
        if any(h in low for h in _CORA_HINTS):
            found.append("cora")
        return found

    def _detect_transform_target(self, low: str) -> Optional[str]:
        for hint in _TRANSFORM_AS_HINTS:
            if hint in low:
                # Sólo muse por ahora; "como X" siempre cae en creativo.
                return "muse"
        return None

    def _primary_technical(self, detected: List[str], low: str) -> str:
        """Elige el motor técnico base para un 'transform-as'."""
        for m in detected:
            if m != "muse":
                return m
        # fallback por si sólo se detectó muse
        if any(h in low for h in _FORGE_C_HINTS):
            return "forge_c"
        if any(h in low for h in _AXIOM_HINTS):
            return "axiom"
        return "cora"

    def _muse_style(self, low: str) -> str:
        if "cuento" in low or "story" in low:
            return "como un cuento corto"
        if "poema" in low or "poem" in low:
            return "como un poema"
        if "historia" in low:
            return "como una historia"
        return "de forma narrativa y creativa"

    def _is_compositional(self, low: str) -> bool:
        return any(h in low for h in _COMPOSITIONAL_HINTS)


# ════════════════════════════════════════════════════════════════════════════
# CompositeOrchestrator — ejecutor
# ════════════════════════════════════════════════════════════════════════════

GenerateFn = Callable[[str, str, int], str]
"""Firma: (motor_name, prompt, max_tokens) → generated_text."""


class CompositeOrchestrator:
    """Ejecuta una Trajectory paso a paso con una generate_fn inyectada.

    generate_fn es agnóstica del pipeline concreto: en producción envuelve
    MoSEPipeline forzando el motor activo; en tests devuelve texto
    determinista por (motor, prompt).
    """

    def __init__(self, generate_fn: GenerateFn) -> None:
        self.generate_fn = generate_fn

    def execute(self, trajectory: Trajectory) -> TrajectoryResult:
        t0 = time.perf_counter()
        results: List[StepResult] = []
        for i, step in enumerate(trajectory.steps):
            prior_outputs = [results[d].output for d in step.depends_on]
            prompt = self._build_prompt(
                query=trajectory.query,
                sub_goal=step.sub_goal,
                prior_outputs=prior_outputs,
            )
            s0 = time.perf_counter()
            out = self.generate_fn(step.motor_name, prompt, step.max_tokens)
            elapsed = (time.perf_counter() - s0) * 1000.0
            results.append(
                StepResult(
                    step_index=i,
                    motor_name=step.motor_name,
                    sub_goal=step.sub_goal,
                    prompt=prompt,
                    output=str(out),
                    elapsed_ms=elapsed,
                )
            )

        fused = TrajectoryUnifier().fuse(trajectory, results)
        total_ms = (time.perf_counter() - t0) * 1000.0
        return TrajectoryResult(
            trajectory=trajectory,
            step_results=results,
            fused_output=fused,
            total_ms=total_ms,
        )

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _build_prompt(
        query: str,
        sub_goal: str,
        prior_outputs: Sequence[str],
    ) -> str:
        lines: List[str] = [f"[QUERY: {query}]", f"[GOAL: {sub_goal}]"]
        for i, out in enumerate(prior_outputs):
            lines.append(f"[PRIOR_{i}: {out}]")
        lines.append("[AION:]")
        return " ".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# TrajectoryUnifier — fusión textual
# ════════════════════════════════════════════════════════════════════════════

class TrajectoryUnifier:
    """Fusiona los outputs de los steps en una única respuesta.

    Reglas:
        - 1 step: pasa el output tal cual.
        - N steps: el output del ÚLTIMO step es la respuesta base, a menos
          que un step explícitamente "comparador" (cora al final) exista.
        - Los outputs intermedios se conservan como trace accesible pero
          no se concatenan al texto final (no queremos salida fragmentada).

    Esta es una implementación de primera versión. Una siguiente versión
    puede pasar los outputs por un modelo resumidor o detectar contradicciones.
    """

    def fuse(self, trajectory: Trajectory, results: Sequence[StepResult]) -> str:
        if not results:
            return ""
        if len(results) == 1:
            return results[0].output.strip()
        # El último step es el que "conduce" la respuesta (unifier o muse final).
        return results[-1].output.strip()
