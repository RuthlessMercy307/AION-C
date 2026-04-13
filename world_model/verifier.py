"""
world_model/verifier.py — Verificación + bucle simulate→verify (Parte 19.2)
=============================================================================

Después de simular, ANTES de generar la respuesta:
  1. El scratch pad final se pasa al VAL (ScratchPadVerifier)
  2. VAL verifica coherencia interna del scratch pad
  3. Si hay inconsistencia → CRE re-simula con corrección
  4. Solo cuando el scratch pad es coherente → decoder genera

Este archivo expone:
  VerificationResult     — passed/issues/notes
  ScratchPadVerifier     — chequeos genéricos + por-motor
  SimulationLoop         — loop "simulate→verify→re-simulate" con max_iters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .scratch_pad import ScratchPad, ScratchPadSchema
from .simulator import WorldSimulator


# ─────────────────────────────────────────────────────────────────────────────
# Resultado
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VerificationResult:
    coherent: bool
    issues:   List[str] = field(default_factory=list)
    notes:    List[str] = field(default_factory=list)

    def add_issue(self, msg: str) -> None:
        self.coherent = False
        self.issues.append(msg)

    def add_note(self, msg: str) -> None:
        self.notes.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Verifier
# ─────────────────────────────────────────────────────────────────────────────


class ScratchPadVerifier:
    """
    Chequea coherencia interna de un ScratchPad:
      - Slots requeridos están presentes
      - Tipos coinciden con el schema
      - Listas no son vacías cuando deberían tener contenido
      - Reglas específicas por motor (axiom: contradicción declarada debe
        tener referencia; cora: si hay efectos, debe haber predicción; etc)
    """

    def verify(self, pad: ScratchPad) -> VerificationResult:
        result = VerificationResult(coherent=True)
        if pad.schema is None:
            result.add_issue("scratch pad has no schema")
            return result

        # 1) slots requeridos
        for spec in pad.schema.slots:
            value = pad.get(spec.index)
            if spec.required and value is None:
                result.add_issue(f"missing required slot: {spec.name}")
                continue
            if value is None:
                continue
            # 2) tipos
            if not isinstance(value, spec.expected_type):
                # Acepta int donde se espera float
                if spec.expected_type is float and isinstance(value, int):
                    pass
                else:
                    result.add_issue(
                        f"slot '{spec.name}' has wrong type: "
                        f"expected {spec.expected_type.__name__}, got {type(value).__name__}"
                    )

        # 3) chequeos específicos por motor
        motor = pad.schema.motor
        if motor == "axiom":
            self._verify_axiom(pad, result)
        elif motor == "cora":
            self._verify_cora(pad, result)
        elif motor == "forge_c":
            self._verify_forge_c(pad, result)
        elif motor == "muse":
            self._verify_muse(pad, result)
        elif motor == "empathy":
            self._verify_empathy(pad, result)

        return result

    # ── chequeos por motor ─────────────────────────────────────────────

    def _verify_axiom(self, pad: ScratchPad, result: VerificationResult) -> None:
        proven = pad.get_by_name("proven") or []
        to_prove = pad.get_by_name("to_prove") or []
        contradiction = pad.get_by_name("contradiction") or ""
        if not proven and not to_prove:
            result.add_issue("axiom: both proven and to_prove are empty")
        # Si hay contradicción declarada, debe tener al menos referencia a una hipótesis
        hyps = pad.get_by_name("hypotheses") or []
        if contradiction and not hyps:
            result.add_issue("axiom: contradiction declared without active hypotheses")

    def _verify_cora(self, pad: ScratchPad, result: VerificationResult) -> None:
        causes = pad.get_by_name("causes") or []
        direct = pad.get_by_name("direct_effects") or []
        prediction = pad.get_by_name("prediction") or ""
        if direct and not causes:
            result.add_issue("cora: effects without causes")
        if (causes or direct) and not prediction:
            result.add_issue("cora: missing prediction despite having causes/effects")

    def _verify_forge_c(self, pad: ScratchPad, result: VerificationResult) -> None:
        variables = pad.get_by_name("variables")
        call_stack = pad.get_by_name("call_stack")
        if not isinstance(variables, dict):
            result.add_issue("forge_c: variables must be a dict")
        if not isinstance(call_stack, list):
            result.add_issue("forge_c: call_stack must be a list")

    def _verify_muse(self, pad: ScratchPad, result: VerificationResult) -> None:
        tension = pad.get_by_name("tension")
        if tension is not None and not (0.0 <= float(tension) <= 1.0):
            result.add_issue(f"muse: tension out of range [0,1]: {tension}")
        conflicts = pad.get_by_name("conflicts") or []
        if not conflicts:
            result.add_issue("muse: no conflicts detected (story needs tension)")

    def _verify_empathy(self, pad: ScratchPad, result: VerificationResult) -> None:
        emotion = pad.get_by_name("emotion") or ""
        strategy = pad.get_by_name("response_strategy") or ""
        # Coherencia: tristeza no debe pedir "celebrar"
        if emotion == "tristeza" and "celebr" in strategy.lower():
            result.add_issue("empathy: strategy 'celebrar' incompatible with emotion 'tristeza'")
        if emotion == "alegría" and "validar enojo" in strategy.lower():
            result.add_issue("empathy: strategy 'validar enojo' incompatible with 'alegría'")


# ─────────────────────────────────────────────────────────────────────────────
# SimulationLoop — el ciclo simulate→verify→re-simulate
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SimulationOutcome:
    """Resultado del SimulationLoop."""
    pad:           ScratchPad
    coherent:      bool
    iterations:    int
    last_result:   VerificationResult
    history:       List[VerificationResult] = field(default_factory=list)


CorrectorFn = Callable[[ScratchPad, VerificationResult], ScratchPad]


def default_corrector(pad: ScratchPad, result: VerificationResult) -> ScratchPad:
    """
    Corrector por defecto: para cada issue conocido, aplica un fix mínimo.
    Si no sabe arreglar el issue, devuelve el pad sin cambios.
    """
    if pad.schema is None:
        return pad
    new_pad = pad.copy()
    motor = pad.schema.motor
    for issue in result.issues:
        if "missing required slot" in issue:
            # Rellenar con un placeholder según el tipo esperado
            slot_name = issue.split(":", 1)[1].strip()
            spec = pad.schema.slot_by_name(slot_name)
            if spec is not None and new_pad.get(spec.index) is None:
                placeholder: Any
                if spec.expected_type is list:
                    placeholder = []
                elif spec.expected_type is dict:
                    placeholder = {}
                elif spec.expected_type is str:
                    placeholder = "(unknown)"
                elif spec.expected_type is float:
                    placeholder = 0.0
                elif spec.expected_type is int:
                    placeholder = 0
                elif spec.expected_type is bool:
                    placeholder = False
                else:
                    placeholder = None
                new_pad.set(spec.index, placeholder)
        elif "tension out of range" in issue and motor == "muse":
            cur = new_pad.get_by_name("tension")
            try:
                clamped = max(0.0, min(1.0, float(cur)))
                new_pad.set_by_name("tension", clamped)
            except Exception:
                new_pad.set_by_name("tension", 0.5)
        elif "no conflicts" in issue and motor == "muse":
            new_pad.set_by_name("conflicts", ["interno: sin definir"])
        elif "missing prediction" in issue and motor == "cora":
            causes = new_pad.get_by_name("causes") or []
            effects = new_pad.get_by_name("direct_effects") or []
            if causes and effects:
                new_pad.set_by_name(
                    "prediction",
                    f"sí, {causes[0]} → {effects[0]}",
                )
            else:
                new_pad.set_by_name("prediction", "no se puede determinar")
    return new_pad


class SimulationLoop:
    """
    Ciclo de simulación recursiva con verificación.

    Uso:
        loop = SimulationLoop(simulator=AxiomSimulator(),
                              verifier=ScratchPadVerifier())
        outcome = loop.run("15% de 240")
        if outcome.coherent:
            ... pasar pad al decoder ...
        else:
            ... fallback ...

    Args:
        simulator:    el WorldSimulator del motor activo
        verifier:     el ScratchPadVerifier (default: instancia nueva)
        corrector:    callable(pad, verification_result) → pad corregido
                      (default: corrector heurístico que rellena slots y clamp)
        max_iters:    iteraciones máximas antes de rendirse (default 3)
    """

    def __init__(
        self,
        simulator: WorldSimulator,
        verifier:  Optional[ScratchPadVerifier] = None,
        corrector: Optional[CorrectorFn] = None,
        max_iters: int = 3,
    ) -> None:
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        self.simulator = simulator
        self.verifier = verifier or ScratchPadVerifier()
        self.corrector = corrector or default_corrector
        self.max_iters = max_iters

    def run(self, query: str, initial_pad: Optional[ScratchPad] = None) -> SimulationOutcome:
        pad = self.simulator.simulate(query, initial_pad)
        history: List[VerificationResult] = []
        last: VerificationResult
        for i in range(1, self.max_iters + 1):
            last = self.verifier.verify(pad)
            history.append(last)
            if last.coherent:
                return SimulationOutcome(
                    pad=pad,
                    coherent=True,
                    iterations=i,
                    last_result=last,
                    history=history,
                )
            # corregir + re-verificar (no re-simulate desde cero, solo aplicar fix)
            pad = self.corrector(pad, last)
        # max_iters agotado
        return SimulationOutcome(
            pad=pad,
            coherent=last.coherent,
            iterations=self.max_iters,
            last_result=last,
            history=history,
        )


__all__ = [
    "VerificationResult",
    "ScratchPadVerifier",
    "SimulationOutcome",
    "SimulationLoop",
    "default_corrector",
    "CorrectorFn",
]
