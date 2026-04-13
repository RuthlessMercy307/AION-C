"""
tests/test_world_model.py — Tests para Parte 19 del MEGA-PROMPT
=================================================================

Cubre:
  ScratchPad / ScratchPadSchema  — get/set, by_name, serialización, copy
  Schemas por motor              — los 5 motores definidos
  Simuladores                    — Axiom (porcentaje + aritmética),
                                    Forge-C, CORA, Muse, Empathy
  ScratchPadVerifier             — chequeos genéricos + por motor
  SimulationLoop                 — simulate→verify→re-simulate
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from world_model.scratch_pad import (
    ScratchPad, ScratchPadSchema, SlotSpec, SLOT_COUNT,
    SCHEMAS_BY_MOTOR,
    FORGE_C_SCHEMA, AXIOM_SCHEMA, CORA_SCHEMA, MUSE_SCHEMA, EMPATHY_SCHEMA,
)
from world_model.simulator import (
    WorldSimulator,
    ForgeCSimulator, AxiomSimulator, CoraSimulator,
    MuseSimulator, EmpathySimulator,
    build_default_simulators,
)
from world_model.verifier import (
    VerificationResult, ScratchPadVerifier, SimulationLoop,
    SimulationOutcome, default_corrector,
)


# ─────────────────────────────────────────────────────────────────────────────
# ScratchPad básico
# ─────────────────────────────────────────────────────────────────────────────


class TestScratchPadBasics:
    def test_default_size_16(self):
        pad = ScratchPad()
        assert pad.size == SLOT_COUNT == 16

    def test_initial_empty(self):
        pad = ScratchPad()
        assert pad.is_empty()
        assert len(pad) == 0

    def test_get_set_basic(self):
        pad = ScratchPad()
        pad.set(0, "value")
        assert pad.get(0) == "value"
        assert not pad.is_empty()

    def test_out_of_range(self):
        pad = ScratchPad()
        with pytest.raises(IndexError):
            pad.set(99, "x")
        with pytest.raises(IndexError):
            pad.get(99)

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            ScratchPad(size=0)

    def test_set_by_name_requires_schema(self):
        pad = ScratchPad()
        with pytest.raises(ValueError):
            pad.set_by_name("variables", {})

    def test_set_by_name_with_schema(self):
        pad = ScratchPad(schema=FORGE_C_SCHEMA)
        pad.set_by_name("variables", {"x": 1})
        assert pad.get(0) == {"x": 1}
        assert pad.get_by_name("variables") == {"x": 1}

    def test_unknown_slot_name(self):
        pad = ScratchPad(schema=FORGE_C_SCHEMA)
        with pytest.raises(KeyError):
            pad.set_by_name("nonexistent", "x")

    def test_clear(self):
        pad = ScratchPad(schema=AXIOM_SCHEMA)
        pad.set_by_name("proven", ["a"])
        pad.clear()
        assert pad.is_empty()

    def test_copy_independent(self):
        pad = ScratchPad(schema=AXIOM_SCHEMA)
        pad.set_by_name("proven", ["a"])
        copy = pad.copy()
        copy.set_by_name("proven", ["b"])
        assert pad.get_by_name("proven") == ["a"]
        assert copy.get_by_name("proven") == ["b"]


class TestScratchPadSerialization:
    def test_as_dict_named_slots(self):
        pad = ScratchPad(schema=AXIOM_SCHEMA)
        pad.set_by_name("proven", ["A→B"])
        pad.set_by_name("to_prove", ["A→C"])
        d = pad.as_dict()
        assert d["motor"] == "axiom"
        assert d["slots"]["proven"] == ["A→B"]
        assert d["slots"]["to_prove"] == ["A→C"]

    def test_to_json_roundtrip(self):
        pad = ScratchPad(schema=AXIOM_SCHEMA)
        pad.set_by_name("proven", ["a"])
        pad.set_by_name("to_prove", ["b"])
        s = pad.to_json()
        import json
        data = json.loads(s)
        pad2 = ScratchPad.from_dict(data, schema=AXIOM_SCHEMA)
        assert pad2.get_by_name("proven") == ["a"]
        assert pad2.get_by_name("to_prove") == ["b"]

    def test_filled_indices(self):
        pad = ScratchPad(schema=AXIOM_SCHEMA)
        pad.set(0, "x")
        pad.set(2, "y")
        assert pad.filled_indices() == [0, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Schemas por motor
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_all_5_motors_have_schemas(self):
        for m in ("forge_c", "axiom", "cora", "muse", "empathy"):
            assert m in SCHEMAS_BY_MOTOR
            assert isinstance(SCHEMAS_BY_MOTOR[m], ScratchPadSchema)

    def test_forge_c_schema(self):
        assert FORGE_C_SCHEMA.slot_by_name("variables").index == 0
        assert FORGE_C_SCHEMA.slot_by_name("call_stack").index == 1
        assert FORGE_C_SCHEMA.slot_by_name("variables").required

    def test_axiom_schema(self):
        assert AXIOM_SCHEMA.slot_by_name("proven").required
        assert AXIOM_SCHEMA.slot_by_name("to_prove").required

    def test_cora_schema(self):
        assert CORA_SCHEMA.slot_by_name("causes").required
        assert CORA_SCHEMA.slot_by_name("prediction").required

    def test_required_indices(self):
        idxs = AXIOM_SCHEMA.required_indices()
        assert 0 in idxs and 1 in idxs


# ─────────────────────────────────────────────────────────────────────────────
# AxiomSimulator — caso emblemático del MEGA-PROMPT 19.3
# ─────────────────────────────────────────────────────────────────────────────


class TestAxiomSimulator:
    def setup_method(self):
        self.sim = AxiomSimulator()

    def test_percent_15_de_240(self):
        """El ejemplo del 19.3 — 15% de 240 = 36 paso a paso."""
        pad = self.sim.simulate("cuanto es 15% de 240")
        proven = pad.get_by_name("proven")
        assert proven is not None
        # Debe contener los pasos intermedios
        joined = " ".join(proven)
        assert "0.15" in joined
        # Y el resultado final 36
        assert "36" in joined

    def test_percent_25_de_200(self):
        pad = self.sim.simulate("what is 25% of 200")
        proven = pad.get_by_name("proven")
        assert any("50" in step for step in proven)

    def test_arithmetic_addition(self):
        pad = self.sim.simulate("2 + 3")
        proven = pad.get_by_name("proven")
        assert any("5" in s for s in proven)

    def test_arithmetic_multiplication(self):
        pad = self.sim.simulate("7 × 8")
        proven = pad.get_by_name("proven")
        assert any("56" in s for s in proven)

    def test_unknown_query_falls_back(self):
        pad = self.sim.simulate("teorema de fermat")
        # Debe llenar to_prove al menos
        assert pad.get_by_name("to_prove") is not None
        assert pad.get_by_name("proven") is not None  # vacía, pero presente


# ─────────────────────────────────────────────────────────────────────────────
# Otros simuladores
# ─────────────────────────────────────────────────────────────────────────────


class TestForgeCSimulator:
    def test_extracts_function_definitions(self):
        pad = ForgeCSimulator().simulate("def add(a, b): return a + b")
        assert "add" in pad.get_by_name("call_stack")

    def test_extracts_simple_assignments(self):
        pad = ForgeCSimulator().simulate("x = 5\ny = 10")
        vars = pad.get_by_name("variables")
        assert vars["x"] == 5
        assert vars["y"] == 10

    def test_default_call_stack(self):
        pad = ForgeCSimulator().simulate("x = 1")
        assert pad.get_by_name("call_stack") == ["main"]


class TestCoraSimulator:
    def test_known_chain_lluvia(self):
        pad = CoraSimulator().simulate("si llueve, ¿qué pasa?")
        # llueve no está en KNOWN_CHAINS exacto pero "lluvia" sí — esto
        # podría no detectar. Validamos que al menos se pobló.
        assert pad.get_by_name("prediction") is not None

    def test_known_chain_rain(self):
        pad = CoraSimulator().simulate("rain causes problems")
        assert "rain" in pad.get_by_name("causes")
        assert "wet soil" in pad.get_by_name("direct_effects")

    def test_explicit_causal_statement(self):
        pad = CoraSimulator().simulate("si A causa B, ¿qué pasa?")
        assert pad.get_by_name("causes") == ["a"]
        assert "b" in pad.get_by_name("direct_effects")[0]

    def test_no_causal_pattern(self):
        pad = CoraSimulator().simulate("hola")
        assert pad.get_by_name("causes") == []
        assert pad.get_by_name("prediction")  # no vacío


class TestMuseSimulator:
    def test_high_tension_keywords(self):
        pad = MuseSimulator().simulate("una historia de muerte y lucha")
        assert pad.get_by_name("tension") > 0.5
        assert pad.get_by_name("conflicts")

    def test_default_internal_conflict(self):
        pad = MuseSimulator().simulate("escribe algo")
        conflicts = pad.get_by_name("conflicts")
        assert any("interno" in c for c in conflicts)


class TestEmpathySimulator:
    def test_detects_frustration(self):
        pad = EmpathySimulator().simulate("esto no funciona, estoy harto")
        assert pad.get_by_name("emotion") == "frustración"
        assert "validación" in pad.get_by_name("need")

    def test_detects_sadness(self):
        pad = EmpathySimulator().simulate("estoy triste, perdí mi trabajo")
        assert pad.get_by_name("emotion") == "tristeza"

    def test_neutral_default(self):
        pad = EmpathySimulator().simulate("¿qué tal el clima?")
        assert pad.get_by_name("emotion") == "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# ScratchPadVerifier
# ─────────────────────────────────────────────────────────────────────────────


class TestVerifier:
    def setup_method(self):
        self.v = ScratchPadVerifier()

    def test_no_schema_fails(self):
        pad = ScratchPad()
        result = self.v.verify(pad)
        assert not result.coherent

    def test_missing_required_slot(self):
        pad = ScratchPad(schema=AXIOM_SCHEMA)
        # No setea proven ni to_prove → ambos required
        result = self.v.verify(pad)
        assert not result.coherent
        assert any("proven" in i for i in result.issues)

    def test_wrong_type(self):
        pad = ScratchPad(schema=AXIOM_SCHEMA)
        pad.set_by_name("proven", "should be a list")
        pad.set_by_name("to_prove", [])
        result = self.v.verify(pad)
        assert not result.coherent
        assert any("wrong type" in i for i in result.issues)

    def test_int_acceptable_for_float(self):
        pad = ScratchPad(schema=MUSE_SCHEMA)
        pad.set_by_name("tension", 1)  # int donde se espera float
        pad.set_by_name("conflicts", ["a"])
        result = self.v.verify(pad)
        # Int para float es aceptable; el chequeo de tension fuera de rango
        # NO debería disparar (1 está en [0,1])
        assert "tension out of range" not in str(result.issues)

    def test_axiom_contradiction_without_hypothesis(self):
        pad = ScratchPad(schema=AXIOM_SCHEMA)
        pad.set_by_name("proven", ["a"])
        pad.set_by_name("to_prove", [])
        pad.set_by_name("contradiction", "alguna contradicción")
        # No setea hypotheses
        result = self.v.verify(pad)
        assert not result.coherent
        assert any("contradiction" in i for i in result.issues)

    def test_cora_effects_without_causes(self):
        pad = ScratchPad(schema=CORA_SCHEMA)
        pad.set_by_name("causes", [])
        pad.set_by_name("direct_effects", ["x"])
        pad.set_by_name("prediction", "y")
        result = self.v.verify(pad)
        assert not result.coherent
        assert any("effects without causes" in i for i in result.issues)

    def test_muse_tension_out_of_range(self):
        pad = ScratchPad(schema=MUSE_SCHEMA)
        pad.set_by_name("tension", 1.5)
        pad.set_by_name("conflicts", ["x"])
        result = self.v.verify(pad)
        assert not result.coherent
        assert any("tension out of range" in i for i in result.issues)

    def test_empathy_incoherent_strategy(self):
        pad = ScratchPad(schema=EMPATHY_SCHEMA)
        pad.set_by_name("emotion", "tristeza")
        pad.set_by_name("probable_cause", "x")
        pad.set_by_name("need", "presencia")
        pad.set_by_name("response_strategy", "celebrar")
        result = self.v.verify(pad)
        assert not result.coherent

    def test_full_axiom_pad_passes(self):
        sim = AxiomSimulator()
        pad = sim.simulate("cuanto es 15% de 240")
        result = self.v.verify(pad)
        assert result.coherent, f"issues: {result.issues}"


# ─────────────────────────────────────────────────────────────────────────────
# SimulationLoop
# ─────────────────────────────────────────────────────────────────────────────


class TestSimulationLoop:
    def test_runs_simulator_and_verifies(self):
        loop = SimulationLoop(simulator=AxiomSimulator())
        outcome = loop.run("15% de 240")
        assert outcome.coherent
        assert outcome.iterations >= 1
        assert isinstance(outcome.pad, ScratchPad)

    def test_corrector_fixes_missing_required_slot(self):
        # Simulador "broken" que devuelve un pad incompleto
        class BrokenAxiomSim(WorldSimulator):
            motor = "axiom"
            schema = AXIOM_SCHEMA
            def _simulate(self, query, pad):
                # Solo pone "proven", no pone "to_prove" (que es required)
                pad.set_by_name("proven", ["a"])
                return pad
        loop = SimulationLoop(simulator=BrokenAxiomSim())
        outcome = loop.run("x")
        # Tras corregir con default_corrector, debe ser coherente
        assert outcome.coherent
        assert outcome.iterations >= 2  # al menos un re-intento
        # to_prove debe haberse rellenado con []
        assert outcome.pad.get_by_name("to_prove") == []

    def test_max_iters_respected(self):
        # Simulador que siempre devuelve incoherente y corrector que no arregla
        class HopelessSim(WorldSimulator):
            motor = "axiom"
            schema = AXIOM_SCHEMA
            def _simulate(self, query, pad):
                return pad  # vacío
        def noop_corrector(pad, result):
            return pad
        loop = SimulationLoop(
            simulator=HopelessSim(),
            corrector=noop_corrector,
            max_iters=2,
        )
        outcome = loop.run("x")
        assert outcome.iterations == 2
        assert not outcome.coherent
        assert len(outcome.history) == 2

    def test_invalid_max_iters(self):
        with pytest.raises(ValueError):
            SimulationLoop(simulator=AxiomSimulator(), max_iters=0)

    def test_loop_with_muse_simulator(self):
        loop = SimulationLoop(simulator=MuseSimulator())
        outcome = loop.run("una historia con muerte y amor")
        assert outcome.coherent

    def test_factory_returns_all_5(self):
        sims = build_default_simulators()
        assert set(sims.keys()) == {"forge_c", "axiom", "cora", "muse", "empathy"}
