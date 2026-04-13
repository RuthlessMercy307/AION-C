"""
tests/test_composition_trajectories.py — Parte 22.5 (Trayectorias compuestas).

Cubre:
    - TrajectoryStep / Trajectory validation (motor válido, deps bien formadas)
    - TrajectoryPlanner: casos simples (1 motor), transform-as, compositional
    - El caso emblemático del MEGA-PROMPT:
        "explica este código como cuento" → [forge_c, muse]
    - CompositeOrchestrator: ejecución en orden, prompts construidos con
      prior outputs cuando hay depends_on
    - TrajectoryUnifier: 1 step pasa, N steps usa el último
    - Límite MAX_TRAJECTORY_DEPTH
    - Serialización to_dict (para enviar al frontend por WS)
"""

from __future__ import annotations

import pytest

from composition.trajectories import (
    TrajectoryStep,
    Trajectory,
    TrajectoryPlanner,
    CompositeOrchestrator,
    TrajectoryUnifier,
    MAX_TRAJECTORY_DEPTH,
    VALID_MOTORS,
)


# ════════════════════════════════════════════════════════════════════════════
# Validación de modelo
# ════════════════════════════════════════════════════════════════════════════

class TestTrajectoryModel:
    def test_empty_steps_rejected(self):
        with pytest.raises(ValueError):
            Trajectory(query="q", steps=[])

    def test_unknown_motor_rejected(self):
        with pytest.raises(ValueError):
            Trajectory(
                query="q",
                steps=[TrajectoryStep(motor_name="nope", sub_goal="x")],
            )

    def test_depends_on_forward_rejected(self):
        # Step 0 no puede depender de Step 1 (forward ref)
        with pytest.raises(ValueError):
            Trajectory(
                query="q",
                steps=[
                    TrajectoryStep("cora", "a", depends_on=[1]),
                    TrajectoryStep("muse", "b", depends_on=[]),
                ],
            )

    def test_depends_on_self_rejected(self):
        with pytest.raises(ValueError):
            Trajectory(
                query="q",
                steps=[TrajectoryStep("cora", "a", depends_on=[0])],
            )

    def test_max_depth_enforced(self):
        too_many = [TrajectoryStep("cora", f"g{i}") for i in range(MAX_TRAJECTORY_DEPTH + 1)]
        with pytest.raises(ValueError):
            Trajectory(query="q", steps=too_many)

    def test_motor_sequence(self):
        t = Trajectory(
            query="q",
            steps=[
                TrajectoryStep("forge_c", "a"),
                TrajectoryStep("muse", "b", depends_on=[0]),
            ],
        )
        assert t.motor_sequence() == ["forge_c", "muse"]

    def test_to_dict_shape(self):
        t = Trajectory(
            query="q",
            steps=[TrajectoryStep("cora", "a")],
            rationale="test",
        )
        d = t.to_dict()
        assert d["query"] == "q"
        assert d["rationale"] == "test"
        assert d["motor_sequence"] == ["cora"]
        assert len(d["steps"]) == 1
        assert d["steps"][0]["motor_name"] == "cora"

    def test_valid_motors_covers_5(self):
        assert VALID_MOTORS == {"cora", "forge_c", "muse", "axiom", "empathy"}


# ════════════════════════════════════════════════════════════════════════════
# TrajectoryPlanner
# ════════════════════════════════════════════════════════════════════════════

class TestTrajectoryPlanner:
    def setup_method(self):
        self.planner = TrajectoryPlanner()

    def test_empty_query_rejected(self):
        with pytest.raises(ValueError):
            self.planner.plan("   ")

    def test_code_query_single_forge_c(self):
        t = self.planner.plan("escribe una función python que sume dos números")
        assert t.motor_sequence() == ["forge_c"]

    def test_emotion_query_single_empathy(self):
        t = self.planner.plan("me siento triste hoy")
        assert t.motor_sequence() == ["empathy"]

    def test_math_query_single_axiom(self):
        t = self.planner.plan("calcula el 15% de 240")
        assert t.motor_sequence() == ["axiom"]

    def test_causal_query_single_cora(self):
        t = self.planner.plan("¿por qué llueve en invierno?")
        assert t.motor_sequence() == ["cora"]

    def test_fallback_to_cora(self):
        t = self.planner.plan("hola, ¿qué tal?")
        assert t.motor_sequence() == ["cora"]

    def test_transform_as_code_to_story(self):
        """Caso emblemático del MEGA-PROMPT: FORGE-C → MUSE."""
        t = self.planner.plan("explica este código como cuento")
        assert t.motor_sequence() == ["forge_c", "muse"]
        # El segundo step debe depender del primero
        assert t.steps[1].depends_on == [0]
        # Rationale visible para UI
        assert "transform-as" in t.rationale

    def test_transform_as_poem(self):
        t = self.planner.plan("cuéntame este teorema como poema")
        seq = t.motor_sequence()
        assert seq[0] in ("axiom", "cora")
        assert seq[-1] == "muse"
        assert "poema" in t.steps[-1].sub_goal.lower()

    def test_compositional_code_and_causal(self):
        """Pregunta compuesta: varios dominios + unificador final."""
        t = self.planner.plan(
            "explica qué es python y por qué se usa tanto en ciencia de datos"
        )
        seq = t.motor_sequence()
        # forge_c y cora detectados, cora al final (unifier)
        assert "forge_c" in seq
        assert seq[-1] == "cora"

    def test_compositional_adds_final_cora_unifier(self):
        t = self.planner.plan("explica rust y python, ¿por qué son diferentes?")
        seq = t.motor_sequence()
        assert seq[-1] == "cora"
        # El último step depende de todos los anteriores
        assert t.steps[-1].depends_on == list(range(len(t.steps) - 1))

    def test_max_depth_custom(self):
        p = TrajectoryPlanner(max_depth=2)
        t = p.plan("explica python y javascript y rust y por qué cambian")
        assert len(t.steps) <= 2

    def test_max_depth_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            TrajectoryPlanner(max_depth=0)
        with pytest.raises(ValueError):
            TrajectoryPlanner(max_depth=MAX_TRAJECTORY_DEPTH + 1)


# ════════════════════════════════════════════════════════════════════════════
# CompositeOrchestrator
# ════════════════════════════════════════════════════════════════════════════

def _stub_generate_fn(responses):
    """Devuelve una generate_fn que produce textos deterministas por motor."""
    def fn(motor: str, prompt: str, max_tokens: int) -> str:
        return responses.get(motor, f"<{motor}:{len(prompt)}>")
    return fn


class TestCompositeOrchestrator:
    def test_single_step_execution(self):
        traj = Trajectory(
            query="test",
            steps=[TrajectoryStep("forge_c", "responde: test")],
        )
        gen = _stub_generate_fn({"forge_c": "def f(): pass"})
        orch = CompositeOrchestrator(gen)
        result = orch.execute(traj)
        assert len(result.step_results) == 1
        assert result.step_results[0].output == "def f(): pass"
        assert result.fused_output == "def f(): pass"
        assert result.total_ms >= 0

    def test_sequence_passes_prior_outputs(self):
        traj = Trajectory(
            query="explica este código como cuento",
            steps=[
                TrajectoryStep("forge_c", "analiza el código"),
                TrajectoryStep(
                    "muse", "reescríbelo como cuento",
                    depends_on=[0],
                ),
            ],
        )

        captured_prompts = {}
        def gen(motor, prompt, max_tokens):
            captured_prompts[motor] = prompt
            if motor == "forge_c":
                return "función que suma números"
            return "érase una vez una función que sumaba números..."

        result = CompositeOrchestrator(gen).execute(traj)
        # El prompt de muse debe contener el output de forge_c
        assert "función que suma números" in captured_prompts["muse"]
        # El final output es el del último step (muse)
        assert result.fused_output.startswith("érase una vez")

    def test_no_depends_on_clean_prompt(self):
        traj = Trajectory(
            query="q",
            steps=[TrajectoryStep("cora", "responde: q")],
        )
        captured = {}
        def gen(motor, prompt, max_tokens):
            captured["prompt"] = prompt
            return "answer"
        CompositeOrchestrator(gen).execute(traj)
        assert "[PRIOR_" not in captured["prompt"]
        assert "[QUERY: q]" in captured["prompt"]
        assert "[GOAL: responde: q]" in captured["prompt"]

    def test_multiple_depends_on(self):
        traj = Trajectory(
            query="q",
            steps=[
                TrajectoryStep("forge_c", "a"),
                TrajectoryStep("muse", "b"),
                TrajectoryStep(
                    "cora", "combina",
                    depends_on=[0, 1],
                ),
            ],
        )
        captured = []
        def gen(motor, prompt, max_tokens):
            captured.append(prompt)
            return f"out-of-{motor}"
        CompositeOrchestrator(gen).execute(traj)
        cora_prompt = captured[2]
        assert "out-of-forge_c" in cora_prompt
        assert "out-of-muse" in cora_prompt

    def test_result_to_dict_serializable(self):
        traj = Trajectory(
            query="q",
            steps=[TrajectoryStep("cora", "a")],
        )
        result = CompositeOrchestrator(_stub_generate_fn({"cora": "x"})).execute(traj)
        d = result.to_dict()
        import json
        # Debe serializar sin errores
        json.dumps(d)
        assert d["fused_output"] == "x"


# ════════════════════════════════════════════════════════════════════════════
# TrajectoryUnifier
# ════════════════════════════════════════════════════════════════════════════

class TestTrajectoryUnifier:
    def test_empty_results(self):
        t = Trajectory(query="q", steps=[TrajectoryStep("cora", "a")])
        assert TrajectoryUnifier().fuse(t, []) == ""

    def test_single_result_passes_through(self):
        from composition.trajectories import StepResult
        t = Trajectory(query="q", steps=[TrajectoryStep("cora", "a")])
        results = [StepResult(0, "cora", "a", "prompt", "  solo answer  ", 1.0)]
        assert TrajectoryUnifier().fuse(t, results) == "solo answer"

    def test_multi_result_uses_last(self):
        from composition.trajectories import StepResult
        t = Trajectory(
            query="q",
            steps=[TrajectoryStep("forge_c", "a"), TrajectoryStep("muse", "b", depends_on=[0])],
        )
        results = [
            StepResult(0, "forge_c", "a", "p0", "analysis", 1.0),
            StepResult(1, "muse", "b", "p1", "story version", 1.0),
        ]
        assert TrajectoryUnifier().fuse(t, results) == "story version"


# ════════════════════════════════════════════════════════════════════════════
# End-to-end: el caso emblemático del MEGA-PROMPT
# ════════════════════════════════════════════════════════════════════════════

class TestEmblematic:
    def test_code_as_story_end_to_end(self):
        """Pipeline completo: planner + composite + unifier."""
        planner = TrajectoryPlanner()
        traj = planner.plan("explica este código como cuento")
        assert traj.motor_sequence() == ["forge_c", "muse"]

        calls = []
        def gen(motor, prompt, max_tokens):
            calls.append(motor)
            if motor == "forge_c":
                return "El código define una función recursiva."
            return "Había una vez una función que se llamaba a sí misma..."

        result = CompositeOrchestrator(gen).execute(traj)
        assert calls == ["forge_c", "muse"]
        assert result.fused_output.startswith("Había una vez")
        # Los dos step_results quedaron accesibles para la UI
        assert len(result.step_results) == 2
        assert result.step_results[0].motor_name == "forge_c"
        assert result.step_results[1].motor_name == "muse"
