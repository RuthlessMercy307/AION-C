"""
tests/test_math_graph_gen.py — Tests para MathGraphGenerator
=============================================================

Verifica que:
  1. MathGraphGenerator genera ejemplos para niveles 1-3
  2. Todos los ejemplos tienen estructura correcta (problem, graph, answer, etc.)
  3. Los grafos contienen MathNode/MathEdge con tipos correctos
  4. verify_math_example() pasa para TODOS los ejemplos generados
  5. Las respuestas son numéricamente correctas (verificadas con eval())
  6. Los tipos de respuesta cubren el abanico esperado por nivel
  7. El generador es determinista con seed fija
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import math

from core.graph import CausalGraph
from motors.axiom.relations import (
    MATH_NODE_TYPES, MATH_RELATIONS,
    MathNode, MathEdge, MathNodeType, MathRelation,
)
from synth.math_graph_gen import (
    MathAnswerType,
    MathExample,
    MathGraphGenerator,
    MathVerificationResult,
    verify_math_example,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_gen(seed: int = 42) -> MathGraphGenerator:
    return MathGraphGenerator(seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATOR INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class TestMathGraphGeneratorInterface:

    def test_instantiates(self):
        assert isinstance(make_gen(), MathGraphGenerator)

    def test_generate_level_1(self):
        ex = make_gen().generate(level=1)
        assert isinstance(ex, MathExample)
        assert ex.complexity_level == 1

    def test_generate_level_2(self):
        ex = make_gen().generate(level=2)
        assert isinstance(ex, MathExample)
        assert ex.complexity_level == 2

    def test_generate_level_3(self):
        ex = make_gen().generate(level=3)
        assert isinstance(ex, MathExample)
        assert ex.complexity_level == 3

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            make_gen().generate(level=0)
        with pytest.raises(ValueError):
            make_gen().generate(level=4)

    def test_generate_batch_count(self):
        batch = make_gen().generate_batch(n=30)
        assert len(batch) == 30

    def test_generate_batch_all_math_examples(self):
        for ex in make_gen().generate_batch(n=20):
            assert isinstance(ex, MathExample)

    def test_deterministic_with_seed(self):
        ex1 = MathGraphGenerator(seed=99).generate(level=1)
        ex2 = MathGraphGenerator(seed=99).generate(level=1)
        assert ex1.problem_text == ex2.problem_text
        assert ex1.answer == ex2.answer

    def test_level_only_batch(self):
        batch = make_gen().generate_batch(n=20, level_distribution={2: 1.0})
        for ex in batch:
            assert ex.complexity_level == 2


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXAMPLE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class TestMathExampleStructure:

    def _check(self, ex: MathExample) -> None:
        assert isinstance(ex.problem_text, str) and len(ex.problem_text) > 0
        assert isinstance(ex.graph, CausalGraph)
        assert isinstance(ex.answer, str) and len(ex.answer) > 0
        assert ex.complexity_level in (1, 2, 3)
        assert isinstance(ex.answer_type, MathAnswerType)
        assert ex.verifiable is True
        assert isinstance(ex.metadata, dict)
        assert isinstance(ex.example_id, str)

    def test_all_levels_structure(self):
        gen = make_gen()
        for level in (1, 2, 3):
            for _ in range(5):
                self._check(gen.generate(level=level))

    def test_graph_has_at_least_2_nodes(self):
        gen = make_gen()
        for level in (1, 2, 3):
            for _ in range(5):
                ex = gen.generate(level=level)
                assert len(ex.graph) >= 2

    def test_graph_has_at_least_1_edge(self):
        gen = make_gen()
        for level in (1, 2, 3):
            for _ in range(5):
                ex = gen.generate(level=level)
                assert len(ex.graph.edges) >= 1

    def test_numeric_or_bool_answer_present(self):
        gen = make_gen()
        for _ in range(30):
            ex = gen.generate(level=1)
            assert (ex.numeric_answer is not None) or (ex.bool_answer is not None)

    def test_unique_example_ids(self):
        gen  = make_gen()
        ids  = [gen.generate(level=1).example_id for _ in range(20)]
        assert len(set(ids)) >= 15


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH NODE AND EDGE TYPES
# ─────────────────────────────────────────────────────────────────────────────

class TestMathGraphTypes:

    def test_nodes_are_math_nodes(self):
        gen = make_gen()
        for level in (1, 2, 3):
            for _ in range(4):
                ex = gen.generate(level=level)
                for node in ex.graph.nodes:
                    assert isinstance(node, MathNode), \
                        f"Expected MathNode, got {type(node)}"
                    assert node.node_type.value in MATH_NODE_TYPES

    def test_edges_are_math_edges(self):
        gen = make_gen()
        for level in (1, 2, 3):
            for _ in range(4):
                ex = gen.generate(level=level)
                for edge in ex.graph.edges:
                    assert isinstance(edge, MathEdge), \
                        f"Expected MathEdge, got {type(edge)}"
                    assert edge.relation.value in MATH_RELATIONS

    def test_no_self_loops(self):
        gen = make_gen()
        for level in (1, 2, 3):
            for _ in range(5):
                ex = gen.generate(level=level)
                for edge in ex.graph.edges:
                    assert edge.source_id != edge.target_id

    def test_edge_indices_assigned(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            for edge in ex.graph.edges:
                assert edge.source_idx >= 0
                assert edge.target_idx >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. VERIFICATION — ALL EXAMPLES PASS
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyMathExample:

    def test_returns_math_verification_result(self):
        ex  = make_gen().generate(level=1)
        res = verify_math_example(ex)
        assert isinstance(res, MathVerificationResult)

    def test_all_level1_pass(self):
        gen = make_gen(seed=1)
        for _ in range(30):
            ex  = gen.generate(level=1)
            res = verify_math_example(ex)
            assert res.passed, f"Level 1 FAIL: {res.reason}\nExample: {ex}\nAnswer: {ex.answer}"

    def test_all_level2_pass(self):
        gen = make_gen(seed=2)
        for _ in range(30):
            ex  = gen.generate(level=2)
            res = verify_math_example(ex)
            assert res.passed, f"Level 2 FAIL: {res.reason}\nExample: {ex}\nAnswer: {ex.answer}"

    def test_all_level3_pass(self):
        gen = make_gen(seed=3)
        for _ in range(30):
            ex  = gen.generate(level=3)
            res = verify_math_example(ex)
            assert res.passed, f"Level 3 FAIL: {res.reason}\nExample: {ex}\nAnswer: {ex.answer}"

    def test_batch_all_pass(self):
        gen   = make_gen(seed=10)
        batch = gen.generate_batch(n=60, level_distribution={1: 0.4, 2: 0.4, 3: 0.2})
        for ex in batch:
            res = verify_math_example(ex)
            assert res.passed, f"FAIL: {res.reason}\n{ex}"

    def test_verification_result_bool(self):
        ex  = make_gen().generate(level=1)
        res = verify_math_example(ex)
        assert bool(res) == res.passed


# ─────────────────────────────────────────────────────────────────────────────
# 5. NUMERICAL CORRECTNESS — SPOT CHECKS WITH EVAL
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalCorrectness:
    """Verifica manualmente algunos tipos de ejemplo con eval()."""

    def test_arithmetic_eval_correct(self):
        gen = make_gen(seed=42)
        found = 0
        for _ in range(50):
            ex = gen.generate(level=1)
            if ex.answer_type == MathAnswerType.ARITHMETIC:
                expr = ex.metadata.get("expr", ex.metadata.get("verify_expr", ""))
                if expr:
                    expected = ex.metadata.get("expected", ex.numeric_answer)
                    computed = int(eval(expr))
                    assert computed == int(expected), \
                        f"eval({expr!r}) = {computed} ≠ {expected}"
                    found += 1
                    if found >= 5:
                        break

    def test_linear_equation_solution_correct(self):
        gen = make_gen(seed=43)
        found = 0
        for _ in range(50):
            ex = gen.generate(level=1)
            if ex.answer_type == MathAnswerType.LINEAR_EQUATION:
                meta = ex.metadata
                if "b" in meta and "c" in meta:
                    b, c = meta["b"], meta["c"]
                    expected_x = meta["expected_x"]
                    # x + b = c → x = c - b
                    assert (c - b) == expected_x, \
                        f"x + {b} = {c} → x = {c-b} ≠ {expected_x}"
                    found += 1
                    if found >= 5:
                        break

    def test_linear_equation_level2_correct(self):
        gen = make_gen(seed=44)
        found = 0
        for _ in range(50):
            ex = gen.generate(level=2)
            if ex.answer_type == MathAnswerType.LINEAR_EQUATION and "a" in ex.metadata:
                a, b, c = ex.metadata["a"], ex.metadata["b"], ex.metadata["c"]
                expected = ex.metadata["expected_x"]
                computed = (c - b) / a
                assert abs(computed - expected) < 1e-9, \
                    f"({c}-{b})/{a} = {computed} ≠ {expected}"
                found += 1
                if found >= 5:
                    break

    def test_inequality_chain_ordered(self):
        gen = make_gen(seed=45)
        found = 0
        for _ in range(50):
            ex = gen.generate(level=2)
            if ex.answer_type == MathAnswerType.INEQUALITY:
                chain = ex.metadata.get("chain")
                if chain:
                    for i in range(len(chain) - 1):
                        assert chain[i] > chain[i + 1], \
                            f"Chain not strictly decreasing at {i}: {chain}"
                    found += 1
                    if found >= 5:
                        break

    def test_divisibility_check_correct(self):
        gen = make_gen(seed=46)
        found = 0
        for _ in range(50):
            ex = gen.generate(level=1)
            if ex.answer_type == MathAnswerType.DIVISIBILITY:
                n = ex.metadata.get("n", 0)
                k = ex.metadata.get("k", 1)
                expected = ex.metadata.get("expected_divisible")
                if expected is not None:
                    actual = (int(n) % int(k) == 0)
                    assert actual == expected, \
                        f"{n} % {k} = {int(n) % int(k)}, divisible={actual} ≠ {expected}"
                    found += 1
                    if found >= 5:
                        break

    def test_parity_sum_is_even(self):
        gen = make_gen(seed=47)
        found = 0
        for _ in range(50):
            ex = gen.generate(level=3)
            if ex.answer_type == MathAnswerType.PARITY_PROOF:
                ve = ex.metadata.get("verify_expr", "")
                if ve:
                    result = eval(ve)
                    assert result == True, \
                        f"Parity proof failed: eval({ve!r}) = {result}"
                    found += 1
                    if found >= 5:
                        break

    def test_algebraic_identity_lhs_eq_rhs(self):
        gen = make_gen(seed=48)
        found = 0
        for _ in range(50):
            ex = gen.generate(level=3)
            if ex.answer_type == MathAnswerType.ALGEBRAIC_ID:
                lhs = ex.metadata.get("verify_lhs", "")
                rhs = ex.metadata.get("verify_rhs", "")
                if lhs and rhs:
                    assert abs(eval(lhs) - eval(rhs)) < 1e-9, \
                        f"Identity fails: eval({lhs!r}) ≠ eval({rhs!r})"
                    found += 1
                    if found >= 5:
                        break

    def test_sqrt_bound_correct(self):
        gen = make_gen(seed=49)
        found = 0
        for _ in range(50):
            ex = gen.generate(level=3)
            if ex.answer_type == MathAnswerType.BOUND:
                k = ex.metadata.get("k")
                r = ex.metadata.get("r")
                if k is not None and r is not None:
                    assert r * r == k, f"r={r} but r²={r*r} ≠ k={k}"
                    assert ex.numeric_answer == float(r)
                    found += 1
                    if found >= 5:
                        break


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANSWER TYPE COVERAGE
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerTypeCoverage:

    def test_level1_covers_multiple_types(self):
        gen   = make_gen(seed=42)
        types = {gen.generate(level=1).answer_type for _ in range(40)}
        assert len(types) >= 3  # arithmetic, linear_eq, inequality, divisibility

    def test_level2_covers_multiple_types(self):
        gen   = make_gen(seed=43)
        types = {gen.generate(level=2).answer_type for _ in range(40)}
        assert len(types) >= 2

    def test_level3_covers_multiple_types(self):
        gen   = make_gen(seed=44)
        types = {gen.generate(level=3).answer_type for _ in range(40)}
        assert len(types) >= 3  # parity, algebraic_id, bound, linear_system

    def test_level1_has_arithmetic(self):
        gen = make_gen(seed=1)
        found = any(
            gen.generate(level=1).answer_type == MathAnswerType.ARITHMETIC
            for _ in range(40)
        )
        assert found

    def test_level3_has_parity_proof(self):
        gen = make_gen(seed=2)
        found = any(
            gen.generate(level=3).answer_type == MathAnswerType.PARITY_PROOF
            for _ in range(40)
        )
        assert found

    def test_level3_has_algebraic_identity(self):
        gen = make_gen(seed=3)
        found = any(
            gen.generate(level=3).answer_type == MathAnswerType.ALGEBRAIC_ID
            for _ in range(40)
        )
        assert found


# ─────────────────────────────────────────────────────────────────────────────
# 7. METADATA CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMathMetadataContracts:

    def _find(self, gen, level, atype, n=60):
        for _ in range(n):
            ex = gen.generate(level=level)
            if ex.answer_type == atype:
                return ex
        return None

    def test_arithmetic_has_expr_and_expected(self):
        ex = self._find(make_gen(seed=10), 1, MathAnswerType.ARITHMETIC)
        assert ex is not None
        assert "expected" in ex.metadata or ex.numeric_answer is not None

    def test_linear_eq_has_expected_x(self):
        ex = self._find(make_gen(seed=11), 1, MathAnswerType.LINEAR_EQUATION)
        assert ex is not None
        assert "expected_x" in ex.metadata

    def test_inequality_has_expected_result(self):
        ex = self._find(make_gen(seed=12), 1, MathAnswerType.INEQUALITY)
        assert ex is not None
        assert "expected_result" in ex.metadata

    def test_divisibility_has_expected_divisible(self):
        ex = self._find(make_gen(seed=13), 1, MathAnswerType.DIVISIBILITY)
        assert ex is not None
        assert "expected_divisible" in ex.metadata

    def test_parity_proof_has_verify_expr(self):
        ex = self._find(make_gen(seed=14), 3, MathAnswerType.PARITY_PROOF)
        assert ex is not None
        assert "verify_expr" in ex.metadata

    def test_algebraic_id_has_verify_lhs_rhs(self):
        ex = self._find(make_gen(seed=15), 3, MathAnswerType.ALGEBRAIC_ID)
        assert ex is not None
        assert "verify_lhs" in ex.metadata
        assert "verify_rhs" in ex.metadata

    def test_bound_has_expected_x(self):
        ex = self._find(make_gen(seed=16), 3, MathAnswerType.BOUND)
        assert ex is not None
        assert "expected_x" in ex.metadata
        assert "k" in ex.metadata
