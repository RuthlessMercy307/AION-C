"""
tests/test_symbolic.py — Tests para Parte 20 del MEGA-PROMPT
==============================================================

Cubre:
  SymbolicGraph              — nodos/aristas, paths, cycles, copy
  AXIOM rules                — transitivity, contradiction, substitution, arithmetic
  FORGE-C rules              — type check, null check, loop detection, dead code
  CORA rules                 — causal transitivity, contradiction, counterfactual
  SymbolicEngine             — apply_all hasta punto fijo, conflict resolution
  build_engine_for_motor     — factory por motor
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from symbolic.graph import SymbolicGraph, SymbolicNode, SymbolicEdge
from symbolic.rules import RuleResult, SymbolicRule
from symbolic.axiom_rules import (
    TransitivityRule, ContradictionRule, SubstitutionRule, ArithmeticRule,
    AXIOM_RULES,
)
from symbolic.forge_c_rules import (
    TypeCheckRule, NullCheckRule, LoopDetectionRule, DeadCodeRule,
    FORGE_C_RULES,
)
from symbolic.cora_rules import (
    CausalTransitivityRule, CausalContradictionRule, CounterfactualRule,
    CORA_RULES,
)
from symbolic.engine import SymbolicEngine, HybridResult, build_engine_for_motor


# ─────────────────────────────────────────────────────────────────────────────
# SymbolicGraph
# ─────────────────────────────────────────────────────────────────────────────


class TestSymbolicGraph:
    def test_add_node_and_find(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode(id="a", label="A"))
        assert g.has_node("a")
        assert g.find_node("a").label == "A"
        assert len(g) == 1

    def test_add_edge_creates_missing_nodes(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge(source="a", target="b", relation="causes"))
        assert g.has_node("a")
        assert g.has_node("b")
        assert g.has_edge("a", "b", "causes")

    def test_add_edge_no_duplicate(self):
        g = SymbolicGraph()
        e = SymbolicEdge(source="a", target="b", relation="causes")
        g.add_edge(e)
        g.add_edge(e)  # mismo
        assert len(g.edges) == 1

    def test_remove_node_removes_edges(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "causes"))
        g.add_edge(SymbolicEdge("b", "c", "causes"))
        assert g.remove_node("b")
        assert not g.has_node("b")
        assert g.edges == []

    def test_edges_from_to(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "implies"))
        g.add_edge(SymbolicEdge("a", "c", "implies"))
        g.add_edge(SymbolicEdge("d", "a", "implies"))
        assert len(g.edges_from("a")) == 2
        assert len(g.edges_to("a")) == 1

    def test_has_path_simple(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "causes"))
        g.add_edge(SymbolicEdge("b", "c", "causes"))
        assert g.has_path("a", "c", relation="causes")
        assert not g.has_path("c", "a")

    def test_has_path_filtered_by_relation(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "causes"))
        g.add_edge(SymbolicEdge("b", "c", "implies"))
        assert g.has_path("a", "b", relation="causes")
        assert not g.has_path("a", "c", relation="causes")

    def test_has_cycle(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "calls"))
        g.add_edge(SymbolicEdge("b", "a", "calls"))
        assert g.has_cycle(relation="calls")

    def test_no_cycle_when_acyclic(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "calls"))
        g.add_edge(SymbolicEdge("b", "c", "calls"))
        assert not g.has_cycle(relation="calls")

    def test_copy_independent(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "causes"))
        g2 = g.copy()
        g2.add_node(SymbolicNode("c"))
        assert not g.has_node("c")

    def test_to_dict(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("a", label="alpha"))
        g.add_edge(SymbolicEdge("a", "b", "causes"))
        d = g.to_dict()
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# AXIOM rules
# ─────────────────────────────────────────────────────────────────────────────


class TestTransitivity:
    def test_simple_chain(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "implies"))
        g.add_edge(SymbolicEdge("b", "c", "implies"))
        result = TransitivityRule().apply(g)
        assert result.modified
        assert g.has_edge("a", "c", "implies")
        assert any(e.props.get("derived_by") == "transitivity" for e in result.added_edges)

    def test_no_op_when_no_chain(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "implies"))
        result = TransitivityRule().apply(g)
        assert not result.modified

    def test_avoids_self_loop(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "implies"))
        g.add_edge(SymbolicEdge("b", "a", "implies"))
        TransitivityRule().apply(g)
        assert not g.has_edge("a", "a", "implies")
        assert not g.has_edge("b", "b", "implies")


class TestContradiction:
    def test_negation_pair(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("p", label="P"))
        g.add_node(SymbolicNode("not_p", label="¬P"))
        result = ContradictionRule().apply(g)
        assert result.modified
        assert any("contradiction" in c for c in result.conflicts)

    def test_implies_and_negates(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "implies"))
        g.add_edge(SymbolicEdge("a", "b", "negates"))
        result = ContradictionRule().apply(g)
        assert result.conflicts

    def test_no_contradiction(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("p", label="P"))
        g.add_node(SymbolicNode("q", label="Q"))
        result = ContradictionRule().apply(g)
        assert not result.conflicts


class TestSubstitution:
    def test_substitutes_in_other_nodes(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("x", label="x"))
        g.add_node(SymbolicNode("five", label="5"))
        g.add_node(SymbolicNode("expr", label="x + 1"))
        g.add_edge(SymbolicEdge("x", "five", "equals"))
        result = SubstitutionRule().apply(g)
        assert result.modified
        assert g.find_node("expr").label == "5 + 1"

    def test_no_substitution_when_no_equals(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("expr", label="x + 1"))
        assert not SubstitutionRule().applies_to(g)


class TestArithmetic:
    def test_simple_addition(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("expr", label="3 + 4"))
        result = ArithmeticRule().apply(g)
        assert result.modified
        # debe haber creado un nodo con label "7"
        result_node = g.find_node("result_expr")
        assert result_node is not None
        assert result_node.label == "7"

    def test_multiplication_decimal(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("expr", label="0.15 × 240"))
        ArithmeticRule().apply(g)
        result_node = g.find_node("result_expr")
        assert result_node is not None
        assert result_node.label == "36"

    def test_division_by_zero_conflict(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("expr", label="5 / 0"))
        result = ArithmeticRule().apply(g)
        assert any("division by zero" in c for c in result.conflicts)

    def test_non_arith_label_ignored(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("text", label="hola"))
        result = ArithmeticRule().apply(g)
        assert not result.modified


# ─────────────────────────────────────────────────────────────────────────────
# FORGE-C rules
# ─────────────────────────────────────────────────────────────────────────────


class TestTypeCheck:
    def test_type_mismatch_conflict(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("v", type="literal", props={"type": "string"}))
        g.add_node(SymbolicNode("p", type="parameter", props={"expected_type": "int"}))
        g.add_edge(SymbolicEdge("v", "p", "passes"))
        result = TypeCheckRule().apply(g)
        assert any("type mismatch" in c for c in result.conflicts)

    def test_matching_types_no_conflict(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("v", type="literal", props={"type": "int"}))
        g.add_node(SymbolicNode("p", type="parameter", props={"expected_type": "int"}))
        g.add_edge(SymbolicEdge("v", "p", "passes"))
        result = TypeCheckRule().apply(g)
        assert not result.conflicts


class TestNullCheck:
    def test_unchecked_use_warns(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("var", type="variable", props={"nullable": True}))
        g.add_node(SymbolicNode("blk", type="block"))
        g.add_edge(SymbolicEdge("blk", "var", "uses"))
        result = NullCheckRule().apply(g)
        assert any("nullable" in n for n in result.notes)

    def test_checked_use_ok(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("var", type="variable", props={"nullable": True}))
        g.add_node(SymbolicNode("blk", type="block"))
        g.add_edge(SymbolicEdge("blk", "var", "checks"))
        g.add_edge(SymbolicEdge("blk", "var", "uses"))
        result = NullCheckRule().apply(g)
        assert not result.notes


class TestLoopDetection:
    def test_detects_recursion_cycle(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("f", "g", "calls"))
        g.add_edge(SymbolicEdge("g", "f", "calls"))
        result = LoopDetectionRule().apply(g)
        assert any("loop" in c for c in result.conflicts)

    def test_no_cycle_no_conflict(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("main", "f", "calls"))
        g.add_edge(SymbolicEdge("f", "g", "calls"))
        result = LoopDetectionRule().apply(g)
        assert not result.conflicts


class TestDeadCode:
    def test_function_without_callers_is_dead(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("orphan", type="function", label="orphan"))
        result = DeadCodeRule().apply(g)
        assert any("dead code" in n for n in result.notes)

    def test_main_not_dead(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("main", type="function", label="main"))
        result = DeadCodeRule().apply(g)
        assert not result.notes

    def test_called_function_not_dead(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("main", type="function", label="main"))
        g.add_node(SymbolicNode("helper", type="function", label="helper"))
        g.add_edge(SymbolicEdge("main", "helper", "calls"))
        result = DeadCodeRule().apply(g)
        assert not result.notes


# ─────────────────────────────────────────────────────────────────────────────
# CORA rules
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalTransitivity:
    def test_chain_a_b_c(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "causes"))
        g.add_edge(SymbolicEdge("b", "c", "causes"))
        result = CausalTransitivityRule().apply(g)
        assert result.modified
        assert g.has_edge("a", "c", "causes")
        new = g.edges_with_relation("causes")
        assert any(e.source == "a" and e.target == "c" for e in new)


class TestCausalContradiction:
    def test_causes_and_prevents_conflict(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "causes"))
        g.add_edge(SymbolicEdge("a", "b", "prevents"))
        result = CausalContradictionRule().apply(g)
        assert any("contradiction" in c for c in result.conflicts)


class TestCounterfactual:
    def test_path_still_exists_after_intervention(self):
        # a → c y a → b → c. Quitar 'b' deja a → c.
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "c", "causes"))
        g.add_edge(SymbolicEdge("a", "b", "causes"))
        g.add_edge(SymbolicEdge("b", "c", "causes"))
        rule = CounterfactualRule()
        assert rule.check(g, intervention="b", target="c") is True

    def test_path_broken_after_intervention(self):
        # Solo a → b → c, sin a → c directo. Quitar 'b' rompe el path.
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "causes"))
        g.add_edge(SymbolicEdge("b", "c", "causes"))
        rule = CounterfactualRule()
        assert rule.check(g, intervention="b", target="c") is False

    def test_intervention_via_node_props(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "causes"))
        g.add_edge(SymbolicEdge("b", "c", "causes"))
        g.add_node(SymbolicNode(
            id="_intervention",
            props={"remove": "b", "target": "c"},
        ))
        result = CounterfactualRule().apply(g)
        assert any("counterfactual" in n for n in result.notes)
        assert any("broken" in n for n in result.notes)


# ─────────────────────────────────────────────────────────────────────────────
# SymbolicEngine — ejecución híbrida
# ─────────────────────────────────────────────────────────────────────────────


class TestSymbolicEngine:
    def test_apply_all_axiom_chain(self):
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "implies"))
        g.add_edge(SymbolicEdge("b", "c", "implies"))
        g.add_edge(SymbolicEdge("c", "d", "implies"))
        engine = SymbolicEngine(rules=AXIOM_RULES)
        result = engine.apply_all(g)
        # Tras aplicar transitividad hasta punto fijo, debe haber a→c, b→d, a→d
        assert g.has_edge("a", "c", "implies")
        assert g.has_edge("b", "d", "implies")
        assert g.has_edge("a", "d", "implies")
        assert "axiom.transitivity" in result.applied_rules

    def test_engine_resolves_causal_conflict(self):
        # Neural propuso A causa B y A previene B → símbolo gana → quita prevents
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "causes"))
        g.add_edge(SymbolicEdge("a", "b", "prevents"))
        engine = SymbolicEngine(rules=CORA_RULES)
        result = engine.apply_all(g)
        assert result.has_conflicts
        # La arista 'prevents' debe haber sido removida
        assert not g.has_edge("a", "b", "prevents")
        # 'causes' debe seguir
        assert g.has_edge("a", "b", "causes")
        assert any(e.relation == "prevents" for e in result.removed_edges)

    def test_engine_resolves_logical_conflict(self):
        # Neural propuso A implies B y A negates B → símbolo gana
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "implies"))
        g.add_edge(SymbolicEdge("a", "b", "negates"))
        engine = SymbolicEngine(rules=AXIOM_RULES)
        result = engine.apply_all(g)
        assert result.has_conflicts
        assert not g.has_edge("a", "b", "negates")
        assert g.has_edge("a", "b", "implies")

    def test_engine_terminates_at_fixpoint(self):
        # Pequeña cadena, debe terminar antes de max_iters
        g = SymbolicGraph()
        g.add_edge(SymbolicEdge("a", "b", "implies"))
        engine = SymbolicEngine(rules=AXIOM_RULES)
        result = engine.apply_all(g, max_iters=10)
        assert result.iterations <= 10

    def test_engine_invalid_max_iters(self):
        engine = SymbolicEngine(rules=AXIOM_RULES)
        with pytest.raises(ValueError):
            engine.apply_all(SymbolicGraph(), max_iters=0)

    def test_engine_arithmetic_extends_graph(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("e", label="0.15 × 240"))
        engine = SymbolicEngine(rules=AXIOM_RULES)
        engine.apply_all(g)
        assert g.has_node("result_e")
        assert g.find_node("result_e").label == "36"

    def test_engine_forge_c_dead_code(self):
        g = SymbolicGraph()
        g.add_node(SymbolicNode("orphan", type="function", label="orphan"))
        engine = SymbolicEngine(rules=FORGE_C_RULES)
        result = engine.apply_all(g)
        assert any("dead code" in n for n in result.notes)


class TestEngineFactory:
    def test_axiom_engine(self):
        e = build_engine_for_motor("axiom")
        assert len(e.rules) == 4

    def test_forge_c_engine(self):
        e = build_engine_for_motor("forge_c")
        assert len(e.rules) == 4

    def test_cora_engine(self):
        e = build_engine_for_motor("cora")
        assert len(e.rules) == 3

    def test_unknown_motor_returns_empty(self):
        e = build_engine_for_motor("unknown")
        assert e.rules == []
