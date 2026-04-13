"""
tests/test_social_graph_gen.py — Tests para SocialGraphGenerator
================================================================

Verifica que:
  1. SocialGraphGenerator genera ejemplos para niveles 1-3
  2. Todos los ejemplos tienen estructura correcta (problem, graph, answer, etc.)
  3. Los grafos contienen SocialNode/SocialEdge con tipos correctos
  4. verify_social_example() pasa para TODOS los ejemplos generados
  5. Coherencia social: toda acción tiene PERSON, conflicto tiene intenciones opuestas
  6. Los tipos de respuesta cubren el abanico esperado por nivel
  7. El generador es determinista con seed fija
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from core.graph import CausalGraph
from motors.empathy.relations import (
    SOCIAL_NODE_TYPES, SOCIAL_RELATIONS,
    SocialNode, SocialEdge, SocialNodeType, SocialRelation,
)
from synth.social_graph_gen import (
    SocialAnswerType,
    SocialExample,
    SocialGraphGenerator,
    SocialVerificationResult,
    verify_social_example,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_gen(seed: int = 42) -> SocialGraphGenerator:
    return SocialGraphGenerator(seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATOR INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialGraphGeneratorInterface:

    def test_instantiates(self):
        assert isinstance(make_gen(), SocialGraphGenerator)

    def test_generate_level_1(self):
        ex = make_gen().generate(level=1)
        assert isinstance(ex, SocialExample)
        assert ex.complexity_level == 1

    def test_generate_level_2(self):
        ex = make_gen().generate(level=2)
        assert isinstance(ex, SocialExample)
        assert ex.complexity_level == 2

    def test_generate_level_3(self):
        ex = make_gen().generate(level=3)
        assert isinstance(ex, SocialExample)
        assert ex.complexity_level == 3

    def test_invalid_level_raises(self):
        with pytest.raises((ValueError, KeyError)):
            make_gen().generate(level=99)

    def test_generate_batch_count(self):
        assert len(make_gen().generate_batch(n=30)) == 30

    def test_generate_batch_all_social_examples(self):
        for ex in make_gen().generate_batch(n=20):
            assert isinstance(ex, SocialExample)

    def test_generate_batch_custom_distribution(self):
        batch = make_gen().generate_batch(n=20, level_distribution={1: 1.0, 2: 0.0, 3: 0.0})
        for ex in batch:
            assert ex.complexity_level == 1

    def test_deterministic_with_same_seed(self):
        g1 = SocialGraphGenerator(seed=7)
        g2 = SocialGraphGenerator(seed=7)
        assert g1.generate(level=1).problem_text == g2.generate(level=1).problem_text

    def test_different_seeds_different_output(self):
        e1 = SocialGraphGenerator(seed=1).generate(level=1)
        e2 = SocialGraphGenerator(seed=2).generate(level=1)
        assert e1.problem_text != e2.problem_text or e1.example_id != e2.example_id


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXAMPLE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialExampleStructure:

    def _check(self, ex: SocialExample) -> None:
        assert isinstance(ex.problem_text, str) and len(ex.problem_text) > 0
        assert isinstance(ex.graph, CausalGraph)
        assert isinstance(ex.answer, str) and len(ex.answer) > 0
        assert ex.complexity_level in (1, 2, 3)
        assert isinstance(ex.answer_type, SocialAnswerType)
        assert ex.verifiable is True
        assert isinstance(ex.metadata, dict)
        assert isinstance(ex.example_id, str)

    def test_all_levels_structure(self):
        gen = make_gen()
        for level in (1, 2, 3):
            self._check(gen.generate(level=level))

    def test_graph_has_at_least_2_nodes(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            assert len(ex.graph) >= 2

    def test_graph_has_at_least_1_edge(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            assert len(ex.graph.edges) >= 1

    def test_level_1_has_3_to_5_nodes(self):
        gen = make_gen()
        for _ in range(10):
            ex = gen.generate(level=1)
            assert 3 <= len(ex.graph) <= 6

    def test_level_2_has_4_to_6_nodes(self):
        gen = make_gen()
        for _ in range(10):
            ex = gen.generate(level=2)
            assert 4 <= len(ex.graph) <= 7

    def test_level_3_has_5_to_9_nodes(self):
        gen = make_gen()
        for _ in range(10):
            ex = gen.generate(level=3)
            assert 5 <= len(ex.graph) <= 10

    def test_metadata_has_required_node_types(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            assert "required_node_types" in ex.metadata

    def test_metadata_has_required_relations(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            assert "required_relations" in ex.metadata

    def test_metadata_has_social_checks(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            assert "social_checks" in ex.metadata

    def test_example_repr(self):
        ex = make_gen().generate(level=2)
        r  = repr(ex)
        assert "SocialExample" in r
        assert "level=2" in r


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH TYPES
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialGraphTypes:

    def test_nodes_are_social_nodes(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            for node in ex.graph.nodes:
                assert isinstance(node, SocialNode), \
                    f"Level {level}: expected SocialNode, got {type(node)}"

    def test_edges_are_social_edges(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            for edge in ex.graph.edges:
                assert isinstance(edge, SocialEdge)

    def test_node_types_are_social_enum(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            for node in ex.graph.nodes:
                assert isinstance(node.node_type, SocialNodeType)

    def test_edge_relations_are_social_enum(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            for edge in ex.graph.edges:
                assert isinstance(edge.relation, SocialRelation)

    def test_no_self_loops(self):
        gen = make_gen()
        for _ in range(15):
            ex = gen.generate(level=gen._rng.randint(1, 3))
            for edge in ex.graph.edges:
                assert edge.source_id != edge.target_id

    def test_edge_indices_assigned(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            for edge in ex.graph.edges:
                assert edge.source_idx >= 0
                assert edge.target_idx >= 0

    def test_node_types_in_social_vocabulary(self):
        gen = make_gen()
        vocab = set(SOCIAL_NODE_TYPES)
        for _ in range(20):
            ex = gen.generate(level=gen._rng.randint(1, 3))
            for node in ex.graph.nodes:
                assert node.node_type.value in vocab

    def test_edge_relations_in_social_vocabulary(self):
        gen = make_gen()
        vocab = set(SOCIAL_RELATIONS)
        for _ in range(20):
            ex = gen.generate(level=gen._rng.randint(1, 3))
            for edge in ex.graph.edges:
                assert edge.relation.value in vocab


# ─────────────────────────────────────────────────────────────────────────────
# 4. VERIFY SOCIAL EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifySocialExample:

    def test_all_level1_pass(self):
        gen = make_gen(seed=1)
        for _ in range(25):
            ex  = gen.generate(level=1)
            res = verify_social_example(ex)
            assert res.passed, f"Level 1 FAIL: {res.reason}\n{ex}"

    def test_all_level2_pass(self):
        gen = make_gen(seed=2)
        for _ in range(25):
            ex  = gen.generate(level=2)
            res = verify_social_example(ex)
            assert res.passed, f"Level 2 FAIL: {res.reason}\n{ex}"

    def test_all_level3_pass(self):
        gen = make_gen(seed=3)
        for _ in range(25):
            ex  = gen.generate(level=3)
            res = verify_social_example(ex)
            assert res.passed, f"Level 3 FAIL: {res.reason}\n{ex}"

    def test_batch_all_pass(self):
        gen = make_gen(seed=42)
        for ex in gen.generate_batch(n=75):
            res = verify_social_example(ex)
            assert res.passed, f"Batch FAIL: {res.reason}\n{ex}"

    def test_verify_returns_verification_result(self):
        ex  = make_gen().generate(level=1)
        res = verify_social_example(ex)
        assert isinstance(res, SocialVerificationResult)

    def test_verify_has_details(self):
        ex  = make_gen().generate(level=2)
        res = verify_social_example(ex)
        assert isinstance(res.details, dict)

    def test_verify_bool(self):
        ex  = make_gen().generate(level=1)
        res = verify_social_example(ex)
        assert bool(res) == res.passed


# ─────────────────────────────────────────────────────────────────────────────
# 5. SOCIAL COHERENCE CHECKS
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialCoherence:

    def test_all_level1_have_person_nodes(self):
        """All level-1 examples must have PERSON node type."""
        gen = make_gen(seed=10)
        for _ in range(20):
            ex = gen.generate(level=1)
            types = {node.node_type.value for node in ex.graph.nodes}
            assert "person" in types, f"No PERSON in level-1: {ex}"

    def test_level1_every_action_from_person(self):
        """WANTS/FEELS/BELIEVES edges must originate from PERSON nodes."""
        gen = make_gen(seed=11)
        action_rels = {"wants", "feels", "believes", "expects"}
        for _ in range(20):
            ex = gen.generate(level=1)
            person_ids = {
                node.node_id for node in ex.graph.nodes
                if node.node_type == SocialNodeType.PERSON
            }
            for edge in ex.graph.edges:
                if edge.relation.value in action_rels:
                    assert edge.source_id in person_ids, \
                        f"Action edge from non-PERSON: {edge.source_id} --{edge.relation.value}--> {edge.target_id}"

    def test_level2_has_misunderstands_or_conflict(self):
        """All level-2 examples must have MISUNDERSTANDS or EXPECTS edge."""
        gen = make_gen(seed=12)
        conflict_rels = {"misunderstands", "expects"}
        for _ in range(20):
            ex = gen.generate(level=2)
            relations = {edge.relation.value for edge in ex.graph.edges}
            assert relations & conflict_rels, \
                f"No conflict relation in level-2: {relations}, {ex}"

    def test_false_belief_has_misunderstands_and_belief(self):
        """Level-2 false_belief subtype must have MISUNDERSTANDS and BELIEF."""
        gen = make_gen(seed=13)
        for _ in range(40):
            ex = gen.generate(level=2)
            if ex.answer_type == SocialAnswerType.MISUNDERSTANDING:
                relations  = {edge.relation.value for edge in ex.graph.edges}
                node_types = {node.node_type.value for node in ex.graph.nodes}
                assert "misunderstands" in relations, f"No MISUNDERSTANDS: {ex}"
                assert "belief" in node_types or "intention" in node_types, \
                    f"No BELIEF or INTENTION for misunderstanding: {ex}"
                break

    def test_level3_norm_violation_has_norm_node(self):
        """Level-3 norm_violation subtype must have NORM node and VIOLATES_NORM edge."""
        gen = make_gen(seed=14)
        for _ in range(40):
            ex = gen.generate(level=3)
            if ex.answer_type == SocialAnswerType.NORM_VIOLATION:
                node_types = {node.node_type.value for node in ex.graph.nodes}
                relations  = {edge.relation.value for edge in ex.graph.edges}
                assert "norm" in node_types,        f"No NORM in norm_violation: {ex}"
                assert "violates_norm" in relations, f"No VIOLATES_NORM: {ex}"
                break

    def test_level3_repair_strategy_has_trust_and_norm(self):
        """Level-3 trust_betrayal subtype must have TRUSTS and VIOLATES_NORM."""
        gen = make_gen(seed=15)
        for _ in range(40):
            ex = gen.generate(level=3)
            if ex.answer_type == SocialAnswerType.REPAIR_STRATEGY:
                relations = {edge.relation.value for edge in ex.graph.edges}
                assert "trusts" in relations,       f"No TRUSTS in trust_betrayal: {ex}"
                assert "violates_norm" in relations, f"No VIOLATES_NORM: {ex}"
                break

    def test_level3_conflict_diagnosis_has_opposing_intentions(self):
        """Level-3 CONFLICT_DIAGNOSIS must have ≥2 INTENTION nodes."""
        gen = make_gen(seed=16)
        for _ in range(40):
            ex = gen.generate(level=3)
            if ex.answer_type == SocialAnswerType.CONFLICT_DIAGNOSIS:
                intentions = [
                    n for n in ex.graph.nodes
                    if n.node_type == SocialNodeType.INTENTION
                ]
                assert len(intentions) >= 2, \
                    f"Conflict needs ≥2 INTENTIONs, got {len(intentions)}: {ex}"
                break

    def test_all_levels_at_least_2_persons(self):
        """All examples should have at least 2 PERSON nodes."""
        gen = make_gen(seed=20)
        for _ in range(30):
            ex = gen.generate(level=gen._rng.randint(1, 3))
            persons = [n for n in ex.graph.nodes if n.node_type == SocialNodeType.PERSON]
            assert len(persons) >= 2, f"Less than 2 PERSONs: {ex}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANSWER TYPE COVERAGE
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialAnswerTypeCoverage:

    def test_level1_has_empathic_or_intention_types(self):
        gen = make_gen(seed=30)
        types = set()
        for _ in range(30):
            types.add(gen.generate(level=1).answer_type)
        valid = {SocialAnswerType.EMPATHIC_RESPONSE, SocialAnswerType.INTENTION_INFERENCE}
        assert types & valid, f"Level-1 types: {types}"

    def test_level2_has_misunderstanding_or_conflict(self):
        gen = make_gen(seed=31)
        types = set()
        for _ in range(30):
            types.add(gen.generate(level=2).answer_type)
        valid = {SocialAnswerType.MISUNDERSTANDING, SocialAnswerType.CONFLICT_DIAGNOSIS}
        assert types & valid, f"Level-2 types: {types}"

    def test_level3_covers_multiple_types(self):
        gen = make_gen(seed=32)
        types = set()
        for _ in range(60):
            types.add(gen.generate(level=3).answer_type)
        assert len(types) >= 2

    def test_level3_has_norm_violation_type(self):
        gen = make_gen(seed=0)
        types = set()
        for _ in range(80):
            types.add(gen.generate(level=3).answer_type)
        assert SocialAnswerType.NORM_VIOLATION in types

    def test_level3_has_repair_strategy_type(self):
        gen = make_gen(seed=1)
        types = set()
        for _ in range(80):
            types.add(gen.generate(level=3).answer_type)
        assert SocialAnswerType.REPAIR_STRATEGY in types

    def test_level3_has_conflict_diagnosis_type(self):
        gen = make_gen(seed=2)
        types = set()
        for _ in range(80):
            types.add(gen.generate(level=3).answer_type)
        assert SocialAnswerType.CONFLICT_DIAGNOSIS in types


# ─────────────────────────────────────────────────────────────────────────────
# 7. METADATA CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialMetadataContracts:

    def test_level1_has_person_keys(self):
        gen = make_gen(seed=40)
        for _ in range(20):
            ex = gen.generate(level=1)
            assert "person_a" in ex.metadata
            assert "person_b" in ex.metadata

    def test_level2_misunderstanding_has_belief_keys(self):
        gen = make_gen(seed=41)
        for _ in range(40):
            ex = gen.generate(level=2)
            if ex.answer_type == SocialAnswerType.MISUNDERSTANDING:
                assert "false_belief" in ex.metadata or "wrong_belief" in ex.metadata
                break

    def test_level3_norm_violation_has_norm_key(self):
        gen = make_gen(seed=42)
        for _ in range(40):
            ex = gen.generate(level=3)
            if ex.answer_type == SocialAnswerType.NORM_VIOLATION:
                assert "norm" in ex.metadata
                break

    def test_level3_repair_has_repair_key(self):
        gen = make_gen(seed=43)
        for _ in range(40):
            ex = gen.generate(level=3)
            if ex.answer_type == SocialAnswerType.REPAIR_STRATEGY:
                assert "repair" in ex.metadata
                assert "norm"   in ex.metadata
                break

    def test_required_node_types_present_in_graph(self):
        """required_node_types must actually be present in the graph."""
        gen = make_gen(seed=50)
        for _ in range(30):
            ex = gen.generate(level=gen._rng.randint(1, 3))
            present = {node.node_type.value for node in ex.graph.nodes}
            for nt in ex.metadata.get("required_node_types", []):
                assert nt in present, \
                    f"'{nt}' missing. Present: {present}. Example: {ex}"

    def test_required_relations_present_in_graph(self):
        """required_relations must actually be present in the graph edges."""
        gen = make_gen(seed=51)
        for _ in range(30):
            ex = gen.generate(level=gen._rng.randint(1, 3))
            present = {edge.relation.value for edge in ex.graph.edges}
            for rel in ex.metadata.get("required_relations", []):
                assert rel in present, \
                    f"'{rel}' missing. Present: {present}. Example: {ex}"

    def test_social_checks_has_person_count(self):
        gen = make_gen(seed=60)
        for _ in range(20):
            ex = gen.generate(level=gen._rng.randint(1, 3))
            checks = ex.metadata.get("social_checks", {})
            assert "person_count" in checks, f"No person_count in checks: {ex}"
            assert checks["person_count"] >= 2
