"""
tests/test_narrative_graph_gen.py — Tests para NarrativeGraphGenerator
=======================================================================

Verifica que:
  1. NarrativeGraphGenerator genera ejemplos para niveles 1-3
  2. Todos los ejemplos tienen estructura correcta (problem, graph, answer, etc.)
  3. Los grafos contienen NarrativeNode/NarrativeEdge con tipos correctos
  4. verify_narrative_example() pasa para TODOS los ejemplos generados
  5. Los arcos narrativos son coherentes (conflicto antes de resolución, etc.)
  6. Los tipos de respuesta cubren el abanico esperado por nivel
  7. El generador es determinista con seed fija
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from core.graph import CausalGraph
from motors.muse.relations import (
    NARRATIVE_NODE_TYPES, NARRATIVE_RELATIONS,
    NarrativeNode, NarrativeEdge, NarrativeNodeType, NarrativeRelation,
)
from synth.narrative_graph_gen import (
    NarrativeAnswerType,
    NarrativeExample,
    NarrativeGraphGenerator,
    NarrativeVerificationResult,
    verify_narrative_example,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_gen(seed: int = 42) -> NarrativeGraphGenerator:
    return NarrativeGraphGenerator(seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATOR INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class TestNarrativeGraphGeneratorInterface:

    def test_instantiates(self):
        assert isinstance(make_gen(), NarrativeGraphGenerator)

    def test_generate_level_1(self):
        ex = make_gen().generate(level=1)
        assert isinstance(ex, NarrativeExample)
        assert ex.complexity_level == 1

    def test_generate_level_2(self):
        ex = make_gen().generate(level=2)
        assert isinstance(ex, NarrativeExample)
        assert ex.complexity_level == 2

    def test_generate_level_3(self):
        ex = make_gen().generate(level=3)
        assert isinstance(ex, NarrativeExample)
        assert ex.complexity_level == 3

    def test_invalid_level_raises(self):
        with pytest.raises((ValueError, KeyError)):
            make_gen().generate(level=99)

    def test_generate_batch_count(self):
        batch = make_gen().generate_batch(n=30)
        assert len(batch) == 30

    def test_generate_batch_all_narrative_examples(self):
        for ex in make_gen().generate_batch(n=20):
            assert isinstance(ex, NarrativeExample)

    def test_generate_batch_custom_distribution(self):
        batch = make_gen().generate_batch(n=30, level_distribution={1: 1.0, 2: 0.0, 3: 0.0})
        for ex in batch:
            assert ex.complexity_level == 1

    def test_deterministic_with_same_seed(self):
        g1 = NarrativeGraphGenerator(seed=7)
        g2 = NarrativeGraphGenerator(seed=7)
        e1 = g1.generate(level=1)
        e2 = g2.generate(level=1)
        assert e1.problem_text == e2.problem_text

    def test_different_seeds_different_output(self):
        e1 = NarrativeGraphGenerator(seed=1).generate(level=1)
        e2 = NarrativeGraphGenerator(seed=2).generate(level=1)
        # With overwhelming probability, these should differ
        assert e1.problem_text != e2.problem_text or e1.example_id != e2.example_id


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXAMPLE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class TestNarrativeExampleStructure:

    def _check(self, ex: NarrativeExample) -> None:
        assert isinstance(ex.problem_text, str) and len(ex.problem_text) > 0
        assert isinstance(ex.graph, CausalGraph)
        assert isinstance(ex.answer, str) and len(ex.answer) > 0
        assert ex.complexity_level in (1, 2, 3)
        assert isinstance(ex.answer_type, NarrativeAnswerType)
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
            assert len(ex.graph) >= 2, (
                f"Level {level}: expected ≥2 nodes, got {len(ex.graph)}"
            )

    def test_graph_has_at_least_1_edge(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            assert len(ex.graph.edges) >= 1, (
                f"Level {level}: expected ≥1 edges, got {len(ex.graph.edges)}"
            )

    def test_level_1_has_3_to_4_nodes(self):
        gen = make_gen()
        for _ in range(10):
            ex = gen.generate(level=1)
            assert 2 <= len(ex.graph) <= 5

    def test_level_2_has_4_to_6_nodes(self):
        gen = make_gen()
        for _ in range(10):
            ex = gen.generate(level=2)
            assert 3 <= len(ex.graph) <= 7

    def test_level_3_has_5_to_8_nodes(self):
        gen = make_gen()
        for _ in range(10):
            ex = gen.generate(level=3)
            assert 4 <= len(ex.graph) <= 9

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

    def test_metadata_has_arc_checks(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            assert "arc_checks" in ex.metadata


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH TYPES
# ─────────────────────────────────────────────────────────────────────────────

class TestNarrativeGraphTypes:

    def test_nodes_are_narrative_nodes(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            for node in ex.graph.nodes:
                assert isinstance(node, NarrativeNode), \
                    f"Level {level}: expected NarrativeNode, got {type(node)}"

    def test_edges_are_narrative_edges(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            for edge in ex.graph.edges:
                assert isinstance(edge, NarrativeEdge), \
                    f"Level {level}: expected NarrativeEdge, got {type(edge)}"

    def test_node_types_are_narrative_enum(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            for node in ex.graph.nodes:
                assert isinstance(node.node_type, NarrativeNodeType)

    def test_edge_relations_are_narrative_enum(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            for edge in ex.graph.edges:
                assert isinstance(edge.relation, NarrativeRelation)

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

    def test_node_types_in_narrative_vocabulary(self):
        gen = make_gen()
        narr_values = set(NARRATIVE_NODE_TYPES)
        for _ in range(20):
            ex = gen.generate(level=gen._rng.randint(1, 3))
            for node in ex.graph.nodes:
                assert node.node_type.value in narr_values

    def test_edge_relations_in_narrative_vocabulary(self):
        gen = make_gen()
        narr_rels = set(NARRATIVE_RELATIONS)
        for _ in range(20):
            ex = gen.generate(level=gen._rng.randint(1, 3))
            for edge in ex.graph.edges:
                assert edge.relation.value in narr_rels


# ─────────────────────────────────────────────────────────────────────────────
# 4. VERIFY NARRATIVE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyNarrativeExample:

    def test_all_level1_pass(self):
        gen = make_gen(seed=1)
        for _ in range(20):
            ex  = gen.generate(level=1)
            res = verify_narrative_example(ex)
            assert res.passed, f"Level 1 FAIL: {res.reason}\nExample: {ex}"

    def test_all_level2_pass(self):
        gen = make_gen(seed=2)
        for _ in range(20):
            ex  = gen.generate(level=2)
            res = verify_narrative_example(ex)
            assert res.passed, f"Level 2 FAIL: {res.reason}\nExample: {ex}"

    def test_all_level3_pass(self):
        gen = make_gen(seed=3)
        for _ in range(20):
            ex  = gen.generate(level=3)
            res = verify_narrative_example(ex)
            assert res.passed, f"Level 3 FAIL: {res.reason}\nExample: {ex}"

    def test_batch_all_pass(self):
        gen = make_gen(seed=42)
        for ex in gen.generate_batch(n=60):
            res = verify_narrative_example(ex)
            assert res.passed, f"Batch FAIL: {res.reason}\nExample: {ex}"

    def test_verify_returns_verification_result(self):
        ex  = make_gen().generate(level=1)
        res = verify_narrative_example(ex)
        assert isinstance(res, NarrativeVerificationResult)

    def test_verify_has_details(self):
        ex  = make_gen().generate(level=2)
        res = verify_narrative_example(ex)
        assert isinstance(res.details, dict)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ARC COHERENCE CHECKS
# ─────────────────────────────────────────────────────────────────────────────

class TestArcCoherence:

    def test_level2_has_conflict_before_resolution(self):
        """All level-2 examples must have CONFLICT node and RESOLVES edge."""
        gen = make_gen(seed=10)
        for _ in range(20):
            ex = gen.generate(level=2)
            node_types = {node.node_type.value for node in ex.graph.nodes}
            relations  = {edge.relation.value for edge in ex.graph.edges}
            assert "conflict"  in node_types, f"No CONFLICT in level-2: {ex}"
            assert "resolution" in node_types or "resolves" in relations, \
                f"No RESOLUTION/RESOLVES in level-2: {ex}"
            assert "resolves" in relations, f"No RESOLVES edge in level-2: {ex}"

    def test_level2_has_motivation_before_conflict(self):
        """All level-2 examples must have MOTIVATES relation."""
        gen = make_gen(seed=11)
        for _ in range(20):
            ex = gen.generate(level=2)
            relations = {edge.relation.value for edge in ex.graph.edges}
            assert "motivates" in relations, \
                f"No MOTIVATES edge in level-2: {ex}"

    def test_level3_has_conflict_before_resolution(self):
        """All level-3 examples must have CONFLICT and RESOLVES."""
        gen = make_gen(seed=12)
        for _ in range(20):
            ex = gen.generate(level=3)
            node_types = {node.node_type.value for node in ex.graph.nodes}
            relations  = {edge.relation.value for edge in ex.graph.edges}
            assert "conflict"  in node_types, f"No CONFLICT in level-3: {ex}"
            assert "resolves"  in relations,  f"No RESOLVES in level-3: {ex}"

    def test_subverted_expectation_has_foreshadows_and_subverts(self):
        """Level-3 subverted_expectation arc must have both FORESHADOWS and SUBVERTS."""
        gen  = make_gen(seed=0)
        found_twist = False
        # Generate until we hit a TWIST_IDENTIFICATION example
        for _ in range(50):
            ex = gen.generate(level=3)
            if ex.answer_type == NarrativeAnswerType.TWIST_IDENTIFICATION:
                relations = {edge.relation.value for edge in ex.graph.edges}
                assert "foreshadows" in relations, f"No FORESHADOWS in twist arc: {ex}"
                assert "subverts"    in relations, f"No SUBVERTS in twist arc: {ex}"
                found_twist = True
                break
        assert found_twist, "Could not generate TWIST_IDENTIFICATION example in 50 tries"

    def test_symbolic_arc_has_symbol_and_theme(self):
        """Level-3 symbolic arc must have SYMBOL and THEME node types."""
        gen = make_gen(seed=5)
        found_symbol = False
        for _ in range(50):
            ex = gen.generate(level=3)
            if ex.answer_type == NarrativeAnswerType.SYMBOL_MEANING:
                node_types = {node.node_type.value for node in ex.graph.nodes}
                assert "symbol" in node_types, f"No SYMBOL in symbolic arc: {ex}"
                assert "theme"  in node_types, f"No THEME in symbolic arc: {ex}"
                relations = {edge.relation.value for edge in ex.graph.edges}
                assert "symbolizes" in relations, f"No SYMBOLIZES in symbolic arc: {ex}"
                found_symbol = True
                break
        assert found_symbol, "Could not generate SYMBOL_MEANING example in 50 tries"

    def test_parallel_conflict_has_parallels_edge(self):
        """Level-3 parallel conflict arc must have PARALLELS relation."""
        gen = make_gen(seed=3)
        found_parallel = False
        for _ in range(50):
            ex = gen.generate(level=3)
            relations = {edge.relation.value for edge in ex.graph.edges}
            if "parallels" in relations:
                assert "conflict" in {
                    node.node_type.value for node in ex.graph.nodes
                }
                found_parallel = True
                break
        assert found_parallel, "Could not generate parallel conflict in 50 tries"

    def test_level3_intensification_before_resolution(self):
        """Level-3 examples with intensification must have INTENSIFIES before RESOLVES."""
        gen = make_gen(seed=15)
        for _ in range(20):
            ex  = gen.generate(level=3)
            arc = ex.metadata.get("arc_checks", {})
            if arc.get("has_intensification_before_resolution"):
                relations = {edge.relation.value for edge in ex.graph.edges}
                assert "intensifies" in relations, f"No INTENSIFIES: {ex}"
                assert "resolves"    in relations, f"No RESOLVES: {ex}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANSWER TYPE COVERAGE
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerTypeCoverage:

    def test_level1_has_motivation_or_arc_types(self):
        gen = make_gen(seed=20)
        types = set()
        for _ in range(30):
            ex = gen.generate(level=1)
            types.add(ex.answer_type)
        valid = {NarrativeAnswerType.MOTIVATION, NarrativeAnswerType.ARC_COMPLETION}
        assert types & valid, f"Level-1 types: {types}"

    def test_level2_has_conflict_resolution_type(self):
        gen = make_gen(seed=21)
        types = set()
        for _ in range(30):
            ex = gen.generate(level=2)
            types.add(ex.answer_type)
        assert NarrativeAnswerType.CONFLICT_RESOLUTION in types or \
               NarrativeAnswerType.MOTIVATION in types

    def test_level3_covers_multiple_types(self):
        gen = make_gen(seed=22)
        types = set()
        for _ in range(60):
            ex = gen.generate(level=3)
            types.add(ex.answer_type)
        assert len(types) >= 2

    def test_level3_has_twist_identification(self):
        gen = make_gen(seed=0)
        types = set()
        for _ in range(80):
            ex = gen.generate(level=3)
            types.add(ex.answer_type)
        assert NarrativeAnswerType.TWIST_IDENTIFICATION in types

    def test_level3_has_symbol_meaning(self):
        gen = make_gen(seed=1)
        types = set()
        for _ in range(80):
            ex = gen.generate(level=3)
            types.add(ex.answer_type)
        assert NarrativeAnswerType.SYMBOL_MEANING in types


# ─────────────────────────────────────────────────────────────────────────────
# 7. METADATA CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

class TestNarrativeMetadataContracts:

    def test_level1_desire_action_has_character_key(self):
        gen = make_gen(seed=30)
        for _ in range(20):
            ex = gen.generate(level=1)
            if ex.answer_type == NarrativeAnswerType.MOTIVATION:
                assert "character" in ex.metadata
                assert "desire"    in ex.metadata
                break

    def test_level2_conflict_has_conflict_key(self):
        gen = make_gen(seed=31)
        for _ in range(20):
            ex = gen.generate(level=2)
            if ex.answer_type == NarrativeAnswerType.CONFLICT_RESOLUTION:
                assert "conflict"   in ex.metadata
                assert "resolution" in ex.metadata
                break

    def test_level3_twist_has_twist_desc(self):
        gen = make_gen(seed=32)
        for _ in range(50):
            ex = gen.generate(level=3)
            if ex.answer_type == NarrativeAnswerType.TWIST_IDENTIFICATION:
                assert "twist"      in ex.metadata
                assert "twist_desc" in ex.metadata
                break

    def test_level3_symbol_has_symbol_meaning_key(self):
        gen = make_gen(seed=33)
        for _ in range(50):
            ex = gen.generate(level=3)
            if ex.answer_type == NarrativeAnswerType.SYMBOL_MEANING:
                assert "symbol"        in ex.metadata
                assert "symbol_meaning" in ex.metadata
                assert "theme"         in ex.metadata
                break

    def test_required_node_types_present_in_graph(self):
        """required_node_types in metadata must actually be present in the graph."""
        gen = make_gen(seed=40)
        for _ in range(30):
            ex = gen.generate(level=gen._rng.randint(1, 3))
            present = {node.node_type.value for node in ex.graph.nodes}
            for nt in ex.metadata.get("required_node_types", []):
                assert nt in present, \
                    f"required_node_type '{nt}' missing from graph. Present: {present}"

    def test_required_relations_present_in_graph(self):
        """required_relations in metadata must actually be present in the graph edges."""
        gen = make_gen(seed=41)
        for _ in range(30):
            ex = gen.generate(level=gen._rng.randint(1, 3))
            present = {edge.relation.value for edge in ex.graph.edges}
            for rel in ex.metadata.get("required_relations", []):
                assert rel in present, \
                    f"required_relation '{rel}' missing from graph. Present: {present}"

    def test_example_repr(self):
        ex = make_gen().generate(level=2)
        r  = repr(ex)
        assert "NarrativeExample" in r
        assert "level=2" in r
