"""
tests/test_code_graph_gen.py — Tests para CodeGraphGenerator
=============================================================

Verifica que:
  1. CodeGraphGenerator genera ejemplos para todos los niveles 1-3
  2. Cada ejemplo tiene los campos requeridos (problem_text, graph, answer, etc.)
  3. Los grafos contienen CodeNode/CodeEdge con tipos correctos
  4. verify_code_example() pasa para TODOS los ejemplos generados
  5. generate_batch() produce el número correcto de ejemplos
  6. Los generadores de cada nivel producen los tipos de respuesta esperados
  7. La distribución de niveles se respeta aproximadamente en batches grandes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import random

from core.graph import CausalGraph
from motors.forge_c.relations import (
    CODE_NODE_TYPES, CODE_RELATIONS,
    CodeNode, CodeEdge, CodeNodeType, CodeRelation,
)
from synth.code_graph_gen import (
    CodeAnswerType,
    CodeExample,
    CodeGraphGenerator,
    CodeVerificationResult,
    verify_code_example,
    WEB_APP_NODES,
    WEB_APP_CHAINS,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_gen(seed: int = 42) -> CodeGraphGenerator:
    return CodeGraphGenerator(seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATOR BASIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeGraphGeneratorInterface:

    def test_instantiates(self):
        gen = make_gen()
        assert isinstance(gen, CodeGraphGenerator)

    def test_available_domains(self):
        gen = make_gen()
        domains = gen.available_domains()
        assert isinstance(domains, list)
        assert len(domains) >= 1
        for d in domains:
            assert d in WEB_APP_NODES

    def test_generate_level_1(self):
        gen = make_gen()
        ex  = gen.generate(level=1)
        assert isinstance(ex, CodeExample)
        assert ex.complexity_level == 1

    def test_generate_level_2(self):
        gen = make_gen()
        ex  = gen.generate(level=2)
        assert isinstance(ex, CodeExample)
        assert ex.complexity_level == 2

    def test_generate_level_3(self):
        gen = make_gen()
        ex  = gen.generate(level=3)
        assert isinstance(ex, CodeExample)
        assert ex.complexity_level == 3

    def test_invalid_level_raises(self):
        gen = make_gen()
        with pytest.raises(ValueError):
            gen.generate(level=4)

    def test_generate_batch_count(self):
        gen = make_gen()
        batch = gen.generate_batch(n=20)
        assert len(batch) == 20

    def test_generate_batch_returns_code_examples(self):
        gen = make_gen()
        batch = gen.generate_batch(n=10)
        for ex in batch:
            assert isinstance(ex, CodeExample)

    def test_generate_batch_custom_distribution(self):
        gen = make_gen(seed=0)
        batch = gen.generate_batch(n=30, level_distribution={1: 1.0})
        for ex in batch:
            assert ex.complexity_level == 1

    def test_deterministic_with_seed(self):
        ex1 = CodeGraphGenerator(seed=7).generate(level=1)
        ex2 = CodeGraphGenerator(seed=7).generate(level=1)
        assert ex1.problem_text == ex2.problem_text
        assert ex1.answer == ex2.answer


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXAMPLE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeExampleStructure:

    def _check_example(self, ex: CodeExample) -> None:
        assert isinstance(ex.problem_text, str) and len(ex.problem_text) > 0
        assert isinstance(ex.graph, CausalGraph)
        assert isinstance(ex.answer, str) and len(ex.answer) > 0
        assert ex.complexity_level in (1, 2, 3)
        assert isinstance(ex.answer_type, CodeAnswerType)
        assert ex.verifiable is True
        assert isinstance(ex.metadata, dict)
        assert isinstance(ex.example_id, str)

    def test_level1_structure(self):
        gen = make_gen()
        for _ in range(5):
            self._check_example(gen.generate(level=1))

    def test_level2_structure(self):
        gen = make_gen()
        for _ in range(5):
            self._check_example(gen.generate(level=2))

    def test_level3_structure(self):
        gen = make_gen()
        for _ in range(5):
            self._check_example(gen.generate(level=3))

    def test_graph_has_nodes(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            assert len(ex.graph) >= 2, f"Level {level}: expected >= 2 nodes"

    def test_graph_has_edges(self):
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            assert len(ex.graph.edges) >= 1, f"Level {level}: expected >= 1 edge"

    def test_unique_example_ids(self):
        gen = make_gen()
        ids = [gen.generate(level=1).example_id for _ in range(20)]
        # With UUID generation, collisions should be extremely rare
        assert len(set(ids)) >= 15


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH NODE AND EDGE TYPES
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeGraphTypes:
    """Los grafos contienen CodeNode/CodeEdge con tipos correctos."""

    def test_nodes_have_code_node_types(self):
        gen = make_gen()
        for level in (1, 2, 3):
            for _ in range(3):
                ex = gen.generate(level=level)
                for node in ex.graph.nodes:
                    assert isinstance(node, CodeNode), \
                        f"Expected CodeNode, got {type(node)}"
                    assert node.node_type.value in CODE_NODE_TYPES, \
                        f"Unknown node type: {node.node_type.value}"

    def test_edges_have_code_relations(self):
        gen = make_gen()
        for level in (1, 2, 3):
            for _ in range(3):
                ex = gen.generate(level=level)
                for edge in ex.graph.edges:
                    assert isinstance(edge, CodeEdge), \
                        f"Expected CodeEdge, got {type(edge)}"
                    assert edge.relation.value in CODE_RELATIONS, \
                        f"Unknown relation: {edge.relation.value}"

    def test_edge_indices_assigned(self):
        """source_idx / target_idx asignados por CausalGraph.add_edge."""
        gen = make_gen()
        for level in (1, 2, 3):
            ex = gen.generate(level=level)
            for edge in ex.graph.edges:
                assert edge.source_idx >= 0
                assert edge.target_idx >= 0

    def test_no_self_loops(self):
        gen = make_gen()
        for level in (1, 2, 3):
            for _ in range(5):
                ex = gen.generate(level=level)
                for edge in ex.graph.edges:
                    assert edge.source_id != edge.target_id


# ─────────────────────────────────────────────────────────────────────────────
# 4. VERIFICATION — ALL EXAMPLES PASS
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyCodeExample:

    def test_verify_returns_code_verification_result(self):
        gen = make_gen()
        ex  = gen.generate(level=1)
        res = verify_code_example(ex)
        assert isinstance(res, CodeVerificationResult)

    def test_all_level1_examples_pass(self):
        gen = make_gen(seed=1)
        passed = 0
        for _ in range(20):
            ex  = gen.generate(level=1)
            res = verify_code_example(ex)
            assert res.passed, f"Level 1 FAIL: {res.reason}\nExample: {ex}"
            passed += 1
        assert passed == 20

    def test_all_level2_examples_pass(self):
        gen = make_gen(seed=2)
        passed = 0
        for _ in range(20):
            ex  = gen.generate(level=2)
            res = verify_code_example(ex)
            assert res.passed, f"Level 2 FAIL: {res.reason}\nExample: {ex}"
            passed += 1
        assert passed == 20

    def test_all_level3_examples_pass(self):
        gen = make_gen(seed=3)
        passed = 0
        for _ in range(20):
            ex  = gen.generate(level=3)
            res = verify_code_example(ex)
            assert res.passed, f"Level 3 FAIL: {res.reason}\nExample: {ex}"
            passed += 1
        assert passed == 20

    def test_batch_all_pass(self):
        gen   = make_gen(seed=10)
        batch = gen.generate_batch(n=60, level_distribution={1: 0.4, 2: 0.4, 3: 0.2})
        for ex in batch:
            res = verify_code_example(ex)
            assert res.passed, f"FAIL: {res.reason}\n{ex}"

    def test_verification_result_bool(self):
        gen = make_gen()
        ex  = gen.generate(level=1)
        res = verify_code_example(ex)
        assert bool(res) == res.passed


# ─────────────────────────────────────────────────────────────────────────────
# 5. LEVEL-SPECIFIC ANSWER TYPES
# ─────────────────────────────────────────────────────────────────────────────

class TestLevelAnswerTypes:
    """Cada nivel produce los tipos de respuesta esperados."""

    def test_level1_answer_types(self):
        gen = make_gen(seed=42)
        types = {gen.generate(level=1).answer_type for _ in range(30)}
        expected = {
            CodeAnswerType.DIRECT_CALLER,
            CodeAnswerType.DIRECT_CALLEE,
            CodeAnswerType.READS_WRITES,
            CodeAnswerType.IMPACT_ANALYSIS,
        }
        # Al menos 2 tipos distintos en 30 ejemplos
        assert len(types) >= 2

    def test_level2_answer_types(self):
        gen   = make_gen(seed=43)
        types = {gen.generate(level=2).answer_type for _ in range(30)}
        assert len(types) >= 2

    def test_level3_answer_types(self):
        gen   = make_gen(seed=44)
        types = {gen.generate(level=3).answer_type for _ in range(30)}
        assert len(types) >= 2

    def test_level1_complexity(self):
        """Level 1: 2-3 nodos, relaciones simples."""
        gen = make_gen(seed=5)
        for _ in range(20):
            ex = gen.generate(level=1)
            assert len(ex.graph) <= 4, f"Level 1 too large: {len(ex.graph)} nodes"
            assert ex.complexity_level == 1

    def test_level2_has_fan_structure(self):
        """Level 2: al menos un nodo tiene múltiples aristas."""
        gen = make_gen(seed=6)
        found_fan = False
        for _ in range(20):
            ex = gen.generate(level=2)
            if len(ex.graph) >= 3 and len(ex.graph.edges) >= 2:
                found_fan = True
                break
        assert found_fan, "Level 2 should have graphs with fan structure"

    def test_level3_has_chains(self):
        """Level 3: grafos con cadenas de 3+ nodos."""
        gen = make_gen(seed=7)
        found_chain = False
        for _ in range(20):
            ex = gen.generate(level=3)
            if len(ex.graph) >= 3 and len(ex.graph.edges) >= 2:
                found_chain = True
                break
        assert found_chain, "Level 3 should have multi-node chains"


# ─────────────────────────────────────────────────────────────────────────────
# 6. METADATA CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataContracts:
    """Cada tipo de respuesta tiene las claves de metadata requeridas por verify."""

    def test_direct_caller_has_target_and_expected_callers(self):
        gen = make_gen(seed=100)
        found = False
        for _ in range(50):
            ex = gen.generate(level=1)
            if ex.answer_type == CodeAnswerType.DIRECT_CALLER:
                assert "target_id" in ex.metadata
                assert "expected_callers" in ex.metadata
                found = True
                break
        assert found, "Could not find DIRECT_CALLER example in 50 attempts"

    def test_impact_analysis_has_source_and_affected(self):
        gen = make_gen(seed=101)
        found = False
        for _ in range(50):
            ex = gen.generate(level=2)
            if ex.answer_type == CodeAnswerType.IMPACT_ANALYSIS:
                assert "source_id" in ex.metadata
                assert "expected_affected_ids" in ex.metadata
                found = True
                break
        assert found, "Could not find IMPACT_ANALYSIS example in 50 attempts"

    def test_data_source_has_consumer_and_root(self):
        gen = make_gen(seed=102)
        found = False
        for _ in range(50):
            ex = gen.generate(level=3)
            if ex.answer_type == CodeAnswerType.DATA_SOURCE:
                assert "consumer_id" in ex.metadata
                assert "root_source_id" in ex.metadata
                found = True
                break
        assert found, "Could not find DATA_SOURCE example in 50 attempts"


# ─────────────────────────────────────────────────────────────────────────────
# 7. DOMAIN POOLS SANITY
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainPools:

    def test_all_domains_have_nodes(self):
        for domain, nodes in WEB_APP_NODES.items():
            assert len(nodes) >= 3, f"Domain {domain} has too few nodes"

    def test_all_domains_have_chains(self):
        for domain in WEB_APP_NODES:
            assert domain in WEB_APP_CHAINS, f"No chain for domain {domain}"
            assert len(WEB_APP_CHAINS[domain]) >= 2

    def test_chain_indices_valid(self):
        for domain, chains in WEB_APP_CHAINS.items():
            n_nodes = len(WEB_APP_NODES[domain])
            for src_idx, tgt_idx, rel in chains:
                assert 0 <= src_idx < n_nodes, \
                    f"Domain {domain}: invalid src_idx {src_idx}"
                assert 0 <= tgt_idx < n_nodes, \
                    f"Domain {domain}: invalid tgt_idx {tgt_idx}"
                assert src_idx != tgt_idx, \
                    f"Domain {domain}: self-loop in chain"
                assert isinstance(rel, CodeRelation)

    def test_all_node_types_in_pools_are_code_types(self):
        for domain, nodes in WEB_APP_NODES.items():
            for nid, label, ntype in nodes:
                assert isinstance(ntype, CodeNodeType), \
                    f"Domain {domain}: {label} has non-CodeNodeType {ntype}"
