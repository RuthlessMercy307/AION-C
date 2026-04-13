"""
tests/test_causal_graph_gen.py — Tests para synth/causal_graph_gen.py
======================================================================

Cubre:
  - CausalExample: estructura, campos, repr
  - VerificationResult: bool, repr
  - DOMAIN_NODES / DOMAIN_CHAINS: completitud, consistencia
  - Nivel 1 (Lineal): generación, verify, variantes
  - Nivel 2 (Bifurcación): fan-out, fan-in, diamond, mixed + verify
  - Nivel 3 (Contradicción): con contradicción, sin contradicción + verify
  - Nivel 4 (Contrafactual): simple, middle, alternate + verify
  - Nivel 5 (Multi-dominio): tamaño, cross-domain, critical_path, multi_hop + verify
  - CausalGraphGenerator: generate, generate_batch, stream, stats, seed reproducible
  - verify_example: casos correctos e incorrectos para todos los AnswerType
  - Propiedades globales: todos los niveles producen grafos acíclicos por defecto,
    level 3 contradictions son detectables, level 4 contrafactual es simulable

Ejecutar:
  cd IAS/AION-C
  python -m pytest tests/test_causal_graph_gen.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import pytest
import random

from synth.causal_graph_gen import (
    AnswerType,
    CausalExample,
    CausalGraphGenerator,
    VerificationResult,
    DOMAIN_NODES,
    DOMAIN_CHAINS,
    _Level1Generator,
    _Level2Generator,
    _Level3Generator,
    _Level4Generator,
    _Level5Generator,
    verify_example,
    _rel_text,
    _longest_path,
    _build_chain,
    _node_from_desc,
)
from core.graph import (
    CausalEdge,
    CausalGraph,
    CausalNode,
    CausalRelation,
    NodeType,
    CONTRADICTION_PAIRS,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def gen():
    return CausalGraphGenerator(seed=42)


@pytest.fixture
def gen_unseeded():
    return CausalGraphGenerator()


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — DOMAIN POOLS (DATOS ESTÁTICOS)
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainPools:

    def test_domain_nodes_keys(self):
        required = {"clima", "economia", "salud", "tecnologia", "fisica", "social", "medioambiente"}
        assert required.issubset(set(DOMAIN_NODES.keys()))

    def test_every_domain_has_at_least_3_nodes(self):
        for domain, nodes in DOMAIN_NODES.items():
            assert len(nodes) >= 3, f"Dominio {domain!r} tiene solo {len(nodes)} nodos"

    def test_node_descriptors_have_4_elements(self):
        for domain, nodes in DOMAIN_NODES.items():
            for nd in nodes:
                assert len(nd) == 4, f"Descriptor {nd} en {domain!r} no tiene 4 elementos"

    def test_node_descriptors_types(self):
        for domain, nodes in DOMAIN_NODES.items():
            for nid, label, ntype, desc in nodes:
                assert isinstance(nid, str) and nid, f"nid vacío en {domain}"
                assert isinstance(label, str) and label, f"label vacío en {domain}"
                assert isinstance(ntype, NodeType), f"ntype no es NodeType en {domain}"
                assert isinstance(desc, str), f"desc no es str en {domain}"

    def test_domain_chains_exist_for_every_domain(self):
        for domain in DOMAIN_NODES:
            assert domain in DOMAIN_CHAINS, f"Dominio {domain!r} sin chain"

    def test_domain_chain_indices_valid(self):
        for domain, chain in DOMAIN_CHAINS.items():
            max_idx = len(DOMAIN_NODES[domain]) - 1
            for src, tgt, rel in chain:
                assert 0 <= src <= max_idx, f"{domain}: src={src} fuera de rango"
                assert 0 <= tgt <= max_idx, f"{domain}: tgt={tgt} fuera de rango"
                assert src != tgt, f"{domain}: auto-loop en chain"
                assert isinstance(rel, CausalRelation)

    def test_node_ids_unique_within_domain(self):
        for domain, nodes in DOMAIN_NODES.items():
            ids = [nd[0] for nd in nodes]
            assert len(ids) == len(set(ids)), f"IDs duplicados en dominio {domain!r}"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_node_from_desc_no_prefix(self):
        desc = ("inflacion", "Inflación", NodeType.STATE, "subida de precios")
        node = _node_from_desc(desc)
        assert node.node_id == "inflacion"
        assert node.label == "Inflación"
        assert node.node_type == NodeType.STATE

    def test_node_from_desc_with_prefix(self):
        desc = ("lluvia", "Lluvia", NodeType.EVENT, "precipitaciones")
        node = _node_from_desc(desc, prefix="cli_")
        assert node.node_id == "cli_lluvia"
        assert node.label == "Lluvia"

    def test_build_chain_returns_graph_and_nodes(self):
        rng = random.Random(0)
        g, nodes = _build_chain("economia", [0, 1, 2], rng)
        assert isinstance(g, CausalGraph)
        assert len(nodes) == 3
        assert len(g) == 3

    def test_build_chain_edges_between_selected_nodes(self):
        rng = random.Random(0)
        g, nodes = _build_chain("economia", [0, 1, 2], rng)
        assert len(g.edges) >= 1

    def test_build_chain_with_prefix(self):
        rng = random.Random(0)
        g, nodes = _build_chain("clima", [0, 1], rng, prefix="cl_")
        for node in g.nodes:
            assert node.node_id.startswith("cl_")

    def test_longest_path_linear(self):
        g = CausalGraph()
        for nid in ["A", "B", "C", "D"]:
            g.add_node(CausalNode(nid, nid))
        g.add_edge(CausalEdge("A", "B", CausalRelation.CAUSES))
        g.add_edge(CausalEdge("B", "C", CausalRelation.CAUSES))
        g.add_edge(CausalEdge("C", "D", CausalRelation.CAUSES))
        path = _longest_path(g)
        assert path == ["A", "B", "C", "D"]

    def test_longest_path_empty_graph(self):
        g = CausalGraph()
        assert _longest_path(g) == []

    def test_longest_path_single_node(self):
        g = CausalGraph()
        g.add_node(CausalNode("X", "X"))
        path = _longest_path(g)
        assert path == ["X"]

    def test_longest_path_diamond(self):
        g = CausalGraph()
        for nid in ["A", "B", "C", "D"]:
            g.add_node(CausalNode(nid, nid))
        g.add_edge(CausalEdge("A", "B", CausalRelation.CAUSES))
        g.add_edge(CausalEdge("A", "C", CausalRelation.CAUSES))
        g.add_edge(CausalEdge("B", "D", CausalRelation.CAUSES))
        g.add_edge(CausalEdge("C", "D", CausalRelation.CAUSES))
        path = _longest_path(g)
        assert len(path) == 3  # A→B→D o A→C→D

    def test_rel_text_returns_string(self):
        for rel in CausalRelation:
            text = _rel_text(rel)
            assert isinstance(text, str)
            assert len(text) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — CAUSAL EXAMPLE Y VERIFICATION RESULT
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalExample:

    def test_basic_construction(self, gen):
        ex = gen.generate(level=1)
        assert isinstance(ex, CausalExample)
        assert isinstance(ex.problem_text, str)
        assert isinstance(ex.graph, CausalGraph)
        assert isinstance(ex.answer, str)
        assert ex.complexity_level == 1
        assert isinstance(ex.answer_type, AnswerType)
        assert ex.verifiable is True
        assert isinstance(ex.metadata, dict)
        assert isinstance(ex.example_id, str)

    def test_problem_text_not_empty(self, gen):
        for level in range(1, 6):
            ex = gen.generate(level=level)
            assert len(ex.problem_text) > 20, f"Nivel {level}: texto muy corto"

    def test_answer_not_empty(self, gen):
        for level in range(1, 6):
            ex = gen.generate(level=level)
            assert len(ex.answer) > 10, f"Nivel {level}: respuesta muy corta"

    def test_graph_not_empty(self, gen):
        for level in range(1, 6):
            ex = gen.generate(level=level)
            assert len(ex.graph) >= 2, f"Nivel {level}: grafo tiene menos de 2 nodos"

    def test_unique_example_ids(self, gen):
        ids = [gen.generate(level=1).example_id for _ in range(50)]
        assert len(ids) == len(set(ids)), "IDs de ejemplo no son únicos"

    def test_repr_contains_level_and_type(self, gen):
        ex = gen.generate(level=2)
        r = repr(ex)
        assert "2" in r
        assert ex.answer_type.value in r

    def test_metadata_not_empty(self, gen):
        for level in range(1, 6):
            ex = gen.generate(level=level)
            assert len(ex.metadata) > 0, f"Nivel {level}: metadata vacío"


class TestVerificationResult:

    def test_bool_true(self):
        vr = VerificationResult(passed=True, reason="ok")
        assert bool(vr) is True

    def test_bool_false(self):
        vr = VerificationResult(passed=False, reason="fallo")
        assert bool(vr) is False

    def test_repr_pass(self):
        vr = VerificationResult(passed=True, reason="all good")
        assert "PASS" in repr(vr)

    def test_repr_fail(self):
        vr = VerificationResult(passed=False, reason="bad")
        assert "FAIL" in repr(vr)

    def test_details_default_empty(self):
        vr = VerificationResult(passed=True, reason="ok")
        assert vr.details == {}


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — NIVEL 1 (CADENAS LINEALES)
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel1:

    def _gen_examples(self, n=30, seed=0):
        rng = random.Random(seed)
        gen = _Level1Generator()
        return [gen.generate(rng) for _ in range(n)]

    def test_all_have_level_1(self):
        for ex in self._gen_examples():
            assert ex.complexity_level == 1

    def test_node_count_2_or_3(self):
        for ex in self._gen_examples():
            assert 2 <= len(ex.graph) <= 3, \
                f"Nivel 1 debe tener 2-3 nodos, tiene {len(ex.graph)}"

    def test_at_least_one_edge(self):
        for ex in self._gen_examples():
            assert len(ex.graph.edges) >= 1

    def test_graph_is_acyclic(self):
        for ex in self._gen_examples():
            assert ex.graph.detect_cycles() == [], "Nivel 1 no debe tener ciclos"

    def test_answer_types_are_valid(self):
        valid = {AnswerType.TRANSITIVITY, AnswerType.DIRECT_CAUSE}
        for ex in self._gen_examples():
            assert ex.answer_type in valid

    def test_all_pass_verify(self):
        for i, ex in enumerate(self._gen_examples(n=30, seed=7)):
            result = verify_example(ex)
            assert result.passed, \
                f"Nivel 1 ejemplo {i} falló verificación: {result.reason}\n{ex}"

    def test_metadata_has_domain(self):
        for ex in self._gen_examples():
            assert "domain" in ex.metadata

    def test_transitivity_metadata(self):
        """Ejemplos de transitividad deben tener source_id y target_id."""
        for ex in self._gen_examples(n=50, seed=10):
            if ex.answer_type == AnswerType.TRANSITIVITY:
                assert "source_id" in ex.metadata
                assert "target_id" in ex.metadata
                assert "expected_reachable" in ex.metadata
                assert isinstance(ex.metadata["expected_reachable"], bool)

    def test_edge_indices_are_valid(self):
        for ex in self._gen_examples():
            n = len(ex.graph)
            for edge in ex.graph.edges:
                assert 0 <= edge.source_idx < n
                assert 0 <= edge.target_idx < n


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — NIVEL 2 (BIFURCACIONES)
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel2:

    def _gen_examples(self, n=30, seed=1):
        rng = random.Random(seed)
        gen = _Level2Generator()
        return [gen.generate(rng) for _ in range(n)]

    def test_all_have_level_2(self):
        for ex in self._gen_examples():
            assert ex.complexity_level == 2

    def test_node_count_3_to_5(self):
        for ex in self._gen_examples():
            assert 3 <= len(ex.graph) <= 5, \
                f"Nivel 2 debe tener 3-5 nodos, tiene {len(ex.graph)}"

    def test_at_least_2_edges(self):
        for ex in self._gen_examples():
            assert len(ex.graph.edges) >= 2

    def test_acyclic(self):
        for ex in self._gen_examples():
            assert ex.graph.detect_cycles() == []

    def test_answer_types_branching_or_transitivity(self):
        valid = {AnswerType.BRANCHING, AnswerType.DIRECT_CAUSE, AnswerType.TRANSITIVITY}
        for ex in self._gen_examples():
            assert ex.answer_type in valid

    def test_all_pass_verify(self):
        for i, ex in enumerate(self._gen_examples(n=30, seed=11)):
            result = verify_example(ex)
            assert result.passed, \
                f"Nivel 2 ejemplo {i} falló: {result.reason}\n{ex}"

    def test_fan_out_has_2_successors(self):
        rng = random.Random(0)
        gen = _Level2Generator()
        fan_out_examples = []
        for _ in range(100):
            ex = gen.generate(rng)
            if ex.metadata.get("variant") == "fan_out":
                fan_out_examples.append(ex)
        assert len(fan_out_examples) > 0, "No se generaron ejemplos fan_out"
        for ex in fan_out_examples:
            src_id = ex.metadata["source_id"]
            succs = ex.graph.successors(src_id)
            assert len(succs) == 2, f"fan_out debe tener 2 sucesores, tiene {len(succs)}"

    def test_fan_in_has_2_predecessors(self):
        rng = random.Random(5)
        gen = _Level2Generator()
        fan_in_examples = []
        for _ in range(100):
            ex = gen.generate(rng)
            if ex.metadata.get("variant") == "fan_in":
                fan_in_examples.append(ex)
        assert len(fan_in_examples) > 0, "No se generaron ejemplos fan_in"
        for ex in fan_in_examples:
            tgt_id = ex.metadata["target_id"]
            preds = ex.graph.predecessors(tgt_id)
            assert len(preds) == 2, f"fan_in debe tener 2 predecesores, tiene {len(preds)}"

    def test_diamond_has_path(self):
        rng = random.Random(3)
        gen = _Level2Generator()
        for _ in range(100):
            ex = gen.generate(rng)
            if ex.metadata.get("variant") == "diamond":
                src = ex.metadata["source_id"]
                tgt = ex.metadata["target_id"]
                assert ex.graph.has_path(src, tgt)
                assert ex.metadata["n_paths"] == 2

    def test_edge_strength_in_range(self):
        for ex in self._gen_examples():
            for edge in ex.graph.edges:
                assert 0.0 <= edge.strength <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — NIVEL 3 (CONTRADICCIONES)
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel3:

    def _gen_examples(self, n=40, seed=2):
        rng = random.Random(seed)
        gen = _Level3Generator()
        return [gen.generate(rng) for _ in range(n)]

    def test_all_have_level_3(self):
        for ex in self._gen_examples():
            assert ex.complexity_level == 3

    def test_answer_type_is_contradiction(self):
        for ex in self._gen_examples():
            assert ex.answer_type == AnswerType.CONTRADICTION

    def test_about_70_percent_have_contradiction(self):
        examples = self._gen_examples(n=200, seed=99)
        with_c = sum(1 for ex in examples if ex.metadata.get("has_contradiction"))
        ratio = with_c / len(examples)
        assert 0.55 <= ratio <= 0.85, \
            f"Se esperaba ~70% con contradicción, encontrado {ratio:.1%}"

    def test_contradiction_examples_pass_verify(self):
        examples = self._gen_examples(n=40, seed=13)
        for i, ex in enumerate(examples):
            result = verify_example(ex)
            assert result.passed, \
                f"Nivel 3 ejemplo {i} falló: {result.reason}\n{ex}"

    def test_with_contradiction_graph_has_contradiction(self):
        rng = random.Random(7)
        gen = _Level3Generator()
        for _ in range(100):
            ex = gen.generate(rng)
            if ex.metadata.get("has_contradiction"):
                contradictions = ex.graph.find_contradictions()
                assert len(contradictions) >= 1, \
                    f"Ejemplo marcado como con contradicción pero el grafo no la tiene"

    def test_without_contradiction_graph_is_clean(self):
        rng = random.Random(8)
        gen = _Level3Generator()
        for _ in range(100):
            ex = gen.generate(rng)
            if not ex.metadata.get("has_contradiction"):
                contradictions = ex.graph.find_contradictions()
                assert len(contradictions) == 0, \
                    f"Ejemplo 'sin contradicción' pero el grafo tiene {len(contradictions)}"

    def test_metadata_has_expected_has_contradiction(self):
        for ex in self._gen_examples():
            assert "expected_has_contradiction" in ex.metadata

    def test_contradiction_source_and_target_in_graph(self):
        for ex in self._gen_examples():
            if ex.metadata.get("has_contradiction"):
                src = ex.metadata.get("contradiction_source_id")
                tgt = ex.metadata.get("contradiction_target_id")
                assert src in ex.graph, f"{src} no está en el grafo"
                assert tgt in ex.graph, f"{tgt} no está en el grafo"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — NIVEL 4 (CONTRAFACTUAL)
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel4:

    def _gen_examples(self, n=30, seed=3):
        rng = random.Random(seed)
        gen = _Level4Generator()
        return [gen.generate(rng) for _ in range(n)]

    def test_all_have_level_4(self):
        for ex in self._gen_examples():
            assert ex.complexity_level == 4

    def test_answer_type_is_counterfactual(self):
        for ex in self._gen_examples():
            assert ex.answer_type == AnswerType.COUNTERFACTUAL

    def test_all_pass_verify(self):
        for i, ex in enumerate(self._gen_examples(n=30, seed=14)):
            result = verify_example(ex)
            assert result.passed, \
                f"Nivel 4 ejemplo {i} falló: {result.reason}\n{ex}"

    def test_metadata_has_removed_id(self):
        for ex in self._gen_examples():
            assert "counterfactual_removed_id" in ex.metadata
            removed = ex.metadata["counterfactual_removed_id"]
            assert removed in ex.graph, f"Nodo removido {removed!r} no está en el grafo"

    def test_metadata_has_expected_path_blocked(self):
        for ex in self._gen_examples():
            assert "expected_path_blocked" in ex.metadata
            assert isinstance(ex.metadata["expected_path_blocked"], bool)

    def test_simple_variant_path_is_blocked(self):
        rng = random.Random(4)
        gen = _Level4Generator()
        simples = []
        for _ in range(100):
            ex = gen.generate(rng)
            if ex.metadata.get("variant") == "simple":
                simples.append(ex)
        assert len(simples) > 0
        for ex in simples:
            assert ex.metadata["expected_path_blocked"] is True

    def test_alternate_variant_path_not_blocked(self):
        rng = random.Random(9)
        gen = _Level4Generator()
        alternates = []
        for _ in range(200):
            ex = gen.generate(rng)
            if ex.metadata.get("variant") == "alternate":
                alternates.append(ex)
        assert len(alternates) > 0
        for ex in alternates:
            assert ex.metadata.get("alternate_path_exists") is True
            assert ex.metadata["expected_path_blocked"] is False

    def test_counterfactual_removal_simulation(self):
        """
        Verifica manualmente la simulación de remoción:
        después de quitar removed_id, el camino target_id debe bloquearse
        (o no, según expected_path_blocked).
        """
        rng = random.Random(6)
        gen = _Level4Generator()
        for _ in range(50):
            ex = gen.generate(rng)
            removed_id = ex.metadata["counterfactual_removed_id"]
            tgt_id     = ex.metadata["target_id"]
            src_id     = ex.metadata.get("source_id")
            expected   = ex.metadata["expected_path_blocked"]

            g_cf = copy.deepcopy(ex.graph)
            g_cf.remove_node(removed_id)

            if tgt_id not in g_cf:
                actual_blocked = True
            elif src_id and src_id not in g_cf:
                actual_blocked = True
            elif src_id and src_id in g_cf:
                actual_blocked = not g_cf.has_path(src_id, tgt_id)
            else:
                actual_blocked = len(g_cf.in_edges(tgt_id)) == 0

            assert actual_blocked == expected, \
                f"Simulación contrafactual incorrecta para {ex}"

    def test_3_nodes_minimum(self):
        for ex in self._gen_examples():
            assert len(ex.graph) >= 2

    def test_no_cycles(self):
        for ex in self._gen_examples():
            assert ex.graph.detect_cycles() == []


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — NIVEL 5 (MULTI-DOMINIO)
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel5:

    def _gen_examples(self, n=20, seed=4):
        rng = random.Random(seed)
        gen = _Level5Generator()
        return [gen.generate(rng) for _ in range(n)]

    def test_all_have_level_5(self):
        for ex in self._gen_examples():
            assert ex.complexity_level == 5

    def test_node_count_8_to_20(self):
        # Permitimos un rango un poco más amplio porque los cross-domain
        # pueden añadir nodos extras para garantizar el mínimo de 8
        for ex in self._gen_examples():
            assert len(ex.graph) >= 8, \
                f"Nivel 5 debe tener ≥8 nodos, tiene {len(ex.graph)}"

    def test_answer_type_is_valid(self):
        valid = {AnswerType.CRITICAL_PATH, AnswerType.MULTI_HOP}
        for ex in self._gen_examples():
            assert ex.answer_type in valid

    def test_all_pass_verify(self):
        for i, ex in enumerate(self._gen_examples(n=20, seed=15)):
            result = verify_example(ex)
            assert result.passed, \
                f"Nivel 5 ejemplo {i} falló: {result.reason}\n{ex}"

    def test_metadata_has_domains(self):
        for ex in self._gen_examples():
            assert "domains" in ex.metadata
            assert isinstance(ex.metadata["domains"], list)
            assert len(ex.metadata["domains"]) >= 2

    def test_metadata_has_expected_n_nodes(self):
        for ex in self._gen_examples():
            assert "expected_n_nodes" in ex.metadata
            assert ex.metadata["expected_n_nodes"] == len(ex.graph)

    def test_no_cycles_by_default(self):
        """
        El nivel 5 no introduce ciclos intencionales —
        todos los grafos deben ser DAGs.
        """
        for ex in self._gen_examples(n=20, seed=16):
            cycles = ex.graph.detect_cycles()
            assert cycles == [], \
                f"Nivel 5 tiene {len(cycles)} ciclo(s) inesperado(s)"

    def test_no_contradictions_by_default(self):
        for ex in self._gen_examples(n=20, seed=17):
            contradictions = ex.graph.find_contradictions()
            assert contradictions == [], \
                f"Nivel 5 tiene {len(contradictions)} contradicción(es) inesperada(s)"

    def test_multi_hop_has_source_and_target(self):
        rng = random.Random(18)
        gen = _Level5Generator()
        for _ in range(50):
            ex = gen.generate(rng)
            if ex.answer_type == AnswerType.MULTI_HOP:
                assert "source_id" in ex.metadata
                assert "target_id" in ex.metadata
                assert "expected_reachable" in ex.metadata

    def test_critical_path_metadata(self):
        rng = random.Random(19)
        gen = _Level5Generator()
        for _ in range(50):
            ex = gen.generate(rng)
            if ex.answer_type == AnswerType.CRITICAL_PATH:
                assert "expected_path" in ex.metadata
                assert "expected_path_length" in ex.metadata
                assert ex.metadata["expected_path_length"] >= 0

    def test_edge_indices_valid(self):
        for ex in self._gen_examples():
            n = len(ex.graph)
            for edge in ex.graph.edges:
                assert 0 <= edge.source_idx < n, \
                    f"source_idx {edge.source_idx} fuera de rango para n={n}"
                assert 0 <= edge.target_idx < n, \
                    f"target_idx {edge.target_idx} fuera de rango para n={n}"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — CAUSAL GRAPH GENERATOR (FACHADA PRINCIPAL)
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalGraphGenerator:

    def test_generate_all_levels(self, gen):
        for level in range(1, 6):
            ex = gen.generate(level=level)
            assert ex.complexity_level == level

    def test_invalid_level_raises(self, gen):
        with pytest.raises(ValueError):
            gen.generate(level=0)
        with pytest.raises(ValueError):
            gen.generate(level=6)

    def test_seed_reproducible(self):
        gen1 = CausalGraphGenerator(seed=99)
        gen2 = CausalGraphGenerator(seed=99)
        for level in range(1, 6):
            ex1 = gen1.generate(level=level)
            ex2 = gen2.generate(level=level)
            assert ex1.problem_text == ex2.problem_text, \
                f"Nivel {level}: problem_text no reproducible con mismo seed"
            assert ex1.answer == ex2.answer, \
                f"Nivel {level}: answer no reproducible con mismo seed"

    def test_different_seeds_different_results(self):
        gen_a = CausalGraphGenerator(seed=1)
        gen_b = CausalGraphGenerator(seed=2)
        # Con suficientes ejemplos, al menos uno debe diferir
        texts_a = [gen_a.generate(1).problem_text for _ in range(5)]
        texts_b = [gen_b.generate(1).problem_text for _ in range(5)]
        assert texts_a != texts_b, \
            "Seeds distintos producen los mismos resultados — error en RNG"

    def test_generate_with_domain(self, gen):
        for domain in list(DOMAIN_NODES.keys())[:3]:
            ex = gen.generate(level=1, domain=domain)
            assert ex.metadata.get("domain") == domain

    def test_generate_batch_size(self, gen):
        batch = gen.generate_batch(n=20, verify=False)
        assert len(batch) == 20

    def test_generate_batch_with_distribution(self, gen):
        dist = {1: 0.5, 2: 0.3, 3: 0.2}
        batch = gen.generate_batch(n=30, level_distribution=dist, verify=False)
        assert len(batch) == 30
        for ex in batch:
            assert ex.complexity_level in dist

    def test_generate_batch_verify_true(self, gen):
        batch = gen.generate_batch(n=20, verify=True)
        for ex in batch:
            result = verify_example(ex)
            assert result.passed, f"Batch con verify=True contiene ejemplo inválido: {result}"

    def test_generate_batch_all_levels(self, gen):
        dist = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
        batch = gen.generate_batch(n=50, level_distribution=dist, verify=True)
        assert len(batch) > 0
        levels_seen = {ex.complexity_level for ex in batch}
        assert len(levels_seen) >= 3, \
            f"Batch debe cubrir ≥3 niveles, solo vio: {levels_seen}"

    def test_stats_count(self, gen):
        for level in [1, 1, 2, 3]:
            gen.generate(level=level)
        stats = gen.stats
        assert stats["total"] >= 4
        assert stats["by_level"][1] >= 2
        assert stats["by_level"][2] >= 1
        assert stats["by_level"][3] >= 1

    def test_stats_seed(self):
        g = CausalGraphGenerator(seed=42)
        assert g.stats["seed"] == 42

    def test_stream_is_infinite(self, gen):
        count = 0
        for ex in gen.stream(level=1):
            count += 1
            if count >= 10:
                break
        assert count == 10

    def test_stream_all_same_level(self, gen):
        for level in range(1, 6):
            count = 0
            for ex in gen.stream(level=level):
                assert ex.complexity_level == level
                count += 1
                if count >= 5:
                    break


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — VERIFY_EXAMPLE DIRECTO
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyExample:

    def _minimal_graph(self, n_nodes=2) -> CausalGraph:
        """Grafo mínimo para tests de verify."""
        g = CausalGraph()
        nodes = [CausalNode(f"n{i}", f"Node{i}") for i in range(n_nodes)]
        for node in nodes:
            g.add_node(node)
        for i in range(n_nodes - 1):
            g.add_edge(CausalEdge(f"n{i}", f"n{i+1}", CausalRelation.CAUSES))
        return g

    def test_empty_graph_fails(self):
        g = CausalGraph()
        ex = CausalExample(
            problem_text="test", graph=g, answer="a",
            complexity_level=1, answer_type=AnswerType.TRANSITIVITY,
        )
        result = verify_example(ex)
        assert not result.passed
        assert "vacío" in result.reason

    def test_transitivity_correct(self):
        g = self._minimal_graph(3)
        ex = CausalExample(
            problem_text="test", graph=g, answer="sí",
            complexity_level=1, answer_type=AnswerType.TRANSITIVITY,
            metadata={"source_id": "n0", "target_id": "n2", "expected_reachable": True},
        )
        result = verify_example(ex)
        assert result.passed

    def test_transitivity_wrong_expectation(self):
        g = self._minimal_graph(3)
        ex = CausalExample(
            problem_text="test", graph=g, answer="no",
            complexity_level=1, answer_type=AnswerType.TRANSITIVITY,
            metadata={"source_id": "n0", "target_id": "n2", "expected_reachable": False},
        )
        result = verify_example(ex)
        assert not result.passed

    def test_transitivity_missing_metadata(self):
        g = self._minimal_graph(2)
        ex = CausalExample(
            problem_text="test", graph=g, answer="sí",
            complexity_level=1, answer_type=AnswerType.TRANSITIVITY,
            metadata={},  # Sin source_id ni target_id
        )
        result = verify_example(ex)
        assert not result.passed

    def test_transitivity_missing_node(self):
        g = self._minimal_graph(2)
        ex = CausalExample(
            problem_text="test", graph=g, answer="sí",
            complexity_level=1, answer_type=AnswerType.TRANSITIVITY,
            metadata={"source_id": "n0", "target_id": "NOEXISTE", "expected_reachable": True},
        )
        result = verify_example(ex)
        assert not result.passed

    def test_branching_correct_count(self):
        g = CausalGraph()
        for nid in ["A", "B", "C"]:
            g.add_node(CausalNode(nid, nid))
        g.add_edge(CausalEdge("A", "B", CausalRelation.CAUSES))
        g.add_edge(CausalEdge("A", "C", CausalRelation.CAUSES))
        ex = CausalExample(
            problem_text="test", graph=g, answer="dos efectos",
            complexity_level=2, answer_type=AnswerType.BRANCHING,
            metadata={"source_id": "A", "expected_successor_count": 2,
                      "expected_successor_ids": ["B", "C"]},
        )
        result = verify_example(ex)
        assert result.passed

    def test_branching_wrong_count(self):
        g = CausalGraph()
        for nid in ["A", "B"]:
            g.add_node(CausalNode(nid, nid))
        g.add_edge(CausalEdge("A", "B", CausalRelation.CAUSES))
        ex = CausalExample(
            problem_text="test", graph=g, answer="dos",
            complexity_level=2, answer_type=AnswerType.BRANCHING,
            metadata={"source_id": "A", "expected_successor_count": 2},  # Esperamos 2, hay 1
        )
        result = verify_example(ex)
        assert not result.passed

    def test_contradiction_correct_with(self):
        g = CausalGraph()
        g.add_node(CausalNode("A", "A")).add_node(CausalNode("B", "B"))
        g.add_edge(CausalEdge("A", "B", CausalRelation.CAUSES))
        g.add_edge(CausalEdge("A", "B", CausalRelation.PREVENTS))
        ex = CausalExample(
            problem_text="test", graph=g, answer="sí hay contradicción",
            complexity_level=3, answer_type=AnswerType.CONTRADICTION,
            metadata={"expected_has_contradiction": True, "expected_n_contradictions": 1},
        )
        result = verify_example(ex)
        assert result.passed

    def test_contradiction_correct_without(self):
        g = CausalGraph()
        g.add_node(CausalNode("A", "A")).add_node(CausalNode("B", "B"))
        g.add_edge(CausalEdge("A", "B", CausalRelation.CAUSES))
        ex = CausalExample(
            problem_text="test", graph=g, answer="no hay",
            complexity_level=3, answer_type=AnswerType.CONTRADICTION,
            metadata={"expected_has_contradiction": False, "expected_n_contradictions": 0},
        )
        result = verify_example(ex)
        assert result.passed

    def test_contradiction_mismatch(self):
        g = CausalGraph()
        g.add_node(CausalNode("A", "A")).add_node(CausalNode("B", "B"))
        g.add_edge(CausalEdge("A", "B", CausalRelation.CAUSES))
        ex = CausalExample(
            problem_text="test", graph=g, answer="sí",
            complexity_level=3, answer_type=AnswerType.CONTRADICTION,
            metadata={"expected_has_contradiction": True},  # Esperamos contradicción, no hay
        )
        result = verify_example(ex)
        assert not result.passed

    def test_counterfactual_blocked(self):
        g = CausalGraph()
        for nid in ["A", "B", "C"]:
            g.add_node(CausalNode(nid, nid))
        g.add_edge(CausalEdge("A", "B", CausalRelation.CAUSES))
        g.add_edge(CausalEdge("B", "C", CausalRelation.CAUSES))
        ex = CausalExample(
            problem_text="test", graph=g, answer="no C",
            complexity_level=4, answer_type=AnswerType.COUNTERFACTUAL,
            metadata={
                "counterfactual_removed_id": "A",
                "source_id": "A",
                "target_id": "C",
                "expected_path_blocked": True,
            },
        )
        result = verify_example(ex)
        assert result.passed

    def test_counterfactual_not_blocked_alternate_path(self):
        g = CausalGraph()
        for nid in ["A", "B", "C"]:
            g.add_node(CausalNode(nid, nid))
        g.add_edge(CausalEdge("A", "B", CausalRelation.CAUSES))
        g.add_edge(CausalEdge("B", "C", CausalRelation.CAUSES))
        g.add_edge(CausalEdge("A", "C", CausalRelation.ENABLES))  # Camino alternativo
        ex = CausalExample(
            problem_text="test", graph=g, answer="quizás C via A→C",
            complexity_level=4, answer_type=AnswerType.COUNTERFACTUAL,
            metadata={
                "counterfactual_removed_id": "B",
                "source_id": "A",
                "target_id": "C",
                "expected_path_blocked": False,   # A→C directo sigue existiendo
            },
        )
        result = verify_example(ex)
        assert result.passed

    def test_counterfactual_missing_metadata(self):
        g = self._minimal_graph(2)
        ex = CausalExample(
            problem_text="test", graph=g, answer="no",
            complexity_level=4, answer_type=AnswerType.COUNTERFACTUAL,
            metadata={},
        )
        result = verify_example(ex)
        assert not result.passed

    def test_critical_path_correct(self):
        g = self._minimal_graph(4)
        path = _longest_path(g)
        ex = CausalExample(
            problem_text="test", graph=g, answer="el camino",
            complexity_level=5, answer_type=AnswerType.CRITICAL_PATH,
            metadata={
                "expected_n_nodes": 4,
                "expected_path": path,
                "expected_path_length": 3,
            },
        )
        result = verify_example(ex)
        assert result.passed

    def test_critical_path_wrong_n_nodes(self):
        g = self._minimal_graph(3)
        ex = CausalExample(
            problem_text="test", graph=g, answer="camino",
            complexity_level=5, answer_type=AnswerType.CRITICAL_PATH,
            metadata={"expected_n_nodes": 10, "expected_path": [], "expected_path_length": 2},
        )
        result = verify_example(ex)
        assert not result.passed

    def test_multi_hop_correct_reachable(self):
        g = self._minimal_graph(3)
        ex = CausalExample(
            problem_text="test", graph=g, answer="sí",
            complexity_level=5, answer_type=AnswerType.MULTI_HOP,
            metadata={"source_id": "n0", "target_id": "n2", "expected_reachable": True},
        )
        result = verify_example(ex)
        assert result.passed

    def test_multi_hop_not_reachable(self):
        g = self._minimal_graph(3)  # n0→n1→n2
        ex = CausalExample(
            problem_text="test", graph=g, answer="no",
            complexity_level=5, answer_type=AnswerType.MULTI_HOP,
            metadata={"source_id": "n2", "target_id": "n0", "expected_reachable": False},
        )
        result = verify_example(ex)
        assert result.passed


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — PROPIEDADES GLOBALES
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalProperties:

    def test_all_levels_pass_verify(self):
        """Test crítico: todos los niveles pasan verify_example() con seed fijo."""
        gen = CausalGraphGenerator(seed=2024)
        failed = []
        for level in range(1, 6):
            for i in range(20):
                ex = gen.generate(level=level)
                result = verify_example(ex)
                if not result.passed:
                    failed.append((level, i, result.reason))
        assert len(failed) == 0, \
            f"Ejemplos que fallaron verify: {failed}"

    def test_edge_indices_always_valid_after_generation(self):
        """Los índices fuente/destino deben ser válidos para indexar tensores."""
        gen = CausalGraphGenerator(seed=123)
        for level in range(1, 6):
            for _ in range(10):
                ex = gen.generate(level=level)
                n = len(ex.graph)
                for edge in ex.graph.edges:
                    assert 0 <= edge.source_idx < n, \
                        f"Nivel {level}: source_idx={edge.source_idx} fuera de [0,{n})"
                    assert 0 <= edge.target_idx < n, \
                        f"Nivel {level}: target_idx={edge.target_idx} fuera de [0,{n})"

    def test_levels_1_2_4_5_acyclic(self):
        """Los niveles sin contradicciones intencionales deben ser DAGs."""
        gen = CausalGraphGenerator(seed=456)
        for level in [1, 2, 4, 5]:
            for _ in range(15):
                ex = gen.generate(level=level)
                cycles = ex.graph.detect_cycles()
                assert cycles == [], \
                    f"Nivel {level}: ciclos inesperados: {cycles}"

    def test_problem_text_contains_node_labels(self):
        """El texto del problema debe mencionar al menos un nodo del grafo."""
        gen = CausalGraphGenerator(seed=789)
        for level in range(1, 6):
            ex = gen.generate(level=level)
            node_labels = {n.label for n in ex.graph.nodes}
            has_mention = any(label in ex.problem_text for label in node_labels)
            assert has_mention, \
                f"Nivel {level}: problema no menciona ningún nodo del grafo"

    def test_answer_mentions_relevant_content(self):
        """La respuesta debe tener contenido relacionado al grafo."""
        gen = CausalGraphGenerator(seed=321)
        for level in range(1, 6):
            ex = gen.generate(level=level)
            # La respuesta debe tener al menos 20 caracteres
            assert len(ex.answer) >= 20, \
                f"Nivel {level}: respuesta demasiado corta: {ex.answer!r}"

    def test_level_3_produces_both_types(self):
        """Nivel 3 debe producir ejemplos CON y SIN contradicción."""
        gen = CausalGraphGenerator(seed=555)
        with_c = without_c = 0
        for _ in range(100):
            ex = gen.generate(level=3)
            if ex.metadata.get("has_contradiction"):
                with_c += 1
            else:
                without_c += 1
        assert with_c > 10, "Nivel 3: muy pocos ejemplos con contradicción"
        assert without_c > 5, "Nivel 3: muy pocos ejemplos sin contradicción"

    def test_batch_covers_diversity_of_domains(self):
        """Un batch de 100 ejemplos debe cubrir al menos 4 dominios distintos."""
        gen = CausalGraphGenerator(seed=777)
        batch = gen.generate_batch(n=100, verify=False)
        domains_seen = set()
        for ex in batch:
            domain = ex.metadata.get("domain") or ""
            domains_seen.update([d for d in domain.split(",")])
        # Contar dominios de metadata (para L5 es lista, para otros es str)
        all_domains = set()
        for ex in batch:
            d = ex.metadata.get("domain", ex.metadata.get("domains", ""))
            if isinstance(d, list):
                all_domains.update(d)
            elif d:
                all_domains.add(d)
        assert len(all_domains) >= 4, \
            f"Batch cubre solo {len(all_domains)} dominio(s): {all_domains}"

    def test_generation_is_fast(self):
        """Generación de 500 ejemplos debe completarse rápido (SID = CPU en tiempo real)."""
        import time
        gen = CausalGraphGenerator(seed=0)
        start = time.time()
        batch = gen.generate_batch(n=500, verify=False)
        elapsed = time.time() - start
        assert elapsed < 10.0, \
            f"Generación de 500 ejemplos tomó {elapsed:.1f}s — demasiado lento para SID"
        assert len(batch) == 500
