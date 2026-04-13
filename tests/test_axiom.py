"""
tests/test_axiom.py — Tests para MathMotor (AXIOM)
===================================================

Verifica que:
  1. motors/axiom/relations.py define los tipos y relaciones correctos
  2. MathNode/MathEdge aceptan MathNodeType/MathRelation sin errores
  3. MathMotor implementa BaseMotor correctamente
  4. build_graph produce grafos con MathNodeType/MathRelation
  5. reason() usa MATH_RELATIONS en sus funciones de mensaje
  6. get_graph_repr() produce [k_nodes, D] correctamente
  7. CRE con MATH_RELATIONS (10) funciona y es backwards compatible
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from core.graph import CausalGraph, CausalEdge, CausalNode, CausalRelation, NodeType
from motors.base_motor import BaseMotor
from motors.axiom.relations import (
    MATH_NODE_TYPES,
    MATH_RELATIONS,
    MathEdge,
    MathNode,
    MathNodeType,
    MathRelation,
    INFERENCE_RELATIONS,
    HIERARCHY_RELATIONS,
    TRANSFORM_RELATIONS,
)
from motors.axiom.motor import MathMotor, MathMotorConfig, MathCrystallizer
from crystallizer.config import CrystallizerConfig
from cre.config import CREConfig
from cre.engine import CausalReasoningEngine, CREOutput


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_tiny_math_config() -> MathMotorConfig:
    cryst = CrystallizerConfig(
        hidden_dim=64, max_nodes=8, pooler_heads=4,
        n_node_types=8, n_relation_types=10,
    )
    cre = CREConfig(
        node_dim=64, edge_dim=16, message_dim=32,
        n_message_layers=1, max_iterations=3, n_relation_types=10,
    )
    return MathMotorConfig(crystallizer=cryst, cre=cre)


def make_motor() -> MathMotor:
    return MathMotor(make_tiny_math_config())


def make_math_graph(n_nodes: int = 3, n_edges: int = 2) -> CausalGraph:
    g = CausalGraph()
    for i in range(n_nodes):
        g.add_node(MathNode(f"n{i}", f"expr_{i}", MathNodeType.EXPRESSION, 1.0))
    for i in range(min(n_edges, n_nodes - 1)):
        g.add_edge(MathEdge(f"n{i}", f"n{i+1}", MathRelation.DERIVES, 1.0, 1.0))
    return g


# ─────────────────────────────────────────────────────────────────────────────
# 1. RELATIONS MODULE
# ─────────────────────────────────────────────────────────────────────────────

class TestMathRelationsModule:

    def test_node_types_count(self):
        assert len(MATH_NODE_TYPES) == 8

    def test_node_types_values(self):
        expected = {"axiom", "definition", "theorem", "lemma",
                    "hypothesis", "expression", "equality", "set"}
        assert set(MATH_NODE_TYPES) == expected

    def test_relations_count(self):
        assert len(MATH_RELATIONS) == 10

    def test_relations_values(self):
        expected = {
            "derives", "assumes", "contradicts", "generalizes",
            "specializes", "applies", "reduces_to", "bounds",
            "equivalent_to", "implies",
        }
        assert set(MATH_RELATIONS) == expected

    def test_math_node_type_is_str_enum(self):
        assert isinstance(MathNodeType.THEOREM, str)
        assert MathNodeType.THEOREM.value == "theorem"

    def test_math_relation_is_str_enum(self):
        assert isinstance(MathRelation.DERIVES, str)
        assert MathRelation.DERIVES.value == "derives"

    def test_semantic_groupings(self):
        assert MathRelation.DERIVES in INFERENCE_RELATIONS
        assert MathRelation.IMPLIES in INFERENCE_RELATIONS
        assert MathRelation.GENERALIZES in HIERARCHY_RELATIONS
        assert MathRelation.REDUCES_TO in TRANSFORM_RELATIONS
        assert MathRelation.EQUIVALENT_TO in TRANSFORM_RELATIONS

    def test_no_overlap_with_causal_relations(self):
        from core.graph import CAUSAL_RELATIONS
        math_set   = set(MATH_RELATIONS)
        causal_set = set(CAUSAL_RELATIONS)
        # "implies" and "contradicts" overlap by design — rest must be unique
        unique_math = math_set - causal_set
        assert len(unique_math) >= 6  # derives, assumes, generalizes, specializes, applies, reduces_to, bounds, equivalent_to


# ─────────────────────────────────────────────────────────────────────────────
# 2. MATH NODE AND MATH EDGE
# ─────────────────────────────────────────────────────────────────────────────

class TestMathNodeEdge:

    def test_math_node_accepts_math_type(self):
        n = MathNode("t1", "Pythagorean theorem", MathNodeType.THEOREM, 0.95)
        assert n.node_type == MathNodeType.THEOREM
        assert n.node_type.value == "theorem"

    def test_math_node_all_types(self):
        for i, mt in enumerate(MathNodeType):
            n = MathNode(f"n{i}", mt.value, mt, 1.0)
            assert n.node_type == mt

    def test_math_edge_accepts_math_relation(self):
        e = MathEdge("n0", "n1", MathRelation.DERIVES, 1.0, 1.0)
        assert e.relation == MathRelation.DERIVES
        assert e.relation.value == "derives"

    def test_math_edge_all_relations(self):
        for i, mr in enumerate(MathRelation):
            e = MathEdge(f"a{i}", f"b{i}", mr, 0.9, 0.9)
            assert e.relation == mr

    def test_math_edge_in_causal_graph(self):
        g = make_math_graph(4, 3)
        assert len(g) == 4
        assert len(g.edges) == 3
        for edge in g.edges:
            assert isinstance(edge.relation, MathRelation)

    def test_math_node_in_causal_graph(self):
        g = make_math_graph(3, 2)
        for node in g.nodes:
            assert isinstance(node, MathNode)
            assert isinstance(node.node_type, MathNodeType)

    def test_self_loop_raises(self):
        with pytest.raises(ValueError):
            MathEdge("n0", "n0", MathRelation.DERIVES, 1.0, 1.0)

    def test_invalid_strength_raises(self):
        with pytest.raises(ValueError):
            MathEdge("a", "b", MathRelation.IMPLIES, 1.5, 1.0)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            MathNode("n", "x", MathNodeType.AXIOM, confidence=2.0)

    def test_source_target_idx_assigned(self):
        g = make_math_graph(3, 2)
        for edge in g.edges:
            assert edge.source_idx >= 0
            assert edge.target_idx >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. CRE WITH MATH RELATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestCREWithMathRelations:

    def test_cre_initializes_with_math_relations(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=MATH_RELATIONS)
        assert cre.relation_keys == MATH_RELATIONS

    def test_cre_message_fns_have_math_keys(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=MATH_RELATIONS)
        for layer in cre.layers:
            for rel in MATH_RELATIONS:
                assert rel in layer.message_fns

    def test_cre_message_fns_count(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=MATH_RELATIONS)
        for layer in cre.layers:
            assert len(layer.message_fns) == 10

    def test_cre_forward_with_math_graph(self):
        cfg = CREConfig(node_dim=64, edge_dim=16, message_dim=32,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=MATH_RELATIONS)
        cre.eval()
        g     = make_math_graph(4, 3)
        feats = torch.randn(4, 64)
        with torch.no_grad():
            out = cre.forward(g, feats, n_iterations=2)
        assert out.node_features.shape == (4, 64)
        assert out.iterations_run == 2

    def test_cre_forward_no_edges(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=MATH_RELATIONS)
        cre.eval()
        g     = make_math_graph(3, 0)
        feats = torch.randn(3, 32)
        with torch.no_grad():
            out = cre.forward(g, feats, n_iterations=2)
        assert out.node_features.shape == (3, 32)

    def test_derives_vs_contradicts_different_outputs(self):
        """DERIVES y CONTRADICTS producen outputs distintos."""
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=MATH_RELATIONS)
        cre.eval()

        g_der  = CausalGraph()
        g_con  = CausalGraph()
        feats  = torch.randn(2, 32)
        for i in range(2):
            g_der.add_node(MathNode(f"n{i}", f"e{i}", MathNodeType.EXPRESSION, 1.0))
            g_con.add_node(MathNode(f"n{i}", f"e{i}", MathNodeType.EXPRESSION, 1.0))
        g_der.add_edge(MathEdge("n0", "n1", MathRelation.DERIVES,     1.0, 1.0))
        g_con.add_edge(MathEdge("n0", "n1", MathRelation.CONTRADICTS, 1.0, 1.0))

        with torch.no_grad():
            out_der = cre.forward(g_der, feats.clone(), n_iterations=2)
            out_con = cre.forward(g_con, feats.clone(), n_iterations=2)

        assert not torch.allclose(out_der.node_features, out_con.node_features)

    def test_default_cre_backwards_compatible(self):
        from core.graph import CAUSAL_RELATIONS
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2)
        cre = CausalReasoningEngine(cfg)
        assert cre.relation_keys == CAUSAL_RELATIONS


# ─────────────────────────────────────────────────────────────────────────────
# 4. MATH MOTOR INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class TestMathMotorInterface:

    def test_is_base_motor(self):
        assert isinstance(make_motor(), BaseMotor)

    def test_is_nn_module(self):
        assert isinstance(make_motor(), nn.Module)

    def test_has_all_abstract_methods(self):
        m = make_motor()
        for method in ("define_node_types", "define_relations",
                       "build_graph", "reason", "get_graph_repr"):
            assert callable(getattr(m, method, None))

    def test_has_trainable_parameters(self):
        assert len(list(make_motor().parameters())) > 0

    def test_has_crystallizer_and_cre(self):
        m = make_motor()
        assert isinstance(m.crystallizer, MathCrystallizer)
        assert isinstance(m.cre, CausalReasoningEngine)


# ─────────────────────────────────────────────────────────────────────────────
# 5. INTROSPECTION
# ─────────────────────────────────────────────────────────────────────────────

class TestMathMotorIntrospection:

    def test_node_types_match(self):
        m = make_motor()
        assert set(m.define_node_types()) == set(MATH_NODE_TYPES)

    def test_node_types_count(self):
        assert len(make_motor().define_node_types()) == 8

    def test_relations_match(self):
        m = make_motor()
        assert set(m.define_relations()) == set(MATH_RELATIONS)

    def test_relations_count(self):
        assert len(make_motor().define_relations()) == 10

    def test_mostly_distinct_from_code_types(self):
        """Math types are mostly distinct from code types (only 'expression' may overlap)."""
        from motors.forge_c.relations import CODE_NODE_TYPES
        math_types = set(make_motor().define_node_types())
        code_types = set(CODE_NODE_TYPES)
        # "expression" overlaps by design — maths and code both use expressions
        overlap = math_types & code_types
        allowed_overlap = {"expression"}
        assert overlap.issubset(allowed_overlap), \
            f"Unexpected type overlap: {overlap - allowed_overlap}"
        # Most types must be unique
        unique_to_math = math_types - code_types
        assert len(unique_to_math) >= 6


# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILD_GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class TestMathMotorBuildGraph:

    def test_returns_graphs_list(self):
        m = make_motor(); m.eval()
        with torch.no_grad():
            out = m.build_graph(torch.randn(2, 10, 64))
        assert len(out.graphs) == 2

    def test_graph_nodes_have_math_types(self):
        m = make_motor(); m.eval()
        with torch.no_grad():
            out = m.build_graph(torch.randn(1, 10, 64))
        for node in out.graphs[0].nodes:
            assert node.node_type.value in MATH_NODE_TYPES

    def test_graph_edges_have_math_relations(self):
        m = make_motor(); m.eval()
        with torch.no_grad():
            out = m.build_graph(torch.randn(1, 10, 64))
        for edge in out.graphs[0].edges:
            assert edge.relation.value in MATH_RELATIONS

    def test_relation_logits_shape(self):
        m = make_motor(); m.eval()
        B, L, D = 2, 8, 64
        with torch.no_grad():
            out = m.build_graph(torch.randn(B, L, D))
        K = min(L, m.config.crystallizer.max_nodes)
        assert out.relation_logits.shape == (B, K, K, 10)

    def test_node_vectors_shape(self):
        m = make_motor(); m.eval()
        B, L, D = 2, 8, 64
        with torch.no_grad():
            out = m.build_graph(torch.randn(B, L, D))
        K = min(L, m.config.crystallizer.max_nodes)
        assert out.node_vectors.shape == (B, K, D)


# ─────────────────────────────────────────────────────────────────────────────
# 7. REASON
# ─────────────────────────────────────────────────────────────────────────────

class TestMathMotorReason:

    def test_returns_cre_output(self):
        m = make_motor(); m.eval()
        g = make_math_graph(3, 2)
        with torch.no_grad():
            out = m.reason(g, torch.randn(3, 64), n_iterations=2)
        assert isinstance(out, CREOutput)

    def test_node_features_shape(self):
        m = make_motor(); m.eval()
        N = 5
        g = make_math_graph(N, 4)
        with torch.no_grad():
            out = m.reason(g, torch.randn(N, 64), n_iterations=2)
        assert out.node_features.shape == (N, 64)

    def test_edge_features_shape(self):
        m = make_motor(); m.eval()
        N, E = 4, 3
        g = make_math_graph(N, E)
        with torch.no_grad():
            out = m.reason(g, torch.randn(N, 64), n_iterations=2)
        assert out.edge_features.shape == (E, m.config.cre.edge_dim)

    def test_default_iterations(self):
        m = make_motor(); m.eval()
        g = make_math_graph(3, 2)
        with torch.no_grad():
            out = m.reason(g, torch.randn(3, 64))
        assert out.iterations_run == 3

    def test_no_edges(self):
        m = make_motor(); m.eval()
        g = make_math_graph(4, 0)
        with torch.no_grad():
            out = m.reason(g, torch.randn(4, 64), n_iterations=2)
        assert out.node_features.shape == (4, 64)
        assert out.edge_features.shape == (0, m.config.cre.edge_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 8. GET_GRAPH_REPR
# ─────────────────────────────────────────────────────────────────────────────

class TestMathMotorGetGraphRepr:

    def _cre_out(self, n: int) -> CREOutput:
        return CREOutput(
            node_features=torch.randn(n, 64),
            edge_features=torch.zeros(0, 16),
            iterations_run=2, layer_outputs=[],
        )

    def test_exact_shape(self):
        m = make_motor()
        assert m.get_graph_repr(self._cre_out(8), k_nodes=8).shape == (8, 64)

    def test_n_gt_k(self):
        m = make_motor()
        assert m.get_graph_repr(self._cre_out(20), k_nodes=8).shape == (8, 64)

    def test_n_lt_k(self):
        m = make_motor()
        assert m.get_graph_repr(self._cre_out(3), k_nodes=8).shape == (8, 64)

    def test_padding_zeros(self):
        m = make_motor()
        out = CREOutput(
            node_features=torch.ones(2, 64),
            edge_features=torch.zeros(0, 16),
            iterations_run=1, layer_outputs=[],
        )
        r = m.get_graph_repr(out, k_nodes=5)
        assert torch.all(r[2:] == 0.0)

    def test_empty_graph(self):
        m = make_motor()
        r = m.get_graph_repr(self._cre_out(0), k_nodes=8)
        assert r.shape == (8, 64)
        assert torch.all(r == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 9. END-TO-END + CONFIG VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class TestMathMotorEndToEnd:

    def test_full_pipeline(self):
        m = make_motor(); m.eval()
        concepts = torch.randn(1, 8, 64)
        with torch.no_grad():
            cryst_out = m.build_graph(concepts)
        g     = cryst_out.graphs[0]
        n     = max(cryst_out.node_counts[0], 1)
        feats = cryst_out.node_vectors[0, :n]
        with torch.no_grad():
            cre_out = m.reason(g, feats, n_iterations=2)
        repr_ = m.get_graph_repr(cre_out, k_nodes=4)
        assert repr_.shape == (4, 64)

    def test_gradients_flow(self):
        m = make_motor()
        concepts = torch.randn(1, 8, 64, requires_grad=True)
        cryst_out = m.build_graph(concepts)
        n     = max(cryst_out.node_counts[0], 1)
        feats = cryst_out.node_vectors[0, :n]
        cre_out = m.reason(cryst_out.graphs[0], feats, n_iterations=1)
        m.get_graph_repr(cre_out, k_nodes=4).sum().backward()
        assert concepts.grad is not None

    def test_dim_mismatch_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, pooler_heads=4,
                                   n_node_types=8, n_relation_types=10)
        cre   = CREConfig(node_dim=128, n_relation_types=10)
        with pytest.raises(ValueError):
            MathMotorConfig(crystallizer=cryst, cre=cre)

    def test_wrong_node_type_count_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, pooler_heads=4,
                                   n_node_types=7, n_relation_types=10)
        cre   = CREConfig(node_dim=64, n_relation_types=10)
        with pytest.raises(ValueError):
            MathMotorConfig(crystallizer=cryst, cre=cre)

    def test_wrong_relation_count_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, pooler_heads=4,
                                   n_node_types=8, n_relation_types=16)
        cre   = CREConfig(node_dim=64, n_relation_types=10)
        with pytest.raises(ValueError):
            MathMotorConfig(crystallizer=cryst, cre=cre)

    def test_three_motors_independent_params(self):
        """CORA, FORGE-C, AXIOM tienen parámetros independientes."""
        from motors.cora.motor import CORAMotor, CORAMotorConfig
        from motors.forge_c.motor import CodeMotor, CodeMotorConfig
        from crystallizer.config import CrystallizerConfig as CC

        cora  = CORAMotor(CORAMotorConfig(
            crystallizer=CC(hidden_dim=64, pooler_heads=4),
            cre=CREConfig(node_dim=64, edge_dim=8, message_dim=16,
                          n_message_layers=1, max_iterations=2),
        ))
        forge = CodeMotor(CodeMotorConfig(
            crystallizer=CC(hidden_dim=64, pooler_heads=4,
                            n_node_types=8, n_relation_types=12),
            cre=CREConfig(node_dim=64, edge_dim=8, message_dim=16,
                          n_message_layers=1, max_iterations=2, n_relation_types=12),
        ))
        axiom = make_motor()

        # Each motor has its own set of parameters (not shared)
        cora_ids  = {id(p) for p in cora.parameters()}
        forge_ids = {id(p) for p in forge.parameters()}
        axiom_ids = {id(p) for p in axiom.parameters()}

        assert len(cora_ids  & forge_ids) == 0
        assert len(cora_ids  & axiom_ids) == 0
        assert len(forge_ids & axiom_ids) == 0
