"""
tests/test_empathy.py — Tests para SocialMotor (EMPATHY)
=========================================================

Verifica que:
  1. motors/empathy/relations.py define los tipos y relaciones correctos
  2. SocialNode/SocialEdge aceptan SocialNodeType/SocialRelation sin errores
  3. SocialMotor implementa BaseMotor correctamente
  4. build_graph produce grafos con SocialNodeType/SocialRelation
  5. reason() usa SOCIAL_RELATIONS en sus funciones de mensaje
  6. get_graph_repr() produce [k_nodes, D] correctamente
  7. CRE con SOCIAL_RELATIONS (10) funciona y es backwards compatible
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from core.graph import CausalGraph
from motors.base_motor import BaseMotor
from motors.empathy.relations import (
    SOCIAL_NODE_TYPES,
    SOCIAL_RELATIONS,
    SocialEdge,
    SocialNode,
    SocialNodeType,
    SocialRelation,
    MENTAL_STATE_RELATIONS,
    NORM_RELATIONS,
    INTERACTION_RELATIONS,
    CONFLICT_RELATIONS,
)
from motors.empathy.motor import SocialMotor, SocialMotorConfig, SocialCrystallizer
from crystallizer.config import CrystallizerConfig
from cre.config import CREConfig
from cre.engine import CausalReasoningEngine, CREOutput


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_tiny_social_config() -> SocialMotorConfig:
    cryst = CrystallizerConfig(
        hidden_dim=64, max_nodes=8, pooler_heads=4,
        n_node_types=8, n_relation_types=10,
    )
    cre = CREConfig(
        node_dim=64, edge_dim=16, message_dim=32,
        n_message_layers=1, max_iterations=3, n_relation_types=10,
    )
    return SocialMotorConfig(crystallizer=cryst, cre=cre)


def make_motor() -> SocialMotor:
    return SocialMotor(make_tiny_social_config())


def make_social_graph(n_nodes: int = 3, n_edges: int = 2) -> CausalGraph:
    g = CausalGraph()
    for i in range(n_nodes):
        g.add_node(SocialNode(f"n{i}", f"agent_{i}", SocialNodeType.PERSON, 1.0))
    for i in range(min(n_edges, n_nodes - 1)):
        g.add_edge(SocialEdge(f"n{i}", f"n{i+1}", SocialRelation.WANTS, 1.0, 1.0))
    return g


# ─────────────────────────────────────────────────────────────────────────────
# 1. RELATIONS MODULE
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialRelationsModule:

    def test_node_types_count(self):
        assert len(SOCIAL_NODE_TYPES) == 8

    def test_node_types_values(self):
        expected = {
            "person", "intention", "belief", "emotion",
            "norm", "context", "relationship", "expectation",
        }
        assert set(SOCIAL_NODE_TYPES) == expected

    def test_relations_count(self):
        assert len(SOCIAL_RELATIONS) == 10

    def test_relations_values(self):
        expected = {
            "wants", "believes", "feels", "expects",
            "violates_norm", "empathizes", "persuades",
            "trusts", "misunderstands", "reciprocates",
        }
        assert set(SOCIAL_RELATIONS) == expected

    def test_social_node_type_is_str_enum(self):
        assert isinstance(SocialNodeType.PERSON, str)
        assert SocialNodeType.PERSON.value == "person"

    def test_social_relation_is_str_enum(self):
        assert isinstance(SocialRelation.WANTS, str)
        assert SocialRelation.WANTS.value == "wants"

    def test_semantic_groupings_mental_state(self):
        assert SocialRelation.WANTS    in MENTAL_STATE_RELATIONS
        assert SocialRelation.BELIEVES in MENTAL_STATE_RELATIONS
        assert SocialRelation.FEELS    in MENTAL_STATE_RELATIONS
        assert SocialRelation.EXPECTS  in MENTAL_STATE_RELATIONS

    def test_semantic_groupings_norm(self):
        assert SocialRelation.VIOLATES_NORM in NORM_RELATIONS

    def test_semantic_groupings_interaction(self):
        assert SocialRelation.EMPATHIZES   in INTERACTION_RELATIONS
        assert SocialRelation.PERSUADES    in INTERACTION_RELATIONS
        assert SocialRelation.TRUSTS       in INTERACTION_RELATIONS
        assert SocialRelation.RECIPROCATES in INTERACTION_RELATIONS

    def test_semantic_groupings_conflict(self):
        assert SocialRelation.MISUNDERSTANDS in CONFLICT_RELATIONS
        assert SocialRelation.VIOLATES_NORM  in CONFLICT_RELATIONS

    def test_node_types_ordered_list(self):
        assert SOCIAL_NODE_TYPES[0] == "person"
        assert SOCIAL_NODE_TYPES[-1] == "expectation"

    def test_relations_ordered_list(self):
        assert SOCIAL_RELATIONS[0] == "wants"
        assert SOCIAL_RELATIONS[-1] == "reciprocates"

    def test_mostly_distinct_from_narrative_relations(self):
        from motors.muse.relations import NARRATIVE_RELATIONS
        soc_set  = set(SOCIAL_RELATIONS)
        narr_set = set(NARRATIVE_RELATIONS)
        unique_to_social = soc_set - narr_set
        assert len(unique_to_social) >= 7

    def test_mostly_distinct_from_math_relations(self):
        from motors.axiom.relations import MATH_RELATIONS
        soc_set  = set(SOCIAL_RELATIONS)
        math_set = set(MATH_RELATIONS)
        unique_to_social = soc_set - math_set
        assert len(unique_to_social) >= 8

    def test_mostly_distinct_from_code_relations(self):
        from motors.forge_c.relations import CODE_RELATIONS
        soc_set  = set(SOCIAL_RELATIONS)
        code_set = set(CODE_RELATIONS)
        unique_to_social = soc_set - code_set
        assert len(unique_to_social) >= 8


# ─────────────────────────────────────────────────────────────────────────────
# 2. SOCIAL NODE AND SOCIAL EDGE
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialNodeEdge:

    def test_social_node_accepts_social_type(self):
        n = SocialNode("p1", "Ana", SocialNodeType.PERSON, 0.95)
        assert n.node_type == SocialNodeType.PERSON
        assert n.node_type.value == "person"

    def test_social_node_all_types(self):
        for i, st in enumerate(SocialNodeType):
            n = SocialNode(f"n{i}", st.value, st, 1.0)
            assert n.node_type == st

    def test_social_edge_accepts_social_relation(self):
        e = SocialEdge("n0", "n1", SocialRelation.WANTS, 1.0, 1.0)
        assert e.relation == SocialRelation.WANTS
        assert e.relation.value == "wants"

    def test_social_edge_all_relations(self):
        for i, sr in enumerate(SocialRelation):
            e = SocialEdge(f"a{i}", f"b{i}", sr, 0.9, 0.9)
            assert e.relation == sr

    def test_social_edge_in_causal_graph(self):
        g = make_social_graph(4, 3)
        assert len(g) == 4
        assert len(g.edges) == 3
        for edge in g.edges:
            assert isinstance(edge.relation, SocialRelation)

    def test_social_node_in_causal_graph(self):
        g = make_social_graph(3, 2)
        for node in g.nodes:
            assert isinstance(node, SocialNode)
            assert isinstance(node.node_type, SocialNodeType)

    def test_self_loop_raises(self):
        with pytest.raises(ValueError):
            SocialEdge("n0", "n0", SocialRelation.WANTS, 1.0, 1.0)

    def test_invalid_strength_raises(self):
        with pytest.raises(ValueError):
            SocialEdge("a", "b", SocialRelation.TRUSTS, 1.5, 1.0)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            SocialNode("n", "x", SocialNodeType.NORM, confidence=2.0)

    def test_source_target_idx_assigned(self):
        g = make_social_graph(3, 2)
        for edge in g.edges:
            assert edge.source_idx >= 0
            assert edge.target_idx >= 0

    def test_mixed_social_types_in_graph(self):
        g = CausalGraph()
        per  = SocialNode("p1", "Ana",        SocialNodeType.PERSON,    1.0)
        bel  = SocialNode("b1", "cree X",     SocialNodeType.BELIEF,    0.9)
        norm = SocialNode("n1", "no mentir",  SocialNodeType.NORM,      1.0)
        emo  = SocialNode("e1", "frustración",SocialNodeType.EMOTION,   0.85)
        g.add_node(per).add_node(bel).add_node(norm).add_node(emo)
        g.add_edge(SocialEdge("p1", "b1", SocialRelation.BELIEVES,     1.0, 1.0))
        g.add_edge(SocialEdge("p1", "n1", SocialRelation.VIOLATES_NORM,0.8, 0.8))
        g.add_edge(SocialEdge("p1", "e1", SocialRelation.FEELS,        0.9, 0.9))
        assert len(g) == 4
        assert len(g.edges) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 3. CRE WITH SOCIAL RELATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestCREWithSocialRelations:

    def test_cre_initializes_with_social_relations(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=SOCIAL_RELATIONS)
        assert cre.relation_keys == SOCIAL_RELATIONS

    def test_cre_message_fns_have_social_keys(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=SOCIAL_RELATIONS)
        for layer in cre.layers:
            for rel in SOCIAL_RELATIONS:
                assert rel in layer.message_fns

    def test_cre_message_fns_count(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=SOCIAL_RELATIONS)
        for layer in cre.layers:
            assert len(layer.message_fns) == 10

    def test_cre_forward_with_social_graph(self):
        cfg = CREConfig(node_dim=64, edge_dim=16, message_dim=32,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=SOCIAL_RELATIONS)
        cre.eval()
        g     = make_social_graph(4, 3)
        feats = torch.randn(4, 64)
        with torch.no_grad():
            out = cre.forward(g, feats, n_iterations=2)
        assert out.node_features.shape == (4, 64)
        assert out.iterations_run == 2

    def test_cre_forward_no_edges(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=SOCIAL_RELATIONS)
        cre.eval()
        g     = make_social_graph(3, 0)
        feats = torch.randn(3, 32)
        with torch.no_grad():
            out = cre.forward(g, feats, n_iterations=2)
        assert out.node_features.shape == (3, 32)

    def test_wants_vs_misunderstands_different_outputs(self):
        """WANTS y MISUNDERSTANDS producen outputs distintos."""
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=SOCIAL_RELATIONS)
        cre.eval()
        g_w = CausalGraph()
        g_m = CausalGraph()
        feats = torch.randn(2, 32)
        for i in range(2):
            g_w.add_node(SocialNode(f"n{i}", f"p{i}", SocialNodeType.PERSON, 1.0))
            g_m.add_node(SocialNode(f"n{i}", f"p{i}", SocialNodeType.PERSON, 1.0))
        g_w.add_edge(SocialEdge("n0", "n1", SocialRelation.WANTS,         1.0, 1.0))
        g_m.add_edge(SocialEdge("n0", "n1", SocialRelation.MISUNDERSTANDS,1.0, 1.0))
        with torch.no_grad():
            out_w = cre.forward(g_w, feats.clone(), n_iterations=2)
            out_m = cre.forward(g_m, feats.clone(), n_iterations=2)
        assert not torch.allclose(out_w.node_features, out_m.node_features)

    def test_trusts_vs_violates_norm_different_outputs(self):
        """TRUSTS y VIOLATES_NORM producen outputs distintos."""
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=SOCIAL_RELATIONS)
        cre.eval()
        g_t = CausalGraph()
        g_v = CausalGraph()
        feats = torch.randn(2, 32)
        for i in range(2):
            g_t.add_node(SocialNode(f"n{i}", f"p{i}", SocialNodeType.PERSON, 1.0))
            g_v.add_node(SocialNode(f"n{i}", f"p{i}", SocialNodeType.PERSON, 1.0))
        g_t.add_edge(SocialEdge("n0", "n1", SocialRelation.TRUSTS,        1.0, 1.0))
        g_v.add_edge(SocialEdge("n0", "n1", SocialRelation.VIOLATES_NORM, 1.0, 1.0))
        with torch.no_grad():
            out_t = cre.forward(g_t, feats.clone(), n_iterations=2)
            out_v = cre.forward(g_v, feats.clone(), n_iterations=2)
        assert not torch.allclose(out_t.node_features, out_v.node_features)

    def test_default_cre_backwards_compatible(self):
        from core.graph import CAUSAL_RELATIONS
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2)
        cre = CausalReasoningEngine(cfg)
        assert cre.relation_keys == CAUSAL_RELATIONS


# ─────────────────────────────────────────────────────────────────────────────
# 4. SOCIAL MOTOR INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialMotorInterface:

    def test_is_base_motor(self):
        assert issubclass(SocialMotor, BaseMotor)

    def test_is_nn_module(self):
        assert issubclass(SocialMotor, nn.Module)

    def test_instantiates(self):
        motor = make_motor()
        assert isinstance(motor, SocialMotor)

    def test_has_crystallizer(self):
        motor = make_motor()
        assert isinstance(motor.crystallizer, SocialCrystallizer)

    def test_has_cre(self):
        motor = make_motor()
        assert isinstance(motor.cre, CausalReasoningEngine)

    def test_define_node_types(self):
        motor = make_motor()
        types = motor.define_node_types()
        assert set(types) == set(SOCIAL_NODE_TYPES)
        assert len(types) == 8

    def test_define_relations(self):
        motor = make_motor()
        rels = motor.define_relations()
        assert set(rels) == set(SOCIAL_RELATIONS)
        assert len(rels) == 10

    def test_cre_uses_social_relations(self):
        motor = make_motor()
        assert motor.cre.relation_keys == SOCIAL_RELATIONS

    def test_config_dim_mismatch_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, n_node_types=8, n_relation_types=10)
        cre   = CREConfig(node_dim=32, edge_dim=8, message_dim=16, n_relation_types=10)
        with pytest.raises(ValueError, match="hidden_dim"):
            SocialMotorConfig(crystallizer=cryst, cre=cre)

    def test_config_wrong_node_types_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, n_node_types=6, n_relation_types=10)
        cre   = CREConfig(node_dim=64, n_relation_types=10)
        with pytest.raises(ValueError):
            SocialMotorConfig(crystallizer=cryst, cre=cre)

    def test_config_wrong_relation_types_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, n_node_types=8, n_relation_types=8)
        cre   = CREConfig(node_dim=64, n_relation_types=8)
        with pytest.raises(ValueError):
            SocialMotorConfig(crystallizer=cryst, cre=cre)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SOCIAL MOTOR INTROSPECTION
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialMotorIntrospection:

    def test_has_parameters(self):
        motor = make_motor()
        assert len(list(motor.parameters())) > 0

    def test_node_type_count_equals_8(self):
        assert len(make_motor().define_node_types()) == 8

    def test_relation_count_equals_10(self):
        assert len(make_motor().define_relations()) == 10

    def test_message_fns_count_per_layer(self):
        motor = make_motor()
        for layer in motor.cre.layers:
            assert len(layer.message_fns) == 10

    def test_message_fns_keys_are_social(self):
        motor = make_motor()
        for layer in motor.cre.layers:
            assert set(layer.message_fns.keys()) == set(SOCIAL_RELATIONS)

    def test_default_config_dimensions(self):
        config = SocialMotorConfig()
        assert config.crystallizer.hidden_dim == config.cre.node_dim
        assert config.crystallizer.n_node_types == 8
        assert config.crystallizer.n_relation_types == 10


# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialMotorBuildGraph:

    def test_build_graph_output_type(self):
        from crystallizer.model import CrystallizerOutput
        motor = make_motor()
        motor.eval()
        with torch.no_grad():
            out = motor.build_graph(torch.randn(2, 16, 64))
        assert isinstance(out, CrystallizerOutput)

    def test_build_graph_returns_graphs(self):
        motor = make_motor()
        motor.eval()
        B = 3
        with torch.no_grad():
            out = motor.build_graph(torch.randn(B, 16, 64))
        assert len(out.graphs) == B

    def test_build_graph_nodes_are_social_nodes(self):
        motor = make_motor()
        motor.eval()
        with torch.no_grad():
            out = motor.build_graph(torch.randn(2, 16, 64))
        for graph in out.graphs:
            for node in graph.nodes:
                assert isinstance(node, SocialNode)

    def test_build_graph_edges_are_social_edges(self):
        motor = make_motor()
        motor.eval()
        with torch.no_grad():
            out = motor.build_graph(torch.randn(2, 16, 64))
        for graph in out.graphs:
            for edge in graph.edges:
                assert isinstance(edge, SocialEdge)

    def test_build_graph_node_types_are_social(self):
        motor = make_motor()
        motor.eval()
        with torch.no_grad():
            out = motor.build_graph(torch.randn(2, 16, 64))
        for graph in out.graphs:
            for node in graph.nodes:
                assert isinstance(node.node_type, SocialNodeType)

    def test_build_graph_edge_relations_are_social(self):
        motor = make_motor()
        motor.eval()
        with torch.no_grad():
            out = motor.build_graph(torch.randn(2, 16, 64))
        for graph in out.graphs:
            for edge in graph.edges:
                assert isinstance(edge.relation, SocialRelation)

    def test_build_graph_node_vectors_shape(self):
        motor = make_motor()
        motor.eval()
        B, L, D = 2, 16, 64
        with torch.no_grad():
            out = motor.build_graph(torch.randn(B, L, D))
        K = min(L, motor.config.crystallizer.max_nodes)
        assert out.node_vectors.shape == (B, K, D)


# ─────────────────────────────────────────────────────────────────────────────
# 7. REASON
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialMotorReason:

    def test_reason_returns_cre_output(self):
        motor = make_motor()
        motor.eval()
        g     = make_social_graph(4, 3)
        feats = torch.randn(4, 64)
        with torch.no_grad():
            out = motor.reason(g, feats, n_iterations=2)
        assert isinstance(out, CREOutput)

    def test_reason_output_shape(self):
        motor = make_motor()
        motor.eval()
        N, D = 5, 64
        g     = make_social_graph(N, N - 1)
        feats = torch.randn(N, D)
        with torch.no_grad():
            out = motor.reason(g, feats, n_iterations=2)
        assert out.node_features.shape == (N, D)

    def test_reason_iterations_run(self):
        motor = make_motor()
        motor.eval()
        g     = make_social_graph(3, 2)
        feats = torch.randn(3, 64)
        with torch.no_grad():
            out = motor.reason(g, feats, n_iterations=3)
        assert out.iterations_run == 3

    def test_reason_modifies_features(self):
        motor = make_motor()
        motor.eval()
        g     = make_social_graph(4, 3)
        feats = torch.randn(4, 64)
        with torch.no_grad():
            out = motor.reason(g, feats.clone(), n_iterations=2)
        assert not torch.allclose(feats, out.node_features)

    def test_reason_no_edges(self):
        motor = make_motor()
        motor.eval()
        g     = make_social_graph(3, 0)
        feats = torch.randn(3, 64)
        with torch.no_grad():
            out = motor.reason(g, feats, n_iterations=2)
        assert out.node_features.shape == (3, 64)


# ─────────────────────────────────────────────────────────────────────────────
# 8. GET GRAPH REPR
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialMotorGetGraphRepr:

    def _cre_output(self, n: int, D: int) -> CREOutput:
        motor = make_motor()
        motor.eval()
        g = make_social_graph(n, n - 1)
        with torch.no_grad():
            return motor.reason(g, torch.randn(n, D), n_iterations=2)

    def test_repr_shape_exact(self):
        motor = make_motor()
        motor.eval()
        out = self._cre_output(5, 64)
        assert motor.get_graph_repr(out, k_nodes=5).shape == (5, 64)

    def test_repr_shape_pad(self):
        motor = make_motor()
        motor.eval()
        out = self._cre_output(3, 64)
        assert motor.get_graph_repr(out, k_nodes=8).shape == (8, 64)

    def test_repr_shape_truncate(self):
        motor = make_motor()
        motor.eval()
        out = self._cre_output(8, 64)
        assert motor.get_graph_repr(out, k_nodes=4).shape == (4, 64)

    def test_repr_zeros_for_empty(self):
        motor = make_motor()
        motor.eval()
        empty = CREOutput(
            node_features=torch.zeros(0, 64),
            edge_features=torch.zeros(0, 16),
            iterations_run=0,
            layer_outputs=[],
        )
        out = motor.get_graph_repr(empty, k_nodes=4)
        assert out.shape == (4, 64)
        assert out.sum() == 0.0

    def test_repr_is_differentiable(self):
        motor = make_motor()
        g     = make_social_graph(4, 3)
        feats = torch.randn(4, 64, requires_grad=True)
        out   = motor.reason(g, feats, n_iterations=2)
        repr_ = motor.get_graph_repr(out, k_nodes=3)
        repr_.sum().backward()
        assert feats.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# 9. END-TO-END
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialMotorEndToEnd:

    def test_full_pipeline_single_batch(self):
        motor = make_motor()
        motor.eval()
        B, L, D = 1, 16, 64
        concepts = torch.randn(B, L, D)
        with torch.no_grad():
            cryst_out = motor.build_graph(concepts)
        graph = cryst_out.graphs[0]
        n_nodes = len(graph)
        if n_nodes == 0:
            return
        node_feats = cryst_out.node_vectors[0, :n_nodes]
        with torch.no_grad():
            cre_out = motor.reason(graph, node_feats, n_iterations=2)
        repr_ = motor.get_graph_repr(cre_out, k_nodes=4)
        assert repr_.shape == (4, D)

    def test_full_pipeline_batch(self):
        motor = make_motor()
        motor.eval()
        with torch.no_grad():
            out = motor.build_graph(torch.randn(4, 16, 64))
        assert len(out.graphs) == 4

    def test_pipeline_gradient_flows(self):
        motor = make_motor()
        concepts = torch.randn(1, 16, 64, requires_grad=True)
        cryst_out = motor.build_graph(concepts)
        graph = cryst_out.graphs[0]
        n_nodes = len(graph)
        if n_nodes == 0:
            return
        node_feats = cryst_out.node_vectors[0, :n_nodes]
        cre_out = motor.reason(graph, node_feats, n_iterations=1)
        motor.get_graph_repr(cre_out, k_nodes=4).sum().backward()
        assert concepts.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# 10. BACKWARDS COMPATIBILITY
# ─────────────────────────────────────────────────────────────────────────────

class TestBackwardsCompatibility:

    def test_default_cre_not_affected_by_empathy(self):
        from core.graph import CAUSAL_RELATIONS
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16, n_message_layers=1)
        cre = CausalReasoningEngine(cfg)
        assert cre.relation_keys == CAUSAL_RELATIONS

    def test_empathy_cre_does_not_affect_cora(self):
        from motors.cora.motor import CORAMotor, CORAMotorConfig
        from core.graph import CAUSAL_RELATIONS
        cora = CORAMotor(CORAMotorConfig())
        assert cora.cre.relation_keys == CAUSAL_RELATIONS

    def test_all_motors_have_different_relation_keys(self):
        from motors.axiom.relations import MATH_RELATIONS
        from motors.muse.relations import NARRATIVE_RELATIONS
        from motors.forge_c.relations import CODE_RELATIONS
        keys = [
            set(SOCIAL_RELATIONS),
            set(MATH_RELATIONS),
            set(NARRATIVE_RELATIONS),
            set(CODE_RELATIONS),
        ]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                # Each pair should have at least some unique relations
                assert keys[i] != keys[j]

    def test_more_relations_more_params(self):
        cfg_10 = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                           n_message_layers=1, n_relation_types=10)
        cfg_5  = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                           n_message_layers=1, n_relation_types=5)
        cre_10 = CausalReasoningEngine(cfg_10, relation_keys=SOCIAL_RELATIONS)
        cre_5  = CausalReasoningEngine(cfg_5,  relation_keys=SOCIAL_RELATIONS[:5])
        params_10 = sum(p.numel() for p in cre_10.parameters())
        params_5  = sum(p.numel() for p in cre_5.parameters())
        assert params_10 > params_5
