"""
tests/test_muse.py — Tests para CreativeMotor (MUSE)
=====================================================

Verifica que:
  1. motors/muse/relations.py define los tipos y relaciones correctos
  2. NarrativeNode/NarrativeEdge aceptan NarrativeNodeType/NarrativeRelation
  3. CreativeMotor implementa BaseMotor correctamente
  4. build_graph produce grafos con NarrativeNodeType/NarrativeRelation
  5. reason() usa NARRATIVE_RELATIONS en sus funciones de mensaje
  6. get_graph_repr() produce [k_nodes, D] correctamente
  7. CRE con NARRATIVE_RELATIONS (10) funciona y es backwards compatible
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from core.graph import CausalGraph, CausalEdge, CausalNode, CausalRelation, NodeType
from motors.base_motor import BaseMotor
from motors.muse.relations import (
    NARRATIVE_NODE_TYPES,
    NARRATIVE_RELATIONS,
    NarrativeEdge,
    NarrativeNode,
    NarrativeNodeType,
    NarrativeRelation,
    MOTIVATION_RELATIONS,
    TENSION_RELATIONS,
    RESOLUTION_RELATIONS,
    SYMBOLIC_RELATIONS,
    CONTRAST_RELATIONS,
)
from motors.muse.motor import CreativeMotor, CreativeMotorConfig, NarrativeCrystallizer
from crystallizer.config import CrystallizerConfig
from cre.config import CREConfig
from cre.engine import CausalReasoningEngine, CREOutput


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_tiny_creative_config() -> CreativeMotorConfig:
    cryst = CrystallizerConfig(
        hidden_dim=64, max_nodes=8, pooler_heads=4,
        n_node_types=8, n_relation_types=10,
    )
    cre = CREConfig(
        node_dim=64, edge_dim=16, message_dim=32,
        n_message_layers=1, max_iterations=3, n_relation_types=10,
    )
    return CreativeMotorConfig(crystallizer=cryst, cre=cre)


def make_motor() -> CreativeMotor:
    return CreativeMotor(make_tiny_creative_config())


def make_narrative_graph(n_nodes: int = 3, n_edges: int = 2) -> CausalGraph:
    g = CausalGraph()
    for i in range(n_nodes):
        g.add_node(NarrativeNode(f"n{i}", f"elem_{i}", NarrativeNodeType.EVENT, 1.0))
    for i in range(min(n_edges, n_nodes - 1)):
        g.add_edge(NarrativeEdge(f"n{i}", f"n{i+1}", NarrativeRelation.MOTIVATES, 1.0, 1.0))
    return g


# ─────────────────────────────────────────────────────────────────────────────
# 1. RELATIONS MODULE
# ─────────────────────────────────────────────────────────────────────────────

class TestNarrativeRelationsModule:

    def test_node_types_count(self):
        assert len(NARRATIVE_NODE_TYPES) == 8

    def test_node_types_values(self):
        expected = {
            "character", "event", "emotion", "theme",
            "symbol", "setting", "conflict", "resolution",
        }
        assert set(NARRATIVE_NODE_TYPES) == expected

    def test_relations_count(self):
        assert len(NARRATIVE_RELATIONS) == 10

    def test_relations_values(self):
        expected = {
            "motivates", "conflicts_with", "develops_into", "symbolizes",
            "parallels", "contrasts", "foreshadows", "resolves",
            "intensifies", "subverts",
        }
        assert set(NARRATIVE_RELATIONS) == expected

    def test_narrative_node_type_is_str_enum(self):
        assert isinstance(NarrativeNodeType.CHARACTER, str)
        assert NarrativeNodeType.CHARACTER.value == "character"

    def test_narrative_relation_is_str_enum(self):
        assert isinstance(NarrativeRelation.MOTIVATES, str)
        assert NarrativeRelation.MOTIVATES.value == "motivates"

    def test_semantic_groupings_motivation(self):
        assert NarrativeRelation.MOTIVATES in MOTIVATION_RELATIONS
        assert NarrativeRelation.DEVELOPS_INTO in MOTIVATION_RELATIONS

    def test_semantic_groupings_tension(self):
        assert NarrativeRelation.CONFLICTS_WITH in TENSION_RELATIONS
        assert NarrativeRelation.INTENSIFIES in TENSION_RELATIONS

    def test_semantic_groupings_resolution(self):
        assert NarrativeRelation.RESOLVES in RESOLUTION_RELATIONS

    def test_semantic_groupings_symbolic(self):
        assert NarrativeRelation.SYMBOLIZES in SYMBOLIC_RELATIONS
        assert NarrativeRelation.PARALLELS in SYMBOLIC_RELATIONS

    def test_semantic_groupings_contrast(self):
        assert NarrativeRelation.CONTRASTS in CONTRAST_RELATIONS
        assert NarrativeRelation.FORESHADOWS in CONTRAST_RELATIONS
        assert NarrativeRelation.SUBVERTS in CONTRAST_RELATIONS

    def test_node_types_ordered_list(self):
        assert NARRATIVE_NODE_TYPES[0] == "character"
        assert NARRATIVE_NODE_TYPES[-1] == "resolution"

    def test_relations_ordered_list(self):
        assert NARRATIVE_RELATIONS[0] == "motivates"
        assert NARRATIVE_RELATIONS[-1] == "subverts"

    def test_mostly_distinct_from_math_relations(self):
        from motors.axiom.relations import MATH_RELATIONS
        narr_set = set(NARRATIVE_RELATIONS)
        math_set = set(MATH_RELATIONS)
        unique_to_narr = narr_set - math_set
        assert len(unique_to_narr) >= 8  # narrative relations are mostly unique

    def test_mostly_distinct_from_code_relations(self):
        from motors.forge_c.relations import CODE_RELATIONS
        narr_set = set(NARRATIVE_RELATIONS)
        code_set = set(CODE_RELATIONS)
        unique_to_narr = narr_set - code_set
        assert len(unique_to_narr) >= 8


# ─────────────────────────────────────────────────────────────────────────────
# 2. NARRATIVE NODE AND NARRATIVE EDGE
# ─────────────────────────────────────────────────────────────────────────────

class TestNarrativeNodeEdge:

    def test_narrative_node_accepts_narrative_type(self):
        n = NarrativeNode("c1", "Elena la protagonista", NarrativeNodeType.CHARACTER, 0.95)
        assert n.node_type == NarrativeNodeType.CHARACTER
        assert n.node_type.value == "character"

    def test_narrative_node_all_types(self):
        for i, nt in enumerate(NarrativeNodeType):
            n = NarrativeNode(f"n{i}", nt.value, nt, 1.0)
            assert n.node_type == nt

    def test_narrative_edge_accepts_narrative_relation(self):
        e = NarrativeEdge("n0", "n1", NarrativeRelation.MOTIVATES, 1.0, 1.0)
        assert e.relation == NarrativeRelation.MOTIVATES
        assert e.relation.value == "motivates"

    def test_narrative_edge_all_relations(self):
        for i, nr in enumerate(NarrativeRelation):
            e = NarrativeEdge(f"a{i}", f"b{i}", nr, 0.9, 0.9)
            assert e.relation == nr

    def test_narrative_edge_in_causal_graph(self):
        g = make_narrative_graph(4, 3)
        assert len(g) == 4
        assert len(g.edges) == 3
        for edge in g.edges:
            assert isinstance(edge.relation, NarrativeRelation)

    def test_narrative_node_in_causal_graph(self):
        g = make_narrative_graph(3, 2)
        for node in g.nodes:
            assert isinstance(node, NarrativeNode)
            assert isinstance(node.node_type, NarrativeNodeType)

    def test_self_loop_raises(self):
        with pytest.raises(ValueError):
            NarrativeEdge("n0", "n0", NarrativeRelation.MOTIVATES, 1.0, 1.0)

    def test_invalid_strength_raises(self):
        with pytest.raises(ValueError):
            NarrativeEdge("a", "b", NarrativeRelation.RESOLVES, 1.5, 1.0)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            NarrativeNode("n", "x", NarrativeNodeType.CONFLICT, confidence=2.0)

    def test_source_target_idx_assigned(self):
        g = make_narrative_graph(3, 2)
        for edge in g.edges:
            assert edge.source_idx >= 0
            assert edge.target_idx >= 0

    def test_mixed_narrative_types_in_graph(self):
        g = CausalGraph()
        char = NarrativeNode("c1", "protagonist",  NarrativeNodeType.CHARACTER,  1.0)
        ev   = NarrativeNode("e1", "the crisis",   NarrativeNodeType.EVENT,      0.9)
        conf = NarrativeNode("cf", "the conflict", NarrativeNodeType.CONFLICT,   0.85)
        res  = NarrativeNode("r1", "resolution",   NarrativeNodeType.RESOLUTION, 1.0)
        g.add_node(char).add_node(ev).add_node(conf).add_node(res)
        g.add_edge(NarrativeEdge("c1", "cf", NarrativeRelation.MOTIVATES,     1.0, 1.0))
        g.add_edge(NarrativeEdge("cf", "e1", NarrativeRelation.INTENSIFIES,   0.8, 0.8))
        g.add_edge(NarrativeEdge("r1", "cf", NarrativeRelation.RESOLVES,      1.0, 1.0))
        assert len(g) == 4
        assert len(g.edges) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 3. CRE WITH NARRATIVE RELATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestCREWithNarrativeRelations:

    def test_cre_initializes_with_narrative_relations(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=NARRATIVE_RELATIONS)
        assert cre.relation_keys == NARRATIVE_RELATIONS

    def test_cre_message_fns_have_narrative_keys(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=NARRATIVE_RELATIONS)
        for layer in cre.layers:
            for rel in NARRATIVE_RELATIONS:
                assert rel in layer.message_fns

    def test_cre_message_fns_count(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=NARRATIVE_RELATIONS)
        for layer in cre.layers:
            assert len(layer.message_fns) == 10

    def test_cre_forward_with_narrative_graph(self):
        cfg = CREConfig(node_dim=64, edge_dim=16, message_dim=32,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=NARRATIVE_RELATIONS)
        cre.eval()
        g     = make_narrative_graph(4, 3)
        feats = torch.randn(4, 64)
        with torch.no_grad():
            out = cre.forward(g, feats, n_iterations=2)
        assert out.node_features.shape == (4, 64)
        assert out.iterations_run == 2

    def test_cre_forward_no_edges(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=NARRATIVE_RELATIONS)
        cre.eval()
        g     = make_narrative_graph(3, 0)
        feats = torch.randn(3, 32)
        with torch.no_grad():
            out = cre.forward(g, feats, n_iterations=2)
        assert out.node_features.shape == (3, 32)

    def test_motivates_vs_subverts_different_outputs(self):
        """MOTIVATES y SUBVERTS producen outputs distintos."""
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=NARRATIVE_RELATIONS)
        cre.eval()

        g_mot = CausalGraph()
        g_sub = CausalGraph()
        feats = torch.randn(2, 32)
        for i in range(2):
            g_mot.add_node(NarrativeNode(f"n{i}", f"e{i}", NarrativeNodeType.EVENT, 1.0))
            g_sub.add_node(NarrativeNode(f"n{i}", f"e{i}", NarrativeNodeType.EVENT, 1.0))
        g_mot.add_edge(NarrativeEdge("n0", "n1", NarrativeRelation.MOTIVATES, 1.0, 1.0))
        g_sub.add_edge(NarrativeEdge("n0", "n1", NarrativeRelation.SUBVERTS,  1.0, 1.0))

        with torch.no_grad():
            out_mot = cre.forward(g_mot, feats.clone(), n_iterations=2)
            out_sub = cre.forward(g_sub, feats.clone(), n_iterations=2)

        assert not torch.allclose(out_mot.node_features, out_sub.node_features)

    def test_resolves_vs_intensifies_different_outputs(self):
        """RESOLVES e INTENSIFIES producen outputs distintos."""
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=10)
        cre = CausalReasoningEngine(cfg, relation_keys=NARRATIVE_RELATIONS)
        cre.eval()

        g_res = CausalGraph()
        g_int = CausalGraph()
        feats = torch.randn(2, 32)
        for i in range(2):
            g_res.add_node(NarrativeNode(f"n{i}", f"e{i}", NarrativeNodeType.EVENT, 1.0))
            g_int.add_node(NarrativeNode(f"n{i}", f"e{i}", NarrativeNodeType.EVENT, 1.0))
        g_res.add_edge(NarrativeEdge("n0", "n1", NarrativeRelation.RESOLVES,   1.0, 1.0))
        g_int.add_edge(NarrativeEdge("n0", "n1", NarrativeRelation.INTENSIFIES,1.0, 1.0))

        with torch.no_grad():
            out_res = cre.forward(g_res, feats.clone(), n_iterations=2)
            out_int = cre.forward(g_int, feats.clone(), n_iterations=2)

        assert not torch.allclose(out_res.node_features, out_int.node_features)

    def test_default_cre_backwards_compatible(self):
        from core.graph import CAUSAL_RELATIONS
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2)
        cre = CausalReasoningEngine(cfg)
        assert cre.relation_keys == CAUSAL_RELATIONS


# ─────────────────────────────────────────────────────────────────────────────
# 4. CREATIVE MOTOR INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class TestCreativeMotorInterface:

    def test_is_base_motor(self):
        assert issubclass(CreativeMotor, BaseMotor)

    def test_is_nn_module(self):
        assert issubclass(CreativeMotor, nn.Module)

    def test_instantiates(self):
        motor = make_motor()
        assert isinstance(motor, CreativeMotor)

    def test_has_crystallizer(self):
        motor = make_motor()
        assert isinstance(motor.crystallizer, NarrativeCrystallizer)

    def test_has_cre(self):
        motor = make_motor()
        assert isinstance(motor.cre, CausalReasoningEngine)

    def test_define_node_types(self):
        motor = make_motor()
        types = motor.define_node_types()
        assert set(types) == set(NARRATIVE_NODE_TYPES)
        assert len(types) == 8

    def test_define_relations(self):
        motor = make_motor()
        rels = motor.define_relations()
        assert set(rels) == set(NARRATIVE_RELATIONS)
        assert len(rels) == 10

    def test_cre_uses_narrative_relations(self):
        motor = make_motor()
        assert motor.cre.relation_keys == NARRATIVE_RELATIONS

    def test_config_dim_mismatch_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, n_node_types=8, n_relation_types=10)
        cre   = CREConfig(node_dim=32, edge_dim=8, message_dim=16, n_relation_types=10)
        with pytest.raises(ValueError, match="hidden_dim"):
            CreativeMotorConfig(crystallizer=cryst, cre=cre)

    def test_config_wrong_node_types_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, n_node_types=6, n_relation_types=10)
        cre   = CREConfig(node_dim=64, n_relation_types=10)
        with pytest.raises(ValueError):
            CreativeMotorConfig(crystallizer=cryst, cre=cre)

    def test_config_wrong_relation_types_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, n_node_types=8, n_relation_types=8)
        cre   = CREConfig(node_dim=64, n_relation_types=8)
        with pytest.raises(ValueError):
            CreativeMotorConfig(crystallizer=cryst, cre=cre)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CREATIVE MOTOR INTROSPECTION
# ─────────────────────────────────────────────────────────────────────────────

class TestCreativeMotorIntrospection:

    def test_has_parameters(self):
        motor = make_motor()
        params = list(motor.parameters())
        assert len(params) > 0

    def test_node_type_count_equals_8(self):
        motor = make_motor()
        assert len(motor.define_node_types()) == 8

    def test_relation_count_equals_10(self):
        motor = make_motor()
        assert len(motor.define_relations()) == 10

    def test_message_fns_count_per_layer(self):
        motor = make_motor()
        for layer in motor.cre.layers:
            assert len(layer.message_fns) == 10

    def test_message_fns_keys_are_narrative(self):
        motor = make_motor()
        for layer in motor.cre.layers:
            keys = set(layer.message_fns.keys())
            assert keys == set(NARRATIVE_RELATIONS)

    def test_default_config_dimensions(self):
        config = CreativeMotorConfig()
        assert config.crystallizer.hidden_dim == config.cre.node_dim
        assert config.crystallizer.n_node_types == 8
        assert config.crystallizer.n_relation_types == 10


# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class TestCreativeMotorBuildGraph:

    def test_build_graph_output_type(self):
        motor = make_motor()
        motor.eval()
        concepts = torch.randn(2, 16, 64)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        from crystallizer.model import CrystallizerOutput
        assert isinstance(out, CrystallizerOutput)

    def test_build_graph_returns_graphs(self):
        motor = make_motor()
        motor.eval()
        B = 3
        concepts = torch.randn(B, 16, 64)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        assert len(out.graphs) == B

    def test_build_graph_nodes_are_narrative_nodes(self):
        motor = make_motor()
        motor.eval()
        concepts = torch.randn(2, 16, 64)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        for graph in out.graphs:
            for node in graph.nodes:
                assert isinstance(node, NarrativeNode)

    def test_build_graph_edges_are_narrative_edges(self):
        motor = make_motor()
        motor.eval()
        concepts = torch.randn(2, 16, 64)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        for graph in out.graphs:
            for edge in graph.edges:
                assert isinstance(edge, NarrativeEdge)

    def test_build_graph_node_types_are_narrative(self):
        motor = make_motor()
        motor.eval()
        concepts = torch.randn(2, 16, 64)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        for graph in out.graphs:
            for node in graph.nodes:
                assert isinstance(node.node_type, NarrativeNodeType)

    def test_build_graph_edge_relations_are_narrative(self):
        motor = make_motor()
        motor.eval()
        concepts = torch.randn(2, 16, 64)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        for graph in out.graphs:
            for edge in graph.edges:
                assert isinstance(edge.relation, NarrativeRelation)

    def test_build_graph_node_vectors_shape(self):
        motor = make_motor()
        motor.eval()
        B, L, D = 2, 16, 64
        concepts = torch.randn(B, L, D)
        with torch.no_grad():
            out = motor.build_graph(concepts)
        K = min(L, motor.config.crystallizer.max_nodes)
        assert out.node_vectors.shape == (B, K, D)


# ─────────────────────────────────────────────────────────────────────────────
# 7. REASON
# ─────────────────────────────────────────────────────────────────────────────

class TestCreativeMotorReason:

    def test_reason_returns_cre_output(self):
        motor = make_motor()
        motor.eval()
        g     = make_narrative_graph(4, 3)
        feats = torch.randn(4, 64)
        with torch.no_grad():
            out = motor.reason(g, feats, n_iterations=2)
        assert isinstance(out, CREOutput)

    def test_reason_output_shape(self):
        motor = make_motor()
        motor.eval()
        N, D = 5, 64
        g     = make_narrative_graph(N, N - 1)
        feats = torch.randn(N, D)
        with torch.no_grad():
            out = motor.reason(g, feats, n_iterations=2)
        assert out.node_features.shape == (N, D)

    def test_reason_iterations_run(self):
        motor = make_motor()
        motor.eval()
        g     = make_narrative_graph(3, 2)
        feats = torch.randn(3, 64)
        with torch.no_grad():
            out = motor.reason(g, feats, n_iterations=3)
        assert out.iterations_run == 3

    def test_reason_modifies_features(self):
        motor = make_motor()
        motor.eval()
        g     = make_narrative_graph(4, 3)
        feats = torch.randn(4, 64)
        with torch.no_grad():
            out = motor.reason(g, feats.clone(), n_iterations=2)
        # Features should be different after message passing (graph has edges)
        assert not torch.allclose(feats, out.node_features)

    def test_reason_no_edges(self):
        motor = make_motor()
        motor.eval()
        g     = make_narrative_graph(3, 0)
        feats = torch.randn(3, 64)
        with torch.no_grad():
            out = motor.reason(g, feats, n_iterations=2)
        assert out.node_features.shape == (3, 64)


# ─────────────────────────────────────────────────────────────────────────────
# 8. GET GRAPH REPR
# ─────────────────────────────────────────────────────────────────────────────

class TestCreativeMotorGetGraphRepr:

    def _get_cre_output(self, n_nodes: int, D: int) -> CREOutput:
        motor = make_motor()
        motor.eval()
        g     = make_narrative_graph(n_nodes, n_nodes - 1)
        feats = torch.randn(n_nodes, D)
        with torch.no_grad():
            return motor.reason(g, feats, n_iterations=2)

    def test_repr_shape_exact(self):
        motor = make_motor()
        motor.eval()
        out = self._get_cre_output(5, 64)
        repr_ = motor.get_graph_repr(out, k_nodes=5)
        assert repr_.shape == (5, 64)

    def test_repr_shape_pad_when_fewer_nodes(self):
        motor = make_motor()
        motor.eval()
        out = self._get_cre_output(3, 64)
        repr_ = motor.get_graph_repr(out, k_nodes=8)
        assert repr_.shape == (8, 64)

    def test_repr_shape_truncate_when_more_nodes(self):
        motor = make_motor()
        motor.eval()
        out = self._get_cre_output(8, 64)
        repr_ = motor.get_graph_repr(out, k_nodes=4)
        assert repr_.shape == (4, 64)

    def test_repr_zeros_for_empty(self):
        motor = make_motor()
        motor.eval()
        empty_out = CREOutput(
            node_features=torch.zeros(0, 64),
            edge_features=torch.zeros(0, 16),
            iterations_run=0,
            layer_outputs=[],
        )
        repr_ = motor.get_graph_repr(empty_out, k_nodes=4)
        assert repr_.shape == (4, 64)
        assert repr_.sum() == 0.0

    def test_repr_is_differentiable(self):
        motor = make_motor()
        g     = make_narrative_graph(4, 3)
        feats = torch.randn(4, 64, requires_grad=True)
        out   = motor.reason(g, feats, n_iterations=2)
        repr_ = motor.get_graph_repr(out, k_nodes=3)
        loss  = repr_.sum()
        loss.backward()
        assert feats.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# 9. END-TO-END
# ─────────────────────────────────────────────────────────────────────────────

class TestCreativeMotorEndToEnd:

    def test_full_pipeline_single_batch(self):
        motor = make_motor()
        motor.eval()
        B, L, D = 1, 16, 64
        concepts = torch.randn(B, L, D)
        with torch.no_grad():
            cryst_out = motor.build_graph(concepts)
        graph   = cryst_out.graphs[0]
        n_nodes = len(graph)
        if n_nodes == 0:
            return  # valid: low threshold, no nodes detected
        node_feats = cryst_out.node_vectors[0, :n_nodes]
        with torch.no_grad():
            cre_out = motor.reason(graph, node_feats, n_iterations=2)
        repr_ = motor.get_graph_repr(cre_out, k_nodes=4)
        assert repr_.shape == (4, D)

    def test_full_pipeline_batch(self):
        motor = make_motor()
        motor.eval()
        B, L, D = 4, 16, 64
        concepts = torch.randn(B, L, D)
        with torch.no_grad():
            cryst_out = motor.build_graph(concepts)
        assert len(cryst_out.graphs) == B

    def test_pipeline_gradient_flows(self):
        motor = make_motor()
        concepts = torch.randn(1, 16, 64, requires_grad=True)
        cryst_out = motor.build_graph(concepts)
        graph     = cryst_out.graphs[0]
        n_nodes   = len(graph)
        if n_nodes == 0:
            return
        node_feats = cryst_out.node_vectors[0, :n_nodes]
        cre_out   = motor.reason(graph, node_feats, n_iterations=1)
        repr_     = motor.get_graph_repr(cre_out, k_nodes=4)
        loss      = repr_.sum()
        loss.backward()
        assert concepts.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# 10. BACKWARDS COMPATIBILITY
# ─────────────────────────────────────────────────────────────────────────────

class TestBackwardsCompatibility:

    def test_default_cre_not_affected_by_muse(self):
        from core.graph import CAUSAL_RELATIONS
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16, n_message_layers=1)
        cre = CausalReasoningEngine(cfg)
        assert cre.relation_keys == CAUSAL_RELATIONS

    def test_muse_cre_does_not_affect_cora_cre(self):
        from motors.cora.motor import CORAMotor, CORAMotorConfig
        from core.graph import CAUSAL_RELATIONS
        cora_config = CORAMotorConfig()
        cora_motor  = CORAMotor(cora_config)
        # CORA uses default CRE (CAUSAL_RELATIONS)
        assert cora_motor.cre.relation_keys == CAUSAL_RELATIONS

    def test_muse_and_axiom_have_different_relation_keys(self):
        from motors.axiom.relations import MATH_RELATIONS
        narr_set = set(NARRATIVE_RELATIONS)
        math_set = set(MATH_RELATIONS)
        assert narr_set != math_set

    def test_more_relations_more_params(self):
        """Motor con más relaciones tiene más parámetros en message_fns."""
        cfg_10 = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                           n_message_layers=1, n_relation_types=10)
        cfg_5  = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                           n_message_layers=1, n_relation_types=5)
        rels_5 = NARRATIVE_RELATIONS[:5]
        cre_10 = CausalReasoningEngine(cfg_10, relation_keys=NARRATIVE_RELATIONS)
        cre_5  = CausalReasoningEngine(cfg_5,  relation_keys=rels_5)
        params_10 = sum(p.numel() for p in cre_10.parameters())
        params_5  = sum(p.numel() for p in cre_5.parameters())
        assert params_10 > params_5
