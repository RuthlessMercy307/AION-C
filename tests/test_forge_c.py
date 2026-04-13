"""
tests/test_forge_c.py — Tests para CodeMotor (FORGE-C)
=======================================================

Verifica que:
  1. motors/forge_c/relations.py define los tipos y relaciones correctos
  2. CodeNode/CodeEdge aceptan CodeNodeType/CodeRelation sin errores
  3. CodeMotor implementa BaseMotor correctamente
  4. build_graph produce grafos con CodeNodeType/CodeRelation
  5. reason() usa CODE_RELATIONS en sus funciones de mensaje
  6. get_graph_repr() produce [k_nodes, D] en todos los casos
  7. CausalReasoningEngine con relation_keys=CODE_RELATIONS funciona
  8. Los tests originales de CORA no se ven afectados por los cambios al CRE
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from core.graph import CausalGraph, CausalEdge, CausalNode, CausalRelation, NodeType
from motors.base_motor import BaseMotor
from motors.forge_c.relations import (
    CODE_NODE_TYPES,
    CODE_RELATIONS,
    CodeEdge,
    CodeNode,
    CodeNodeType,
    CodeRelation,
    CALL_RELATIONS,
    HIER_RELATIONS,
    DEPS_RELATIONS,
)
from motors.forge_c.motor import CodeMotor, CodeMotorConfig, CodeCrystallizer
from crystallizer.config import CrystallizerConfig
from cre.config import CREConfig
from cre.engine import CausalReasoningEngine, CREOutput


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_tiny_code_config() -> CodeMotorConfig:
    cryst = CrystallizerConfig(
        hidden_dim=64,
        max_nodes=8,
        pooler_heads=4,
        n_node_types=8,
        n_relation_types=12,
    )
    cre = CREConfig(node_dim=64, edge_dim=16, message_dim=32,
                    n_message_layers=1, max_iterations=3, n_relation_types=12)
    return CodeMotorConfig(crystallizer=cryst, cre=cre)


def make_motor() -> CodeMotor:
    return CodeMotor(make_tiny_code_config())


def make_code_graph(n_nodes: int = 3, n_edges: int = 2) -> CausalGraph:
    """Grafo con CodeNode y CodeEdge."""
    g = CausalGraph()
    for i in range(n_nodes):
        g.add_node(CodeNode(
            node_id=f"fn{i}",
            label=f"function_{i}",
            node_type=CodeNodeType.FUNCTION,
            confidence=1.0,
        ))
    for i in range(min(n_edges, n_nodes - 1)):
        g.add_edge(CodeEdge(
            source_id=f"fn{i}",
            target_id=f"fn{i+1}",
            relation=CodeRelation.CALLS,
            strength=1.0,
            confidence=1.0,
        ))
    return g


# ─────────────────────────────────────────────────────────────────────────────
# 1. RELATIONS MODULE
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeRelationsModule:
    """relations.py define los tipos y relaciones correctos."""

    def test_code_node_types_count(self):
        assert len(CODE_NODE_TYPES) == 8

    def test_code_node_types_values(self):
        expected = {"function", "class", "module", "variable",
                    "expression", "error", "test", "config"}
        assert set(CODE_NODE_TYPES) == expected

    def test_code_relations_count(self):
        assert len(CODE_RELATIONS) == 12

    def test_code_relations_values(self):
        expected = {
            "calls", "imports", "inherits", "mutates", "reads",
            "returns", "throws", "depends_on", "tests",
            "implements", "overrides", "data_flows_to",
        }
        assert set(CODE_RELATIONS) == expected

    def test_code_node_type_is_str_enum(self):
        assert isinstance(CodeNodeType.FUNCTION, str)
        assert CodeNodeType.FUNCTION.value == "function"

    def test_code_relation_is_str_enum(self):
        assert isinstance(CodeRelation.CALLS, str)
        assert CodeRelation.CALLS.value == "calls"

    def test_semantic_groupings(self):
        assert CodeRelation.CALLS in CALL_RELATIONS
        assert CodeRelation.DATA_FLOWS_TO in CALL_RELATIONS
        assert CodeRelation.INHERITS in HIER_RELATIONS
        assert CodeRelation.IMPORTS in DEPS_RELATIONS


# ─────────────────────────────────────────────────────────────────────────────
# 2. CODENODE AND CODEEDGE
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeNodeEdge:
    """CodeNode/CodeEdge aceptan code types sin errores en runtime."""

    def test_code_node_accepts_code_node_type(self):
        node = CodeNode(
            node_id="fn1",
            label="my_function",
            node_type=CodeNodeType.FUNCTION,
            confidence=0.9,
        )
        assert node.node_type == CodeNodeType.FUNCTION
        assert node.node_type.value == "function"

    def test_code_node_all_types(self):
        for ct in CodeNodeType:
            n = CodeNode(node_id=f"n_{ct.value}", label=ct.value, node_type=ct, confidence=1.0)
            assert n.node_type == ct

    def test_code_edge_accepts_code_relation(self):
        edge = CodeEdge(
            source_id="fn1",
            target_id="fn2",
            relation=CodeRelation.CALLS,
            strength=1.0,
            confidence=1.0,
        )
        assert edge.relation == CodeRelation.CALLS
        assert edge.relation.value == "calls"

    def test_code_edge_all_relations(self):
        for i, cr in enumerate(CodeRelation):
            e = CodeEdge(source_id=f"a{i}", target_id=f"b{i}",
                         relation=cr, strength=0.8, confidence=0.9)
            assert e.relation == cr

    def test_code_edge_in_causal_graph(self):
        """CausalGraph acepta CodeNode y CodeEdge sin errores."""
        g = make_code_graph(3, 2)
        assert len(g) == 3
        assert len(g.edges) == 2
        for edge in g.edges:
            assert isinstance(edge.relation, CodeRelation)

    def test_code_edge_self_loop_raises(self):
        with pytest.raises(ValueError):
            CodeEdge(source_id="fn1", target_id="fn1",
                     relation=CodeRelation.CALLS, strength=1.0, confidence=1.0)

    def test_code_edge_invalid_strength_raises(self):
        with pytest.raises(ValueError):
            CodeEdge(source_id="a", target_id="b",
                     relation=CodeRelation.CALLS, strength=1.5, confidence=1.0)

    def test_code_node_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            CodeNode(node_id="n", label="x", node_type=CodeNodeType.ERROR, confidence=1.5)

    def test_source_target_idx_assigned(self):
        """CausalGraph asigna source_idx/target_idx correctamente."""
        g = make_code_graph(3, 2)
        for edge in g.edges:
            assert edge.source_idx >= 0
            assert edge.target_idx >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. CRE WITH CODE RELATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestCREWithCodeRelations:
    """CausalReasoningEngine funciona con relation_keys=CODE_RELATIONS."""

    def test_cre_initializes_with_code_relations(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=12)
        cre = CausalReasoningEngine(cfg, relation_keys=CODE_RELATIONS)
        assert cre.relation_keys == CODE_RELATIONS

    def test_cre_message_fns_have_code_relation_keys(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=12)
        cre = CausalReasoningEngine(cfg, relation_keys=CODE_RELATIONS)
        for layer in cre.layers:
            for rel in CODE_RELATIONS:
                assert rel in layer.message_fns, f"Missing key: {rel}"

    def test_cre_does_not_have_causal_relation_keys(self):
        """Motor de código no debe tener funciones de mensaje para relaciones causales."""
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=12)
        cre = CausalReasoningEngine(cfg, relation_keys=CODE_RELATIONS)
        for layer in cre.layers:
            assert "causes" not in layer.message_fns

    def test_cre_forward_with_code_graph(self):
        cfg = CREConfig(node_dim=64, edge_dim=16, message_dim=32,
                        n_message_layers=1, max_iterations=2, n_relation_types=12)
        cre = CausalReasoningEngine(cfg, relation_keys=CODE_RELATIONS)
        cre.eval()
        graph = make_code_graph(4, 3)
        feats = torch.randn(4, 64)
        with torch.no_grad():
            out = cre.forward(graph, feats, n_iterations=2)
        assert out.node_features.shape == (4, 64)
        assert out.iterations_run == 2

    def test_cre_forward_no_edges(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=12)
        cre = CausalReasoningEngine(cfg, relation_keys=CODE_RELATIONS)
        cre.eval()
        graph = make_code_graph(3, 0)
        feats = torch.randn(3, 32)
        with torch.no_grad():
            out = cre.forward(graph, feats, n_iterations=2)
        assert out.node_features.shape == (3, 32)

    def test_default_cre_still_uses_causal_relations(self):
        """Sin relation_keys, el CRE usa CAUSAL_RELATIONS (backwards compatible)."""
        from core.graph import CAUSAL_RELATIONS
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2)
        cre = CausalReasoningEngine(cfg)
        assert cre.relation_keys == CAUSAL_RELATIONS

    def test_cre_gradients_with_code_relations(self):
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2, n_relation_types=12)
        cre = CausalReasoningEngine(cfg, relation_keys=CODE_RELATIONS)
        graph = make_code_graph(3, 2)
        feats = torch.randn(3, 32, requires_grad=True)
        out  = cre.forward(graph, feats, n_iterations=2)
        out.node_features.sum().backward()
        assert feats.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# 4. CODE MOTOR IMPLEMENTS BASE MOTOR
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeMotorInterface:
    """CodeMotor implementa todos los métodos de BaseMotor."""

    def test_code_motor_is_base_motor(self):
        m = make_motor()
        assert isinstance(m, BaseMotor)

    def test_code_motor_is_nn_module(self):
        m = make_motor()
        assert isinstance(m, nn.Module)

    def test_has_all_abstract_methods(self):
        m = make_motor()
        for method in ("define_node_types", "define_relations",
                       "build_graph", "reason", "get_graph_repr"):
            assert callable(getattr(m, method, None)), f"Missing method: {method}"

    def test_has_trainable_parameters(self):
        m = make_motor()
        assert len(list(m.parameters())) > 0

    def test_has_crystallizer_and_cre(self):
        m = make_motor()
        assert hasattr(m, "crystallizer")
        assert hasattr(m, "cre")
        assert isinstance(m.crystallizer, CodeCrystallizer)
        assert isinstance(m.cre, CausalReasoningEngine)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEFINE_NODE_TYPES AND DEFINE_RELATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeMotorIntrospection:

    def test_node_types_match_code_types(self):
        m = make_motor()
        assert set(m.define_node_types()) == set(CODE_NODE_TYPES)

    def test_node_types_count(self):
        m = make_motor()
        assert len(m.define_node_types()) == 8

    def test_relations_match_code_relations(self):
        m = make_motor()
        assert set(m.define_relations()) == set(CODE_RELATIONS)

    def test_relations_count(self):
        m = make_motor()
        assert len(m.define_relations()) == 12

    def test_node_types_do_not_include_causal_types(self):
        m = make_motor()
        # CodeMotor no usa tipos causales como "entity", "event", "state"
        code_types = set(m.define_node_types())
        causal_types = {"entity", "event", "state", "action", "hypothesis", "fact", "question"}
        assert code_types.isdisjoint(causal_types)

    def test_relations_do_not_include_causal_relations(self):
        m = make_motor()
        code_rels   = set(m.define_relations())
        causal_rels = {"causes", "enables", "prevents", "leads_to"}
        assert code_rels.isdisjoint(causal_rels)


# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILD_GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeMotorBuildGraph:

    def test_returns_graphs_list(self):
        m = make_motor()
        m.eval()
        with torch.no_grad():
            out = m.build_graph(torch.randn(2, 10, 64))
        assert len(out.graphs) == 2

    def test_graphs_are_causal_graphs(self):
        m = make_motor()
        m.eval()
        with torch.no_grad():
            out = m.build_graph(torch.randn(2, 10, 64))
        for g in out.graphs:
            assert isinstance(g, CausalGraph)

    def test_graph_edges_have_code_relations(self):
        m = make_motor()
        m.eval()
        with torch.no_grad():
            out = m.build_graph(torch.randn(1, 10, 64))
        g = out.graphs[0]
        for edge in g.edges:
            assert edge.relation.value in CODE_RELATIONS, \
                f"Edge relation {edge.relation.value!r} not in CODE_RELATIONS"

    def test_graph_nodes_have_code_node_types(self):
        m = make_motor()
        m.eval()
        with torch.no_grad():
            out = m.build_graph(torch.randn(1, 12, 64))
        g = out.graphs[0]
        for node in g.nodes:
            assert node.node_type.value in CODE_NODE_TYPES, \
                f"Node type {node.node_type.value!r} not in CODE_NODE_TYPES"

    def test_node_vectors_shape(self):
        m = make_motor()
        m.eval()
        B, L, D = 2, 10, 64
        with torch.no_grad():
            out = m.build_graph(torch.randn(B, L, D))
        K = min(L, m.config.crystallizer.max_nodes)
        assert out.node_vectors.shape == (B, K, D)

    def test_relation_logits_shape(self):
        m = make_motor()
        m.eval()
        B, L, D = 2, 10, 64
        with torch.no_grad():
            out = m.build_graph(torch.randn(B, L, D))
        K = min(L, m.config.crystallizer.max_nodes)
        assert out.relation_logits.shape == (B, K, K, 12)


# ─────────────────────────────────────────────────────────────────────────────
# 7. REASON
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeMotorReason:

    def test_returns_cre_output(self):
        m = make_motor()
        m.eval()
        graph = make_code_graph(3, 2)
        feats = torch.randn(3, 64)
        with torch.no_grad():
            out = m.reason(graph, feats, n_iterations=2)
        assert isinstance(out, CREOutput)

    def test_node_features_shape(self):
        m = make_motor()
        m.eval()
        N = 4
        graph = make_code_graph(N, 3)
        feats = torch.randn(N, 64)
        with torch.no_grad():
            out = m.reason(graph, feats, n_iterations=2)
        assert out.node_features.shape == (N, 64)

    def test_edge_features_shape(self):
        m = make_motor()
        m.eval()
        N, E = 4, 3
        graph = make_code_graph(N, E)
        feats = torch.randn(N, 64)
        with torch.no_grad():
            out = m.reason(graph, feats, n_iterations=2)
        assert out.edge_features.shape == (E, m.config.cre.edge_dim)

    def test_iterations_run(self):
        m = make_motor()
        m.eval()
        graph = make_code_graph(3, 2)
        with torch.no_grad():
            out = m.reason(graph, torch.randn(3, 64), n_iterations=3)
        assert out.iterations_run == 3

    def test_different_code_relations_produce_different_outputs(self):
        """CALLS y INHERITS producen outputs distintos (distintas message fns)."""
        m = make_motor()
        m.eval()
        feats = torch.randn(2, 64)

        g_calls = CausalGraph()
        g_inh   = CausalGraph()
        for i in range(2):
            g_calls.add_node(CodeNode(f"n{i}", f"fn{i}", CodeNodeType.FUNCTION, 1.0))
            g_inh.add_node(CodeNode(f"n{i}", f"cls{i}", CodeNodeType.CLASS, 1.0))
        g_calls.add_edge(CodeEdge("n0", "n1", CodeRelation.CALLS, 1.0, 1.0))
        g_inh.add_edge(CodeEdge("n0", "n1", CodeRelation.INHERITS, 1.0, 1.0))

        with torch.no_grad():
            out_calls = m.reason(g_calls, feats.clone(), n_iterations=2)
            out_inh   = m.reason(g_inh,   feats.clone(), n_iterations=2)

        assert not torch.allclose(out_calls.node_features, out_inh.node_features), \
            "CALLS and INHERITS should produce different node features"


# ─────────────────────────────────────────────────────────────────────────────
# 8. GET_GRAPH_REPR
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeMotorGetGraphRepr:

    def _cre_output(self, n: int, D: int = 64) -> CREOutput:
        return CREOutput(
            node_features=torch.randn(n, D),
            edge_features=torch.zeros(0, 16),
            iterations_run=2,
            layer_outputs=[],
        )

    def test_output_shape_exact(self):
        m = make_motor()
        out = m.get_graph_repr(self._cre_output(8), k_nodes=8)
        assert out.shape == (8, 64)

    def test_output_shape_n_gt_k(self):
        m = make_motor()
        out = m.get_graph_repr(self._cre_output(20), k_nodes=8)
        assert out.shape == (8, 64)

    def test_output_shape_n_lt_k(self):
        m = make_motor()
        out = m.get_graph_repr(self._cre_output(3), k_nodes=8)
        assert out.shape == (8, 64)

    def test_padding_is_zeros(self):
        m = make_motor()
        cre_out = CREOutput(
            node_features=torch.ones(2, 64),
            edge_features=torch.zeros(0, 16),
            iterations_run=1, layer_outputs=[],
        )
        out = m.get_graph_repr(cre_out, k_nodes=5)
        assert torch.all(out[2:] == 0.0)
        assert torch.all(out[:2] == 1.0)

    def test_empty_graph(self):
        m = make_motor()
        out = m.get_graph_repr(self._cre_output(0), k_nodes=8)
        assert out.shape == (8, 64)
        assert torch.all(out == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 9. END-TO-END INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeMotorEndToEnd:

    def test_full_pipeline(self):
        m = make_motor()
        m.eval()
        concepts = torch.randn(1, 10, 64)
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
        n = max(cryst_out.node_counts[0], 1)
        feats = cryst_out.node_vectors[0, :n]
        cre_out = m.reason(cryst_out.graphs[0], feats, n_iterations=1)
        repr_   = m.get_graph_repr(cre_out, k_nodes=4)
        repr_.sum().backward()
        assert concepts.grad is not None

    def test_config_dimension_mismatch_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, pooler_heads=4,
                                   n_node_types=8, n_relation_types=12)
        cre   = CREConfig(node_dim=128, n_relation_types=12)
        with pytest.raises(ValueError):
            CodeMotorConfig(crystallizer=cryst, cre=cre)

    def test_config_wrong_node_type_count_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, pooler_heads=4,
                                   n_node_types=7,   # wrong: should be 8
                                   n_relation_types=12)
        cre   = CREConfig(node_dim=64, n_relation_types=12)
        with pytest.raises(ValueError):
            CodeMotorConfig(crystallizer=cryst, cre=cre)

    def test_config_wrong_relation_count_raises(self):
        cryst = CrystallizerConfig(hidden_dim=64, pooler_heads=4,
                                   n_node_types=8,
                                   n_relation_types=16)  # wrong: should be 12
        cre   = CREConfig(node_dim=64, n_relation_types=12)
        with pytest.raises(ValueError):
            CodeMotorConfig(crystallizer=cryst, cre=cre)


# ─────────────────────────────────────────────────────────────────────────────
# 10. BACKWARDS COMPATIBILITY — CORA CRE MUST STILL WORK
# ─────────────────────────────────────────────────────────────────────────────

class TestBackwardsCompatibility:
    """Los cambios al CRE no rompen el uso existente (CORA-style CRE)."""

    def test_causal_cre_default_relation_keys(self):
        from core.graph import CAUSAL_RELATIONS
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2)
        cre = CausalReasoningEngine(cfg)
        assert cre.relation_keys == CAUSAL_RELATIONS

    def test_causal_cre_message_fns_count(self):
        from core.graph import CAUSAL_RELATIONS
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2)
        cre = CausalReasoningEngine(cfg)
        for layer in cre.layers:
            assert len(layer.message_fns) == len(CAUSAL_RELATIONS)

    def test_causal_cre_forward_with_causal_graph(self):
        """CRE original funciona con grafos causales después de los cambios."""
        cfg = CREConfig(node_dim=32, edge_dim=8, message_dim=16,
                        n_message_layers=1, max_iterations=2)
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        g = CausalGraph()
        g.add_node(CausalNode("n0", "A", NodeType.EVENT, 1.0))
        g.add_node(CausalNode("n1", "B", NodeType.STATE, 1.0))
        g.add_edge(CausalEdge("n0", "n1", CausalRelation.CAUSES, 1.0, 1.0))

        feats = torch.randn(2, 32)
        with torch.no_grad():
            out = cre.forward(g, feats, n_iterations=2)
        assert out.node_features.shape == (2, 32)
        assert out.iterations_run == 2
