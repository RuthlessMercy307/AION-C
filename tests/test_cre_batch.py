"""
tests/test_cre_batch.py — Tests para CausalReasoningEngine.forward_batch
=========================================================================

Verifica que el batch forward produce resultados equivalentes al forward
individual por grafo, y que el batching funciona correctamente con:
    - grafos de distinto número de nodos (padding)
    - grafos con distinto número de aristas
    - batch size = 1 (degenerado)
    - grafos sin aristas
    - gradientes a través del batch forward
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

from core.graph import CausalEdge, CausalGraph, CausalNode, CausalRelation, NodeType
from cre import CREConfig, CausalReasoningEngine


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_cfg(**kwargs) -> CREConfig:
    defaults = dict(node_dim=32, edge_dim=16, message_dim=24,
                    n_message_layers=2, max_iterations=3, n_relation_types=16,
                    use_convergence_gate=False)
    defaults.update(kwargs)
    return CREConfig(**defaults)


def make_graph(n_nodes: int, n_edges: int = 0,
               relation: CausalRelation = CausalRelation.CAUSES) -> CausalGraph:
    """Construye un grafo con n_nodes nodos y n_edges aristas lineales."""
    graph = CausalGraph()
    for i in range(n_nodes):
        graph.add_node(CausalNode(
            node_id=f"n{i}", label=f"node{i}",
            node_type=NodeType.EVENT, confidence=0.9,
        ))
    for i in range(min(n_edges, n_nodes - 1)):
        graph.add_edge(CausalEdge(
            source_id=f"n{i}", target_id=f"n{i+1}",
            relation=relation, strength=0.8, confidence=0.9,
        ))
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestForwardBatchShapes:
    """Los shapes de salida deben coincidir con los de entrada."""

    def test_output_count_matches_input(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()
        graphs = [make_graph(3, 2), make_graph(5, 4), make_graph(2, 1)]
        feats  = [torch.randn(3, 32), torch.randn(5, 32), torch.randn(2, 32)]
        with torch.no_grad():
            outs = cre.forward_batch(graphs, feats, n_iterations=2)
        assert len(outs) == 3

    def test_node_output_shapes(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()
        sizes = [3, 7, 2, 5]
        graphs = [make_graph(n, n - 1) for n in sizes]
        feats  = [torch.randn(n, 32) for n in sizes]
        with torch.no_grad():
            outs = cre.forward_batch(graphs, feats, n_iterations=2)
        for i, (n, out) in enumerate(zip(sizes, outs)):
            assert out.node_features.shape == (n, 32), \
                f"Graph {i}: expected ({n}, 32), got {out.node_features.shape}"

    def test_edge_output_shapes(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()
        n_edges_list = [2, 4, 0, 3]
        graphs = [make_graph(5, ne) for ne in n_edges_list]
        feats  = [torch.randn(5, 32) for _ in n_edges_list]
        with torch.no_grad():
            outs = cre.forward_batch(graphs, feats, n_iterations=2)
        for ne, out in zip(n_edges_list, outs):
            assert out.edge_features.shape == (ne, 16)

    def test_iterations_run_recorded(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()
        graphs = [make_graph(3, 2), make_graph(4, 3)]
        feats  = [torch.randn(3, 32), torch.randn(4, 32)]
        with torch.no_grad():
            outs = cre.forward_batch(graphs, feats, n_iterations=3)
        for out in outs:
            assert out.iterations_run == 3


class TestForwardBatchEquivalence:
    """
    forward_batch con un grafo debe producir el mismo resultado que forward()
    individual (mismos pesos, mismo input → mismo output).
    """

    def test_single_graph_equivalence(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        graph = make_graph(4, 3)
        feats = torch.randn(4, 32)
        n_iters = 2

        with torch.no_grad():
            out_single = cre.forward(graph, feats, n_iterations=n_iters)
            out_batch  = cre.forward_batch([graph], [feats], n_iterations=n_iters)

        torch.testing.assert_close(
            out_single.node_features,
            out_batch[0].node_features,
            atol=1e-5, rtol=1e-5,
            msg="forward_batch single-graph != forward() single-graph",
        )

    def test_multi_graph_each_matches_single(self):
        """Cada grafo en el batch debe coincidir con su forward individual."""
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        specs = [(3, 2), (5, 4), (2, 1), (4, 3)]
        graphs = [make_graph(n, e) for n, e in specs]
        feats  = [torch.randn(n, 32) for n, _ in specs]
        n_iters = 2

        with torch.no_grad():
            singles = [
                cre.forward(g, f, n_iterations=n_iters)
                for g, f in zip(graphs, feats)
            ]
            batch_outs = cre.forward_batch(graphs, feats, n_iterations=n_iters)

        for i, (s, b) in enumerate(zip(singles, batch_outs)):
            torch.testing.assert_close(
                s.node_features, b.node_features,
                atol=1e-5, rtol=1e-5,
                msg=f"Graph {i}: batch output differs from single forward",
            )


class TestForwardBatchEdgeCases:
    """Casos límite: batch vacío, grafo sin aristas, batch_size=1."""

    def test_empty_batch_returns_empty_list(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()
        with torch.no_grad():
            outs = cre.forward_batch([], [])
        assert outs == []

    def test_graph_without_edges(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()
        graph = make_graph(4, 0)   # no edges
        feats = torch.randn(4, 32)
        with torch.no_grad():
            outs = cre.forward_batch([graph], [feats], n_iterations=2)
        assert outs[0].node_features.shape == (4, 32)
        assert outs[0].edge_features.shape == (0, 16)

    def test_all_graphs_without_edges(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()
        graphs = [make_graph(3, 0), make_graph(5, 0)]
        feats  = [torch.randn(3, 32), torch.randn(5, 32)]
        with torch.no_grad():
            outs = cre.forward_batch(graphs, feats, n_iterations=2)
        assert outs[0].node_features.shape == (3, 32)
        assert outs[1].node_features.shape == (5, 32)

    def test_mixed_with_and_without_edges(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()
        graphs = [make_graph(3, 2), make_graph(4, 0), make_graph(5, 3)]
        feats  = [torch.randn(3, 32), torch.randn(4, 32), torch.randn(5, 32)]
        with torch.no_grad():
            outs = cre.forward_batch(graphs, feats, n_iterations=2)
        assert len(outs) == 3
        assert outs[0].node_features.shape == (3, 32)
        assert outs[1].node_features.shape == (4, 32)
        assert outs[2].node_features.shape == (5, 32)

    def test_single_node_graphs(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()
        graphs = [make_graph(1, 0), make_graph(1, 0)]
        feats  = [torch.randn(1, 32), torch.randn(1, 32)]
        with torch.no_grad():
            outs = cre.forward_batch(graphs, feats, n_iterations=2)
        assert outs[0].node_features.shape == (1, 32)
        assert outs[1].node_features.shape == (1, 32)


class TestForwardBatchGradients:
    """Gradientes fluyen correctamente a través de forward_batch."""

    def test_gradients_flow_to_input(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)

        graphs = [make_graph(3, 2), make_graph(4, 3)]
        feats  = [torch.randn(3, 32, requires_grad=True),
                  torch.randn(4, 32, requires_grad=True)]

        outs = cre.forward_batch(graphs, feats, n_iterations=2)
        loss = sum(o.node_features.sum() for o in outs)
        loss.backward()

        for i, f in enumerate(feats):
            assert f.grad is not None, f"No gradient for input {i}"
            assert not torch.all(f.grad == 0), f"Zero gradient for input {i}"

    def test_gradients_flow_to_parameters(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)

        graphs = [make_graph(4, 3)]
        feats  = [torch.randn(4, 32)]

        outs = cre.forward_batch(graphs, feats, n_iterations=2)
        loss = outs[0].node_features.sum()
        loss.backward()

        params_with_grad = [
            p for p in cre.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(params_with_grad) > 0, "No parameters received gradients"


class TestForwardBatchVariableSizes:
    """Padding funciona con grafos de tamaños muy distintos."""

    def test_large_size_difference(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()
        graphs = [make_graph(2, 1), make_graph(20, 15)]
        feats  = [torch.randn(2, 32), torch.randn(20, 32)]
        with torch.no_grad():
            outs = cre.forward_batch(graphs, feats, n_iterations=2)
        assert outs[0].node_features.shape == (2,  32)
        assert outs[1].node_features.shape == (20, 32)

    def test_padding_does_not_leak_into_output(self):
        """
        Los nodos de padding (dummy) no deben contaminar los nodos reales.
        Verifica que un grafo pequeño en batch da el mismo resultado
        que un grafo pequeño solo (sin padding de otro grafo).
        """
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        small_graph = make_graph(2, 1)
        large_graph = make_graph(10, 8)
        feats_small = torch.randn(2, 32)
        feats_large = torch.randn(10, 32)

        with torch.no_grad():
            out_alone = cre.forward_batch([small_graph], [feats_small], n_iterations=2)
            out_mixed = cre.forward_batch(
                [small_graph, large_graph],
                [feats_small, feats_large],
                n_iterations=2,
            )

        # The small graph result should be identical in both cases
        torch.testing.assert_close(
            out_alone[0].node_features,
            out_mixed[0].node_features,
            atol=1e-5, rtol=1e-5,
            msg="Small graph output changed when batched with larger graph",
        )

    def test_different_relation_types(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()
        g1 = make_graph(3, 2, CausalRelation.CAUSES)
        g2 = make_graph(3, 2, CausalRelation.PREVENTS)
        f1 = torch.randn(3, 32)
        f2 = f1.clone()
        with torch.no_grad():
            outs = cre.forward_batch([g1, g2], [f1, f2], n_iterations=2)
        # Same input features but different relation types → different outputs
        assert not torch.allclose(outs[0].node_features, outs[1].node_features), \
            "CAUSES and PREVENTS should produce different outputs"
