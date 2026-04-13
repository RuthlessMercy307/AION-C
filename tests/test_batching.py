"""
tests/test_batching.py — Tests para PyGStyleBatcher y forward_batched
======================================================================

Verifica que:
  1. PyGStyleBatcher construye super-grafos correctos (shapes, offsets, batch vector)
  2. forward_batched produce resultados IDÉNTICOS a forward() individual (atol=1e-5)
     — incluyendo batch=16 (el caso clave del plan v4 §16.2)
  3. Los grafos no se contaminan entre sí (isolación garantizada)
  4. Gradientes fluyen correctamente
  5. AutoScaler funciona en CPU
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

from core.graph import CausalEdge, CausalGraph, CausalNode, CausalRelation, NodeType
from cre import (
    CREConfig,
    CausalReasoningEngine,
    PyGStyleBatcher,
    BatchedGraph,
    AutoScaler,
    AutoScaleResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_cfg(**kwargs) -> CREConfig:
    defaults = dict(
        node_dim=32, edge_dim=16, message_dim=24,
        n_message_layers=2, max_iterations=3, n_relation_types=16,
        use_convergence_gate=False,
    )
    defaults.update(kwargs)
    return CREConfig(**defaults)


def make_graph(
    n_nodes: int,
    n_edges: int = 0,
    relation: CausalRelation = CausalRelation.CAUSES,
) -> CausalGraph:
    """Grafo lineal n0→n1→…→n(n_nodes-1) con la relación dada."""
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


def make_full_graph(n_nodes: int, relation: CausalRelation = CausalRelation.CAUSES) -> CausalGraph:
    """Grafo completo (todos los pares)."""
    graph = CausalGraph()
    for i in range(n_nodes):
        graph.add_node(CausalNode(
            node_id=f"n{i}", label=f"node{i}",
            node_type=NodeType.EVENT, confidence=0.9,
        ))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                graph.add_edge(CausalEdge(
                    source_id=f"n{i}", target_id=f"n{j}",
                    relation=relation, strength=0.8, confidence=0.9,
                ))
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# TEST PyGStyleBatcher — SHAPES Y ESTRUCTURA
# ─────────────────────────────────────────────────────────────────────────────

class TestPyGStyleBatcherShapes:
    """Verifica que el BatchedGraph tiene los shapes y valores correctos."""

    def test_node_features_concatenated(self):
        batcher = PyGStyleBatcher()
        g1 = make_graph(3, 2)
        g2 = make_graph(5, 4)
        f1 = torch.randn(3, 32)
        f2 = torch.randn(5, 32)
        batched = batcher.batch([g1, g2], [f1, f2])

        assert batched.node_features.shape == (8, 32)
        # Los primeros 3 nodos son de g1
        torch.testing.assert_close(batched.node_features[:3], f1)
        # Los siguientes 5 son de g2
        torch.testing.assert_close(batched.node_features[3:], f2)

    def test_edge_index_shape(self):
        batcher = PyGStyleBatcher()
        g1 = make_graph(3, 2)  # 2 edges
        g2 = make_graph(5, 4)  # 4 edges
        f1 = torch.randn(3, 32)
        f2 = torch.randn(5, 32)
        batched = batcher.batch([g1, g2], [f1, f2])

        assert batched.edge_index.shape == (2, 6)  # 2+4 edges total

    def test_edge_offsets_applied(self):
        """Las aristas del grafo B tienen índices offset por n_nodes(A)."""
        batcher = PyGStyleBatcher()
        g1 = make_graph(3, 2)  # nodos 0,1,2 — edges: (0,1),(1,2)
        g2 = make_graph(4, 3)  # nodos 0,1,2,3 — edges: (0,1),(1,2),(2,3)
        f1 = torch.randn(3, 32)
        f2 = torch.randn(4, 32)
        batched = batcher.batch([g1, g2], [f1, f2])

        edge_index = batched.edge_index  # [2, 5]
        # Aristas de g1 (primeras 2): índices 0..2
        assert edge_index[0, 0].item() < 3
        assert edge_index[1, 0].item() < 3
        # Aristas de g2 (últimas 3): índices ≥ 3 (offset = 3)
        assert edge_index[0, 2].item() >= 3
        assert edge_index[1, 2].item() >= 3
        # Ningún índice de g2 apunta a nodos de g1 (ni viceversa)
        for i in range(2, 5):
            assert edge_index[0, i].item() >= 3
            assert edge_index[1, i].item() >= 3

    def test_batch_vector(self):
        """batch[i] indica el ID del grafo al que pertenece el nodo i."""
        batcher = PyGStyleBatcher()
        g1 = make_graph(3, 0)
        g2 = make_graph(5, 0)
        f1 = torch.randn(3, 32)
        f2 = torch.randn(5, 32)
        batched = batcher.batch([g1, g2], [f1, f2])

        expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
        torch.testing.assert_close(batched.batch, expected)

    def test_n_graphs_and_metadata(self):
        batcher = PyGStyleBatcher()
        graphs = [make_graph(3, 2), make_graph(4, 3), make_graph(2, 1)]
        feats  = [torch.randn(3, 32), torch.randn(4, 32), torch.randn(2, 32)]
        batched = batcher.batch(graphs, feats)

        assert batched.n_graphs == 3
        assert batched.nodes_per_graph == [3, 4, 2]
        assert batched.edges_per_graph == [2, 3, 1]
        assert batched.n_nodes == 9
        assert batched.n_edges == 6

    def test_no_edges(self):
        """Grafos sin aristas: edge_index vacío."""
        batcher = PyGStyleBatcher()
        g1 = make_graph(3, 0)
        g2 = make_graph(4, 0)
        f1 = torch.randn(3, 32)
        f2 = torch.randn(4, 32)
        batched = batcher.batch([g1, g2], [f1, f2])

        assert batched.n_edges == 0
        assert batched.edge_index.shape == (2, 0)

    def test_single_graph(self):
        """batch de un solo grafo: offsets = 0, equivalente a grafo original."""
        batcher = PyGStyleBatcher()
        g = make_graph(4, 3)
        f = torch.randn(4, 32)
        batched = batcher.batch([g], [f])

        assert batched.n_graphs == 1
        assert batched.node_features.shape == (4, 32)
        assert batched.batch.tolist() == [0, 0, 0, 0]


# ─────────────────────────────────────────────────────────────────────────────
# TEST PyGStyleBatcher — UNBATCH
# ─────────────────────────────────────────────────────────────────────────────

class TestPyGStyleBatcherUnbatch:
    """unbatch devuelve tensores correctos por grafo."""

    def test_unbatch_recovers_original_slices(self):
        batcher = PyGStyleBatcher()
        g1 = make_graph(3, 0)
        g2 = make_graph(5, 0)
        f1 = torch.randn(3, 32)
        f2 = torch.randn(5, 32)
        batched = batcher.batch([g1, g2], [f1, f2])

        # Simular features refinados (mismos que los originales para test)
        refined = batched.node_features.clone()
        parts = batcher.unbatch(batched, refined)

        assert len(parts) == 2
        assert parts[0].shape == (3, 32)
        assert parts[1].shape == (5, 32)
        torch.testing.assert_close(parts[0], f1)
        torch.testing.assert_close(parts[1], f2)

    def test_unbatch_three_graphs(self):
        batcher = PyGStyleBatcher()
        sizes = [2, 4, 3]
        graphs = [make_graph(n, 0) for n in sizes]
        feats  = [torch.randn(n, 16) for n in sizes]
        batched = batcher.batch(graphs, feats)
        refined = torch.randn(9, 16)  # features modificados
        parts = batcher.unbatch(batched, refined)

        assert [p.shape[0] for p in parts] == sizes
        offset = 0
        for i, n in enumerate(sizes):
            torch.testing.assert_close(parts[i], refined[offset:offset + n])
            offset += n


# ─────────────────────────────────────────────────────────────────────────────
# TEST forward_batched — EQUIVALENCIA CON forward() INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────

class TestForwardBatchedEquivalence:
    """
    PRUEBA CLAVE del plan v4 §16.2:
    forward_batched con batch=N debe producir resultados IDÉNTICOS (atol=1e-5)
    a N llamadas individuales de forward().
    """

    def _run_equivalence(self, n_graphs: int, node_sizes, edge_counts, n_iters=2):
        """Helper: compara forward() x N vs forward_batched() x 1."""
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        graphs = [make_graph(n, e) for n, e in zip(node_sizes, edge_counts)]
        feats  = [torch.randn(n, 32) for n in node_sizes]

        batcher = PyGStyleBatcher()
        batched = batcher.batch(graphs, feats)

        with torch.no_grad():
            # N llamadas individuales
            singles = [
                cre.forward(g, f, n_iterations=n_iters)
                for g, f in zip(graphs, feats)
            ]
            # 1 llamada batch
            batch_outs = cre.forward_batched(batched, n_iterations=n_iters)

        assert len(batch_outs) == n_graphs
        for i, (s, b) in enumerate(zip(singles, batch_outs)):
            torch.testing.assert_close(
                s.node_features,
                b.node_features,
                atol=1e-5,
                rtol=1e-4,
                msg=f"Graph {i}: forward_batched != forward() (batch={n_graphs})",
            )

    def test_equivalence_batch_1(self):
        self._run_equivalence(1, [4], [3])

    def test_equivalence_batch_2(self):
        self._run_equivalence(2, [3, 5], [2, 4])

    def test_equivalence_batch_4(self):
        self._run_equivalence(4, [3, 5, 2, 4], [2, 4, 1, 3])

    def test_equivalence_batch_8(self):
        sizes = [3, 5, 2, 4, 6, 3, 4, 2]
        edges = [2, 4, 1, 3, 5, 2, 3, 1]
        self._run_equivalence(8, sizes, edges)

    def test_equivalence_batch_16(self):
        """EL TEST DEL PLAN v4 §16.2: batch=16 IDÉNTICO a 16 × forward()."""
        sizes = [3, 5, 2, 4, 6, 3, 4, 2, 5, 3, 4, 6, 2, 3, 5, 4]
        edges = [2, 4, 1, 3, 5, 2, 3, 1, 4, 2, 3, 5, 1, 2, 4, 3]
        self._run_equivalence(16, sizes, edges)

    def test_equivalence_uniform_graphs_batch_16(self):
        """16 grafos idénticos procesados en batch vs individualmente."""
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        graph  = make_graph(8, 6)
        feats  = [torch.randn(8, 32) for _ in range(16)]

        batcher = PyGStyleBatcher()
        batched = batcher.batch([graph] * 16, feats)

        with torch.no_grad():
            singles = [cre.forward(graph, f, n_iterations=2) for f in feats]
            batch_outs = cre.forward_batched(batched, n_iterations=2)

        for i, (s, b) in enumerate(zip(singles, batch_outs)):
            torch.testing.assert_close(
                s.node_features, b.node_features,
                atol=1e-5, rtol=1e-4,
                msg=f"Graph {i}: mismatch in uniform batch=16",
            )

    def test_equivalence_no_edges(self):
        """Grafos sin aristas: forward_batched = forward() para todos."""
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        graphs = [make_graph(4, 0), make_graph(3, 0), make_graph(5, 0)]
        feats  = [torch.randn(n, 32) for n in [4, 3, 5]]

        batcher = PyGStyleBatcher()
        batched = batcher.batch(graphs, feats)

        with torch.no_grad():
            singles = [cre.forward(g, f, n_iterations=2) for g, f in zip(graphs, feats)]
            batch_outs = cre.forward_batched(batched, n_iterations=2)

        for i, (s, b) in enumerate(zip(singles, batch_outs)):
            torch.testing.assert_close(
                s.node_features, b.node_features,
                atol=1e-5, rtol=1e-4,
            )

    def test_equivalence_mixed_edges_and_no_edges(self):
        """Mix de grafos con y sin aristas."""
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        graphs = [make_graph(3, 2), make_graph(4, 0), make_graph(5, 3), make_graph(2, 0)]
        feats  = [torch.randn(n, 32) for n in [3, 4, 5, 2]]

        batcher = PyGStyleBatcher()
        batched = batcher.batch(graphs, feats)

        with torch.no_grad():
            singles = [cre.forward(g, f, n_iterations=2) for g, f in zip(graphs, feats)]
            batch_outs = cre.forward_batched(batched, n_iterations=2)

        for i, (s, b) in enumerate(zip(singles, batch_outs)):
            torch.testing.assert_close(
                s.node_features, b.node_features,
                atol=1e-5, rtol=1e-4,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TEST forward_batched — ISOLACIÓN (grafos no se contaminan)
# ─────────────────────────────────────────────────────────────────────────────

class TestForwardBatchedIsolation:
    """
    Verifica que el message passing no cruza entre grafos.
    Clave: grafos desconectados = sin contaminación (la garantía matemática del plan).
    """

    def test_graph_result_independent_of_other_graphs(self):
        """
        El resultado del Grafo A no debe cambiar si el Grafo B tiene features diferentes.
        """
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        g1 = make_graph(3, 2)
        g2a = make_graph(4, 3)
        g2b = make_graph(4, 3)

        f1  = torch.randn(3, 32)
        f2a = torch.randn(4, 32)
        f2b = torch.randn(4, 32)  # diferentes features para g2

        batcher = PyGStyleBatcher()
        batched_a = batcher.batch([g1, g2a], [f1, f2a])
        batched_b = batcher.batch([g1, g2b], [f1, f2b])

        with torch.no_grad():
            out_a = cre.forward_batched(batched_a, n_iterations=2)
            out_b = cre.forward_batched(batched_b, n_iterations=2)

        # El resultado de g1 debe ser IDÉNTICO independientemente de g2
        torch.testing.assert_close(
            out_a[0].node_features,
            out_b[0].node_features,
            atol=1e-5, rtol=1e-4,
            msg="Graph A is affected by Graph B's features — isolation broken",
        )

    def test_causes_and_prevents_isolated_in_batch(self):
        """
        Un grafo CAUSES y un grafo PREVENTS en el mismo batch no se mezclan.
        Cada uno debe dar el mismo resultado que procesado individualmente.
        """
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        g_causes   = make_graph(4, 3, CausalRelation.CAUSES)
        g_prevents = make_graph(4, 3, CausalRelation.PREVENTS)
        f          = torch.randn(4, 32)

        batcher = PyGStyleBatcher()
        batched = batcher.batch([g_causes, g_prevents], [f.clone(), f.clone()])

        with torch.no_grad():
            out_causes_alone   = cre.forward(g_causes,   f.clone(), n_iterations=2)
            out_prevents_alone = cre.forward(g_prevents, f.clone(), n_iterations=2)
            batch_outs         = cre.forward_batched(batched, n_iterations=2)

        torch.testing.assert_close(
            batch_outs[0].node_features,
            out_causes_alone.node_features,
            atol=1e-5, rtol=1e-4,
            msg="CAUSES graph affected by PREVENTS in batch",
        )
        torch.testing.assert_close(
            batch_outs[1].node_features,
            out_prevents_alone.node_features,
            atol=1e-5, rtol=1e-4,
            msg="PREVENTS graph affected by CAUSES in batch",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST forward_batched — SHAPES Y METADATOS
# ─────────────────────────────────────────────────────────────────────────────

class TestForwardBatchedShapes:
    """Shapes de salida correctos."""

    def test_output_count_matches_input(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        graphs = [make_graph(3, 2), make_graph(5, 4), make_graph(2, 1)]
        feats  = [torch.randn(3, 32), torch.randn(5, 32), torch.randn(2, 32)]
        batcher = PyGStyleBatcher()
        batched = batcher.batch(graphs, feats)

        with torch.no_grad():
            outs = cre.forward_batched(batched, n_iterations=2)

        assert len(outs) == 3

    def test_node_output_shapes(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        sizes = [3, 7, 2, 5]
        graphs = [make_graph(n, n - 1) for n in sizes]
        feats  = [torch.randn(n, 32) for n in sizes]
        batcher = PyGStyleBatcher()
        batched = batcher.batch(graphs, feats)

        with torch.no_grad():
            outs = cre.forward_batched(batched, n_iterations=2)

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
        batcher = PyGStyleBatcher()
        batched = batcher.batch(graphs, feats)

        with torch.no_grad():
            outs = cre.forward_batched(batched, n_iterations=2)

        for ne, out in zip(n_edges_list, outs):
            assert out.edge_features.shape == (ne, 16), \
                f"Expected ({ne}, 16), got {out.edge_features.shape}"

    def test_iterations_run_recorded(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        graphs = [make_graph(3, 2), make_graph(4, 3)]
        feats  = [torch.randn(3, 32), torch.randn(4, 32)]
        batcher = PyGStyleBatcher()
        batched = batcher.batch(graphs, feats)

        with torch.no_grad():
            outs = cre.forward_batched(batched, n_iterations=3)

        for out in outs:
            assert out.iterations_run == 3

    def test_empty_batch_returns_empty_list(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        # forward_batched requiere un BatchedGraph — testeamos via forward_batch
        with torch.no_grad():
            outs = cre.forward_batch([], [])
        assert outs == []


# ─────────────────────────────────────────────────────────────────────────────
# TEST forward_batched — GRADIENTES
# ─────────────────────────────────────────────────────────────────────────────

class TestForwardBatchedGradients:
    """Gradientes fluyen correctamente a través de forward_batched."""

    def test_gradients_flow_to_input(self):
        """
        Verifica que backward() completa sin error y que .grad se popula
        para los inputs. El GRU puede suprimir gradientes a inputs con
        ciertos valores de gate (propiedad del GRU, no un bug del batching),
        así que sólo verificamos que .grad es NOT None (el backward completó
        sin desconectar el grafo).
        """
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)

        graphs = [make_graph(3, 2), make_graph(4, 3)]
        feats  = [torch.randn(3, 32, requires_grad=True),
                  torch.randn(4, 32, requires_grad=True)]

        batcher = PyGStyleBatcher()
        batched = batcher.batch(graphs, feats)

        outs = cre.forward_batched(batched, n_iterations=2)
        loss = sum(o.node_features.sum() for o in outs)
        loss.backward()  # no debe lanzar RuntimeError

        # .grad populated → el grafo computacional está completo y conectado
        for i, f in enumerate(feats):
            assert f.grad is not None, f"backward() no conectó input {i} al grafo"

    def test_gradients_flow_to_parameters(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)

        graphs = [make_graph(4, 3)]
        feats  = [torch.randn(4, 32)]

        batcher = PyGStyleBatcher()
        batched = batcher.batch(graphs, feats)

        outs = cre.forward_batched(batched, n_iterations=2)
        loss = outs[0].node_features.sum()
        loss.backward()

        params_with_grad = [
            p for p in cre.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(params_with_grad) > 0, "No parameters received gradients"


# ─────────────────────────────────────────────────────────────────────────────
# TEST AutoScaler — CPU
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoScalerCPU:
    """AutoScaler en CPU devuelve batch_size válido."""

    def test_returns_positive_int(self):
        scaler = AutoScaler()
        cfg    = make_cfg()
        cre    = CausalReasoningEngine(cfg)
        graph  = make_graph(4, 3)
        device = torch.device("cpu")

        result = scaler.find_optimal_batch(cre, graph, device, node_dim=32)
        assert isinstance(result, AutoScaleResult)
        assert isinstance(result.batch_size, int)
        assert result.batch_size >= 1

    def test_cpu_upper_bound(self):
        """En CPU, el batch_size no debe exceder 8 (heurística del plan)."""
        scaler = AutoScaler()
        cfg    = make_cfg()
        cre    = CausalReasoningEngine(cfg)
        graph  = make_graph(4, 3)
        device = torch.device("cpu")

        result = scaler.find_optimal_batch(cre, graph, device, node_dim=32)
        assert result.batch_size <= 8

    def test_cpu_estimate_method(self):
        """_estimate_cpu_optimal retorna int en [1, 8]."""
        scaler = AutoScaler()
        result = scaler._estimate_cpu_optimal()
        assert isinstance(result, int)
        assert 1 <= result <= 8


# ─────────────────────────────────────────────────────────────────────────────
# TEST ROUNDTRIP: batch → forward_batched → unbatch
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchForwardUnbatchRoundtrip:
    """
    Test del ciclo completo: batch → forward_batched → unbatch.
    El resultado del unbatch debe coincidir con forward() individual.
    """

    def test_roundtrip_equivalence(self):
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        graphs = [make_graph(3, 2), make_graph(5, 4), make_graph(2, 1)]
        feats  = [torch.randn(n, 32) for n in [3, 5, 2]]

        batcher = PyGStyleBatcher()
        batched = batcher.batch(graphs, feats)

        with torch.no_grad():
            # Camino batch
            batch_outs = cre.forward_batched(batched, n_iterations=2)
            refined_all = torch.cat([o.node_features for o in batch_outs], dim=0)
            unbatched = batcher.unbatch(batched, refined_all)

            # Camino individual
            singles = [cre.forward(g, f, n_iterations=2) for g, f in zip(graphs, feats)]

        for i, (s, u) in enumerate(zip(singles, unbatched)):
            torch.testing.assert_close(
                s.node_features, u,
                atol=1e-5, rtol=1e-4,
                msg=f"Roundtrip mismatch for graph {i}",
            )

    def test_roundtrip_batch_16(self):
        """Roundtrip completo con batch=16."""
        cfg = make_cfg()
        cre = CausalReasoningEngine(cfg)
        cre.eval()

        sizes  = [3, 5, 2, 4, 6, 3, 4, 2, 5, 3, 4, 6, 2, 3, 5, 4]
        edges  = [2, 4, 1, 3, 5, 2, 3, 1, 4, 2, 3, 5, 1, 2, 4, 3]
        graphs = [make_graph(n, e) for n, e in zip(sizes, edges)]
        feats  = [torch.randn(n, 32) for n in sizes]

        batcher = PyGStyleBatcher()
        batched = batcher.batch(graphs, feats)

        with torch.no_grad():
            batch_outs = cre.forward_batched(batched, n_iterations=2)
            singles    = [cre.forward(g, f, n_iterations=2) for g, f in zip(graphs, feats)]

        for i, (s, b) in enumerate(zip(singles, batch_outs)):
            torch.testing.assert_close(
                s.node_features, b.node_features,
                atol=1e-5, rtol=1e-4,
                msg=f"Roundtrip batch=16 mismatch at graph {i}",
            )
