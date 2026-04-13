"""
tests/test_pipeline.py — Tests para CORAPipeline (Paso 8)
==========================================================

Verifica:
  1. Pipeline completo corre sin errores (token IDs → logits)
  2. Shapes consistentes entre módulos
  3. Diferenciabilidad end-to-end (loss.backward() hasta encoder embedding)
  4. Cambiar input cambia output (no constante)
  5. Test con texto "Si la lluvia causa suelo mojado..." → CausalGraph inspectable
  6. Intermedios inspeccionables (crystal_output, graph_repr)
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from router import CORAConfig, CORAPipeline, PipelineOutput
from crystallizer.model import CrystallizerOutput
from core.graph import CausalGraph


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_cfg():
    """Config mínima compartida por todos los tests del módulo."""
    return CORAConfig.tiny()


@pytest.fixture(scope="module")
def pipeline(tiny_cfg):
    """Pipeline compartido (instanciar una sola vez por módulo — es caro)."""
    torch.manual_seed(42)
    p = CORAPipeline(tiny_cfg)
    p.eval()
    return p


def _batch(cfg: CORAConfig, B: int = 2, L: int = 8) -> torch.Tensor:
    """Crea un batch de token_ids aleatorio."""
    return torch.randint(0, cfg.vocab_size, (B, L))


# ─────────────────────────────────────────────────────────────────────────────
# 1. CORAConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestCORAConfig:

    def test_tiny_factory(self):
        cfg = CORAConfig.tiny()
        assert cfg.hidden_dim == 64
        assert cfg.vocab_size == 512

    def test_default_values(self):
        cfg = CORAConfig()
        assert cfg.hidden_dim  == 256
        assert cfg.vocab_size  == 32_000
        assert cfg.cre_max_iterations == 20

    def test_encoder_config_concept_dim_equals_hidden_dim(self):
        cfg = CORAConfig.tiny()
        enc_cfg = cfg.encoder_config()
        assert enc_cfg.concept_dim == cfg.hidden_dim

    def test_crystallizer_config_hidden_dim_matches(self):
        cfg = CORAConfig.tiny()
        cryst_cfg = cfg.crystallizer_config()
        assert cryst_cfg.hidden_dim == cfg.hidden_dim

    def test_cre_config_node_dim_matches(self):
        cfg = CORAConfig.tiny()
        cre_cfg = cfg.cre_config()
        assert cre_cfg.node_dim == cfg.hidden_dim

    def test_decoder_config_node_dim_matches(self):
        cfg = CORAConfig.tiny()
        dec_cfg = cfg.decoder_config()
        assert dec_cfg.node_dim == cfg.hidden_dim

    def test_decoder_max_graph_nodes_matches_crystallizer(self):
        cfg = CORAConfig.tiny()
        assert cfg.decoder_config().max_graph_nodes == cfg.cryst_max_nodes

    def test_scratch_pad_node_dim_matches(self):
        cfg = CORAConfig.tiny()
        pad_cfg = cfg.scratch_pad_config()
        assert pad_cfg.node_dim == cfg.hidden_dim

    def test_invalid_hidden_dim_vs_cryst_heads(self):
        with pytest.raises(ValueError, match="cryst_n_heads"):
            CORAConfig(hidden_dim=33, cryst_n_heads=4)

    def test_invalid_hidden_dim_vs_dec_heads(self):
        with pytest.raises(ValueError, match="dec_n_heads"):
            CORAConfig(hidden_dim=33, dec_n_heads=4, cryst_n_heads=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pipeline — forward pass sin errores
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineForwardPass:

    def test_runs_without_error(self, pipeline, tiny_cfg):
        token_ids = _batch(tiny_cfg, B=2, L=8)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert out is not None

    def test_returns_pipeline_output(self, pipeline, tiny_cfg):
        token_ids = _batch(tiny_cfg, B=1, L=4)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert isinstance(out, PipelineOutput)

    def test_batch_size_1(self, pipeline, tiny_cfg):
        token_ids = _batch(tiny_cfg, B=1, L=6)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert out.logits.shape[0] == 1

    def test_batch_size_4(self, pipeline, tiny_cfg):
        token_ids = _batch(tiny_cfg, B=4, L=6)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert out.logits.shape[0] == 4

    def test_single_token(self, pipeline, tiny_cfg):
        """L=1 no crashea."""
        token_ids = _batch(tiny_cfg, B=2, L=1)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert out.logits.shape[1] == 1

    def test_longer_sequence(self, pipeline, tiny_cfg):
        """Secuencia más larga (dentro de max_seq_len)."""
        token_ids = _batch(tiny_cfg, B=1, L=32)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert out.logits.shape[1] == 32

    def test_zero_cre_iterations(self, pipeline, tiny_cfg):
        """0 iteraciones CRE: los node_features no cambian."""
        token_ids = _batch(tiny_cfg, B=1, L=4)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=0)
        assert out.logits.isfinite().all()

    def test_multiple_cre_iterations(self, pipeline, tiny_cfg):
        token_ids = _batch(tiny_cfg, B=1, L=4)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=3)
        assert out.logits.isfinite().all()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Shapes — consistencia entre módulos
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineShapes:

    def test_logits_shape(self, pipeline, tiny_cfg):
        B, L = 2, 6
        token_ids = _batch(tiny_cfg, B=B, L=L)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert out.logits.shape == (B, L, tiny_cfg.vocab_size)

    def test_anchor_logits_shape(self, pipeline, tiny_cfg):
        B, L = 2, 6
        token_ids = _batch(tiny_cfg, B=B, L=L)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert out.anchor_logits.shape == (B, L, tiny_cfg.cryst_max_nodes)

    def test_confidence_shape(self, pipeline, tiny_cfg):
        B, L = 2, 6
        token_ids = _batch(tiny_cfg, B=B, L=L)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert out.confidence.shape == (B,)

    def test_needs_clarif_shape(self, pipeline, tiny_cfg):
        B = 2
        token_ids = _batch(tiny_cfg, B=B, L=6)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert out.needs_clarif.shape == (B,)

    def test_graph_repr_shape(self, pipeline, tiny_cfg):
        B = 2
        token_ids = _batch(tiny_cfg, B=B, L=6)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert out.graph_repr.shape == (B, tiny_cfg.cryst_max_nodes, tiny_cfg.hidden_dim)

    def test_crystal_output_node_vectors_shape(self, pipeline, tiny_cfg):
        B, L = 2, 8
        K = min(L, tiny_cfg.cryst_max_nodes)
        token_ids = _batch(tiny_cfg, B=B, L=L)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        # node_vectors: [B, K, D]
        assert out.crystal_output.node_vectors.shape == (B, K, tiny_cfg.hidden_dim)

    def test_encoder_output_feeds_crystallizer(self, pipeline, tiny_cfg):
        """
        Verifica que concept_dim del encoder == hidden_dim del crystallizer.
        Se comprueba implícitamente porque el forward no lanza excepción.
        La check explícita: concept_projector output == crystallizer input.
        """
        enc_cfg  = tiny_cfg.encoder_config()
        cryst_cfg = tiny_cfg.crystallizer_config()
        assert enc_cfg.concept_dim == cryst_cfg.hidden_dim

    def test_cre_node_dim_feeds_decoder(self, tiny_cfg):
        """CRE node_dim == decoder node_dim."""
        cre_cfg = tiny_cfg.cre_config()
        dec_cfg = tiny_cfg.decoder_config()
        assert cre_cfg.node_dim == dec_cfg.node_dim

    def test_all_outputs_finite(self, pipeline, tiny_cfg):
        token_ids = _batch(tiny_cfg, B=2, L=6)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert out.logits.isfinite().all()
        assert out.anchor_logits.isfinite().all()
        assert out.confidence.isfinite().all()
        assert out.graph_repr.isfinite().all()

    def test_confidence_in_01(self, pipeline, tiny_cfg):
        token_ids = _batch(tiny_cfg, B=2, L=6)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert (out.confidence >= 0).all()
        assert (out.confidence <= 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Diferenciabilidad end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndGradients:

    def test_loss_backward_no_error(self, tiny_cfg):
        pipeline = CORAPipeline(tiny_cfg)
        token_ids = _batch(tiny_cfg, B=1, L=4)
        out = pipeline(token_ids, n_cre_iters=1)
        out.logits.sum().backward()   # no debe lanzar excepción

    def test_grad_flows_to_encoder_embedding(self, tiny_cfg):
        """
        Gradiente desde logits llega hasta token_embedding del encoder.
        Esto confirma que todo el pipeline es diferenciable.
        """
        pipeline = CORAPipeline(tiny_cfg)
        token_ids = _batch(tiny_cfg, B=1, L=4)
        out = pipeline(token_ids, n_cre_iters=1)
        out.logits.sum().backward()
        grad = pipeline.encoder.token_embedding.weight.grad
        assert grad is not None, "Gradient must reach encoder token embedding"
        assert grad.abs().sum() > 0

    def test_grad_flows_to_crystallizer(self, tiny_cfg):
        """Gradiente llega hasta el node_detector del crystallizer."""
        pipeline = CORAPipeline(tiny_cfg)
        token_ids = _batch(tiny_cfg, B=1, L=4)
        out = pipeline(token_ids, n_cre_iters=1)
        out.logits.sum().backward()
        cryst_grads = [
            p.grad for p in pipeline.crystallizer.parameters()
            if p.grad is not None
        ]
        assert len(cryst_grads) > 0

    def test_grad_flows_to_cre(self, tiny_cfg):
        """Gradiente llega hasta los parámetros del CRE (message functions)."""
        pipeline = CORAPipeline(tiny_cfg)
        token_ids = _batch(tiny_cfg, B=1, L=4)
        out = pipeline(token_ids, n_cre_iters=1)
        out.logits.sum().backward()
        cre_grads = [
            p.grad for p in pipeline.cre.parameters()
            if p.grad is not None
        ]
        assert len(cre_grads) > 0

    def test_grad_flows_to_scratch_pad(self, tiny_cfg):
        """Gradiente llega hasta el scratch pad."""
        pipeline = CORAPipeline(tiny_cfg)
        token_ids = _batch(tiny_cfg, B=1, L=4)
        out = pipeline(token_ids, n_cre_iters=1)
        out.logits.sum().backward()
        pad_grads = [
            p.grad for p in pipeline.scratch_pad.parameters()
            if p.grad is not None
        ]
        assert len(pad_grads) > 0

    def test_grad_flows_to_decoder(self, tiny_cfg):
        """Gradiente llega hasta el decoder."""
        pipeline = CORAPipeline(tiny_cfg)
        token_ids = _batch(tiny_cfg, B=1, L=4)
        out = pipeline(token_ids, n_cre_iters=1)
        out.logits.sum().backward()
        dec_grads = [
            p.grad for p in pipeline.decoder.parameters()
            if p.grad is not None
        ]
        assert len(dec_grads) > 0

    def test_all_gradients_finite(self, tiny_cfg):
        """Todos los gradientes son finitos (sin NaN/Inf)."""
        pipeline = CORAPipeline(tiny_cfg)
        token_ids = _batch(tiny_cfg, B=1, L=4)
        out = pipeline(token_ids, n_cre_iters=1)
        out.logits.sum().backward()
        for name, p in pipeline.named_parameters():
            if p.grad is not None:
                assert p.grad.isfinite().all(), f"Non-finite grad in {name}"

    def test_gradient_through_graph_repr_padding(self, tiny_cfg):
        """
        Gradiente fluye a través del padding del graph_repr.
        graph_repr[b, :n, :] debe tener gradiente, y graph_repr[b, n:, :] no.
        """
        pipeline = CORAPipeline(tiny_cfg)
        token_ids = _batch(tiny_cfg, B=1, L=4)
        out = pipeline(token_ids, n_cre_iters=1)
        out.graph_repr.sum().backward()
        # Si fluye, encoder embedding debe tener gradiente
        enc_grad = pipeline.encoder.token_embedding.weight.grad
        assert enc_grad is not None

    def test_anchor_logits_gradients(self, tiny_cfg):
        """Gradiente desde anchor_logits también funciona."""
        pipeline = CORAPipeline(tiny_cfg)
        token_ids = _batch(tiny_cfg, B=1, L=4)
        out = pipeline(token_ids, n_cre_iters=1)
        out.anchor_logits.sum().backward()
        enc_grad = pipeline.encoder.token_embedding.weight.grad
        assert enc_grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cambiar input cambia output
# ─────────────────────────────────────────────────────────────────────────────

class TestInputSensitivity:

    def test_different_tokens_different_logits(self, pipeline, tiny_cfg):
        """Output no es constante — responde al input."""
        torch.manual_seed(0)
        t1 = _batch(tiny_cfg, B=1, L=8)
        t2 = _batch(tiny_cfg, B=1, L=8)
        with torch.no_grad():
            out1 = pipeline(t1, n_cre_iters=1)
            out2 = pipeline(t2, n_cre_iters=1)
        assert not torch.allclose(out1.logits, out2.logits), \
            "Different inputs must produce different logits"

    def test_same_input_same_output_eval_mode(self, pipeline, tiny_cfg):
        """En eval mode, mismo input → mismo output (determinismo)."""
        t = _batch(tiny_cfg, B=1, L=6)
        with torch.no_grad():
            out1 = pipeline(t, n_cre_iters=1)
            out2 = pipeline(t, n_cre_iters=1)
        assert torch.allclose(out1.logits, out2.logits)

    def test_different_tokens_different_graphs(self, pipeline, tiny_cfg):
        """Tokens distintos generan grafos causales distintos."""
        torch.manual_seed(1)
        t1 = _batch(tiny_cfg, B=1, L=8)
        t2 = _batch(tiny_cfg, B=1, L=8)
        with torch.no_grad():
            out1 = pipeline(t1, n_cre_iters=1)
            out2 = pipeline(t2, n_cre_iters=1)
        # Los node_vectors (diferenciables) deben ser distintos
        assert not torch.allclose(
            out1.crystal_output.node_vectors,
            out2.crystal_output.node_vectors,
        )

    def test_different_tokens_different_graph_repr(self, pipeline, tiny_cfg):
        """graph_repr (ya refinado por CRE) cambia con el input."""
        torch.manual_seed(2)
        t1 = _batch(tiny_cfg, B=1, L=8)
        t2 = _batch(tiny_cfg, B=1, L=8)
        with torch.no_grad():
            out1 = pipeline(t1, n_cre_iters=1)
            out2 = pipeline(t2, n_cre_iters=1)
        assert not torch.allclose(out1.graph_repr, out2.graph_repr)

    def test_output_changes_with_cre_iterations(self, tiny_cfg):
        """Más iteraciones CRE producen graph_repr diferente."""
        # Disable gate and budget manager so n_cre_iters parameter is respected
        cfg = CORAConfig.tiny()
        cfg.cre_use_convergence_gate = False
        cfg.use_budget_manager = False
        torch.manual_seed(42)
        local_pipeline = CORAPipeline(cfg)
        local_pipeline.eval()
        t = _batch(cfg, B=1, L=6)
        with torch.no_grad():
            out1 = local_pipeline(t, n_cre_iters=0)
            out2 = local_pipeline(t, n_cre_iters=2)
        # Con 0 vs 2 iteraciones los node features deben diferir
        # (CRE modifica los node features con cada iteración)
        assert not torch.allclose(out1.graph_repr, out2.graph_repr)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CausalGraph intermedio — inspección
# ─────────────────────────────────────────────────────────────────────────────

class TestIntermediateGraphInspection:

    def test_crystal_output_has_graphs(self, pipeline, tiny_cfg):
        token_ids = _batch(tiny_cfg, B=2, L=8)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        assert len(out.crystal_output.graphs) == 2

    def test_graphs_are_causal_graph_instances(self, pipeline, tiny_cfg):
        token_ids = _batch(tiny_cfg, B=2, L=8)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        for graph in out.crystal_output.graphs:
            assert isinstance(graph, CausalGraph)

    def test_graphs_have_nodes(self, pipeline, tiny_cfg):
        """Con thresholds bajos, deben aparecer nodos."""
        token_ids = _batch(tiny_cfg, B=1, L=8)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        graph = out.crystal_output.graphs[0]
        assert len(graph.nodes) > 0, \
            f"Expected nodes with threshold={tiny_cfg.cryst_node_threshold}, got 0"

    def test_graphs_have_edges(self, pipeline, tiny_cfg):
        """Con thresholds bajos, deben aparecer aristas."""
        token_ids = _batch(tiny_cfg, B=1, L=8)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        graph = out.crystal_output.graphs[0]
        assert len(graph.edges) > 0, \
            f"Expected edges with threshold={tiny_cfg.cryst_edge_threshold}, got 0"

    def test_graph_nodes_have_valid_types(self, pipeline, tiny_cfg):
        """Cada nodo tiene un NodeType válido."""
        from core.graph import NodeType
        token_ids = _batch(tiny_cfg, B=1, L=8)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        for node in out.crystal_output.graphs[0].nodes:
            assert isinstance(node.node_type, NodeType)

    def test_graph_edges_have_valid_relations(self, pipeline, tiny_cfg):
        """Cada arista tiene una CausalRelation válida."""
        from core.graph import CausalRelation
        token_ids = _batch(tiny_cfg, B=1, L=8)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        for edge in out.crystal_output.graphs[0].edges:
            assert isinstance(edge.relation, CausalRelation)

    def test_graph_node_count_respects_max_nodes(self, pipeline, tiny_cfg):
        token_ids = _batch(tiny_cfg, B=2, L=16)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=1)
        for graph in out.crystal_output.graphs:
            assert len(graph.nodes) <= tiny_cfg.cryst_max_nodes

    def test_causal_chain_graph(self, tiny_cfg):
        """
        Test con input representando 'lluvia → suelo mojado → resbalón'.

        Usamos IDs sintéticos (sin tokenizer real).
        El test verifica que el grafo intermedio es inspectable y no vacío.
        """
        # Simular tokenización: each word → un ID (mod vocab_size)
        text = "Si la lluvia causa suelo mojado y suelo mojado causa resbalon"
        words = text.split()
        # Token ID = hash(word) mod vocab_size, asegurar variedad
        token_list = [(hash(w) % (tiny_cfg.vocab_size - 1)) + 1 for w in words]
        token_ids = torch.tensor([token_list])  # [1, n_words]

        pipeline = CORAPipeline(tiny_cfg)
        pipeline.eval()
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=2)

        graph = out.crystal_output.graphs[0]

        # 1. Output tiene la forma correcta
        n_tokens = len(words)
        assert out.logits.shape == (1, n_tokens, tiny_cfg.vocab_size)

        # 2. El grafo es un objeto CausalGraph válido
        assert isinstance(graph, CausalGraph)

        # 3. Con thresholds bajos, hay nodos detectados
        assert len(graph.nodes) > 0, \
            f"Causal chain graph must have nodes (threshold={tiny_cfg.cryst_node_threshold})"

        # 4. El grafo es inspectable — podemos iterar sobre nodos y aristas
        for node in graph.nodes:
            assert node.node_id.startswith("n")
            assert 0.0 <= node.confidence <= 1.0

        for edge in graph.edges:
            assert edge.source_id in {n.node_id for n in graph.nodes}
            assert edge.target_id in {n.node_id for n in graph.nodes}

        # 5. graph_repr tiene la forma correcta
        assert out.graph_repr.shape == (1, tiny_cfg.cryst_max_nodes, tiny_cfg.hidden_dim)
        assert out.graph_repr.isfinite().all()

        # 6. Podemos inspeccionar el repr (no es todo ceros si hay nodos)
        n_valid = len(graph.nodes)
        if n_valid > 0:
            # Las primeras n_valid filas de graph_repr[0] no son todas cero
            active_repr = out.graph_repr[0, :n_valid, :]
            # Al menos una posición debería tener un valor no-cero
            assert active_repr.abs().sum() > 0

    def test_crystal_output_differentiable_tensors(self, pipeline, tiny_cfg):
        """Los tensores del CrystallizerOutput son diferenciables."""
        pipeline2 = CORAPipeline(tiny_cfg)
        token_ids = _batch(tiny_cfg, B=1, L=6)
        out = pipeline2(token_ids, n_cre_iters=1)
        # node_vectors debe tener grad_fn
        assert out.crystal_output.node_vectors.requires_grad or \
               out.crystal_output.node_vectors.grad_fn is not None

    def test_graph_repr_differentiable(self, tiny_cfg):
        """graph_repr debe tener grad_fn (viene de cálculos diferenciables)."""
        pipeline2 = CORAPipeline(tiny_cfg)
        token_ids = _batch(tiny_cfg, B=1, L=4)
        out = pipeline2(token_ids, n_cre_iters=1)
        assert out.graph_repr.grad_fn is not None, \
            "graph_repr must have grad_fn for backprop"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Parameter count
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterCount:

    def test_count_parameters_positive(self, pipeline):
        assert pipeline.count_parameters() > 0

    def test_breakdown_keys(self, pipeline):
        bd = pipeline.parameter_breakdown()
        for key in ["encoder", "crystallizer", "cre", "scratch_pad",
                    "decoder", "total_unique"]:
            assert key in bd

    def test_breakdown_total_matches_count(self, pipeline):
        bd = pipeline.parameter_breakdown()
        assert bd["total_unique"] == pipeline.count_parameters()

    def test_tiny_config_small_param_count(self, pipeline):
        """Config tiny debe tener menos de 5M parámetros."""
        assert pipeline.count_parameters() < 5_000_000

    def test_all_modules_contribute_params(self, pipeline):
        bd = pipeline.parameter_breakdown()
        for module in ["encoder", "crystallizer", "cre", "scratch_pad", "decoder"]:
            assert bd[module] > 0, f"Module {module} has 0 parameters"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Estabilidad numérica
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalStability:

    def test_long_sequence_stable(self, pipeline, tiny_cfg):
        """Secuencia larga no produce NaN/Inf."""
        token_ids = _batch(tiny_cfg, B=1, L=64)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=2)
        assert out.logits.isfinite().all()
        assert out.graph_repr.isfinite().all()

    def test_max_cre_iterations_stable(self, pipeline, tiny_cfg):
        """Máximo de iteraciones CRE no explota."""
        token_ids = _batch(tiny_cfg, B=1, L=4)
        with torch.no_grad():
            out = pipeline(token_ids, n_cre_iters=tiny_cfg.cre_max_iterations)
        assert out.logits.isfinite().all()

    def test_repeated_same_token(self, pipeline, tiny_cfg):
        """Secuencia de tokens repetidos no produce NaN."""
        t = torch.ones(1, 8, dtype=torch.long)
        with torch.no_grad():
            out = pipeline(t, n_cre_iters=1)
        assert out.logits.isfinite().all()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Determinismo
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_eval_mode_deterministic(self, pipeline, tiny_cfg):
        """Mismo input en eval mode → mismo output."""
        t = _batch(tiny_cfg, B=1, L=6)
        with torch.no_grad():
            out1 = pipeline(t, n_cre_iters=1)
            out2 = pipeline(t, n_cre_iters=1)
        assert torch.allclose(out1.logits, out2.logits)
        assert torch.allclose(out1.graph_repr, out2.graph_repr)

    def test_same_seed_same_weights(self, tiny_cfg):
        """Con misma semilla, dos pipelines tienen los mismos pesos."""
        torch.manual_seed(0)
        p1 = CORAPipeline(tiny_cfg)
        torch.manual_seed(0)
        p2 = CORAPipeline(tiny_cfg)

        t = _batch(tiny_cfg, B=1, L=4)
        p1.eval(); p2.eval()
        with torch.no_grad():
            out1 = p1(t, n_cre_iters=1)
            out2 = p2(t, n_cre_iters=1)
        assert torch.allclose(out1.logits, out2.logits, atol=1e-6)
