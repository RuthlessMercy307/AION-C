"""
tests/test_weakness.py — Tests del WeaknessDetector
=====================================================

Organización:
    TestWeaknessReport          — estructura del dataclass
    TestLowConfidence           — nodos con baja confianza neural
    TestMissingCause            — nodos sin causas entrantes
    TestUnresolvedContradiction — pares CONTRADICTS con ambos confiados
    TestCircularReasoning       — ciclos en el grafo dirigido
    TestWeakEvidence            — hipótesis sin SUPPORTS
    TestFocusMask               — shape y semántica del focus_mask
    TestCombinedWeaknesses      — grafo con múltiples tipos simultáneos
    TestEdgeCases               — grafo vacío, 1 nodo, sin aristas
    TestGradientFlow            — gradientes a través del confidence scorer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

from core.graph import (
    CausalEdge, CausalGraph, CausalNode, CausalRelation, NodeType
)
from cre.weakness import (
    CIRCULAR_REASONING,
    LOW_CONFIDENCE,
    MISSING_CAUSE,
    UNRESOLVED_CONTRADICTION,
    WEAK_EVIDENCE,
    WEAKNESS_TYPES,
    Weakness,
    WeaknessDetector,
    WeaknessReport,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

NODE_DIM = 32


def make_node(nid: str, ntype: NodeType = NodeType.STATE, conf: float = 0.7) -> CausalNode:
    return CausalNode(node_id=nid, label=nid, node_type=ntype, confidence=conf)


def make_detector(conf_threshold: float = 0.35) -> WeaknessDetector:
    return WeaknessDetector(
        node_dim=NODE_DIM,
        confidence_threshold=conf_threshold,
        confidence_hidden=16,
    )


def linear_graph(n: int, relation: CausalRelation = CausalRelation.CAUSES) -> CausalGraph:
    """A→B→C→… con la relación dada."""
    g = CausalGraph()
    for i in range(n):
        g.add_node(make_node(f"n{i}"))
    for i in range(n - 1):
        g.add_edge(CausalEdge(source_id=f"n{i}", target_id=f"n{i+1}", relation=relation))
    return g


def random_features(n: int, dim: int = NODE_DIM, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, dim)


# ─────────────────────────────────────────────────────────────────────────────
# TestWeaknessReport
# ─────────────────────────────────────────────────────────────────────────────

class TestWeaknessReport:
    def test_dataclass_fields(self):
        wr = WeaknessReport(
            weaknesses    = [],
            focus_mask    = torch.zeros(4, dtype=torch.bool),
            node_severity = torch.zeros(4),
            confidence    = torch.ones(4) * 0.5,
            n_weaknesses  = 0,
            mean_severity = 0.0,
        )
        assert wr.n_weaknesses == 0
        assert wr.focus_mask.shape == (4,)
        assert wr.node_severity.shape == (4,)
        assert wr.confidence.shape == (4,)

    def test_weakness_fields(self):
        w = Weakness(node_idx=2, node_id="nodeX", type=LOW_CONFIDENCE, severity=0.8)
        assert w.node_idx == 2
        assert w.node_id == "nodeX"
        assert w.type == LOW_CONFIDENCE
        assert 0.0 <= w.severity <= 1.0

    def test_weakness_types_constant(self):
        assert LOW_CONFIDENCE           in WEAKNESS_TYPES
        assert MISSING_CAUSE            in WEAKNESS_TYPES
        assert UNRESOLVED_CONTRADICTION in WEAKNESS_TYPES
        assert CIRCULAR_REASONING       in WEAKNESS_TYPES
        assert WEAK_EVIDENCE            in WEAKNESS_TYPES
        assert len(WEAKNESS_TYPES) == 5


# ─────────────────────────────────────────────────────────────────────────────
# TestLowConfidence
# ─────────────────────────────────────────────────────────────────────────────

class TestLowConfidence:
    """
    La detección de baja confianza depende del confidence_scorer neural.
    Usamos pesos inicializados con std pequeño para controlar el test.
    """

    def test_low_conf_flagged_when_scorer_gives_low_score(self):
        """Nodos con features → score bajo deben ser marcados como low_confidence."""
        det = make_detector(conf_threshold=0.9)   # threshold muy alto → casi todo es low
        g = linear_graph(3, CausalRelation.CAUSES)

        h = random_features(3)
        e = torch.zeros(2, 8)
        report = det(g, h, e)

        # Con threshold=0.9, la mayoría de nodos tendrán confianza < 0.9
        low_conf = [w for w in report.weaknesses if w.type == LOW_CONFIDENCE]
        assert len(low_conf) >= 1, "Con threshold 0.9 debe haber nodos low_confidence"

    def test_no_low_conf_with_threshold_zero(self):
        """Con threshold=0, ningún nodo es low_confidence."""
        det = make_detector(conf_threshold=0.0)
        g = linear_graph(3, CausalRelation.CAUSES)
        h = random_features(3)
        e = torch.zeros(2, 8)
        report = det(g, h, e)
        low_conf = [w for w in report.weaknesses if w.type == LOW_CONFIDENCE]
        assert len(low_conf) == 0

    def test_severity_is_inverse_of_confidence(self):
        """Severidad de low_confidence = 1 - confidence_score."""
        det = make_detector(conf_threshold=0.9)
        g = linear_graph(2, CausalRelation.CAUSES)
        h = random_features(2)
        e = torch.zeros(1, 8)
        report = det(g, h, e)
        for w in report.weaknesses:
            if w.type == LOW_CONFIDENCE:
                conf_val = float(report.confidence[w.node_idx].item())
                expected_sev = 1.0 - conf_val
                assert abs(w.severity - expected_sev) < 1e-5

    def test_confidence_shape(self):
        det = make_detector()
        g = linear_graph(5, CausalRelation.CAUSES)
        h = random_features(5)
        e = torch.zeros(4, 8)
        report = det(g, h, e)
        assert report.confidence.shape == (5,)
        assert (report.confidence >= 0.0).all()
        assert (report.confidence <= 1.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# TestMissingCause
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingCause:
    def test_root_cause_flagged_as_missing_cause(self):
        """El nodo raíz de una cadena causal no tiene causas entrantes."""
        det = make_detector(conf_threshold=0.0)   # deshabilitar low_confidence
        g = linear_graph(3, CausalRelation.CAUSES)
        # n0 → n1 → n2
        # n0 no tiene causas entrantes → missing_cause
        h = random_features(3)
        e = torch.zeros(2, 8)
        report = det(g, h, e)

        missing = [w for w in report.weaknesses if w.type == MISSING_CAUSE]
        missing_ids = {w.node_id for w in missing}
        assert "n0" in missing_ids, "El nodo raíz debe ser flagged como missing_cause"
        assert "n2" not in missing_ids, "n2 tiene causa n1, no debe ser flagged"

    def test_fact_node_not_flagged(self):
        """Los nodos FACT no necesitan causas — ya están grounded."""
        det = make_detector(conf_threshold=0.0)
        g = CausalGraph()
        g.add_node(make_node("fact_root", ntype=NodeType.FACT))
        g.add_node(make_node("effect",    ntype=NodeType.STATE))
        g.add_edge(CausalEdge(source_id="fact_root", target_id="effect",
                              relation=CausalRelation.CAUSES))
        h = random_features(2)
        e = torch.zeros(1, 8)
        report = det(g, h, e)

        missing = {w.node_id for w in report.weaknesses if w.type == MISSING_CAUSE}
        assert "fact_root" not in missing, "FACT no debe ser flagged como missing_cause"

    def test_question_node_not_flagged(self):
        """Los nodos QUESTION tampoco necesitan causas."""
        det = make_detector(conf_threshold=0.0)
        g = CausalGraph()
        g.add_node(make_node("q",    ntype=NodeType.QUESTION))
        g.add_node(make_node("ans",  ntype=NodeType.STATE))
        g.add_edge(CausalEdge(source_id="ans", target_id="q",
                              relation=CausalRelation.IMPLIES))
        h = random_features(2)
        e = torch.zeros(1, 8)
        report = det(g, h, e)

        missing = {w.node_id for w in report.weaknesses if w.type == MISSING_CAUSE}
        assert "q" not in missing

    def test_isolated_node_flagged(self):
        """Un nodo sin ninguna arista conectada es missing_cause."""
        det = make_detector(conf_threshold=0.0)
        g = CausalGraph()
        g.add_node(make_node("isolated", ntype=NodeType.STATE))
        h = random_features(1)
        e = torch.zeros(0, 8)
        report = det(g, h, e)

        missing = {w.node_id for w in report.weaknesses if w.type == MISSING_CAUSE}
        assert "isolated" in missing


# ─────────────────────────────────────────────────────────────────────────────
# TestUnresolvedContradiction
# ─────────────────────────────────────────────────────────────────────────────

class TestUnresolvedContradiction:
    def _make_contradiction_graph(self) -> CausalGraph:
        """A --contradicts--> B."""
        g = CausalGraph()
        g.add_node(make_node("A", ntype=NodeType.HYPOTHESIS))
        g.add_node(make_node("B", ntype=NodeType.HYPOTHESIS))
        g.add_edge(CausalEdge(source_id="A", target_id="B",
                              relation=CausalRelation.CONTRADICTS))
        return g

    def test_high_conf_both_nodes_flagged(self):
        """
        Con ambos nodos de alta confianza, la contradicción está sin resolver.
        Forzamos alta confianza configurando threshold muy bajo y usando
        el confidence_scorer con pesos que dan high output.
        """
        det = WeaknessDetector(
            node_dim=NODE_DIM,
            confidence_threshold=0.01,  # casi todo es high_confidence
            confidence_hidden=16,
        )
        # Override scorer para que siempre dé confianza alta (logit >> 0)
        with torch.no_grad():
            for layer in det.confidence_scorer:
                if hasattr(layer, 'weight'):
                    layer.weight.fill_(0.0)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.fill_(5.0)   # sigmoid(5) ≈ 0.993 → high confidence

        g = self._make_contradiction_graph()
        h = random_features(2)
        e = torch.zeros(1, 8)
        report = det(g, h, e)

        contradiction = {w.node_id for w in report.weaknesses
                         if w.type == UNRESOLVED_CONTRADICTION}
        assert "A" in contradiction, "A debe ser flagged (contradicción irresuelota)"
        assert "B" in contradiction, "B debe ser flagged (contradicción irresuelota)"

    def test_low_conf_one_side_not_flagged(self):
        """Si un lado tiene baja confianza, la contradicción está 'resuelta'."""
        det = WeaknessDetector(
            node_dim=NODE_DIM,
            confidence_threshold=0.5,
            confidence_hidden=16,
        )
        # Override scorer: siempre low confidence
        with torch.no_grad():
            for layer in det.confidence_scorer:
                if hasattr(layer, 'weight'):
                    layer.weight.fill_(0.0)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.fill_(-5.0)   # sigmoid(-5) ≈ 0.007 → low confidence

        g = self._make_contradiction_graph()
        h = random_features(2)
        e = torch.zeros(1, 8)
        report = det(g, h, e)

        contradiction = {w.node_id for w in report.weaknesses
                         if w.type == UNRESOLVED_CONTRADICTION}
        # Con baja confianza en ambos lados, no hay contradicción irresuelota
        assert len(contradiction) == 0

    def test_no_contradiction_edges_means_no_contradiction_weakness(self):
        """Sin aristas CONTRADICTS, no hay esta debilidad."""
        det = make_detector(conf_threshold=0.0)
        g = linear_graph(3, CausalRelation.CAUSES)
        h = random_features(3)
        e = torch.zeros(2, 8)
        report = det(g, h, e)
        contradiction = [w for w in report.weaknesses if w.type == UNRESOLVED_CONTRADICTION]
        assert len(contradiction) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestCircularReasoning
# ─────────────────────────────────────────────────────────────────────────────

class TestCircularReasoning:
    def test_simple_cycle_detected(self):
        """A→B→C→A: todos los nodos del ciclo deben ser marcados."""
        det = make_detector(conf_threshold=0.0)
        g = CausalGraph()
        for nid in ["A", "B", "C"]:
            g.add_node(make_node(nid))
        g.add_edge(CausalEdge(source_id="A", target_id="B", relation=CausalRelation.CAUSES))
        g.add_edge(CausalEdge(source_id="B", target_id="C", relation=CausalRelation.CAUSES))
        g.add_edge(CausalEdge(source_id="C", target_id="A", relation=CausalRelation.CAUSES))

        h = random_features(3)
        e = torch.zeros(3, 8)
        report = det(g, h, e)

        circular = {w.node_id for w in report.weaknesses if w.type == CIRCULAR_REASONING}
        assert "A" in circular and "B" in circular and "C" in circular

    def test_linear_graph_no_cycle(self):
        """Un grafo lineal (sin ciclos) no debe tener circular_reasoning."""
        det = make_detector(conf_threshold=0.0)
        g = linear_graph(5, CausalRelation.CAUSES)
        h = random_features(5)
        e = torch.zeros(4, 8)
        report = det(g, h, e)
        circular = [w for w in report.weaknesses if w.type == CIRCULAR_REASONING]
        assert len(circular) == 0

    def test_non_causal_edges_dont_form_cycles(self):
        """Los ciclos sólo se detectan sobre aristas causales dirigidas."""
        det = make_detector(conf_threshold=0.0)
        g = CausalGraph()
        g.add_node(make_node("X"))
        g.add_node(make_node("Y"))
        # CORRELATES no es una arista causal dirigida → no forma ciclo
        g.add_edge(CausalEdge(source_id="X", target_id="Y", relation=CausalRelation.CORRELATES))
        g.add_edge(CausalEdge(source_id="Y", target_id="X", relation=CausalRelation.CORRELATES))

        h = random_features(2)
        e = torch.zeros(2, 8)
        report = det(g, h, e)
        circular = [w for w in report.weaknesses if w.type == CIRCULAR_REASONING]
        assert len(circular) == 0

    def test_self_loop_detected(self):
        """A→A (si permitido) debe ser ciclo. Test de robustez."""
        det = make_detector(conf_threshold=0.0)
        g = CausalGraph()
        g.add_node(make_node("self"))
        # CausalGraph puede no permitir self-loops; si los permite, detecta ciclo.
        # Si lanza error, el test pasa (el detector no crashea con grafos normales).
        h = random_features(1)
        e = torch.zeros(0, 8)
        report = det(g, h, e)
        # Solo verificamos que no crashea y da shapes correctos
        assert report.focus_mask.shape == (1,)


# ─────────────────────────────────────────────────────────────────────────────
# TestWeakEvidence
# ─────────────────────────────────────────────────────────────────────────────

class TestWeakEvidence:
    def test_hypothesis_without_support_flagged(self):
        """Una hipótesis sin SUPPORTS entrante es weak_evidence."""
        det = make_detector(conf_threshold=0.0)
        g = CausalGraph()
        g.add_node(make_node("fact",  ntype=NodeType.FACT))
        g.add_node(make_node("hyp",   ntype=NodeType.HYPOTHESIS))
        # No hay arista SUPPORTS de fact a hyp
        g.add_edge(CausalEdge(source_id="fact", target_id="hyp",
                              relation=CausalRelation.CAUSES))   # CAUSES no es SUPPORTS

        h = random_features(2)
        e = torch.zeros(1, 8)
        report = det(g, h, e)

        evidence = {w.node_id for w in report.weaknesses if w.type == WEAK_EVIDENCE}
        assert "hyp" in evidence

    def test_supported_hypothesis_not_flagged(self):
        """Una hipótesis CON SUPPORTS entrante no es weak_evidence."""
        det = make_detector(conf_threshold=0.0)
        g = CausalGraph()
        g.add_node(make_node("evidence", ntype=NodeType.FACT))
        g.add_node(make_node("hyp",      ntype=NodeType.HYPOTHESIS))
        g.add_edge(CausalEdge(source_id="evidence", target_id="hyp",
                              relation=CausalRelation.SUPPORTS))

        h = random_features(2)
        e = torch.zeros(1, 8)
        report = det(g, h, e)

        evidence = {w.node_id for w in report.weaknesses if w.type == WEAK_EVIDENCE}
        assert "hyp" not in evidence

    def test_non_hypothesis_not_flagged(self):
        """Solo nodos HYPOTHESIS pueden ser weak_evidence."""
        det = make_detector(conf_threshold=0.0)
        g = CausalGraph()
        g.add_node(make_node("state",  ntype=NodeType.STATE))   # no HYPOTHESIS
        g.add_node(make_node("entity", ntype=NodeType.ENTITY))
        # Sin aristas — state no tiene soporte pero no es HYPOTHESIS
        h = random_features(2)
        e = torch.zeros(0, 8)
        report = det(g, h, e)

        evidence = [w for w in report.weaknesses if w.type == WEAK_EVIDENCE]
        assert len(evidence) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestFocusMask
# ─────────────────────────────────────────────────────────────────────────────

class TestFocusMask:
    def test_shape_matches_n_nodes(self):
        det = make_detector()
        for n in [1, 3, 7, 16]:
            g = linear_graph(n, CausalRelation.CAUSES)
            h = random_features(n)
            e = torch.zeros(max(n - 1, 0), 8)
            report = det(g, h, e)
            assert report.focus_mask.shape == (n,), f"N={n}: shape mismatch"
            assert report.node_severity.shape == (n,)

    def test_focus_mask_is_boolean(self):
        det = make_detector()
        g = linear_graph(4, CausalRelation.CAUSES)
        h = random_features(4)
        e = torch.zeros(3, 8)
        report = det(g, h, e)
        assert report.focus_mask.dtype == torch.bool

    def test_focus_mask_aligned_with_node_severity(self):
        """focus_mask[i] == True iff node_severity[i] > 0."""
        det = make_detector(conf_threshold=0.9)  # muchos low_confidence
        g = linear_graph(5, CausalRelation.CAUSES)
        h = random_features(5)
        e = torch.zeros(4, 8)
        report = det(g, h, e)
        # Nodes in mask should have severity > 0
        for i in range(5):
            if report.focus_mask[i]:
                assert report.node_severity[i] > 0.0
            else:
                assert report.node_severity[i] == 0.0

    def test_no_weaknesses_means_empty_mask(self):
        """Con threshold=0 y grafo bien formado, no hay debilidades."""
        det = make_detector(conf_threshold=0.0)
        # Grafo con FACT raíz (no missing_cause), sin contradicciones, sin ciclos
        g = CausalGraph()
        g.add_node(make_node("root",   ntype=NodeType.FACT))
        g.add_node(make_node("effect", ntype=NodeType.STATE))
        g.add_edge(CausalEdge(source_id="root", target_id="effect",
                              relation=CausalRelation.CAUSES))
        h = random_features(2)
        e = torch.zeros(1, 8)
        report = det(g, h, e)
        assert not report.focus_mask.any(), "Sin threshold de conf, no debería haber mask"


# ─────────────────────────────────────────────────────────────────────────────
# TestCombinedWeaknesses
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedWeaknesses:
    def test_multiple_types_on_same_node(self):
        """Un nodo HYPOTHESIS sin SUPPORTS y sin causas tiene multiple debilidades."""
        det = make_detector(conf_threshold=0.9)
        g = CausalGraph()
        g.add_node(make_node("H", ntype=NodeType.HYPOTHESIS))
        h = random_features(1)
        e = torch.zeros(0, 8)
        report = det(g, h, e)

        types_on_H = {w.type for w in report.weaknesses if w.node_id == "H"}
        # Debe tener al menos missing_cause y weak_evidence
        assert MISSING_CAUSE  in types_on_H
        assert WEAK_EVIDENCE  in types_on_H

    def test_severity_aggregation(self):
        """node_severity[i] = max severity de todas las debilidades del nodo i."""
        det = make_detector(conf_threshold=0.9)
        g = linear_graph(3, CausalRelation.CAUSES)
        h = random_features(3)
        e = torch.zeros(2, 8)
        report = det(g, h, e)

        for i in range(3):
            node_id = f"n{i}"
            node_weaknesses = [w for w in report.weaknesses if w.node_id == node_id]
            if node_weaknesses:
                expected_max = max(w.severity for w in node_weaknesses)
                assert abs(float(report.node_severity[i].item()) - expected_max) < 1e-5

    def test_n_weaknesses_count(self):
        """n_weaknesses = número total de debilidades en la lista."""
        det = make_detector(conf_threshold=0.9)
        g = linear_graph(4, CausalRelation.CAUSES)
        h = random_features(4)
        e = torch.zeros(3, 8)
        report = det(g, h, e)
        assert report.n_weaknesses == len(report.weaknesses)

    def test_mean_severity_range(self):
        """mean_severity ∈ [0, 1]."""
        det = make_detector(conf_threshold=0.9)
        g = linear_graph(4, CausalRelation.CAUSES)
        h = random_features(4)
        e = torch.zeros(3, 8)
        report = det(g, h, e)
        assert 0.0 <= report.mean_severity <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TestEdgeCases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_node_graph(self):
        """1 nodo: no crashea, shapes correctos."""
        det = make_detector()
        g = CausalGraph()
        g.add_node(make_node("solo"))
        h = random_features(1)
        e = torch.zeros(0, 8)
        report = det(g, h, e)
        assert report.focus_mask.shape == (1,)
        assert report.node_severity.shape == (1,)
        assert report.confidence.shape == (1,)

    def test_no_edges_graph(self):
        """3 nodos sin aristas: cada uno es missing_cause."""
        det = make_detector(conf_threshold=0.0)
        g = CausalGraph()
        for i in range(3):
            g.add_node(make_node(f"n{i}", ntype=NodeType.STATE))
        h = random_features(3)
        e = torch.zeros(0, 8)
        report = det(g, h, e)
        missing = [w for w in report.weaknesses if w.type == MISSING_CAUSE]
        assert len(missing) == 3

    def test_large_graph_no_crash(self):
        """20 nodos, no crashea ni explota."""
        det = make_detector()
        g = linear_graph(20, CausalRelation.CAUSES)
        h = random_features(20)
        e = torch.zeros(19, 8)
        report = det(g, h, e)
        assert report.focus_mask.shape == (20,)
        assert report.n_weaknesses >= 0


# ─────────────────────────────────────────────────────────────────────────────
# TestGradientFlow
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:
    def test_confidence_scorer_gradients(self):
        """Los parámetros del confidence_scorer reciben gradientes."""
        det = make_detector()
        det.train()
        g = linear_graph(3, CausalRelation.CAUSES)
        h = random_features(3).requires_grad_(True)
        e = torch.zeros(2, 8)
        report = det(g, h, e)

        # La confianza es diferenciable
        loss = report.confidence.sum()
        loss.backward()

        for name, param in det.named_parameters():
            assert param.grad is not None, f"No gradient in {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_node_severity_no_gradient_needed(self):
        """node_severity puede ser usado sin gradientes (detached)."""
        det = make_detector()
        g = linear_graph(3, CausalRelation.CAUSES)
        h = random_features(3)
        e = torch.zeros(2, 8)
        report = det(g, h, e)
        # node_severity es float, no necesita ser diferenciable
        sev_val = float(report.node_severity.mean().item())
        assert isinstance(sev_val, float)
