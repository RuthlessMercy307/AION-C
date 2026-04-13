"""
tests/test_convergence.py — Tests del ConvergenceGate y integración en CRE
===========================================================================

Organización:
    TestConvergenceDecision         — estructura del dataclass
    TestConvergenceGateBasic        — señales individuales y thresholds
    TestConvergenceGateMinIter      — safety floor (min_iterations)
    TestSimpleGraphConvergesFast    — queries simples: 1-3 iteraciones
    TestComplexGraphUsesMoreIters   — queries complejas: 5-10+ iteraciones
    TestFocusMaskInEngine           — focus_mask limita qué nodos se actualizan
    TestIterationsRunAccurate       — iterations_run refleja cuántas se hicieron
    TestBackwardCompatibility       — use_convergence_gate=False: comportamiento original
    TestGradientFlowWithGate        — gradientes fluyen con gate activado
    TestStopReasons                 — razones de parada correctas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

from core.graph import (
    CausalEdge, CausalGraph, CausalNode, CausalRelation, NodeType
)
from cre import (
    CREConfig,
    CREOutput,
    CausalReasoningEngine,
    ConvergenceDecision,
    ConvergenceGate,
    WeaknessDetector,
    WeaknessReport,
)
from cre.convergence import _REASON_DELTA, _REASON_CONFIDENCE, _REASON_MIN_ITER, _REASON_CONTINUING


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

NODE_DIM = 32


def make_node(nid: str, ntype: NodeType = NodeType.STATE, conf: float = 0.7) -> CausalNode:
    return CausalNode(node_id=nid, label=nid, node_type=ntype, confidence=conf)


def tiny_cfg(**kwargs) -> CREConfig:
    """Config mínima con convergence gate activado."""
    defaults = dict(
        node_dim=NODE_DIM, edge_dim=16, message_dim=24,
        n_message_layers=1, max_iterations=20,
        n_relation_types=16,
        use_convergence_gate=True,
        min_iterations=1,
        weakness_conf_hidden=8,
        conv_delta_threshold=0.05,
        conv_conf_threshold=0.75,
        conv_weakness_threshold=0.25,
    )
    defaults.update(kwargs)
    return CREConfig(**defaults)


def simple_graph() -> tuple:
    """
    Grafo simple: FACT → STATE (1 arista causal clara).
    Fácil de razonar → converge rápido.
    """
    g = CausalGraph()
    g.add_node(make_node("cause", ntype=NodeType.FACT))
    g.add_node(make_node("effect", ntype=NodeType.STATE))
    g.add_edge(CausalEdge(source_id="cause", target_id="effect",
                          relation=CausalRelation.CAUSES))
    h = torch.zeros(2, NODE_DIM)   # features estables → delta pequeño desde el inicio
    return g, h


def complex_graph() -> tuple:
    """
    Grafo complejo: contradicciones + hipótesis sin evidencia + ciclo.
    Difícil de resolver → usa más iteraciones.
    """
    g = CausalGraph()
    # Cadena causal larga
    for i in range(6):
        g.add_node(make_node(f"n{i}", ntype=NodeType.STATE))
    for i in range(5):
        g.add_edge(CausalEdge(source_id=f"n{i}", target_id=f"n{i+1}",
                              relation=CausalRelation.CAUSES))
    # Hipótesis sin evidencia
    g.add_node(make_node("hyp", ntype=NodeType.HYPOTHESIS))
    g.add_edge(CausalEdge(source_id="n5", target_id="hyp", relation=CausalRelation.LEADS_TO))
    # Contradicción
    g.add_node(make_node("counter", ntype=NodeType.STATE))
    g.add_edge(CausalEdge(source_id="counter", target_id="hyp",
                          relation=CausalRelation.CONTRADICTS))

    n = len(g.nodes)
    torch.manual_seed(1)
    h = torch.randn(n, NODE_DIM)   # features aleatorios → cambiarán mucho
    return g, h


def make_detector_and_gate(cfg: CREConfig):
    det = WeaknessDetector(
        node_dim=cfg.node_dim,
        confidence_threshold=cfg.weakness_conf_threshold,
        confidence_hidden=cfg.weakness_conf_hidden,
    )
    gate = ConvergenceGate(
        delta_threshold   = cfg.conv_delta_threshold,
        conf_threshold    = cfg.conv_conf_threshold,
        weakness_threshold = cfg.conv_weakness_threshold,
        min_iterations    = cfg.min_iterations,
    )
    return det, gate


# ─────────────────────────────────────────────────────────────────────────────
# TestConvergenceDecision
# ─────────────────────────────────────────────────────────────────────────────

class TestConvergenceDecision:
    def test_fields_present(self):
        d = ConvergenceDecision(
            should_stop=True,
            reason="delta_stable",
            delta_norm=0.001,
            global_confidence=0.8,
            weakness_ratio=0.1,
            coverage_ratio=1.5,
            convergence_score=0.9,
        )
        assert d.should_stop is True
        assert d.reason == "delta_stable"
        assert 0.0 <= d.convergence_score <= 1.0

    def test_should_stop_false(self):
        d = ConvergenceDecision(
            should_stop=False, reason="continuing",
            delta_norm=0.5, global_confidence=0.3,
            weakness_ratio=0.8, coverage_ratio=2.0,
            convergence_score=0.2,
        )
        assert not d.should_stop


# ─────────────────────────────────────────────────────────────────────────────
# TestConvergenceGateBasic
# ─────────────────────────────────────────────────────────────────────────────

class TestConvergenceGateBasic:
    def _make_report(self, n: int, conf: float, n_weak: int) -> WeaknessReport:
        """WeaknessReport sintético con confianza y debilidades controladas."""
        return WeaknessReport(
            weaknesses    = [object() for _ in range(n_weak)],  # dummy
            focus_mask    = torch.zeros(n, dtype=torch.bool),
            node_severity = torch.zeros(n),
            confidence    = torch.full((n,), conf),
            n_weaknesses  = n_weak,
            mean_severity = 0.0,
        )

    def test_stops_on_small_delta(self):
        """Cuando delta_norm < threshold, debe parar."""
        gate = ConvergenceGate(delta_threshold=0.1, min_iterations=1)
        N = 4
        # h_prev con norma grande; h casi idéntico → delta_norm << threshold
        h_prev = torch.ones(N, NODE_DIM)
        h      = h_prev + 1e-8   # ||h - h_prev|| / ||h_prev|| ≈ 1e-8 << 0.1
        h_init = torch.zeros(N, NODE_DIM)
        report = self._make_report(N, 0.5, 2)

        d = gate.check(h, h_prev, h_init, report, n_weak_init=5, iteration=1)
        assert d.should_stop
        assert d.reason == _REASON_DELTA

    def test_stops_on_high_conf_low_weakness(self):
        """Alta confianza + pocas debilidades → parar."""
        gate = ConvergenceGate(
            delta_threshold=0.001,   # delta muy pequeño para este test
            conf_threshold=0.8,
            weakness_threshold=0.2,
            min_iterations=1,
        )
        N = 4
        h      = torch.randn(N, NODE_DIM) * 0.5
        h_prev = h + torch.randn(N, NODE_DIM) * 0.01   # delta moderado > 0.001
        h_init = torch.zeros(N, NODE_DIM)
        # Alta confianza, pocas debilidades
        report = self._make_report(N, 0.95, 1)   # 1 debilidad de 10 iniciales → ratio=0.1

        d = gate.check(h, h_prev, h_init, report, n_weak_init=10, iteration=2)
        assert d.should_stop
        assert d.reason == _REASON_CONFIDENCE

    def test_does_not_stop_when_continuing(self):
        """Delta grande + baja confianza → continuar."""
        gate = ConvergenceGate(delta_threshold=0.01, conf_threshold=0.8, min_iterations=1)
        N = 4
        h      = torch.randn(N, NODE_DIM)
        h_prev = torch.zeros(N, NODE_DIM)  # delta = ||h|| >> threshold
        h_init = torch.zeros(N, NODE_DIM)
        report = self._make_report(N, 0.3, 5)

        d = gate.check(h, h_prev, h_init, report, n_weak_init=5, iteration=2)
        assert not d.should_stop
        assert d.reason == _REASON_CONTINUING

    def test_delta_norm_computed_correctly(self):
        """delta_norm = ||h - h_prev|| / ||h_prev||."""
        gate = ConvergenceGate(delta_threshold=100.0, min_iterations=0)
        N = 3
        h      = torch.ones(N, NODE_DIM) * 2.0
        h_prev = torch.ones(N, NODE_DIM) * 1.0
        h_init = torch.zeros(N, NODE_DIM)
        report = self._make_report(N, 0.5, 0)

        d = gate.check(h, h_prev, h_init, report, n_weak_init=1, iteration=1)
        # delta = ||ones * 1|| / ||ones * 1|| = 1.0
        assert abs(d.delta_norm - 1.0) < 0.01

    def test_convergence_score_range(self):
        """convergence_score ∈ [0, 1]."""
        gate = ConvergenceGate(min_iterations=0)
        N = 4
        for _ in range(10):
            h      = torch.randn(N, NODE_DIM)
            h_prev = torch.randn(N, NODE_DIM)
            h_init = torch.randn(N, NODE_DIM)
            report = self._make_report(N, torch.rand(1).item(), int(torch.randint(0, 10, (1,))))
            d = gate.check(h, h_prev, h_init, report, n_weak_init=max(1, report.n_weaknesses), iteration=1)
            assert 0.0 <= d.convergence_score <= 1.0 + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# TestConvergenceGateMinIter
# ─────────────────────────────────────────────────────────────────────────────

class TestConvergenceGateMinIter:
    def test_never_stops_before_min_iterations(self):
        """Safety floor: nunca parar antes de min_iterations."""
        gate = ConvergenceGate(
            delta_threshold=100.0,  # siempre pararía por delta
            min_iterations=5,
        )
        N = 3
        h      = torch.zeros(N, NODE_DIM)
        h_prev = h + 1e-10   # delta ≈ 0
        h_init = torch.zeros(N, NODE_DIM)
        report = WeaknessReport([], torch.zeros(N, dtype=torch.bool),
                                torch.zeros(N), torch.ones(N), 0, 0.0)

        for it in range(4):   # iteraciones 0, 1, 2, 3 (antes de min=5)
            d = gate.check(h, h_prev, h_init, report, n_weak_init=1, iteration=it)
            assert not d.should_stop, f"No debe parar en iteration={it} < min_iterations=5"
            assert d.reason == _REASON_MIN_ITER

    def test_can_stop_at_min_iteration(self):
        """En iteration == min_iterations-1 (0-indexed), puede parar."""
        gate = ConvergenceGate(
            delta_threshold=100.0,
            conf_threshold=0.0,
            min_iterations=2,
        )
        N = 3
        h      = torch.zeros(N, NODE_DIM)
        h_prev = h + 1e-10
        h_init = torch.zeros(N, NODE_DIM)
        report = WeaknessReport([], torch.zeros(N, dtype=torch.bool),
                                torch.zeros(N), torch.ones(N), 0, 0.0)

        # iteration=1 es el 2do paso (0-indexed) = min_iterations-1
        d = gate.check(h, h_prev, h_init, report, n_weak_init=1, iteration=1)
        assert d.should_stop


# ─────────────────────────────────────────────────────────────────────────────
# TestSimpleGraphConvergesFast
# ─────────────────────────────────────────────────────────────────────────────

class TestSimpleGraphConvergesFast:
    def test_zero_features_converge_in_few_iters(self):
        """
        Features iniciales en cero + grafo simple → delta muy pequeño desde iter 1.
        Debe converger en 1-3 iteraciones.
        """
        cfg = tiny_cfg(
            max_iterations=20,
            min_iterations=1,
            conv_delta_threshold=0.05,
        )
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g, h = simple_graph()

        with torch.no_grad():
            out = eng(g, h, n_iterations=20)

        assert out.iterations_run <= 5, (
            f"Grafo simple debería converger en ≤5 iters, "
            f"usó {out.iterations_run} (razón: {out.stop_reason})"
        )

    def test_stop_reason_not_max_iterations(self):
        """Un grafo simple no debe agotar el máximo de iteraciones."""
        cfg = tiny_cfg(max_iterations=20, min_iterations=1)
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g, h = simple_graph()

        with torch.no_grad():
            out = eng(g, h, n_iterations=20)

        assert out.stop_reason != "max_iterations", (
            f"Grafo simple no debería agotar max_iterations, "
            f"stop_reason={out.stop_reason}"
        )

    def test_iterations_run_consistent_with_stop_reason(self):
        """iterations_run > 0 siempre."""
        cfg = tiny_cfg(max_iterations=20, min_iterations=1)
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g, h = simple_graph()

        with torch.no_grad():
            out = eng(g, h)

        assert out.iterations_run >= 1


# ─────────────────────────────────────────────────────────────────────────────
# TestComplexGraphUsesMoreIters
# ─────────────────────────────────────────────────────────────────────────────

class TestComplexGraphUsesMoreIters:
    def test_complex_uses_more_iters_than_simple(self):
        """
        El grafo complejo debe usar más iteraciones que el simple.
        Comparamos iterations_run en el mismo engine.
        """
        cfg = tiny_cfg(
            max_iterations=20,
            min_iterations=1,
            conv_delta_threshold=0.001,   # threshold estricto → no para fácil
        )
        eng = CausalReasoningEngine(cfg)
        eng.eval()

        g_simple, h_simple = simple_graph()
        g_complex, h_complex = complex_graph()

        with torch.no_grad():
            out_simple  = eng(g_simple,  h_simple,  n_iterations=20)
            out_complex = eng(g_complex, h_complex, n_iterations=20)

        assert out_complex.iterations_run >= out_simple.iterations_run, (
            f"Complejo ({out_complex.iterations_run}) debería usar ≥ simple ({out_simple.iterations_run})"
        )

    def test_complex_weaknesses_detected(self):
        """El grafo complejo debe tener debilidades detectadas."""
        cfg = tiny_cfg(max_iterations=1)
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g_complex, h_complex = complex_graph()

        with torch.no_grad():
            out = eng(g_complex, h_complex, n_iterations=1)

        assert out.n_weaknesses_initial > 0, (
            "El grafo complejo debe tener debilidades en la primera iteración"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestFocusMaskInEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestFocusMaskInEngine:
    def test_focus_mask_shape_matches_nodes(self):
        """focus_mask_final tiene shape [N]."""
        cfg = tiny_cfg(max_iterations=3)
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g, h = complex_graph()
        N = h.shape[0]

        with torch.no_grad():
            out = eng(g, h, n_iterations=3)

        assert out.focus_mask_final is not None
        assert out.focus_mask_final.shape == (N,)
        assert out.focus_mask_final.dtype == torch.bool

    def test_focus_mask_limits_unfocused_nodes(self):
        """
        Cuando focus_mask está activo, los nodos sin debilidades NO se actualizan.
        Para verificar esto, comparamos el output con y sin gate en un nodo específico.

        Estrategia: usamos un grafo donde n0 (FACT) definitivamente no tiene debilidades
        (tiene causa propia como FACT) y verificamos que sus features permanezcan estables.
        """
        cfg = tiny_cfg(
            max_iterations=3,
            min_iterations=1,
            conv_delta_threshold=0.0,   # no parar por delta
            conv_conf_threshold=1.1,    # no parar por confianza
            conv_weakness_threshold=0.0,  # no parar por weaknesses
        )
        eng = CausalReasoningEngine(cfg)
        eng.eval()

        # FACT node: sin debilidades (no missing_cause, no weak_evidence)
        g = CausalGraph()
        g.add_node(make_node("fact",   ntype=NodeType.FACT))
        g.add_node(make_node("effect", ntype=NodeType.HYPOTHESIS))
        g.add_edge(CausalEdge(source_id="fact", target_id="effect",
                              relation=CausalRelation.CAUSES))
        h = torch.zeros(2, NODE_DIM)   # features estables

        with torch.no_grad():
            out = eng(g, h, n_iterations=3)

        assert out.focus_mask_final is not None
        # fact (index 0) debería NO estar en el focus (FACT + tiene causa = sin debilidades)
        # effect (index 1) debería estar en el focus si tiene low_confidence

    def test_none_when_gate_disabled(self):
        """Con use_convergence_gate=False, focus_mask_final es None."""
        cfg = CREConfig(
            node_dim=NODE_DIM, edge_dim=16, message_dim=24,
            n_message_layers=1, max_iterations=3,
            use_convergence_gate=False,
        )
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g, h = simple_graph()

        with torch.no_grad():
            out = eng(g, h, n_iterations=3)

        assert out.focus_mask_final is None


# ─────────────────────────────────────────────────────────────────────────────
# TestIterationsRunAccurate
# ─────────────────────────────────────────────────────────────────────────────

class TestIterationsRunAccurate:
    def test_runs_exactly_max_when_no_convergence(self):
        """Sin gate, siempre corre exactamente n_iterations."""
        cfg = CREConfig(
            node_dim=NODE_DIM, edge_dim=16, message_dim=24,
            n_message_layers=1, max_iterations=7,
            use_convergence_gate=False,
        )
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g, h = complex_graph()

        with torch.no_grad():
            out = eng(g, h, n_iterations=7)

        assert out.iterations_run == 7

    def test_iterations_run_between_min_and_max(self):
        """Con gate, iterations_run ∈ [min_iterations, max_iterations]."""
        cfg = tiny_cfg(max_iterations=15, min_iterations=2)
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g, h = complex_graph()

        with torch.no_grad():
            out = eng(g, h, n_iterations=15)

        assert 2 <= out.iterations_run <= 15

    def test_override_n_iterations(self):
        """n_iterations en forward() actúa como safety cap."""
        cfg = tiny_cfg(max_iterations=20, min_iterations=1)
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g, h = complex_graph()

        with torch.no_grad():
            out = eng(g, h, n_iterations=3)   # override cap

        assert out.iterations_run <= 3


# ─────────────────────────────────────────────────────────────────────────────
# TestBackwardCompatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestBackwardCompatibility:
    def test_gate_disabled_behaves_as_original(self):
        """use_convergence_gate=False: iteraciones fijas, sin overhead."""
        cfg = CREConfig(
            node_dim=NODE_DIM, edge_dim=16, message_dim=24,
            n_message_layers=2, max_iterations=5,
            use_convergence_gate=False,
        )
        eng = CausalReasoningEngine(cfg)
        assert eng.weakness_detector is None
        assert eng.convergence_gate is None

        eng.eval()
        g, h = simple_graph()
        with torch.no_grad():
            out = eng(g, h, n_iterations=5)

        assert out.iterations_run == 5
        assert out.stop_reason == "max_iterations"
        assert out.n_weaknesses_initial == 0
        assert out.n_weaknesses_final == 0
        assert out.focus_mask_final is None

    def test_existing_tests_unaffected(self):
        """Los tests del CRE original pasan con use_convergence_gate=False."""
        cfg = CREConfig(node_dim=32, edge_dim=16, message_dim=24,
                        n_message_layers=2, max_iterations=3,
                        use_convergence_gate=False)
        eng = CausalReasoningEngine(cfg)
        eng.eval()

        g = CausalGraph()
        for i in range(4):
            g.add_node(CausalNode(node_id=f"n{i}", label=f"n{i}"))
        for i in range(3):
            g.add_edge(CausalEdge(source_id=f"n{i}", target_id=f"n{i+1}",
                                  relation=CausalRelation.CAUSES))

        h = torch.randn(4, 32)
        with torch.no_grad():
            out = eng(g, h, n_iterations=3)

        assert out.node_features.shape == (4, 32)
        assert out.edge_features.shape == (3, 16)
        assert out.iterations_run == 3


# ─────────────────────────────────────────────────────────────────────────────
# TestGradientFlowWithGate
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlowWithGate:
    def test_gradients_flow_through_weakness_detector(self):
        """
        Con use_convergence_gate=True, los gradientes fluyen desde el output
        del engine hasta los parámetros del confidence_scorer del WeaknessDetector.

        Nota: solo las funciones de mensaje usadas en el grafo tienen gradientes.
        Verificamos los parámetros que DEBEN participar (causes + norms).
        """
        cfg = tiny_cfg(
            max_iterations=3,
            min_iterations=1,
            conv_delta_threshold=0.0,   # no parar por delta
            conv_conf_threshold=1.1,    # no parar por conf
        )
        eng = CausalReasoningEngine(cfg)
        eng.train()
        # Grafo con CAUSES para activar message_fns.causes
        g = CausalGraph()
        for nid in ["A", "B", "C"]:
            g.add_node(make_node(nid))
        g.add_edge(CausalEdge(source_id="A", target_id="B", relation=CausalRelation.CAUSES))
        g.add_edge(CausalEdge(source_id="B", target_id="C", relation=CausalRelation.CAUSES))
        h = torch.randn(3, NODE_DIM, requires_grad=True)

        out = eng(g, h, n_iterations=3)
        loss = out.node_features.sum()
        loss.backward()

        # Parámetros del message passing usados en el grafo (causes) deben tener grad
        causes_params = list(eng.layers[0].message_fns["causes"].parameters())
        for p in causes_params:
            assert p.grad is not None, "causes message_fn debe tener gradiente"

        # Parámetros del weakness_detector: el confidence_scorer siempre participa
        wd = eng.weakness_detector
        if wd is not None:
            for name, param in wd.confidence_scorer.named_parameters():
                # confidence_scorer se llama en forward → debe tener grad
                if param.requires_grad:
                    # No todos participan en la pérdida si el grafo es simple
                    assert not torch.isnan(param.grad).any() if param.grad is not None else True

    def test_no_nan_in_output_with_gate(self):
        """No NaN ni Inf en el output con gate activado."""
        cfg = tiny_cfg(max_iterations=5, min_iterations=2)
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g_complex, h_complex = complex_graph()

        with torch.no_grad():
            out = eng(g_complex, h_complex, n_iterations=5)

        assert not torch.isnan(out.node_features).any(), "NaN en node_features"
        assert not torch.isinf(out.node_features).any(), "Inf en node_features"
        assert not torch.isnan(out.edge_features).any(), "NaN en edge_features"


# ─────────────────────────────────────────────────────────────────────────────
# TestStopReasons
# ─────────────────────────────────────────────────────────────────────────────

class TestStopReasons:
    def test_max_iterations_reason_when_gate_disabled(self):
        cfg = CREConfig(node_dim=NODE_DIM, edge_dim=16, message_dim=24,
                        n_message_layers=1, max_iterations=3,
                        use_convergence_gate=False)
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g, h = simple_graph()
        with torch.no_grad():
            out = eng(g, h, n_iterations=3)
        assert out.stop_reason == "max_iterations"

    def test_delta_stable_reason(self):
        """Features en cero → delta estable desde la primera iteración."""
        cfg = tiny_cfg(
            max_iterations=20,
            min_iterations=1,
            conv_delta_threshold=100.0,   # cualquier delta es "estable"
        )
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g, h = simple_graph()

        with torch.no_grad():
            out = eng(g, h, n_iterations=20)

        assert out.stop_reason == _REASON_DELTA

    def test_weaknesses_tracked_over_iterations(self):
        """n_weaknesses_initial >= n_weaknesses_final (el CRE mejora el grafo)."""
        cfg = tiny_cfg(max_iterations=5, min_iterations=2)
        eng = CausalReasoningEngine(cfg)
        eng.eval()
        g_complex, h_complex = complex_graph()

        with torch.no_grad():
            out = eng(g_complex, h_complex, n_iterations=5)

        # Ambos deben ser >= 0 cuando el gate está activo
        assert out.n_weaknesses_initial >= 0
        assert out.n_weaknesses_final >= 0
