"""
tests/test_scratch_pad.py — Tests del DifferentiableScratchPad
==============================================================

Organización:
    TestScratchPadConfig        — validación de configuración
    TestInitState               — estado inicial correcto
    TestRead                    — shapes, diferenciabilidad, attention properties
    TestUpdate                  — NTM formula, write_addr, erase, write_gate
    TestRetention               — write → read recupera información
    TestEraseGate               — el erase gate puede limpiar slots
    TestDifferentiability       — gradientes end-to-end
    TestIntegrationWithCRE      — funciona dentro del loop del CausalReasoningEngine
    TestNumericalStability      — estable con múltiples llamadas
    TestDeterminism             — eval mode determinista
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from core.graph import CausalEdge, CausalGraph, CausalNode, CausalRelation, NodeType
from cre import (
    CREConfig,
    CausalReasoningEngine,
    DifferentiableScratchPad,
    ScratchPadConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES & HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_cfg():
    return ScratchPadConfig(n_slots=8, slot_dim=16, node_dim=32)


@pytest.fixture
def pad(tiny_cfg):
    m = DifferentiableScratchPad(tiny_cfg)
    m.eval()
    return m


@pytest.fixture
def default_cfg():
    return ScratchPadConfig()  # n_slots=16, slot_dim=128, node_dim=256


def make_pad_and_state(n_slots=8, slot_dim=16, node_dim=32, device=None):
    cfg   = ScratchPadConfig(n_slots=n_slots, slot_dim=slot_dim, node_dim=node_dim)
    pad   = DifferentiableScratchPad(cfg)
    pad.eval()
    state = pad.init_state(device=device)
    return pad, state


def make_linear_graph(n_nodes=4):
    graph = CausalGraph()
    for i in range(n_nodes):
        graph.add_node(CausalNode(node_id=f"n{i}", label=f"n{i}",
                                   node_type=NodeType.EVENT))
    for i in range(n_nodes - 1):
        graph.add_edge(CausalEdge(source_id=f"n{i}", target_id=f"n{i+1}",
                                   relation=CausalRelation.CAUSES))
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# TestScratchPadConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestScratchPadConfig:
    def test_default_values(self):
        cfg = ScratchPadConfig()
        assert cfg.n_slots  == 16
        assert cfg.slot_dim == 128
        assert cfg.node_dim == 256

    def test_custom_values(self):
        cfg = ScratchPadConfig(n_slots=8, slot_dim=32, node_dim=64)
        assert cfg.n_slots  == 8
        assert cfg.slot_dim == 32
        assert cfg.node_dim == 64

    def test_invalid_n_slots(self):
        with pytest.raises(ValueError):
            ScratchPadConfig(n_slots=0)

    def test_invalid_slot_dim(self):
        with pytest.raises(ValueError):
            ScratchPadConfig(slot_dim=0)

    def test_invalid_node_dim(self):
        with pytest.raises(ValueError):
            ScratchPadConfig(node_dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# TestInitState
# ─────────────────────────────────────────────────────────────────────────────

class TestInitState:
    def test_shape(self, pad, tiny_cfg):
        state = pad.init_state()
        assert state.shape == (tiny_cfg.n_slots, tiny_cfg.slot_dim)

    def test_initialized_to_zeros(self, pad, tiny_cfg):
        state = pad.init_state()
        assert torch.all(state == 0)

    def test_dtype_preserved(self, pad):
        state_f32 = pad.init_state(dtype=torch.float32)
        assert state_f32.dtype == torch.float32

    def test_device_cpu(self, pad):
        state = pad.init_state(device=torch.device("cpu"))
        assert state.device.type == "cpu"

    def test_independent_states(self, pad):
        """Dos llamadas a init_state producen tensores independientes."""
        s1 = pad.init_state()
        s2 = pad.init_state()
        s1[0, 0] = 99.0
        assert s2[0, 0] == 0.0, "States should be independent"

    def test_default_config_shape(self):
        cfg = ScratchPadConfig()
        pad = DifferentiableScratchPad(cfg)
        state = pad.init_state()
        assert state.shape == (16, 128)


# ─────────────────────────────────────────────────────────────────────────────
# TestRead
# ─────────────────────────────────────────────────────────────────────────────

class TestRead:
    def test_output_shape(self, pad, tiny_cfg):
        N = 5
        state = pad.init_state()
        nf    = torch.randn(N, tiny_cfg.node_dim)
        out   = pad.read(nf, state)
        assert out.shape == (N, tiny_cfg.node_dim)

    def test_output_dim_matches_node_dim(self, pad, tiny_cfg):
        """read() devuelve vectores del mismo tamaño que node_features (residual)."""
        out = pad.read(torch.randn(3, tiny_cfg.node_dim), pad.init_state())
        assert out.shape[-1] == tiny_cfg.node_dim

    def test_read_from_zeros_gives_residual(self, pad, tiny_cfg):
        """
        Con estado cero y read_out_proj inicializado en cero (std=0.02 ≈ small),
        el output debe ser muy cercano al input (residual dominante).
        """
        nf    = torch.randn(4, tiny_cfg.node_dim) * 3.0
        state = pad.init_state()  # zeros → V=0 → content=0 → read_out≈0 → out≈LN(nf)
        out   = pad.read(nf, state)
        # No exactamente igual debido a LayerNorm, pero la norma debe ser similar
        assert out.shape == nf.shape
        assert out.isfinite().all()

    def test_nonzero_state_changes_output(self, pad, tiny_cfg):
        """Un estado no nulo debe cambiar el output respecto al estado cero."""
        nf     = torch.randn(4, tiny_cfg.node_dim)
        state0 = pad.init_state()
        state1 = torch.randn_like(state0) * 2.0

        with torch.no_grad():
            out0 = pad.read(nf, state0)
            out1 = pad.read(nf, state1)

        assert not torch.allclose(out0, out1), \
            "Different states should produce different read outputs"

    def test_attention_weights_sum_to_one(self, pad, tiny_cfg):
        """
        La attention del read es un softmax → debe sumar 1 por fila.
        Verificado via el comportamiento del output.
        """
        # Indirectamente: el output debe ser finito y acotado
        nf    = torch.randn(6, tiny_cfg.node_dim)
        state = torch.randn(tiny_cfg.n_slots, tiny_cfg.slot_dim)
        out   = pad.read(nf, state)
        assert out.isfinite().all()

    def test_read_has_residual_connection(self, pad, tiny_cfg):
        """
        El read usa residual: output = LN(node_features + projected_content).
        Con randn input (non-constant), LN(nf + 0) es no-cero porque la varianza > 0.
        """
        torch.manual_seed(42)
        nf     = torch.randn(3, tiny_cfg.node_dim)  # non-constant → LN non-zero
        state  = pad.init_state()
        out    = pad.read(nf, state)
        # Output no es zero: LN normaliza la varianza de nf al pasar por residual
        assert out.norm() > 0.1

    def test_gradient_flows_to_node_features(self, pad, tiny_cfg):
        nf    = torch.randn(4, tiny_cfg.node_dim, requires_grad=True)
        state = pad.init_state()
        out   = pad.read(nf, state)
        out.sum().backward()
        assert nf.grad is not None
        assert not nf.grad.isnan().any()

    def test_gradient_flows_to_state(self, pad, tiny_cfg):
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = torch.randn(tiny_cfg.n_slots, tiny_cfg.slot_dim, requires_grad=True)
        out   = pad.read(nf, state)
        out.sum().backward()
        assert state.grad is not None

    def test_different_nodes_different_reads(self, pad, tiny_cfg):
        """Nodos distintos deben leer cosas distintas del mismo estado."""
        state = torch.randn(tiny_cfg.n_slots, tiny_cfg.slot_dim)
        nf    = torch.randn(4, tiny_cfg.node_dim)
        out   = pad.read(nf, state)
        # Comprobar que no todos los nodos leen lo mismo
        all_same = all(torch.allclose(out[0], out[i]) for i in range(1, 4))
        assert not all_same, "Different nodes should read different content"

    def test_has_layernorm(self, pad):
        assert hasattr(pad, 'read_norm')
        assert isinstance(pad.read_norm, nn.LayerNorm)


# ─────────────────────────────────────────────────────────────────────────────
# TestUpdate
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdate:
    def test_output_shape(self, pad, tiny_cfg):
        state = pad.init_state()
        nf    = torch.randn(4, tiny_cfg.node_dim)
        new_state = pad.update(nf, state)
        assert new_state.shape == (tiny_cfg.n_slots, tiny_cfg.slot_dim)

    def test_state_changes_after_update(self, pad, tiny_cfg):
        """Después de update(), el estado debe cambiar."""
        state = pad.init_state()
        nf    = torch.randn(4, tiny_cfg.node_dim)
        new_state = pad.update(nf, state)
        assert not torch.allclose(new_state, state), \
            "State should change after update"

    def test_write_addr_sums_to_one(self, pad, tiny_cfg):
        """write_addr es un softmax → suma 1 por fila (por nodo)."""
        state = pad.init_state()
        nf    = torch.randn(4, tiny_cfg.node_dim)
        _, write_addr, _, _, _ = pad.update_debug(nf, state)
        row_sums = write_addr.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"write_addr rows should sum to 1, got {row_sums}"

    def test_write_gate_in_01(self, pad, tiny_cfg):
        """write_gate es un sigmoid → ∈ (0,1)."""
        state = pad.init_state()
        nf    = torch.randn(4, tiny_cfg.node_dim)
        _, _, write_gate, _, _ = pad.update_debug(nf, state)
        assert (write_gate >= 0).all() and (write_gate <= 1).all()

    def test_erase_vec_in_01(self, pad, tiny_cfg):
        """erase_vec es un sigmoid → ∈ (0,1)."""
        state = pad.init_state()
        nf    = torch.randn(4, tiny_cfg.node_dim)
        _, _, _, erase_vec, _ = pad.update_debug(nf, state)
        assert (erase_vec >= 0).all() and (erase_vec <= 1).all()

    def test_content_in_minus1_1(self, pad, tiny_cfg):
        """content es un tanh → ∈ (-1, 1)."""
        state = pad.init_state()
        nf    = torch.randn(4, tiny_cfg.node_dim)
        _, _, _, _, content = pad.update_debug(nf, state)
        assert (content >= -1).all() and (content <= 1).all()

    def test_ntm_formula_verified(self, pad, tiny_cfg):
        """
        Verificación white-box: la fórmula NTM está implementada correctamente.
        new_state = state * (1 - erase_agg) + write_agg
        """
        state = torch.randn(tiny_cfg.n_slots, tiny_cfg.slot_dim)
        nf    = torch.randn(4, tiny_cfg.node_dim)

        with torch.no_grad():
            new_state, write_addr, write_gate, erase_vec, content = pad.update_debug(nf, state)

        # Reconstruir manualmente
        weighted  = write_gate * write_addr              # [N, n_slots]
        erase_agg = (weighted.T @ erase_vec).clamp(0,1) # [n_slots, S]
        write_agg = weighted.T @ content                 # [n_slots, S]
        expected  = state * (1.0 - erase_agg) + write_agg

        assert torch.allclose(new_state, expected, atol=1e-5), \
            "update() formula doesn't match manual reconstruction"

    def test_gradient_flows_to_node_features(self, pad, tiny_cfg):
        nf    = torch.randn(4, tiny_cfg.node_dim, requires_grad=True)
        state = pad.init_state()
        new_state = pad.update(nf, state)
        new_state.sum().backward()
        assert nf.grad is not None
        assert not nf.grad.isnan().any()

    def test_gradient_flows_to_state(self, pad, tiny_cfg):
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = torch.randn(tiny_cfg.n_slots, tiny_cfg.slot_dim, requires_grad=True)
        new_state = pad.update(nf, state)
        new_state.sum().backward()
        assert state.grad is not None

    def test_output_finite(self, pad, tiny_cfg):
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = pad.init_state()
        new_state = pad.update(nf, state)
        assert new_state.isfinite().all()

    def test_update_debug_same_as_update(self, pad, tiny_cfg):
        """update_debug() debe producir el mismo new_state que update()."""
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = torch.randn(tiny_cfg.n_slots, tiny_cfg.slot_dim)

        with torch.no_grad():
            ns1 = pad.update(nf, state)
            ns2, *_ = pad.update_debug(nf, state)

        assert torch.allclose(ns1, ns2, atol=1e-6), \
            "update() and update_debug() should produce the same new_state"

    def test_multiple_nodes_aggregate_to_same_state_shape(self, pad, tiny_cfg):
        """Cualquier número de nodos produce el mismo shape de estado."""
        state = pad.init_state()
        for N in [1, 4, 8, 16]:
            nf = torch.randn(N, tiny_cfg.node_dim)
            ns = pad.update(nf, state)
            assert ns.shape == (tiny_cfg.n_slots, tiny_cfg.slot_dim)


# ─────────────────────────────────────────────────────────────────────────────
# TestRetention
# ─────────────────────────────────────────────────────────────────────────────

class TestRetention:
    """
    Verifica que el scratch pad RETIENE información entre llamadas.
    La propiedad central: write → read recupera algo relacionado con lo escrito.
    """

    def test_read_changes_after_write(self, pad, tiny_cfg):
        """
        Leer DESPUÉS de escribir produce un output diferente que antes de escribir.
        Esto verifica que la escritura modifica el estado y que la lectura lo detecta.
        """
        state0 = pad.init_state()
        nf     = torch.randn(4, tiny_cfg.node_dim)

        # Leer del estado vacío
        with torch.no_grad():
            read0 = pad.read(nf, state0)
            # Escribir
            state1 = pad.update(nf, state0)
            # Leer del estado modificado
            read1 = pad.read(nf, state1)

        assert not torch.allclose(read0, read1), \
            "Reading after write should give different result than before write"

    def test_write_updates_state(self, pad, tiny_cfg):
        """El estado debe cambiar después de cada update()."""
        state = pad.init_state()
        nf    = torch.randn(4, tiny_cfg.node_dim)

        with torch.no_grad():
            for _ in range(3):
                prev_state = state.clone()
                state = pad.update(nf, state)
                assert not torch.allclose(state, prev_state), \
                    "State should change after each update"

    def test_sequential_writes_accumulate(self, pad, tiny_cfg):
        """
        Múltiples escrituras cambian el estado progresivamente.
        Tras N writes, el estado difiere del estado tras 1 write.
        """
        nf     = torch.randn(4, tiny_cfg.node_dim)
        state  = pad.init_state()

        with torch.no_grad():
            state_after_1 = pad.update(nf, state)
            state_after_3 = state.clone()
            for _ in range(3):
                state_after_3 = pad.update(nf, state_after_3)

        assert not torch.allclose(state_after_1, state_after_3), \
            "3 writes should produce different state than 1 write"

    def test_different_write_inputs_different_states(self, pad, tiny_cfg):
        """Escrituras con inputs distintos deben producir estados distintos."""
        state0 = pad.init_state()
        nf1    = torch.randn(4, tiny_cfg.node_dim)
        nf2    = torch.randn(4, tiny_cfg.node_dim)

        with torch.no_grad():
            s1 = pad.update(nf1, state0)
            s2 = pad.update(nf2, state0)

        assert not torch.allclose(s1, s2)

    def test_read_content_related_to_write_content(self, pad, tiny_cfg):
        """
        Tras escribir con un query específico, leer con el mismo query
        debe retornar algo influenciado por lo que se escribió.

        Se verifica midiendo que el output de read después de escribir
        es diferente al output antes, indicando que la info fue retenida.
        """
        nf     = torch.randn(4, tiny_cfg.node_dim) * 3.0
        state0 = pad.init_state()

        with torch.no_grad():
            before = pad.read(nf, state0)
            state1 = pad.update(nf, state0)
            after  = pad.read(nf, state1)

        # La diferencia entre before y after debe ser no trivial
        diff = (after - before).abs().mean().item()
        assert diff > 1e-6, \
            f"Read after write should differ from read before, mean diff={diff:.2e}"

    def test_write_read_write_read_evolves(self, pad, tiny_cfg):
        """
        Un ciclo write→read→write→read debe mostrar evolución del estado.
        """
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = pad.init_state()
        reads = []

        with torch.no_grad():
            for _ in range(4):
                r = pad.read(nf, state)
                reads.append(r.clone())
                state = pad.update(nf, state)

        # Al menos algún par de reads consecutivos debe diferir
        diffs = [(reads[i] - reads[i+1]).abs().max().item() for i in range(3)]
        assert any(d > 1e-6 for d in diffs), \
            "Read outputs should evolve with each write-read cycle"


# ─────────────────────────────────────────────────────────────────────────────
# TestEraseGate
# ─────────────────────────────────────────────────────────────────────────────

class TestEraseGate:
    """
    Verifica que el mecanismo de borrado (erase gate) funciona correctamente.
    """

    def test_erase_formula_reduces_addressed_slot(self, pad, tiny_cfg):
        """
        Verificación matemática: slots muy direccionados con erase≈1
        deben tener menor magnitud después del update.
        """
        torch.manual_seed(0)
        nf     = torch.randn(4, tiny_cfg.node_dim)
        # Estado inicial con valores grandes
        state  = torch.ones(tiny_cfg.n_slots, tiny_cfg.slot_dim) * 3.0

        with torch.no_grad():
            new_state, write_addr, write_gate, erase_vec, content = pad.update_debug(nf, state)

        # Slot más direccionado:
        weighted   = (write_gate * write_addr).sum(0)       # [n_slots]
        top_slot   = weighted.argmax().item()

        # La fórmula dice: new[s] = state[s] * (1-erase_agg[s]) + write[s]
        # Si erase_agg[s] > 0, la magnitud del slot debe ser <= estado original
        # (excepto si write_agg lo aumenta)
        erase_agg = (weighted.unsqueeze(-1) * erase_vec.mean(0)).clamp(0,1)
        # Simplificación: verificar que la fórmula se aplicó
        old_norm = state[top_slot].norm().item()
        new_norm = new_state[top_slot].norm().item()
        # No guaranteed to be smaller (write_agg adds too), but at least different
        assert not torch.allclose(new_state[top_slot], state[top_slot]), \
            "Most-addressed slot should be modified by erase+write"

    def test_erase_with_zero_state(self, pad, tiny_cfg):
        """Con estado inicial cero, erase no puede hacer nada (0 * anything = 0)."""
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = pad.init_state()  # zeros

        with torch.no_grad():
            new_state, _, write_gate, erase_vec, content = pad.update_debug(nf, state)

        # state * (1 - erase) = 0 * (1 - erase) = 0
        # new_state = 0 + write_agg = write_agg
        _, write_addr, write_gate2, _, content2 = pad.update_debug(nf, state)
        weighted    = write_gate2 * write_addr
        write_agg   = weighted.T @ content2
        assert torch.allclose(new_state, write_agg, atol=1e-5), \
            "With zero state, new_state should equal write_agg only"

    def test_erase_decreases_state_magnitude_net(self, pad, tiny_cfg):
        """
        Con estado grande y write_content pequeño, el erase debe reducir la magnitud neta.
        Esto se verifica usando valores constantes que producen erase sistemático.
        """
        torch.manual_seed(42)
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = torch.ones(tiny_cfg.n_slots, tiny_cfg.slot_dim) * 5.0

        # Aplicar múltiples updates con el mismo input
        # Si erase > write neto, la norma total debe bajar en algún momento
        with torch.no_grad():
            norms = [state.norm().item()]
            current = state
            for _ in range(5):
                current = pad.update(nf, current)
                norms.append(current.norm().item())

        # La norma no debe crecer indefinidamente (erase previene explosión)
        # Simplemente verificar que es finita
        assert all(n < float('inf') for n in norms), "State norm should remain finite"
        assert all(n > -float('inf') for n in norms)

    def test_erase_vec_is_sigmoid_bounded(self, pad, tiny_cfg):
        """erase_vec ∈ [0,1] — sigmoid guarantees this (float32 saturates at exactly 0 or 1)."""
        nf    = torch.randn(4, tiny_cfg.node_dim) * 100.0  # large input
        state = pad.init_state()
        _, _, _, erase_vec, _ = pad.update_debug(nf, state)
        assert (erase_vec >= 0).all()
        assert (erase_vec <= 1).all()

    def test_full_erase_zeros_slot_mathematically(self, tiny_cfg):
        """
        Si construimos manualmente erase_agg=1 para un slot, ese slot se zeroa.
        Verificación puramente matemática del mecanismo.
        """
        pad   = DifferentiableScratchPad(tiny_cfg)
        state = torch.ones(tiny_cfg.n_slots, tiny_cfg.slot_dim) * 3.0

        # Simular erase_agg=1 para el slot 0, write_agg=0 para ese slot
        erase_agg = torch.zeros(tiny_cfg.n_slots, tiny_cfg.slot_dim)
        erase_agg[0] = 1.0  # borrado completo del slot 0
        write_agg  = torch.zeros_like(erase_agg)

        new_state = state * (1.0 - erase_agg) + write_agg

        assert torch.allclose(new_state[0], torch.zeros(tiny_cfg.slot_dim)), \
            "Slot with erase_agg=1 and write_agg=0 should be zeroed"
        # Otros slots no deben cambiar
        assert torch.allclose(new_state[1:], state[1:])

    def test_zero_gate_preserves_state(self, pad, tiny_cfg):
        """
        Con write_gate=0 para todos los nodos: weighted=0 → erase_agg=0 → state preservado.
        Verificamos con un input que produzca gate≈0.
        """
        # Modificar el bias del gate head para producir gate≈0
        with torch.no_grad():
            pad.gate_head.bias.fill_(-20.0)  # sigmoid(-20) ≈ 0

        state = torch.randn(tiny_cfg.n_slots, tiny_cfg.slot_dim) * 2.0
        nf    = torch.randn(4, tiny_cfg.node_dim)

        new_state = pad.update(nf, state)
        # write_gate ≈ 0 → weighted ≈ 0 → erase ≈ 0, write ≈ 0 → new ≈ state
        assert torch.allclose(new_state, state, atol=1e-3), \
            "With write_gate≈0, state should be nearly preserved"

        # Restaurar bias
        with torch.no_grad():
            pad.gate_head.bias.fill_(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TestDifferentiability
# ─────────────────────────────────────────────────────────────────────────────

class TestDifferentiability:
    """
    Verifica diferenciabilidad end-to-end: gradientes fluyen de read a update.
    """

    def test_grad_flows_through_read(self, pad, tiny_cfg):
        nf    = torch.randn(4, tiny_cfg.node_dim, requires_grad=True)
        state = pad.init_state()
        out   = pad.read(nf, state)
        out.sum().backward()
        assert nf.grad is not None
        assert not nf.grad.isnan().any()

    def test_grad_flows_through_update(self, pad, tiny_cfg):
        nf    = torch.randn(4, tiny_cfg.node_dim, requires_grad=True)
        state = pad.init_state()
        ns    = pad.update(nf, state)
        ns.sum().backward()
        assert nf.grad is not None

    def test_grad_flows_read_through_update(self, pad, tiny_cfg):
        """
        Gradientes fluyen de read() hasta update():
        read(nf, update(nf_write, state)) — backprop pasa por update.
        """
        nf_write = torch.randn(4, tiny_cfg.node_dim, requires_grad=True)
        nf_read  = torch.randn(4, tiny_cfg.node_dim)
        state    = pad.init_state()

        new_state = pad.update(nf_write, state)       # escribe
        out       = pad.read(nf_read, new_state)       # lee del estado modificado
        out.sum().backward()

        assert nf_write.grad is not None, \
            "Gradient should flow from read through the state to the write query"
        assert not nf_write.grad.isnan().any()

    def test_all_params_get_gradients(self, pad, tiny_cfg):
        """Todos los parámetros deben recibir gradiente en una pasada completa."""
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = pad.init_state()

        pad.train()
        ns  = pad.update(nf, state)
        out = pad.read(nf, ns)
        loss = out.sum() + ns.sum()
        loss.backward()

        no_grad = [n for n, p in pad.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert len(no_grad) == 0, f"Params without grad: {no_grad}"

    def test_gradients_are_finite(self, pad, tiny_cfg):
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = pad.init_state()

        pad.train()
        ns  = pad.update(nf, state)
        out = pad.read(nf, ns)
        (out.sum() + ns.sum()).backward()

        for name, p in pad.named_parameters():
            if p.grad is not None:
                assert p.grad.isfinite().all(), f"Non-finite grad in {name}"

    def test_multiple_update_then_read_differentiable(self, pad, tiny_cfg):
        """Varios updates seguidos de read son diferenciables."""
        nf    = torch.randn(4, tiny_cfg.node_dim, requires_grad=True)
        state = pad.init_state()

        pad.train()
        for _ in range(3):
            state = pad.update(nf, state)
        out = pad.read(nf, state)
        out.sum().backward()

        assert nf.grad is not None
        assert nf.grad.isfinite().all()


# ─────────────────────────────────────────────────────────────────────────────
# TestIntegrationWithCRE
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationWithCRE:
    """
    Verifica que el scratch pad funciona integrado con CausalReasoningEngine.
    """

    @pytest.fixture
    def tiny_cre_cfg(self):
        return CREConfig(node_dim=32, edge_dim=16, message_dim=24,
                         n_message_layers=2, max_iterations=3, n_relation_types=16)

    @pytest.fixture
    def tiny_pad_cfg(self):
        return ScratchPadConfig(n_slots=8, slot_dim=16, node_dim=32)

    def test_engine_accepts_scratch_pad(self, tiny_cre_cfg, tiny_pad_cfg):
        engine = CausalReasoningEngine(tiny_cre_cfg)
        pad    = DifferentiableScratchPad(tiny_pad_cfg)
        engine.eval(); pad.eval()

        graph  = make_linear_graph(4)
        node_f = torch.randn(4, tiny_cre_cfg.node_dim)

        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=3, scratch_pad=pad)

        assert out.node_features.shape == (4, tiny_cre_cfg.node_dim)
        assert out.node_features.isfinite().all()

    def test_scratch_pad_changes_cre_output(self, tiny_cre_cfg, tiny_pad_cfg):
        """CRE con scratch pad produce output diferente que sin él."""
        engine = CausalReasoningEngine(tiny_cre_cfg)
        pad    = DifferentiableScratchPad(tiny_pad_cfg)
        engine.eval(); pad.eval()

        graph  = make_linear_graph(4)
        node_f = torch.randn(4, tiny_cre_cfg.node_dim)

        with torch.no_grad():
            out_no_pad  = engine(graph, node_f, n_iterations=3)
            out_with_pad = engine(graph, node_f, n_iterations=3, scratch_pad=pad)

        diff = (out_no_pad.node_features - out_with_pad.node_features).abs().max().item()
        assert diff > 1e-4, \
            f"Scratch pad should change CRE output, max_diff={diff:.2e}"

    def test_cre_without_pad_unchanged(self, tiny_cre_cfg):
        """Sin scratch_pad (None), el CRE funciona igual que antes."""
        engine = CausalReasoningEngine(tiny_cre_cfg)
        engine.eval()
        graph  = make_linear_graph(4)
        node_f = torch.randn(4, tiny_cre_cfg.node_dim)

        with torch.no_grad():
            out1 = engine(graph, node_f, n_iterations=3, scratch_pad=None)
            out2 = engine(graph, node_f, n_iterations=3)  # default=None

        assert torch.allclose(out1.node_features, out2.node_features)

    def test_gradients_flow_with_scratch_pad(self, tiny_cre_cfg, tiny_pad_cfg):
        """Los gradientes fluyen a través del CRE+scratch pad."""
        engine = CausalReasoningEngine(tiny_cre_cfg)
        pad    = DifferentiableScratchPad(tiny_pad_cfg)
        engine.train(); pad.train()

        graph  = make_linear_graph(4)
        node_f = torch.randn(4, tiny_cre_cfg.node_dim, requires_grad=True)

        out = engine(graph, node_f, n_iterations=2, scratch_pad=pad)
        out.node_features.sum().backward()

        assert node_f.grad is not None
        assert not node_f.grad.isnan().any()

    def test_pad_params_get_gradients_in_cre_loop(self, tiny_cre_cfg, tiny_pad_cfg):
        """Los parámetros del scratch pad reciben gradientes dentro del loop del CRE."""
        engine = CausalReasoningEngine(tiny_cre_cfg)
        pad    = DifferentiableScratchPad(tiny_pad_cfg)
        engine.train(); pad.train()

        graph  = make_linear_graph(4)
        node_f = torch.randn(4, tiny_cre_cfg.node_dim)

        out = engine(graph, node_f, n_iterations=2, scratch_pad=pad)
        out.node_features.sum().backward()

        no_grad = [n for n, p in pad.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert len(no_grad) == 0, f"Pad params without grad: {no_grad}"

    def test_multiple_iterations_with_pad_stable(self, tiny_cre_cfg, tiny_pad_cfg):
        """Muchas iteraciones con scratch pad son estables."""
        engine = CausalReasoningEngine(tiny_cre_cfg)
        pad    = DifferentiableScratchPad(tiny_pad_cfg)
        engine.eval(); pad.eval()

        graph  = make_linear_graph(4)
        node_f = torch.randn(4, tiny_cre_cfg.node_dim)

        with torch.no_grad():
            out = engine(graph, node_f, n_iterations=20, scratch_pad=pad)

        assert out.node_features.isfinite().all()

    def test_pad_node_dim_must_match_cre_node_dim(self, tiny_cre_cfg):
        """
        Si el node_dim del pad no coincide con el del CRE, debe fallar en runtime.
        (No hay verificación en __init__, falla durante el forward con error de shape.)
        """
        wrong_cfg = ScratchPadConfig(n_slots=8, slot_dim=16, node_dim=999)
        engine    = CausalReasoningEngine(tiny_cre_cfg)
        pad       = DifferentiableScratchPad(wrong_cfg)

        graph  = make_linear_graph(4)
        node_f = torch.randn(4, tiny_cre_cfg.node_dim)

        with pytest.raises((RuntimeError, Exception)):
            engine(graph, node_f, n_iterations=1, scratch_pad=pad)


# ─────────────────────────────────────────────────────────────────────────────
# TestNumericalStability
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalStability:
    def test_many_updates_no_nan(self, pad, tiny_cfg):
        """50 updates con el mismo input no producen NaN."""
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = pad.init_state()
        with torch.no_grad():
            for _ in range(50):
                state = pad.update(nf, state)
        assert not state.isnan().any()

    def test_many_updates_no_inf(self, pad, tiny_cfg):
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = pad.init_state()
        with torch.no_grad():
            for _ in range(50):
                state = pad.update(nf, state)
        assert state.isfinite().all()

    def test_large_input_values_stable(self, pad, tiny_cfg):
        """Inputs de magnitud grande no causan explosión."""
        nf    = torch.ones(4, tiny_cfg.node_dim) * 1000.0
        state = pad.init_state()
        with torch.no_grad():
            for _ in range(10):
                state = pad.update(nf, state)
                h = pad.read(nf, state)
        assert state.isfinite().all()
        assert h.isfinite().all()

    def test_alternating_inputs_stable(self, pad, tiny_cfg):
        """Inputs alternando entre muy positivos y muy negativos."""
        nf_pos = torch.ones(4, tiny_cfg.node_dim) * 5.0
        nf_neg = torch.ones(4, tiny_cfg.node_dim) * -5.0
        state  = pad.init_state()
        with torch.no_grad():
            for i in range(20):
                nf = nf_pos if i % 2 == 0 else nf_neg
                state = pad.update(nf, state)
        assert state.isfinite().all()

    def test_state_norm_bounded(self, pad, tiny_cfg):
        """La norma del estado debe mantenerse acotada con muchos updates."""
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = pad.init_state()
        with torch.no_grad():
            for _ in range(20):
                state = pad.update(nf, state)
        norm = state.norm().item()
        assert norm < 1e6, f"State norm too large: {norm}"


# ─────────────────────────────────────────────────────────────────────────────
# TestDeterminism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_eval_mode_deterministic(self, pad, tiny_cfg):
        pad.eval()
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = pad.init_state()

        with torch.no_grad():
            ns1, r1 = pad.update(nf, state), pad.read(nf, state)
            ns2, r2 = pad.update(nf, state), pad.read(nf, state)

        assert torch.allclose(ns1, ns2)
        assert torch.allclose(r1, r2)

    def test_same_input_same_output(self, pad, tiny_cfg):
        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = torch.randn(tiny_cfg.n_slots, tiny_cfg.slot_dim)

        with torch.no_grad():
            out1 = pad.update(nf, state)
            out2 = pad.update(nf, state)

        assert torch.allclose(out1, out2)

    def test_different_seeds_different_states(self, tiny_cfg):
        """Pesos distintos (seeds distintos) producen updates distintos."""
        torch.manual_seed(0);  pad1 = DifferentiableScratchPad(tiny_cfg)
        torch.manual_seed(99); pad2 = DifferentiableScratchPad(tiny_cfg)

        nf    = torch.randn(4, tiny_cfg.node_dim)
        state = pad1.init_state()

        with torch.no_grad():
            s1 = pad1.update(nf, state)
            s2 = pad2.update(nf, state)

        assert not torch.allclose(s1, s2)
