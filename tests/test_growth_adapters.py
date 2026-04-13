"""
tests/test_growth_adapters.py — tests para growth/ (Parte 22 del MEGA-PROMPT).

Cubre:
    LoRALinear       — forward, init, enable/disable, grad propagation
    AdapterPack      — build, state_dict roundtrip, size accounting
    attach/detach    — equivalencia bit-a-bit tras detach, double-attach guard
    AdapterRegistry  — save/load/list/delete, route_by_query, persistencia
    GrowthPolicy     — umbrales 22.4, sub_motor vs expand, límite de adapters
    Integration      — aprender 5 "conceptos" → 5 adapters persistidos,
                       los 10 "originales" siguen funcionando bit-a-bit
                       cuando los adapters no están attached.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from growth.adapters import (
    LoRAConfig,
    LoRALinear,
    AdapterPack,
    build_adapter_pack,
    attach_adapter_pack,
    detach_adapter_pack,
    freeze_base_parameters,
    auto_target_paths,
)
from growth.registry import AdapterRegistry, AdapterMeta
from growth.policy import GrowthDecision, GrowthPolicy, decide_growth


# ════════════════════════════════════════════════════════════════════════════
# Fake motor — evita cargar un MoSEPipeline real en tests unitarios
# ════════════════════════════════════════════════════════════════════════════

class FakeCrystallizer(nn.Module):
    def __init__(self, d_in: int, d_hid: int) -> None:
        super().__init__()
        self.project = nn.Linear(d_in, d_hid)
        self.out = nn.Linear(d_hid, d_hid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.out(torch.relu(self.project(x))))


class FakeCRE(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(d, d)
        self.message = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.message(torch.relu(self.input_proj(x)))


class FakeMotor(nn.Module):
    """Imita la estructura .crystallizer / .cre de los motores reales."""
    def __init__(self, d_in: int = 8, d_hid: int = 16) -> None:
        super().__init__()
        self.crystallizer = FakeCrystallizer(d_in, d_hid)
        self.cre = FakeCRE(d_hid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cre(self.crystallizer(x))


ALL_TARGETS = [
    "crystallizer.project",
    "crystallizer.out",
    "cre.input_proj",
    "cre.message",
]


def _deterministic_motor(seed: int = 123) -> FakeMotor:
    g = torch.Generator().manual_seed(seed)
    m = FakeMotor()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5, generator=g))
    return m


# ════════════════════════════════════════════════════════════════════════════
# LoRALinear
# ════════════════════════════════════════════════════════════════════════════

class TestLoRALinear:
    def test_forward_shape(self):
        base = nn.Linear(8, 16)
        lora = LoRALinear(base, LoRAConfig(rank=4, alpha=8))
        x = torch.randn(2, 3, 8)
        y = lora(x)
        assert y.shape == (2, 3, 16)

    def test_zero_init_matches_base(self):
        """Con B=0 el delta es 0 — la salida debe igualar al base exactamente."""
        base = nn.Linear(8, 16)
        lora = LoRALinear(base, LoRAConfig(rank=4, alpha=8))
        x = torch.randn(5, 8)
        y_base = base(x)
        y_lora = lora(x)
        assert torch.allclose(y_base, y_lora, atol=1e-7)

    def test_nonzero_after_training_step(self):
        base = nn.Linear(8, 16)
        lora = LoRALinear(base, LoRAConfig(rank=4, alpha=8))
        with torch.no_grad():
            lora.lora_B.add_(0.1)  # simular update
        x = torch.randn(5, 8)
        y_base = base(x)
        y_lora = lora(x)
        assert not torch.allclose(y_base, y_lora)

    def test_disabled_matches_base(self):
        base = nn.Linear(8, 16)
        lora = LoRALinear(base, LoRAConfig(rank=4, alpha=8))
        with torch.no_grad():
            lora.lora_B.add_(0.3)
        lora.enabled = False
        x = torch.randn(5, 8)
        assert torch.allclose(lora(x), base(x), atol=1e-7)

    def test_rejects_non_linear(self):
        with pytest.raises(TypeError):
            LoRALinear(nn.Conv1d(4, 4, 3), LoRAConfig())  # type: ignore[arg-type]

    def test_rejects_zero_rank(self):
        with pytest.raises(ValueError):
            LoRALinear(nn.Linear(4, 4), LoRAConfig(rank=0))

    def test_grad_on_lora_only(self):
        base = nn.Linear(8, 16)
        lora = LoRALinear(base, LoRAConfig(rank=4, alpha=8))
        # Congelar base
        for p in base.parameters():
            p.requires_grad_(False)
        x = torch.randn(3, 8)
        y = lora(x).sum()
        # Forzamos un path no-cero
        with torch.no_grad():
            lora.lora_B.add_(0.1)
        y = lora(x).sum()
        y.backward()
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert base.weight.grad is None
        assert base.bias.grad is None

    def test_pack_adapter_state_dict_contains_only_lora(self):
        """El state_dict filtrado del pack no incluye pesos base, sólo lora_A/B."""
        m = _deterministic_motor()
        pack = build_adapter_pack(
            m, ["crystallizer.project"], LoRAConfig(rank=4), "c", "forge_c"
        )
        state = pack.adapter_state_dict()
        keys = set(state.keys())
        # Sin "weight" ni "bias" del base.
        assert all("base" not in k for k in keys)
        assert all(k.endswith("lora_A") or k.endswith("lora_B") for k in keys)

    def test_param_count(self):
        base = nn.Linear(8, 16)
        lora = LoRALinear(base, LoRAConfig(rank=4))
        expected = 4 * 8 + 16 * 4
        assert lora.num_adapter_parameters() == expected


# ════════════════════════════════════════════════════════════════════════════
# AdapterPack
# ════════════════════════════════════════════════════════════════════════════

class TestAdapterPack:
    def test_build_covers_all_targets(self):
        m = _deterministic_motor()
        pack = build_adapter_pack(
            m, ALL_TARGETS, LoRAConfig(rank=4), "python", "forge_c"
        )
        assert set(pack.target_paths()) == set(ALL_TARGETS)
        assert pack.num_adapter_parameters() > 0

    def test_get_unknown_path_raises(self):
        m = _deterministic_motor()
        pack = build_adapter_pack(m, ALL_TARGETS[:1], LoRAConfig(), "c1", "forge_c")
        with pytest.raises(KeyError):
            pack.get("does.not.exist")

    def test_state_dict_roundtrip(self):
        m = _deterministic_motor()
        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(rank=4), "c", "forge_c")
        # Mutar pesos para diferenciar de init
        for path in ALL_TARGETS:
            with torch.no_grad():
                pack.get(path).lora_B.normal_(0, 0.1)
        state = pack.adapter_state_dict()
        # Construir otro pack y cargar
        m2 = _deterministic_motor()
        pack2 = build_adapter_pack(m2, ALL_TARGETS, LoRAConfig(rank=4), "c", "forge_c")
        pack2.load_adapter_state_dict(state)
        # Valores iguales
        for path in ALL_TARGETS:
            assert torch.allclose(pack.get(path).lora_B, pack2.get(path).lora_B)
            assert torch.allclose(pack.get(path).lora_A, pack2.get(path).lora_A)

    def test_set_enabled_propagates(self):
        m = _deterministic_motor()
        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(rank=4), "c", "forge_c")
        pack.set_enabled(False)
        for path in ALL_TARGETS:
            assert pack.get(path).enabled is False
        pack.set_enabled(True)
        for path in ALL_TARGETS:
            assert pack.get(path).enabled is True

    def test_size_bytes_positive(self):
        m = _deterministic_motor()
        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(rank=8), "c", "forge_c")
        assert pack.size_bytes() > 0

    def test_duplicate_add_layer_rejected(self):
        m = _deterministic_motor()
        pack = build_adapter_pack(m, ["crystallizer.project"], LoRAConfig(), "c", "forge_c")
        lora = LoRALinear(m.crystallizer.project, LoRAConfig())  # type: ignore[arg-type]
        with pytest.raises(KeyError):
            pack.add_layer("crystallizer.project", lora)


# ════════════════════════════════════════════════════════════════════════════
# attach / detach
# ════════════════════════════════════════════════════════════════════════════

class TestAttachDetach:
    def test_attach_replaces_linear_with_lora(self):
        m = _deterministic_motor()
        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(), "c", "forge_c")
        attach_adapter_pack(m, pack)
        assert isinstance(m.crystallizer.project, LoRALinear)
        assert isinstance(m.cre.message, LoRALinear)
        assert pack.attached

    def test_detach_restores_original_linear(self):
        m = _deterministic_motor()
        original_proj = m.crystallizer.project
        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(), "c", "forge_c")
        attach_adapter_pack(m, pack)
        detach_adapter_pack(m, pack)
        assert m.crystallizer.project is original_proj
        assert not pack.attached

    def test_output_unchanged_after_detach(self):
        """Garantía dura: 10/10 originales intactos tras attach/detach."""
        m = _deterministic_motor()
        x = torch.randn(4, 3, 8)
        y_before = m(x).detach().clone()

        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(rank=4), "c", "forge_c")
        attach_adapter_pack(m, pack)
        # Mutar pesos del adapter (simula entrenamiento)
        for path in ALL_TARGETS:
            with torch.no_grad():
                pack.get(path).lora_B.normal_(0, 0.5)
        # Con adapter activo DEBE diferir
        y_adapted = m(x)
        assert not torch.allclose(y_before, y_adapted, atol=1e-5)

        # Detach → salida bit-a-bit
        detach_adapter_pack(m, pack)
        y_after = m(x)
        assert torch.allclose(y_before, y_after, atol=1e-7)

    def test_disabled_pack_matches_base_even_while_attached(self):
        m = _deterministic_motor()
        x = torch.randn(4, 8)
        y_before = m(x).detach().clone()

        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(rank=4), "c", "forge_c")
        attach_adapter_pack(m, pack)
        for path in ALL_TARGETS:
            with torch.no_grad():
                pack.get(path).lora_B.normal_(0, 0.5)
        pack.set_enabled(False)
        y_disabled = m(x)
        assert torch.allclose(y_before, y_disabled, atol=1e-7)

    def test_double_attach_raises(self):
        m = _deterministic_motor()
        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(), "c", "forge_c")
        attach_adapter_pack(m, pack)
        with pytest.raises(RuntimeError):
            attach_adapter_pack(m, pack)

    def test_detach_idempotent(self):
        m = _deterministic_motor()
        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(), "c", "forge_c")
        # Sin haber hecho attach, detach es no-op
        detach_adapter_pack(m, pack)
        assert not pack.attached

    def test_freeze_base_leaves_lora_trainable(self):
        m = _deterministic_motor()
        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(rank=4), "c", "forge_c")
        attach_adapter_pack(m, pack)
        frozen = freeze_base_parameters(m)
        assert frozen > 0
        # lora_A / lora_B siguen con grad
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        assert any("lora_A" in n for n in trainable)
        assert any("lora_B" in n for n in trainable)
        assert not any(n.endswith("weight") and "lora" not in n for n in trainable)


# ════════════════════════════════════════════════════════════════════════════
# AdapterRegistry
# ════════════════════════════════════════════════════════════════════════════

class TestAdapterRegistry:
    def test_save_and_list(self, tmp_path: Path):
        reg = AdapterRegistry(tmp_path)
        m = _deterministic_motor()
        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(rank=4), "python", "forge_c")
        meta = reg.save(pack, parent_brain_version="v1", tags=["lang", "code"])
        assert meta.concept_name == "python"
        lst = reg.list()
        assert len(lst) == 1
        assert lst[0].concept_name == "python"
        assert lst[0].rank == 4
        assert lst[0].num_params == pack.num_adapter_parameters()

    def test_persistence_on_disk(self, tmp_path: Path):
        reg = AdapterRegistry(tmp_path)
        m = _deterministic_motor()
        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(rank=4), "rust", "forge_c")
        reg.save(pack)
        # Nuevo registry apuntando al mismo root — debe ver el adapter.
        reg2 = AdapterRegistry(tmp_path)
        lst = reg2.list()
        assert len(lst) == 1
        assert lst[0].concept_name == "rust"

    def test_load_into_roundtrip(self, tmp_path: Path):
        reg = AdapterRegistry(tmp_path)
        m = _deterministic_motor()
        pack = build_adapter_pack(m, ALL_TARGETS, LoRAConfig(rank=4), "go", "forge_c")
        for path in ALL_TARGETS:
            with torch.no_grad():
                pack.get(path).lora_B.normal_(0, 0.2)
        reg.save(pack)

        # Otro motor/pack y carga
        m2 = _deterministic_motor()
        pack2 = build_adapter_pack(m2, ALL_TARGETS, LoRAConfig(rank=4), "go", "forge_c")
        reg.load_into(pack2)
        for path in ALL_TARGETS:
            assert torch.allclose(pack.get(path).lora_B, pack2.get(path).lora_B)

    def test_list_filter_by_motor(self, tmp_path: Path):
        reg = AdapterRegistry(tmp_path)
        m = _deterministic_motor()
        reg.save(build_adapter_pack(m, ["crystallizer.project"], LoRAConfig(), "c1", "forge_c"))
        reg.save(build_adapter_pack(m, ["cre.message"], LoRAConfig(), "c2", "axiom"))
        fc = reg.list(motor_name="forge_c")
        assert len(fc) == 1
        assert fc[0].motor_name == "forge_c"
        assert reg.count() == 2

    def test_delete_removes_from_disk(self, tmp_path: Path):
        reg = AdapterRegistry(tmp_path)
        m = _deterministic_motor()
        reg.save(build_adapter_pack(m, ALL_TARGETS, LoRAConfig(), "python", "forge_c"))
        assert reg.exists("forge_c", "python")
        assert reg.delete("forge_c", "python")
        assert not reg.exists("forge_c", "python")
        assert reg.list() == []
        assert not reg.delete("forge_c", "python")  # segunda vez no-op

    def test_route_by_query_name_hit(self, tmp_path: Path):
        reg = AdapterRegistry(tmp_path)
        m = _deterministic_motor()
        reg.save(build_adapter_pack(m, ALL_TARGETS, LoRAConfig(), "rust", "forge_c"))
        reg.save(build_adapter_pack(m, ALL_TARGETS, LoRAConfig(), "python", "forge_c"))
        hits = reg.route_by_query("enséñame rust por favor", motor_name="forge_c")
        assert any(h.concept_name == "rust" for h in hits)
        assert not any(h.concept_name == "python" for h in hits)

    def test_route_by_query_tag_hit(self, tmp_path: Path):
        reg = AdapterRegistry(tmp_path)
        m = _deterministic_motor()
        reg.save(
            build_adapter_pack(m, ALL_TARGETS, LoRAConfig(), "ownership", "forge_c"),
            tags=["rust", "memory"],
        )
        hits = reg.route_by_query("tengo dudas con rust", motor_name="forge_c")
        assert len(hits) == 1
        assert hits[0].concept_name == "ownership"

    def test_update_meta_persists(self, tmp_path: Path):
        reg = AdapterRegistry(tmp_path)
        m = _deterministic_motor()
        reg.save(build_adapter_pack(m, ALL_TARGETS, LoRAConfig(), "python", "forge_c"))
        meta = reg.get_meta("forge_c", "python")
        meta.reward_score = 0.85
        meta.usage_count = 12
        reg.update_meta(meta)
        reg2 = AdapterRegistry(tmp_path)
        meta2 = reg2.get_meta("forge_c", "python")
        assert meta2.reward_score == 0.85
        assert meta2.usage_count == 12


# ════════════════════════════════════════════════════════════════════════════
# GrowthPolicy
# ════════════════════════════════════════════════════════════════════════════

class TestGrowthPolicy:
    def test_high_accuracy_no_growth(self):
        assert decide_growth(0.85) == GrowthDecision.NO_GROWTH
        assert decide_growth(0.70) == GrowthDecision.NO_GROWTH

    def test_mid_accuracy_adapter(self):
        assert decide_growth(0.55) == GrowthDecision.ADAPTER
        assert decide_growth(0.30) == GrowthDecision.ADAPTER

    def test_low_accuracy_expand_or_submotor(self):
        assert decide_growth(0.10, domain_distinct=False) == GrowthDecision.EXPAND_MOTOR
        assert decide_growth(0.10, domain_distinct=True) == GrowthDecision.SUB_MOTOR

    def test_full_motor_escalates(self):
        # Rango adapter pero el motor está lleno → debe subir un escalón
        d = decide_growth(
            0.5, current_adapters_in_motor=8, domain_distinct=False
        )
        assert d == GrowthDecision.EXPAND_MOTOR

    def test_full_motor_distinct_domain_becomes_sub(self):
        d = decide_growth(
            0.5, current_adapters_in_motor=8, domain_distinct=True
        )
        assert d == GrowthDecision.SUB_MOTOR

    def test_out_of_range_accuracy_raises(self):
        with pytest.raises(ValueError):
            decide_growth(1.2)
        with pytest.raises(ValueError):
            decide_growth(-0.1)

    def test_custom_policy_thresholds(self):
        p = GrowthPolicy(adapter_lower=0.2, adapter_upper=0.8, max_adapters_per_motor=3)
        assert decide_growth(0.25, policy=p) == GrowthDecision.ADAPTER
        assert decide_growth(0.81, policy=p) == GrowthDecision.NO_GROWTH


# ════════════════════════════════════════════════════════════════════════════
# Integration: 5 conceptos + 10 originales protegidos
# ════════════════════════════════════════════════════════════════════════════

class TestFiveConceptsIntegration:
    """Simula el escenario del MEGA-PROMPT: aprender 5 conceptos y verificar
    que los 10 'originales' siguen funcionando bit-a-bit cuando los adapters
    están detached (producción normal sin routing a ninguno).
    """

    def test_learn_five_concepts_preserves_originals(self, tmp_path: Path):
        motor = _deterministic_motor(seed=42)
        reg = AdapterRegistry(tmp_path)

        # "Exam" de 10 inputs originales → capturamos el output de referencia.
        torch.manual_seed(0)
        exam_inputs = [torch.randn(1, 8) for _ in range(10)]
        exam_outputs = [motor(x).detach().clone() for x in exam_inputs]

        # Aprender 5 conceptos: por cada uno construimos un pack, lo
        # adjuntamos, mutamos los pesos del lora (simula fine-tune),
        # guardamos al registry y detach.
        concepts = ["python", "rust", "go", "sql", "bash"]
        for name in concepts:
            pack = build_adapter_pack(
                motor, ALL_TARGETS, LoRAConfig(rank=4, alpha=8), name, "forge_c"
            )
            attach_adapter_pack(motor, pack)
            for path in ALL_TARGETS:
                with torch.no_grad():
                    pack.get(path).lora_B.normal_(0, 0.3)
            # Verifica que con el adapter activo el motor responde distinto.
            y_with = motor(exam_inputs[0])
            assert not torch.allclose(y_with, exam_outputs[0], atol=1e-5)
            # Guardar y detach — a partir de aquí el motor vuelve a base.
            reg.save(pack, tags=[name])
            detach_adapter_pack(motor, pack)

        # 5 adapters en disco
        assert reg.count(motor_name="forge_c") == 5
        names = {m.concept_name for m in reg.list()}
        assert names == set(concepts)

        # Los 10 originales siguen dando la MISMA salida bit-a-bit.
        for x, y_ref in zip(exam_inputs, exam_outputs):
            y_now = motor(x)
            assert torch.allclose(y_now, y_ref, atol=1e-7), (
                "Original exam output diverged — base weights were mutated."
            )

    def test_auto_target_paths_fake_motor(self):
        m = _deterministic_motor()
        # El fake motor no tiene q_proj/out_proj, pero sí "project"
        paths = auto_target_paths(m)
        assert "crystallizer.project" in paths

    def test_auto_target_paths_max_limit(self):
        m = _deterministic_motor()
        paths = auto_target_paths(m, patterns=["project", "message", "input_proj", "out"], max_targets=2)
        assert len(paths) <= 2

    def test_real_code_motor_detach_preserves_weights(self):
        """Smoke test contra CodeMotor real: attach+detach no altera pesos base."""
        from motors.forge_c.motor import CodeMotor, CodeMotorConfig

        motor = CodeMotor(CodeMotorConfig())
        # Snapshot de los pesos base relevantes
        targets = auto_target_paths(motor, max_targets=6)
        assert len(targets) > 0, "CodeMotor should expose proj-like layers"

        snap = {}
        for t in targets:
            parent_attrs = t.split(".")
            mod = motor
            for a in parent_attrs:
                mod = getattr(mod, a)
            snap[t] = mod.weight.detach().clone()  # type: ignore[attr-defined]

        pack = build_adapter_pack(
            motor, targets, LoRAConfig(rank=4, alpha=8), "python", "forge_c"
        )
        attach_adapter_pack(motor, pack)
        # Entrenar los lora con random updates
        for t in targets:
            with torch.no_grad():
                pack.get(t).lora_B.normal_(0, 0.2)
        detach_adapter_pack(motor, pack)

        # Los pesos base NO deben haber cambiado
        for t in targets:
            mod = motor
            for a in t.split("."):
                mod = getattr(mod, a)
            assert torch.allclose(mod.weight, snap[t], atol=1e-7), (  # type: ignore[attr-defined]
                f"Base weight of {t} was mutated — LoRA leaked into base."
            )

    def test_load_saved_adapter_reproduces_effect(self, tmp_path: Path):
        """Cargar un adapter guardado debe reproducir su efecto en la salida."""
        motor = _deterministic_motor(seed=7)
        reg = AdapterRegistry(tmp_path)

        pack = build_adapter_pack(
            motor, ALL_TARGETS, LoRAConfig(rank=4), "sql", "forge_c"
        )
        attach_adapter_pack(motor, pack)
        for path in ALL_TARGETS:
            with torch.no_grad():
                pack.get(path).lora_B.normal_(0, 0.25)
        x = torch.randn(2, 8)
        y_with_effect = motor(x).detach().clone()
        reg.save(pack)
        detach_adapter_pack(motor, pack)

        # Motor fresh (otro proceso) — rebuild y cargar
        motor2 = _deterministic_motor(seed=7)
        pack2 = build_adapter_pack(
            motor2, ALL_TARGETS, LoRAConfig(rank=4), "sql", "forge_c"
        )
        reg.load_into(pack2)
        attach_adapter_pack(motor2, pack2)
        y_reloaded = motor2(x)
        assert torch.allclose(y_with_effect, y_reloaded, atol=1e-6)
