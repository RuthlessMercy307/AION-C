"""
growth/adapters.py — LoRA-style adapters para motores AION-C (Parte 22.1).

Un adapter es una actualización low-rank sobre un conjunto de nn.Linear de un
motor. El motor base permanece CONGELADO y el adapter entrena sólo dos matrices
pequeñas (A de rango r y B de rango r) por cada Linear afectado.

Diseño:
    - LoRALinear envuelve un nn.Linear existente SIN clonar sus pesos.
    - AdapterPack agrupa todos los LoRALinear de un adapter (target_name → LoRALinear).
    - attach_adapter_pack monkey-patcha los Linear del motor para que su forward
      pase por los LoRALinear correspondientes.
    - detach_adapter_pack revierte exactamente ese patch.
    - Guardado: sólo los state_dicts de los lora_A / lora_B — NUNCA se
      reguarda el base, que ya vive en brain/vN/weights.pt.

Garantía estructural:
    Después de detach_adapter_pack, una query produce EXACTAMENTE la misma salida
    que antes de attach_adapter_pack (bit-a-bit). Esto es lo que protege los
    "10/10 originales" cuando se aprenden N conceptos nuevos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple

import math
import torch
import torch.nn as nn


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class LoRAConfig:
    """Hiperparámetros de un adapter LoRA.

    rank:   dimensión del cuello de botella (r). 4-16 es el rango útil típico.
    alpha:  factor de escala del delta. El delta efectivo es (alpha/rank) * B A x.
    dropout: dropout sobre la entrada al adapter durante training (0.0 = off).
    init_scale: escala de init de A (Kaiming uniform). B empieza en 0 → delta=0.
    """
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    init_scale: float = 1.0

    def scaling(self) -> float:
        if self.rank <= 0:
            return 0.0
        return float(self.alpha) / float(self.rank)


# ════════════════════════════════════════════════════════════════════════════
# LoRALinear
# ════════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """Wrap de un nn.Linear con una actualización low-rank entrenable.

    Forward:
        y = base(x) + scaling * (dropout(x) @ A^T @ B^T)

    donde A: [rank, in_features] y B: [out_features, rank].
    El base es REFERENCIADO (no clonado); queda como atributo pero NO se
    registra como submódulo — así state_dict(LoRALinear) contiene sólo A y B.

    Enable/disable:
        if not self.enabled → forward degenera en base(x) exacto.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear requires nn.Linear, got {type(base).__name__}")
        if config.rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {config.rank}")

        self.config = config
        # Base se registra como submódulo normal para que siga siendo
        # descubrible desde motor.named_parameters() tras el attach. El
        # guardado selectivo (sólo lora_A/B) vive en AdapterPack.adapter_state_dict().
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features

        # Parámetros entrenables del adapter
        self.lora_A = nn.Parameter(torch.zeros(config.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, config.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5) / max(config.init_scale, 1e-6))
        # B empieza en 0 para que el delta inicial sea exactamente 0.
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=config.dropout) if config.dropout > 0 else nn.Identity()
        self.enabled: bool = True
        self._scaling = config.scaling()

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.config.rank}, alpha={self.config.alpha}, enabled={self.enabled}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if not self.enabled or self._scaling == 0.0:
            return base_out
        # delta = scaling * (x @ A^T) @ B^T
        x_drop = self.dropout(x)
        lora = torch.nn.functional.linear(x_drop, self.lora_A)   # [..., rank]
        lora = torch.nn.functional.linear(lora, self.lora_B)      # [..., out]
        return base_out + self._scaling * lora

    def num_adapter_parameters(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


# ════════════════════════════════════════════════════════════════════════════
# AdapterPack
# ════════════════════════════════════════════════════════════════════════════

class AdapterPack(nn.Module):
    """Agrupa los LoRALinear de un adapter, indexados por ruta (dotted path).

    Ejemplo de rutas: "crystallizer.project", "cre.input_proj".
    """

    def __init__(
        self,
        concept_name: str,
        motor_name: str,
        config: LoRAConfig,
    ) -> None:
        super().__init__()
        self.concept_name = concept_name
        self.motor_name = motor_name
        self.config = config
        self.layers: nn.ModuleDict = nn.ModuleDict()
        self._attached: bool = False
        self._target_paths: List[str] = []

    # ── Construcción ──────────────────────────────────────────────────────
    def add_layer(self, path: str, lora_linear: LoRALinear) -> None:
        key = _escape_path(path)
        if key in self.layers:
            raise KeyError(f"Layer already present in pack: {path}")
        # ModuleDict no admite puntos — usamos una clave escapada.
        self.layers[key] = lora_linear
        self._target_paths.append(path)

    def target_paths(self) -> List[str]:
        return list(self._target_paths)

    def get(self, path: str) -> LoRALinear:
        return self.layers[_escape_path(path)]  # type: ignore[return-value]

    # ── Enable/disable sin re-attach ──────────────────────────────────────
    def set_enabled(self, enabled: bool) -> None:
        for layer in self.layers.values():
            layer.enabled = bool(enabled)  # type: ignore[attr-defined]

    @property
    def attached(self) -> bool:
        return self._attached

    # ── Conteo ────────────────────────────────────────────────────────────
    def num_adapter_parameters(self) -> int:
        return sum(l.num_adapter_parameters() for l in self.layers.values())  # type: ignore[attr-defined]

    def size_bytes(self) -> int:
        # fp32 por default → 4 bytes/param
        total = 0
        for l in self.layers.values():
            for p in (l.lora_A, l.lora_B):  # type: ignore[attr-defined]
                total += p.numel() * p.element_size()
        return total

    # ── State dict sólo del adapter ───────────────────────────────────────
    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict que contiene SÓLO los lora_A/lora_B de cada layer.

        Nunca incluye los pesos base, que pertenecen al motor congelado.
        """
        out: Dict[str, torch.Tensor] = {}
        for key, layer in self.layers.items():
            out[f"{key}.lora_A"] = layer.lora_A.detach().clone()  # type: ignore[attr-defined]
            out[f"{key}.lora_B"] = layer.lora_B.detach().clone()  # type: ignore[attr-defined]
        return out

    def load_adapter_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for key, layer in self.layers.items():
            a = state[f"{key}.lora_A"]
            b = state[f"{key}.lora_B"]
            with torch.no_grad():
                layer.lora_A.copy_(a)  # type: ignore[attr-defined]
                layer.lora_B.copy_(b)  # type: ignore[attr-defined]


def _escape_path(path: str) -> str:
    return path.replace(".", "__")


def _unescape_path(key: str) -> str:
    return key.replace("__", ".")


# ════════════════════════════════════════════════════════════════════════════
# attach / detach
# ════════════════════════════════════════════════════════════════════════════

def _resolve_parent(root: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    cur: nn.Module = root
    for p in parts[:-1]:
        cur = getattr(cur, p)
    return cur, parts[-1]


def _find_linear(root: nn.Module, dotted: str) -> nn.Linear:
    parent, attr = _resolve_parent(root, dotted)
    mod = getattr(parent, attr)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"Target '{dotted}' is not nn.Linear (got {type(mod).__name__})")
    return mod


def build_adapter_pack(
    motor: nn.Module,
    target_paths: Iterable[str],
    config: LoRAConfig,
    concept_name: str,
    motor_name: str,
) -> AdapterPack:
    """Construye un AdapterPack detectando los Linear en los paths dados.

    NO adjunta todavía el pack al motor — sólo crea los LoRALinear.
    """
    pack = AdapterPack(concept_name=concept_name, motor_name=motor_name, config=config)
    for path in target_paths:
        base = _find_linear(motor, path)
        lora = LoRALinear(base, config)
        pack.add_layer(path, lora)
    return pack


def attach_adapter_pack(motor: nn.Module, pack: AdapterPack) -> None:
    """Monkey-patcha los Linear del motor reemplazándolos por sus LoRALinear.

    Post-condición:
        Para cada path en pack.target_paths(), el atributo del padre apunta al
        LoRALinear en vez de al Linear original. El Linear original sigue vivo
        como `lora_linear.base` y conserva sus pesos (congelados).
    """
    if pack._attached:
        raise RuntimeError(f"Pack {pack.concept_name} already attached")
    for path in pack.target_paths():
        parent, attr = _resolve_parent(motor, path)
        lora = pack.get(path)
        # Sanity: el Linear vivo en el motor debe ser el mismo que base del lora.
        current = getattr(parent, attr)
        if current is not lora.base:
            raise RuntimeError(
                f"Motor layer at '{path}' does not match lora.base — "
                f"motor was mutated since the pack was built."
            )
        setattr(parent, attr, lora)
    pack._attached = True


def detach_adapter_pack(motor: nn.Module, pack: AdapterPack) -> None:
    """Revierte attach_adapter_pack — restaura los Linear originales.

    Post-condición:
        El motor queda bit-a-bit equivalente al estado previo al attach,
        porque el base del lora nunca fue tocado.
    """
    if not pack._attached:
        return
    for path in pack.target_paths():
        parent, attr = _resolve_parent(motor, path)
        lora = pack.get(path)
        current = getattr(parent, attr)
        if current is not lora:
            # Alguien cambió el layer — no podemos revertir con seguridad.
            raise RuntimeError(
                f"Cannot detach '{path}': current layer is not the expected LoRALinear."
            )
        setattr(parent, attr, lora.base)
    pack._attached = False


def auto_target_paths(
    motor: nn.Module,
    patterns: Optional[List[str]] = None,
    max_targets: int = 8,
) -> List[str]:
    """Heurística para elegir Linear target de un motor sin configurar a mano.

    Por default prefiere proyecciones de atención (q/k/v/out_proj) y los
    proyectores principales — los puntos de mayor impacto para un LoRA.

    Args:
        motor: nn.Module con Linear en su interior.
        patterns: substrings a priorizar. Default: ["q_proj","k_proj","v_proj",
                  "out_proj","input_proj","project","output_proj"].
        max_targets: número máximo de paths a devolver.

    Returns:
        Lista de dotted-paths (compatibles con build_adapter_pack).
    """
    if patterns is None:
        patterns = [
            "q_proj", "k_proj", "v_proj", "out_proj",
            "input_proj", "output_proj", "project",
        ]
    matches: List[str] = []
    for name, mod in motor.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(pat in name for pat in patterns):
            matches.append(name)
    return matches[:max_targets]


def freeze_base_parameters(motor: nn.Module) -> int:
    """Pone requires_grad=False en TODOS los parámetros base del motor.

    No toca los lora_A / lora_B de packs ya adjuntos — esos tienen sus propios
    parámetros nuevos que quedarán con requires_grad=True.

    Returns: número de parámetros congelados.
    """
    frozen = 0
    for name, p in motor.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            continue
        if p.requires_grad:
            p.requires_grad_(False)
            frozen += p.numel()
    return frozen
