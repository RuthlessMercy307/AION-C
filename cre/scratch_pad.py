"""
cre/scratch_pad.py — DifferentiableScratchPad
==============================================

Memoria de trabajo diferenciable para el CausalReasoningEngine.

POR QUÉ UN SCRATCH PAD:
    El message passing propaga información LOCAL (vecino → nodo).
    Pero el razonamiento requiere recordar hechos GLOBALES entre iteraciones:
        "Estoy intentando demostrar X"
        "Ya establecí que A causa B en la iteración 3"
        "La contradicción entre nodo 2 y nodo 5 aún no está resuelta"

    El scratch pad es la implementación de esa "memoria de trabajo":
        - Persiste entre iteraciones del CRE (no se reinicia en cada paso)
        - Los nodos pueden LEER de ella (recuerdan contexto global)
        - Los nodos pueden ESCRIBIR en ella (registran hallazgos)
        - La escritura es SELECTIVA: solo actualiza los slots relevantes

ARQUITECTURA (Neural Turing Machine simplificada):

    ESTADO: Tensor [n_slots, slot_dim]
        - n_slots slots independientes de memoria
        - Cada slot es un vector de slot_dim dimensiones
        - Inicializado en cero (mente en blanco al inicio de cada inferencia)

    READ (cross-attention):
        Q = read_q_proj(node_features)    [N, slot_dim]
        K = read_k_proj(state)            [n_slots, slot_dim]
        V = read_v_proj(state)            [n_slots, slot_dim]
        attn = softmax(Q @ K.T / √slot_dim)  [N, n_slots]
        out  = attn @ V                   [N, slot_dim]
        return LN(node_features + read_out_proj(out))  residual + norm

    UPDATE (NTM erase-write):
        write_addr = softmax(addr_proj(nodes) @ state.T)  [N, n_slots]
        write_gate = sigmoid(gate_head(nodes))             [N, 1]
        erase_vec  = sigmoid(erase_head(nodes))           [N, slot_dim]
        content    = tanh(content_head(nodes))            [N, slot_dim]

        # Aggregate over all nodes (each node votes on each slot)
        weighted = write_gate * write_addr               [N, n_slots]
        erase_agg  = weighted.T @ erase_vec              [n_slots, slot_dim], clamp [0,1]
        write_agg  = weighted.T @ content                [n_slots, slot_dim]

        new_state = state * (1 - erase_agg) + write_agg

    SEMANTICS:
        erase_vec ≈ 1  → "olvidar lo que hay en este slot"
        write_gate ≈ 0 → "no escribir en este slot ahora"
        erase_vec ≈ 0  → "preservar el contenido"

DIFFERENTIABILITY:
    Todas las operaciones son diferenciables:
        - softmax (gradient through attention weights)
        - sigmoid (smooth gate)
        - tanh (smooth content)
        - matmul y scatter implícito vía matmul
        - erase: multiplicación element-wise (smooth)

    Gradientes fluyen de read() hacia update() vía el estado compartido.
    Esto permite que el training sepa "qué escribir para que la lectura sea útil".

INTEGRATION WITH CRE:
    En cada iteración del CausalReasoningEngine:
        h = scratch_pad.read(h, pad_state)
        pad_state = scratch_pad.update(h, pad_state)

    El estado pad_state se inicializa al inicio de cada forward() del engine
    y se propaga iteración a iteración.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScratchPadConfig:
    """
    Configuración del DifferentiableScratchPad.

    Tiny para testing: n_slots=16, slot_dim=128
    node_dim debe coincidir con CREConfig.node_dim.
    """
    n_slots:  int = 16    # número de slots de memoria
    slot_dim: int = 128   # dimensión de cada slot
    node_dim: int = 256   # dimensión de los nodos (= CREConfig.node_dim)
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {self.n_slots}")
        if self.slot_dim < 1:
            raise ValueError(f"slot_dim must be >= 1, got {self.slot_dim}")
        if self.node_dim < 1:
            raise ValueError(f"node_dim must be >= 1, got {self.node_dim}")


# ─────────────────────────────────────────────────────────────────────────────
# DIFFERENTIABLE SCRATCH PAD
# ─────────────────────────────────────────────────────────────────────────────

class DifferentiableScratchPad(nn.Module):
    """
    Memoria de trabajo diferenciable para el CRE.

    Uso básico:
        config = ScratchPadConfig(n_slots=16, slot_dim=128, node_dim=256)
        pad    = DifferentiableScratchPad(config)

        # Inicializar estado (al comienzo de cada inferencia)
        state = pad.init_state(device=device)        # [16, 128]

        # Leer: nodos consultan la memoria
        h_aug = pad.read(node_features, state)       # [N, 256]

        # Actualizar: nodos escriben a la memoria
        state = pad.update(node_features, state)     # [16, 128]

    Uso con CRE (la integración que implementa el loop):
        for iteration in range(n_iterations):
            h, e = layer(h, e, graph)
            h     = pad.read(h, pad_state)
            pad_state = pad.update(h, pad_state)
    """

    def __init__(self, config: ScratchPadConfig) -> None:
        super().__init__()
        self.config = config
        N = config.node_dim
        S = config.slot_dim
        K = config.n_slots

        # ── Mecanismo de lectura (cross-attention) ────────────────────────────
        # Nodos como queries; slots como keys y values
        self.read_q_proj   = nn.Linear(N, S, bias=False)
        self.read_k_proj   = nn.Linear(S, S, bias=False)
        self.read_v_proj   = nn.Linear(S, S, bias=False)
        self.read_out_proj = nn.Linear(S, N, bias=False)  # proyecto de vuelta a node_dim
        self.read_norm     = nn.LayerNorm(N, eps=config.norm_eps)

        self._read_scale = math.sqrt(S) ** -1  # 1/√slot_dim para estabilidad

        # ── Mecanismo de escritura (NTM erase-write) ──────────────────────────
        # Dirección: qué slot escribir
        self.addr_proj    = nn.Linear(N, S, bias=False)

        # Gate de escritura: cuánto escribir (escalar por nodo)
        self.gate_head    = nn.Linear(N, 1)

        # Vector de borrado: qué dimensiones borrar (por nodo)
        self.erase_head   = nn.Linear(N, S)

        # Contenido a escribir (por nodo)
        self.content_head = nn.Linear(N, S)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── API pública ───────────────────────────────────────────────────────────

    def init_state(
        self,
        device: torch.device | None = None,
        dtype:  torch.dtype  | None = None,
    ) -> torch.Tensor:
        """
        Crea un estado inicial vacío (ceros).

        Returns: [n_slots, slot_dim]
            Representación de una memoria de trabajo en blanco.
        """
        return torch.zeros(
            self.config.n_slots, self.config.slot_dim,
            device=device, dtype=dtype,
        )

    def read(
        self,
        node_features: torch.Tensor,  # [N, node_dim]
        state:         torch.Tensor,  # [n_slots, slot_dim]
    ) -> torch.Tensor:
        """
        Los nodos consultan la memoria de trabajo mediante cross-attention.

        Cada nodo genera una query y atiende a todos los slots.
        El resultado se agrega (residual + LayerNorm) a las features del nodo.

        Args:
            node_features: [N, node_dim]
            state:         [n_slots, slot_dim]

        Returns:
            enriched: [N, node_dim]
                node_features + info recuperada de los slots (residual + LN)
        """
        # Proyectar nodos a queries y slots a keys/values
        Q = self.read_q_proj(node_features)          # [N, S]
        K = self.read_k_proj(state)                  # [n_slots, S]
        V = self.read_v_proj(state)                  # [n_slots, S]

        # Attention: cada nodo "mira" todos los slots
        attn_logits  = (Q @ K.T) * self._read_scale  # [N, n_slots]
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [N, n_slots], suma 1

        # Agregar contenido de los slots
        slot_content = attn_weights @ V              # [N, S]

        # Proyectar de vuelta a node_dim y aplicar residual + LN
        read_out = self.read_out_proj(slot_content)  # [N, N_dim]
        # Cast read_out to node_features dtype: under AMP, Linear ops return FP16
        # but node_features may be FP32 (crystallizer output). Addition requires
        # matching dtypes.
        return self.read_norm(node_features + read_out.to(dtype=node_features.dtype))

    def update(
        self,
        node_features: torch.Tensor,  # [N, node_dim]
        state:         torch.Tensor,  # [n_slots, slot_dim]
    ) -> torch.Tensor:
        """
        Los nodos escriben en la memoria de trabajo (NTM erase-write).

        Pipeline:
            1. write_addr: softmax → qué slot atender (distribution over slots)
            2. write_gate: sigmoid → con qué fuerza escribir (escalar)
            3. erase_vec:  sigmoid → qué dimensiones borrar (por slot)
            4. content:    tanh   → qué escribir

        Fórmula de actualización:
            weighted = write_gate * write_addr         # [N, n_slots]
            erase    = (weighted.T @ erase_vec).clamp(0,1)  # [n_slots, S]
            content  = weighted.T @ tanh_content        # [n_slots, S]
            new_state = state * (1 - erase) + content

        Args:
            node_features: [N, node_dim]
            state:         [n_slots, slot_dim]

        Returns:
            new_state: [n_slots, slot_dim]
        """
        N = node_features.shape[0]
        S = self.config.slot_dim

        # ── Paso 1: write address (cuál slot) ─────────────────────────────────
        addr_q      = self.addr_proj(node_features)      # [N, S]
        addr_logits = addr_q @ state.T                   # [N, n_slots]
        write_addr  = torch.softmax(addr_logits, dim=-1) # [N, n_slots], suma 1

        # ── Paso 2: write gate (cuánto escribir) ──────────────────────────────
        write_gate = torch.sigmoid(self.gate_head(node_features))  # [N, 1]

        # ── Paso 3: erase vector (qué borrar) ─────────────────────────────────
        erase_vec = torch.sigmoid(self.erase_head(node_features))  # [N, S]

        # ── Paso 4: contenido a escribir ──────────────────────────────────────
        content = torch.tanh(self.content_head(node_features))     # [N, S]

        # ── Agregación sobre todos los nodos ──────────────────────────────────
        # weighted[n, s] = write_gate[n] × write_addr[n, s]
        # Indica la contribución del nodo n al slot s.
        weighted = write_gate * write_addr                # [N, n_slots]

        # erase_agg[s, d] = Σ_n weighted[n,s] × erase_vec[n,d]  ∈ [0,1]
        # Cuánto se borra de cada dimensión d del slot s.
        erase_agg = (weighted.T @ erase_vec).clamp(0.0, 1.0)  # [n_slots, S]

        # write_agg[s, d] = Σ_n weighted[n,s] × content[n,d]
        # Qué contenido nuevo se agrega al slot s.
        write_agg = weighted.T @ content                  # [n_slots, S]

        # ── Actualización NTM ─────────────────────────────────────────────────
        # Primero borrar, luego escribir: new = old * (1 - erase) + content
        # Cast state to match erase_agg dtype: under AMP, matmul/sigmoid return FP16
        # but state was initialised with node_features.dtype which may be FP32.
        # Element-wise * and + require matching dtypes.
        state = state.to(dtype=erase_agg.dtype)
        new_state = state * (1.0 - erase_agg) + write_agg

        return new_state

    def update_debug(
        self,
        node_features: torch.Tensor,
        state:         torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Versión de update() que expone los tensores intermedios para testing/debugging.

        Returns:
            (new_state, write_addr, write_gate, erase_vec, content)
            - new_state:  [n_slots, slot_dim]
            - write_addr: [N, n_slots]  — distribution softmax sobre slots
            - write_gate: [N, 1]        — sigmoid, fuerza de escritura
            - erase_vec:  [N, slot_dim] — sigmoid, qué borrar
            - content:    [N, slot_dim] — tanh, qué escribir
        """
        addr_q      = self.addr_proj(node_features)
        write_addr  = torch.softmax(addr_q @ state.T, dim=-1)
        write_gate  = torch.sigmoid(self.gate_head(node_features))
        erase_vec   = torch.sigmoid(self.erase_head(node_features))
        content     = torch.tanh(self.content_head(node_features))

        weighted  = write_gate * write_addr
        erase_agg = (weighted.T @ erase_vec).clamp(0.0, 1.0)
        write_agg = weighted.T @ content
        state = state.to(dtype=erase_agg.dtype)
        new_state = state * (1.0 - erase_agg) + write_agg

        return new_state, write_addr, write_gate, erase_vec, content
