"""
encoder/mamba_layer.py — Mamba-style Selective State Space Model para AION-C
=============================================================================

Implementa el StreamEncoder como una Selective State Space Machine (S6),
la arquitectura del paper "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces" (Gu & Dao, 2023).

Por qué Mamba para el encoder de AION-C:
  - O(L) memoria: no hay attention matrix (no O(L²))
  - Estado recurrente: información del pasado comprimida en el state vector
  - Selectividad: B, C, Δ dependen del input → puede filtrar o enfatizar
  - Nota del plan: "Buenos para IO" — el SE solo necesita entender secuencias,
    no razonar sobre ellas. El razonamiento es trabajo del CEC.

Implementación PURA PYTORCH — sin kernels CUDA custom (mamba-ssm, etc.)
El scan selectivo se hace en Python con operaciones torch estándar.

Componentes:
  StreamEncoderConfig — configuración única para todo el encoder
  RMSNorm             — normalización Root Mean Square
  GatedFFN            — feed-forward gateado (SwiGLU)
  SelectiveSSM        — el S6 scan: núcleo del modelo
  MambaLayer          — SSM block + GatedFFN con residuals
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StreamEncoderConfig:
    """
    Configuración del StreamEncoder basado en Mamba.

    Tiny (testing):
        hidden_dim=256, n_layers=4, state_dim=16, vocab_size=32000
        d_inner = expand * hidden_dim = 512
        dt_rank  = ceil(256/16) = 16
        concept_dim = 128

    Parámetros aproximados (tiny): ~13M
    """
    vocab_size: int  = 32_000   # Tamaño del vocabulario
    hidden_dim: int  = 256      # D: dimensión del modelo
    n_layers:   int  = 4        # Número de MambaLayers
    state_dim:  int  = 16       # N: dimensión del estado SSM
    expand:     int  = 2        # D_inner = expand × D
    d_conv:     int  = 4        # Ancho de la convolución causal local
    dt_rank:    int  = 0        # Rango de Δ (0 = auto: ceil(D/16))
    concept_dim: int = 128      # Dimensión del espacio conceptual de salida
    ffn_mult:   int  = 4        # FFN inner_dim = ffn_mult × D
    dropout:    float = 0.0
    bias:       bool  = False
    rms_eps:    float = 1e-5

    def __post_init__(self) -> None:
        if self.dt_rank == 0:
            self.dt_rank = math.ceil(self.hidden_dim / 16)

    @property
    def d_inner(self) -> int:
        """Dimensión interna del SSM (D_inner = expand × D)."""
        return self.expand * self.hidden_dim


# ─────────────────────────────────────────────────────────────────────────────
# RMS NORM
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Más eficiente que LayerNorm (sin mean centering).
    Usado en LLaMA, Mamba, Mistral.

    x_norm = x / rms(x)  donde  rms(x) = sqrt(mean(x²) + ε)
    output = weight * x_norm
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms      = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(rms + self.eps)
        return self.weight * x_normed


# ─────────────────────────────────────────────────────────────────────────────
# GATED FFN (SwiGLU)
# ─────────────────────────────────────────────────────────────────────────────

class GatedFFN(nn.Module):
    """
    Gated Feed-Forward Network — variante SwiGLU.

    output = W_down( SiLU(W_gate(x)) ⊙ W_up(x) )

    El gate aprendido (SiLU(W_gate)) actúa como filtro selectivo:
    - Dimensiones relevantes para el concepto: gate ≈ 1 → pasan
    - Dimensiones irrelevantes: gate ≈ 0 → se suprimen

    Equivalente FLOPs a FFN estándar con ffn_mult×4 pero con mayor
    capacidad expresiva por el gating.
    """

    def __init__(self, dim: int, ffn_mult: int = 4, bias: bool = False) -> None:
        super().__init__()
        inner = dim * ffn_mult
        self.w_gate = nn.Linear(dim, inner, bias=bias)
        self.w_up   = nn.Linear(dim, inner, bias=bias)
        self.w_down = nn.Linear(inner, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))   # [B, L, inner]
        up   = self.w_up(x)             # [B, L, inner]
        return self.w_down(gate * up)   # [B, L, D]


# ─────────────────────────────────────────────────────────────────────────────
# SELECTIVE STATE SPACE MODEL (S6)
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model — núcleo del Mamba encoder.

    Ecuaciones de estado:
        h_t = Ā_t · h_{t-1} + B̄_t · u_t     [state update]
        y_t = C_t · h_t + D · u_t             [output]

    Donde Ā_t, B̄_t son discretizaciones ZOH input-dependientes:
        Ā_t = exp(Δ_t ⊗ A)
        B̄_t = Δ_t · B_t · u_t

    La "selectividad" viene de que Δ, B, C son funciones del input u.
    Esto permite al modelo ignorar información irrelevante (Δ→0)
    o copiar input directamente al estado (Δ→∞).

    Complejidad:
        Memoria: O(B·L·D·N) — lineal en L (vs O(B·L²·H) de attention)
        Compute: O(B·L·D·N) — lineal en L

    Implementación: scan secuencial en PyTorch puro.
    Optimización futura: parallel prefix scan o kernel CUDA custom.

    Args:
        config: StreamEncoderConfig
    """

    def __init__(self, config: StreamEncoderConfig) -> None:
        super().__init__()
        D       = config.d_inner
        N       = config.state_dim
        dt_rank = config.dt_rank

        self.d_inner  = D
        self.d_state  = N
        self.dt_rank  = dt_rank

        # x_proj: u → [dt_raw, B_ssm, C_ssm]  (Δ, B, C son input-dependientes)
        self.x_proj = nn.Linear(D, dt_rank + N * 2, bias=False)

        # dt_proj: dt_rank → D  (expande Δ a dimensión completa)
        self.dt_proj = nn.Linear(dt_rank, D, bias=True)

        # A: parámetro fijo en estructura, log-parametrizado para estabilidad
        # Inicialización: A[d,n] = -(n+1)  (todos negativos → decaimiento estable)
        A_init   = torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(0).expand(D, N)
        self.A_log = nn.Parameter(torch.log(A_init))    # [D, N]
        self.A_log._no_weight_decay = True              # type: ignore[assignment]

        # D: skip connection (u → y directo)
        self.D = nn.Parameter(torch.ones(D))
        self.D._no_weight_decay = True                  # type: ignore[assignment]

        # Inicialización de dt_proj inspirada en Mamba paper
        # Asegura que dt inicial produzca discretizaciones razonables
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt_init = torch.exp(
            torch.rand(D) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_weight_decay = True       # type: ignore[assignment]

        # Para inspección/testing — tamaño del tensor A_bar en último forward
        self._last_A_bar_numel: int = 0

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:             [B, L, D_inner] — input al SSM (post-conv, post-SiLU)
            return_states: si True, devuelve la trayectoria de estados h

        Returns:
            y:      [B, L, D_inner]
            states: [B, L, D_inner, N] si return_states else None
        """
        B, L, D = x.shape
        N = self.d_state

        # A: negativo (decaimiento estable). Shape: [D, N]
        A = -torch.exp(self.A_log.float())  # [D, N]

        # ── Proyecciones input-dependientes ──────────────────────────────────
        # Δ (dt), B, C son funciones del input → "selectividad"
        x_dbl = self.x_proj(x)             # [B, L, dt_rank + 2N]
        dt_raw, B_ssm, C_ssm = torch.split(
            x_dbl,
            [self.dt_rank, N, N],
            dim=-1,
        )

        # Δ: positivo y proyectado a D dimensiones
        dt = F.softplus(self.dt_proj(dt_raw))   # [B, L, D]

        # ── Discretización ZOH ────────────────────────────────────────────────
        # Ā_bar[b,l,d,n] = exp(Δ[b,l,d] · A[d,n])
        # Shape: [B, L, D, N]
        A_bar = torch.exp(
            dt.unsqueeze(-1) *                     # [B, L, D, 1]
            A.unsqueeze(0).unsqueeze(0)            # [1, 1, D, N]
        )                                          # [B, L, D, N]  ← O(L), no O(L²)

        # B̄_bar[b,l,d,n] = Δ[b,l,d] · B[b,l,n] · x[b,l,d]
        # Shape: [B, L, D, N]
        delta_B_x = (
            dt.unsqueeze(-1)   *   # [B, L, D, 1]
            B_ssm.unsqueeze(2) *   # [B, L, 1, N]
            x.unsqueeze(-1)        # [B, L, D, 1]
        )                          # [B, L, D, N]

        # Registrar tamaño para tests de escalado
        self._last_A_bar_numel = A_bar.numel()

        # ── Scan secuencial ───────────────────────────────────────────────────
        # h[b,d,n]: estado recurrente — tamaño CONSTANTE en L
        h   = x.new_zeros(B, D, N)
        ys: List[torch.Tensor] = []
        states_list: List[torch.Tensor] = []

        for t in range(L):
            # State update: h_t = Ā_t · h_{t-1} + B̄_t
            h = A_bar[:, t] * h + delta_B_x[:, t]     # [B, D, N]

            if return_states:
                states_list.append(h.detach().clone())

            # Output: y_t = Σ_n(C[b,t,n] · h[b,d,n]) + D · x[b,t,d]
            y_t = (C_ssm[:, t].unsqueeze(1) * h).sum(-1)   # [B, D]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                     # [B, L, D]

        # Skip connection (D: aprendido, no decay)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)  # [B, L, D]

        states = (
            torch.stack(states_list, dim=1)    # [B, L, D, N]
            if return_states else None
        )

        return y, states


# ─────────────────────────────────────────────────────────────────────────────
# MAMBA LAYER = SSM BLOCK + GATED FFN
# ─────────────────────────────────────────────────────────────────────────────

class MambaLayer(nn.Module):
    """
    Un bloque completo del encoder Mamba.

    Estructura (pre-norm, como GPT-2 / LLaMA):
      ┌─────────────────────────────────────────┐
      │  residual = x                           │
      │  h = RMSNorm(x)                         │
      │  h = in_proj(h) → (x_in, z)            │
      │  x_in = Conv1d(x_in) [causal]           │
      │  x_in = SiLU(x_in)                     │
      │  y, states = SelectiveSSM(x_in)         │
      │  y = y ⊙ SiLU(z)    [gating]           │
      │  y = out_proj(y)                        │
      │  x = residual + dropout(y)              │
      │  ─────────────────────────────────────  │
      │  residual = x                           │
      │  h = RMSNorm(x)                         │
      │  x = residual + dropout(GatedFFN(h))    │
      └─────────────────────────────────────────┘

    El gating interno (z) es la segunda forma de selectividad:
    - El SSM produce y (información del pasado comprimida)
    - z es una proyección directa del input actual
    - y * SiLU(z) combina memoria y input actual adaptativamente
    """

    def __init__(self, config: StreamEncoderConfig) -> None:
        super().__init__()
        D       = config.hidden_dim
        D_inner = config.d_inner

        # ── Mamba SSM block ──────────────────────────────────────────────────
        self.norm1 = RMSNorm(D, eps=config.rms_eps)

        # Proyección entrada: D → 2·D_inner (x_in y z)
        self.in_proj = nn.Linear(D, D_inner * 2, bias=config.bias)

        # Convolución causal depthwise para contexto local
        self.conv1d = nn.Conv1d(
            in_channels  = D_inner,
            out_channels = D_inner,
            kernel_size  = config.d_conv,
            padding      = config.d_conv - 1,   # → causal al recortar en L
            groups       = D_inner,              # depthwise
            bias         = True,
        )

        self.ssm = SelectiveSSM(config)

        # Proyección salida: D_inner → D
        self.out_proj = nn.Linear(D_inner, D, bias=config.bias)

        # ── Gated FFN ────────────────────────────────────────────────────────
        self.norm2 = RMSNorm(D, eps=config.rms_eps)
        self.ffn   = GatedFFN(D, ffn_mult=config.ffn_mult, bias=config.bias)

        self.drop  = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:             [B, L, D]
            return_states: si True, retorna trayectoria de estados SSM

        Returns:
            output: [B, L, D]
            states: [B, L, D_inner, N] o None
        """
        B, L, _ = x.shape

        # ── SSM block ────────────────────────────────────────────────────────
        residual = x
        h = self.norm1(x)                           # [B, L, D]

        # Proyectar a espacio expandido: (x_in, z)
        xz   = self.in_proj(h)                      # [B, L, 2·D_inner]
        x_in, z = xz.chunk(2, dim=-1)              # cada [B, L, D_inner]

        # Convolución causal local (agrega contexto de los últimos d_conv tokens)
        x_in = x_in.transpose(1, 2)                # [B, D_inner, L]
        x_in = self.conv1d(x_in)[:, :, :L]         # recortar → causal
        x_in = x_in.transpose(1, 2)                # [B, L, D_inner]
        x_in = F.silu(x_in)

        # Selective SSM
        y, states = self.ssm(x_in, return_states=return_states)   # [B, L, D_inner]

        # Gating con z (selectividad sobre qué parte del pasado usar)
        y = y * F.silu(z)

        # Proyectar de vuelta a D
        y = self.out_proj(y)                        # [B, L, D]
        x = residual + self.drop(y)

        # ── Gated FFN ────────────────────────────────────────────────────────
        residual = x
        x = residual + self.drop(self.ffn(self.norm2(x)))

        return x, states
