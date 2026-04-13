"""
router/config_3_5b.py — MoSEScaleConfig: configuración de escala para MoSE
===========================================================================

Provee tres fábricas de configuración:
  - .tiny()        : ≈ 1.5 M parámetros (igual a MoSEConfig.tiny(), para tests)
  - .medium()      : ≈ 410 M parámetros (D=768, 6 capas encoder/decoder)
  - .production()  : ≈ 3.44 B parámetros (objetivo 3.5 B ±10%)

Métodos:
  - count_params()       : estimación analítica (sin instanciar el modelo)
  - count_params_real()  : cuenta real (instancia MoSEPipeline; solo para tiny)
  - estimate_vram_bf16() : estimación de VRAM en GB para bf16

Fórmulas verificadas contra los módulos reales con ratio ≈ 1.000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE RELACIONES (de los módulos del proyecto)
# ─────────────────────────────────────────────────────────────────────────────

# Extraídas de core/graph.py y motors/*/relations.py
_CORA_RELATIONS   = 16   # len(CAUSAL_RELATIONS)
_CORA_NODE_TYPES  = 7    # len(NODE_TYPES)
_FORGE_RELATIONS  = 12   # len(CODE_RELATIONS)
_FORGE_NODE_TYPES = 8    # len(CODE_NODE_TYPES)
_AXIOM_RELATIONS  = 10   # len(MATH_RELATIONS)
_AXIOM_NODE_TYPES = 8    # len(MATH_NODE_TYPES)
_MUSE_RELATIONS   = 10   # len(NARRATIVE_RELATIONS)
_MUSE_NODE_TYPES  = 8    # len(NARRATIVE_NODE_TYPES)
_EMPATHY_RELATIONS = 10  # len(SOCIAL_RELATIONS)
_EMPATHY_NODE_TYPES = 8  # len(SOCIAL_NODE_TYPES)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: fórmulas analíticas de parámetros
# ─────────────────────────────────────────────────────────────────────────────


def _mamba_layer_params(
    D: int,
    state_dim: int,
    expand: int = 2,
    d_conv: int = 4,
    ffn_mult: int = 4,
) -> int:
    """
    Parámetros exactos de un MambaLayer (SSM block + GatedFFN).

    Fórmula verificada con ratio 1.000 contra módulos reales.

    Estructura:
      RMSNorm(D)                         [D]
      Linear(D, 2·Di, bias=False)        [2·D·Di]
      Conv1d(Di, Di, C, groups=Di, bias) [Di·C + Di]
      SelectiveSSM:
        x_proj Linear(Di, R+2N, bias=F)  [Di·(R+2N)]
        dt_proj Linear(R, Di, bias=T)    [R·Di + Di]
        A_log Parameter                  [Di·N]
        D Parameter                      [Di]
      Linear(Di, D, bias=False)          [Di·D]
      RMSNorm(D)                         [D]
      GatedFFN(D, ffn_mult):
        w_gate, w_up, w_down             [3·ffn_mult·D²]

    Args:
        D:         hidden_dim
        state_dim: N (SSM state dimension)
        expand:    E (D_inner = expand × D)
        d_conv:    kernel size of causal conv1d
        ffn_mult:  GatedFFN inner multiplier
    """
    Di = expand * D
    R  = max(1, D // 16)     # dt_rank
    N  = state_dim

    return (
        D                       # norm1 (RMSNorm weight)
        + 2 * D * Di            # in_proj (no bias)
        + Di * d_conv + Di      # conv1d (depthwise + bias)
        + Di * (R + 2 * N)      # x_proj (no bias)
        + R * Di + Di           # dt_proj (with bias)
        + Di * N                # A_log
        + Di                    # D parameter
        + Di * D                # out_proj (no bias)
        + D                     # norm2 (RMSNorm weight)
        + 3 * ffn_mult * D * D  # GatedFFN: w_gate + w_up + w_down (no bias)
    )


def _hybrid_decoder_layer_params(
    D: int,
    state_dim: int,
    expand: int = 2,
    d_conv: int = 4,
    ffn_mult: int = 4,
    n_heads: int = 4,
    node_dim: Optional[int] = None,
) -> int:
    """
    Parámetros de un HybridDecoderLayer.

    Estructura:
      MambaLayer(D)
      LayerNorm(D)
      MultiheadAttention(D, n_heads, bias=False)  ← cross-attn al grafo
      [Linear(node_dim, D) si node_dim != D]      ← graph_proj
      LayerNorm(D)
      MultiheadAttention(D, n_heads, bias=False)  ← cross-attn al encoder
      LayerNorm(D)
      GatedFFN(D, ffn_mult, bias=False)

    Las 3 LayerNorm tienen weight+bias → 2·D cada una.
    Los MultiheadAttention con bias=False: 4·D² (in_proj + out_proj).
    """
    if node_dim is None:
        node_dim = D

    mamba   = _mamba_layer_params(D, state_dim, expand, d_conv, ffn_mult)
    ln      = 2 * D          # LayerNorm (weight + bias)
    mha     = 4 * D * D      # MultiheadAttention, bias=False: 3D²+D²
    g_proj  = 0 if node_dim == D else node_dim * D   # graph_proj (no bias)
    extra_ffn = 3 * ffn_mult * D * D   # GatedFFN (no bias)

    return (
        mamba
        + ln + mha + g_proj    # cross-attn graph + norm
        + ln + mha             # cross-attn encoder + norm
        + ln + extra_ffn       # extra GatedFFN + norm
    )


def _cre_message_layer_params(
    D: int,
    n_relations: int,
) -> int:
    """
    Parámetros de un CausalMessagePassingLayer.

    Estructura por relación:
      Linear(2D+E, M)  +  Linear(M, M)   (message_fn, with bias)

    Más componentes compartidos:
      AttentiveAggregator: Linear(M+D, M//2) + Linear(M//2, 1, no bias) + LayerNorm(M)
      ManualGRUCell(M, D): weight_ih + weight_hh + bias_ih + bias_hh
      _EdgeUpdater: Linear(2D+E, E) + Linear(E, E) + LayerNorm(E)
      node_norm: LayerNorm(D)
    """
    E = max(16, D // 4)   # edge_dim
    M = max(32, D // 2)   # message_dim
    R = n_relations

    # message functions (R relaciones)
    per_fn = (2 * D + E) * M + M + M * M + M   # Linear(2D+E, M, bias) + Linear(M, M, bias)
    msg_fns = R * per_fn

    # AttentiveAggregator
    aggregator = (
        (M + D) * (M // 2) + (M // 2)   # Linear(M+D, M//2, bias)
        + M // 2                          # Linear(M//2, 1, no bias)
        + 2 * M                           # LayerNorm(M)
    )

    # ManualGRUCell(input=M, hidden=D)
    gru = 3 * D * M + 3 * D * D + 3 * D + 3 * D   # weight_ih, weight_hh, bias_ih, bias_hh

    # _EdgeUpdater
    edge_updater = (
        (2 * D + E) * E + E    # Linear(2D+E, E, bias)
        + E * E + E             # Linear(E, E, bias)
        + 2 * E                 # LayerNorm(E)
    )

    # node_norm
    node_norm = 2 * D

    return msg_fns + aggregator + gru + edge_updater + node_norm


def _cre_params(
    D: int,
    n_relations: int,
    n_message_layers: int = 1,
    use_convergence_gate: bool = True,
) -> int:
    """
    Parámetros de un CausalReasoningEngine.

    Incluye:
      - n_message_layers capas compartidas (WEIGHT SHARING: n_layers parámetros, no n_layers×max_iterations)
      - edge_type_embedding: Embedding(n_relations, edge_dim)
      - edge_feat_projector: Linear(edge_dim+2, edge_dim, no bias)
      - WeaknessDetector (si use_convergence_gate=True)
        Linear(D, 64, bias) + Linear(64, 1, bias)
      - ConvergenceGate: sin parámetros propios
    """
    E = max(16, D // 4)

    layers   = n_message_layers * _cre_message_layer_params(D, n_relations)
    emb      = n_relations * E         # edge_type_embedding
    proj     = (E + 2) * E             # edge_feat_projector (no bias)

    weakness = 0
    if use_convergence_gate:
        C = 64   # weakness_conf_hidden (default)
        weakness = D * C + C + C + 1   # Linear(D, C, bias) + Linear(C, 1, bias)

    return layers + emb + proj + weakness


def _crystallizer_params(
    D: int,
    n_node_types: int,
    n_relation_types: int,
    rel_hidden: int = 64,
    conf_hidden: int = 64,
) -> int:
    """
    Parámetros de un GraphCrystallizer.

    Componentes:
      NodeDetector:
        node_scorer:       Linear(D,D,bias) + Linear(D,1,no_bias)
        type_classifier:   Linear(D, n_node_types, bias)
        confidence_head:   Linear(D, C, bias) + Linear(C, 1, no_bias)

      CrossAttentionPooler:
        q/k/v/out_proj:    4 × Linear(D, D, no_bias)
        LayerNorm(D)

      AsymmetricRelationScorer:
        source_proj:       Linear(D, R×H, no_bias)
        target_proj:       Linear(D, R×H, no_bias)
        refiner:           Linear(R, 2R, bias) + Linear(2R, R, bias)
    """
    T = n_node_types
    R = n_relation_types
    H = rel_hidden
    C = conf_hidden

    # NodeDetector
    node_scorer  = D * D + D + D          # Linear(D,D,bias) + Linear(D,1,no_bias)
    type_clf     = D * T + T              # Linear(D, T, bias)
    conf_head    = D * C + C + C          # Linear(D,C,bias) + Linear(C,1,no_bias)
    node_det     = node_scorer + type_clf + conf_head

    # CrossAttentionPooler: 4 linears (no bias) + LayerNorm
    pooler = 4 * D * D + 2 * D

    # AsymmetricRelationScorer
    src_proj  = D * R * H                       # no bias
    tgt_proj  = D * R * H                       # no bias
    refiner   = R * (2 * R) + (2 * R) + (2 * R) * R + R   # Linear(R,2R,bias) + Linear(2R,R,bias)
    rel_scr   = src_proj + tgt_proj + refiner

    return node_det + pooler + rel_scr


def _encoder_params(
    D: int,
    n_layers: int,
    vocab_size: int,
    state_dim: int,
    expand: int = 2,
    d_conv: int = 4,
    ffn_mult: int = 4,
    concept_dim: Optional[int] = None,
) -> int:
    """Parámetros del StreamEncoder."""
    if concept_dim is None:
        concept_dim = D

    emb     = vocab_size * D                                # token_embedding
    layers  = n_layers * _mamba_layer_params(D, state_dim, expand, d_conv, ffn_mult)
    norm    = D                                             # RMSNorm (weight only)
    c_proj  = D * concept_dim                              # concept_projector (no bias)

    return emb + layers + norm + c_proj


def _decoder_params(
    D: int,
    n_layers: int,
    vocab_size: int,
    state_dim: int,
    expand: int = 2,
    d_conv: int = 4,
    ffn_mult: int = 4,
    n_heads: int = 4,
    max_seq_len: int = 2048,
    max_graph_nodes: int = 32,
    node_dim: Optional[int] = None,
) -> int:
    """
    Parámetros únicos del StreamDecoder (weight-tying: lm_head no cuenta).

    Componentes:
      token_embedding + pos_embedding
      n_layers × HybridDecoderLayer
      final_norm (LayerNorm)
      anchor_head (con bias)
      meta_head: proj (D+node_dim→D, bias) + confidence_head + clarification_head
    """
    if node_dim is None:
        node_dim = D

    emb     = vocab_size * D                 # token_embedding (lm_head es tied)
    pos_emb = max_seq_len * D                # pos_embedding
    layers  = n_layers * _hybrid_decoder_layer_params(
        D, state_dim, expand, d_conv, ffn_mult, n_heads, node_dim
    )
    final_norm   = 2 * D                     # LayerNorm (weight + bias)
    anchor_head  = D * max_graph_nodes + max_graph_nodes  # with bias

    # meta_head:  proj Linear(D+node_dim, D, bias) + 2 heads Linear(D, 1, bias)
    meta_proj    = (D + node_dim) * D + D
    meta_heads   = 2 * (D + 1)
    meta_head    = meta_proj + meta_heads

    return emb + pos_emb + layers + final_norm + anchor_head + meta_head


def _orchestrator_params(D: int, mlp_hidden: int, n_motors: int = 5) -> int:
    """
    Parámetros del Orchestrator (MLP clasificador con LayerNorm).

    Estructura:
      Linear(D, H)     [with bias]
      LayerNorm(H)     [weight + bias]
      Linear(H, H//2)  [with bias]
      Linear(H//2, N)  [with bias]
    """
    H = mlp_hidden
    N = n_motors
    return (
        D * H + H           # Linear(D, H, bias)
        + 2 * H             # LayerNorm(H)
        + H * (H // 2) + (H // 2)   # Linear(H, H//2, bias)
        + (H // 2) * N + N          # Linear(H//2, N, bias)
    )


def _unifier_params(D: int, n_heads: int = 4) -> int:
    """
    Parámetros del Unifier (cross-attention + MLP).

    cross_attn: MultiheadAttention con bias=True (default PyTorch):
      in_proj_weight [3D, D] + in_proj_bias [3D] + out_proj.weight [D,D] + out_proj.bias [D]
      = 3D² + 3D + D² + D = 4D² + 4D

    fusion_mlp:
      LayerNorm(D) [2D] + Linear(D, 2D, bias) [2D²+2D]
      + Linear(2D, D, bias) [2D²+D] + LayerNorm(D) [2D]
      = 4D² + 7D

    input_norm: LayerNorm(D) [2D]
    """
    return (
        4 * D * D + 4 * D    # cross_attn (with bias)
        + 2 * D              # input_norm
        + 4 * D * D + 7 * D  # fusion_mlp
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG CLASS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MoSEScaleConfig:
    """
    Configuración de escala para el pipeline MoSE completo.

    Permite dimensiones heterogéneas por componente (encoder/decoder/motores).
    MoSEConfig usa un único hidden_dim para todo el pipeline; MoSEScaleConfig
    permite diseñar configuraciones de producción donde encoder, decoder y
    motores tienen dimensiones distintas.

    Uso:
        cfg = MoSEScaleConfig.production()
        total = cfg.count_params()      # ≈ 3.44 B
        vram  = cfg.estimate_vram_bf16()  # ≈ 14 GB inference

    count_params_real() solo es seguro para .tiny() (no OOM).
    """

    vocab_size: int = 32_000

    # ── Encoder ──────────────────────────────────────────────────────────────
    enc_dim:       int = 256
    enc_layers:    int = 4
    enc_state_dim: int = 16
    enc_expand:    int = 2
    enc_d_conv:    int = 4
    enc_ffn_mult:  int = 4

    # ── Decoder ──────────────────────────────────────────────────────────────
    dec_dim:            int = 256
    dec_layers:         int = 4
    dec_state_dim:      int = 16
    dec_expand:         int = 2
    dec_d_conv:         int = 4
    dec_ffn_mult:       int = 4
    dec_n_heads:        int = 4
    dec_max_seq_len:    int = 2048
    dec_max_graph_nodes: int = 32
    dec_node_dim:       int = 256   # node_dim para decoder (debe coincidir con unifier output)

    # ── Motores pequeños: CORA, AXIOM, MUSE, EMPATHY ─────────────────────────
    small_motor_dim:            int = 256
    small_motor_cre_msg_layers: int = 1

    # ── Motor FORGE-C ─────────────────────────────────────────────────────────
    forge_c_dim:            int = 256
    forge_c_cre_msg_layers: int = 1

    # ── Orchestrator ──────────────────────────────────────────────────────────
    orch_dim:        int = 256
    orch_mlp_hidden: int = 128

    # ── Unifier ───────────────────────────────────────────────────────────────
    unif_dim:    int = 256
    unif_n_heads: int = 4

    # ─────────────────────────────────────────────────────────────────────────
    # FACTORY METHODS
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def tiny(cls) -> "MoSEScaleConfig":
        """
        Configuración mínima para tests rápidos.

        Dimensiones iguales a MoSEConfig.tiny() → count_params_real() válido.
        Total: ≈ 1.52 M parámetros.
        """
        return cls(
            vocab_size            = 512,
            enc_dim               = 64,
            enc_layers            = 2,
            enc_state_dim         = 4,
            enc_expand            = 2,
            enc_d_conv            = 4,
            enc_ffn_mult          = 2,
            dec_dim               = 64,
            dec_layers            = 2,
            dec_state_dim         = 4,
            dec_expand            = 2,
            dec_d_conv            = 4,
            dec_ffn_mult          = 2,
            dec_n_heads           = 4,
            dec_max_seq_len       = 128,
            dec_max_graph_nodes   = 8,
            dec_node_dim          = 64,
            small_motor_dim            = 64,
            small_motor_cre_msg_layers = 1,
            forge_c_dim            = 64,
            forge_c_cre_msg_layers = 1,
            orch_dim               = 64,
            orch_mlp_hidden        = 32,
            unif_dim               = 64,
            unif_n_heads           = 4,
        )

    @classmethod
    def medium(cls) -> "MoSEScaleConfig":
        """
        Configuración intermedia.

        D=768 uniforme, 6 capas encoder/decoder, 2 CRE msg layers.
        Total: ≈ 410 M parámetros.
        """
        return cls(
            vocab_size            = 32_000,
            enc_dim               = 768,
            enc_layers            = 6,
            enc_state_dim         = 16,
            enc_expand            = 2,
            enc_d_conv            = 4,
            enc_ffn_mult          = 4,
            dec_dim               = 768,
            dec_layers            = 6,
            dec_state_dim         = 16,
            dec_expand            = 2,
            dec_d_conv            = 4,
            dec_ffn_mult          = 4,
            dec_n_heads           = 8,
            dec_max_seq_len       = 2048,
            dec_max_graph_nodes   = 32,
            dec_node_dim          = 768,
            small_motor_dim            = 768,
            small_motor_cre_msg_layers = 2,
            forge_c_dim            = 768,
            forge_c_cre_msg_layers = 2,
            orch_dim               = 768,
            orch_mlp_hidden        = 1024,
            unif_dim               = 768,
            unif_n_heads           = 8,
        )

    @classmethod
    def production(cls) -> "MoSEScaleConfig":
        """
        Configuración de producción objetivo ≈ 3.5 B parámetros.

        Breakdown aproximado:
          Encoder  (D=1024, 14L):          ≈  303 M
          Decoder  (D=1536, 32L):          ≈ 2951 M
          CORA     (D=768,  CRE-msg=2):    ≈   37 M
          AXIOM    (D=768,  CRE-msg=2):    ≈   27 M
          MUSE     (D=768,  CRE-msg=2):    ≈   27 M
          EMPATHY  (D=768,  CRE-msg=2):    ≈   27 M
          FORGE-C  (D=1024, CRE-msg=2):    ≈   53 M
          Orch     (D=1024, H=2048):       ≈    4 M
          Unifier  (D=1024):               ≈    8 M
          ─────────────────────────────────────────
          TOTAL:                           ≈ 3437 M ≈ 3.44 B

        Dentro del rango objetivo ±10%: [3.15 B, 3.85 B] ✓
        VRAM bf16 inferencia: ≈ 14 GB  (< 130 GB) ✓
        VRAM bf16 training:   ≈ 41 GB  (< 130 GB) ✓
        """
        return cls(
            vocab_size            = 32_000,
            enc_dim               = 1024,
            enc_layers            = 14,
            enc_state_dim         = 16,
            enc_expand            = 2,
            enc_d_conv            = 4,
            enc_ffn_mult          = 4,
            dec_dim               = 1536,
            dec_layers            = 32,
            dec_state_dim         = 16,
            dec_expand            = 2,
            dec_d_conv            = 4,
            dec_ffn_mult          = 4,
            dec_n_heads           = 16,
            dec_max_seq_len       = 2048,
            dec_max_graph_nodes   = 32,
            dec_node_dim          = 1536,
            small_motor_dim            = 768,
            small_motor_cre_msg_layers = 2,
            forge_c_dim            = 1024,
            forge_c_cre_msg_layers = 2,
            orch_dim               = 1024,
            orch_mlp_hidden        = 2048,
            unif_dim               = 1024,
            unif_n_heads           = 8,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PARAM COUNT
    # ─────────────────────────────────────────────────────────────────────────

    def count_params(self) -> int:
        """
        Cuenta analítica de parámetros usando fórmulas exactas verificadas.

        NO instancia ningún módulo — solo aritmética.
        Válida para las tres configuraciones (tiny, medium, production).
        """
        total = 0

        # ── Encoder ──────────────────────────────────────────────────────────
        total += _encoder_params(
            D          = self.enc_dim,
            n_layers   = self.enc_layers,
            vocab_size = self.vocab_size,
            state_dim  = self.enc_state_dim,
            expand     = self.enc_expand,
            d_conv     = self.enc_d_conv,
            ffn_mult   = self.enc_ffn_mult,
            concept_dim= self.enc_dim,   # concept_dim = hidden_dim en MoSEConfig
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        total += _decoder_params(
            D               = self.dec_dim,
            n_layers        = self.dec_layers,
            vocab_size      = self.vocab_size,
            state_dim       = self.dec_state_dim,
            expand          = self.dec_expand,
            d_conv          = self.dec_d_conv,
            ffn_mult        = self.dec_ffn_mult,
            n_heads         = self.dec_n_heads,
            max_seq_len     = self.dec_max_seq_len,
            max_graph_nodes = self.dec_max_graph_nodes,
            node_dim        = self.dec_node_dim,
        )

        # ── CORA motor (D, T=7, R=16) ─────────────────────────────────────────
        D_s = self.small_motor_dim
        total += _crystallizer_params(D_s, _CORA_NODE_TYPES,   _CORA_RELATIONS)
        total += _cre_params(D_s, _CORA_RELATIONS,   self.small_motor_cre_msg_layers)

        # ── AXIOM motor (D, T=8, R=10) ───────────────────────────────────────
        total += _crystallizer_params(D_s, _AXIOM_NODE_TYPES,  _AXIOM_RELATIONS)
        total += _cre_params(D_s, _AXIOM_RELATIONS,  self.small_motor_cre_msg_layers)

        # ── MUSE motor (D, T=8, R=10) ────────────────────────────────────────
        total += _crystallizer_params(D_s, _MUSE_NODE_TYPES,   _MUSE_RELATIONS)
        total += _cre_params(D_s, _MUSE_RELATIONS,   self.small_motor_cre_msg_layers)

        # ── EMPATHY motor (D, T=8, R=10) ─────────────────────────────────────
        total += _crystallizer_params(D_s, _EMPATHY_NODE_TYPES, _EMPATHY_RELATIONS)
        total += _cre_params(D_s, _EMPATHY_RELATIONS, self.small_motor_cre_msg_layers)

        # ── FORGE-C motor (D_fc, T=8, R=12) ──────────────────────────────────
        D_fc = self.forge_c_dim
        total += _crystallizer_params(D_fc, _FORGE_NODE_TYPES, _FORGE_RELATIONS)
        total += _cre_params(D_fc, _FORGE_RELATIONS, self.forge_c_cre_msg_layers)

        # ── Orchestrator ──────────────────────────────────────────────────────
        total += _orchestrator_params(
            D          = self.orch_dim,
            mlp_hidden = self.orch_mlp_hidden,
            n_motors   = 5,
        )

        # ── Unifier ───────────────────────────────────────────────────────────
        total += _unifier_params(D=self.unif_dim, n_heads=self.unif_n_heads)

        return total

    def count_params_real(self) -> int:
        """
        Instancia MoSEPipeline y cuenta parámetros reales.

        ADVERTENCIA: Solo seguro para tiny() (< 2 M params).
        Para medium() puede tardar; para production() dará OOM.

        Usa MoSEConfig.tiny() directamente para garantizar compatibilidad.
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        from router.pipeline import MoSEConfig, MoSEPipeline

        cfg   = MoSEConfig.tiny()
        model = MoSEPipeline(cfg)

        seen: set = set()
        total = 0
        for p in model.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                if p.requires_grad:
                    total += p.numel()
        return total

    # ─────────────────────────────────────────────────────────────────────────
    # VRAM ESTIMATE
    # ─────────────────────────────────────────────────────────────────────────

    def estimate_vram_bf16(self, training: bool = False) -> float:
        """
        Estimación de VRAM en GB para bf16.

        Inferencia (training=False):
          pesos bf16 + activaciones/KV-cache overhead ≈ 2× pesos
          → 2 × params × 2 bytes / 1e9

        Entrenamiento (training=True):
          pesos bf16 + gradientes bf16 + estados AdamW fp32 (m + v)
          → pesos(2) + grads(2) + adam_states(8) = 12 bytes/param
          → params × 12 / 1e9

        Returns:
            VRAM estimada en GB.
        """
        params = self.count_params()
        if not training:
            # Inference: weights (bf16) + activations/KV cache overhead (~2x)
            return (params * 2 * 2) / 1e9
        else:
            # Training: bf16 weights + bf16 grads + fp32 Adam (m+v)
            return (params * 12) / 1e9

    # ─────────────────────────────────────────────────────────────────────────
    # REPR
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Resumen compacto de la configuración y sus parámetros."""
        total = self.count_params()
        vram_inf  = self.estimate_vram_bf16(training=False)
        vram_train = self.estimate_vram_bf16(training=True)
        return (
            f"MoSEScaleConfig("
            f"enc_dim={self.enc_dim}, dec_dim={self.dec_dim}, "
            f"motor_dim={self.small_motor_dim}) | "
            f"params={total / 1e6:.1f}M | "
            f"vram_infer={vram_inf:.1f}GB | vram_train={vram_train:.1f}GB"
        )
