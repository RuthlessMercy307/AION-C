"""
cre/config.py — Configuración del Causal Reasoning Engine
==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CREConfig:
    """
    Hiperparámetros del CausalReasoningEngine.

    Configuración tiny para testing:
        node_dim=256, edge_dim=64, message_dim=128,
        n_message_layers=2, max_iterations=20

    WEIGHT SHARING:
        Los n_message_layers capas se comparten entre TODAS las iteraciones.
        Total de parámetros = parámetros de 2 capas (no de 2 × 20 = 40 capas).
        Total de FLOPs = 2 capas × max_iterations (más barato que 40 capas distintas).

    n_relation_types:
        Debe coincidir con len(CAUSAL_RELATIONS) = 16.
        Cada relación tiene su propia función de mensaje en CausalMessagePassingLayer.
    """

    # Dimensiones principales
    node_dim:      int = 256   # dimensión de los vectores de nodo
    edge_dim:      int = 64    # dimensión de los vectores de arista
    message_dim:   int = 128   # dimensión de los mensajes entre nodos

    # Arquitectura
    n_message_layers: int = 2  # capas de MP por iteración (compartidas)
    max_iterations:   int = 20  # iteraciones de refinamiento por defecto

    # Vocabulario (debe coincidir con core/graph.py)
    n_relation_types: int = 16  # len(CausalRelation)

    # Estabilidad numérica
    norm_eps: float = 1e-6

    # ── ConvergenceGate (parada dinámica) ────────────────────────────────────
    # use_convergence_gate=False preserva el comportamiento anterior (iteraciones fijas).
    # Activar para beneficiarse de parada adaptativa.
    use_convergence_gate:  bool  = True   # activar parada dinámica
    min_iterations:        int   = 1      # safety floor: nunca parar antes de esto

    # Thresholds del ConvergenceGate
    conv_delta_threshold:   float = 0.01   # ||Δh||/||h|| < esto → features estables
    conv_conf_threshold:    float = 0.75   # confianza media > esto → nodos seguros
    conv_weakness_threshold: float = 0.25  # ratio debilidades < esto → mayoría resuelta

    # ── WeaknessDetector ─────────────────────────────────────────────────────
    # Sólo se instancia cuando use_convergence_gate=True.
    weakness_conf_threshold: float = 0.35  # sigmoid < esto → low_confidence
    weakness_conf_hidden:    int   = 64    # dim oculta del confidence scorer

    # ── SparseMoE ────────────────────────────────────────────────────────────
    # use_moe=False preserva el comportamiento anterior (sin MoE).
    # El MoE actúa DESPUÉS del message passing por iteración.
    use_moe:               bool  = False  # activar SparseMoE post-MP
    moe_n_groups:          int   = 4      # grupos de expertos
    moe_experts_per_group: int   = 4      # expertos por grupo (total = n_groups × per_group)
    moe_active_experts:    int   = 2      # top-k activos por nodo
    moe_expert_hidden_mult: int  = 2      # multiplicador oculto dentro de cada experto
    moe_load_balance_weight: float = 0.01  # peso de la load balance loss

    def __post_init__(self) -> None:
        if self.n_message_layers < 1:
            raise ValueError(f"n_message_layers must be >= 1, got {self.n_message_layers}")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.message_dim < 1:
            raise ValueError(f"message_dim must be >= 1, got {self.message_dim}")
        if self.min_iterations < 1:
            raise ValueError(f"min_iterations must be >= 1, got {self.min_iterations}")
        if self.min_iterations > self.max_iterations:
            raise ValueError(
                f"min_iterations ({self.min_iterations}) > max_iterations ({self.max_iterations})"
            )
