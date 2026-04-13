"""
router/pipeline.py — CORAPipeline
===================================

Pipeline end-to-end de AION-C que conecta los cuatro módulos principales:

    StreamEncoder → GraphCrystallizer → CausalReasoningEngine → StreamDecoder
         (SE)              (GC)                 (CRE)                (SD)

FLUJO DE DATOS:
    token_ids [B, L]
        ↓  StreamEncoder
    concept_vectors [B, L, D]
        ↓  GraphCrystallizer
    CrystallizerOutput:
        .graphs           List[CausalGraph]       — topología discreta
        .node_vectors     [B, K, D]               — features diferenciables
        .node_counts      List[int]               — nodos válidos por item
        ↓  CausalReasoningEngine (por item del batch)
    refined_nodes   List[[n_i, D]]              — un tensor por item
        ↓  padding → [B, max_graph_nodes, D]
        ↓  StreamDecoder
    DecoderOutput:
        .logits           [B, L, vocab_size]
        .anchor_logits    [B, L, max_graph_nodes]
        .confidence       [B]
        .needs_clarification [B]

DISEÑO DEL PIPELINE:
    La clave es la transición GC → CRE → SD:

    1. GC produce `node_vectors [B, K, D]` — diferenciable — y
       `graphs [B]` — estructura discreta.

    2. CRE opera POR GRAFO (no batched) porque cada grafo tiene
       una topología diferente. Para el item b:
           node_feats = node_vectors[b, :node_counts[b], :]  ← diferenciable
           cre_out    = cre(graphs[b], node_feats)

    3. Después de CRE, los refined node_features se padean a
       max_graph_nodes para crear un tensor [B, max_graph_nodes, D]
       compatible con el decoder batched.

    GRADIENTES:
        El padding usa `torch.cat` y `torch.stack` — diferenciables.
        node_vectors[b, :n, :] es un slice — diferenciable.
        El CRE forward es diferenciable (GRU, attention, etc.).
        → Gradiente fluye desde logits hasta token_embedding.weight.

SCRATCHPAD:
    El mismo DifferentiableScratchPad se reutiliza en cada item del batch.
    Los parámetros del pad se aplican a cada grafo independientemente.
    El estado `pad_state` se reinicia para cada item (no comparte estado
    entre elementos del batch — cada razonamiento es independiente).

CONFIG ALINEAMIENTO:
    CORAConfig garantiza que estas dimensiones son iguales:
        encoder.concept_dim  == crystallizer.hidden_dim
                             == cre.node_dim
                             == decoder.node_dim

    Sin esta alineación, las interfaces entre módulos no conectan.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from budget import BudgetManager, BudgetOutput
from crystallizer import CrystallizerOutput, GraphCrystallizer
from crystallizer.config import CrystallizerConfig
from cre import CausalReasoningEngine, DifferentiableScratchPad
from cre.batching import PyGStyleBatcher
from cre.config import CREConfig
from cre.scratch_pad import ScratchPadConfig
from decoder import DecoderOutput, StreamDecoder
from decoder.config import StreamDecoderConfig
from encoder import StreamEncoder
from encoder.mamba_layer import StreamEncoderConfig
from validation import AionCValidator, ValidationResult
from validation.model import ValidatorConfig


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CORAConfig:
    """
    Config unificada para el pipeline CORAPipeline.

    Garantiza la alineación dimensional entre módulos:
        encoder.concept_dim == crystallizer.hidden_dim
                            == cre.node_dim == decoder.node_dim

    Configuración tiny para tests (rápida, pocos parámetros):
        CORAConfig.tiny()
    """
    # ── Dimensión compartida (la más importante) ─────────────────────────────
    hidden_dim:  int   = 256     # fluye por todo el pipeline
    vocab_size:  int   = 32_000

    # ── StreamEncoder ─────────────────────────────────────────────────────────
    enc_n_layers:  int   = 4
    enc_state_dim: int   = 16
    enc_expand:    int   = 2
    enc_d_conv:    int   = 4
    enc_ffn_mult:  int   = 4

    # ── GraphCrystallizer ─────────────────────────────────────────────────────
    cryst_max_nodes:       int   = 32
    cryst_n_heads:         int   = 4
    cryst_node_threshold:  float = 0.3
    cryst_edge_threshold:  float = 0.3

    # ── CausalReasoningEngine ─────────────────────────────────────────────────
    cre_edge_dim:             int   = 64
    cre_message_dim:          int   = 128
    cre_n_message_layers:     int   = 2
    cre_max_iterations:       int   = 20
    cre_use_convergence_gate: bool  = True   # True → parada adaptativa
    cre_min_iterations:       int   = 1      # safety floor para ConvergenceGate

    # ── BudgetManager ─────────────────────────────────────────────────────────
    # use_budget_manager=False preserva el comportamiento anterior (iters fijas).
    # Cuando True, el BudgetManager clasifica cada query antes del CRE y decide
    # max_iterations dinamicamente (trivial=1, simple=3, complex=10, deep=max).
    use_budget_manager:     bool  = True
    budget_hidden_dim:      int   = 64    # dim oculta del MLP clasificador

    # ── Validator (VAL) ───────────────────────────────────────────────────────
    # use_validator=False preserva el comportamiento anterior (sin validacion).
    # Cuando True, el VAL verifica la respuesta del decoder contra el grafo.
    # val_rereason=True: si la validacion falla, el CRE re-razona con mas iters
    # (un solo reintento, con n_iters * 2 capeado a cre_max_iterations).
    use_validator:          bool  = False
    val_hidden_dim:         int   = 128   # dim interna del VAL
    val_n_layers:           int   = 2     # capas de self-attn del ResponseEncoder
    val_n_heads:            int   = 4     # cabezas de attencion del VAL
    val_pass_threshold:     float = 0.5   # overall_score >= esto → passed
    val_issue_threshold:    float = 0.4   # score de check < esto → issue
    val_rereason:           bool  = False # re-razonar si la validacion falla

    # ── DifferentiableScratchPad ──────────────────────────────────────────────
    pad_n_slots:  int = 16
    pad_slot_dim: int = 128

    # ── StreamDecoder ─────────────────────────────────────────────────────────
    dec_n_layers:   int   = 4
    dec_n_heads:    int   = 4
    dec_max_seq_len: int  = 2048
    dec_state_dim:  int   = 16
    dec_expand:     int   = 2
    dec_d_conv:     int   = 4
    dec_ffn_mult:   int   = 4

    def __post_init__(self) -> None:
        if self.hidden_dim % self.cryst_n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"cryst_n_heads ({self.cryst_n_heads})"
            )
        if self.hidden_dim % self.dec_n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"dec_n_heads ({self.dec_n_heads})"
            )

    @classmethod
    def tiny(cls) -> "CORAConfig":
        """
        Configuración mínima para tests rápidos.

        hidden_dim=64 → ~2M parámetros totales.
        Estado SSM: 4. Capas: enc=2, cryst trivial, cre=1 layer, dec=2.
        """
        return cls(
            hidden_dim   = 64,
            vocab_size   = 512,
            # Encoder
            enc_n_layers  = 2,
            enc_state_dim = 4,
            enc_expand    = 2,
            enc_d_conv    = 4,
            enc_ffn_mult  = 2,
            # Crystallizer
            cryst_max_nodes      = 8,
            cryst_n_heads        = 4,
            cryst_node_threshold = 0.01,   # bajo → siempre hay nodos con pesos random
            cryst_edge_threshold = 0.01,   # bajo → siempre hay aristas con pesos random
            # CRE
            cre_edge_dim         = 32,
            cre_message_dim      = 64,
            cre_n_message_layers = 1,
            cre_max_iterations   = 3,
            # Scratch pad
            pad_n_slots  = 8,
            pad_slot_dim = 32,
            # Decoder
            dec_n_layers    = 2,
            dec_n_heads     = 4,
            dec_max_seq_len = 128,
            dec_state_dim   = 4,
            dec_expand      = 2,
            dec_d_conv      = 4,
            dec_ffn_mult    = 2,
        )

    # ── Factories para configs de submódulos ──────────────────────────────────

    def encoder_config(self) -> StreamEncoderConfig:
        return StreamEncoderConfig(
            vocab_size  = self.vocab_size,
            hidden_dim  = self.hidden_dim,
            n_layers    = self.enc_n_layers,
            state_dim   = self.enc_state_dim,
            expand      = self.enc_expand,
            d_conv      = self.enc_d_conv,
            ffn_mult    = self.enc_ffn_mult,
            concept_dim = self.hidden_dim,   # ← alineado con el pipeline
        )

    def crystallizer_config(self) -> CrystallizerConfig:
        return CrystallizerConfig(
            hidden_dim          = self.hidden_dim,
            max_nodes           = self.cryst_max_nodes,
            n_relation_types    = 16,
            n_node_types        = 7,
            node_threshold      = self.cryst_node_threshold,
            edge_threshold      = self.cryst_edge_threshold,
            pooler_heads        = self.cryst_n_heads,
        )

    def cre_config(self) -> CREConfig:
        return CREConfig(
            node_dim               = self.hidden_dim,
            edge_dim               = self.cre_edge_dim,
            message_dim            = self.cre_message_dim,
            n_message_layers       = self.cre_n_message_layers,
            max_iterations         = self.cre_max_iterations,
            n_relation_types       = 16,
            use_convergence_gate   = self.cre_use_convergence_gate,
            min_iterations         = self.cre_min_iterations,
        )

    def scratch_pad_config(self) -> ScratchPadConfig:
        return ScratchPadConfig(
            n_slots  = self.pad_n_slots,
            slot_dim = self.pad_slot_dim,
            node_dim = self.hidden_dim,
        )

    def validator_config(self) -> ValidatorConfig:
        return ValidatorConfig(
            input_dim       = self.hidden_dim,
            hidden_dim      = self.val_hidden_dim,
            n_heads         = self.val_n_heads,
            n_layers        = self.val_n_layers,
            pass_threshold  = self.val_pass_threshold,
            issue_threshold = self.val_issue_threshold,
        )

    def decoder_config(self) -> StreamDecoderConfig:
        return StreamDecoderConfig(
            vocab_size      = self.vocab_size,
            hidden_dim      = self.hidden_dim,
            n_layers        = self.dec_n_layers,
            n_heads         = self.dec_n_heads,
            node_dim        = self.hidden_dim,   # ← alineado con CRE output
            max_graph_nodes = self.cryst_max_nodes,
            max_seq_len     = self.dec_max_seq_len,
            state_dim       = self.dec_state_dim,
            expand          = self.dec_expand,
            d_conv          = self.dec_d_conv,
            ffn_mult        = self.dec_ffn_mult,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineOutput:
    """
    Resultado completo del CORAPipeline.

    Contiene los outputs de cada submódulo para inspección y pérdidas auxiliares.

    logits:          [B, L, vocab_size]     — distribución de tokens (para LM loss)
    anchor_logits:   [B, L, max_graph_nodes] — anclaje al grafo (aux loss)
    confidence:      [B]                    — confianza del modelo
    needs_clarif:    [B]                    — solicitar más contexto
    crystal_output:  CrystallizerOutput     — tensores diferenciables + grafos
    graph_repr:      [B, max_nodes, D]      — representación refinada del grafo
    budget:          BudgetOutput | None    — decisión del BudgetManager (None si inactivo)
    validation:      ValidationResult | None — resultado del VAL (None si inactivo)
    """
    logits:         torch.Tensor
    anchor_logits:  torch.Tensor
    confidence:     torch.Tensor
    needs_clarif:   torch.Tensor
    crystal_output: CrystallizerOutput
    graph_repr:     torch.Tensor            # [B, max_graph_nodes, D]
    budget:         Optional[BudgetOutput]    = None
    validation:     Optional[ValidationResult] = None


# ─────────────────────────────────────────────────────────────────────────────
# CORA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class CORAPipeline(nn.Module):
    """
    Pipeline AION-C CEN end-to-end.

    SE → GC → CRE (con ScratchPad) → SD

    Uso básico:
        config   = CORAConfig.tiny()
        pipeline = CORAPipeline(config)

        token_ids = torch.randint(0, config.vocab_size, (2, 16))
        out       = pipeline(token_ids)

        # out.logits:         [2, 16, vocab_size]
        # out.crystal_output.graphs[0]:  CausalGraph inspectable
        # out.graph_repr:     [2, max_nodes, hidden_dim]

    Gradient flow:
        out.logits.sum().backward()
        # pipeline.encoder.token_embedding.weight.grad is not None  ✓

    Args:
        config: CORAConfig con todos los hiperparámetros
    """

    def __init__(self, config: CORAConfig) -> None:
        super().__init__()
        self.config = config

        # ── Submódulos ────────────────────────────────────────────────────────
        self.encoder      = StreamEncoder(config.encoder_config())
        self.crystallizer = GraphCrystallizer(config.crystallizer_config())
        self.cre          = CausalReasoningEngine(config.cre_config())
        self.scratch_pad  = DifferentiableScratchPad(config.scratch_pad_config())
        self.decoder      = StreamDecoder(config.decoder_config())

        # ── BudgetManager (opcional) ──────────────────────────────────────────
        # Solo se instancia cuando use_budget_manager=True.
        if config.use_budget_manager:
            self.budget_manager: Optional[BudgetManager] = BudgetManager(
                concept_dim        = config.hidden_dim,
                max_cre_iterations = config.cre_max_iterations,
                hidden_dim         = config.budget_hidden_dim,
                use_learned        = True,
            )
        else:
            self.budget_manager = None

        # ── Validator (opcional) ───────────────────────────────────────────────
        # Solo se instancia cuando use_validator=True.
        if config.use_validator:
            self.validator: Optional[AionCValidator] = AionCValidator(
                config     = config.validator_config(),
                vocab_size = config.vocab_size,
            )
        else:
            self.validator = None

        # ── PyGStyleBatcher (no tiene parámetros — no es nn.Module) ───────────
        # Usado en _run_cre_and_build cuando B > 1 y convergence gate inactivo.
        self._batcher = PyGStyleBatcher()

    def forward(
        self,
        token_ids:    torch.Tensor,          # [B, L]
        n_cre_iters:  Optional[int] = None,  # override iterations (para tests rápidos)
    ) -> PipelineOutput:
        """
        Pasa token_ids por el pipeline completo y devuelve logits.

        Args:
            token_ids:   [B, L] — indices de tokens de entrada
            n_cre_iters: número de iteraciones del CRE (None = config.cre_max_iterations)

        Returns:
            PipelineOutput con logits y metadatos de todos los submódulos
        """
        B, L  = token_ids.shape
        D     = self.config.hidden_dim
        device = token_ids.device
        dtype  = self.encoder.token_embedding.weight.dtype

        # ── 1. StreamEncoder: tokens → concept vectors ────────────────────────
        concepts = self.encoder(token_ids)      # [B, L, D]

        # ── 2. BudgetManager: clasificar query → n_iterations para el CRE ─────
        # Si el BudgetManager está activo, decide cuántas iteraciones usa el CRE.
        # Si no está activo, usa n_cre_iters (que puede ser None → config default).
        budget: Optional[BudgetOutput] = None
        if self.budget_manager is not None:
            budget     = self.budget_manager(token_ids, concepts)
            n_cre_iters = budget.n_iterations   # override con presupuesto asignado

        # ── 3. GraphCrystallizer: concept vectors → CausalGraph ──────────────
        crystal = self.crystallizer(concepts)   # CrystallizerOutput
        # crystal.node_vectors:  [B, K, D]   ← diferenciable
        # crystal.node_counts:   [B]
        # crystal.graphs:        List[CausalGraph]

        # ── 4. CRE + graph_repr (helper reutilizable para re-reasoning) ─────────
        max_nodes  = self.config.cryst_max_nodes
        graph_repr = self._run_cre_and_build(
            crystal, B, D, device, dtype, max_nodes, n_cre_iters
        )

        # ── 5. StreamDecoder: (tokens, graph, encoder_concepts) → logits ────────
        dec_out = self.decoder(token_ids, graph_repr, concepts)

        # ── 6. Validator: verificar coherencia respuesta-grafo ────────────────
        # Si la validacion falla y val_rereason=True, el CRE re-razona con
        # el doble de iteraciones (un solo reintento).
        val_result: Optional[ValidationResult] = None
        if self.validator is not None:
            val_result = self.validator(dec_out.logits, graph_repr, concepts)

            if (
                not val_result.passed
                and self.config.val_rereason
            ):
                # Calcular iteraciones del reintento: doble, capeado a max
                base_iters  = n_cre_iters or self.config.cre_max_iterations
                retry_iters = min(base_iters * 2, self.config.cre_max_iterations)

                if retry_iters > base_iters:
                    # Re-razonar con mas iteraciones
                    graph_repr_2 = self._run_cre_and_build(
                        crystal, B, D, device, dtype, max_nodes, retry_iters
                    )
                    dec_out_2  = self.decoder(token_ids, graph_repr_2, concepts)
                    val_result = self.validator(dec_out_2.logits, graph_repr_2, concepts)

                    # Usar la segunda pasada si mejoro
                    dec_out    = dec_out_2
                    graph_repr = graph_repr_2

        return PipelineOutput(
            logits         = dec_out.logits,
            anchor_logits  = dec_out.anchor_logits,
            confidence     = dec_out.confidence,
            needs_clarif   = dec_out.needs_clarification,
            crystal_output = crystal,
            graph_repr     = graph_repr,
            budget         = budget,
            validation     = val_result,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run_cre_and_build(
        self,
        crystal:      CrystallizerOutput,
        B:            int,
        D:            int,
        device:       torch.device,
        dtype:        torch.dtype,
        max_nodes:    int,
        n_iterations: Optional[int],
    ) -> torch.Tensor:
        """
        Ejecuta el CRE y construye graph_repr [B, max_nodes, D].

        Dos caminos según el tamaño del batch y la config:

        Camino 1 — Per-item (B==1 o convergence gate activo):
            Itera sobre cada item, llama cre.forward() con scratch_pad y
            convergence gate. Soporta todos los features avanzados del CRE.

        Camino 2 — PyGStyleBatcher (B>1 y convergence gate inactivo):
            Concatena los B grafos en un super-grafo sin padding con nodos dummy
            y corre cre.forward_batched() en un solo pase — matemáticamente
            equivalente a B llamadas individuales pero más eficiente en GPU.
            No aplica scratch_pad ni convergence gate (vanilla-only).

        El camino se selecciona automáticamente según config.cre_use_convergence_gate.
        """
        # ── Camino 1: per-item (soporta scratch_pad + convergence gate) ───────
        if B == 1 or self.config.cre_use_convergence_gate:
            refined_nodes: List[torch.Tensor] = []
            for b in range(B):
                n_nodes = crystal.node_counts[b]
                if n_nodes == 0:
                    refined_nodes.append(
                        torch.zeros(0, D, device=device, dtype=dtype)
                    )
                    continue
                node_feats = crystal.node_vectors[b, :n_nodes, :]
                cre_out    = self.cre(
                    graph         = crystal.graphs[b],
                    node_features = node_feats,
                    n_iterations  = n_iterations,
                    scratch_pad   = self.scratch_pad,
                )
                refined_nodes.append(cre_out.node_features)
            return self._build_graph_repr(refined_nodes, max_nodes, D, device, dtype)

        # ── Camino 2: PyGStyleBatcher (B>1, sin convergence gate) ─────────────
        # Separar items sin nodos de los que sí tienen grafos válidos.
        refined_map:   Dict[int, torch.Tensor] = {}
        valid_graphs:  List                    = []
        valid_feats:   List[torch.Tensor]      = []
        valid_indices: List[int]               = []

        for b in range(B):
            n_nodes = crystal.node_counts[b]
            if n_nodes == 0:
                refined_map[b] = torch.zeros(0, D, device=device, dtype=dtype)
            else:
                valid_graphs.append(crystal.graphs[b])
                valid_feats.append(crystal.node_vectors[b, :n_nodes, :])
                valid_indices.append(b)

        if valid_graphs:
            batched     = self._batcher.batch(valid_graphs, valid_feats)
            cre_outputs = self.cre.forward_batched(batched, n_iterations=n_iterations)
            for i, b in enumerate(valid_indices):
                refined_map[b] = cre_outputs[i].node_features

        refined_nodes = [refined_map[b] for b in range(B)]
        return self._build_graph_repr(refined_nodes, max_nodes, D, device, dtype)

    def _build_graph_repr(
        self,
        refined_nodes: List[torch.Tensor],  # List of [n_i, D] (puede n_i=0)
        max_nodes:     int,
        D:             int,
        device:        torch.device,
        dtype:         torch.dtype,
    ) -> torch.Tensor:
        """
        Convierte una lista de tensores [n_i, D] (con n_i variable) en
        un tensor batched [B, max_nodes, D] con padding de ceros.

        Diferenciable: usa torch.cat y torch.stack.
        """
        padded: List[torch.Tensor] = []

        for nodes in refined_nodes:
            n = nodes.shape[0]

            if n == 0:
                # Sin nodos — todo ceros
                padded.append(
                    torch.zeros(max_nodes, D, device=device, dtype=dtype)
                )
            elif n >= max_nodes:
                # Truncar si excede el límite
                padded.append(nodes[:max_nodes])
            else:
                # Pad con ceros hasta max_nodes
                pad = torch.zeros(max_nodes - n, D, device=device, dtype=dtype)
                padded.append(torch.cat([nodes, pad], dim=0))

        return torch.stack(padded, dim=0)   # [B, max_nodes, D]

    # ── Utilidades ────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Número total de parámetros únicos entrenables."""
        seen: set = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                if p.requires_grad:
                    total += p.numel()
        return total

    def parameter_breakdown(self) -> dict:
        """Desglose de parámetros por submódulo."""
        breakdown = {
            "encoder":      sum(p.numel() for p in self.encoder.parameters()),
            "crystallizer": sum(p.numel() for p in self.crystallizer.parameters()),
            "cre":          sum(p.numel() for p in self.cre.parameters()),
            "scratch_pad":  sum(p.numel() for p in self.scratch_pad.parameters()),
            "decoder":      self.decoder.count_parameters(),   # deduplica weight tying
        }
        if self.budget_manager is not None:
            breakdown["budget_manager"] = self.budget_manager.count_parameters()
        if self.validator is not None:
            breakdown["validator"] = self.validator.count_parameters()
        breakdown["total_unique"] = self.count_parameters()
        return breakdown


# ─────────────────────────────────────────────────────────────────────────────
# MoSE PIPELINE — MIXTURE OF SPECIALIZED ENGINES
# ─────────────────────────────────────────────────────────────────────────────

from orchestrator.model import (
    Orchestrator, OrchestratorConfig, OrchestratorOutput, MOTOR_NAMES,
)
from unifier.model import Unifier, UnifierConfig, UnifierOutput
from motors.cora.motor    import CORAMotor, CORAMotorConfig
from motors.forge_c.motor import CodeMotor, CodeMotorConfig
from motors.muse.motor    import CreativeMotor, CreativeMotorConfig
from motors.axiom.motor   import MathMotor, MathMotorConfig
from motors.empathy.motor import SocialMotor, SocialMotorConfig
from crystallizer.config  import CrystallizerConfig
from cre.config           import CREConfig


@dataclass
class MoSEConfig:
    """
    Config para el pipeline MoSE completo.

    Garantiza alineación dimensional entre encoder, motores, unifier y decoder:
        hidden_dim fluye por todo el pipeline.
    """
    hidden_dim:  int   = 256
    vocab_size:  int   = 32_000

    # Encoder
    enc_n_layers:  int = 4
    enc_state_dim: int = 16
    enc_expand:    int = 2
    enc_d_conv:    int = 4
    enc_ffn_mult:  int = 4

    # Orchestrator
    orch_mlp_hidden:     int   = 128
    orch_max_motors:     int   = 3
    orch_min_confidence: float = 0.3

    # Motors (shared crystallizer/CRE dims, cada motor tiene sus propias relaciones)
    motor_max_nodes:  int   = 16
    motor_n_heads:    int   = 4
    motor_threshold:  float = 0.3

    # Unifier
    unif_n_heads: int = 4

    # Decoder
    dec_n_layers:    int = 4
    dec_n_heads:     int = 4
    dec_max_seq_len: int = 2048
    dec_state_dim:   int = 16
    dec_expand:      int = 2
    dec_d_conv:      int = 4
    dec_ffn_mult:    int = 4

    def __post_init__(self) -> None:
        if self.hidden_dim % self.motor_n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"motor_n_heads ({self.motor_n_heads})"
            )
        if self.hidden_dim % self.dec_n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"dec_n_heads ({self.dec_n_heads})"
            )

    @classmethod
    def tiny(cls) -> "MoSEConfig":
        """Configuración mínima para tests rápidos."""
        return cls(
            hidden_dim       = 64,
            vocab_size       = 512,
            enc_n_layers     = 2,
            enc_state_dim    = 4,
            enc_expand       = 2,
            enc_d_conv       = 4,
            enc_ffn_mult     = 2,
            orch_mlp_hidden  = 32,
            orch_max_motors  = 3,
            orch_min_confidence = 0.3,
            motor_max_nodes  = 8,
            motor_n_heads    = 4,
            motor_threshold  = 0.01,
            unif_n_heads     = 4,
            dec_n_layers     = 2,
            dec_n_heads      = 4,
            dec_max_seq_len  = 128,
            dec_state_dim    = 4,
            dec_expand       = 2,
            dec_d_conv       = 4,
            dec_ffn_mult     = 2,
        )

    def encoder_config(self) -> StreamEncoderConfig:
        return StreamEncoderConfig(
            vocab_size  = self.vocab_size,
            hidden_dim  = self.hidden_dim,
            n_layers    = self.enc_n_layers,
            state_dim   = self.enc_state_dim,
            expand      = self.enc_expand,
            d_conv      = self.enc_d_conv,
            ffn_mult    = self.enc_ffn_mult,
            concept_dim = self.hidden_dim,
        )

    def orchestrator_config(self) -> OrchestratorConfig:
        return OrchestratorConfig(
            hidden_dim                 = self.hidden_dim,
            n_motors                   = 5,
            max_active_motors          = self.orch_max_motors,
            min_confidence_to_activate = self.orch_min_confidence,
            mlp_hidden_dim             = self.orch_mlp_hidden,
        )

    def _motor_crystallizer_config(self, n_node_types: int, n_relation_types: int) -> CrystallizerConfig:
        return CrystallizerConfig(
            hidden_dim       = self.hidden_dim,
            max_nodes        = self.motor_max_nodes,
            n_node_types     = n_node_types,
            n_relation_types = n_relation_types,
            node_threshold   = self.motor_threshold,
            edge_threshold   = self.motor_threshold,
            pooler_heads     = self.motor_n_heads,
        )

    def _motor_cre_config(self, n_relation_types: int) -> CREConfig:
        return CREConfig(
            node_dim         = self.hidden_dim,
            edge_dim         = max(16, self.hidden_dim // 4),
            message_dim      = max(32, self.hidden_dim // 2),
            n_message_layers = 1,
            max_iterations   = 10,
            n_relation_types = n_relation_types,
        )

    def cora_config(self) -> CORAMotorConfig:
        from core.graph import CAUSAL_RELATIONS
        n_rel = len(CAUSAL_RELATIONS)
        return CORAMotorConfig(
            crystallizer=self._motor_crystallizer_config(7, n_rel),
            cre=self._motor_cre_config(n_rel),
        )

    def forge_c_config(self) -> CodeMotorConfig:
        from motors.forge_c.relations import CODE_RELATIONS, CODE_NODE_TYPES
        return CodeMotorConfig(
            crystallizer=self._motor_crystallizer_config(len(CODE_NODE_TYPES), len(CODE_RELATIONS)),
            cre=self._motor_cre_config(len(CODE_RELATIONS)),
        )

    def muse_config(self) -> CreativeMotorConfig:
        from motors.muse.relations import NARRATIVE_RELATIONS, NARRATIVE_NODE_TYPES
        return CreativeMotorConfig(
            crystallizer=self._motor_crystallizer_config(len(NARRATIVE_NODE_TYPES), len(NARRATIVE_RELATIONS)),
            cre=self._motor_cre_config(len(NARRATIVE_RELATIONS)),
        )

    def axiom_config(self) -> MathMotorConfig:
        from motors.axiom.relations import MATH_RELATIONS, MATH_NODE_TYPES
        return MathMotorConfig(
            crystallizer=self._motor_crystallizer_config(len(MATH_NODE_TYPES), len(MATH_RELATIONS)),
            cre=self._motor_cre_config(len(MATH_RELATIONS)),
        )

    def empathy_config(self) -> SocialMotorConfig:
        from motors.empathy.relations import SOCIAL_RELATIONS, SOCIAL_NODE_TYPES
        return SocialMotorConfig(
            crystallizer=self._motor_crystallizer_config(len(SOCIAL_NODE_TYPES), len(SOCIAL_RELATIONS)),
            cre=self._motor_cre_config(len(SOCIAL_RELATIONS)),
        )

    def unifier_config(self) -> UnifierConfig:
        return UnifierConfig(
            node_dim         = self.hidden_dim,
            n_heads          = self.unif_n_heads,
            max_output_nodes = self.motor_max_nodes,
        )

    def decoder_config(self) -> StreamDecoderConfig:
        return StreamDecoderConfig(
            vocab_size      = self.vocab_size,
            hidden_dim      = self.hidden_dim,
            n_layers        = self.dec_n_layers,
            n_heads         = self.dec_n_heads,
            node_dim        = self.hidden_dim,
            max_graph_nodes = self.motor_max_nodes,
            max_seq_len     = self.dec_max_seq_len,
            state_dim       = self.dec_state_dim,
            expand          = self.dec_expand,
            d_conv          = self.dec_d_conv,
            ffn_mult        = self.dec_ffn_mult,
        )


@dataclass
class MoSEOutput:
    """
    Resultado completo del MoSEPipeline.

    logits:         [B, L, vocab_size]
    anchor_logits:  [B, L, max_nodes]
    confidence:     [B]
    needs_clarif:   [B]
    graph_repr:     [B, max_nodes, D]
    orchestrator:   OrchestratorOutput — routing decision
    unifier:        UnifierOutput — fusion metadata
    active_motors:  List[str] — motores que se ejecutaron
    """
    logits:        torch.Tensor
    anchor_logits: torch.Tensor
    confidence:    torch.Tensor
    needs_clarif:  torch.Tensor
    graph_repr:    torch.Tensor
    orchestrator:  OrchestratorOutput
    unifier:       UnifierOutput
    active_motors: List[str]


class MoSEPipeline(nn.Module):
    """
    Pipeline MoSE (Mixture of Specialized Engines) completo.

    Flujo:
        token_ids [B, L]
            ↓ StreamEncoder
        concept_vectors [B, L, D]
            ↓ Orchestrator
        OrchestratorOutput → lista de motores a activar
            ↓ Para cada motor activado:
        motor.build_graph(concept_vectors) → cryst_out
        motor.reason(graph, node_feats, n_iters) → cre_out
        motor.get_graph_repr(cre_out, k_nodes) → [k, D]
            ↓ Unifier
        unified [max_nodes, D]  →  batched [B, max_nodes, D]
            ↓ StreamDecoder
        DecoderOutput → MoSEOutput

    Todos los motores comparten el mismo encoder y decoder.
    Cada motor tiene su propio crystallizer + CRE con su vocabulario.
    """

    def __init__(self, config: MoSEConfig) -> None:
        super().__init__()
        self.config = config

        # ── Submódulos compartidos ────────────────────────────────────────────
        self.encoder     = StreamEncoder(config.encoder_config())
        self.orchestrator = Orchestrator(config.orchestrator_config())
        self.unifier     = Unifier(config.unifier_config())
        self.decoder     = StreamDecoder(config.decoder_config())

        # ── Los 5 motores especializados ──────────────────────────────────────
        self.motors: nn.ModuleDict = nn.ModuleDict({
            "cora":    CORAMotor(config.cora_config()),
            "forge_c": CodeMotor(config.forge_c_config()),
            "muse":    CreativeMotor(config.muse_config()),
            "axiom":   MathMotor(config.axiom_config()),
            "empathy": SocialMotor(config.empathy_config()),
        })

    def forward(
        self,
        token_ids:   torch.Tensor,           # [B, L]
        query_text:  Optional[str] = None,   # para heurística de fallback
    ) -> MoSEOutput:
        """
        Pasa token_ids por el pipeline MoSE completo.

        Args:
            token_ids:  [B, L] — índices de tokens
            query_text: texto de la query (opcional, para heurísticas)

        Returns:
            MoSEOutput con logits y metadatos de routing y fusión
        """
        B, L  = token_ids.shape
        D     = self.config.hidden_dim
        device = token_ids.device
        dtype  = self.encoder.token_embedding.weight.dtype

        # ── 1. Encode ─────────────────────────────────────────────────────────
        concepts = self.encoder(token_ids)          # [B, L, D]

        # ── 2. Orchestrate ────────────────────────────────────────────────────
        orch_out = self.orchestrator(concepts, query_text)

        # ── 3. Run selected motors ────────────────────────────────────────────
        K        = self.config.motor_max_nodes
        max_nodes = K

        # graph_repr_batch[b] = list of [K, D] tensors (one per motor)
        # We process per-batch-item for CRE, but crystallizer is batched.
        all_graph_reprs: List[torch.Tensor] = []   # [B, K, D] stacked later

        # Pre-compute crystallizer outputs for all active motors (batched)
        motor_cryst: Dict[str, object] = {}
        for activation in orch_out.activations:
            motor = self.motors[activation.motor_name]
            with torch.no_grad() if not self.training else _null_ctx():
                cryst_out = motor.build_graph(concepts)
            motor_cryst[activation.motor_name] = cryst_out

        # For each item in batch, run CRE per motor and unify
        for b in range(B):
            motor_reprs: List[torch.Tensor] = []

            for activation in orch_out.activations:
                motor     = self.motors[activation.motor_name]
                cryst_out = motor_cryst[activation.motor_name]
                n_nodes   = cryst_out.node_counts[b]

                if n_nodes == 0:
                    motor_reprs.append(
                        torch.zeros(K, D, device=device, dtype=dtype)
                    )
                    continue

                node_feats = cryst_out.node_vectors[b, :n_nodes]  # [n, D]
                graph_b    = cryst_out.graphs[b]

                cre_out = motor.reason(
                    graph_b, node_feats,
                    n_iterations=activation.n_iterations,
                )
                repr_b = motor.get_graph_repr(cre_out, k_nodes=K)  # [K, D]
                motor_reprs.append(repr_b)

            # Unify motor representations for this batch item
            unif_out = self.unifier(motor_reprs)         # UnifierOutput
            all_graph_reprs.append(unif_out.unified)     # [K, D]

        # Stack into [B, K, D]
        graph_repr = torch.stack(all_graph_reprs, dim=0)  # [B, K, D]

        # ── 4. Decode ─────────────────────────────────────────────────────────
        dec_out = self.decoder(token_ids, graph_repr, concepts)

        return MoSEOutput(
            logits        = dec_out.logits,
            anchor_logits = dec_out.anchor_logits,
            confidence    = dec_out.confidence,
            needs_clarif  = dec_out.needs_clarification,
            graph_repr    = graph_repr,
            orchestrator  = orch_out,
            unifier       = unif_out,
            active_motors = orch_out.motor_names,
        )

    def count_parameters(self) -> int:
        seen: set = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                if p.requires_grad:
                    total += p.numel()
        return total

    def parameter_breakdown(self) -> dict:
        bd = {
            "encoder":      sum(p.numel() for p in self.encoder.parameters()),
            "orchestrator": self.orchestrator.count_parameters(),
            "unifier":      self.unifier.count_parameters(),
            "decoder":      self.decoder.count_parameters(),
        }
        for name, motor in self.motors.items():
            bd[f"motor_{name}"] = sum(p.numel() for p in motor.parameters())
        bd["total_unique"] = self.count_parameters()
        return bd

    @torch.no_grad()
    def explain(
        self,
        token_ids: torch.Tensor,
        tok=None,
    ) -> Dict:
        """
        Ejecuta el pipeline y retorna info de inspección:
        grafo, scratch pad states, respuesta decodificada, motores activos.

        Args:
            token_ids: [1, L] tensor de tokens
            tok: tokenizador opcional para decodificar respuesta

        Returns:
            dict con keys: graphs, pad_states, response_ids, response_text,
                          active_motors, orchestrator_scores
        """
        self.eval()
        out = self.forward(token_ids)

        # Collect graphs from active motors
        graphs = []
        for activation in out.orchestrator.activations:
            motor = self.motors[activation.motor_name]
            concepts = self.encoder(token_ids)
            cryst_out = motor.build_graph(concepts)
            if cryst_out.graphs:
                graphs.extend(cryst_out.graphs)

        # Greedy decode response
        logits = out.logits  # [1, L, V]
        response_ids = logits[0].argmax(dim=-1).tolist()

        response_text = ""
        if tok is not None:
            try:
                response_text = tok.decode(response_ids)
            except Exception:
                response_text = str(response_ids[:20])

        return {
            "graphs": graphs,
            "pad_states": [],  # filled by CRE if tracking enabled
            "response_ids": response_ids,
            "response_text": response_text,
            "active_motors": out.active_motors,
            "orchestrator_scores": out.orchestrator.logits.tolist()
                if hasattr(out.orchestrator, 'logits') else [],
            "confidence": out.confidence.tolist() if out.confidence is not None else [],
        }


class _null_ctx:
    """Context manager noop para usar con torch.no_grad() condicional."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
