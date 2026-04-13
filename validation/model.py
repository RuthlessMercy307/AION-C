"""
validation/model.py — AionCValidator (VAL)
===========================================

Modelo encoder-only pequeno que verifica si la respuesta generada
es coherente con el CausalGraph producido por el CRE.

POSICION EN EL PIPELINE:
    StreamDecoder → logits [B, L, V]
                        ↓
                 AionCValidator
                     ↓         ↓
              ValidationResult  (si falla + rereason_on_fail)
                                    ↓
                            CRE re-razona (mas iters)
                            Decoder regenera
                            Validator re-verifica

POR QUE:
    El decoder puede generar texto que suena fluido pero:
    - Contradice una relacion causal del grafo (CAUSES vs PREVENTS)
    - Menciona entidades que no estan en el grafo (alucinacion)
    - No responde a lo que se pregunto (incompletitud)
    - Es inconsistente internamente respecto al grafo
    El VAL actua como red de seguridad antes de entregar la respuesta.

LOS 4 CHECKS:
    faithfulness:   texto alineado con nodos del grafo (no inventa hechos)
    consistency:    texto no contradice relaciones del grafo
    completeness:   texto responde lo que se pregunto (cubre el input)
    hallucination:  texto no afirma cosas fuera del grafo + input

ARQUITECTURA:
    1. ResponseEncoder: proyecta logits → [B, L, H], n_layers de self-attn, mean pool
    2. GraphEncoder:    proyecta graph_repr → [B, K, H], mean pool
    3. InputEncoder:    proyecta input_concepts → [B, L, H], mean pool (si disponible)
    4. CrossAttn R→G:   response (Q) atiende a graph nodes (K, V) → cross_out [B, H]
    5. 4 CheckHeads:    cada uno con inputs semanticamente distintos

INPUT AL VALIDATOR:
    response_logits: [B, L, vocab_size]  — distribución del decoder
    graph_repr:      [B, max_nodes, D]   — features refinadas del CRE
    input_concepts:  [B, L, D] opcional  — output del encoder (contexto de la pregunta)
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidatorConfig:
    """
    Configuracion del AionCValidator.

    input_dim:       dimension del pipeline (hidden_dim de CORAConfig)
    hidden_dim:      dimension interna del VAL (puede ser distinta del pipeline)
    n_heads:         cabezas de atencion
    n_layers:        capas del ResponseEncoder
    pass_threshold:  overall_score >= esto → passed=True
    issue_threshold: score de un check < esto → emite ValidationIssue
    norm_eps:        epsilon de LayerNorm
    """
    input_dim:       int   = 256
    hidden_dim:      int   = 128
    n_heads:         int   = 4
    n_layers:        int   = 2
    pass_threshold:  float = 0.5
    issue_threshold: float = 0.4
    norm_eps:        float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    """
    Un problema detectado por el validator en un check especifico.

    check:    nombre del check ("faithfulness" | "consistency" |
                                "completeness" | "hallucination")
    score:    score del check (0-1, menor = peor)
    severity: "warning" si 0.3 <= score < issue_threshold,
              "error"   si score < 0.3
    """
    check:    str
    score:    float
    severity: str   # "warning" | "error"

    def __repr__(self) -> str:
        return f"ValidationIssue({self.check}, score={self.score:.3f}, {self.severity})"


@dataclass
class ValidationResult:
    """
    Resultado completo del AionCValidator.

    passed:         True si overall_score >= pass_threshold
    overall_score:  media geometrica de los 4 scores (0-1)
    faithfulness:   alineacion texto-grafo (1=fiel al grafo)
    consistency:    coherencia interna texto-relaciones (1=consistente)
    completeness:   cobertura de la pregunta (1=responde todo)
    hallucination:  ausencia de alucinaciones (1=sin alucinaciones)
    issues:         lista de ValidationIssue para checks que fallaron
    """
    passed:         bool
    overall_score:  float
    faithfulness:   float
    consistency:    float
    completeness:   float
    hallucination:  float
    issues:         List[ValidationIssue]

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ValidationResult({status}, overall={self.overall_score:.3f}, "
            f"faith={self.faithfulness:.3f}, cons={self.consistency:.3f}, "
            f"comp={self.completeness:.3f}, hall={self.hallucination:.3f}, "
            f"issues={len(self.issues)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SUB-MODULOS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

class _FFNBlock(nn.Module):
    """FFN de 2 capas con GELU y LayerNorm pre-norm."""

    def __init__(self, hidden_dim: int, ffn_mult: int = 2, norm_eps: float = 1e-6) -> None:
        super().__init__()
        ffn_dim = hidden_dim * ffn_mult
        self.norm = nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.fc1  = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.fc2  = nn.Linear(ffn_dim, hidden_dim, bias=False)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc2(F.gelu(self.fc1(x)))
        return residual + x


class _SelfAttnBlock(nn.Module):
    """Bloque de self-attention con pre-LayerNorm y residual."""

    def __init__(self, hidden_dim: int, n_heads: int, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = n_heads,
            batch_first = True,
            bias        = False,
        )
        self.ffn = _FFNBlock(hidden_dim, norm_eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        return self.ffn(x)


class _CheckHead(nn.Module):
    """
    Cabeza binaria para un check especifico.

    Linear(in_dim → hidden_dim) → GELU → Linear(hidden_dim → 1) → sigmoid
    Score en [0, 1]; 1 = sin problemas.
    """

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B] scores in (0, 1)."""
        return torch.sigmoid(self.net(x).squeeze(-1))


# ─────────────────────────────────────────────────────────────────────────────
# AION-C VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class AionCValidator(nn.Module):
    """
    Verificador encoder-only para el pipeline AION-C.

    Realiza 4 checks sobre la respuesta generada vs el CausalGraph:

        faithfulness  — la respuesta se adhiere a los conceptos del grafo
        consistency   — la respuesta no contradice las relaciones del grafo
        completeness  — la respuesta cubre lo que se pregunto
        hallucination — la respuesta no va mas alla del grafo + input

    Uso:
        config = ValidatorConfig(input_dim=64, hidden_dim=128, n_layers=2)
        val    = AionCValidator(config, vocab_size=512)

        result = val(response_logits, graph_repr, input_concepts)
        # result.passed:        True/False
        # result.overall_score: 0.0-1.0
        # result.issues:        lista de ValidationIssue

    Args:
        config:     ValidatorConfig con hiperparametros
        vocab_size: tamanio del vocabulario del modelo (para proyectar logits)
    """

    def __init__(self, config: ValidatorConfig, vocab_size: int) -> None:
        super().__init__()
        self.config     = config
        H               = config.hidden_dim
        D               = config.input_dim

        # ── Proyectores de entrada ────────────────────────────────────────────
        # Proyecta la distribucion de tokens (logits softmax) al espacio del VAL
        self.response_proj = nn.Linear(vocab_size, H, bias=False)

        # Proyecta graph_repr (dim D del pipeline) al espacio del VAL
        self.graph_proj    = nn.Linear(D, H, bias=False)

        # Proyecta input_concepts (dim D) al espacio del VAL
        self.input_proj    = nn.Linear(D, H, bias=False)

        # ── ResponseEncoder: n_layers de self-attention ───────────────────────
        # Codifica la secuencia de response antes de pooling
        self.response_encoder = nn.ModuleList([
            _SelfAttnBlock(H, config.n_heads, config.norm_eps)
            for _ in range(config.n_layers)
        ])
        self.response_norm = nn.LayerNorm(H, eps=config.norm_eps)

        # ── GraphEncoder: proyeccion + LayerNorm ──────────────────────────────
        self.graph_norm = nn.LayerNorm(H, eps=config.norm_eps)

        # ── Cross-attention: response (Q) → graph nodes (K, V) ───────────────
        # Mide cuanto del grafo esta representado en la respuesta
        self.cross_attn_norm = nn.LayerNorm(H, eps=config.norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = H,
            num_heads   = config.n_heads,
            batch_first = True,
            bias        = False,
        )

        # ── 4 CheckHeads (inputs semanticamente distintos) ────────────────────
        #
        # faithfulness:  cross_out → como response_pool cubre el grafo
        #                input: cross_out [H]
        self.head_faithfulness  = _CheckHead(H, H // 2)

        # consistency:   coherencia global entre texto y grafo
        #                input: cat(response_pool, graph_pool) [2H]
        self.head_consistency   = _CheckHead(H * 2, H // 2)

        # completeness:  cuanto cubre la respuesta el input (pregunta original)
        #                input: cat(response_pool, input_pool) [2H]
        #                Si no hay input_concepts, usa graph_pool como proxy
        self.head_completeness  = _CheckHead(H * 2, H // 2)

        # hallucination: cuanto de la respuesta NO puede ser explicado por grafo + input
        #                input: cat(response_pool - cross_out, response_pool) [2H]
        #                diferencia entre response y su reconstruccion via grafo
        self.head_hallucination = _CheckHead(H * 2, H // 2)

        # ── Init ──────────────────────────────────────────────────────────────
        nn.init.normal_(self.response_proj.weight, std=0.02)
        nn.init.normal_(self.graph_proj.weight,    std=0.02)
        nn.init.normal_(self.input_proj.weight,    std=0.02)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        response_logits: torch.Tensor,           # [B, L, vocab_size]
        graph_repr:      torch.Tensor,           # [B, max_nodes, D]
        input_concepts:  Optional[torch.Tensor] = None,  # [B, L, D]
    ) -> ValidationResult:
        """
        Verifica la coherencia de la respuesta con el grafo causal.

        Args:
            response_logits: [B, L, vocab_size] — logits del StreamDecoder
            graph_repr:      [B, max_nodes, D]  — graph repr del CRE (post-padding)
            input_concepts:  [B, L, D] opcional — concept vectors del encoder

        Returns:
            ValidationResult con scores [0, 1] para cada check y lista de issues
        """
        B = response_logits.shape[0]

        # ── 1. Codificar respuesta ────────────────────────────────────────────
        # Soft-embedding: softmax(logits) @ proj → [B, L, H]
        response_probs = F.softmax(response_logits, dim=-1)    # [B, L, V]
        response_enc   = self.response_proj(response_probs)    # [B, L, H]

        # n_layers de self-attention
        for block in self.response_encoder:
            response_enc = block(response_enc)                 # [B, L, H]
        response_enc = self.response_norm(response_enc)

        response_pool = response_enc.mean(dim=1)               # [B, H]

        # ── 2. Codificar grafo ────────────────────────────────────────────────
        graph_enc  = self.graph_proj(graph_repr)               # [B, K, H]
        graph_enc  = self.graph_norm(graph_enc)
        graph_pool = graph_enc.mean(dim=1)                     # [B, H]

        # ── 3. Codificar input (pregunta) ─────────────────────────────────────
        if input_concepts is not None:
            input_enc  = self.input_proj(input_concepts)       # [B, L, H]
            input_pool = input_enc.mean(dim=1)                 # [B, H]
        else:
            # Sin input_concepts, usar graph_pool como proxy de la pregunta
            input_pool = graph_pool

        # ── 4. Cross-attention: response (Q) atiende a graph nodes (K, V) ─────
        h = self.cross_attn_norm(response_pool.unsqueeze(1))   # [B, 1, H]
        cross_out, _ = self.cross_attn(
            query = h,
            key   = graph_enc,
            value = graph_enc,
        )                                                      # [B, 1, H]
        cross_out = cross_out.squeeze(1)                       # [B, H]

        # ── 5. Compute 4 check scores ─────────────────────────────────────────

        # faithfulness: cuanto de graph_repr aparece en la respuesta
        #   cross_out captura la proyeccion de la respuesta sobre el grafo
        faith_score = self.head_faithfulness(cross_out)        # [B]

        # consistency: coherencia global texto-grafo
        #   usa la representacion global de ambos
        cons_score  = self.head_consistency(
            torch.cat([response_pool, graph_pool], dim=-1)     # [B, 2H]
        )                                                      # [B]

        # completeness: la respuesta cubre la pregunta original
        comp_score  = self.head_completeness(
            torch.cat([response_pool, input_pool], dim=-1)     # [B, 2H]
        )                                                      # [B]

        # hallucination: parte de la respuesta NO explicada por el grafo
        #   residual = response_pool - cross_out → mide lo "ajeno al grafo"
        #   score alto = poca diferencia = poca alucinacion
        residual    = response_pool - cross_out                # [B, H]
        hall_score  = self.head_hallucination(
            torch.cat([residual, response_pool], dim=-1)       # [B, 2H]
        )                                                      # [B]

        # ── 6. Agregar a ValidationResult por item del batch ─────────────────
        # Para batch > 1, devolvemos el resultado del item mas critico (min score).
        # En produccion batch=1 siempre.
        faith_val = float(faith_score.min().item())
        cons_val  = float(cons_score.min().item())
        comp_val  = float(comp_score.min().item())
        hall_val  = float(hall_score.min().item())

        overall = _geometric_mean([faith_val, cons_val, comp_val, hall_val])

        passed = overall >= self.config.pass_threshold

        issues = _build_issues(
            scores = {
                "faithfulness":  faith_val,
                "consistency":   cons_val,
                "completeness":  comp_val,
                "hallucination": hall_val,
            },
            threshold = self.config.issue_threshold,
        )

        return ValidationResult(
            passed        = passed,
            overall_score = overall,
            faithfulness  = faith_val,
            consistency   = cons_val,
            completeness  = comp_val,
            hallucination = hall_val,
            issues        = issues,
        )

    # ── Utilidades ────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS PRIVADOS
# ─────────────────────────────────────────────────────────────────────────────

def _geometric_mean(scores: List[float]) -> float:
    """
    Media geometrica de una lista de scores.
    Penaliza mas que la media aritmetica cuando un score es muy bajo.
    Resultado en [0, 1].
    """
    if not scores:
        return 0.0
    product = 1.0
    for s in scores:
        product *= max(s, 1e-9)   # evitar log(0)
    return product ** (1.0 / len(scores))


def _build_issues(
    scores:    dict,
    threshold: float,
) -> List[ValidationIssue]:
    """
    Construye la lista de ValidationIssue para checks que fallaron.

    severity:
        score < 0.3       → "error"
        0.3 <= score < threshold → "warning"
    """
    issues: List[ValidationIssue] = []
    for check, score in scores.items():
        if score < threshold:
            severity = "error" if score < 0.3 else "warning"
            issues.append(ValidationIssue(check=check, score=score, severity=severity))
    return issues
