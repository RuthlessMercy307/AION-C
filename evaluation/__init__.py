"""
evaluation/ — Eval prompts canónicos + métricas para training del 1.1B
========================================================================

Componentes:
  eval_prompts.py — 50 prompts fijos (10 × 5 dominios) con respuestas
                    esperadas para exact_match y BLEU
  metrics.py      — bleu_score, exact_match, generation_quality_score
"""

from .eval_prompts import (
    EvalPrompt, EVAL_PROMPTS,
    prompts_by_domain, prompts_for_domain,
)
from .metrics import (
    tokenize, ngrams, bleu_score, multi_reference_bleu,
    exact_match, contains_any,
    GenerationQualityResult, generation_quality_score,
)

__all__ = [
    "EvalPrompt",
    "EVAL_PROMPTS",
    "prompts_by_domain",
    "prompts_for_domain",
    "tokenize",
    "ngrams",
    "bleu_score",
    "multi_reference_bleu",
    "exact_match",
    "contains_any",
    "GenerationQualityResult",
    "generation_quality_score",
]
