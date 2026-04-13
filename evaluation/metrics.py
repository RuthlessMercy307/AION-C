"""
evaluation/metrics.py — Métricas de evaluación para training del 1.1B
=========================================================================

  - tokenize / ngrams      → primitivas
  - bleu_score             → BLEU-1+2 con smoothing y brevity penalty
  - multi_reference_bleu   → max BLEU sobre lista de referencias
  - exact_match            → substring case-insensitive
  - generation_quality_score → función principal: combina exact_match + BLEU
                                + routing_accuracy en un score 0..1

Sin dependencias externas (no nltk, no sacrebleu). Implementado a mano para
que el script de training pueda correr en Vast sin instalar nada extra.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Primitivas
# ─────────────────────────────────────────────────────────────────────────────


_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñü0-9]+|[^\sA-Za-zÁÉÍÓÚÑáéíóúñü0-9]")


def tokenize(text: str) -> List[str]:
    """Tokenización simple: palabras + signos como tokens separados."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Devuelve la lista de n-gramas como tuplas."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# BLEU
# ─────────────────────────────────────────────────────────────────────────────


def _modified_precision(
    reference_tokens: List[str],
    hypothesis_tokens: List[str],
    n: int,
) -> float:
    """
    BLEU modified n-gram precision con clipping.
    Devuelve 0..1 o 0 si no hay hypothesis n-grams.
    """
    hyp_ngrams = ngrams(hypothesis_tokens, n)
    if not hyp_ngrams:
        return 0.0
    ref_counts: Dict[Tuple[str, ...], int] = {}
    for ng in ngrams(reference_tokens, n):
        ref_counts[ng] = ref_counts.get(ng, 0) + 1
    matches = 0
    for ng in hyp_ngrams:
        if ref_counts.get(ng, 0) > 0:
            matches += 1
            ref_counts[ng] -= 1
    return matches / len(hyp_ngrams)


def bleu_score(reference: str, hypothesis: str, max_n: int = 2) -> float:
    """
    BLEU-1+2 con smoothing aditivo (epsilon=1e-3) y brevity penalty.

    Para sentencias cortas (~5-30 tokens) BLEU-1 y BLEU-2 son más estables
    que BLEU-4. Para evals de respuestas conversacionales no usamos BLEU-3+.

    Returns:
        score en [0, 1]
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    if max_n < 1:
        max_n = 1

    precisions = []
    for n in range(1, max_n + 1):
        p = _modified_precision(ref_tokens, hyp_tokens, n)
        # Smoothing aditivo para evitar 0 en geometric mean
        precisions.append(max(p, 1e-3))

    # Geometric mean de precisions
    log_precision_sum = sum(math.log(p) for p in precisions) / len(precisions)
    geo_mean = math.exp(log_precision_sum)

    # Brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_len / max(1, hyp_len))

    return float(geo_mean * bp)


def multi_reference_bleu(references: List[str], hypothesis: str, max_n: int = 2) -> float:
    """
    Devuelve el BLEU MÁXIMO sobre la lista de referencias. Si references
    está vacía, devuelve 0.0.
    """
    if not references:
        return 0.0
    return max(bleu_score(r, hypothesis, max_n=max_n) for r in references)


# ─────────────────────────────────────────────────────────────────────────────
# Exact match
# ─────────────────────────────────────────────────────────────────────────────


def exact_match(text: str, expected: str) -> bool:
    """
    Substring match case-insensitive. Devuelve True si `expected` está en `text`
    (ambos lowercased + stripped).
    """
    if not text or not expected:
        return False
    t = text.lower().strip()
    e = expected.lower().strip()
    return e in t


def contains_any(text: str, candidates: List[str]) -> bool:
    """True si alguno de los candidates está como substring en text."""
    if not text:
        return False
    t = text.lower()
    return any(c.lower() in t for c in candidates if c)


# ─────────────────────────────────────────────────────────────────────────────
# Generation quality score (la métrica principal del training)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GenerationQualityResult:
    """Resultado completo de generation_quality_score."""
    n_prompts:        int
    exact_match:      float                  # 0..1
    bleu:             float                  # 0..1
    routing_accuracy: float                  # 0..1
    combined:         float                  # 0..1 (weighted)
    per_domain:       Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_prompt:       List[Dict[str, Any]]   = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_prompts":         self.n_prompts,
            "exact_match":       round(self.exact_match, 4),
            "bleu":              round(self.bleu, 4),
            "routing_accuracy":  round(self.routing_accuracy, 4),
            "combined":          round(self.combined, 4),
            "per_domain":        self.per_domain,
            "per_prompt":        self.per_prompt,
        }


# Pesos del combined score (deben sumar 1.0)
WEIGHT_EXACT_MATCH      = 0.40
WEIGHT_BLEU             = 0.20
WEIGHT_ROUTING_ACCURACY = 0.40


def generation_quality_score(
    prompts: List[Any],
    generator_fn: Callable[[str], Tuple[str, Optional[str]]],
    max_n: int = 2,
) -> GenerationQualityResult:
    """
    Calcula generation_quality_score sobre una lista de EvalPrompts.

    Args:
        prompts:      lista de EvalPrompt (o objetos con los mismos campos)
        generator_fn: callable(query: str) -> (generated_text, top_motor)
                      top_motor puede ser None si no hay routing info
        max_n:        max n-gram para BLEU (default 2)

    Returns:
        GenerationQualityResult con todas las métricas y breakdown por dominio.

    Combined score formula:
        0.40 * exact_match + 0.20 * bleu + 0.40 * routing_accuracy

    El weighting baja BLEU porque es ruidoso en respuestas cortas y prioriza
    exact_match (substring requerido) y routing_accuracy (motor correcto).
    """
    n = len(prompts)
    if n == 0:
        return GenerationQualityResult(
            n_prompts=0, exact_match=0.0, bleu=0.0,
            routing_accuracy=0.0, combined=0.0,
        )

    sum_exact = 0
    sum_bleu  = 0.0
    sum_route = 0
    per_prompt: List[Dict[str, Any]] = []
    per_domain_counts: Dict[str, Dict[str, float]] = {}

    for p in prompts:
        try:
            generated, top_motor = generator_fn(p.query)
        except Exception as exc:
            generated, top_motor = "", None
            per_prompt.append({
                "query":     p.query,
                "domain":    p.domain,
                "generated": "",
                "error":     str(exc)[:120],
                "exact":     False,
                "bleu":      0.0,
                "routing":   False,
            })
            d = per_domain_counts.setdefault(p.domain, {"n": 0, "exact": 0, "bleu": 0.0, "routing": 0})
            d["n"] += 1
            continue

        em = exact_match(generated, p.expected_substring)
        b = multi_reference_bleu(getattr(p, "references", []) or [], generated, max_n=max_n)
        routing_correct = (top_motor == p.domain)

        sum_exact += int(em)
        sum_bleu  += b
        sum_route += int(routing_correct)

        d = per_domain_counts.setdefault(p.domain, {"n": 0, "exact": 0, "bleu": 0.0, "routing": 0})
        d["n"]       += 1
        d["exact"]   += int(em)
        d["bleu"]    += b
        d["routing"] += int(routing_correct)

        per_prompt.append({
            "query":     p.query[:80],
            "domain":    p.domain,
            "motor":     top_motor or "?",
            "generated": (generated or "")[:120],
            "exact":     em,
            "bleu":      round(b, 3),
            "routing":   routing_correct,
        })

    em_avg    = sum_exact / n
    bleu_avg  = sum_bleu / n
    route_avg = sum_route / n
    combined  = (
        WEIGHT_EXACT_MATCH * em_avg
        + WEIGHT_BLEU * bleu_avg
        + WEIGHT_ROUTING_ACCURACY * route_avg
    )

    # Normaliza per-domain
    per_domain: Dict[str, Dict[str, float]] = {}
    for dom, c in per_domain_counts.items():
        n_dom = max(1, c["n"])
        per_domain[dom] = {
            "n":       int(c["n"]),
            "exact":   round(c["exact"] / n_dom, 4),
            "bleu":    round(c["bleu"] / n_dom, 4),
            "routing": round(c["routing"] / n_dom, 4),
        }

    return GenerationQualityResult(
        n_prompts=n,
        exact_match=em_avg,
        bleu=bleu_avg,
        routing_accuracy=route_avg,
        combined=combined,
        per_domain=per_domain,
        per_prompt=per_prompt,
    )


__all__ = [
    "tokenize",
    "ngrams",
    "bleu_score",
    "multi_reference_bleu",
    "exact_match",
    "contains_any",
    "GenerationQualityResult",
    "generation_quality_score",
    "WEIGHT_EXACT_MATCH",
    "WEIGHT_BLEU",
    "WEIGHT_ROUTING_ACCURACY",
]
