"""
tools/dataset_analyzer.py — DatasetQualityAnalyzer
====================================================

Analiza un dataset de CausalExamples ANTES de entrenar para evitar
desperdiciar horas de GPU con datos de baja calidad.

METRICAS:
    correctness:      fraccion de ejemplos que pasan verify_example() [>= 0.95 para A]
    diversity:        entropia de AnswerTypes y niveles (mayor = mas diverso)
    level_balance:    distribucion de niveles L1-L5 (cuanto se aleja de uniforme)
    relation_coverage: cuantas de las 16 CausalRelations aparecen en el dataset
    entity_spans:     fraccion de ejemplos con al menos un span valido != (-1,-1)

CALIFICACION FINAL (plan v3, seccion 10.4):
    A — overall >= 0.90   (dataset excelente, listo para GPU)
    B — overall >= 0.75   (dataset bueno, mejoras menores)
    C — overall >= 0.50   (dataset aceptable, mejoras necesarias)
    D — overall <  0.50   (dataset deficiente, no usar)

USO COMO MODULO:
    gen      = CausalGraphGenerator(seed=42)
    examples = gen.generate_batch(n=1000)
    analyzer = DatasetQualityAnalyzer()
    report   = analyzer.analyze(examples)
    print(report.summary())
    print(analyzer.recommend(report))

USO COMO SCRIPT:
    python -m tools.dataset_analyzer --n 5000
    python -m tools.dataset_analyzer --n 1000 --seed 0 --verbose
"""

from __future__ import annotations

import argparse
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.graph import CausalRelation
from synth.causal_graph_gen import (
    AnswerType,
    CausalExample,
    CausalGraphGenerator,
    verify_example,
)

# Todas las relaciones posibles (16 en total)
ALL_RELATIONS: List[str] = [r.value for r in CausalRelation]
N_ALL_RELATIONS: int     = len(ALL_RELATIONS)   # 16

# Todos los tipos de respuesta (7 en total)
ALL_ANSWER_TYPES: List[str] = [t.value for t in AnswerType]
N_ALL_ANSWER_TYPES: int     = len(ALL_ANSWER_TYPES)   # 7

# Todos los dominios conocidos (7 en total)
ALL_DOMAINS: List[str] = [
    "clima", "economia", "salud", "tecnologia",
    "fisica", "social", "medioambiente",
]
N_ALL_DOMAINS: int = len(ALL_DOMAINS)   # 7

# Niveles de complejidad
ALL_LEVELS: List[int] = [1, 2, 3, 4, 5]

# Umbrales de calificacion (plan v3, seccion 10.4)
GRADE_THRESHOLDS = {"A": 0.90, "B": 0.75, "C": 0.50}


# ─────────────────────────────────────────────────────────────────────────────
# METRICAS INDIVIDUALES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CorrectnessMetric:
    """Fraccion de ejemplos que pasan verify_example()."""
    n_total:  int
    n_passed: int
    score:    float   # n_passed / n_total

    @property
    def label(self) -> str:
        return f"{self.n_passed}/{self.n_total} ({self.score:.1%})"


@dataclass
class DiversityMetric:
    """Entropia normalizada de AnswerTypes, niveles y dominios."""
    answer_type_entropy: float  # [0, 1] — 1 = uniforme sobre 7 tipos
    level_entropy:       float  # [0, 1] — 1 = uniforme sobre 5 niveles
    domain_entropy:      float  # [0, 1] — 1 = uniforme sobre 7 dominios
    score:               float  # media de las 3 entropias normalizadas

    # Distribuciones absolutas para el reporte
    answer_type_counts: Dict[str, int] = field(default_factory=dict)
    level_counts:       Dict[int, int] = field(default_factory=dict)
    domain_counts:      Dict[str, int] = field(default_factory=dict)


@dataclass
class LevelBalanceMetric:
    """
    Que tan bien distribuidos estan los 5 niveles de complejidad.

    score = 1 - distancia_normalizada_a_distribución_uniforme
    Donde la distancia es la suma de diferencias absolutas entre
    la distribucion real y la ideal (1/5 por nivel).
    """
    level_fractions: Dict[int, float]   # nivel → fraccion real
    score:           float              # 0-1, 1 = perfectamente uniforme


@dataclass
class RelationCoverageMetric:
    """Cuantas de las 16 CausalRelations aparecen en el dataset."""
    n_covered:   int              # numero de relaciones con al menos 1 uso
    n_total:     int = 16
    score:       float = 0.0      # n_covered / n_total
    counts:      Dict[str, int] = field(default_factory=dict)   # relacion → n_apariciones
    missing:     List[str]      = field(default_factory=list)   # relaciones ausentes

    def __post_init__(self) -> None:
        self.score = self.n_covered / self.n_total if self.n_total > 0 else 0.0


@dataclass
class EntitySpansMetric:
    """Fraccion de ejemplos con al menos un span valido != (-1, -1)."""
    n_with_valid_spans: int
    n_total:            int
    score:              float   # n_with_valid_spans / n_total

    @property
    def label(self) -> str:
        return f"{self.n_with_valid_spans}/{self.n_total} ({self.score:.1%})"


# ─────────────────────────────────────────────────────────────────────────────
# REPORTE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    """
    Reporte de calidad de un dataset de CausalExamples.

    Contiene todas las metricas y la calificacion final A/B/C/D.
    """
    n_examples:        int
    correctness:       CorrectnessMetric
    diversity:         DiversityMetric
    level_balance:     LevelBalanceMetric
    relation_coverage: RelationCoverageMetric
    entity_spans:      EntitySpansMetric
    overall_score:     float    # media ponderada de todos los scores
    grade:             str      # "A" | "B" | "C" | "D"

    # Pesos de cada metrica en el overall_score
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "correctness":       0.35,
        "diversity":         0.20,
        "level_balance":     0.20,
        "relation_coverage": 0.15,
        "entity_spans":      0.10,
    })

    def summary(self) -> str:
        """
        Devuelve un resumen legible del reporte.
        """
        sep = "-" * 60
        lines = [
            "=" * 60,
            f"  DATASET QUALITY REPORT — {self.n_examples} ejemplos",
            "=" * 60,
            f"  Calificacion final:  {self.grade}  (overall = {self.overall_score:.3f})",
            sep,
            f"  Correctness:        {self.correctness.score:.3f}  "
            f"({self.correctness.label})",
            f"  Diversity:          {self.diversity.score:.3f}  "
            f"(answer_type H={self.diversity.answer_type_entropy:.3f}, "
            f"level H={self.diversity.level_entropy:.3f}, "
            f"domain H={self.diversity.domain_entropy:.3f})",
            f"  Level balance:      {self.level_balance.score:.3f}  "
            f"(dist={_format_level_dist(self.level_balance.level_fractions)})",
            f"  Relation coverage:  {self.relation_coverage.score:.3f}  "
            f"({self.relation_coverage.n_covered}/{self.relation_coverage.n_total} relaciones)",
            f"  Entity spans:       {self.entity_spans.score:.3f}  "
            f"({self.entity_spans.label})",
            sep,
        ]
        if self.relation_coverage.missing:
            lines.append(
                f"  Relaciones ausentes: {', '.join(self.relation_coverage.missing)}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class DatasetQualityAnalyzer:
    """
    Analiza la calidad de un dataset de CausalExamples.

    Ejecuta 5 metricas, las pondera y produce una calificacion A/B/C/D.

    Uso:
        analyzer = DatasetQualityAnalyzer()
        report   = analyzer.analyze(examples)
        print(report.summary())
        print(analyzer.recommend(report))

    Pesos de las metricas (configurables via constructor):
        correctness:       35%  — lo mas importante: datos correctos
        diversity:         20%  — variedad de tipos y dominios
        level_balance:     20%  — distribucion balanceada de dificultad
        relation_coverage: 15%  — uso de todas las relaciones causales
        entity_spans:      10%  — spans de entidades validos
    """

    # Umbrales para calificacion de metricas individuales (cuanto se considera "bueno")
    CORRECTNESS_TARGET       = 0.95   # >=95% correcto → perfecta correctness
    SPAN_TARGET              = 0.80   # >=80% con spans → perfecto
    RELATION_COVERAGE_TARGET = 1.00   # 16/16 → perfecto

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.weights = weights or {
            "correctness":       0.35,
            "diversity":         0.20,
            "level_balance":     0.20,
            "relation_coverage": 0.15,
            "entity_spans":      0.10,
        }
        # Normalizar pesos
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    # ── Metodo principal ──────────────────────────────────────────────────────

    def analyze(self, examples: List[CausalExample]) -> QualityReport:
        """
        Analiza una lista de CausalExamples y devuelve un QualityReport.

        Args:
            examples: lista de CausalExample a analizar

        Returns:
            QualityReport con todas las metricas y calificacion final
        """
        if not examples:
            return self._empty_report()

        n = len(examples)

        correctness       = self._compute_correctness(examples)
        diversity         = self._compute_diversity(examples)
        level_balance     = self._compute_level_balance(examples)
        relation_coverage = self._compute_relation_coverage(examples)
        entity_spans      = self._compute_entity_spans(examples)

        overall = (
            self.weights["correctness"]       * correctness.score +
            self.weights["diversity"]         * diversity.score +
            self.weights["level_balance"]     * level_balance.score +
            self.weights["relation_coverage"] * relation_coverage.score +
            self.weights["entity_spans"]      * entity_spans.score
        )
        overall = min(1.0, max(0.0, overall))
        grade   = _score_to_grade(overall)

        return QualityReport(
            n_examples        = n,
            correctness       = correctness,
            diversity         = diversity,
            level_balance     = level_balance,
            relation_coverage = relation_coverage,
            entity_spans      = entity_spans,
            overall_score     = overall,
            grade             = grade,
        )

    # ── Recomendaciones ───────────────────────────────────────────────────────

    def recommend(self, report: QualityReport) -> str:
        """
        Devuelve recomendaciones concretas para mejorar la calidad del dataset.

        Args:
            report: QualityReport previamente calculado con analyze()

        Returns:
            str con recomendaciones en lenguaje natural
        """
        lines: List[str] = [
            f"RECOMENDACIONES (grade={report.grade}, overall={report.overall_score:.3f})"
        ]

        if report.grade == "A":
            lines.append("  Dataset excelente. Listo para GPU.")
            return "\n".join(lines)

        # Correctness
        if report.correctness.score < 0.95:
            n_bad = report.n_examples - report.correctness.n_passed
            lines.append(
                f"  [CORRECTNESS {report.correctness.score:.1%}] "
                f"{n_bad} ejemplos fallan verify_example(). "
                "Regenerar con verify=True o revisar generadores de nivel alto."
            )

        # Diversity — answer types
        if report.diversity.answer_type_entropy < 0.7:
            dominant = _dominant_key(report.diversity.answer_type_counts)
            lines.append(
                f"  [DIVERSITY tipos {report.diversity.answer_type_entropy:.2f}] "
                f"Tipo '{dominant}' domina. "
                "Balancear AnswerTypes en la generacion (ajustar level_distribution)."
            )

        # Diversity — dominios
        if report.diversity.domain_entropy < 0.7:
            dominant = _dominant_key(report.diversity.domain_counts)
            lines.append(
                f"  [DIVERSITY dominios {report.diversity.domain_entropy:.2f}] "
                f"Dominio '{dominant}' domina. "
                "Pasar domain= especifico en generate() para forzar balance."
            )

        # Level balance
        if report.level_balance.score < 0.75:
            fracs = report.level_balance.level_fractions
            over  = [f"L{l}" for l, f in fracs.items() if f > 0.25]
            under = [f"L{l}" for l, f in fracs.items() if f < 0.10]
            if over:
                lines.append(
                    f"  [LEVEL BALANCE {report.level_balance.score:.2f}] "
                    f"Niveles sobre-representados: {', '.join(over)}. "
                    "Usar level_distribution={{1:0.2, 2:0.2, 3:0.2, 4:0.2, 5:0.2}}."
                )
            if under:
                lines.append(
                    f"  [LEVEL BALANCE] Niveles bajo-representados: {', '.join(under)}. "
                    "Generar ejemplos extra para esos niveles."
                )

        # Relation coverage
        if report.relation_coverage.score < 1.0:
            missing = report.relation_coverage.missing
            lines.append(
                f"  [RELATIONS {report.relation_coverage.n_covered}/16] "
                f"Relaciones sin usar: {', '.join(missing)}. "
                "Usar gen.generate(level=3) o gen.generate(level=5) para "
                "relaciones como CONTRADICTS, WEAKENS, ANALOGOUS_TO."
            )

        # Entity spans
        if report.entity_spans.score < 0.80:
            n_bad = report.n_examples - report.entity_spans.n_with_valid_spans
            lines.append(
                f"  [ENTITY SPANS {report.entity_spans.score:.1%}] "
                f"{n_bad} ejemplos sin spans validos. "
                "Revisar compute_entity_spans() o templates de problem_text."
            )

        if len(lines) == 1:
            lines.append("  Sin problemas criticos detectados.")

        return "\n".join(lines)

    # ── Computo de metricas ───────────────────────────────────────────────────

    @staticmethod
    def _compute_correctness(examples: List[CausalExample]) -> CorrectnessMetric:
        """
        Fraccion de ejemplos que pasan verify_example().
        """
        n_passed = sum(1 for ex in examples if verify_example(ex).passed)
        score    = n_passed / len(examples)
        return CorrectnessMetric(
            n_total  = len(examples),
            n_passed = n_passed,
            score    = score,
        )

    @staticmethod
    def _compute_diversity(examples: List[CausalExample]) -> DiversityMetric:
        """
        Entropia normalizada de AnswerTypes, niveles y dominios.

        La entropia normalizada es H / H_max, donde H_max = log2(n_categorias).
        Una distribucion perfectamente uniforme tiene entropia normalizada = 1.0.
        """
        # AnswerType distribution
        type_counts: Dict[str, int] = Counter(ex.answer_type.value for ex in examples)
        type_entropy = _normalized_entropy(
            list(type_counts.values()), N_ALL_ANSWER_TYPES
        )

        # Level distribution
        level_counts: Dict[int, int] = Counter(ex.complexity_level for ex in examples)
        level_entropy = _normalized_entropy(
            list(level_counts.values()), len(ALL_LEVELS)
        )

        # Domain distribution — inferido del metadata (si existe) o del grafo
        domain_counts: Dict[str, int] = Counter(
            ex.metadata.get("domain", "unknown") for ex in examples
        )
        # Ignorar "unknown" para el calculo de entropia
        known_domain_counts = {k: v for k, v in domain_counts.items() if k != "unknown"}
        domain_entropy = _normalized_entropy(
            list(known_domain_counts.values()), N_ALL_DOMAINS
        ) if known_domain_counts else 0.0

        overall = (type_entropy + level_entropy + domain_entropy) / 3.0

        return DiversityMetric(
            answer_type_entropy = type_entropy,
            level_entropy       = level_entropy,
            domain_entropy      = domain_entropy,
            score               = overall,
            answer_type_counts  = dict(type_counts),
            level_counts        = dict(level_counts),
            domain_counts       = dict(domain_counts),
        )

    @staticmethod
    def _compute_level_balance(examples: List[CausalExample]) -> LevelBalanceMetric:
        """
        Score de balance de niveles: 1 - distancia_total_a_distribución_uniforme.

        Distancia = sum |real_i - ideal_i| / 2  (normalizado a [0,1]).
        Score = 1 - distancia.
        """
        n        = len(examples)
        counts   = Counter(ex.complexity_level for ex in examples)
        ideal    = 1.0 / len(ALL_LEVELS)
        fracs    = {lvl: counts.get(lvl, 0) / n for lvl in ALL_LEVELS}

        total_variation = sum(abs(fracs[lvl] - ideal) for lvl in ALL_LEVELS) / 2.0
        score           = 1.0 - total_variation

        return LevelBalanceMetric(
            level_fractions = fracs,
            score           = max(0.0, score),
        )

    @staticmethod
    def _compute_relation_coverage(examples: List[CausalExample]) -> RelationCoverageMetric:
        """
        Cuantas de las 16 CausalRelations aparecen en al menos una arista del dataset.
        """
        seen: Counter = Counter()
        for ex in examples:
            for edge in ex.graph.edges:
                seen[edge.relation.value] += 1

        n_covered = sum(1 for rel in ALL_RELATIONS if rel in seen)
        missing   = [rel for rel in ALL_RELATIONS if rel not in seen]

        return RelationCoverageMetric(
            n_covered = n_covered,
            n_total   = N_ALL_RELATIONS,
            counts    = dict(seen),
            missing   = missing,
        )

    @staticmethod
    def _compute_entity_spans(examples: List[CausalExample]) -> EntitySpansMetric:
        """
        Fraccion de ejemplos con al menos un span valido (diferente de (-1, -1)).
        """
        n_valid = sum(
            1 for ex in examples
            if any(span != (-1, -1) for span in ex.entity_spans)
        )
        score = n_valid / len(examples)
        return EntitySpansMetric(
            n_with_valid_spans = n_valid,
            n_total            = len(examples),
            score              = score,
        )

    # ── Dataset vacio ─────────────────────────────────────────────────────────

    def _empty_report(self) -> QualityReport:
        return QualityReport(
            n_examples        = 0,
            correctness       = CorrectnessMetric(0, 0, 0.0),
            diversity         = DiversityMetric(0.0, 0.0, 0.0, 0.0),
            level_balance     = LevelBalanceMetric({}, 0.0),
            relation_coverage = RelationCoverageMetric(0, missing=list(ALL_RELATIONS)),
            entity_spans      = EntitySpansMetric(0, 0, 0.0),
            overall_score     = 0.0,
            grade             = "D",
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS PRIVADOS
# ─────────────────────────────────────────────────────────────────────────────

def _normalized_entropy(counts: List[int], n_categories: int) -> float:
    """
    Entropia de Shannon normalizada por el maximo posible (log2 n_categories).

    Resultado en [0, 1]. 1.0 = distribucion perfectamente uniforme.
    Si n_categories <= 1 o no hay datos, devuelve 0.0.
    """
    total = sum(counts)
    if total == 0 or n_categories <= 1:
        return 0.0
    h_max = math.log2(n_categories)
    if h_max == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return min(1.0, h / h_max)


def _score_to_grade(score: float) -> str:
    """A (>= 0.90) / B (>= 0.75) / C (>= 0.50) / D (< 0.50)."""
    if score >= GRADE_THRESHOLDS["A"]:
        return "A"
    if score >= GRADE_THRESHOLDS["B"]:
        return "B"
    if score >= GRADE_THRESHOLDS["C"]:
        return "C"
    return "D"


def _dominant_key(counts: Dict) -> str:
    """Devuelve la clave con mayor recuento."""
    if not counts:
        return "N/A"
    return max(counts, key=counts.get)


def _format_level_dist(fracs: Dict[int, float]) -> str:
    return " ".join(f"L{l}={fracs.get(l, 0):.0%}" for l in ALL_LEVELS)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "python -m tools.dataset_analyzer",
        description = "Analiza la calidad de un dataset sintetico de CausalExamples.",
    )
    p.add_argument(
        "--n", type=int, default=500,
        help="Numero de ejemplos a generar y analizar (default: 500)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Semilla aleatoria (default: None = aleatorio)",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Mostrar detalle de relaciones y tipos",
    )
    p.add_argument(
        "--no-verify", dest="verify", action="store_false", default=True,
        help="No verificar ejemplos durante la generacion (mas rapido)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser  = _build_parser()
    args    = parser.parse_args(argv)

    print(f"Generando {args.n} ejemplos (seed={args.seed}, verify={args.verify})...")
    gen      = CausalGraphGenerator(seed=args.seed)
    examples = gen.generate_batch(n=args.n, verify=args.verify)
    print(f"Generados: {len(examples)} ejemplos.")

    analyzer = DatasetQualityAnalyzer()
    report   = analyzer.analyze(examples)

    print()
    print(report.summary())
    print()
    print(analyzer.recommend(report))

    if args.verbose:
        print()
        print("DETALLE — Distribucion de relaciones:")
        for rel in ALL_RELATIONS:
            cnt = report.relation_coverage.counts.get(rel, 0)
            bar = "#" * min(cnt // max(1, len(examples) // 50), 40)
            print(f"  {rel:<20} {cnt:>5}  {bar}")

        print()
        print("DETALLE — Distribucion de AnswerTypes:")
        for atype in ALL_ANSWER_TYPES:
            cnt = report.diversity.answer_type_counts.get(atype, 0)
            pct = cnt / max(1, len(examples))
            bar = "#" * int(pct * 50)
            print(f"  {atype:<20} {cnt:>5}  ({pct:.1%})  {bar}")


if __name__ == "__main__":
    main()
