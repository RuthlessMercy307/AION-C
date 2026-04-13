"""
motors/axiom/relations.py — Tipos de nodo y relaciones para el motor AXIOM
===========================================================================

AXIOM razona sobre GRAFOS DE PRUEBAS MATEMÁTICAS en lugar de grafos causales.
Razona como un matemático: axiomas → derivaciones → QED.

TIPOS DE NODO (8):
  AXIOM       — verdad asumida sin necesidad de demostración
  DEFINITION  — definición formal de un objeto o propiedad
  THEOREM     — proposición que debe/ha sido demostrada
  LEMMA       — resultado auxiliar en el camino a un teorema
  HYPOTHESIS  — suposición temporal (para demostración por contradicción)
  EXPRESSION  — expresión matemática (fórmula, término)
  EQUALITY    — igualdad o desigualdad (relación entre expresiones)
  SET         — conjunto, dominio o espacio matemático

RELACIONES DE PRUEBA (10):
  DERIVES        — paso A se deriva lógicamente de paso B
  ASSUMES        — proposición A asume hipótesis B
  CONTRADICTS    — resultado A contradice hipótesis B (reductio ad absurdum)
  GENERALIZES    — teorema A es caso general de B
  SPECIALIZES    — A es instancia particular de regla B
  APPLIES        — paso A aplica (instancia) la regla/teorema B
  REDUCES_TO     — problema A se transforma/reduce a problema B
  BOUNDS         — expresión A acota (≤ o ≥) expresión B
  EQUIVALENT_TO  — A y B son equivalentes por transformación algebraica
  IMPLIES        — proposición A implica lógicamente proposición B

MathNode y MathEdge son subclases finas de CausalNode/CausalEdge que
omiten la coerción de tipos para aceptar MathNodeType/MathRelation.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dataclasses import dataclass
from enum import Enum
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# ENUMERACIONES
# ─────────────────────────────────────────────────────────────────────────────

class MathNodeType(str, Enum):
    """
    Tipos de nodo en el grafo de prueba matemática de AXIOM.

    Semántica para el message passing:
    - AXIOM:       fuente de verdad → propaga certeza máxima
    - HYPOTHESIS:  verdad temporal → sus derivados son condicionales
    - THEOREM:     destino de la prueba → alta energía si no está demostrado
    - EXPRESSION:  objeto manipulable → recibe EQUIVALENT_TO, REDUCES_TO
    - EQUALITY:    relación entre expresiones → une EXPRESSION con DERIVES
    """
    AXIOM      = "axiom"
    DEFINITION = "definition"
    THEOREM    = "theorem"
    LEMMA      = "lemma"
    HYPOTHESIS = "hypothesis"
    EXPRESSION = "expression"
    EQUALITY   = "equality"
    SET        = "set"


class MathRelation(str, Enum):
    """
    Vocabulario de relaciones de prueba matemática para AXIOM.

    Convención: A --relation--> B (todas dirigidas).

    INFERENCIA LÓGICA
      DERIVES       A se obtiene de B por deducción
      IMPLIES       A implica B (lógica proposicional/predicados)
      ASSUMES       A utiliza B como hipótesis de trabajo

    REFUTACIÓN
      CONTRADICTS   A contradice B (para demostración por contradicción)

    JERARQUÍA TEÓRICA
      GENERALIZES   A es el caso general del que B es instancia
      SPECIALIZES   A es instancia particular de la regla/teorema B
      APPLIES       A aplica (instancia) la regla/teorema B a un caso concreto

    TRANSFORMACIÓN ALGEBRAICA
      REDUCES_TO    A se transforma en B (simplificación/sustitución)
      EQUIVALENT_TO A y B son algebraicamente equivalentes (=)

    COTAS
      BOUNDS        A establece una cota superior o inferior sobre B
    """
    DERIVES       = "derives"
    ASSUMES       = "assumes"
    CONTRADICTS   = "contradicts"
    GENERALIZES   = "generalizes"
    SPECIALIZES   = "specializes"
    APPLIES       = "applies"
    REDUCES_TO    = "reduces_to"
    BOUNDS        = "bounds"
    EQUIVALENT_TO = "equivalent_to"
    IMPLIES       = "implies"


# Listas ordenadas — el índice es estable (lo usa el embedding de relaciones)
MATH_NODE_TYPES: List[str] = [t.value for t in MathNodeType]
MATH_RELATIONS:  List[str] = [r.value for r in MathRelation]

# Agrupaciones semánticas
INFERENCE_RELATIONS: set = {MathRelation.DERIVES, MathRelation.IMPLIES, MathRelation.ASSUMES}
REFUTATION_RELATIONS: set = {MathRelation.CONTRADICTS}
HIERARCHY_RELATIONS: set = {MathRelation.GENERALIZES, MathRelation.SPECIALIZES, MathRelation.APPLIES}
TRANSFORM_RELATIONS: set = {MathRelation.REDUCES_TO, MathRelation.EQUIVALENT_TO}
BOUND_RELATIONS:     set = {MathRelation.BOUNDS}


# ─────────────────────────────────────────────────────────────────────────────
# MATH NODE Y MATH EDGE — WRAPPERS SIN COERCIÓN DE TIPO
# ─────────────────────────────────────────────────────────────────────────────

from core.graph import CausalNode, CausalEdge


@dataclass
class MathNode(CausalNode):
    """
    Variante de CausalNode que acepta MathNodeType sin coercionar a NodeType.
    Se usa en grafos de prueba matemática de AXIOM.
    """
    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        # No coercionar node_type — preservar MathNodeType tal cual


@dataclass
class MathEdge(CausalEdge):
    """
    Variante de CausalEdge que acepta MathRelation sin coercionar a CausalRelation.
    `edge.relation.value` devuelve el string de la relación (p.ej. "derives").
    """
    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"strength must be in [0, 1], got {self.strength!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        if self.source_id == self.target_id:
            raise ValueError(
                f"Self-loops not allowed: source_id == target_id == {self.source_id!r}"
            )
        # No coercionar relation — preservar MathRelation tal cual
