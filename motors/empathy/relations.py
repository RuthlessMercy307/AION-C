"""
motors/empathy/relations.py — Tipos de nodo y relaciones para el motor EMPATHY
===============================================================================

EMPATHY razona sobre GRAFOS SOCIALES.
Modela intenciones, creencias, normas y emociones de personas.

TIPOS DE NODO (8):
  PERSON       — un individuo o grupo social
  INTENTION    — lo que alguien quiere lograr
  BELIEF       — lo que alguien cree verdadero (puede ser falso)
  EMOTION      — lo que alguien siente
  NORM         — una regla social (explícita o implícita)
  CONTEXT      — la situación social actual
  RELATIONSHIP — relación entre dos personas o grupos
  EXPECTATION  — lo que alguien espera que ocurra

RELACIONES SOCIALES (10):
  WANTS         — persona A desea resultado/acción B
  BELIEVES      — persona A cree proposición/hecho B
  FEELS         — persona A siente emoción B
  EXPECTS       — persona A espera evento/reacción B
  VIOLATES_NORM — acción/persona A viola norma social B
  EMPATHIZES    — persona A comprende y comparte la emoción de B
  PERSUADES     — persona A intenta convencer a persona B
  TRUSTS        — persona A confía en persona/institución B
  MISUNDERSTANDS — persona A malinterpreta la intención de B
  RECIPROCATES  — persona A corresponde la acción/gesto de B

SocialNode y SocialEdge son subclases finas de CausalNode/CausalEdge que
omiten la coerción de tipos para aceptar SocialNodeType/SocialRelation.
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

class SocialNodeType(str, Enum):
    """
    Tipos de nodo en el grafo social de EMPATHY.

    Semántica para el message passing:
    - PERSON:       agente principal → propaga intenciones y emociones
    - INTENTION:    objetivo perseguido → fuente de acción y conflicto
    - BELIEF:       modelo mental → puede ser verdadero o falso (MISUNDERSTANDS)
    - EMOTION:      estado afectivo → influye en TRUSTS, EMPATHIZES, PERSUADES
    - NORM:         restricción social → su violación genera conflicto
    - CONTEXT:      ambiente → modula la interpretación de creencias y normas
    - RELATIONSHIP: vínculo entre PERSONs → canal de TRUSTS y RECIPROCATES
    - EXPECTATION:  predicción → su violación genera sorpresa y reacción
    """
    PERSON       = "person"
    INTENTION    = "intention"
    BELIEF       = "belief"
    EMOTION      = "emotion"
    NORM         = "norm"
    CONTEXT      = "context"
    RELATIONSHIP = "relationship"
    EXPECTATION  = "expectation"


class SocialRelation(str, Enum):
    """
    Vocabulario de relaciones sociales para EMPATHY.

    Convención: A --relation--> B (todas dirigidas).

    DESEO / CREENCIA / EMOCIÓN
      WANTS         A (persona) quiere lograr B (intención/resultado)
      BELIEVES      A (persona) cree verdadero B (creencia/proposición)
      FEELS         A (persona) experimenta B (emoción)
      EXPECTS       A (persona) anticipa B (expectativa/evento)

    NORMAS
      VIOLATES_NORM A (acción/persona) viola la norma social B

    INTERACCIÓN SOCIAL
      EMPATHIZES    A comprende y valida la emoción/estado de B
      PERSUADES     A intenta modificar la creencia o acción de B
      TRUSTS        A confía en la honestidad/competencia de B
      MISUNDERSTANDS A malinterpreta la intención o estado de B
      RECIPROCATES  A responde correspondiendo el gesto de B
    """
    WANTS          = "wants"
    BELIEVES       = "believes"
    FEELS          = "feels"
    EXPECTS        = "expects"
    VIOLATES_NORM  = "violates_norm"
    EMPATHIZES     = "empathizes"
    PERSUADES      = "persuades"
    TRUSTS         = "trusts"
    MISUNDERSTANDS = "misunderstands"
    RECIPROCATES   = "reciprocates"


# Listas ordenadas — el índice es estable (lo usa el embedding de relaciones)
SOCIAL_NODE_TYPES: List[str] = [t.value for t in SocialNodeType]
SOCIAL_RELATIONS:  List[str] = [r.value for r in SocialRelation]

# Agrupaciones semánticas
MENTAL_STATE_RELATIONS:  set = {SocialRelation.WANTS, SocialRelation.BELIEVES,
                                 SocialRelation.FEELS, SocialRelation.EXPECTS}
NORM_RELATIONS:          set = {SocialRelation.VIOLATES_NORM}
INTERACTION_RELATIONS:   set = {SocialRelation.EMPATHIZES, SocialRelation.PERSUADES,
                                 SocialRelation.TRUSTS, SocialRelation.RECIPROCATES}
CONFLICT_RELATIONS:      set = {SocialRelation.MISUNDERSTANDS, SocialRelation.VIOLATES_NORM}


# ─────────────────────────────────────────────────────────────────────────────
# SOCIAL NODE Y SOCIAL EDGE — WRAPPERS SIN COERCIÓN DE TIPO
# ─────────────────────────────────────────────────────────────────────────────

from core.graph import CausalNode, CausalEdge


@dataclass
class SocialNode(CausalNode):
    """
    Variante de CausalNode que acepta SocialNodeType sin coercionar a NodeType.
    Se usa en grafos sociales de EMPATHY.
    """
    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        # No coercionar node_type — preservar SocialNodeType tal cual


@dataclass
class SocialEdge(CausalEdge):
    """
    Variante de CausalEdge que acepta SocialRelation sin coercionar a CausalRelation.
    `edge.relation.value` devuelve el string de la relación (p.ej. "wants").
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
        # No coercionar relation — preservar SocialRelation tal cual
