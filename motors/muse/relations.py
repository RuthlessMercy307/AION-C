"""
motors/muse/relations.py — Tipos de nodo y relaciones para el motor MUSE
=========================================================================

MUSE razona sobre GRAFOS NARRATIVOS en lugar de grafos causales.
No razona sobre lógica sino sobre TENSIÓN y ARCO.

TIPOS DE NODO (8):
  CHARACTER   — un personaje o voz narrativa
  EVENT       — algo que ocurre en la narrativa
  EMOTION     — un estado emocional de un personaje
  THEME       — un tema abstracto o idea central
  SYMBOL      — un símbolo o metáfora
  SETTING     — un lugar o ambiente narrativo
  CONFLICT    — un conflicto central o tensión
  RESOLUTION  — una resolución de conflicto

RELACIONES NARRATIVAS (10):
  MOTIVATES      — emoción/evento A motiva acción/evento B
  CONFLICTS_WITH — personaje/elemento A conflictúa con B
  DEVELOPS_INTO  — evento A evoluciona/transforma en B
  SYMBOLIZES     — objeto A simboliza tema/emoción B
  PARALLELS      — evento A refleja/espeja evento B
  CONTRASTS      — personaje/elemento A contrasta con B
  FORESHADOWS    — evento A presagia evento B
  RESOLVES       — evento A resuelve conflicto B
  INTENSIFIES    — evento A aumenta la tensión de conflicto B
  SUBVERTS       — evento A subvierte la expectativa de B

NarrativeNode y NarrativeEdge son subclases finas de CausalNode/CausalEdge que
omiten la coerción de tipos para aceptar NarrativeNodeType/NarrativeRelation.
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

class NarrativeNodeType(str, Enum):
    """
    Tipos de nodo en el grafo narrativo de MUSE.

    Semántica para el message passing:
    - CHARACTER:   agente de acción → propaga motivación hacia EVENT
    - CONFLICT:    fuente de tensión → alta energía hasta que RESOLUTION la absorbe
    - RESOLUTION:  nodo destino del arco → tensión cero cuando está alcanzado
    - EMOTION:     estado interno → influye en decisiones de CHARACTER
    - SYMBOL:      nodo de metáfora → conecta THEME con elementos concretos
    - THEME:       nodo abstracto → propaga significado hacia SYMBOL y EVENT
    - EVENT:       nodo de acción → conecta CHARACTER con CONFLICT/RESOLUTION
    - SETTING:     contexto → influye en el tono de CHARACTER y EVENT
    """
    CHARACTER  = "character"
    EVENT      = "event"
    EMOTION    = "emotion"
    THEME      = "theme"
    SYMBOL     = "symbol"
    SETTING    = "setting"
    CONFLICT   = "conflict"
    RESOLUTION = "resolution"


class NarrativeRelation(str, Enum):
    """
    Vocabulario de relaciones narrativas para MUSE.

    Convención: A --relation--> B (todas dirigidas).

    MOTIVACIÓN / CAUSALIDAD NARRATIVA
      MOTIVATES      A (emoción/evento) impulsa B (acción/decisión)
      DEVELOPS_INTO  A evoluciona narrativamente en B

    TENSIÓN
      CONFLICTS_WITH A y B están en oposición activa
      INTENSIFIES    A aumenta la tensión/urgencia del conflicto B

    RESOLUCIÓN
      RESOLVES       A es el evento que cierra/resuelve el conflicto B

    SIMBOLISMO
      SYMBOLIZES     A representa o encarna el significado de B
      PARALLELS      A y B son reflejos estructurales el uno del otro

    CONTRASTE / EXPECTATIVA
      CONTRASTS      A y B están en oposición de carácter o valor
      FORESHADOWS    A anticipa o señala hacia el evento B
      SUBVERTS       A invierte la expectativa creada por B
    """
    MOTIVATES      = "motivates"
    CONFLICTS_WITH = "conflicts_with"
    DEVELOPS_INTO  = "develops_into"
    SYMBOLIZES     = "symbolizes"
    PARALLELS      = "parallels"
    CONTRASTS      = "contrasts"
    FORESHADOWS    = "foreshadows"
    RESOLVES       = "resolves"
    INTENSIFIES    = "intensifies"
    SUBVERTS       = "subverts"


# Listas ordenadas — el índice es estable (lo usa el embedding de relaciones)
NARRATIVE_NODE_TYPES: List[str] = [t.value for t in NarrativeNodeType]
NARRATIVE_RELATIONS:  List[str] = [r.value for r in NarrativeRelation]

# Agrupaciones semánticas
MOTIVATION_RELATIONS:  set = {NarrativeRelation.MOTIVATES, NarrativeRelation.DEVELOPS_INTO}
TENSION_RELATIONS:     set = {NarrativeRelation.CONFLICTS_WITH, NarrativeRelation.INTENSIFIES}
RESOLUTION_RELATIONS:  set = {NarrativeRelation.RESOLVES}
SYMBOLIC_RELATIONS:    set = {NarrativeRelation.SYMBOLIZES, NarrativeRelation.PARALLELS}
CONTRAST_RELATIONS:    set = {NarrativeRelation.CONTRASTS, NarrativeRelation.FORESHADOWS, NarrativeRelation.SUBVERTS}


# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVE NODE Y NARRATIVE EDGE — WRAPPERS SIN COERCIÓN DE TIPO
# ─────────────────────────────────────────────────────────────────────────────

from core.graph import CausalNode, CausalEdge


@dataclass
class NarrativeNode(CausalNode):
    """
    Variante de CausalNode que acepta NarrativeNodeType sin coercionar a NodeType.
    Se usa en grafos narrativos de MUSE.
    """
    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        # No coercionar node_type — preservar NarrativeNodeType tal cual


@dataclass
class NarrativeEdge(CausalEdge):
    """
    Variante de CausalEdge que acepta NarrativeRelation sin coercionar a CausalRelation.
    `edge.relation.value` devuelve el string de la relación (p.ej. "motivates").
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
        # No coercionar relation — preservar NarrativeRelation tal cual
