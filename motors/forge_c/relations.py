"""
motors/forge_c/relations.py — Tipos de nodo y relaciones para el motor FORGE-C
================================================================================

FORGE-C razona sobre GRAFOS DE CÓDIGO en lugar de grafos causales abstractos.
Los 8 tipos de nodo y las 12 relaciones de código son ortogonales a los de CORA:
tienen semántica distinta y funciones de mensaje distintas (no se comparten pesos).

TIPOS DE NODO (8):
  FUNCTION    — función o método (algo que ejecuta)
  CLASS       — clase o struct (algo que agrupa estado + comportamiento)
  MODULE      — archivo, módulo o paquete (unidad de organización)
  VARIABLE    — variable, atributo o campo (algo que almacena estado)
  EXPRESSION  — expresión, literal o valor (algo que evalúa a un resultado)
  ERROR       — excepción, error o condición de fallo
  TEST        — test, assertion o verificación
  CONFIG      — constante, configuración o parámetro de entorno

RELACIONES DE CÓDIGO (12):
  CALLS          — función A invoca a función B
  IMPORTS        — módulo A importa módulo/símbolo B
  INHERITS       — clase A hereda de clase B
  MUTATES        — función A escribe/modifica variable B
  READS          — función A lee variable B
  RETURNS        — función A retorna valor/tipo B
  THROWS         — función A puede lanzar error B
  DEPENDS_ON     — módulo A depende de módulo B (build/runtime)
  TESTS          — test A verifica función/clase B
  IMPLEMENTS     — clase A implementa interface B
  OVERRIDES      — método A sobreescribe método B del padre
  DATA_FLOWS_TO  — el output de A es input de B (flujo de datos)

CodeNode y CodeEdge son subclases finas de CausalNode/CausalEdge que
omiten la coerción de tipos para aceptar CodeNodeType/CodeRelation
en lugar de NodeType/CausalRelation.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# ENUMERACIONES
# ─────────────────────────────────────────────────────────────────────────────

class CodeNodeType(str, Enum):
    """
    Tipos de nodo en el grafo de código de FORGE-C.

    Cada tipo tiene implicaciones semánticas para el message passing:
    - FUNCTION:   ejecutable → puede CALLS, RETURNS, THROWS, MUTATES, READS
    - CLASS:      agrupador → puede INHERITS, IMPLEMENTS, OVERRIDES
    - MODULE:     contenedor → puede IMPORTS, DEPENDS_ON
    - VARIABLE:   estado     → puede ser MUTATES, READS target
    - EXPRESSION: valor      → puede ser RETURNS source, DATA_FLOWS_TO source
    - ERROR:      fallo      → puede ser THROWS target
    - TEST:       verificador → puede TESTS target
    - CONFIG:     constante  → puede ser READS target, DEPENDS_ON source
    """
    FUNCTION   = "function"
    CLASS      = "class"
    MODULE     = "module"
    VARIABLE   = "variable"
    EXPRESSION = "expression"
    ERROR      = "error"
    TEST       = "test"
    CONFIG     = "config"


class CodeRelation(str, Enum):
    """
    Vocabulario de relaciones de código para FORGE-C.

    Convención: A --relation--> B (todas son dirigidas).

    INVOCACIÓN / CONTROL
      CALLS          A invoca B directamente (llamada de función)
      DATA_FLOWS_TO  El output de A es input de B (flujo de datos implícito)

    ORGANIZACIÓN / ESTRUCTURA
      IMPORTS        A importa B (módulo o símbolo)
      INHERITS       A hereda de B (jerarquía de clases)
      DEPENDS_ON     A depende de B en tiempo de build/runtime

    CONTRATO / INTERFAZ
      IMPLEMENTS     A implementa la interfaz B
      OVERRIDES      A sobreescribe el método B del padre

    ACCESO A ESTADO
      MUTATES        A escribe o modifica B
      READS          A lee o consulta B

    FLUJO DE ERRORES / OUTPUT
      RETURNS        A retorna el tipo o valor B
      THROWS         A puede lanzar la excepción B

    VERIFICACIÓN
      TESTS          A verifica el comportamiento de B
    """
    CALLS         = "calls"
    IMPORTS       = "imports"
    INHERITS      = "inherits"
    MUTATES       = "mutates"
    READS         = "reads"
    RETURNS       = "returns"
    THROWS        = "throws"
    DEPENDS_ON    = "depends_on"
    TESTS         = "tests"
    IMPLEMENTS    = "implements"
    OVERRIDES     = "overrides"
    DATA_FLOWS_TO = "data_flows_to"


# Listas ordenadas — el índice es estable (lo usa el embedding de relaciones)
CODE_NODE_TYPES: List[str] = [t.value for t in CodeNodeType]
CODE_RELATIONS:  List[str] = [r.value for r in CodeRelation]

# Agrupaciones semánticas (útiles para análisis y generación sintética)
WRITE_RELATIONS: set = {CodeRelation.MUTATES, CodeRelation.RETURNS}
READ_RELATIONS:  set = {CodeRelation.READS, CodeRelation.DATA_FLOWS_TO}
CALL_RELATIONS:  set = {CodeRelation.CALLS, CodeRelation.DATA_FLOWS_TO}
HIER_RELATIONS:  set = {CodeRelation.INHERITS, CodeRelation.IMPLEMENTS, CodeRelation.OVERRIDES}
DEPS_RELATIONS:  set = {CodeRelation.IMPORTS, CodeRelation.DEPENDS_ON}


# ─────────────────────────────────────────────────────────────────────────────
# CODE NODE Y CODE EDGE — WRAPPERS SIN COERCIÓN DE TIPO
# ─────────────────────────────────────────────────────────────────────────────

# Importamos las clases base después de fijar el path
from core.graph import CausalNode, CausalEdge


@dataclass
class CodeNode(CausalNode):
    """
    Variante de CausalNode que acepta CodeNodeType sin coercionar a NodeType.

    Se usa en grafos de código de FORGE-C. El campo node_type puede contener
    un valor de CodeNodeType (en lugar de NodeType) sin errores en runtime.
    """

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        # No coercionar node_type — preservar CodeNodeType tal cual


@dataclass
class CodeEdge(CausalEdge):
    """
    Variante de CausalEdge que acepta CodeRelation sin coercionar a CausalRelation.

    Se usa en grafos de código de FORGE-C. El campo relation puede contener
    un valor de CodeRelation (en lugar de CausalRelation) sin errores en runtime.
    `edge.relation.value` devuelve el string de la relación (p.ej. "calls").
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
        # No coercionar relation — preservar CodeRelation tal cual
