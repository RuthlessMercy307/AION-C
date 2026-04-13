"""
synth/narrative_graph_gen.py — Generador de Grafos Narrativos para MUSE
========================================================================

Motor de datos sintéticos para el motor MUSE (razonamiento narrativo/creativo).
Genera mini-narrativas con estructura verificable:

  - Arco lineal:       personaje A desea X → acción → resultado
  - Con conflicto:     personaje A desea X, conflicto con B → resolución C
  - Con subversión:    setup → expectativa → giro inesperado (SUBVERTS)

Tres niveles de complejidad:

  Nivel 1 — Arco lineal (3-4 nodos)
             "Personaje desea algo → toma acción → logra resultado"
             Verificar: hay MOTIVATES antes de EVENT, hay desarrollo al final
             Subtypes: desire_action_result, setting_event_emotion

  Nivel 2 — Arco con conflicto (4-6 nodos)
             "Personaje desea X, entra en conflicto con antagonista,
              conflicto se intensifica, hay resolución"
             Verificar: hay CONFLICT antes de RESOLUTION,
                        hay MOTIVATES antes del conflicto
             Subtypes: protagonist_antagonist, internal_conflict, rival_resolution

  Nivel 3 — Arco con subversión/simbolismo (5-8 nodos)
             "Setup con expectativa → giro (SUBVERTS) o doble conflicto
              o símbolo que refuerza tema"
             Verificar: hay FORESHADOWS antes de SUBVERTS,
                        la resolución llega después de al menos 1 intensificación
             Subtypes: subverted_expectation, symbolic_arc, parallel_conflict, double_arc

Contrato de cada NarrativeExample:
  - problem_text:     descripción de la mini-narrativa en lenguaje natural
  - graph:            CausalGraph con NarrativeNode/NarrativeEdge
  - answer:           respuesta a la pregunta narrativa
  - complexity_level: 1-3
  - answer_type:      NarrativeAnswerType
  - metadata:         parámetros para verify_narrative_example()
  - example_id:       UUID reproducible

Verificación de coherencia narrativa:
  - has_motivation_before_action:  ¿hay un nodo MOTIVATES/EMOTION antes de EVENT?
  - has_conflict_before_resolution: ¿hay CONFLICT antes de RESOLUTION en el grafo?
  - has_foreshadowing_before_payoff: ¿hay FORESHADOWS antes de SUBVERTS/RESOLVES?
  - arc_is_complete:                ¿el grafo tiene el tipo de nodo esperado para el nivel?

Uso básico:
    gen = NarrativeGraphGenerator(seed=42)
    ex  = gen.generate(level=1)
    res = verify_narrative_example(ex)
    assert res.passed

    batch = gen.generate_batch(n=100, level_distribution={1: 0.4, 2: 0.4, 3: 0.2})
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.graph import CausalGraph
from motors.muse.relations import (
    NARRATIVE_RELATIONS,
    NarrativeEdge,
    NarrativeNode,
    NarrativeNodeType,
    NarrativeRelation,
)


# ─────────────────────────────────────────────────────────────────────────────
# TIPOS DE RESPUESTA NARRATIVA
# ─────────────────────────────────────────────────────────────────────────────

class NarrativeAnswerType(str, Enum):
    ARC_COMPLETION      = "arc_completion"      # ¿Cómo termina el arco?
    CONFLICT_RESOLUTION = "conflict_resolution" # ¿Cómo se resuelve el conflicto?
    MOTIVATION          = "motivation"          # ¿Qué motiva al personaje?
    THEME_IDENTIFICATION= "theme_identification"# ¿Cuál es el tema central?
    TWIST_IDENTIFICATION= "twist_identification"# ¿Cuál es el giro narrativo?
    SYMBOL_MEANING      = "symbol_meaning"      # ¿Qué simboliza el objeto X?


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO DE VERIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NarrativeVerificationResult:
    """
    Resultado de verify_narrative_example().

    passed:  True si el grafo tiene coherencia narrativa verificable
    reason:  Explicación en lenguaje natural del resultado
    details: Datos cuantitativos para debugging
    """
    passed:  bool
    reason:  str
    details: Dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"NarrativeVerificationResult({status}: {self.reason})"


# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVE EXAMPLE — UNIDAD ATÓMICA DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NarrativeExample:
    """
    Unidad atómica de entrenamiento para MUSE.

    El grafo captura el arco narrativo completo con NarrativeNode/NarrativeEdge.

    metadata siempre incluye las claves necesarias para verify_narrative_example():
      - required_node_types:  List[str] — tipos que deben aparecer en el grafo
      - required_relations:   List[str] — relaciones que deben aparecer en aristas
      - arc_checks:           Dict — checks de coherencia específicos por nivel
    """
    problem_text:     str
    graph:            CausalGraph
    answer:           str
    complexity_level: int
    answer_type:      NarrativeAnswerType
    verifiable:       bool  = True
    metadata:         Dict  = field(default_factory=dict)
    example_id:       str   = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def __repr__(self) -> str:
        return (
            f"NarrativeExample(level={self.complexity_level}, "
            f"type={self.answer_type.value}, "
            f"nodes={len(self.graph)}, id={self.example_id})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — CONSTRUCCIÓN DE GRAFOS
# ─────────────────────────────────────────────────────────────────────────────

def _narr_node(
    nid: str, label: str, ntype: NarrativeNodeType, conf: float = 1.0
) -> NarrativeNode:
    return NarrativeNode(
        node_id=nid, label=label, node_type=ntype,
        confidence=conf, grounded=True
    )


def _narr_edge(
    src: str, tgt: str, rel: NarrativeRelation,
    strength: float = 1.0, conf: float = 1.0
) -> NarrativeEdge:
    return NarrativeEdge(
        source_id=src, target_id=tgt, relation=rel,
        strength=strength, confidence=conf
    )


def _node_types_in_graph(graph: CausalGraph) -> Set[str]:
    """Retorna el conjunto de valores de NarrativeNodeType presentes en el grafo."""
    return {node.node_type.value for node in graph.nodes}


def _relations_in_graph(graph: CausalGraph) -> Set[str]:
    """Retorna el conjunto de valores de NarrativeRelation presentes en aristas."""
    return {edge.relation.value for edge in graph.edges}


# ─────────────────────────────────────────────────────────────────────────────
# DATOS NARRATIVOS — NOMBRES, EMOCIONES, SETTINGS, TEMAS
# ─────────────────────────────────────────────────────────────────────────────

_CHARACTERS = [
    "Elena", "Marcos", "Sofía", "Andrés", "Lucía", "Tomás",
    "Valentina", "Carlos", "Isabela", "Diego", "Camila", "Javier",
]

_EMOTIONS = [
    ("esperanza", "la esperanza de un futuro mejor"),
    ("miedo",     "el miedo a lo desconocido"),
    ("amor",      "el amor que lo sostiene"),
    ("rabia",     "la rabia acumulada por años"),
    ("soledad",   "la soledad que lo consume"),
    ("ambición",  "la ambición que lo ciega"),
    ("culpa",     "la culpa que lo paraliza"),
    ("curiosidad","la curiosidad que lo impulsa"),
]

_DESIRES = [
    "recuperar lo que perdió",
    "demostrar su valor",
    "encontrar la verdad",
    "proteger a su familia",
    "escapar de su pasado",
    "conquistar el reconocimiento",
    "salvar a alguien que ama",
    "cumplir una promesa",
]

_SETTINGS = [
    ("una ciudad lluviosa en invierno",   "ciudad"),
    ("un pueblo olvidado en la montaña",  "pueblo"),
    ("una gran mansión abandonada",       "mansión"),
    ("el laboratorio de una corporación", "laboratorio"),
    ("un mercado bullicioso al amanecer", "mercado"),
    ("la orilla de un río al atardecer",  "río"),
]

_THEMES = [
    ("la redención",          "el camino hacia el perdón propio"),
    ("la identidad",          "quién somos cuando nadie nos ve"),
    ("el poder",              "cómo el poder corrompe y libera"),
    ("la lealtad",            "a quién somos verdaderamente leales"),
    ("la pérdida",            "lo que queda después de perder algo"),
    ("la transformación",     "nadie sale igual de una crisis"),
]

_SYMBOLS = [
    ("un reloj roto",          "el tiempo que no se puede recuperar"),
    ("una carta sin enviar",   "las palabras que nunca se dijeron"),
    ("una semilla en cenizas", "el renacimiento posible"),
    ("un espejo distorsionado","la percepción distorsionada de uno mismo"),
    ("una llave oxidada",      "una puerta que ya no existe"),
    ("una vela encendida",     "la esperanza frágil pero persistente"),
]

_CONFLICT_TYPES = [
    "una disputa por la herencia familiar",
    "una traición que destruyó su amistad",
    "una mentira que define toda su relación",
    "una competencia que los enfrenta directamente",
    "un secreto que uno guarda y el otro busca",
    "una deuda moral que ninguno puede pagar",
]

_RESOLUTION_TYPES = [
    ("reconciliación",  "ambos encuentran la manera de perdonarse"),
    ("sacrificio",      "uno cede todo para salvar lo que importa"),
    ("revelación",      "la verdad cambia todo lo que creían saber"),
    ("separación",      "reconocen que su camino juntos ha terminado"),
    ("transformación",  "ninguno sale siendo la misma persona"),
    ("aceptación",      "aprenden a vivir con lo que no puede cambiarse"),
]

_TWISTS = [
    ("el aparente antagonista era quien más lo apoyaba",
     "la persona que parecía el enemigo resultó ser el mayor aliado"),
    ("su deseo cumplido resultó ser exactamente lo que no necesitaba",
     "conseguir lo que quería reveló que lo que buscaba estaba en otra parte"),
    ("la amenaza era interna, no externa",
     "el verdadero conflicto siempre había sido con uno mismo"),
    ("el símbolo que temía era lo que lo protegía",
     "aquello que representaba su miedo era en realidad su escudo"),
    ("el final que evitó era el único que podría haberle dado paz",
     "al huir del destino que temía, lo convocó"),
]


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 1 — ARCO LINEAL (3-4 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level1Generator:
    """
    Nivel 1: Arco narrativo lineal de un único conflicto.

    Subtypes:
      desire_action_result:  CHARACTER desea X → EVENT (acción) → EMOTION (resultado)
      setting_event_emotion: SETTING → EVENT → EMOTION (impacto emocional)
    """

    _SUBTYPES = ["desire_action_result", "setting_event_emotion"]

    def generate(self, rng: random.Random) -> NarrativeExample:
        sub = rng.choice(self._SUBTYPES)
        if sub == "desire_action_result":
            return self._desire_action_result(rng)
        else:
            return self._setting_event_emotion(rng)

    def _desire_action_result(self, rng: random.Random) -> NarrativeExample:
        """
        CHARACTER --MOTIVATES--> EVENT --DEVELOPS_INTO--> EMOTION
        Pregunta: ¿Qué motiva al personaje?
        """
        name   = rng.choice(_CHARACTERS)
        desire = rng.choice(_DESIRES)
        emo_val, emo_desc = rng.choice(_EMOTIONS)

        action = rng.choice([
            f"decide actuar para {desire}",
            f"toma la decisión más difícil de su vida",
            f"da el primer paso hacia su objetivo",
            f"confronta lo que siempre evitó",
        ])
        result_emo = rng.choice([
            f"descubre que {emo_desc}",
            f"siente {emo_val} por primera vez en años",
            f"la {emo_val} lo transforma desde adentro",
        ])

        g = CausalGraph()
        char  = _narr_node("char", f"{name}: quiere {desire}",        NarrativeNodeType.CHARACTER)
        event = _narr_node("ev1",  f"{name} {action}",                NarrativeNodeType.EVENT)
        emot  = _narr_node("em1",  result_emo,                        NarrativeNodeType.EMOTION)
        g.add_node(char).add_node(event).add_node(emot)
        g.add_edge(_narr_edge("char", "ev1", NarrativeRelation.MOTIVATES))
        g.add_edge(_narr_edge("ev1",  "em1", NarrativeRelation.DEVELOPS_INTO))
        g.root_question = f"¿Qué motiva a {name}?"

        return NarrativeExample(
            problem_text=(
                f"{name} quiere {desire}. "
                f"Para lograrlo, {action}. "
                f"El resultado es que {result_emo}."
            ),
            graph=g,
            answer=f"{name} está motivado/a por el deseo de {desire}.",
            complexity_level=1,
            answer_type=NarrativeAnswerType.MOTIVATION,
            metadata={
                "character": name,
                "desire": desire,
                "emotion": emo_val,
                "required_node_types": ["character", "event", "emotion"],
                "required_relations":  ["motivates", "develops_into"],
                "arc_checks": {
                    "has_motivation": True,
                    "motivation_before_action": True,
                },
            },
        )

    def _setting_event_emotion(self, rng: random.Random) -> NarrativeExample:
        """
        SETTING --FORESHADOWS--> EVENT --DEVELOPS_INTO--> EMOTION
        Pregunta: ¿Cómo termina el arco?
        """
        name        = rng.choice(_CHARACTERS)
        set_desc, _ = rng.choice(_SETTINGS)
        emo_val, emo_desc = rng.choice(_EMOTIONS)

        event_desc = rng.choice([
            f"algo inesperado interrumpe la rutina de {name}",
            f"{name} se enfrenta a una decisión que no esperaba",
            f"un encuentro fortuito cambia el día de {name}",
            f"{name} descubre algo que no debía saber",
        ])
        emo_result = rng.choice([
            f"{name} siente {emo_val} que no sabía que llevaba dentro",
            f"la {emo_val} de {name} se hace insostenible",
            f"{name} acepta finalmente {emo_desc}",
        ])

        g = CausalGraph()
        sett  = _narr_node("set1", set_desc,    NarrativeNodeType.SETTING)
        event = _narr_node("ev1",  event_desc,  NarrativeNodeType.EVENT)
        emot  = _narr_node("em1",  emo_result,  NarrativeNodeType.EMOTION)
        g.add_node(sett).add_node(event).add_node(emot)
        g.add_edge(_narr_edge("set1", "ev1", NarrativeRelation.FORESHADOWS))
        g.add_edge(_narr_edge("ev1",  "em1", NarrativeRelation.DEVELOPS_INTO))
        g.root_question = f"¿Cómo termina el arco emocional de {name}?"

        return NarrativeExample(
            problem_text=(
                f"En {set_desc}, {event_desc}. "
                f"Como resultado, {emo_result}."
            ),
            graph=g,
            answer=emo_result,
            complexity_level=1,
            answer_type=NarrativeAnswerType.ARC_COMPLETION,
            metadata={
                "character": name,
                "setting":   set_desc,
                "emotion":   emo_val,
                "required_node_types": ["setting", "event", "emotion"],
                "required_relations":  ["foreshadows", "develops_into"],
                "arc_checks": {
                    "has_setting": True,
                    "setting_before_event": True,
                },
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 2 — ARCO CON CONFLICTO (4-6 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level2Generator:
    """
    Nivel 2: Arco con conflicto central y resolución.

    Subtypes:
      protagonist_antagonist: CHARACTER_A vs CHARACTER_B → CONFLICT → RESOLUTION
      internal_conflict:      CHARACTER → EMOTION → CONFLICT → RESOLUTION
      rival_resolution:       dos CHARACTERs + CONFLICT con INTENSIFIES + RESOLVES
    """

    _SUBTYPES = ["protagonist_antagonist", "internal_conflict", "rival_resolution"]

    def generate(self, rng: random.Random) -> NarrativeExample:
        sub = rng.choice(self._SUBTYPES)
        if sub == "protagonist_antagonist":
            return self._protagonist_antagonist(rng)
        elif sub == "internal_conflict":
            return self._internal_conflict(rng)
        else:
            return self._rival_resolution(rng)

    def _protagonist_antagonist(self, rng: random.Random) -> NarrativeExample:
        """
        char_a --MOTIVATES--> conflict <--CONFLICTS_WITH-- char_b
        conflict --INTENSIFIES--> conflict (auto via ev_int)
        ev_res --RESOLVES--> conflict
        """
        names = rng.sample(_CHARACTERS, 2)
        name_a, name_b = names[0], names[1]
        desire  = rng.choice(_DESIRES)
        conf_type = rng.choice(_CONFLICT_TYPES)
        res_type, res_desc = rng.choice(_RESOLUTION_TYPES)

        intensify_event = rng.choice([
            f"la tensión entre {name_a} y {name_b} llega a su punto máximo",
            f"un nuevo incidente agrava el conflicto",
            f"la situación se vuelve irreversible",
        ])
        resolution_event = rng.choice([
            f"{name_a} y {name_b} se enfrentan por última vez",
            f"la verdad sale a la luz",
            f"uno de ellos toma una decisión definitiva",
        ])

        g = CausalGraph()
        char_a   = _narr_node("cha", f"{name_a}: quiere {desire}",    NarrativeNodeType.CHARACTER)
        char_b   = _narr_node("chb", f"{name_b}: se opone",           NarrativeNodeType.CHARACTER)
        conflict = _narr_node("cf1", conf_type,                       NarrativeNodeType.CONFLICT)
        ev_int   = _narr_node("ei1", intensify_event,                  NarrativeNodeType.EVENT)
        res      = _narr_node("res", resolution_event,                 NarrativeNodeType.RESOLUTION)
        g.add_node(char_a).add_node(char_b).add_node(conflict).add_node(ev_int).add_node(res)
        g.add_edge(_narr_edge("cha", "cf1", NarrativeRelation.MOTIVATES))
        g.add_edge(_narr_edge("chb", "cf1", NarrativeRelation.CONFLICTS_WITH))
        g.add_edge(_narr_edge("ei1", "cf1", NarrativeRelation.INTENSIFIES))
        g.add_edge(_narr_edge("res", "cf1", NarrativeRelation.RESOLVES))
        g.root_question = f"¿Cómo se resuelve el conflicto entre {name_a} y {name_b}?"

        return NarrativeExample(
            problem_text=(
                f"{name_a} quiere {desire}, pero {name_b} se interpone en su camino "
                f"a través de {conf_type}. "
                f"{intensify_event}. "
                f"Finalmente, {resolution_event}: {res_desc}."
            ),
            graph=g,
            answer=f"El conflicto se resuelve mediante {res_type}: {res_desc}.",
            complexity_level=2,
            answer_type=NarrativeAnswerType.CONFLICT_RESOLUTION,
            metadata={
                "protagonist": name_a,
                "antagonist":  name_b,
                "conflict":    conf_type,
                "resolution":  res_type,
                "required_node_types": ["character", "conflict", "event", "resolution"],
                "required_relations":  ["motivates", "conflicts_with", "intensifies", "resolves"],
                "arc_checks": {
                    "has_conflict_before_resolution": True,
                    "has_motivation_before_conflict":  True,
                },
            },
        )

    def _internal_conflict(self, rng: random.Random) -> NarrativeExample:
        """
        CHARACTER --MOTIVATES--> EMOTION --CONFLICTS_WITH--> CONFLICT --RESOLVES--> RESOLUTION
        """
        name = rng.choice(_CHARACTERS)
        desire = rng.choice(_DESIRES)
        emo_val, emo_desc = rng.choice(_EMOTIONS)
        conf_type = rng.choice(_CONFLICT_TYPES)
        res_type, res_desc = rng.choice(_RESOLUTION_TYPES)

        g = CausalGraph()
        char   = _narr_node("cha", f"{name}: quiere {desire}",    NarrativeNodeType.CHARACTER)
        emot   = _narr_node("em1", f"{emo_val}: {emo_desc}",      NarrativeNodeType.EMOTION)
        conf   = _narr_node("cf1", conf_type,                     NarrativeNodeType.CONFLICT)
        res    = _narr_node("res", f"resolución: {res_desc}",     NarrativeNodeType.RESOLUTION)
        g.add_node(char).add_node(emot).add_node(conf).add_node(res)
        g.add_edge(_narr_edge("cha", "em1", NarrativeRelation.MOTIVATES))
        g.add_edge(_narr_edge("em1", "cf1", NarrativeRelation.CONFLICTS_WITH))
        g.add_edge(_narr_edge("res", "cf1", NarrativeRelation.RESOLVES))
        g.root_question = f"¿Qué motiva a {name} en este conflicto interno?"

        return NarrativeExample(
            problem_text=(
                f"{name} quiere {desire}. "
                f"Sin embargo, {emo_desc} genera {conf_type}. "
                f"Al final: {res_desc}."
            ),
            graph=g,
            answer=f"{name} está motivado/a por el deseo de {desire}, pero {emo_desc} lo/la lleva al conflicto.",
            complexity_level=2,
            answer_type=NarrativeAnswerType.MOTIVATION,
            metadata={
                "character":  name,
                "emotion":    emo_val,
                "conflict":   conf_type,
                "resolution": res_type,
                "required_node_types": ["character", "emotion", "conflict", "resolution"],
                "required_relations":  ["motivates", "conflicts_with", "resolves"],
                "arc_checks": {
                    "has_conflict_before_resolution": True,
                    "has_motivation_before_conflict":  True,
                },
            },
        )

    def _rival_resolution(self, rng: random.Random) -> NarrativeExample:
        """
        char_a --CONFLICTS_WITH--> char_b
        char_a --MOTIVATES--> cf1
        cf1 <--INTENSIFIES-- ev_int1
        cf1 <--INTENSIFIES-- ev_int2
        res --RESOLVES--> cf1
        """
        names = rng.sample(_CHARACTERS, 2)
        name_a, name_b = names[0], names[1]
        desire   = rng.choice(_DESIRES)
        conf_type= rng.choice(_CONFLICT_TYPES)
        res_type, res_desc = rng.choice(_RESOLUTION_TYPES)

        int1 = rng.choice([
            "un primer enfrentamiento que eleva la tensión",
            "una traición que lo complica todo",
            "un malentendido que agrava la situación",
        ])
        int2 = rng.choice([
            "la crisis llega a un punto sin retorno",
            "un tercero involuntario lo empeora",
            "el tiempo presiona y la tensión se dispara",
        ])

        g = CausalGraph()
        cha    = _narr_node("cha", f"{name_a}: {desire}",        NarrativeNodeType.CHARACTER)
        chb    = _narr_node("chb", f"{name_b}: se opone",        NarrativeNodeType.CHARACTER)
        cf1    = _narr_node("cf1", conf_type,                    NarrativeNodeType.CONFLICT)
        ei1    = _narr_node("ei1", int1,                         NarrativeNodeType.EVENT)
        ei2    = _narr_node("ei2", int2,                         NarrativeNodeType.EVENT)
        res    = _narr_node("res", f"resolución: {res_desc}",    NarrativeNodeType.RESOLUTION)
        g.add_node(cha).add_node(chb).add_node(cf1).add_node(ei1).add_node(ei2).add_node(res)
        g.add_edge(_narr_edge("cha", "cf1", NarrativeRelation.MOTIVATES))
        g.add_edge(_narr_edge("cha", "chb", NarrativeRelation.CONFLICTS_WITH))
        g.add_edge(_narr_edge("ei1", "cf1", NarrativeRelation.INTENSIFIES))
        g.add_edge(_narr_edge("ei2", "cf1", NarrativeRelation.INTENSIFIES))
        g.add_edge(_narr_edge("res", "cf1", NarrativeRelation.RESOLVES))
        g.root_question = f"¿Cómo se resuelve el conflicto entre {name_a} y {name_b}?"

        return NarrativeExample(
            problem_text=(
                f"{name_a} busca {desire}, pero esto genera {conf_type} con {name_b}. "
                f"{int1}. {int2}. "
                f"Finalmente: {res_desc}."
            ),
            graph=g,
            answer=f"El conflicto se resuelve mediante {res_type}: {res_desc}.",
            complexity_level=2,
            answer_type=NarrativeAnswerType.CONFLICT_RESOLUTION,
            metadata={
                "protagonist": name_a,
                "antagonist":  name_b,
                "conflict":    conf_type,
                "resolution":  res_type,
                "intensification_count": 2,
                "required_node_types": ["character", "conflict", "event", "resolution"],
                "required_relations":  ["motivates", "conflicts_with", "intensifies", "resolves"],
                "arc_checks": {
                    "has_conflict_before_resolution":    True,
                    "has_motivation_before_conflict":     True,
                    "has_intensification_before_resolution": True,
                    "intensification_count": 2,
                },
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 3 — ARCO CON SUBVERSIÓN/SIMBOLISMO (5-8 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level3Generator:
    """
    Nivel 3: Arco complejo con subversión, simbolismo o doble conflicto.

    Subtypes:
      subverted_expectation: FORESHADOWS → CONFLICT → SUBVERTS expectativa
      symbolic_arc:          CHARACTER + SYMBOL --SYMBOLIZES--> THEME → RESOLUTION
      parallel_conflict:     dos CHARACTERs con PARALLELS en sus conflictos
      double_arc:            dos CONFLICTs que convergen en una RESOLUTION
    """

    _SUBTYPES = ["subverted_expectation", "symbolic_arc", "parallel_conflict", "double_arc"]

    def generate(self, rng: random.Random) -> NarrativeExample:
        sub = rng.choice(self._SUBTYPES)
        if sub == "subverted_expectation":
            return self._subverted_expectation(rng)
        elif sub == "symbolic_arc":
            return self._symbolic_arc(rng)
        elif sub == "parallel_conflict":
            return self._parallel_conflict(rng)
        else:
            return self._double_arc(rng)

    def _subverted_expectation(self, rng: random.Random) -> NarrativeExample:
        """
        SETTING --FORESHADOWS--> EVENT_setup
        CHARACTER --MOTIVATES--> CONFLICT
        CONFLICT --INTENSIFIES (via ei)--> CONFLICT
        EVENT_twist --SUBVERTS--> EVENT_setup
        RESOLUTION --RESOLVES--> CONFLICT
        """
        name         = rng.choice(_CHARACTERS)
        desire       = rng.choice(_DESIRES)
        set_desc, _  = rng.choice(_SETTINGS)
        conf_type    = rng.choice(_CONFLICT_TYPES)
        twist_text, twist_desc = rng.choice(_TWISTS)
        res_type, res_desc = rng.choice(_RESOLUTION_TYPES)

        setup_event = rng.choice([
            f"todo parecía indicar que {name} lograría {desire}",
            f"las señales apuntaban hacia un final esperado",
            f"la situación parecía controlada para {name}",
        ])
        intensify = rng.choice([
            "la tensión alcanza su punto álgido",
            "el conflicto parece irresoluble",
            "todo se complica inesperadamente",
        ])

        g = CausalGraph()
        sett     = _narr_node("set1", set_desc,            NarrativeNodeType.SETTING)
        char     = _narr_node("cha",  f"{name}: {desire}", NarrativeNodeType.CHARACTER)
        ev_setup = _narr_node("evs",  setup_event,         NarrativeNodeType.EVENT)
        cf1      = _narr_node("cf1",  conf_type,           NarrativeNodeType.CONFLICT)
        ev_int   = _narr_node("ei1",  intensify,           NarrativeNodeType.EVENT)
        ev_twist = _narr_node("etw",  twist_text,          NarrativeNodeType.EVENT)
        res      = _narr_node("res",  res_desc,            NarrativeNodeType.RESOLUTION)
        g.add_node(sett).add_node(char).add_node(ev_setup).add_node(cf1)
        g.add_node(ev_int).add_node(ev_twist).add_node(res)
        g.add_edge(_narr_edge("set1", "evs", NarrativeRelation.FORESHADOWS))
        g.add_edge(_narr_edge("cha",  "cf1", NarrativeRelation.MOTIVATES))
        g.add_edge(_narr_edge("ei1",  "cf1", NarrativeRelation.INTENSIFIES))
        g.add_edge(_narr_edge("etw",  "evs", NarrativeRelation.SUBVERTS))
        g.add_edge(_narr_edge("res",  "cf1", NarrativeRelation.RESOLVES))
        g.root_question = f"¿Cuál es el giro narrativo en la historia de {name}?"

        return NarrativeExample(
            problem_text=(
                f"En {set_desc}, {name} busca {desire}. "
                f"{setup_event}. "
                f"Pero {intensify}. "
                f"El giro: {twist_desc}. "
                f"Finalmente: {res_desc}."
            ),
            graph=g,
            answer=f"El giro narrativo es que {twist_desc}.",
            complexity_level=3,
            answer_type=NarrativeAnswerType.TWIST_IDENTIFICATION,
            metadata={
                "character":  name,
                "twist":      twist_text,
                "twist_desc": twist_desc,
                "resolution": res_type,
                "required_node_types": ["setting", "character", "event", "conflict", "resolution"],
                "required_relations":  ["foreshadows", "motivates", "intensifies", "subverts", "resolves"],
                "arc_checks": {
                    "has_foreshadowing_before_subversion": True,
                    "has_conflict_before_resolution":       True,
                    "has_intensification_before_resolution": True,
                    "has_subversion": True,
                },
            },
        )

    def _symbolic_arc(self, rng: random.Random) -> NarrativeExample:
        """
        SYMBOL --SYMBOLIZES--> THEME
        CHARACTER --MOTIVATES--> EVENT
        EVENT --DEVELOPS_INTO--> CONFLICT
        THEME --PARALLELS--> CONFLICT
        RESOLUTION --RESOLVES--> CONFLICT
        """
        name          = rng.choice(_CHARACTERS)
        desire        = rng.choice(_DESIRES)
        sym_text, sym_meaning = rng.choice(_SYMBOLS)
        theme_name, theme_desc = rng.choice(_THEMES)
        conf_type     = rng.choice(_CONFLICT_TYPES)
        res_type, res_desc = rng.choice(_RESOLUTION_TYPES)

        event_text = rng.choice([
            f"{name} encuentra {sym_text}",
            f"{sym_text} aparece en el momento más inesperado",
            f"{name} no puede ignorar {sym_text}",
        ])

        g = CausalGraph()
        sym   = _narr_node("sym",  sym_text,                      NarrativeNodeType.SYMBOL)
        thm   = _narr_node("thm",  f"tema: {theme_name}",         NarrativeNodeType.THEME)
        char  = _narr_node("cha",  f"{name}: {desire}",           NarrativeNodeType.CHARACTER)
        event = _narr_node("ev1",  event_text,                    NarrativeNodeType.EVENT)
        cf1   = _narr_node("cf1",  conf_type,                     NarrativeNodeType.CONFLICT)
        res   = _narr_node("res",  res_desc,                      NarrativeNodeType.RESOLUTION)
        g.add_node(sym).add_node(thm).add_node(char).add_node(event)
        g.add_node(cf1).add_node(res)
        g.add_edge(_narr_edge("sym",  "thm", NarrativeRelation.SYMBOLIZES))
        g.add_edge(_narr_edge("cha",  "ev1", NarrativeRelation.MOTIVATES))
        g.add_edge(_narr_edge("ev1",  "cf1", NarrativeRelation.DEVELOPS_INTO))
        g.add_edge(_narr_edge("thm",  "cf1", NarrativeRelation.PARALLELS))
        g.add_edge(_narr_edge("res",  "cf1", NarrativeRelation.RESOLVES))
        g.root_question = f"¿Qué simboliza {sym_text} en la historia de {name}?"

        return NarrativeExample(
            problem_text=(
                f"En la historia de {name}, {sym_text} aparece como símbolo de {sym_meaning}. "
                f"Este símbolo refleja el tema de {theme_name}: {theme_desc}. "
                f"{event_text} desencadena {conf_type}. "
                f"Finalmente: {res_desc}."
            ),
            graph=g,
            answer=f"{sym_text} simboliza {sym_meaning}, que refleja el tema central de {theme_name}.",
            complexity_level=3,
            answer_type=NarrativeAnswerType.SYMBOL_MEANING,
            metadata={
                "character":    name,
                "symbol":       sym_text,
                "symbol_meaning": sym_meaning,
                "theme":        theme_name,
                "resolution":   res_type,
                "required_node_types": ["symbol", "theme", "character", "event", "conflict", "resolution"],
                "required_relations":  ["symbolizes", "motivates", "develops_into", "parallels", "resolves"],
                "arc_checks": {
                    "has_symbol":                   True,
                    "symbol_before_resolution":     True,
                    "has_conflict_before_resolution": True,
                    "has_theme":                    True,
                },
            },
        )

    def _parallel_conflict(self, rng: random.Random) -> NarrativeExample:
        """
        Dos CHARACTERs con conflictos paralelos que se reflejan entre sí.
        char_a --MOTIVATES--> cf_a
        char_b --MOTIVATES--> cf_b
        cf_a --PARALLELS--> cf_b
        ev_int --INTENSIFIES--> cf_a
        res --RESOLVES--> cf_a, res --RESOLVES--> cf_b
        """
        names   = rng.sample(_CHARACTERS, 2)
        name_a, name_b = names[0], names[1]
        desire_a = rng.choice(_DESIRES)
        desire_b = rng.choice(_DESIRES)
        conf_types = rng.sample(_CONFLICT_TYPES, 2)
        res_type, res_desc = rng.choice(_RESOLUTION_TYPES)

        intensify = rng.choice([
            "sus caminos se cruzan y los conflictos se fusionan",
            "la coincidencia de sus crisis amplifica ambas tensiones",
            "un mismo evento los afecta simultáneamente",
        ])

        g = CausalGraph()
        cha   = _narr_node("cha",  f"{name_a}: {desire_a}",    NarrativeNodeType.CHARACTER)
        chb   = _narr_node("chb",  f"{name_b}: {desire_b}",    NarrativeNodeType.CHARACTER)
        cfa   = _narr_node("cfa",  conf_types[0],              NarrativeNodeType.CONFLICT)
        cfb   = _narr_node("cfb",  conf_types[1],              NarrativeNodeType.CONFLICT)
        ei1   = _narr_node("ei1",  intensify,                  NarrativeNodeType.EVENT)
        res   = _narr_node("res",  res_desc,                   NarrativeNodeType.RESOLUTION)
        g.add_node(cha).add_node(chb).add_node(cfa).add_node(cfb).add_node(ei1).add_node(res)
        g.add_edge(_narr_edge("cha", "cfa", NarrativeRelation.MOTIVATES))
        g.add_edge(_narr_edge("chb", "cfb", NarrativeRelation.MOTIVATES))
        g.add_edge(_narr_edge("cfa", "cfb", NarrativeRelation.PARALLELS))
        g.add_edge(_narr_edge("ei1", "cfa", NarrativeRelation.INTENSIFIES))
        g.add_edge(_narr_edge("res", "cfa", NarrativeRelation.RESOLVES))
        g.add_edge(_narr_edge("res", "cfb", NarrativeRelation.RESOLVES))
        g.root_question = f"¿Qué conecta los conflictos de {name_a} y {name_b}?"

        return NarrativeExample(
            problem_text=(
                f"{name_a} lucha con {conf_types[0]} mientras {name_b} enfrenta {conf_types[1]}. "
                f"Sus arcos son paralelos: {intensify}. "
                f"Ambos encuentran resolución mediante {res_type}."
            ),
            graph=g,
            answer=(
                f"Los conflictos de {name_a} y {name_b} son paralelos: "
                f"ambos reflejan {conf_types[0]} y {conf_types[1]} respectivamente, "
                f"y se resuelven juntos mediante {res_type}."
            ),
            complexity_level=3,
            answer_type=NarrativeAnswerType.THEME_IDENTIFICATION,
            metadata={
                "character_a":   name_a,
                "character_b":   name_b,
                "conflict_a":    conf_types[0],
                "conflict_b":    conf_types[1],
                "resolution":    res_type,
                "required_node_types": ["character", "conflict", "event", "resolution"],
                "required_relations":  ["motivates", "parallels", "intensifies", "resolves"],
                "arc_checks": {
                    "has_parallel_conflicts":            True,
                    "has_conflict_before_resolution":     True,
                    "has_intensification_before_resolution": True,
                    "resolution_closes_both_conflicts":   True,
                },
            },
        )

    def _double_arc(self, rng: random.Random) -> NarrativeExample:
        """
        Dos conflictos que convergen en una sola resolución.
        char --MOTIVATES--> cf1
        char --MOTIVATES--> cf2
        cf1 --DEVELOPS_INTO--> cf2  (el segundo empeora al primero)
        ev --INTENSIFIES--> cf2
        res --RESOLVES--> cf1
        res --RESOLVES--> cf2
        SYMBOL --SYMBOLIZES--> THEME (dimensión simbólica)
        """
        name        = rng.choice(_CHARACTERS)
        desire      = rng.choice(_DESIRES)
        conf_types  = rng.sample(_CONFLICT_TYPES, 2)
        sym_text, sym_meaning = rng.choice(_SYMBOLS)
        theme_name, theme_desc = rng.choice(_THEMES)
        res_type, res_desc = rng.choice(_RESOLUTION_TYPES)

        intensify = rng.choice([
            f"el segundo conflicto se agrava por el primero",
            f"la presión de ambos frentes es insostenible",
            f"resolver uno parece imposible sin afectar al otro",
        ])

        g = CausalGraph()
        char  = _narr_node("cha",  f"{name}: {desire}",         NarrativeNodeType.CHARACTER)
        cf1   = _narr_node("cf1",  conf_types[0],               NarrativeNodeType.CONFLICT)
        cf2   = _narr_node("cf2",  conf_types[1],               NarrativeNodeType.CONFLICT)
        sym   = _narr_node("sym",  sym_text,                    NarrativeNodeType.SYMBOL)
        thm   = _narr_node("thm",  f"tema: {theme_name}",       NarrativeNodeType.THEME)
        ei1   = _narr_node("ei1",  intensify,                   NarrativeNodeType.EVENT)
        res   = _narr_node("res",  res_desc,                    NarrativeNodeType.RESOLUTION)
        g.add_node(char).add_node(cf1).add_node(cf2).add_node(sym)
        g.add_node(thm).add_node(ei1).add_node(res)
        g.add_edge(_narr_edge("cha", "cf1", NarrativeRelation.MOTIVATES))
        g.add_edge(_narr_edge("cha", "cf2", NarrativeRelation.MOTIVATES))
        g.add_edge(_narr_edge("cf1", "cf2", NarrativeRelation.DEVELOPS_INTO))
        g.add_edge(_narr_edge("sym", "thm", NarrativeRelation.SYMBOLIZES))
        g.add_edge(_narr_edge("ei1", "cf2", NarrativeRelation.INTENSIFIES))
        g.add_edge(_narr_edge("res", "cf1", NarrativeRelation.RESOLVES))
        g.add_edge(_narr_edge("res", "cf2", NarrativeRelation.RESOLVES))
        g.root_question = f"¿Cómo se resuelven los dos conflictos de {name}?"

        return NarrativeExample(
            problem_text=(
                f"{name} enfrenta simultáneamente {conf_types[0]} y {conf_types[1]}. "
                f"El primero alimenta al segundo: {intensify}. "
                f"{sym_text} simboliza {sym_meaning}, reflejando el tema de {theme_name}. "
                f"Al final, ambos conflictos se cierran mediante {res_type}: {res_desc}."
            ),
            graph=g,
            answer=(
                f"Los dos conflictos de {name} se resuelven juntos mediante {res_type}. "
                f"{res_desc}. El símbolo de {sym_text} ({sym_meaning}) "
                f"encarna el tema de {theme_name}."
            ),
            complexity_level=3,
            answer_type=NarrativeAnswerType.CONFLICT_RESOLUTION,
            metadata={
                "character":  name,
                "conflict_1": conf_types[0],
                "conflict_2": conf_types[1],
                "symbol":     sym_text,
                "theme":      theme_name,
                "resolution": res_type,
                "required_node_types": ["character", "conflict", "symbol", "theme", "event", "resolution"],
                "required_relations":  ["motivates", "develops_into", "symbolizes", "intensifies", "resolves"],
                "arc_checks": {
                    "has_double_conflict":                True,
                    "has_symbol":                         True,
                    "has_conflict_before_resolution":     True,
                    "has_intensification_before_resolution": True,
                    "resolution_closes_both_conflicts":   True,
                },
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN DE COHERENCIA NARRATIVA
# ─────────────────────────────────────────────────────────────────────────────

def verify_narrative_example(ex: NarrativeExample) -> NarrativeVerificationResult:
    """
    Verifica que el arco narrativo del ejemplo es estructuralmente coherente.

    Checks:
      1. required_node_types están presentes en el grafo
      2. required_relations están presentes en las aristas
      3. arc_checks específicos por nivel/subtipo:
         - has_motivation_before_action: hay un nodo CHARACTER con MOTIVATES
         - has_conflict_before_resolution: hay CONFLICT y RESOLUTION, con arista RESOLVES
         - has_foreshadowing_before_subversion: hay FORESHADOWS y SUBVERTS
         - has_intensification_before_resolution: hay INTENSIFIES antes de RESOLVES
    """
    graph    = ex.graph
    meta     = ex.metadata
    arc      = meta.get("arc_checks", {})

    node_types = _node_types_in_graph(graph)
    relations  = _relations_in_graph(graph)

    # ── Check 1: required_node_types ──────────────────────────────────────
    required_ntypes = meta.get("required_node_types", [])
    missing_types = [nt for nt in required_ntypes if nt not in node_types]
    if missing_types:
        return NarrativeVerificationResult(
            False,
            f"Missing required node types: {missing_types}",
            {"present": list(node_types), "required": required_ntypes},
        )

    # ── Check 2: required_relations ───────────────────────────────────────
    required_rels = meta.get("required_relations", [])
    missing_rels = [r for r in required_rels if r not in relations]
    if missing_rels:
        return NarrativeVerificationResult(
            False,
            f"Missing required relations: {missing_rels}",
            {"present": list(relations), "required": required_rels},
        )

    # ── Check 3: has_conflict_before_resolution ───────────────────────────
    if arc.get("has_conflict_before_resolution"):
        # Verificar que existe al menos un CONFLICT y un RESOLUTION y la arista RESOLVES
        if "conflict" not in node_types:
            return NarrativeVerificationResult(
                False, "Arc requires CONFLICT node type but none found",
                {"node_types": list(node_types)},
            )
        if "resolution" not in node_types:
            return NarrativeVerificationResult(
                False, "Arc requires RESOLUTION node type but none found",
                {"node_types": list(node_types)},
            )
        if "resolves" not in relations:
            return NarrativeVerificationResult(
                False, "Arc requires RESOLVES relation but none found",
                {"relations": list(relations)},
            )

    # ── Check 4: has_motivation_before_conflict ───────────────────────────
    if arc.get("has_motivation_before_conflict"):
        if "motivates" not in relations:
            return NarrativeVerificationResult(
                False, "Arc requires MOTIVATES relation before conflict",
                {"relations": list(relations)},
            )

    # ── Check 5: has_intensification_before_resolution ────────────────────
    if arc.get("has_intensification_before_resolution"):
        if "intensifies" not in relations:
            return NarrativeVerificationResult(
                False, "Arc requires INTENSIFIES relation before resolution",
                {"relations": list(relations)},
            )

    # ── Check 6: has_foreshadowing_before_subversion ──────────────────────
    if arc.get("has_foreshadowing_before_subversion"):
        if "foreshadows" not in relations:
            return NarrativeVerificationResult(
                False, "Arc requires FORESHADOWS before SUBVERTS",
                {"relations": list(relations)},
            )
        if "subverts" not in relations:
            return NarrativeVerificationResult(
                False, "Arc requires SUBVERTS but none found",
                {"relations": list(relations)},
            )

    # ── Check 7: has_subversion ───────────────────────────────────────────
    if arc.get("has_subversion"):
        if "subverts" not in relations:
            return NarrativeVerificationResult(
                False, "Arc requires SUBVERTS relation",
                {"relations": list(relations)},
            )

    # ── Check 8: has_symbol ───────────────────────────────────────────────
    if arc.get("has_symbol"):
        if "symbol" not in node_types:
            return NarrativeVerificationResult(
                False, "Arc requires SYMBOL node type",
                {"node_types": list(node_types)},
            )
        if "symbolizes" not in relations:
            return NarrativeVerificationResult(
                False, "Arc requires SYMBOLIZES relation",
                {"relations": list(relations)},
            )

    # ── Check 9: has_theme ────────────────────────────────────────────────
    if arc.get("has_theme"):
        if "theme" not in node_types:
            return NarrativeVerificationResult(
                False, "Arc requires THEME node type",
                {"node_types": list(node_types)},
            )

    # ── Check 10: resolution_closes_both_conflicts ────────────────────────
    if arc.get("resolution_closes_both_conflicts"):
        # Hay al menos 2 nodos CONFLICT
        conflict_nodes = [
            node.node_id for node in graph.nodes
            if node.node_type == NarrativeNodeType.CONFLICT
        ]
        if len(conflict_nodes) < 2:
            return NarrativeVerificationResult(
                False,
                f"Arc requires ≥2 CONFLICT nodes, found {len(conflict_nodes)}",
                {"conflict_nodes": conflict_nodes},
            )
        # Hay al menos 2 aristas RESOLVES
        resolve_edges = [
            e for e in graph.edges
            if e.relation == NarrativeRelation.RESOLVES
        ]
        if len(resolve_edges) < 2:
            return NarrativeVerificationResult(
                False,
                f"Arc requires ≥2 RESOLVES edges for dual conflict, found {len(resolve_edges)}",
                {"resolve_edge_count": len(resolve_edges)},
            )

    # ── Check 11: intensification_count ──────────────────────────────────
    expected_count = arc.get("intensification_count")
    if expected_count is not None:
        intensify_edges = [
            e for e in graph.edges
            if e.relation == NarrativeRelation.INTENSIFIES
        ]
        if len(intensify_edges) < expected_count:
            return NarrativeVerificationResult(
                False,
                f"Arc requires {expected_count} INTENSIFIES edges, found {len(intensify_edges)}",
                {"intensify_count": len(intensify_edges)},
            )

    # ── Check 12: has_parallel_conflicts ─────────────────────────────────
    if arc.get("has_parallel_conflicts"):
        if "parallels" not in relations:
            return NarrativeVerificationResult(
                False, "Arc requires PARALLELS relation for parallel conflicts",
                {"relations": list(relations)},
            )

    # ── Check 13: has_double_conflict ─────────────────────────────────────
    if arc.get("has_double_conflict"):
        conflict_nodes = [
            node.node_id for node in graph.nodes
            if node.node_type == NarrativeNodeType.CONFLICT
        ]
        if len(conflict_nodes) < 2:
            return NarrativeVerificationResult(
                False,
                f"Arc requires ≥2 CONFLICT nodes (double arc), found {len(conflict_nodes)}",
                {"conflict_nodes": conflict_nodes},
            )

    return NarrativeVerificationResult(True, "All narrative arc checks passed", {
        "node_types": list(node_types),
        "relations":  list(relations),
        "level":      ex.complexity_level,
    })


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class NarrativeGraphGenerator:
    """
    Generador principal de ejemplos narrativos para MUSE.

    Uso:
        gen   = NarrativeGraphGenerator(seed=42)
        ex    = gen.generate(level=1)
        batch = gen.generate_batch(n=100, level_distribution={1: 0.4, 2: 0.4, 3: 0.2})
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng  = random.Random(seed)
        self._gens = {
            1: _Level1Generator(),
            2: _Level2Generator(),
            3: _Level3Generator(),
        }

    def generate(self, level: int = 1) -> NarrativeExample:
        """Genera un NarrativeExample del nivel indicado."""
        if level not in self._gens:
            raise ValueError(f"level must be 1, 2 or 3, got {level}")
        return self._gens[level].generate(self._rng)

    def generate_batch(
        self,
        n: int,
        level_distribution: Optional[Dict[int, float]] = None,
    ) -> List[NarrativeExample]:
        """
        Genera n ejemplos con distribución de niveles configurable.

        Args:
            n:                   número de ejemplos a generar
            level_distribution:  dict {nivel: peso} (se normaliza internamente)
                                 Default: {1: 0.4, 2: 0.4, 3: 0.2}
        """
        if level_distribution is None:
            level_distribution = {1: 0.4, 2: 0.4, 3: 0.2}
        levels  = list(level_distribution.keys())
        weights = list(level_distribution.values())
        return [
            self.generate(self._rng.choices(levels, weights=weights, k=1)[0])
            for _ in range(n)
        ]
