"""
synth/social_graph_gen.py — Generador de Grafos Sociales para EMPATHY
=======================================================================

Motor de datos sintéticos para el motor EMPATHY (razonamiento social/empático).
Genera situaciones sociales con estructura verificable:

  - Interacción simple:   persona A quiere X, persona B responde → resultado
  - Malentendido:         A cree algo falso sobre B → conflicto implícito
  - Conflicto de normas:  A viola norma que B valora → tensión social explícita

Tres niveles de complejidad:

  Nivel 1 — Interacción simple (3-4 nodos)
             "A quiere X → B da/responde → resultado"
             Verificar: toda acción tiene un nodo PERSON con WANTS/FEELS
             Subtypes: request_response, emotional_support, trust_exchange

  Nivel 2 — Malentendido (4-6 nodos)
             "A cree algo de B que no es verdad →
              A actúa en base a esa creencia → B se siente mal"
             Verificar: hay MISUNDERSTANDS con un BELIEF involucrado,
                        hay al menos 2 PERSONs con intenciones distintas
             Subtypes: false_belief, misread_intention, expectation_violation

  Nivel 3 — Conflicto de normas (5-8 nodos)
             "A viola una norma que B valora → conflicto →
              hay o no hay reparación (RECIPROCATES / EMPATHIZES)"
             Verificar: VIOLATES_NORM con NORM presente,
                        hay PERSON con WANTS opuesto a otro PERSON
             Subtypes: norm_violation, persuasion_attempt, cultural_clash, trust_betrayal

Contrato de cada SocialExample:
  - problem_text:       descripción de la situación social en lenguaje natural
  - graph:              CausalGraph con SocialNode/SocialEdge
  - answer:             respuesta a la pregunta social/empática
  - complexity_level:   1-3
  - answer_type:        SocialAnswerType
  - metadata:           parámetros para verify_social_example()
  - example_id:         UUID reproducible

Verificación de coherencia social:
  - every_action_has_person:     todo WANTS/FEELS/BELIEVES apunta desde un PERSON
  - conflict_has_opposing_wants: si hay conflicto, hay ≥2 PERSONs con WANTS distintos
  - misunderstanding_has_belief: MISUNDERSTANDS involucra un BELIEF
  - norm_violation_has_norm:     VIOLATES_NORM apunta a un nodo NORM

Uso básico:
    gen = SocialGraphGenerator(seed=42)
    ex  = gen.generate(level=1)
    res = verify_social_example(ex)
    assert res.passed

    batch = gen.generate_batch(n=100, level_distribution={1: 0.4, 2: 0.4, 3: 0.2})
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.graph import CausalGraph
from motors.empathy.relations import (
    SOCIAL_RELATIONS,
    SocialEdge,
    SocialNode,
    SocialNodeType,
    SocialRelation,
)


# ─────────────────────────────────────────────────────────────────────────────
# TIPOS DE RESPUESTA SOCIAL
# ─────────────────────────────────────────────────────────────────────────────

class SocialAnswerType(str, Enum):
    EMPATHIC_RESPONSE   = "empathic_response"    # ¿Qué respuesta empática dar?
    CONFLICT_DIAGNOSIS  = "conflict_diagnosis"   # ¿Por qué hay conflicto?
    MISUNDERSTANDING    = "misunderstanding"      # ¿Qué malentendió A?
    NORM_VIOLATION      = "norm_violation"        # ¿Qué norma se violó?
    INTENTION_INFERENCE = "intention_inference"   # ¿Qué intenta realmente A?
    REPAIR_STRATEGY     = "repair_strategy"       # ¿Cómo reparar la relación?


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO DE VERIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SocialVerificationResult:
    """
    Resultado de verify_social_example().

    passed:  True si el grafo tiene coherencia social verificable
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
        return f"SocialVerificationResult({status}: {self.reason})"


# ─────────────────────────────────────────────────────────────────────────────
# SOCIAL EXAMPLE — UNIDAD ATÓMICA DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SocialExample:
    """
    Unidad atómica de entrenamiento para EMPATHY.

    El grafo captura la dinámica social con SocialNode/SocialEdge.

    metadata siempre incluye las claves necesarias para verify_social_example():
      - required_node_types:  List[str] — tipos que deben aparecer en el grafo
      - required_relations:   List[str] — relaciones que deben aparecer en aristas
      - social_checks:        Dict — checks de coherencia específicos por nivel
    """
    problem_text:     str
    graph:            CausalGraph
    answer:           str
    complexity_level: int
    answer_type:      SocialAnswerType
    verifiable:       bool  = True
    metadata:         Dict  = field(default_factory=dict)
    example_id:       str   = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def __repr__(self) -> str:
        return (
            f"SocialExample(level={self.complexity_level}, "
            f"type={self.answer_type.value}, "
            f"nodes={len(self.graph)}, id={self.example_id})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — CONSTRUCCIÓN DE GRAFOS
# ─────────────────────────────────────────────────────────────────────────────

def _soc_node(
    nid: str, label: str, ntype: SocialNodeType, conf: float = 1.0
) -> SocialNode:
    return SocialNode(
        node_id=nid, label=label, node_type=ntype,
        confidence=conf, grounded=True
    )


def _soc_edge(
    src: str, tgt: str, rel: SocialRelation,
    strength: float = 1.0, conf: float = 1.0
) -> SocialEdge:
    return SocialEdge(
        source_id=src, target_id=tgt, relation=rel,
        strength=strength, confidence=conf
    )


def _node_types_in_graph(graph: CausalGraph) -> Set[str]:
    return {node.node_type.value for node in graph.nodes}


def _relations_in_graph(graph: CausalGraph) -> Set[str]:
    return {edge.relation.value for edge in graph.edges}


# ─────────────────────────────────────────────────────────────────────────────
# DATOS SOCIALES — NOMBRES, EMOCIONES, DESEOS, NORMAS, CONTEXTOS
# ─────────────────────────────────────────────────────────────────────────────

_NAMES = [
    "Ana", "Luis", "María", "Pedro", "Sofía", "Javier",
    "Claudia", "Roberto", "Valentina", "Diego", "Elena", "Marcos",
]

_WANTS = [
    "ser escuchado/a sin ser juzgado/a",
    "recibir ayuda con una tarea difícil",
    "que le pidan disculpas",
    "que respeten su espacio personal",
    "que le expliquen lo que pasó",
    "sentirse valorado/a en el grupo",
    "que cumplan una promesa",
    "privacidad en un asunto personal",
    "reconocimiento por su trabajo",
    "apoyo emocional en un momento difícil",
]

_BELIEFS = [
    ("que el otro está enojado con él/ella",        False),
    ("que la situación es su culpa",                False),
    ("que el otro lo/la ignoró a propósito",        False),
    ("que no le importa a nadie",                   False),
    ("que el otro entiende cómo se siente",         True),
    ("que la relación sigue siendo sólida",         True),
    ("que el otro actuó de buena fe",               True),
    ("que hay una solución posible",                True),
]

_EMOTIONS = [
    "frustración",
    "tristeza",
    "confusión",
    "alivio",
    "gratitud",
    "incomodidad",
    "sorpresa",
    "enojo",
    "esperanza",
    "vergüenza",
]

_NORMS = [
    "respetar los límites personales de los demás",
    "cumplir los compromisos asumidos",
    "no compartir información privada sin permiso",
    "pedir disculpas cuando se causa daño",
    "escuchar antes de responder",
    "tratar a todos con dignidad",
    "no interrumpir cuando alguien habla",
    "agradecer los gestos de ayuda",
    "no responsabilizar a otros por errores propios",
    "mantener la confidencialidad de lo compartido",
]

_CONTEXTS = [
    ("una conversación entre amigos cercanos",        "amistad"),
    ("una reunión de trabajo bajo presión",           "trabajo"),
    ("una discusión familiar en una cena",            "familia"),
    ("un intercambio por mensaje de texto",           "digital"),
    ("una situación pública con otras personas cerca","público"),
    ("un momento de crisis emocional",               "crisis"),
    ("una negociación entre colegas",                "colegas"),
    ("un reencuentro después de un conflicto previo", "reencuentro"),
]

_RELATIONSHIPS = [
    "amistad de larga data",
    "relación laboral formal",
    "vínculo familiar",
    "conocidos recientes",
    "pareja sentimental",
    "compañeros de estudio",
]

_REPAIR_STRATEGIES = [
    "pedir disculpas sinceras y explicar la intención real",
    "escuchar activamente sin interrumpir ni defenderse",
    "proponer una solución concreta al daño causado",
    "reconocer el error y preguntar qué necesita la otra persona",
    "dar espacio y retomar la conversación cuando ambos estén calmados",
    "buscar un mediador de confianza para facilitar el diálogo",
]


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 1 — INTERACCIÓN SIMPLE (3-4 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level1Generator:
    """
    Nivel 1: Interacción social directa, sin conflicto.

    Subtypes:
      request_response:   A --WANTS--> INT, A --FEELS--> EM, B --RECIPROCATES--> A
      emotional_support:  A --FEELS--> EM, B --EMPATHIZES--> A, resultado positivo
      trust_exchange:     A --TRUSTS--> B, B --RECIPROCATES--> A (vínculo positivo)
    """

    _SUBTYPES = ["request_response", "emotional_support", "trust_exchange"]

    def generate(self, rng: random.Random) -> SocialExample:
        sub = rng.choice(self._SUBTYPES)
        if sub == "request_response":
            return self._request_response(rng)
        elif sub == "emotional_support":
            return self._emotional_support(rng)
        else:
            return self._trust_exchange(rng)

    def _request_response(self, rng: random.Random) -> SocialExample:
        """
        per_a --WANTS--> int_a
        per_a --FEELS--> em_a
        per_b --RECIPROCATES--> per_a
        Pregunta: ¿Qué respuesta empática dar?
        """
        names = rng.sample(_NAMES, 2)
        name_a, name_b = names[0], names[1]
        want  = rng.choice(_WANTS)
        emo   = rng.choice(_EMOTIONS)
        ctx_desc, _ = rng.choice(_CONTEXTS)

        response = rng.choice([
            f"le pregunta a {name_a} cómo puede ayudar",
            f"valida los sentimientos de {name_a} sin juzgar",
            f"ofrece su apoyo concreto a {name_a}",
            f"escucha atentamente y responde con cuidado",
        ])

        g = CausalGraph()
        per_a = _soc_node("pa",  f"{name_a}",         SocialNodeType.PERSON)
        per_b = _soc_node("pb",  f"{name_b}",         SocialNodeType.PERSON)
        int_a = _soc_node("ia",  want,                 SocialNodeType.INTENTION)
        em_a  = _soc_node("ea",  emo,                  SocialNodeType.EMOTION)
        g.add_node(per_a).add_node(per_b).add_node(int_a).add_node(em_a)
        g.add_edge(_soc_edge("pa", "ia", SocialRelation.WANTS))
        g.add_edge(_soc_edge("pa", "ea", SocialRelation.FEELS))
        g.add_edge(_soc_edge("pb", "pa", SocialRelation.RECIPROCATES))
        g.root_question = f"¿Qué debería hacer {name_b} para responder empáticamente a {name_a}?"

        return SocialExample(
            problem_text=(
                f"En {ctx_desc}, {name_a} siente {emo} y quiere {want}. "
                f"{name_b} {response}."
            ),
            graph=g,
            answer=(
                f"{name_b} debería {response}, reconociendo que {name_a} "
                f"siente {emo} y quiere {want}."
            ),
            complexity_level=1,
            answer_type=SocialAnswerType.EMPATHIC_RESPONSE,
            metadata={
                "person_a": name_a, "person_b": name_b,
                "want": want, "emotion": emo,
                "required_node_types": ["person", "intention", "emotion"],
                "required_relations":  ["wants", "feels", "reciprocates"],
                "social_checks": {
                    "every_action_has_person": True,
                    "person_count": 2,
                },
            },
        )

    def _emotional_support(self, rng: random.Random) -> SocialExample:
        """
        per_a --FEELS--> em_a
        per_b --EMPATHIZES--> per_a
        per_b --WANTS--> int_b  (ayudar)
        Pregunta: ¿Qué intenta realmente B?
        """
        names = rng.sample(_NAMES, 2)
        name_a, name_b = names[0], names[1]
        emo   = rng.choice(_EMOTIONS)
        support_want = rng.choice([
            f"apoyar emocionalmente a {name_a}",
            f"que {name_a} sepa que no está solo/a",
            f"ayudar a {name_a} a procesar lo que siente",
            f"estar presente para {name_a} sin presionarlo/a",
        ])
        ctx_desc, _ = rng.choice(_CONTEXTS)

        g = CausalGraph()
        per_a = _soc_node("pa", f"{name_a}",   SocialNodeType.PERSON)
        per_b = _soc_node("pb", f"{name_b}",   SocialNodeType.PERSON)
        em_a  = _soc_node("ea", emo,            SocialNodeType.EMOTION)
        int_b = _soc_node("ib", support_want,   SocialNodeType.INTENTION)
        g.add_node(per_a).add_node(per_b).add_node(em_a).add_node(int_b)
        g.add_edge(_soc_edge("pa", "ea", SocialRelation.FEELS))
        g.add_edge(_soc_edge("pb", "pa", SocialRelation.EMPATHIZES))
        g.add_edge(_soc_edge("pb", "ib", SocialRelation.WANTS))
        g.root_question = f"¿Qué intenta realmente {name_b}?"

        return SocialExample(
            problem_text=(
                f"En {ctx_desc}, {name_a} siente {emo}. "
                f"{name_b} lo/la nota y quiere {support_want}."
            ),
            graph=g,
            answer=f"{name_b} intenta {support_want}. Su motivación es empática.",
            complexity_level=1,
            answer_type=SocialAnswerType.INTENTION_INFERENCE,
            metadata={
                "person_a": name_a, "person_b": name_b,
                "emotion": emo, "support_want": support_want,
                "required_node_types": ["person", "emotion", "intention"],
                "required_relations":  ["feels", "empathizes", "wants"],
                "social_checks": {
                    "every_action_has_person": True,
                    "person_count": 2,
                },
            },
        )

    def _trust_exchange(self, rng: random.Random) -> SocialExample:
        """
        per_a --TRUSTS--> per_b
        per_b --RECIPROCATES--> per_a
        rel   describes the relationship
        Pregunta: ¿Cómo reparar la relación?
        """
        names = rng.sample(_NAMES, 2)
        name_a, name_b = names[0], names[1]
        rel_type = rng.choice(_RELATIONSHIPS)
        ctx_desc, _ = rng.choice(_CONTEXTS)
        emo = rng.choice(_EMOTIONS)

        trust_act = rng.choice([
            f"comparte algo personal con {name_b}",
            f"confía en {name_b} con una responsabilidad importante",
            f"le pide consejo a {name_b} sobre algo íntimo",
        ])
        recip_act = rng.choice([
            f"responde con honestidad y cuidado",
            f"honra esa confianza con discreción",
            f"demuestra que merece esa confianza",
        ])

        g = CausalGraph()
        per_a = _soc_node("pa",  f"{name_a}",         SocialNodeType.PERSON)
        per_b = _soc_node("pb",  f"{name_b}",         SocialNodeType.PERSON)
        rel   = _soc_node("rel", rel_type,             SocialNodeType.RELATIONSHIP)
        em_a  = _soc_node("ea",  emo,                  SocialNodeType.EMOTION)
        g.add_node(per_a).add_node(per_b).add_node(rel).add_node(em_a)
        g.add_edge(_soc_edge("pa", "pb",  SocialRelation.TRUSTS))
        g.add_edge(_soc_edge("pb", "pa",  SocialRelation.RECIPROCATES))
        g.add_edge(_soc_edge("pa", "ea",  SocialRelation.FEELS))
        g.root_question = f"¿Cómo se mantiene la confianza entre {name_a} y {name_b}?"

        return SocialExample(
            problem_text=(
                f"En {ctx_desc}, entre {name_a} y {name_b} hay una {rel_type}. "
                f"{name_a} {trust_act}. {name_b} {recip_act}. "
                f"{name_a} siente {emo}."
            ),
            graph=g,
            answer=(
                f"La confianza se mantiene porque {name_a} {trust_act} "
                f"y {name_b} {recip_act}."
            ),
            complexity_level=1,
            answer_type=SocialAnswerType.EMPATHIC_RESPONSE,
            metadata={
                "person_a": name_a, "person_b": name_b,
                "relationship": rel_type, "emotion": emo,
                "required_node_types": ["person", "relationship", "emotion"],
                "required_relations":  ["trusts", "reciprocates", "feels"],
                "social_checks": {
                    "every_action_has_person": True,
                    "person_count": 2,
                },
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 2 — MALENTENDIDO (4-6 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level2Generator:
    """
    Nivel 2: Malentendido o violación de expectativa.

    Subtypes:
      false_belief:          A cree X (falso) sobre B → actúa mal → B se siente mal
      misread_intention:     A malinterpreta intención de B → MISUNDERSTANDS
      expectation_violation: A espera Y de B → B hace Z ≠ Y → conflicto
    """

    _SUBTYPES = ["false_belief", "misread_intention", "expectation_violation"]

    def generate(self, rng: random.Random) -> SocialExample:
        sub = rng.choice(self._SUBTYPES)
        if sub == "false_belief":
            return self._false_belief(rng)
        elif sub == "misread_intention":
            return self._misread_intention(rng)
        else:
            return self._expectation_violation(rng)

    def _false_belief(self, rng: random.Random) -> SocialExample:
        """
        per_a --BELIEVES--> bel_a (falsa)
        per_a --MISUNDERSTANDS--> per_b
        per_b --FEELS--> em_b (consecuencia)
        per_a --WANTS--> int_a
        """
        names = rng.sample(_NAMES, 2)
        name_a, name_b = names[0], names[1]
        belief_text, _ = rng.choice(_BELIEFS)  # ignoramos is_true, usamos creencia falsa
        false_belief = rng.choice([
            f"que {name_b} está ignorándolo/a a propósito",
            f"que {name_b} habló mal de él/ella",
            f"que a {name_b} no le importa la amistad",
            f"que {name_b} lo/la está evitando",
        ])
        emo_b = rng.choice(["tristeza", "frustración", "confusión", "sorpresa"])
        want_a = rng.choice(_WANTS)
        ctx_desc, _ = rng.choice(_CONTEXTS)

        real_reason = rng.choice([
            f"en realidad {name_b} estaba pasando por un momento difícil",
            f"la realidad es que {name_b} simplemente estaba muy ocupado/a",
            f"lo que ocurrió fue un malentendido sin intención alguna",
        ])

        g = CausalGraph()
        per_a = _soc_node("pa",  f"{name_a}",           SocialNodeType.PERSON)
        per_b = _soc_node("pb",  f"{name_b}",           SocialNodeType.PERSON)
        bel_a = _soc_node("ba",  false_belief,           SocialNodeType.BELIEF)
        em_b  = _soc_node("eb",  emo_b,                  SocialNodeType.EMOTION)
        int_a = _soc_node("ia",  want_a,                 SocialNodeType.INTENTION)
        g.add_node(per_a).add_node(per_b).add_node(bel_a).add_node(em_b).add_node(int_a)
        g.add_edge(_soc_edge("pa", "ba",  SocialRelation.BELIEVES))
        g.add_edge(_soc_edge("pa", "pb",  SocialRelation.MISUNDERSTANDS))
        g.add_edge(_soc_edge("pb", "eb",  SocialRelation.FEELS))
        g.add_edge(_soc_edge("pa", "ia",  SocialRelation.WANTS))
        g.root_question = f"¿Qué malentendió {name_a}?"

        return SocialExample(
            problem_text=(
                f"En {ctx_desc}, {name_a} cree {false_belief}. "
                f"Basándose en esto, {name_a} quiere {want_a}. "
                f"Pero {real_reason}. "
                f"{name_b} siente {emo_b} ante la reacción de {name_a}."
            ),
            graph=g,
            answer=(
                f"{name_a} malentendió: creyó {false_belief}, "
                f"pero {real_reason}."
            ),
            complexity_level=2,
            answer_type=SocialAnswerType.MISUNDERSTANDING,
            metadata={
                "person_a": name_a, "person_b": name_b,
                "false_belief": false_belief, "real_reason": real_reason,
                "emotion_b": emo_b,
                "required_node_types": ["person", "belief", "emotion", "intention"],
                "required_relations":  ["believes", "misunderstands", "feels", "wants"],
                "social_checks": {
                    "every_action_has_person":   True,
                    "misunderstanding_has_belief": True,
                    "person_count": 2,
                },
            },
        )

    def _misread_intention(self, rng: random.Random) -> SocialExample:
        """
        per_b --WANTS--> int_b_real  (intención real positiva)
        per_a --MISUNDERSTANDS--> per_b
        per_a --BELIEVES--> bel_a_wrong (creencia incorrecta sobre intención de B)
        per_a --FEELS--> em_a
        ctx contextualization
        """
        names = rng.sample(_NAMES, 2)
        name_a, name_b = names[0], names[1]
        real_intent = rng.choice([
            f"ayudar a {name_a} con buena intención",
            f"dar un consejo honesto sin herir",
            f"poner límites saludables",
            f"ser transparente sobre sus limitaciones",
        ])
        wrong_belief = rng.choice([
            f"que {name_b} quería dejarlo/la de lado",
            f"que {name_b} estaba siendo condescendiente",
            f"que {name_b} no lo/la considera capaz",
            f"que {name_b} estaba siendo indiferente",
        ])
        emo_a = rng.choice(["enojo", "tristeza", "confusión", "incomodidad"])
        ctx_desc, _ = rng.choice(_CONTEXTS)

        g = CausalGraph()
        per_a  = _soc_node("pa",  f"{name_a}",         SocialNodeType.PERSON)
        per_b  = _soc_node("pb",  f"{name_b}",         SocialNodeType.PERSON)
        int_b  = _soc_node("ib",  real_intent,          SocialNodeType.INTENTION)
        bel_a  = _soc_node("ba",  wrong_belief,         SocialNodeType.BELIEF)
        em_a   = _soc_node("ea",  emo_a,                SocialNodeType.EMOTION)
        g.add_node(per_a).add_node(per_b).add_node(int_b).add_node(bel_a).add_node(em_a)
        g.add_edge(_soc_edge("pb", "ib",  SocialRelation.WANTS))
        g.add_edge(_soc_edge("pa", "pb",  SocialRelation.MISUNDERSTANDS))
        g.add_edge(_soc_edge("pa", "ba",  SocialRelation.BELIEVES))
        g.add_edge(_soc_edge("pa", "ea",  SocialRelation.FEELS))
        g.root_question = f"¿Qué malinterpretó {name_a} sobre {name_b}?"

        return SocialExample(
            problem_text=(
                f"En {ctx_desc}, {name_b} quería {real_intent}. "
                f"Pero {name_a} interpretó que {wrong_belief}. "
                f"Como resultado, {name_a} siente {emo_a}."
            ),
            graph=g,
            answer=(
                f"{name_a} malinterpretó la intención de {name_b}: "
                f"creyó {wrong_belief}, cuando en realidad {name_b} quería {real_intent}."
            ),
            complexity_level=2,
            answer_type=SocialAnswerType.MISUNDERSTANDING,
            metadata={
                "person_a": name_a, "person_b": name_b,
                "real_intent": real_intent, "wrong_belief": wrong_belief,
                "emotion_a": emo_a,
                "required_node_types": ["person", "intention", "belief", "emotion"],
                "required_relations":  ["wants", "misunderstands", "believes", "feels"],
                "social_checks": {
                    "every_action_has_person":   True,
                    "misunderstanding_has_belief": True,
                    "person_count": 2,
                },
            },
        )

    def _expectation_violation(self, rng: random.Random) -> SocialExample:
        """
        per_a --EXPECTS--> exp_a
        per_b --WANTS--> int_b   (intención diferente a expectativa de A)
        per_a --FEELS--> em_a    (decepción)
        per_a --MISUNDERSTANDS--> per_b  (no entiende por qué B hizo lo que hizo)
        ctx
        """
        names = rng.sample(_NAMES, 2)
        name_a, name_b = names[0], names[1]
        expectation = rng.choice([
            f"que {name_b} estaría disponible cuando lo/la necesitara",
            f"que {name_b} cumpliría el compromiso acordado",
            f"que {name_b} le consultaría antes de decidir",
            f"que {name_b} mantendría la confidencialidad",
        ])
        actual_intent = rng.choice([
            f"priorizar su propio bienestar ese día",
            f"resolver otro compromiso urgente primero",
            f"actuar rápido para no perder la oportunidad",
            f"compartir la información creyendo que no era privada",
        ])
        emo_a = rng.choice(["decepción", "tristeza", "confusión", "enojo"])
        ctx_desc, _ = rng.choice(_CONTEXTS)

        g = CausalGraph()
        per_a = _soc_node("pa",  f"{name_a}",       SocialNodeType.PERSON)
        per_b = _soc_node("pb",  f"{name_b}",       SocialNodeType.PERSON)
        exp_a = _soc_node("ea",  expectation,        SocialNodeType.EXPECTATION)
        int_b = _soc_node("ib",  actual_intent,      SocialNodeType.INTENTION)
        em_a  = _soc_node("ema", emo_a,              SocialNodeType.EMOTION)
        g.add_node(per_a).add_node(per_b).add_node(exp_a).add_node(int_b).add_node(em_a)
        g.add_edge(_soc_edge("pa", "ea",  SocialRelation.EXPECTS))
        g.add_edge(_soc_edge("pb", "ib",  SocialRelation.WANTS))
        g.add_edge(_soc_edge("pa", "ema", SocialRelation.FEELS))
        g.add_edge(_soc_edge("pa", "pb",  SocialRelation.MISUNDERSTANDS))
        g.root_question = f"¿Por qué hay conflicto entre {name_a} y {name_b}?"

        return SocialExample(
            problem_text=(
                f"En {ctx_desc}, {name_a} esperaba {expectation}. "
                f"Sin embargo, {name_b} decidió {actual_intent}. "
                f"{name_a} siente {emo_a} y no entiende la decisión de {name_b}."
            ),
            graph=g,
            answer=(
                f"El conflicto surge porque {name_a} esperaba {expectation}, "
                f"pero {name_b} priorizó {actual_intent}. "
                f"La expectativa de {name_a} no coincidió con la intención de {name_b}."
            ),
            complexity_level=2,
            answer_type=SocialAnswerType.CONFLICT_DIAGNOSIS,
            metadata={
                "person_a": name_a, "person_b": name_b,
                "expectation": expectation, "actual_intent": actual_intent,
                "emotion_a": emo_a,
                "required_node_types": ["person", "expectation", "intention", "emotion"],
                "required_relations":  ["expects", "wants", "feels", "misunderstands"],
                "social_checks": {
                    "every_action_has_person":          True,
                    "conflict_has_opposing_intentions": False,  # A tiene EXPECTATION, no INTENTION
                    "misunderstanding_has_belief":      False,
                    "person_count": 2,
                },
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR NIVEL 3 — CONFLICTO DE NORMAS (5-8 NODOS)
# ─────────────────────────────────────────────────────────────────────────────

class _Level3Generator:
    """
    Nivel 3: Conflicto social complejo con normas, persuasión o traición de confianza.

    Subtypes:
      norm_violation:      A viola norma que B valora → conflicto → posible reparación
      persuasion_attempt:  A intenta convencer a B de algo → B resiste o cede
      cultural_clash:      normas distintas generan conflicto sin mala fe
      trust_betrayal:      A traiciona la confianza de B → reparación necesaria
    """

    _SUBTYPES = ["norm_violation", "persuasion_attempt", "cultural_clash", "trust_betrayal"]

    def generate(self, rng: random.Random) -> SocialExample:
        sub = rng.choice(self._SUBTYPES)
        if sub == "norm_violation":
            return self._norm_violation(rng)
        elif sub == "persuasion_attempt":
            return self._persuasion_attempt(rng)
        elif sub == "cultural_clash":
            return self._cultural_clash(rng)
        else:
            return self._trust_betrayal(rng)

    def _norm_violation(self, rng: random.Random) -> SocialExample:
        """
        ctx contexto
        per_a --WANTS--> int_a
        per_a --VIOLATES_NORM--> norm
        per_b --BELIEVES--> bel_b (que la norma es importante)
        per_b --FEELS--> em_b
        per_a --WANTS--> int_repair (reparación)
        per_a --RECIPROCATES--> per_b
        """
        names = rng.sample(_NAMES, 2)
        name_a, name_b = names[0], names[1]
        norm  = rng.choice(_NORMS)
        want_a = rng.choice(_WANTS)
        emo_b  = rng.choice(["enojo", "tristeza", "decepción", "incomodidad"])
        ctx_desc, _ = rng.choice(_CONTEXTS)
        repair = rng.choice(_REPAIR_STRATEGIES)

        violation_desc = rng.choice([
            f"actúa sin considerar los límites de {name_b}",
            f"comparte algo que {name_b} le pidió que guardara",
            f"no cumple el compromiso que tenía con {name_b}",
            f"interrumpe a {name_b} repetidamente sin darse cuenta",
        ])

        g = CausalGraph()
        ctx   = _soc_node("ctx", ctx_desc,                    SocialNodeType.CONTEXT)
        per_a = _soc_node("pa",  f"{name_a}",                 SocialNodeType.PERSON)
        per_b = _soc_node("pb",  f"{name_b}",                 SocialNodeType.PERSON)
        int_a = _soc_node("ia",  want_a,                      SocialNodeType.INTENTION)
        norm_ = _soc_node("nm",  norm,                        SocialNodeType.NORM)
        bel_b = _soc_node("bb",  f"que es importante {norm}", SocialNodeType.BELIEF)
        em_b  = _soc_node("eb",  emo_b,                       SocialNodeType.EMOTION)
        g.add_node(ctx).add_node(per_a).add_node(per_b).add_node(int_a)
        g.add_node(norm_).add_node(bel_b).add_node(em_b)
        g.add_edge(_soc_edge("pa", "ia",  SocialRelation.WANTS))
        g.add_edge(_soc_edge("pa", "nm",  SocialRelation.VIOLATES_NORM))
        g.add_edge(_soc_edge("pb", "bb",  SocialRelation.BELIEVES))
        g.add_edge(_soc_edge("pb", "eb",  SocialRelation.FEELS))
        g.add_edge(_soc_edge("pa", "pb",  SocialRelation.RECIPROCATES))
        g.root_question = f"¿Qué norma violó {name_a} y cómo puede repararlo?"

        return SocialExample(
            problem_text=(
                f"En {ctx_desc}, {name_a} quiere {want_a} y {violation_desc}. "
                f"Esto viola la norma de {norm}, que {name_b} valora. "
                f"{name_b} siente {emo_b}."
            ),
            graph=g,
            answer=(
                f"{name_a} violó la norma de {norm}. "
                f"Para reparar, debería: {repair}."
            ),
            complexity_level=3,
            answer_type=SocialAnswerType.NORM_VIOLATION,
            metadata={
                "person_a": name_a, "person_b": name_b,
                "norm": norm, "emotion_b": emo_b, "repair": repair,
                "required_node_types": ["context", "person", "intention", "norm", "belief", "emotion"],
                "required_relations":  ["wants", "violates_norm", "believes", "feels", "reciprocates"],
                "social_checks": {
                    "every_action_has_person": True,
                    "norm_violation_has_norm":  True,
                    "conflict_has_opposing_wants": False,
                    "person_count": 2,
                },
            },
        )

    def _persuasion_attempt(self, rng: random.Random) -> SocialExample:
        """
        per_a --WANTS--> int_a
        per_a --BELIEVES--> bel_a
        per_a --PERSUADES--> per_b
        per_b --BELIEVES--> bel_b (diferente a bel_a)
        per_b --WANTS--> int_b (distinto de int_a)
        per_b --FEELS--> em_b
        ctx
        """
        names = rng.sample(_NAMES, 2)
        name_a, name_b = names[0], names[1]
        want_a = rng.choice(_WANTS)
        want_b = rng.choice(_WANTS)
        while want_b == want_a:
            want_b = rng.choice(_WANTS)
        bel_a_text = rng.choice([
            "que su propuesta es la mejor opción",
            "que el otro está equivocado en su postura",
            "que un cambio de perspectiva beneficiaría a ambos",
            "que la solución es más simple de lo que parece",
        ])
        bel_b_text = rng.choice([
            "que su posición es correcta y fundamentada",
            "que le están presionando injustamente",
            "que hay información que el otro no está considerando",
            "que la propuesta no tiene en cuenta sus necesidades",
        ])
        emo_b  = rng.choice(["incomodidad", "frustración", "confusión", "sorpresa"])
        ctx_desc, _ = rng.choice(_CONTEXTS)

        g = CausalGraph()
        ctx   = _soc_node("ctx", ctx_desc,   SocialNodeType.CONTEXT)
        per_a = _soc_node("pa",  f"{name_a}", SocialNodeType.PERSON)
        per_b = _soc_node("pb",  f"{name_b}", SocialNodeType.PERSON)
        int_a = _soc_node("ia",  want_a,      SocialNodeType.INTENTION)
        int_b = _soc_node("ib",  want_b,      SocialNodeType.INTENTION)
        bel_a = _soc_node("ba",  bel_a_text,  SocialNodeType.BELIEF)
        bel_b = _soc_node("bb",  bel_b_text,  SocialNodeType.BELIEF)
        em_b  = _soc_node("eb",  emo_b,       SocialNodeType.EMOTION)
        g.add_node(ctx).add_node(per_a).add_node(per_b)
        g.add_node(int_a).add_node(int_b).add_node(bel_a).add_node(bel_b).add_node(em_b)
        g.add_edge(_soc_edge("pa", "ia",  SocialRelation.WANTS))
        g.add_edge(_soc_edge("pa", "ba",  SocialRelation.BELIEVES))
        g.add_edge(_soc_edge("pa", "pb",  SocialRelation.PERSUADES))
        g.add_edge(_soc_edge("pb", "ib",  SocialRelation.WANTS))
        g.add_edge(_soc_edge("pb", "bb",  SocialRelation.BELIEVES))
        g.add_edge(_soc_edge("pb", "eb",  SocialRelation.FEELS))
        g.root_question = (
            f"¿Por qué no tiene éxito el intento de persuasión de {name_a}?"
        )

        return SocialExample(
            problem_text=(
                f"En {ctx_desc}, {name_a} quiere {want_a} y cree {bel_a_text}. "
                f"Intenta convencer a {name_b}. "
                f"Pero {name_b} quiere {want_b} y cree {bel_b_text}. "
                f"{name_b} siente {emo_b}."
            ),
            graph=g,
            answer=(
                f"La persuasión no funciona porque {name_a} y {name_b} tienen "
                f"intenciones y creencias opuestas: {name_a} cree {bel_a_text} "
                f"y {name_b} cree {bel_b_text}."
            ),
            complexity_level=3,
            answer_type=SocialAnswerType.CONFLICT_DIAGNOSIS,
            metadata={
                "person_a": name_a, "person_b": name_b,
                "belief_a": bel_a_text, "belief_b": bel_b_text,
                "want_a": want_a, "want_b": want_b,
                "required_node_types": ["context", "person", "intention", "belief", "emotion"],
                "required_relations":  ["wants", "believes", "persuades", "feels"],
                "social_checks": {
                    "every_action_has_person":          True,
                    "conflict_has_opposing_intentions": True,
                    "person_count": 2,
                },
            },
        )

    def _cultural_clash(self, rng: random.Random) -> SocialExample:
        """
        Dos normas distintas (NORM_a y NORM_b) generan conflicto sin mala fe.
        per_a --BELIEVES--> bel_a (su norma es la adecuada)
        per_b --BELIEVES--> bel_b (su norma es la adecuada)
        per_a --VIOLATES_NORM--> norm_b  (viola norma de B sin saberlo)
        per_b --FEELS--> em_b
        ctx
        """
        names = rng.sample(_NAMES, 2)
        name_a, name_b = names[0], names[1]
        norms = rng.sample(_NORMS, 2)
        norm_a, norm_b = norms[0], norms[1]
        emo_b = rng.choice(["incomodidad", "confusión", "sorpresa", "tristeza"])
        ctx_desc, _ = rng.choice(_CONTEXTS)

        bel_a_text = f"que lo correcto es {norm_a}"
        bel_b_text = f"que lo correcto es {norm_b}"
        repair = rng.choice(_REPAIR_STRATEGIES)

        g = CausalGraph()
        ctx   = _soc_node("ctx",  ctx_desc,   SocialNodeType.CONTEXT)
        per_a = _soc_node("pa",   f"{name_a}", SocialNodeType.PERSON)
        per_b = _soc_node("pb",   f"{name_b}", SocialNodeType.PERSON)
        norm_b_node = _soc_node("nb",  norm_b, SocialNodeType.NORM)
        bel_a = _soc_node("ba",   bel_a_text, SocialNodeType.BELIEF)
        bel_b = _soc_node("bb",   bel_b_text, SocialNodeType.BELIEF)
        em_b  = _soc_node("eb",   emo_b,       SocialNodeType.EMOTION)
        g.add_node(ctx).add_node(per_a).add_node(per_b)
        g.add_node(norm_b_node).add_node(bel_a).add_node(bel_b).add_node(em_b)
        g.add_edge(_soc_edge("pa", "ba",  SocialRelation.BELIEVES))
        g.add_edge(_soc_edge("pb", "bb",  SocialRelation.BELIEVES))
        g.add_edge(_soc_edge("pa", "nb",  SocialRelation.VIOLATES_NORM))
        g.add_edge(_soc_edge("pb", "eb",  SocialRelation.FEELS))
        g.add_edge(_soc_edge("pa", "pb",  SocialRelation.MISUNDERSTANDS))
        g.root_question = f"¿Por qué hay tensión entre {name_a} y {name_b}?"

        return SocialExample(
            problem_text=(
                f"En {ctx_desc}, {name_a} actúa según {norm_a}. "
                f"{name_b} tiene una norma distinta: {norm_b}. "
                f"Sin mala fe, {name_a} viola la norma que {name_b} valora. "
                f"{name_b} siente {emo_b}."
            ),
            graph=g,
            answer=(
                f"La tensión surge por un choque de normas: {name_a} sigue {norm_a} "
                f"y {name_b} valora {norm_b}. Ninguno actúa de mala fe. "
                f"Para resolverlo: {repair}."
            ),
            complexity_level=3,
            answer_type=SocialAnswerType.NORM_VIOLATION,
            metadata={
                "person_a": name_a, "person_b": name_b,
                "norm_a": norm_a, "norm_b": norm_b, "emotion_b": emo_b,
                "required_node_types": ["context", "person", "norm", "belief", "emotion"],
                "required_relations":  ["believes", "violates_norm", "feels", "misunderstands"],
                "social_checks": {
                    "every_action_has_person": True,
                    "norm_violation_has_norm":  True,
                    "person_count": 2,
                },
            },
        )

    def _trust_betrayal(self, rng: random.Random) -> SocialExample:
        """
        per_a --TRUSTS--> per_b
        per_b --WANTS--> int_b (que viola la confianza)
        per_b --VIOLATES_NORM--> norm
        per_a --FEELS--> em_a
        per_a --EXPECTS--> exp_a (que no ocurriría)
        per_b --RECIPROCATES--> per_a (intento de reparación)
        """
        names = rng.sample(_NAMES, 2)
        name_a, name_b = names[0], names[1]
        norm  = rng.choice(_NORMS)
        emo_a = rng.choice(["traición", "tristeza", "enojo", "confusión"])
        want_b = rng.choice([
            f"conseguir una ventaja personal",
            f"resolver su propio problema sin medir consecuencias",
            f"avanzar más rápido sin consultar",
            f"protegerse a sí mismo/a en ese momento",
        ])
        expectation = rng.choice([
            f"que {name_b} mantendría la confidencialidad",
            f"que {name_b} la/lo consultaría antes de actuar",
            f"que {name_b} respetaría el acuerdo implícito",
        ])
        ctx_desc, _ = rng.choice(_CONTEXTS)
        repair = rng.choice(_REPAIR_STRATEGIES)
        rel_type = rng.choice(_RELATIONSHIPS)

        g = CausalGraph()
        ctx   = _soc_node("ctx", ctx_desc,   SocialNodeType.CONTEXT)
        per_a = _soc_node("pa",  f"{name_a}", SocialNodeType.PERSON)
        per_b = _soc_node("pb",  f"{name_b}", SocialNodeType.PERSON)
        int_b = _soc_node("ib",  want_b,      SocialNodeType.INTENTION)
        norm_ = _soc_node("nm",  norm,        SocialNodeType.NORM)
        em_a  = _soc_node("ea",  emo_a,       SocialNodeType.EMOTION)
        exp_a = _soc_node("ex",  expectation, SocialNodeType.EXPECTATION)
        rel   = _soc_node("rel", rel_type,    SocialNodeType.RELATIONSHIP)
        g.add_node(ctx).add_node(per_a).add_node(per_b).add_node(int_b)
        g.add_node(norm_).add_node(em_a).add_node(exp_a).add_node(rel)
        g.add_edge(_soc_edge("pa", "pb",  SocialRelation.TRUSTS))
        g.add_edge(_soc_edge("pb", "ib",  SocialRelation.WANTS))
        g.add_edge(_soc_edge("pb", "nm",  SocialRelation.VIOLATES_NORM))
        g.add_edge(_soc_edge("pa", "ea",  SocialRelation.FEELS))
        g.add_edge(_soc_edge("pa", "ex",  SocialRelation.EXPECTS))
        g.add_edge(_soc_edge("pb", "pa",  SocialRelation.RECIPROCATES))
        g.root_question = f"¿Cómo puede {name_b} reparar la confianza de {name_a}?"

        return SocialExample(
            problem_text=(
                f"En {ctx_desc}, {name_a} y {name_b} tienen una {rel_type}. "
                f"{name_a} confía en {name_b} y esperaba {expectation}. "
                f"Pero {name_b} quiso {want_b} y violó {norm}. "
                f"{name_a} siente {emo_a}."
            ),
            graph=g,
            answer=(
                f"Para reparar la confianza, {name_b} debería: {repair}. "
                f"La norma violada fue: {norm}."
            ),
            complexity_level=3,
            answer_type=SocialAnswerType.REPAIR_STRATEGY,
            metadata={
                "person_a": name_a, "person_b": name_b,
                "norm": norm, "emotion_a": emo_a, "repair": repair,
                "required_node_types": [
                    "context", "person", "intention", "norm",
                    "emotion", "expectation", "relationship"
                ],
                "required_relations":  [
                    "trusts", "wants", "violates_norm",
                    "feels", "expects", "reciprocates"
                ],
                "social_checks": {
                    "every_action_has_person": True,
                    "norm_violation_has_norm":  True,
                    "person_count": 2,
                },
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN DE COHERENCIA SOCIAL
# ─────────────────────────────────────────────────────────────────────────────

def verify_social_example(ex: SocialExample) -> SocialVerificationResult:
    """
    Verifica que el grafo social del ejemplo es estructuralmente coherente.

    Checks:
      1. required_node_types están presentes en el grafo
      2. required_relations están presentes en las aristas
      3. social_checks específicos:
         - every_action_has_person:     hay ≥1 nodo PERSON con arista WANTS/FEELS/BELIEVES
         - person_count:                hay al menos N nodos PERSON
         - misunderstanding_has_belief: si hay MISUNDERSTANDS, hay BELIEF en el grafo
         - norm_violation_has_norm:     si hay VIOLATES_NORM, hay NORM en el grafo
         - conflict_has_opposing_intentions: si hay conflicto, hay ≥2 INTENTION
    """
    graph   = ex.graph
    meta    = ex.metadata
    checks  = meta.get("social_checks", {})

    node_types = _node_types_in_graph(graph)
    relations  = _relations_in_graph(graph)

    # ── Check 1: required_node_types ──────────────────────────────────────
    required_ntypes = meta.get("required_node_types", [])
    missing_types = [nt for nt in required_ntypes if nt not in node_types]
    if missing_types:
        return SocialVerificationResult(
            False,
            f"Missing required node types: {missing_types}",
            {"present": list(node_types), "required": required_ntypes},
        )

    # ── Check 2: required_relations ───────────────────────────────────────
    required_rels = meta.get("required_relations", [])
    missing_rels = [r for r in required_rels if r not in relations]
    if missing_rels:
        return SocialVerificationResult(
            False,
            f"Missing required relations: {missing_rels}",
            {"present": list(relations), "required": required_rels},
        )

    # ── Check 3: every_action_has_person ─────────────────────────────────
    if checks.get("every_action_has_person"):
        person_nodes = {
            node.node_id for node in graph.nodes
            if node.node_type == SocialNodeType.PERSON
        }
        action_rels = {
            SocialRelation.WANTS.value, SocialRelation.FEELS.value,
            SocialRelation.BELIEVES.value, SocialRelation.EXPECTS.value,
        }
        edges_with_action = [
            e for e in graph.edges
            if e.relation.value in action_rels
        ]
        if edges_with_action:
            # All action edges must originate from a PERSON
            orphans = [
                e for e in edges_with_action
                if e.source_id not in person_nodes
            ]
            if orphans:
                return SocialVerificationResult(
                    False,
                    f"Action edges not originating from PERSON: "
                    f"{[(e.source_id, e.relation.value) for e in orphans]}",
                    {"person_nodes": list(person_nodes)},
                )

    # ── Check 4: person_count ─────────────────────────────────────────────
    expected_persons = checks.get("person_count", 0)
    if expected_persons > 0:
        actual_persons = sum(
            1 for node in graph.nodes
            if node.node_type == SocialNodeType.PERSON
        )
        if actual_persons < expected_persons:
            return SocialVerificationResult(
                False,
                f"Expected ≥{expected_persons} PERSON nodes, found {actual_persons}",
                {"actual_persons": actual_persons},
            )

    # ── Check 5: misunderstanding_has_belief ─────────────────────────────
    if checks.get("misunderstanding_has_belief", False):
        if "misunderstands" in relations and "belief" not in node_types:
            return SocialVerificationResult(
                False,
                "MISUNDERSTANDS present but no BELIEF node in graph",
                {"node_types": list(node_types), "relations": list(relations)},
            )

    # ── Check 6: norm_violation_has_norm ─────────────────────────────────
    if checks.get("norm_violation_has_norm", False):
        if "violates_norm" in relations and "norm" not in node_types:
            return SocialVerificationResult(
                False,
                "VIOLATES_NORM present but no NORM node in graph",
                {"node_types": list(node_types)},
            )

    # ── Check 7: conflict_has_opposing_intentions ─────────────────────────
    if checks.get("conflict_has_opposing_intentions", False):
        intention_nodes = [
            node for node in graph.nodes
            if node.node_type == SocialNodeType.INTENTION
        ]
        if len(intention_nodes) < 2:
            return SocialVerificationResult(
                False,
                f"Conflict requires ≥2 INTENTION nodes, found {len(intention_nodes)}",
                {"intention_count": len(intention_nodes)},
            )

    return SocialVerificationResult(True, "All social coherence checks passed", {
        "node_types": list(node_types),
        "relations":  list(relations),
        "level":      ex.complexity_level,
    })


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class SocialGraphGenerator:
    """
    Generador principal de ejemplos sociales para EMPATHY.

    Uso:
        gen   = SocialGraphGenerator(seed=42)
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

    def generate(self, level: int = 1) -> SocialExample:
        """Genera un SocialExample del nivel indicado."""
        if level not in self._gens:
            raise ValueError(f"level must be 1, 2 or 3, got {level}")
        return self._gens[level].generate(self._rng)

    def generate_batch(
        self,
        n: int,
        level_distribution: Optional[Dict[int, float]] = None,
    ) -> List[SocialExample]:
        """
        Genera n ejemplos con distribución de niveles configurable.

        Args:
            n:                   número de ejemplos a generar
            level_distribution:  dict {nivel: peso}
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
