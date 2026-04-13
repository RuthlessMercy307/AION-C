"""
synth/conversational_gen.py — 5K ejemplos conversacionales multi-turn
=======================================================================

Genera diálogos de 2-4 turnos en formato canónico:

    [USER: hola]
    [AION: hola, en qué te ayudo?]
    [USER: cuéntame de python]
    [AION: python es un lenguaje de programación...]
    [EOS]

Cubre todos los dominios (general, cora, forge_c, axiom, muse, empathy)
y bilingüe (es/en). Determinista por seed.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Tuple

from .canonical_format import CanonicalRecord, CanonicalTurn, build_record


# ─────────────────────────────────────────────────────────────────────────────
# Plantillas por dominio (es/en mezcladas)
# ─────────────────────────────────────────────────────────────────────────────


GREETINGS_ES = [
    ("hola", "hola, en qué te ayudo?"),
    ("buenas", "buenas, dime qué necesitas"),
    ("hey", "hey, cómo va?"),
    ("buenos días", "buenos días, qué hacemos hoy?"),
    ("qué tal", "bien, listo. dime"),
]
GREETINGS_EN = [
    ("hi", "hi, what's up?"),
    ("hello", "hello, how can I help?"),
    ("hey", "hey, what do you need?"),
    ("good morning", "good morning, what's on your mind?"),
    ("hey there", "hey, what can I do for you?"),
]

FOLLOWUPS_GENERAL_ES = [
    ("dime más", "dime qué parte específicamente y profundizo"),
    ("explícame", "claro, te lo explico paso a paso"),
    ("y eso?", "buena pregunta, te aclaro"),
    ("entiendo", "perfecto, seguimos"),
    ("no entendí", "a ver, te lo digo de otra forma"),
]
FOLLOWUPS_GENERAL_EN = [
    ("tell me more", "sure, which part should I expand?"),
    ("explain", "ok, here's the breakdown"),
    ("got it", "good, let's keep going"),
    ("not clear", "let me try a different angle"),
]

# Topics & Q/A pairs por dominio (lengua mezclada)
TOPIC_PAIRS = {
    "forge_c": [
        ("qué es python?", "python es un lenguaje de programación de alto nivel, interpretado, con tipado dinámico"),
        ("cómo funciona un for loop?", "un for loop itera sobre una secuencia ejecutando el cuerpo en cada elemento"),
        ("qué es una función?", "una función es un bloque de código reutilizable que toma entradas y devuelve salidas"),
        ("what is git?", "git is a distributed version control system for tracking code changes"),
        ("how do decorators work?", "a decorator wraps a function to extend its behavior without modifying it"),
        ("qué es una lista?", "una lista es una colección ordenada y mutable de elementos"),
        ("what is JSON?", "JSON is a lightweight data format using key-value pairs and arrays"),
        ("qué es recursión?", "recursión es cuando una función se llama a sí misma con un caso base que la detiene"),
    ],
    "axiom": [
        ("cuánto es 12 + 8?", "20"),
        ("cuánto es 7 × 9?", "63"),
        ("qué es la integral?", "la integral mide el área bajo una curva, es la operación inversa de la derivada"),
        ("what is a prime number?", "a prime number is a natural number greater than 1 with no divisors other than 1 and itself"),
        ("cómo se resuelve x+5=12?", "se resta 5 de ambos lados: x = 12 - 5 = 7"),
        ("what is pi?", "pi is the ratio of a circle's circumference to its diameter, approximately 3.14159"),
        ("qué es el teorema de Pitágoras?", "en un triángulo rectángulo, a² + b² = c², donde c es la hipotenusa"),
        ("how do percentages work?", "X% of Y means X/100 multiplied by Y"),
    ],
    "cora": [
        ("por qué llueve?", "el agua se evapora, sube, se condensa en nubes y cae cuando las gotas son pesadas"),
        ("what causes earthquakes?", "tectonic plates moving and releasing stored energy along faults"),
        ("por qué los precios suben?", "demanda mayor que oferta, o aumento en los costos de producción, o exceso de dinero"),
        ("what is the greenhouse effect?", "gases trapping heat in the atmosphere, warming the planet"),
        ("cómo se forman las estrellas?", "nubes de gas y polvo colapsan por gravedad hasta encender fusión nuclear"),
        ("why do leaves change color?", "chlorophyll breaks down in autumn revealing other pigments already in the leaf"),
    ],
    "muse": [
        ("escribe un haiku", "luz en la rama, el viento mueve el agua, el día comienza"),
        ("write a metaphor for time", "time is a river that never touches the same water twice"),
        ("inventa una historia corta", "el robot encontró una flor entre el concreto. no entendía la belleza, pero se quedó a mirar"),
        ("describe a sunset", "the sky bled orange into purple, the day exhaling its last warm breath"),
    ],
    "empathy": [
        ("estoy frustrado", "te entiendo. cuéntame qué pasó, lo vemos juntos"),
        ("I feel anxious", "that's hard. one breath at a time. what's pressing on you right now?"),
        ("perdí mi trabajo", "lo siento mucho. es un golpe. cuando estés listo, miramos opciones sin prisa"),
        ("I'm tired of everything", "I hear you. you don't have to fix it all today. what's the smallest thing that would help?"),
    ],
    "general": [
        ("qué puedes hacer?", "puedo razonar, escribir código, resolver matemáticas, crear historias y entender emociones"),
        ("what's your name?", "I'm AION-C, built with the MoSE architecture"),
        ("quién te creó?", "fui creado por Jesús"),
        ("what is MoSE?", "Mixture of Specialized Engines: 5 motors, each for a kind of reasoning"),
    ],
}

CLOSINGS_ES = [
    ("gracias", "de nada"),
    ("genial", "perfecto, cualquier cosa avísame"),
    ("eso es todo", "listo, aquí estoy si necesitas más"),
    ("adiós", "hasta luego"),
]
CLOSINGS_EN = [
    ("thanks", "you're welcome"),
    ("perfect", "great, anything else?"),
    ("that's all", "ok, I'm here if you need more"),
    ("bye", "bye, take care"),
]


def _pick(rng: random.Random, lang: str, *pools_es_en) -> Tuple[str, str]:
    """Selecciona un par (q, a) del pool según idioma."""
    pool = pools_es_en[0] if lang == "es" else pools_es_en[1]
    return rng.choice(pool)


def generate_conversational(n: int = 5000, seed: int = 42) -> Iterator[CanonicalRecord]:
    """
    Genera n diálogos multi-turn (2-4 turnos), bilingües,
    cubriendo todos los dominios.
    """
    rng = random.Random(seed)
    domains = list(TOPIC_PAIRS.keys())

    for i in range(n):
        lang = "es" if rng.random() < 0.5 else "en"
        domain = rng.choice(domains)
        n_turns = rng.choices([2, 3, 4], weights=[0.4, 0.4, 0.2], k=1)[0]

        # Turno 1: greeting
        g_user, g_aion = _pick(rng, lang, GREETINGS_ES, GREETINGS_EN)

        # Turno 2: topic question (domain-specific)
        topic_user, topic_aion = rng.choice(TOPIC_PAIRS[domain])

        extras: List[CanonicalTurn] = [
            CanonicalTurn(user=topic_user, aion=topic_aion),
        ]

        # Turno 3 (opcional): followup
        if n_turns >= 3:
            f_user, f_aion = _pick(rng, lang, FOLLOWUPS_GENERAL_ES, FOLLOWUPS_GENERAL_EN)
            extras.append(CanonicalTurn(user=f_user, aion=f_aion))

        # Turno 4 (opcional): closing
        if n_turns >= 4:
            c_user, c_aion = _pick(rng, lang, CLOSINGS_ES, CLOSINGS_EN)
            extras.append(CanonicalTurn(user=c_user, aion=c_aion))

        yield build_record(
            user=g_user,
            aion=g_aion,
            extra_turns=extras,
            domain=domain,
            language=lang,
            type="multi_turn",
            metadata={"seed_index": i, "n_turns": n_turns},
        )


__all__ = ["generate_conversational"]
