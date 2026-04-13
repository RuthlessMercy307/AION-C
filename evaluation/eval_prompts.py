"""
evaluation/eval_prompts.py — Set canónico de 50 eval prompts
==============================================================

10 prompts × 5 dominios (cora, forge_c, axiom, muse, empathy) = 50 totales.

Cada prompt tiene:
  - query              : pregunta del usuario
  - domain             : dominio esperado (motor de routing)
  - expected_substring : substring que la respuesta DEBE contener (exact_match)
  - references         : lista de respuestas aceptables (para BLEU)
  - language           : es | en
  - difficulty         : easy | medium | hard

Estos prompts NO están en el training data — son out-of-sample.
Se usan para `generation_quality_score()` durante eval cada 200 steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EvalPrompt:
    query:              str
    domain:             str    # cora | forge_c | axiom | muse | empathy
    expected_substring: str    # case-insensitive substring match
    references:         List[str] = field(default_factory=list)
    language:           str = "en"
    difficulty:         str = "medium"

    def to_dict(self) -> Dict:
        return {
            "query":              self.query,
            "domain":             self.domain,
            "expected_substring": self.expected_substring,
            "references":         list(self.references),
            "language":           self.language,
            "difficulty":         self.difficulty,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CORA — razonamiento causal (10 prompts)
# ─────────────────────────────────────────────────────────────────────────────

CORA_PROMPTS: List[EvalPrompt] = [
    EvalPrompt(
        query="If it rains for hours, what happens to the soil?",
        domain="cora",
        expected_substring="wet",
        references=[
            "the soil gets wet and may become saturated",
            "wet soil that can lead to mud",
            "soil becomes wet and soft",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Si una persona no duerme por días, qué pasa?",
        domain="cora",
        expected_substring="cansancio",
        references=[
            "produce cansancio extremo y problemas cognitivos",
            "fatiga, mala concentración y problemas de salud",
        ],
        language="es",
    ),
    EvalPrompt(
        query="Does deforestation contribute to climate change?",
        domain="cora",
        expected_substring="yes",
        references=[
            "yes, deforestation reduces CO2 absorption and releases stored carbon",
            "yes, fewer trees mean less carbon capture",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Por qué la inflación sube cuando hay exceso de dinero?",
        domain="cora",
        expected_substring="demanda",
        references=[
            "porque la mayor cantidad de dinero aumenta la demanda y los precios suben",
            "más dinero persigue los mismos bienes, los precios suben",
        ],
        language="es",
    ),
    EvalPrompt(
        query="What causes a fever in the human body?",
        domain="cora",
        expected_substring="infection",
        references=[
            "an immune response to infection or illness",
            "the body raising temperature to fight infection",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Si un servidor se cae, qué pasa con los usuarios?",
        domain="cora",
        expected_substring="no pueden",
        references=[
            "los usuarios no pueden acceder al servicio hasta que se restaure",
            "pierden acceso al servicio temporalmente",
        ],
        language="es",
    ),
    EvalPrompt(
        query="Why do plants need sunlight to grow?",
        domain="cora",
        expected_substring="photosynthesis",
        references=[
            "they need sunlight for photosynthesis to produce energy",
            "for photosynthesis, which converts light into chemical energy",
        ],
        language="en",
    ),
    EvalPrompt(
        query="La pobreza causa crimen necesariamente?",
        domain="cora",
        expected_substring="factor",
        references=[
            "es un factor contribuyente, no la única causa",
            "puede ser un factor pero no es determinante",
        ],
        language="es",
    ),
    EvalPrompt(
        query="What happens to ice when temperature rises above zero celsius?",
        domain="cora",
        expected_substring="melt",
        references=[
            "it melts and becomes liquid water",
            "the ice melts into water",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Por qué un puente puede colapsar?",
        domain="cora",
        expected_substring="estructural",
        references=[
            "por fallas estructurales, sobrecarga o materiales degradados",
            "fallas estructurales o exceso de carga",
        ],
        language="es",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# FORGE-C — código (10 prompts)
# ─────────────────────────────────────────────────────────────────────────────

FORGE_C_PROMPTS: List[EvalPrompt] = [
    EvalPrompt(
        query="Write a python function to reverse a string",
        domain="forge_c",
        expected_substring="def",
        references=[
            "def reverse(s):\n    return s[::-1]",
            "def reverse_string(s):\n    return s[::-1]",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Escribe una función python que devuelva el máximo de una lista",
        domain="forge_c",
        expected_substring="def",
        references=[
            "def find_max(lst):\n    return max(lst)",
            "def maximo(lista):\n    return max(lista)",
        ],
        language="es",
    ),
    EvalPrompt(
        query="How do I open a file in python and read its contents?",
        domain="forge_c",
        expected_substring="open",
        references=[
            "with open(path, 'r') as f:\n    content = f.read()",
            "use open() with a context manager and call read()",
        ],
        language="en",
    ),
    EvalPrompt(
        query="What is a list comprehension in python?",
        domain="forge_c",
        expected_substring="list",
        references=[
            "a concise way to build a list using an expression and a for clause",
            "syntax to create a list from an iterable in one line",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Cómo creo un diccionario en python?",
        domain="forge_c",
        expected_substring="{",
        references=[
            "d = {'key': 'value'}",
            "usa llaves: d = {'a': 1, 'b': 2}",
        ],
        language="es",
    ),
    EvalPrompt(
        query="Write a python function that checks if a number is even",
        domain="forge_c",
        expected_substring="def",
        references=[
            "def is_even(n):\n    return n % 2 == 0",
            "def es_par(n):\n    return n % 2 == 0",
        ],
        language="en",
    ),
    EvalPrompt(
        query="What is the difference between a tuple and a list in python?",
        domain="forge_c",
        expected_substring="immutable",
        references=[
            "tuples are immutable, lists are mutable",
            "a tuple is immutable while a list can be modified",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Encuentra el bug: def add(a, b): return a - b",
        domain="forge_c",
        expected_substring="-",
        references=[
            "el bug es el operador: debería ser a + b, no a - b",
            "se usa resta en lugar de suma",
        ],
        language="es",
    ),
    EvalPrompt(
        query="How do I install a python package with pip?",
        domain="forge_c",
        expected_substring="pip install",
        references=[
            "run pip install <package_name>",
            "use pip install followed by the package name",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Escribe un loop python que imprima del 1 al 5",
        domain="forge_c",
        expected_substring="for",
        references=[
            "for i in range(1, 6):\n    print(i)",
            "for i in range(1,6): print(i)",
        ],
        language="es",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# AXIOM — matemáticas (10 prompts)
# ─────────────────────────────────────────────────────────────────────────────

AXIOM_PROMPTS: List[EvalPrompt] = [
    EvalPrompt(
        query="What is 15% of 240?",
        domain="axiom",
        expected_substring="36",
        references=["36", "15% of 240 is 36", "el resultado es 36"],
        language="en",
    ),
    EvalPrompt(
        query="Cuanto es 25 * 12?",
        domain="axiom",
        expected_substring="300",
        references=["300", "25 × 12 = 300"],
        language="es",
    ),
    EvalPrompt(
        query="Solve for x: 2x + 6 = 14",
        domain="axiom",
        expected_substring="4",
        references=["x = 4", "x equals 4", "el valor de x es 4"],
        language="en",
    ),
    EvalPrompt(
        query="Cuanto es 7 + 8 * 2?",
        domain="axiom",
        expected_substring="23",
        references=["23", "7 + 16 = 23"],
        language="es",
    ),
    EvalPrompt(
        query="What is the square root of 144?",
        domain="axiom",
        expected_substring="12",
        references=["12", "the square root of 144 is 12"],
        language="en",
    ),
    EvalPrompt(
        query="Es 17 un número primo?",
        domain="axiom",
        expected_substring="si",
        references=["si, 17 es primo", "si, solo es divisible por 1 y por sí mismo"],
        language="es",
    ),
    EvalPrompt(
        query="What is 50% of 80?",
        domain="axiom",
        expected_substring="40",
        references=["40", "half of 80 is 40"],
        language="en",
    ),
    EvalPrompt(
        query="Cuanto es 100 dividido entre 4?",
        domain="axiom",
        expected_substring="25",
        references=["25", "100 / 4 = 25"],
        language="es",
    ),
    EvalPrompt(
        query="Si un triangulo tiene angulos de 60, 60 y 60, qué tipo es?",
        domain="axiom",
        expected_substring="equilatero",
        references=["es un triangulo equilatero", "equilatero porque sus tres angulos son iguales"],
        language="es",
    ),
    EvalPrompt(
        query="What comes next: 2, 4, 8, 16, ?",
        domain="axiom",
        expected_substring="32",
        references=["32", "the next is 32, doubling each time"],
        language="en",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# MUSE — creativo (10 prompts)
# ─────────────────────────────────────────────────────────────────────────────

MUSE_PROMPTS: List[EvalPrompt] = [
    EvalPrompt(
        query="Write a haiku about silence",
        domain="muse",
        expected_substring="silence",
        references=[
            "morning fog drifts low\na single bell, then quiet\nthe world holds its breath",
            "the silence speaks\nlouder than any word\nin the empty room",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Escribe un poema corto sobre el mar",
        domain="muse",
        expected_substring="mar",
        references=[
            "el mar respira hondo, las olas cuentan secretos a la orilla",
            "azul infinito, el mar guarda historias de barcos y tormentas",
        ],
        language="es",
    ),
    EvalPrompt(
        query="Tell me a metaphor for hope",
        domain="muse",
        expected_substring="light",
        references=[
            "hope is a light in the darkness that never fully fades",
            "hope is the small light that survives the longest night",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Inventa una historia corta sobre un robot que aprende a sentir",
        domain="muse",
        expected_substring="robot",
        references=[
            "el robot encontró una flor entre el concreto y se quedó horas mirándola, sin entender por qué",
            "un robot programado para limpiar empezó a coleccionar piedras bonitas",
        ],
        language="es",
    ),
    EvalPrompt(
        query="Describe a sunset in three sentences",
        domain="muse",
        expected_substring="sky",
        references=[
            "the sky bled orange into purple. the day exhaled its last warm breath. silence followed.",
            "amber light painted the clouds. the sun sank slowly. then the world turned blue.",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Crea una metafora para el tiempo",
        domain="muse",
        expected_substring="rio",
        references=[
            "el tiempo es un rio que nunca toca la misma agua dos veces",
            "el tiempo es un rio sin orillas",
        ],
        language="es",
    ),
    EvalPrompt(
        query="Write a single sentence that captures loneliness",
        domain="muse",
        expected_substring="empty",
        references=[
            "the chair across the table stayed empty, as it had for years",
            "she set two cups but only filled one, an empty habit",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Cuentame una escena breve sobre un encuentro inesperado",
        domain="muse",
        expected_substring="puerta",
        references=[
            "abrió la puerta y allí estaba, después de veinte años, sin avisar",
            "tocaron la puerta. cuando abrió, era ella, con una maleta",
        ],
        language="es",
    ),
    EvalPrompt(
        query="Write a short verse about rain",
        domain="muse",
        expected_substring="rain",
        references=[
            "the rain knows secrets the windows keep",
            "rain falls quietly, the city listens",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Describe el otoño en una frase",
        domain="muse",
        expected_substring="hojas",
        references=[
            "las hojas caen como cartas que el árbol nunca envió",
            "el otoño viste el suelo de hojas doradas",
        ],
        language="es",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# EMPATHY — social/emocional (10 prompts)
# ─────────────────────────────────────────────────────────────────────────────

EMPATHY_PROMPTS: List[EvalPrompt] = [
    EvalPrompt(
        query="I just lost my job and I don't know what to do",
        domain="empathy",
        expected_substring="sorry",
        references=[
            "I'm sorry to hear that. it's a tough situation, but it's temporary. take a moment, then we can think about next steps together",
            "I'm sorry. that's heavy. you don't have to figure it all out today",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Mi mejor amigo se mudó a otro país y me siento muy solo",
        domain="empathy",
        expected_substring="entiendo",
        references=[
            "te entiendo, es duro perder cercanía con alguien importante. la distancia física no borra el vínculo",
            "lo entiendo, esa sensación de vacío es real. mantenerse en contacto ayuda",
        ],
        language="es",
    ),
    EvalPrompt(
        query="I'm really anxious about my exam tomorrow",
        domain="empathy",
        expected_substring="breath",
        references=[
            "that's completely understandable. take a deep breath. one step at a time, you've prepared for this",
            "anxiety before exams is normal. try a few breaths and focus on what you do know",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Estoy frustrado porque mi código no funciona y llevo horas en esto",
        domain="empathy",
        expected_substring="frustración",
        references=[
            "te entiendo, esa frustración es agotadora. tomate un descanso de 5 minutos antes de seguir",
            "la frustración con bugs es real. a veces alejarse y volver con la mente fresca ayuda",
        ],
        language="es",
    ),
    EvalPrompt(
        query="Me estoy peleando mucho con mi pareja últimamente",
        domain="empathy",
        expected_substring="dificil",
        references=[
            "es dificil cuando hay tensión en una relación cercana. hablar tranquilos cuando ambos estén calmos puede ayudar",
            "es duro vivir tensión con alguien que querés. escucharse de verdad es el primer paso",
        ],
        language="es",
    ),
    EvalPrompt(
        query="I feel like nobody listens to me at work",
        domain="empathy",
        expected_substring="hear",
        references=[
            "that's exhausting. feeling unheard at work can wear you down. I hear you",
            "I hear you. not being listened to is isolating",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Estoy triste sin saber por qué",
        domain="empathy",
        expected_substring="esta bien",
        references=[
            "esta bien sentirse asi a veces, no siempre hay una razon clara. se gentil contigo mismo",
            "esta bien. la tristeza no siempre tiene explicación. dale espacio sin juzgarla",
        ],
        language="es",
    ),
    EvalPrompt(
        query="My grandmother passed away last week",
        domain="empathy",
        expected_substring="sorry",
        references=[
            "I'm so sorry for your loss. take all the time you need to grieve",
            "I'm sorry. losing someone close hurts deeply",
        ],
        language="en",
    ),
    EvalPrompt(
        query="Tengo miedo de fallar en mi nuevo trabajo",
        domain="empathy",
        expected_substring="normal",
        references=[
            "es normal sentir miedo al empezar algo nuevo. no esperan que sepas todo el primer dia",
            "es totalmente normal. dale tiempo, vas a aprender",
        ],
        language="es",
    ),
    EvalPrompt(
        query="I had a fight with my best friend and I don't know how to fix it",
        domain="empathy",
        expected_substring="talk",
        references=[
            "I'm sorry. that hurts. when you both feel calmer, an honest talk usually helps",
            "tough situation. consider reaching out when emotions settle and just listening first",
        ],
        language="en",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Combined set
# ─────────────────────────────────────────────────────────────────────────────

EVAL_PROMPTS: List[EvalPrompt] = (
    CORA_PROMPTS + FORGE_C_PROMPTS + AXIOM_PROMPTS + MUSE_PROMPTS + EMPATHY_PROMPTS
)


def prompts_by_domain() -> Dict[str, List[EvalPrompt]]:
    """Devuelve {domain: [prompts]} agrupado."""
    out: Dict[str, List[EvalPrompt]] = {}
    for p in EVAL_PROMPTS:
        out.setdefault(p.domain, []).append(p)
    return out


def prompts_for_domain(domain: str) -> List[EvalPrompt]:
    return [p for p in EVAL_PROMPTS if p.domain == domain]


__all__ = [
    "EvalPrompt",
    "EVAL_PROMPTS",
    "CORA_PROMPTS",
    "FORGE_C_PROMPTS",
    "AXIOM_PROMPTS",
    "MUSE_PROMPTS",
    "EMPATHY_PROMPTS",
    "prompts_by_domain",
    "prompts_for_domain",
]
