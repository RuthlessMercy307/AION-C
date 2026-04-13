"""
synth/knowledge/ — Pools de conocimiento factual real para AION-C.

Cada submódulo expone una función `facts()` que devuelve una lista de
dicts con el formato:

    {
        "topic": "python",
        "subtopic": "builtins",
        "q_en": "What does the len() function do?",
        "a_en": "It returns the number of items in an object...",
        "q_es": "¿Qué hace la función len()?",
        "a_es": "Devuelve el número de elementos de un objeto...",
        "difficulty": "easy",
    }

El generador principal (`synth.real_knowledge_gen`) itera sobre estos
pools y produce CanonicalRecord en formato `[USER:][AION:][EOS]`.

Los pools contienen hechos REALES del mundo, no texto sintético. Cada
línea es una afirmación factual verificable en Wikipedia o en
documentación oficial.
"""

from synth.knowledge import (
    programming,
    mathematics,
    science,
    history_geography,
    technology,
    language_patterns,
)


def all_facts():
    """Concatena todos los pools de todos los módulos."""
    out = []
    out.extend(programming.facts())
    out.extend(mathematics.facts())
    out.extend(science.facts())
    out.extend(history_geography.facts())
    out.extend(technology.facts())
    out.extend(language_patterns.facts())
    return out
