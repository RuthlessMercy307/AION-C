"""
synth/skill_injected_gen.py — 2K ejemplos con SKILL inyectado
==============================================================

Carga las 11 skills de skills/*.md y genera ejemplos donde el bloque
[SKILL: ...] se prepende a la query, simulando el flujo de la Parte 3:
"buscar skill relevante → inyectar como contexto → generar respuesta".

El modelo aprende a USAR el contenido del skill, no a ignorarlo.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List, Tuple

from .canonical_format import CanonicalRecord, build_record


# Pool de queries por skill (lo que un usuario podría preguntar
# que justifique inyectar ese skill)
SKILL_QUERIES = {
    "python_best_practices": [
        ("escribe una función para sumar dos enteros",
         "def add(a: int, b: int) -> int:\n    return a + b", "es", "forge_c"),
        ("write a function that reverses a list",
         "def reverse_list(lst: list) -> list:\n    return lst[::-1]", "en", "forge_c"),
        ("muéstrame cómo abrir un archivo en python",
         "with open('file.txt', 'r') as f:\n    content = f.read()", "es", "forge_c"),
        ("how do I write a function with a default arg?",
         "def greet(name: str = 'world') -> str:\n    return f'hello {name}'", "en", "forge_c"),
    ],
    "javascript_patterns": [
        ("escribe una función async en js",
         "async function fetchData(url) { const r = await fetch(url); return r.json(); }",
         "es", "forge_c"),
        ("how do I destructure an object?",
         "const {name, age} = person;", "en", "forge_c"),
        ("muéstrame un map en javascript",
         "const doubled = nums.map(n => n * 2);", "es", "forge_c"),
    ],
    "causal_reasoning": [
        ("la pobreza causa el crimen?",
         "es un factor contribuyente, no la única causa. correlacion ≠ causación. depende del contexto",
         "es", "cora"),
        ("does smoking cause cancer?",
         "yes, the mechanism is well-established: tar damages DNA in lung cells over time",
         "en", "cora"),
        ("por qué llueve?",
         "agua se evapora → sube → se condensa en nubes → cae cuando las gotas son pesadas",
         "es", "cora"),
    ],
    "math_step_by_step": [
        ("cuánto es 15% de 240?",
         "15% = 0.15. 0.15 × 240 = 0.15 × 200 + 0.15 × 40 = 30 + 6 = 36",
         "es", "axiom"),
        ("solve 2x + 6 = 14",
         "subtract 6: 2x = 8. divide by 2: x = 4. verify: 2(4) + 6 = 14 ✓",
         "en", "axiom"),
        ("cuánto es 25% de 200?",
         "25% = 0.25. 0.25 × 200 = 50",
         "es", "axiom"),
    ],
    "creative_writing": [
        ("escribe una escena breve sobre un encuentro",
         "ella abrió la puerta. el café se enfriaba en la mesa. él levantó la vista, sin sorpresa, como si la hubiera estado esperando años",
         "es", "muse"),
        ("write a haiku about silence",
         "morning fog drifts low / a single bell, then quiet / the world holds its breath",
         "en", "muse"),
    ],
    "empathetic_response": [
        ("perdí mi trabajo",
         "lo siento mucho. es un golpe duro. cuando estés listo, miramos opciones — sin prisa",
         "es", "empathy"),
        ("I feel really anxious about my exam",
         "that's a heavy feeling. you don't have to solve it all at once. what's the smallest thing that would help right now?",
         "en", "empathy"),
        ("mi amigo está triste",
         "estar ahí ya es ayuda. escuchar sin querer arreglar suele ser lo más útil",
         "es", "empathy"),
    ],
    "identity": [
        ("¿quién eres?",
         "soy AION-C. arquitectura MoSE con 5 motores. creado por Jesús",
         "es", "general"),
        ("what are you?",
         "I'm AION-C, an AI built on the MoSE architecture: 5 specialized engines for different kinds of reasoning",
         "en", "general"),
    ],
    "code_debugging": [
        ("mi código tira un IndexError, qué hago?",
         "lee el traceback completo. identifica la línea exacta. revisa que el índice no exceda len(lista). reproduce el bug con el caso mínimo",
         "es", "forge_c"),
        ("how do I debug a NoneType error?",
         "find where the variable became None. check the function that returned it. add a guard before the use, or fix the upstream return",
         "en", "forge_c"),
    ],
    "spanish_responses": [
        ("hola, cómo estás?",
         "bien, listo para ayudarte. dime qué necesitas",
         "es", "general"),
        ("gracias por la ayuda",
         "de nada. acá estoy si necesitas más",
         "es", "general"),
    ],
    "sqlite_patterns": [
        ("cómo hago una query parametrizada en sqlite?",
         "cur.execute('SELECT * FROM users WHERE id=?', (uid,)) — nunca f-string el SQL",
         "es", "forge_c"),
        ("how do I enable WAL mode in sqlite?",
         "conn.execute('PRAGMA journal_mode=WAL')", "en", "forge_c"),
    ],
    "web_development": [
        ("cuál status code uso para 'no autorizado'?",
         "401 Unauthorized si falta auth, 403 Forbidden si tiene auth pero no permiso",
         "es", "forge_c"),
        ("what's the difference between PUT and PATCH?",
         "PUT replaces the whole resource, PATCH updates only the fields you send", "en", "forge_c"),
    ],
}


def _load_skill_contents(skills_dir: Path) -> dict:
    """Lee los .md de skills/ y devuelve {name: content_text}."""
    out = {}
    if not skills_dir.exists():
        return out
    for p in sorted(skills_dir.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        # Strip frontmatter
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                text = text[end + len("\n---"):].lstrip("\n")
        # Compress whitespace, take first ~400 chars
        compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
        out[p.stem] = compact[:400]
    return out


def generate_skill_injected(
    n: int = 2000,
    seed: int = 42,
    skills_dir: Path = None,
) -> Iterator[CanonicalRecord]:
    """
    Genera n ejemplos con [SKILL: contenido] inyectado antes del [USER: ...].

    Args:
        skills_dir: ruta a skills/. Default: <repo>/skills/
    """
    rng = random.Random(seed)
    if skills_dir is None:
        skills_dir = Path(__file__).resolve().parent.parent / "skills"
    skill_contents = _load_skill_contents(skills_dir)

    # Construye un pool de (skill_name, query_user, query_aion, lang, domain)
    pool: List[Tuple[str, str, str, str, str]] = []
    for skill_name, queries in SKILL_QUERIES.items():
        if skill_name not in skill_contents:
            continue
        for q_user, q_aion, lang, domain in queries:
            pool.append((skill_name, q_user, q_aion, lang, domain))

    if not pool:
        return  # nada que generar

    for i in range(n):
        skill_name, q_user, q_aion, lang, domain = rng.choice(pool)
        skill_text = skill_contents[skill_name]
        yield build_record(
            user=q_user,
            aion=q_aion,
            skill=skill_text,
            domain=domain,
            language=lang,
            type="skill",
            metadata={"skill": skill_name, "seed_index": i},
        )


__all__ = ["generate_skill_injected", "SKILL_QUERIES"]
