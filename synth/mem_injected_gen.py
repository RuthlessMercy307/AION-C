"""
synth/mem_injected_gen.py — 2K ejemplos con MEM inyectada
============================================================

Genera ejemplos donde un bloque [MEM: ...] se prepende a la query,
simulando que MEM tiene un dato relevante recuperado por la búsqueda
semántica de la Parte 8. El modelo aprende a USAR ese dato.

Cubre: hechos del usuario (nombre, lenguaje, proyectos, preferencias)
       hechos técnicos guardados (jwt_pattern, sqlite_wal, etc.)
       errores previos (error_log)
"""

from __future__ import annotations

import random
from typing import Iterator, List, Tuple

from .canonical_format import CanonicalRecord, build_record


# (mem_text, user_query, aion_response, lang, domain)
USER_FACT_CASES: List[Tuple[str, str, str, str, str]] = [
    ("name=Jesus, lang=es",
     "hola",
     "hola Jesus, en qué te ayudo?",
     "es", "general"),
    ("name=Maria, level=advanced, projects=[aion-c]",
     "ayúdame con un bug",
     "claro Maria, mostrame el bug y el traceback completo",
     "es", "forge_c"),
    ("name=Alex, lang=en, tone=casual",
     "hey",
     "hey Alex, what's up?",
     "en", "general"),
    ("name=Sara, projects=[react-app]",
     "cómo va el proyecto?",
     "el react-app sigue pendiente del refactor del router. quieres retomarlo?",
     "es", "general"),
    ("editor=vscode, lang=en",
     "show me a quick snippet",
     "here you go (in vscode-friendly format): const x = 5;",
     "en", "forge_c"),
]

TECH_FACT_CASES: List[Tuple[str, str, str, str, str]] = [
    ("jwt_pattern: use PyJWT with HS256, 1h expiry",
     "cómo creo un JWT?",
     "usa PyJWT con HS256 y expiración de 1h, como guardamos antes",
     "es", "forge_c"),
    ("sqlite_wal: enable with PRAGMA journal_mode=WAL for concurrency",
     "how do I improve sqlite write performance?",
     "enable WAL mode: conn.execute('PRAGMA journal_mode=WAL')",
     "en", "forge_c"),
    ("python_typing: prefer | over Union in 3.10+",
     "qué uso para tipos opcionales?",
     "en python 3.10+ usa `int | None` directamente, no `Optional[int]`",
     "es", "forge_c"),
    ("rate_limit: redis with sliding window, 100 req/min",
     "cómo hago rate limiting?",
     "implementa sliding window en redis con límite de 100 req/min, como lo hicimos antes",
     "es", "forge_c"),
]

ERROR_LOG_CASES: List[Tuple[str, str, str, str, str]] = [
    ("error: NoneType has no attribute 'split' | cause: forgot None check after dict.get | prevention: use .get(k, '') with default",
     "tengo un error con .split() en None",
     "ya pasó antes: usaste .get() sin default, devolvió None y luego .split() falla. usa .get(k, '')",
     "es", "forge_c"),
    ("error: SQL injection in user search | cause: f-string interpolation in query | prevention: parameterized queries",
     "estoy formateando una query de SQL, está bien?",
     "no, eso es SQL injection. usa parámetros: cur.execute('SELECT ... WHERE x=?', (val,))",
     "es", "forge_c"),
    ("error: index out of range in pagination | cause: didn't validate page number | prevention: clamp page to [1, max_page]",
     "my pagination crashes on the last page",
     "I've seen this before — clamp the page number to [1, max_page] before slicing",
     "en", "forge_c"),
]


def generate_mem_injected(n: int = 2000, seed: int = 42) -> Iterator[CanonicalRecord]:
    """Genera n ejemplos con [MEM: ...] inyectado."""
    rng = random.Random(seed)
    pool = USER_FACT_CASES + TECH_FACT_CASES + ERROR_LOG_CASES

    for i in range(n):
        mem_text, user, aion, lang, domain = rng.choice(pool)
        # Identifica el tipo de fact para metadata
        if (mem_text, user, aion, lang, domain) in USER_FACT_CASES:
            mem_kind = "user_fact"
        elif (mem_text, user, aion, lang, domain) in TECH_FACT_CASES:
            mem_kind = "tech_fact"
        else:
            mem_kind = "error_log"

        yield build_record(
            user=user,
            aion=aion,
            mem=mem_text,
            domain=domain,
            language=lang,
            type="mem",
            metadata={"mem_kind": mem_kind, "seed_index": i},
        )


__all__ = ["generate_mem_injected"]
