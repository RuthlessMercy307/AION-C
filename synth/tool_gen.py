"""
synth/tool_gen.py — 3K ejemplos con tool calls
================================================

Genera ejemplos donde el modelo decide invocar una tool, recibe el
[RESULT: ...] y produce su [AION: ...] final con el resultado integrado.

Cubre los 8 tools del Tool system (Parte 4):
  write_file, edit_file, read_file, run_code, call_api,
  search_web, search_mem, store_mem
"""

from __future__ import annotations

import json
import random
from typing import Iterator, List, Tuple

from .canonical_format import CanonicalRecord, build_record


# ─────────────────────────────────────────────────────────────────────────────
# Templates por tool
# ─────────────────────────────────────────────────────────────────────────────


WRITE_FILE_CASES = [
    {
        "user": "crea un archivo hello.py que imprima hola mundo",
        "filename": "hello.py",
        "content": "print('hola mundo')",
        "aion_tpl": "Listo, creé {fn} con un print de 'hola mundo'.",
        "lang": "es",
    },
    {
        "user": "create a file greet.py that prints hello world",
        "filename": "greet.py",
        "content": "print('hello world')",
        "aion_tpl": "Done, I created {fn} with a print of 'hello world'.",
        "lang": "en",
    },
    {
        "user": "guarda esto en notes.txt: comprar leche, llamar a María",
        "filename": "notes.txt",
        "content": "comprar leche\nllamar a María",
        "aion_tpl": "Listo, guardé las dos notas en {fn}.",
        "lang": "es",
    },
    {
        "user": "save a sum function to math_utils.py",
        "filename": "math_utils.py",
        "content": "def add(a, b):\n    return a + b",
        "aion_tpl": "Saved an add() function to {fn}.",
        "lang": "en",
    },
    {
        "user": "crea config.json con port 8080",
        "filename": "config.json",
        "content": '{"port": 8080}',
        "aion_tpl": "Listo, {fn} con port=8080.",
        "lang": "es",
    },
]

EDIT_FILE_CASES = [
    {
        "user": "en hello.py cambia 'hola' por 'hello'",
        "filename": "hello.py",
        "old": "hola",
        "new": "hello",
        "aion_tpl": "Cambié 'hola' por 'hello' en {fn}.",
        "lang": "es",
    },
    {
        "user": "in app.py rename the function 'foo' to 'bar'",
        "filename": "app.py",
        "old": "foo",
        "new": "bar",
        "aion_tpl": "Renamed 'foo' to 'bar' in {fn}.",
        "lang": "en",
    },
    {
        "user": "en config.json sube el puerto a 9090",
        "filename": "config.json",
        "old": "8080",
        "new": "9090",
        "aion_tpl": "Cambié el puerto de 8080 a 9090 en {fn}.",
        "lang": "es",
    },
]

READ_FILE_CASES = [
    {
        "user": "léeme app.py",
        "filename": "app.py",
        "result": "import os\nimport sys\n# main entry",
        "aion_tpl": "El archivo {fn} importa os y sys, y tiene el comentario 'main entry'.",
        "lang": "es",
    },
    {
        "user": "show me requirements.txt",
        "filename": "requirements.txt",
        "result": "flask==3.0.0\ntorch==2.6.0",
        "aion_tpl": "{fn} declara flask 3.0.0 y torch 2.6.0.",
        "lang": "en",
    },
]

RUN_CODE_CASES = [
    {
        "user": "calcula el factorial de 5",
        "language": "python",
        "code": "import math; print(math.factorial(5))",
        "result": "120",
        "aion_tpl": "El factorial de 5 es 120.",
        "lang": "es",
    },
    {
        "user": "what's 2 to the 16th power?",
        "language": "python",
        "code": "print(2**16)",
        "result": "65536",
        "aion_tpl": "2^16 = 65536.",
        "lang": "en",
    },
    {
        "user": "lista los archivos del directorio actual",
        "language": "bash",
        "code": "ls",
        "result": "app.py\nrequirements.txt\nREADME.md",
        "aion_tpl": "Encontré 3 archivos: app.py, requirements.txt y README.md.",
        "lang": "es",
    },
]

CALL_API_CASES = [
    {
        "user": "consulta la API de github para el repo aion-c",
        "url": "https://api.github.com/repos/jesus/aion-c",
        "method": "GET",
        "result": '{"stars": 42, "forks": 7}',
        "aion_tpl": "El repo tiene 42 stars y 7 forks.",
        "lang": "es",
    },
    {
        "user": "fetch the weather for Madrid",
        "url": "https://api.weather.example/v1/madrid",
        "method": "GET",
        "result": '{"temp": 22, "condition": "sunny"}',
        "aion_tpl": "Madrid: 22°C and sunny.",
        "lang": "en",
    },
]

SEARCH_WEB_CASES = [
    {
        "user": "busca cómo crear un JWT en python",
        "query": "create JWT token in python",
        "result": "Use the PyJWT library: jwt.encode(payload, secret, algorithm='HS256')",
        "aion_tpl": "Usá la librería PyJWT: jwt.encode(payload, secret, algorithm='HS256').",
        "lang": "es",
    },
    {
        "user": "search for sqlite WAL mode",
        "query": "sqlite WAL mode",
        "result": "PRAGMA journal_mode=WAL enables write-ahead logging for better concurrency.",
        "aion_tpl": "Run `PRAGMA journal_mode=WAL` to enable write-ahead logging.",
        "lang": "en",
    },
]

SEARCH_MEM_CASES = [
    {
        "user": "qué sé sobre el proyecto aion?",
        "query": "aion project",
        "result": "[0.92] aion: MoSE architecture with 5 motors",
        "aion_tpl": "Tengo guardado: aion usa la arquitectura MoSE con 5 motores.",
        "lang": "es",
    },
    {
        "user": "what do I know about JWT?",
        "query": "JWT auth",
        "result": "[0.88] jwt_pattern: use PyJWT with HS256",
        "aion_tpl": "I have a note: use PyJWT with HS256 for JWT auth.",
        "lang": "en",
    },
]

STORE_MEM_CASES = [
    {
        "user": "recuerda que mi editor preferido es vscode",
        "key": "editor_preference",
        "value": "vscode",
        "domain": "user_model",
        "aion_tpl": "Listo, guardé que tu editor preferido es vscode.",
        "lang": "es",
    },
    {
        "user": "remember I work on the aion project",
        "key": "current_project",
        "value": "aion",
        "domain": "user_model",
        "aion_tpl": "Got it, noted that you're working on the aion project.",
        "lang": "en",
    },
]


def _tool_call(action: str, input_payload) -> str:
    """JSON serializa un tool call para el bloque [TOOL: ...]."""
    return json.dumps({"action": action, "input": input_payload}, ensure_ascii=False)


def generate_tool_calls(n: int = 3000, seed: int = 42) -> Iterator[CanonicalRecord]:
    """Genera n ejemplos con tool calls, distribuidos entre los 8 tools."""
    rng = random.Random(seed)

    # Pool: cada elemento es (action_name, case_dict, builder_callable)
    builders = []

    def add(action: str, cases, payload_fn, result_fn):
        for c in cases:
            builders.append((action, c, payload_fn, result_fn))

    add("write_file", WRITE_FILE_CASES,
        lambda c: {"path": c["filename"], "content": c["content"]},
        lambda c: f"File written: {c['filename']}")

    add("edit_file", EDIT_FILE_CASES,
        lambda c: {"path": c["filename"], "old": c["old"], "new": c["new"]},
        lambda c: f"Edited: {c['filename']} (1 replacement)")

    add("read_file", READ_FILE_CASES,
        lambda c: c["filename"],
        lambda c: c["result"])

    add("run_code", RUN_CODE_CASES,
        lambda c: {"language": c["language"], "code": c["code"]},
        lambda c: c["result"])

    add("call_api", CALL_API_CASES,
        lambda c: {"url": c["url"], "method": c["method"]},
        lambda c: c["result"])

    add("search_web", SEARCH_WEB_CASES,
        lambda c: c["query"],
        lambda c: c["result"])

    add("search_mem", SEARCH_MEM_CASES,
        lambda c: c["query"],
        lambda c: c["result"])

    add("store_mem", STORE_MEM_CASES,
        lambda c: {"key": c["key"], "value": c["value"], "domain": c["domain"]},
        lambda c: f"Stored: {c['key']} (domain={c['domain']})")

    for i in range(n):
        action, case, payload_fn, result_fn = rng.choice(builders)
        tool_text = _tool_call(action, payload_fn(case))
        result_text = result_fn(case)
        # Format the AION response: substituye {fn} si la tool tiene filename
        aion_text = case["aion_tpl"]
        if "{fn}" in aion_text and "filename" in case:
            aion_text = aion_text.format(fn=case["filename"])

        yield build_record(
            user=case["user"],
            aion=aion_text,
            tool=tool_text,
            result=result_text,
            domain="forge_c" if action in ("write_file", "edit_file", "read_file", "run_code") else "general",
            language=case["lang"],
            type="tool",
            metadata={"action": action, "seed_index": i},
        )


__all__ = ["generate_tool_calls"]
