"""
synth/search_web_gen.py — Genera ejemplos de uso de la tool search_web.

Objetivo: el modelo aprende CUÁNDO buscar y CÓMO formular una query
buena. Cada ejemplo es un turno con:

    [USER: pregunta que requiere información no integrada]
    [TOOL: {"tool":"search_web","query":"query optimizada"}]
    [RESULT: resumen simulado de resultados]
    [AION: respuesta que cita lo encontrado]
    [EOS]

Los resultados son simulados pero plausibles (títulos de artículos
reales de Wikipedia donde aplica, con snippets cortos). Esto enseña
el PATRÓN, no hechos específicos — el hecho real lo trae search_web
en runtime.

Categorías:
    news          — noticias recientes / eventos del año
    api_new       — versiones o features nuevas de APIs
    current_data  — precios, stats, métricas actuales
    private       — datos del usuario que el modelo no tiene
    obscure       — temas especializados fuera del corpus
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from synth.canonical_format import CanonicalRecord, build_record


# ════════════════════════════════════════════════════════════════════════════
# Pools de templates por categoría
# ════════════════════════════════════════════════════════════════════════════

# Each entry: (user_q, optimized_query, simulated_result, aion_response)
# where {PLACEHOLDERS} are filled from context pools at generation time.

_NEWS_TEMPLATES_EN = [
    (
        "what happened in the {topic} world this week?",
        "{topic} news this week",
        "Top story: {topic} industry shifts reported on 2026-04 by major outlets. Several developments in the {subfield} area.",
        "Based on the search, the main {topic} news this week centers on developments in {subfield}. For the specific details you should click through to the source articles.",
    ),
    (
        "is there any breaking news about {topic} today?",
        "{topic} breaking news today",
        "Recent: article on {topic} from 2026-04-{day} discussing {subfield}.",
        "Yes — I found a recent article from 2026-04-{day} about {topic}, specifically covering {subfield}. Want me to search for more sources to cross-check?",
    ),
    (
        "what's the latest status of the {topic} situation?",
        "latest status {topic}",
        "Coverage from 2026-04 shows {topic} is still {state} with expected developments in {subfield}.",
        "According to sources dated 2026-04, the {topic} situation is currently {state}, with upcoming developments expected in {subfield}.",
    ),
]

_NEWS_TEMPLATES_ES = [
    (
        "¿qué pasó con {topic} esta semana?",
        "noticias {topic} esta semana",
        "Historia principal: cambios en la industria de {topic} reportados en 2026-04. Varios avances en el área de {subfield}.",
        "Según la búsqueda, la noticia principal sobre {topic} esta semana se centra en avances en {subfield}. Para detalles específicos, revisa los artículos fuente.",
    ),
    (
        "¿hay alguna noticia de última hora sobre {topic}?",
        "noticias última hora {topic}",
        "Reciente: artículo sobre {topic} del 2026-04-{day} discutiendo {subfield}.",
        "Sí — encontré un artículo reciente del 2026-04-{day} sobre {topic}, cubriendo específicamente {subfield}. ¿Busco más fuentes para contrastar?",
    ),
]

_NEWS_TOPICS = [
    ("AI", "ai large language models"),
    ("climate change", "climate policy"),
    ("space exploration", "space missions"),
    ("quantum computing", "quantum hardware"),
    ("robotics", "humanoid robots"),
    ("biotech", "gene editing"),
    ("cybersecurity", "zero-day exploits"),
    ("renewable energy", "battery storage"),
    ("semiconductors", "chip fabrication"),
    ("economy", "interest rates"),
]

_NEWS_TOPICS_ES = [
    ("IA", "modelos de lenguaje"),
    ("cambio climático", "política climática"),
    ("exploración espacial", "misiones espaciales"),
    ("computación cuántica", "hardware cuántico"),
    ("robótica", "robots humanoides"),
    ("biotecnología", "edición genética"),
    ("ciberseguridad", "exploits zero-day"),
    ("energías renovables", "baterías"),
    ("semiconductores", "fabricación de chips"),
    ("economía", "tasas de interés"),
]

_STATES = ["evolving", "stabilizing", "escalating", "under review", "progressing"]
_STATES_ES = ["en evolución", "estabilizándose", "escalando", "en revisión", "en progreso"]


_API_TEMPLATES_EN = [
    (
        "what's new in {framework} {version}?",
        "{framework} {version} release notes",
        "{framework} {version} released on 2026-04-{day}. New features include {feature}.",
        "{framework} {version} was released on 2026-04-{day}. Key new features: {feature}. Check the official changelog for the full list.",
    ),
    (
        "how do I use the new {api} endpoint?",
        "{api} endpoint documentation {framework}",
        "{framework} docs show {api} with params: id (required), filter (optional). Returns JSON.",
        "According to the {framework} docs, the {api} endpoint takes a required 'id' parameter and an optional 'filter'. It returns JSON. Example: GET /{api}?id=123.",
    ),
    (
        "what are the breaking changes in {framework} {version}?",
        "{framework} {version} breaking changes",
        "{framework} {version} deprecates {feature} and removes legacy {api} endpoint. Migration guide available.",
        "Breaking changes in {framework} {version}: deprecated {feature} and removed the legacy {api} endpoint. There's a migration guide if you need to update.",
    ),
]

_API_TEMPLATES_ES = [
    (
        "¿qué hay nuevo en {framework} {version}?",
        "{framework} {version} release notes",
        "{framework} {version} lanzado el 2026-04-{day}. Nuevas funciones: {feature}.",
        "{framework} {version} se lanzó el 2026-04-{day}. Funciones clave: {feature}. Revisa el changelog oficial para la lista completa.",
    ),
    (
        "¿cómo se usa el nuevo endpoint {api}?",
        "{api} endpoint documentación {framework}",
        "Los docs de {framework} muestran {api} con parámetros: id (requerido), filter (opcional). Devuelve JSON.",
        "Según los docs de {framework}, el endpoint {api} toma un 'id' requerido y un 'filter' opcional. Devuelve JSON. Ejemplo: GET /{api}?id=123.",
    ),
]

_FRAMEWORKS = [
    ("Next.js", "15.2"),
    ("React", "19.1"),
    ("Vue", "3.5"),
    ("Svelte", "5.0"),
    ("Django", "5.1"),
    ("Flask", "3.0"),
    ("FastAPI", "0.115"),
    ("Rails", "8.0"),
    ("Laravel", "11"),
    ("Spring Boot", "3.3"),
]

_APIS = ["/v2/users", "/v3/search", "/api/completions", "/v1/embeddings", "/chat/send"]
_FEATURES = [
    "streaming SSR",
    "partial prerender",
    "edge runtime",
    "server actions",
    "view transitions",
    "zero-config SSG",
    "structured outputs",
    "tool use",
]


_CURRENT_DATA_TEMPLATES_EN = [
    (
        "what is the current price of {asset}?",
        "{asset} price today",
        "Live price for {asset} as of 2026-04: approximately ${price} per unit.",
        "According to the search result, {asset} is trading around ${price} per unit as of 2026-04. Prices fluctuate — double-check on a live ticker for decisions.",
    ),
    (
        "how many active users does {service} have?",
        "{service} monthly active users 2026",
        "{service} reports {users} million monthly active users as of Q1 2026.",
        "Based on the latest numbers I found, {service} has approximately {users} million monthly active users as of Q1 2026.",
    ),
    (
        "what is the current stable version of {tool}?",
        "{tool} latest stable release",
        "{tool} {version} released 2026-04-{day}, now the latest stable.",
        "The current stable version of {tool} is {version}, released on 2026-04-{day}.",
    ),
]

_CURRENT_DATA_TEMPLATES_ES = [
    (
        "¿cuál es el precio actual de {asset}?",
        "precio {asset} hoy",
        "Precio en vivo de {asset} en 2026-04: aproximadamente ${price} por unidad.",
        "Según el resultado de búsqueda, {asset} cotiza alrededor de ${price} por unidad en 2026-04. Los precios fluctúan — verifícalo en un ticker en vivo para decisiones.",
    ),
    (
        "¿cuántos usuarios activos tiene {service}?",
        "{service} usuarios activos mensuales 2026",
        "{service} reporta {users} millones de usuarios activos mensuales en Q1 2026.",
        "Según los últimos datos que encontré, {service} tiene aproximadamente {users} millones de usuarios activos mensuales en Q1 2026.",
    ),
]

_ASSETS = ["Bitcoin", "Ethereum", "gold", "oil", "silver"]
_PRICES = ["67,500", "3,200", "2,150", "85", "29"]
_SERVICES = ["ChatGPT", "Claude.ai", "Gemini", "Llama.ai", "Copilot"]
_USERS = ["180", "120", "250", "90", "75"]
_TOOLS_STABLE = [
    ("Python", "3.13.2"),
    ("Node.js", "22.5.0"),
    ("Rust", "1.85"),
    ("Go", "1.22"),
    ("TypeScript", "5.6"),
]


_PRIVATE_TEMPLATES_EN = [
    (
        "what was the budget we agreed on for the {project} project?",
        None,  # no tool — model should refuse + explain why
        None,
        "I don't have access to your private records or past conversations about the {project} project. I can't search the web for that either — it's internal data. If you share the agreed budget with me, I can keep it in this session's memory.",
    ),
    (
        "when is my next meeting with {person}?",
        None,
        None,
        "I don't have access to your calendar, and searching the web wouldn't help here — meeting schedules are private. Check your calendar app directly, or paste the relevant invite and I can help interpret it.",
    ),
    (
        "what were the action items from yesterday's meeting?",
        None,
        None,
        "I have no record of your meeting — that's private data outside my reach. If you paste the meeting notes or a summary, I'll help you organize the action items.",
    ),
]

_PRIVATE_TEMPLATES_ES = [
    (
        "¿cuál fue el presupuesto que acordamos para el proyecto {project}?",
        None,
        None,
        "No tengo acceso a tus registros privados ni a conversaciones anteriores sobre el proyecto {project}. Tampoco puedo buscar eso en la web — son datos internos. Si me compartes el presupuesto, lo guardo en la memoria de esta sesión.",
    ),
    (
        "¿cuándo es mi próxima reunión con {person}?",
        None,
        None,
        "No tengo acceso a tu calendario, y buscar en la web no ayudaría — los horarios de reuniones son privados. Revisa tu calendario directamente, o pégame la invitación relevante y te ayudo a interpretarla.",
    ),
]

_PROJECTS = ["Apollo", "Nebula", "Orion", "Helios", "Titan"]
_PEOPLE = ["María", "Juan", "Sarah", "David", "the CEO"]


_OBSCURE_TEMPLATES_EN = [
    (
        "what is the {topic} algorithm used for in practice?",
        "{topic} algorithm use cases",
        "The {topic} algorithm, described in a 2024 paper by Smith et al., is used for {usecase}.",
        "Based on the search, the {topic} algorithm is primarily used for {usecase}, according to the Smith et al. paper from 2024.",
    ),
    (
        "how does the {technique} technique work?",
        "{technique} technique explained",
        "{technique} works by {mechanism}, commonly applied to {domain} problems.",
        "From what I found, {technique} works by {mechanism}. It's most commonly applied to {domain} problems.",
    ),
]

_OBSCURE_TOPICS_EN = [
    ("HNSW", "approximate nearest neighbor search in high dimensions"),
    ("Reed-Solomon", "error correction in data storage and transmission"),
    ("FFT", "converting between time and frequency domain representations"),
    ("Raft", "consensus in distributed systems"),
    ("Bloom filter", "probabilistic set membership testing"),
    ("MCTS", "decision-making in game trees and planning"),
    ("CRDTs", "collaborative editing and conflict-free replication"),
]

_TECHNIQUES = [
    ("knowledge distillation", "training a smaller student model to mimic a larger teacher", "deep learning"),
    ("mixed precision training", "using fp16 and fp32 together", "neural network training"),
    ("gradient checkpointing", "recomputing activations during backward pass to save memory", "large model training"),
    ("prompt engineering", "crafting the input to elicit desired model behavior", "LLM applications"),
    ("retrieval augmented generation", "combining a retriever with a generator model", "knowledge-grounded QA"),
]


# ════════════════════════════════════════════════════════════════════════════
# Generation
# ════════════════════════════════════════════════════════════════════════════

def _build_tool_call(query: str) -> str:
    return json.dumps({"tool": "search_web", "query": query})


def _make_search_record(
    user_text: str,
    tool_query: str,
    result: str,
    aion: str,
    language: str,
    subcategory: str,
) -> CanonicalRecord:
    return build_record(
        user=user_text,
        aion=aion,
        tool=_build_tool_call(tool_query),
        result=result,
        domain="metacognitive",
        language=language,
        type="search_web_usage",
        metadata={
            "subcategory": subcategory,
            "uses_tool": "search_web",
            "expected_motor_sequence": ["cora"],
        },
    )


def _make_refusal_record(
    user_text: str,
    aion: str,
    language: str,
    subcategory: str,
) -> CanonicalRecord:
    """Para datos privados: NO se llama el tool, el modelo explica por qué."""
    return build_record(
        user=user_text,
        aion=aion,
        domain="metacognitive",
        language=language,
        type="search_web_usage",
        metadata={
            "subcategory": subcategory,
            "uses_tool": None,
            "expected_motor_sequence": ["cora"],
        },
    )


def gen_news(n: int, rng: random.Random) -> List[CanonicalRecord]:
    out: List[CanonicalRecord] = []
    for i in range(n):
        if i % 2 == 0:
            tpl = rng.choice(_NEWS_TEMPLATES_EN)
            topic, subfield = rng.choice(_NEWS_TOPICS)
            state = rng.choice(_STATES)
            lang = "en"
        else:
            tpl = rng.choice(_NEWS_TEMPLATES_ES)
            topic, subfield = rng.choice(_NEWS_TOPICS_ES)
            state = rng.choice(_STATES_ES)
            lang = "es"
        day = rng.randint(1, 28)
        ctx = dict(topic=topic, subfield=subfield, state=state, day=day)
        user_q, tool_q, result, aion = [s.format(**ctx) for s in tpl]
        out.append(_make_search_record(user_q, tool_q, result, aion, lang, "news"))
    return out


def gen_api_new(n: int, rng: random.Random) -> List[CanonicalRecord]:
    out: List[CanonicalRecord] = []
    for i in range(n):
        if i % 2 == 0:
            tpl = rng.choice(_API_TEMPLATES_EN)
            lang = "en"
        else:
            tpl = rng.choice(_API_TEMPLATES_ES)
            lang = "es"
        fw, ver = rng.choice(_FRAMEWORKS)
        api = rng.choice(_APIS)
        feat = rng.choice(_FEATURES)
        day = rng.randint(1, 28)
        ctx = dict(framework=fw, version=ver, api=api, feature=feat, day=day)
        user_q, tool_q, result, aion = [s.format(**ctx) for s in tpl]
        out.append(_make_search_record(user_q, tool_q, result, aion, lang, "api_new"))
    return out


def gen_current_data(n: int, rng: random.Random) -> List[CanonicalRecord]:
    out: List[CanonicalRecord] = []
    for i in range(n):
        if i % 2 == 0:
            tpl = rng.choice(_CURRENT_DATA_TEMPLATES_EN)
            lang = "en"
        else:
            tpl = rng.choice(_CURRENT_DATA_TEMPLATES_ES)
            lang = "es"
        asset = rng.choice(_ASSETS)
        price = rng.choice(_PRICES)
        service = rng.choice(_SERVICES)
        users = rng.choice(_USERS)
        tool_name, version = rng.choice(_TOOLS_STABLE)
        day = rng.randint(1, 28)
        ctx = dict(asset=asset, price=price, service=service, users=users,
                   tool=tool_name, version=version, day=day)
        user_q, tool_q, result, aion = [s.format(**ctx) for s in tpl]
        out.append(_make_search_record(user_q, tool_q, result, aion, lang, "current_data"))
    return out


def gen_private(n: int, rng: random.Random) -> List[CanonicalRecord]:
    out: List[CanonicalRecord] = []
    for i in range(n):
        if i % 2 == 0:
            tpl = rng.choice(_PRIVATE_TEMPLATES_EN)
            lang = "en"
        else:
            tpl = rng.choice(_PRIVATE_TEMPLATES_ES)
            lang = "es"
        project = rng.choice(_PROJECTS)
        person = rng.choice(_PEOPLE)
        user_q = tpl[0].format(project=project, person=person)
        aion = tpl[3].format(project=project, person=person)
        out.append(_make_refusal_record(user_q, aion, lang, "private"))
    return out


def gen_obscure(n: int, rng: random.Random) -> List[CanonicalRecord]:
    out: List[CanonicalRecord] = []
    for i in range(n):
        tpl = rng.choice(_OBSCURE_TEMPLATES_EN)
        if "algorithm" in tpl[0]:
            topic, usecase = rng.choice(_OBSCURE_TOPICS_EN)
            ctx = dict(topic=topic, usecase=usecase)
        else:
            tech, mechanism, domain = rng.choice(_TECHNIQUES)
            ctx = dict(technique=tech, mechanism=mechanism, domain=domain)
        user_q, tool_q, result, aion = [s.format(**ctx) for s in tpl]
        out.append(_make_search_record(user_q, tool_q, result, aion, "en", "obscure"))
    return out


GENERATORS: List[Tuple[str, callable]] = [
    ("news",         gen_news),
    ("api_new",      gen_api_new),
    ("current_data", gen_current_data),
    ("private",      gen_private),
    ("obscure",      gen_obscure),
]


def generate_all(target_total: int = 3000, seed: int = 2024) -> List[CanonicalRecord]:
    rng = random.Random(seed)
    per_cat = target_total // len(GENERATORS)
    out: List[CanonicalRecord] = []
    for _, fn in GENERATORS:
        out.extend(fn(per_cat, rng))
    return out


def write_jsonl(records: List[CanonicalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=3000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", type=str, default="datasets/search_web_3k.jsonl")
    args = p.parse_args()
    records = generate_all(target_total=args.total, seed=args.seed)
    out_path = Path(args.out)
    write_jsonl(records, out_path)
    by_sub = {}
    for r in records:
        sub = r.metadata.get("subcategory", "?")
        by_sub[sub] = by_sub.get(sub, 0) + 1
    print(f"Generated {len(records)} search_web usage records -> {out_path}")
    print(f"  by subcategory: {by_sub}")


if __name__ == "__main__":  # pragma: no cover
    main()
