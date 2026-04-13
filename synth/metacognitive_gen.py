"""
synth/metacognitive_gen.py — Generador de ejemplos metacognitivos (Parte 25+).

Cinco categorías × 500 ejemplos cada una = 2,500 registros canónicos listos
para concatenarse al dataset canonical principal.

Categorías:
    out_of_knowledge         — usar search_web + citar
    propose_learning         — proponer auto-learn en vez de alucinar
    context_overflow         — resumir historial y responder breve
    out_of_capability        — honestidad sobre límites duros
    low_confidence_disclaimer — respuesta con marcadores verbales de certeza

Uso:
    python -m synth.metacognitive_gen \
        --count 2500 \
        --out datasets/metacognitive_2500.jsonl

Seed determinista por default → los 2,500 ejemplos son siempre los mismos.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from synth.canonical_format import (
    CanonicalRecord,
    CanonicalTurn,
    build_record,
)


# ════════════════════════════════════════════════════════════════════════════
# Pools de contenido
# ════════════════════════════════════════════════════════════════════════════

# Cada pool: (query_template, tool_result, answer_template), bilingüe.

# ── out_of_knowledge ──────────────────────────────────────────────────────
OOK_EN = [
    (
        "what's the latest stable version of {pkg}?",
        "{pkg} {ver} released {date} per official release notes",
        "According to the official release notes, the latest stable version of {pkg} is {ver}, released on {date}.",
    ),
    (
        "did the {fw} team ship anything new this week?",
        "{fw} {ver} announced on {date}, adds {feat}",
        "Yes — {fw} {ver} was announced on {date} and it adds {feat}. I pulled that from the official changelog.",
    ),
    (
        "what are the current pricing tiers for {svc}?",
        "{svc} pricing (retrieved today): free, plus $20/mo, pro $100/mo",
        "Per the pricing page I just checked, {svc} has three tiers: free, plus at $20/mo and pro at $100/mo. These can change so I recommend re-checking before a purchase decision.",
    ),
    (
        "is the {api} endpoint still deprecated in {ver}?",
        "{api} deprecated in {ver}, removed in {ver_next}; use {repl} instead",
        "Based on the docs I just queried, {api} was deprecated in {ver} and is scheduled for removal in {ver_next}. The recommended replacement is {repl}.",
    ),
    (
        "what's the current status of the {pkg} issue {num}?",
        "issue #{num} on {pkg}: status=open, last updated {date}, assignee=maintainer",
        "Issue #{num} on {pkg} is currently open (last updated {date}) and assigned to a maintainer. I don't have visibility into what they said in comments unless you want me to fetch them.",
    ),
]
OOK_ES = [
    (
        "¿cuál es la versión estable más reciente de {pkg}?",
        "{pkg} {ver} publicado {date} según release notes oficial",
        "Según las release notes oficiales, la versión estable más reciente de {pkg} es {ver}, publicada el {date}.",
    ),
    (
        "¿el equipo de {fw} sacó algo nuevo esta semana?",
        "{fw} {ver} anunciado el {date}, agrega {feat}",
        "Sí — {fw} {ver} se anunció el {date} y agrega {feat}. Lo saqué del changelog oficial.",
    ),
    (
        "¿cuánto cuesta hoy {svc}?",
        "{svc} pricing (retrieved today): free, plus $20/mes, pro $100/mes",
        "Según la página de precios que acabo de consultar, {svc} tiene tres tiers: free, plus a $20/mes y pro a $100/mes. Los precios pueden cambiar — verificá antes de comprar.",
    ),
    (
        "¿el endpoint {api} sigue deprecated en {ver}?",
        "{api} deprecated en {ver}, se elimina en {ver_next}; usar {repl}",
        "Según los docs que acabo de consultar, {api} quedó deprecated en {ver} y se elimina en {ver_next}. El reemplazo recomendado es {repl}.",
    ),
    (
        "¿en qué estado está el issue {num} del repo {pkg}?",
        "issue #{num} en {pkg}: estado=abierto, último update {date}, asignado a maintainer",
        "El issue #{num} en {pkg} está abierto (último update {date}) y asignado a un maintainer. No veo el detalle de los comentarios salvo que quieras que los traiga.",
    ),
]

OOK_PKGS = ["Rust", "Python", "Node.js", "PyTorch", "Transformers", "Deno", "Bun"]
OOK_FWS = ["Next.js", "Svelte", "React", "Vue", "Astro", "Remix"]
OOK_SVCS = ["ChatGPT", "Claude API", "Gemini", "Vercel", "Supabase"]
OOK_APIS = ["v1/completions", "/users.me", "/chat/send", "/embeddings"]


# ── propose_learning ──────────────────────────────────────────────────────
PL_EN = [
    (
        "explain how to use the internal '{lib}' library we use at my company",
        "I don't have that knowledge integrated — I never saw '{lib}' in my training or in this session's memory. Would you like me to learn it? If you give me a document or a repo, I can start an auto-learn cycle and in ~5 minutes I'll have a specialized adapter. Meanwhile, tell me briefly what it does and I'll help with what I can infer.",
    ),
    (
        "how does our proprietary {dom} pipeline work?",
        "I have no training data on your internal pipeline — proprietary systems don't leak into public corpora. I can learn it on demand: point me at a README or a design doc and I'll create an adapter. Or describe the main stages and I'll reason from first principles for this turn.",
    ),
    (
        "tell me about the {proj} project at Acme Corp",
        "I have zero visibility into internal Acme projects and I won't guess. If you can paste the design doc or a summary, I'll happily learn it (auto-learn cycle, ~5 min, creates an adapter). Otherwise I can discuss similar public projects if that helps calibrate.",
    ),
    (
        "help me debug this custom DSL we built called {dsl}",
        "'{dsl}' isn't in my training. Two options: (1) paste its grammar or a couple of example programs and I'll infer enough to help you THIS turn; (2) propose an auto-learn cycle — give me the full spec and I'll create a FORGE-C adapter for it.",
    ),
    (
        "what are our SLAs with {vendor}?",
        "I don't have access to your vendor contracts — that's private data. I can't invent numbers here. If you paste the SLA doc, I'll parse it and keep the relevant clauses in session memory. No auto-learn on legal docs by default unless you explicitly ask.",
    ),
]
PL_ES = [
    (
        "explicame cómo usar la librería interna '{lib}' de mi empresa",
        "No tengo ese conocimiento integrado — nunca vi '{lib}' en mi training ni en la memoria de esta sesión. ¿Querés que lo aprenda? Si me das un documento o un repo, puedo iniciar un ciclo de auto-learn y en ~5 minutos tendré un adapter especializado. Mientras tanto, decime brevemente qué hace y te ayudo con lo que pueda inferir.",
    ),
    (
        "¿cómo funciona nuestro pipeline propietario de {dom}?",
        "No tengo training sobre tu pipeline interno — los sistemas propietarios no están en los corpora públicos. Lo puedo aprender on-demand: pasame un README o un design doc y te armo un adapter. O describime las etapas principales y razono desde cero por esta vez.",
    ),
    (
        "contame sobre el proyecto {proj} de Acme Corp",
        "No tengo visibilidad de proyectos internos de Acme y no voy a adivinar. Si me pegás el design doc o un resumen, lo aprendo con gusto (ciclo auto-learn, ~5 min, crea un adapter). Si no, puedo discutir proyectos públicos similares si te sirve de referencia.",
    ),
    (
        "ayudame a debuggear este DSL que construimos llamado {dsl}",
        "'{dsl}' no está en mi training. Dos opciones: (1) pegame la gramática o un par de programas de ejemplo y infiero lo suficiente para ayudarte ESTE turno; (2) proponé un ciclo de auto-learn — dame el spec completo y creo un adapter de FORGE-C para esto.",
    ),
    (
        "¿qué SLAs tenemos con {vendor}?",
        "No tengo acceso a los contratos de tus vendors — eso es data privada. No puedo inventar números acá. Si pegás el SLA, lo parseo y guardo las cláusulas relevantes en memoria de sesión. Auto-learn sobre docs legales no está en el default, pedímelo explícito si querés.",
    ),
]

PL_LIBS = ["auth-core-v3", "payments-edge", "frontend-primitives", "graph-store-v2"]
PL_DOMS = ["ETL", "ingestion", "risk scoring", "ranking"]
PL_PROJS = ["Nebula", "Orion", "Titan-X", "Helios"]
PL_DSLS = ["calc-lang", "configql", "policy-dsl", "rulex"]
PL_VENDORS = ["AWS", "Snowflake", "Datadog", "Cloudflare"]


# ── context_overflow ──────────────────────────────────────────────────────
# Generamos multi-turn largo → agregamos [SUMMARY:] → respuesta breve coherente
CTX_EN_SCENARIOS = [
    {
        "turns": [
            ("I'm hitting an intermittent 504 on /search",  "Can you describe the pattern — random or load-related?"),
            ("It only happens during peak hours",           "Got it. Is it the ALB timing out or the upstream service?"),
            ("ALB drops the conn at 30 seconds",             "So the upstream is fine past 30s but the ALB gives up. That's an idle_timeout config."),
            ("yes, I'd like to bump it to 60",               "Think twice — long timeouts hide real issues. How slow are your slowest queries?"),
            ("p99 is 45s",                                   "Then 60s is reasonable as a short-term fix while you profile the slow queries. Do you want the terraform snippet?"),
            ("yes please",                                   "Here it is: `lb_target_group { idle_timeout_seconds = 60 }`. Apply in staging first."),
            ("applied, staging looks good",                  "Good. Do you want to roll to prod or wait for 24h of stats?"),
            ("let's wait 24h and check",                     "Smart. Set a reminder and come back tomorrow with the p99 graph."),
        ],
        "summary": "user reporting intermittent 504 on /search during peak; identified ALB idle_timeout=30s dropping slow p99 queries (45s); agreed on bumping to 60s in staging as short-term fix; staging applied ok; waiting 24h before prod rollout",
        "final_user": "ok, it's been 24h — the p99 graph looks stable",
        "final_aion": "Great — if p99 held steady under 50s and there were no 504s in the last 24h, you're safe to roll the `idle_timeout_seconds = 60` change to prod. Do it in off-peak if you can, and keep the slow-query profiling as follow-up work (the real fix).",
    },
    {
        "turns": [
            ("our vector search is slow",                    "How slow, on what index size, what hardware?"),
            ("2s per query, 10M vectors, A10 GPU",           "That's an order of magnitude off. What index type?"),
            ("flat L2",                                      "There's your problem. Flat is O(N) — at 10M you need IVF or HNSW."),
            ("which one?",                                   "HNSW is faster for queries, IVF is smaller in RAM. What's your constraint?"),
            ("RAM is tight, 32GB only",                      "Then IVF_PQ. You'll trade a little recall for a lot of memory savings."),
            ("what recall should I expect?",                 "For 10M vectors with IVF_PQ at 256 subquantizers you'll see around 92-94% recall@10 vs flat baseline."),
            ("acceptable. how do I build it?",               "Use faiss: `faiss.index_factory(d, 'IVF4096,PQ64')` then train on a sample of 100k vectors."),
        ],
        "summary": "user has slow vector search (2s/query) with 10M vectors on A10 GPU; current index is flat L2 (O(N)); RAM constraint 32GB rules out HNSW; agreed on IVF_PQ; expected recall ~93%",
        "final_user": "ok I'll try that. what's the training time?",
        "final_aion": "Training IVF_PQ with 100k sample vectors and 4096 centroids takes ~3-5 minutes on an A10. Build the full index after, that's another 10-15 min for 10M vectors. After that queries drop from 2s to roughly 5-10ms.",
    },
]

CTX_ES_SCENARIOS = [
    {
        "turns": [
            ("tengo un 504 intermitente en /buscar",           "¿aparece en horas pico o random?"),
            ("sólo en pico",                                  "Es el ALB o el upstream que timeoutea?"),
            ("el ALB a los 30s",                              "El upstream aguanta más de 30s pero el ALB se rinde. Es idle_timeout."),
            ("quiero subirlo a 60",                           "Pensalo dos veces — timeouts largos esconden problemas reales. ¿Cuál es tu p99?"),
            ("p99 45s",                                       "Entonces 60s es razonable como fix temporal mientras profileás las queries lentas."),
            ("dale",                                          "`lb_target_group { idle_timeout_seconds = 60 }`. Aplicá primero en staging."),
            ("aplicado, staging bien",                        "Perfecto. ¿Vas a prod o esperás 24h?"),
            ("esperamos 24h",                                 "Smart. Poneme un reminder y volvé mañana con el gráfico p99."),
        ],
        "summary": "usuario reporta 504 intermitente en /buscar en pico; ALB con idle_timeout=30s corta las queries lentas (p99=45s); acuerdo de subirlo a 60s en staging como fix temporal; staging ok; espera 24h antes de prod",
        "final_user": "pasaron las 24h, el p99 quedó estable",
        "final_aion": "Listo — si p99 se mantuvo bajo 50s y no hubo 504s en las últimas 24h, podés rollear `idle_timeout_seconds = 60` a prod. Hacelo en horario de bajo tráfico si podés. Y dejá el profiling de las queries lentas como follow-up — esa es la fix real.",
    },
]


# ── out_of_capability ────────────────────────────────────────────────────
OOC_EN = [
    (
        "write me a complete Linux kernel from scratch with scheduler and VM manager",
        "This exceeds what I can do in a useful single response. A real Linux kernel is hundreds of thousands of lines with hardware-specific design. Proposals: (1) I can scaffold a minimal xv6-style kernel in C (~1000 lines); (2) I can explain Linux's high-level architecture; (3) if you pick a concrete subsystem (e.g. round-robin scheduler), I can implement that well. Which do you prefer?",
    ),
    (
        "prove the Riemann hypothesis for me",
        "Nobody has proven Riemann — that's a million-dollar open problem I'm certainly not going to solve here. What I can do: explain the statement rigorously, walk through the known partial results (proof it holds for all zeros with |Im| up to ~10^13), and discuss the approaches that have attacked it. Which would help?",
    ),
    (
        "simulate a full quantum computer with 100 qubits",
        "A 100-qubit simulation needs 2^100 complex amplitudes in memory — physically impossible on any classical machine, not just mine. If you want, I can (a) write you a simulator for a small number of qubits (up to ~25 realistically) using Qiskit or similar, (b) explain the structure of the state vector and why it blows up, or (c) point you at IBM's real quantum hardware if you want actual 100-qubit circuits.",
    ),
    (
        "reverse engineer this compiled binary and give me the full original C code",
        "Exact recovery of the original C source is not generally possible — compilation is lossy (variable names, comments, macro expansion all gone). What I CAN do: disassemble a small portion and explain the control flow, identify library function calls, and reconstruct plausible C that has the same behavior — but it won't be the original. Want me to start with a specific function?",
    ),
    (
        "design a perfect no-failure distributed database with strong consistency and infinite scale",
        "That's the CAP theorem speaking — there's no such thing as a system that is simultaneously consistent, available, and partition-tolerant under network failures. Every real system picks a trade-off. What are YOUR actual constraints (latency targets, failure tolerance, geography)? I can design for those.",
    ),
]
OOC_ES = [
    (
        "escribime un kernel Linux completo desde cero con scheduler y VM manager",
        "Esto excede lo que puedo hacer en una respuesta útil. Un kernel Linux real son cientos de miles de líneas con decisiones específicas del hardware. Propuestas: (1) te armo un esqueleto mínimo tipo xv6 en C (~1000 líneas); (2) te explico la arquitectura alta de Linux; (3) si elegís un subsistema concreto (ej. scheduler round-robin), ese sí lo implemento bien. ¿Cuál preferís?",
    ),
    (
        "demostrame la hipótesis de Riemann",
        "Nadie demostró Riemann — es un problema abierto de un millón de dólares, no lo voy a resolver acá. Lo que SÍ puedo: explicarte el enunciado con rigor, recorrer los resultados parciales conocidos (probado para ceros con |Im| hasta ~10^13), y discutir los abordajes que se intentaron. ¿Cuál te sirve?",
    ),
    (
        "simulame una computadora cuántica completa de 100 qubits",
        "Una simulación de 100 qubits necesita 2^100 amplitudes complejas en memoria — físicamente imposible en cualquier máquina clásica. Lo que puedo hacer: (a) simulador para pocos qubits (hasta ~25 realista) con Qiskit; (b) explicar la estructura del state vector y por qué explota; (c) apuntarte al hardware cuántico real de IBM si querés circuitos de 100 qubits de verdad.",
    ),
    (
        "reversá este binario compilado y devolveme el código C original",
        "Recuperar el C original exacto no es posible en general — la compilación pierde info (nombres de variables, comentarios, macros). Lo que SÍ puedo: desensamblar una parte, explicar el flujo de control, identificar calls a librerías y reconstruir C plausible con el mismo comportamiento — pero no será el original. ¿Empezamos por una función específica?",
    ),
    (
        "diseñá una base de datos distribuida perfecta, sin fallos, con consistencia fuerte y escala infinita",
        "Eso choca con el teorema CAP — no existe un sistema simultáneamente consistente, disponible y tolerante a particiones bajo fallos de red. Todos los reales eligen un trade-off. ¿Cuáles son TUS restricciones reales (latencia, tolerancia a fallos, geografía)? Diseño para eso.",
    ),
]


# ── low_confidence_disclaimer ────────────────────────────────────────────
LCD_EN = [
    (
        "how much energy does training a 1B model for 10h on a 4090 use?",
        "I'm not 100% sure — it depends on batch size and real utilization — but I'd put it around 4-5 kWh: a 4090 has TDP 450W, at ~80% utilization that's 360W sustained, and 360W × 10h = 3.6 kWh, plus ~10% overhead for the rest of the machine. If you want the exact number, send me the nvidia-smi log and I'll compute it.",
    ),
    (
        "what's the half-life of caffeine in a healthy adult?",
        "I recall it's around 5-6 hours on average, but there's wide individual variation — genetics (CYP1A2 polymorphism), liver function, pregnancy and some medications can push it from 3h to 10h. If you need it precise for a specific case, check with a doctor or pharmacist.",
    ),
    (
        "how many parameters does GPT-3 davinci have?",
        "I'm pretty sure it's 175B parameters (the original GPT-3 paper) — I'm confident in that specific number because it was heavily cited. But OpenAI released a few variants and the exact naming got confusing, so if you need it for a paper, double-check against the arXiv preprint (Brown et al., 2020).",
    ),
    (
        "what was the GDP of Japan in 2020?",
        "I recall it was around $5 trillion USD, roughly — not an exact figure from me. For anything you'd cite, pull from the World Bank or IMF databases where the number is definitive.",
    ),
    (
        "does Rust's borrow checker detect use-after-free at compile time?",
        "Yes — that's one of Rust's main guarantees. I'm very confident here: the borrow checker enforces that no reference outlives its referent, which prevents use-after-free at compile time. The only ways to get UAF in Rust are via `unsafe` or via FFI with C code.",
    ),
]
LCD_ES = [
    (
        "¿cuánta energía consume entrenar un modelo 1B 10h en una 4090?",
        "No estoy 100% seguro porque depende del batch size y del utilization real, pero diría que en el orden de 4-5 kWh: una 4090 tiene TDP 450W, a 80% utilización ≈ 360W sostenidos, y 360W × 10h = 3.6 kWh, más ~10% de overhead del resto de la máquina. Si querés el exacto, pasame el nvidia-smi log del run y lo calculo.",
    ),
    (
        "¿cuál es la vida media de la cafeína en un adulto sano?",
        "Me suena que es cerca de 5-6 horas en promedio, pero hay mucha variación individual — genética (polimorfismo de CYP1A2), función hepática, embarazo y algunos medicamentos lo pueden mover de 3h a 10h. Si lo necesitás preciso para un caso específico, consultá con un médico o farmacéutico.",
    ),
    (
        "¿cuántos parámetros tiene GPT-3 davinci?",
        "Estoy bastante seguro que son 175B (el paper original de GPT-3) — confío en ese número porque fue citado mil veces. Pero OpenAI sacó varias variantes y el naming quedó confuso, así que si lo necesitás para un paper, chequealo contra el preprint de arXiv (Brown et al., 2020).",
    ),
    (
        "¿cuál fue el PBI de Japón en 2020?",
        "Me suena que fue cerca de 5 billones de dólares, aproximadamente — no es un número exacto de memoria. Para algo que vayas a citar, sacalo del World Bank o del FMI donde el dato es definitivo.",
    ),
    (
        "¿el borrow checker de Rust detecta use-after-free en compile time?",
        "Sí — es una de las garantías principales de Rust. Acá sí estoy muy seguro: el borrow checker fuerza que ninguna referencia viva más que su referente, lo que previene use-after-free en compile time. Las únicas formas de tener UAF en Rust son vía `unsafe` o FFI con código C.",
    ),
]


# ════════════════════════════════════════════════════════════════════════════
# Rellenadores
# ════════════════════════════════════════════════════════════════════════════

def _fill_ook(template: str, rng: random.Random) -> Dict[str, str]:
    return {
        "pkg": rng.choice(OOK_PKGS),
        "ver": f"{rng.randint(1,20)}.{rng.randint(0,15)}.{rng.randint(0,9)}",
        "ver_next": f"{rng.randint(1,20)}.{rng.randint(16,30)}.0",
        "date": f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        "fw": rng.choice(OOK_FWS),
        "feat": rng.choice([
            "streaming SSR", "parallel routes", "partial prerender",
            "edge runtime", "turbo mode", "zero-config SSG",
        ]),
        "svc": rng.choice(OOK_SVCS),
        "api": rng.choice(OOK_APIS),
        "repl": rng.choice(["v2/completions", "/me", "/conversations", "/embed"]),
        "num": str(rng.randint(100, 9999)),
    }


def _fill_pl(template: str, rng: random.Random) -> Dict[str, str]:
    return {
        "lib": rng.choice(PL_LIBS),
        "dom": rng.choice(PL_DOMS),
        "proj": rng.choice(PL_PROJS),
        "dsl": rng.choice(PL_DSLS),
        "vendor": rng.choice(PL_VENDORS),
    }


# ════════════════════════════════════════════════════════════════════════════
# Generadores por categoría
# ════════════════════════════════════════════════════════════════════════════

def gen_out_of_knowledge(n: int, rng: random.Random) -> List[CanonicalRecord]:
    out: List[CanonicalRecord] = []
    pools = [(OOK_EN, "en"), (OOK_ES, "es")]
    for i in range(n):
        pool, lang = pools[i % 2]
        q_tpl, r_tpl, a_tpl = rng.choice(pool)
        fill = _fill_ook(q_tpl, rng)
        user = q_tpl.format(**fill)
        tool_json = json.dumps({"tool": "search_web", "query": user})
        result = r_tpl.format(**fill)
        aion = a_tpl.format(**fill)
        rec = build_record(
            user=user,
            aion=aion,
            tool=tool_json,
            result=result,
            domain="metacognitive",
            language=lang,
            type="metacognitive",
            metadata={
                "subcategory": "out_of_knowledge",
                "expected_motor_sequence": ["cora"],
                "uses_tool": "search_web",
            },
        )
        out.append(rec)
    return out


def gen_propose_learning(n: int, rng: random.Random) -> List[CanonicalRecord]:
    out: List[CanonicalRecord] = []
    pools = [(PL_EN, "en"), (PL_ES, "es")]
    for i in range(n):
        pool, lang = pools[i % 2]
        q_tpl, a_tpl = rng.choice(pool)
        fill = _fill_pl(q_tpl, rng)
        user = q_tpl.format(**fill)
        aion = a_tpl.format(**fill)
        # 30% de los casos son multi-turn: el usuario responde y el modelo actúa
        is_multi = rng.random() < 0.30
        extra_turns = None
        if is_multi:
            if lang == "en":
                follow_user = rng.choice([
                    "no, just answer with what you have",
                    "yes, please start the auto-learn cycle",
                    "here's the readme: <paste>",
                ])
                follow_aion = (
                    "Got it — here's what I can say from general knowledge, marked as uncertain. "
                    "I'll add a disclaimer at the end."
                )
                if "auto-learn" in follow_user:
                    follow_aion = (
                        "Starting auto-learn now. I'll create an adapter and ping you when it's ready."
                    )
            else:
                follow_user = rng.choice([
                    "no, respondé con lo que tengas",
                    "dale, iniciá el auto-learn",
                    "ahí va el readme: <paste>",
                ])
                follow_aion = (
                    "Ok — te respondo con lo general que sí sé, marcado como incierto. "
                    "Agrego un disclaimer al final."
                )
                if "auto-learn" in follow_user:
                    follow_aion = (
                        "Iniciando auto-learn. Creo el adapter y te aviso cuando esté listo."
                    )
            extra_turns = [CanonicalTurn(user=follow_user, aion=follow_aion)]
        rec = build_record(
            user=user,
            aion=aion,
            extra_turns=extra_turns,
            domain="metacognitive",
            language=lang,
            type="metacognitive",
            metadata={
                "subcategory": "propose_learning",
                "expected_motor_sequence": ["cora"],
            },
        )
        out.append(rec)
    return out


def gen_context_overflow(n: int, rng: random.Random) -> List[CanonicalRecord]:
    out: List[CanonicalRecord] = []
    pools = [(CTX_EN_SCENARIOS, "en"), (CTX_ES_SCENARIOS, "es")]
    for i in range(n):
        pool, lang = pools[i % 2]
        scen = rng.choice(pool)
        # Decidimos: 60% de casos llevan el historial completo + [SUMMARY:]
        #            40% SÓLO llevan el [SUMMARY:] (test de compacto puro)
        include_full_history = rng.random() < 0.60

        if include_full_history:
            first_u, first_a = scen["turns"][0]
            extra = [CanonicalTurn(user=u, aion=a) for u, a in scen["turns"][1:]]
            # El summary es un [MEM:] que antecede al USER final
            extra.append(CanonicalTurn(
                user=scen["final_user"],
                aion=scen["final_aion"],
            ))
            rec = build_record(
                user=first_u, aion=first_a,
                mem=f"SUMMARY: {scen['summary']}",
                extra_turns=extra,
                domain="metacognitive",
                language=lang,
                type="metacognitive",
                metadata={
                    "subcategory": "context_overflow",
                    "mode": "full_history_plus_summary",
                    "expected_motor_sequence": ["cora"],
                },
            )
        else:
            # Sólo summary + pregunta final: entrena a confiar en el resumen
            rec = build_record(
                user=scen["final_user"],
                aion=scen["final_aion"],
                mem=f"SUMMARY: {scen['summary']}",
                domain="metacognitive",
                language=lang,
                type="metacognitive",
                metadata={
                    "subcategory": "context_overflow",
                    "mode": "summary_only",
                    "expected_motor_sequence": ["cora"],
                },
            )
        out.append(rec)
    return out


def gen_out_of_capability(n: int, rng: random.Random) -> List[CanonicalRecord]:
    out: List[CanonicalRecord] = []
    pools = [(OOC_EN, "en"), (OOC_ES, "es")]
    for i in range(n):
        pool, lang = pools[i % 2]
        user, aion = rng.choice(pool)
        rec = build_record(
            user=user,
            aion=aion,
            domain="metacognitive",
            language=lang,
            type="metacognitive",
            metadata={
                "subcategory": "out_of_capability",
                "expected_motor_sequence": ["cora"],
            },
        )
        out.append(rec)
    return out


def gen_low_confidence(n: int, rng: random.Random) -> List[CanonicalRecord]:
    out: List[CanonicalRecord] = []
    pools = [(LCD_EN, "en"), (LCD_ES, "es")]
    for i in range(n):
        pool, lang = pools[i % 2]
        user, aion = rng.choice(pool)
        rec = build_record(
            user=user,
            aion=aion,
            domain="metacognitive",
            language=lang,
            type="metacognitive",
            metadata={
                "subcategory": "low_confidence_disclaimer",
                "expected_motor_sequence": ["cora"],
            },
        )
        out.append(rec)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Pipeline completo
# ════════════════════════════════════════════════════════════════════════════

GENERATORS: List[Tuple[str, callable]] = [
    ("out_of_knowledge",         gen_out_of_knowledge),
    ("propose_learning",         gen_propose_learning),
    ("context_overflow",         gen_context_overflow),
    ("out_of_capability",        gen_out_of_capability),
    ("low_confidence_disclaimer", gen_low_confidence),
]


def generate_all(per_category: int = 500, seed: int = 2024) -> List[CanonicalRecord]:
    """Genera per_category × 5 ejemplos metacognitivos.

    Returns: lista total de CanonicalRecord.
    """
    rng = random.Random(seed)
    out: List[CanonicalRecord] = []
    for name, fn in GENERATORS:
        sub = fn(per_category, rng)
        out.extend(sub)
    return out


def write_jsonl(records: List[CanonicalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-category", type=int, default=500)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", type=str, default="datasets/metacognitive_2500.jsonl")
    args = p.parse_args()

    records = generate_all(per_category=args.per_category, seed=args.seed)
    out_path = Path(args.out)
    write_jsonl(records, out_path)

    # Stats
    by_sub: Dict[str, int] = {}
    by_lang: Dict[str, int] = {0: 0, 1: 0}  # type: ignore[dict-item]
    by_lang_dict: Dict[str, int] = {}
    for r in records:
        sub = r.metadata.get("subcategory", "?")
        by_sub[sub] = by_sub.get(sub, 0) + 1
        by_lang_dict[r.language] = by_lang_dict.get(r.language, 0) + 1
    print(f"Generated {len(records)} metacognitive records -> {out_path}")
    print(f"  by subcategory: {by_sub}")
    print(f"  by language: {by_lang_dict}")


if __name__ == "__main__":  # pragma: no cover
    main()
