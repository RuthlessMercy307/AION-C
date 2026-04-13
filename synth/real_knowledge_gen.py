"""
synth/real_knowledge_gen.py — Generador del dataset de conocimiento real.

Consume los pools de hechos reales en `synth/knowledge/` y produce
CanonicalRecord listos para serializarse como JSONL. Bilingüe (en/es),
categorías balanceadas, determinista via seed.

Uso:
    python -m synth.real_knowledge_gen --out datasets/real_knowledge.jsonl
    # Escribe aproximadamente 8K-12K records y un stats JSON adjunto.

El templating mapea cada hecho estructurado a 1-2 records canonical:
    {
      "q_en": "What is ...?",
      "a_en": "... is ...",
      "q_es": "¿Qué es...?",
      "a_es": "... es ...",
    }
    → record_en: [USER: What is ...?][AION: ... is ...][EOS]
    → record_es: [USER: ¿Qué es...?][AION: ... es ...][EOS]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

from synth.knowledge import all_facts
from synth.canonical_format import CanonicalRecord, build_record


# ════════════════════════════════════════════════════════════════════════════
# Mapeo de topic → domain canonical del dataset
# ════════════════════════════════════════════════════════════════════════════

TOPIC_TO_DOMAIN = {
    "programming": "forge_c",
    "math":        "axiom",
    "science":     "general",
    "history":     "general",
    "geography":   "general",
    "technology":  "forge_c",
    "language":    "general",
}


def _domain_for(fact: Dict) -> str:
    return TOPIC_TO_DOMAIN.get(fact["topic"], "general")


# ════════════════════════════════════════════════════════════════════════════
# Generación
# ════════════════════════════════════════════════════════════════════════════

# Variaciones de phrasing para escalar cada hecho a 3-4 records por idioma
# sin perder calidad (las respuestas son las mismas, solo la pregunta varía).

_EN_PHRASE_PREFIXES = [
    "",  # original
    "Briefly: ",
    "Quick question — ",
    "Hey, ",
]
# Los rewrites vienen en dos grupos:
#   1) "matching" — sólo aplican si el patrón coincide
#   2) "universal" — se aplican SIEMPRE (prefijos/sufijos que garantizan
#      una variante adicional sin requerir coincidencia)
# De esta forma cada hecho produce ≥4 variantes por idioma aunque su
# pregunta original no coincida con los patrones de "matching".

_EN_MATCHING_REWRITES = [
    lambda q: q.replace("What is ", "Can you explain "),
    lambda q: q.replace("What is ", "I'd like to know about "),
    lambda q: q.replace("What does ", "Could you tell me what "),
    lambda q: q.replace("Explain ", "Describe "),
    lambda q: q.replace("Tell me about ", "Could you introduce "),
    lambda q: q.replace("What was ", "Who remembers what "),
    lambda q: q.replace("Who was ", "Can you tell me who "),
]

_EN_UNIVERSAL = [
    lambda q: q,                          # original
    lambda q: "Quick question: " + q,
    lambda q: "Hi — " + q[0].lower() + q[1:],
    lambda q: q.rstrip("?.") + ", in plain English?",
    lambda q: "Short answer please: " + q,
    lambda q: "I was wondering, " + q[0].lower() + q[1:],
    lambda q: q.rstrip("?.") + ". I need a basic introduction.",
    lambda q: "Could you help me understand — " + q[0].lower() + q[1:],
]

_ES_MATCHING_REWRITES = [
    lambda q: q.replace("¿Qué es ", "¿Puedes explicarme "),
    lambda q: q.replace("¿Qué es ", "Me gustaría saber qué es "),
    lambda q: q.replace("¿Qué hace ", "¿Podrías decirme qué hace "),
    lambda q: q.replace("Explica ", "Descríbeme "),
    lambda q: q.replace("Háblame de ", "¿Me cuentas sobre "),
    lambda q: q.replace("¿Qué fue ", "¿Me recuerdas qué fue "),
    lambda q: q.replace("¿Quién fue ", "¿Puedes decirme quién fue "),
]

_ES_UNIVERSAL = [
    lambda q: q,                          # original
    lambda q: "Pregunta rápida: " + q,
    lambda q: "Hola — " + q[0].lower() + q[1:],
    lambda q: q.rstrip("?.") + ", en palabras sencillas?",
    lambda q: "Respuesta breve por favor: " + q,
    lambda q: "Me preguntaba: " + q[0].lower() + q[1:],
    lambda q: q.rstrip("?.") + ". Necesito una introducción básica.",
    lambda q: "Ayúdame a entender: " + q[0].lower() + q[1:],
]


def _apply_safely(fn, q: str):
    try:
        return fn(q)
    except Exception:
        return None


def _vary(q: str, matching, universal, max_variants: int) -> List[str]:
    """Produce hasta `max_variants` variantes únicas de la query."""
    out: List[str] = []
    seen = set()
    # Primero los universales (garantiza la variedad mínima)
    for fn in universal:
        v = _apply_safely(fn, q)
        if v and v not in seen:
            seen.add(v)
            out.append(v)
        if len(out) >= max_variants:
            return out
    # Luego los matching (aportan diversidad si aplican)
    for fn in matching:
        v = _apply_safely(fn, q)
        if v and v != q and v not in seen:
            seen.add(v)
            out.append(v)
        if len(out) >= max_variants:
            return out
    return out


def _vary_en(q: str, max_variants: int = 4) -> List[str]:
    return _vary(q, _EN_MATCHING_REWRITES, _EN_UNIVERSAL, max_variants)


def _vary_es(q: str, max_variants: int = 4) -> List[str]:
    return _vary(q, _ES_MATCHING_REWRITES, _ES_UNIVERSAL, max_variants)


def fact_to_records(fact: Dict, max_variants_per_lang: int = 6) -> List[CanonicalRecord]:
    """Convierte un hecho estructurado en 4-6 CanonicalRecord (EN y ES con variantes)."""
    domain = _domain_for(fact)
    out: List[CanonicalRecord] = []
    meta = {
        "topic": fact["topic"],
        "subtopic": fact.get("subtopic", ""),
        "difficulty": fact.get("difficulty", "medium"),
        "source": "curated_facts",
    }

    en_variants = _vary_en(fact["q_en"], max_variants=max_variants_per_lang)
    for q in en_variants:
        out.append(build_record(
            user=q,
            aion=fact["a_en"],
            domain=domain,
            language="en",
            type="real_knowledge",
            metadata=dict(meta),
        ))

    es_variants = _vary_es(fact["q_es"], max_variants=max_variants_per_lang)
    for q in es_variants:
        out.append(build_record(
            user=q,
            aion=fact["a_es"],
            domain=domain,
            language="es",
            type="real_knowledge",
            metadata=dict(meta),
        ))

    return out


def generate_all(seed: int = 2024) -> List[CanonicalRecord]:
    facts = all_facts()
    rng = random.Random(seed)
    # Shuffle de hechos para que las categorías estén entremezcladas
    rng.shuffle(facts)

    records: List[CanonicalRecord] = []
    for fact in facts:
        records.extend(fact_to_records(fact))

    return records


def write_jsonl(records: List[CanonicalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def compute_stats(records: List[CanonicalRecord]) -> Dict:
    by_topic: Dict[str, int] = {}
    by_subtopic: Dict[str, int] = {}
    by_lang: Dict[str, int] = {}
    by_domain: Dict[str, int] = {}
    eos_count = 0
    for r in records:
        topic = r.metadata.get("topic", "?")
        subtopic = r.metadata.get("subtopic", "?")
        by_topic[topic] = by_topic.get(topic, 0) + 1
        by_subtopic[subtopic] = by_subtopic.get(subtopic, 0) + 1
        by_lang[r.language] = by_lang.get(r.language, 0) + 1
        by_domain[r.domain] = by_domain.get(r.domain, 0) + 1
        if r.text.rstrip().endswith("[EOS]") and r.text.count("[EOS]") == 1:
            eos_count += 1
    return {
        "total": len(records),
        "eos_ok": eos_count,
        "eos_pct": round(100 * eos_count / max(len(records), 1), 2),
        "by_topic": by_topic,
        "by_subtopic": dict(sorted(by_subtopic.items(), key=lambda x: -x[1])),
        "by_language": by_lang,
        "by_domain": by_domain,
    }


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", type=str, default="datasets/real_knowledge.jsonl")
    args = p.parse_args()

    records = generate_all(seed=args.seed)
    out_path = Path(args.out)
    write_jsonl(records, out_path)

    stats = compute_stats(records)
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Generated {len(records)} real knowledge records -> {out_path}")
    print(f"EOS ok: {stats['eos_ok']}/{stats['total']} ({stats['eos_pct']}%)")
    print(f"  by topic: {stats['by_topic']}")
    print(f"  by language: {stats['by_language']}")


if __name__ == "__main__":  # pragma: no cover
    main()
