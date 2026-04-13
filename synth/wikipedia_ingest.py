"""
synth/wikipedia_ingest.py — Scraper opcional de Wikipedia para escalar el
dataset de conocimiento real offline.

Uso típico (después de que `real_knowledge_gen.py` haya producido el seed
inicial, si querés más volumen):

    python -m synth.wikipedia_ingest \
        --seeds topics_en.txt topics_es.txt \
        --out datasets/wiki_ingested.jsonl \
        --max-per-topic 3 \
        --pause 0.5

Cómo funciona:
    1. Lee una lista de títulos de Wikipedia (uno por línea).
    2. Por cada título, consulta la API REST oficial para obtener el
       primer párrafo (extract) en el idioma del archivo.
    3. Genera 1-3 Q&A en formato canonical a partir del título + extract.
    4. Escribe cada record al JSONL de salida.

Requisitos:
    - Internet (usa `tools.search_web_real.wikipedia_search` o la API
      REST directa).
    - No requiere API key.
    - Respeta rate limit del search_web_real (10 req/min por defecto).
    - Puede correr en background mientras entrenas.

Nota:
    Wikipedia es considerablemente confiable pero no infalible. Los
    hechos extraídos deben ser revisados si se usan para decisiones
    críticas. Para un modelo de scale intermedio AION-C, alcanza con
    la calidad típica de un primer párrafo editorial.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from synth.canonical_format import build_record


# ════════════════════════════════════════════════════════════════════════════
# Wikipedia REST API client (no deps)
# ════════════════════════════════════════════════════════════════════════════

def _wiki_summary(title: str, lang: str = "en", timeout: float = 10.0) -> Optional[dict]:
    """Fetch the page summary from the Wikipedia REST API.

    Returns the parsed JSON dict, or None on any error.
    """
    import urllib.request
    import urllib.parse
    import urllib.error
    t_enc = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{t_enc}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AION-C/1.0 (research; github.com/aion-c)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError, TimeoutError):
        return None


# ════════════════════════════════════════════════════════════════════════════
# Seeds helpers
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_SEEDS_EN = [
    "Python (programming language)", "JavaScript", "Rust (programming language)",
    "Go (programming language)", "C (programming language)", "SQL",
    "HTTP", "HTTPS", "TLS", "DNS", "TCP/IP", "UDP", "IPv6",
    "Mathematics", "Algebra", "Calculus", "Probability theory",
    "Physics", "Chemistry", "Biology", "Evolution",
    "French Revolution", "World War II", "Cold War", "Renaissance",
    "Wikipedia", "Machine learning", "Neural network", "Large language model",
]

DEFAULT_SEEDS_ES = [
    "Python (lenguaje de programación)", "JavaScript", "Rust (lenguaje de programación)",
    "SQL", "HTTP", "DNS", "Matemáticas", "Cálculo", "Física",
    "Química", "Biología", "Evolución biológica",
    "Revolución francesa", "Segunda Guerra Mundial", "Guerra Fría",
    "Inteligencia artificial", "Red neuronal", "Aprendizaje automático",
]


def _load_seeds(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ════════════════════════════════════════════════════════════════════════════
# Record generation from a wiki summary
# ════════════════════════════════════════════════════════════════════════════

def _trim(text: str, max_chars: int = 380) -> str:
    if len(text) <= max_chars:
        return text
    # cortar en el punto más cercano
    cut = text.rfind(". ", 0, max_chars)
    if cut > 100:
        return text[: cut + 1]
    return text[:max_chars] + "..."


def _records_from_summary(title: str, extract: str, lang: str) -> List[dict]:
    """Produces 1-3 canonical records from a wiki page summary."""
    extract = _trim(extract, 380)
    out = []
    if lang == "en":
        qs = [
            f"What is {title}?",
            f"Can you tell me about {title}?",
            f"Give me a short introduction to {title}.",
        ]
    else:
        qs = [
            f"¿Qué es {title}?",
            f"¿Puedes contarme sobre {title}?",
            f"Dame una breve introducción a {title}.",
        ]
    for q in qs:
        rec = build_record(
            user=q,
            aion=extract,
            domain="general",
            language=lang,
            type="wiki_ingest",
            metadata={
                "topic": "wikipedia",
                "subtopic": title,
                "source": f"wikipedia_{lang}",
                "difficulty": "medium",
            },
        )
        out.append(rec.to_dict())
    return out


# ════════════════════════════════════════════════════════════════════════════
# Main runner
# ════════════════════════════════════════════════════════════════════════════

def run(
    seeds_en: List[str],
    seeds_es: List[str],
    out_path: Path,
    max_per_topic: int = 3,
    pause_sec: float = 0.5,
    verbose: bool = False,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    n_topics_ok = 0
    n_topics_err = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for lang, seeds in [("en", seeds_en), ("es", seeds_es)]:
            for title in seeds:
                data = _wiki_summary(title, lang=lang)
                if data is None:
                    n_topics_err += 1
                    if verbose:
                        print(f"  [miss] {lang}: {title}", file=sys.stderr)
                    time.sleep(pause_sec)
                    continue
                extract = data.get("extract", "").strip()
                if not extract:
                    n_topics_err += 1
                    time.sleep(pause_sec)
                    continue
                canonical_title = data.get("title", title)
                records = _records_from_summary(canonical_title, extract, lang)[:max_per_topic]
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_records += 1
                n_topics_ok += 1
                if verbose:
                    print(f"  [ok] {lang}: {canonical_title} ({len(records)} records)")
                time.sleep(pause_sec)

    return {
        "topics_ok": n_topics_ok,
        "topics_err": n_topics_err,
        "records_written": n_records,
        "out_path": str(out_path),
    }


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser()
    p.add_argument("--seeds-en", type=Path, default=None,
                   help="File with Wikipedia titles in English, one per line")
    p.add_argument("--seeds-es", type=Path, default=None,
                   help="File with Wikipedia titles in Spanish, one per line")
    p.add_argument("--out", type=Path, default=Path("datasets/wiki_ingested.jsonl"))
    p.add_argument("--max-per-topic", type=int, default=3)
    p.add_argument("--pause", type=float, default=0.5,
                   help="Pause between requests (seconds) to respect rate limits")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--use-defaults", action="store_true",
                   help="Use the DEFAULT_SEEDS built-in instead of files")
    args = p.parse_args()

    seeds_en = _load_seeds(args.seeds_en) if args.seeds_en else []
    seeds_es = _load_seeds(args.seeds_es) if args.seeds_es else []
    if args.use_defaults:
        if not seeds_en:
            seeds_en = DEFAULT_SEEDS_EN
        if not seeds_es:
            seeds_es = DEFAULT_SEEDS_ES

    if not seeds_en and not seeds_es:
        print("ERROR: no seeds provided. Use --seeds-en / --seeds-es or --use-defaults",
              file=sys.stderr)
        sys.exit(1)

    stats = run(
        seeds_en=seeds_en,
        seeds_es=seeds_es,
        out_path=args.out,
        max_per_topic=args.max_per_topic,
        pause_sec=args.pause,
        verbose=args.verbose,
    )
    print(f"Done: {stats['records_written']} records from "
          f"{stats['topics_ok']} ok topics, {stats['topics_err']} errors")
    print(f"  -> {stats['out_path']}")


if __name__ == "__main__":  # pragma: no cover
    main()
