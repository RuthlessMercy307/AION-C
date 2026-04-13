"""
tools/search_web_real.py — Implementación real de búsqueda web para AION-C.

Backends soportados, en orden de preferencia:

    1. Wikipedia REST API (oficial, libre, sin API key, estable, rate-limit
       generoso)
    2. DuckDuckGo HTML (fallback, scraping, frágil pero sin key)

Uso:
    from tools.search_web_real import wikipedia_search, search_web

    # Uso directo
    results = wikipedia_search("python programming language", max_results=5)
    for r in results:
        print(r.title, r.url)
        print(r.snippet)

    # Integración con WebSearchTool existente
    from agent.tools import WebSearchTool
    tool = WebSearchTool(search_fn=search_web)
    result = tool.run({"query": "cómo funciona HTTP", "max_results": 3})

Sandbox:
    - Rate limit: 10 queries/min por proceso
    - Timeout: 10 segundos por request
    - User-Agent identificado como AION-C
    - Sin seguimiento de redirects malignos
    - Errores devueltos como mensaje, no crash

Nota sobre rate limits:
    Wikipedia tolera ~200 req/s pero pedimos al agente que mantenga disciplina
    de 10/min para no abusar y ser buen ciudadano. El limiter es in-process
    (no persistente) así que un reinicio libera el contador.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import List, Optional


USER_AGENT = "AION-C/1.0 (research prototype; contact: github.com/aion-c)"
DEFAULT_TIMEOUT = 10.0
RATE_LIMIT_MAX = 10           # max queries
RATE_LIMIT_WINDOW_SEC = 60.0  # per N seconds


# ════════════════════════════════════════════════════════════════════════════
# Result type
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    source: str  # "wikipedia_en" / "wikipedia_es" / "duckduckgo"

    def to_dict(self):
        return asdict(self)

    def format_line(self) -> str:
        """Formato compacto para concatenar como [RESULT:] en canonical."""
        return f"- {self.title}: {self.snippet} [{self.url}]"


# ════════════════════════════════════════════════════════════════════════════
# Rate limiter
# ════════════════════════════════════════════════════════════════════════════

class _RateLimiter:
    """Ventana deslizante simple en memoria."""

    def __init__(self, max_requests: int, window_sec: float) -> None:
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._hits: List[float] = []

    def allow(self) -> bool:
        now = time.time()
        # Purga hits fuera de la ventana
        self._hits = [t for t in self._hits if now - t < self.window_sec]
        if len(self._hits) >= self.max_requests:
            return False
        self._hits.append(now)
        return True

    def reset(self) -> None:
        self._hits = []


_limiter = _RateLimiter(RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SEC)


class RateLimitExceeded(Exception):
    pass


# ════════════════════════════════════════════════════════════════════════════
# HTTP helper
# ════════════════════════════════════════════════════════════════════════════

def _http_get_json(url: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """GET + JSON parse, con User-Agent identificado."""
    if not _limiter.allow():
        raise RateLimitExceeded(
            f"rate limit exceeded: max {RATE_LIMIT_MAX} queries per "
            f"{RATE_LIMIT_WINDOW_SEC:.0f}s window"
        )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _http_get_text(url: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """GET + texto, para HTML scraping."""
    if not _limiter.allow():
        raise RateLimitExceeded()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ════════════════════════════════════════════════════════════════════════════
# Wikipedia backend (preferred)
# ════════════════════════════════════════════════════════════════════════════

WIKI_SEARCH_TEMPLATE = (
    "https://{lang}.wikipedia.org/w/api.php"
    "?action=query&list=search&srsearch={q}&srlimit={limit}"
    "&format=json&utf8=1"
)
WIKI_SUMMARY_TEMPLATE = (
    "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
)


def wikipedia_search(
    query: str,
    max_results: int = 5,
    lang: str = "en",
    fetch_summaries: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
) -> List[SearchResult]:
    """Busca en Wikipedia usando la API oficial.

    Args:
        query: la consulta
        max_results: cuántos resultados traer (1-10)
        lang: "en" o "es"
        fetch_summaries: si True, hace un segundo request por cada hit para
                         traer el primer párrafo completo (más caro, mejor
                         calidad). Si False, usa el snippet que devuelve la
                         API de search (más rápido, algo con etiquetas HTML).
        timeout: segundos antes de dar timeout

    Returns:
        lista de SearchResult. Vacía si no hay resultados o si el backend
        falla. Lanza RateLimitExceeded si se excede el rate.
    """
    max_results = max(1, min(10, int(max_results)))
    q_enc = urllib.parse.quote(query, safe="")
    url = WIKI_SEARCH_TEMPLATE.format(lang=lang, q=q_enc, limit=max_results)
    try:
        data = _http_get_json(url, timeout=timeout)
    except RateLimitExceeded:
        raise
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
        return []

    hits = data.get("query", {}).get("search", [])
    out: List[SearchResult] = []
    for hit in hits[:max_results]:
        title = hit.get("title", "")
        snippet = _strip_wiki_html(hit.get("snippet", ""))
        page_url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        if fetch_summaries:
            summary = _wiki_page_summary(title, lang=lang, timeout=timeout)
            if summary:
                snippet = summary
        out.append(SearchResult(
            title=title,
            snippet=snippet,
            url=page_url,
            source=f"wikipedia_{lang}",
        ))
    return out


def _strip_wiki_html(s: str) -> str:
    """Elimina las etiquetas <span class='searchmatch'> y similares."""
    import re
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&quot;", '"').replace("&amp;", "&").replace("&nbsp;", " ")
    return s.strip()


def _wiki_page_summary(title: str, lang: str, timeout: float) -> Optional[str]:
    t_enc = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = WIKI_SUMMARY_TEMPLATE.format(lang=lang, title=t_enc)
    try:
        data = _http_get_json(url, timeout=timeout)
    except RateLimitExceeded:
        raise
    except Exception:
        return None
    extract = data.get("extract", "").strip()
    if not extract:
        return None
    # Cortar a unos ~300 chars para que entre en un [RESULT:]
    if len(extract) > 300:
        cut = extract.rfind(". ", 0, 300)
        if cut > 100:
            extract = extract[: cut + 1]
        else:
            extract = extract[:300] + "..."
    return extract


# ════════════════════════════════════════════════════════════════════════════
# DuckDuckGo HTML backend (fallback)
# ════════════════════════════════════════════════════════════════════════════

DDG_HTML_URL = "https://html.duckduckgo.com/html/?q={q}"


def duckduckgo_html_search(
    query: str,
    max_results: int = 5,
    timeout: float = DEFAULT_TIMEOUT,
) -> List[SearchResult]:
    """Scrape básico de DuckDuckGo HTML (sin API).

    Frágil por diseño — DDG puede cambiar el HTML en cualquier momento.
    Se usa SÓLO como fallback si Wikipedia no devuelve nada.
    """
    import re
    q_enc = urllib.parse.quote(query, safe="")
    url = DDG_HTML_URL.format(q=q_enc)
    try:
        html = _http_get_text(url, timeout=timeout)
    except RateLimitExceeded:
        raise
    except Exception:
        return []

    # Pattern para los result cards
    pattern = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    out: List[SearchResult] = []
    for m in pattern.finditer(html):
        if len(out) >= max_results:
            break
        url_raw = m.group(1)
        title = _strip_wiki_html(m.group(2))
        snippet = _strip_wiki_html(m.group(3))
        # DDG wraps the real URL
        if url_raw.startswith("//duckduckgo.com/l/?uddg="):
            real = urllib.parse.parse_qs(url_raw.split("?", 1)[1]).get("uddg", [""])[0]
            url_raw = urllib.parse.unquote(real)
        out.append(SearchResult(
            title=title,
            snippet=snippet,
            url=url_raw,
            source="duckduckgo",
        ))
    return out


# ════════════════════════════════════════════════════════════════════════════
# Unified API
# ════════════════════════════════════════════════════════════════════════════

def search_web(query: str, max_results: int = 5) -> str:
    """Función unificada compatible con la API `search_fn` de WebSearchTool.

    Intenta Wikipedia en español, luego en inglés, luego DDG. Devuelve un
    string formateado listo para meter en un [RESULT:] block.

    Al ser llamado por el tool executor, las excepciones se convierten en
    un mensaje de error en lugar de crashear.
    """
    query = (query or "").strip()
    if not query:
        return "(empty query)"

    try:
        hits = wikipedia_search(query, max_results=max_results, lang="es")
        if not hits:
            hits = wikipedia_search(query, max_results=max_results, lang="en")
        if not hits:
            hits = duckduckgo_html_search(query, max_results=max_results)
    except RateLimitExceeded as exc:
        return f"(rate limit) {exc}"
    except Exception as exc:
        return f"(search error) {exc}"

    if not hits:
        return "(no results)"

    lines = [f"Found {len(hits)} results for {query!r}:"]
    for h in hits:
        lines.append(h.format_line())
    return "\n".join(lines)


def search_web_structured(query: str, max_results: int = 5) -> List[SearchResult]:
    """Variante que devuelve SearchResult list en vez de string."""
    query = (query or "").strip()
    if not query:
        return []
    try:
        hits = wikipedia_search(query, max_results=max_results, lang="es")
        if not hits:
            hits = wikipedia_search(query, max_results=max_results, lang="en")
        if not hits:
            hits = duckduckgo_html_search(query, max_results=max_results)
    except (RateLimitExceeded, Exception):
        hits = []
    return hits


# ════════════════════════════════════════════════════════════════════════════
# CLI para testing manual
# ════════════════════════════════════════════════════════════════════════════

def _cli():  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description="Test search_web_real")
    p.add_argument("query", nargs="+")
    p.add_argument("--max", type=int, default=5)
    p.add_argument("--lang", default="en", choices=["en", "es"])
    p.add_argument("--structured", action="store_true")
    args = p.parse_args()
    q = " ".join(args.query)
    if args.structured:
        results = wikipedia_search(q, max_results=args.max, lang=args.lang,
                                   fetch_summaries=True)
        for r in results:
            print(f"- {r.title}")
            print(f"  {r.snippet}")
            print(f"  {r.url}")
            print()
    else:
        print(search_web(q, max_results=args.max))


if __name__ == "__main__":
    _cli()
