"""
tests/test_search_web_real.py — tests del search_web real.

Los tests NO hacen hits a la red. Usan monkeypatching sobre el HTTP
helper para simular respuestas de Wikipedia y DuckDuckGo.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tools import search_web_real as swr
from tools.search_web_real import (
    SearchResult,
    _RateLimiter,
    _strip_wiki_html,
    duckduckgo_html_search,
    search_web,
    search_web_structured,
    wikipedia_search,
    RateLimitExceeded,
)


# ════════════════════════════════════════════════════════════════════════════
# Unit: SearchResult
# ════════════════════════════════════════════════════════════════════════════

class TestSearchResult:
    def test_format_line(self):
        r = SearchResult("Python", "a language", "https://x", "wikipedia_en")
        line = r.format_line()
        assert "Python" in line
        assert "a language" in line
        assert "https://x" in line

    def test_to_dict(self):
        r = SearchResult("T", "S", "U", "wikipedia_en")
        d = r.to_dict()
        assert d["title"] == "T"
        assert d["source"] == "wikipedia_en"


# ════════════════════════════════════════════════════════════════════════════
# Unit: _strip_wiki_html
# ════════════════════════════════════════════════════════════════════════════

class TestStripHtml:
    def test_removes_tags(self):
        s = '<span class="searchmatch">Python</span> is a language'
        assert _strip_wiki_html(s) == "Python is a language"

    def test_decodes_entities(self):
        s = "Say &quot;hi&quot; &amp; bye"
        assert _strip_wiki_html(s) == 'Say "hi" & bye'

    def test_trims_whitespace(self):
        assert _strip_wiki_html("  hello  ") == "hello"


# ════════════════════════════════════════════════════════════════════════════
# Unit: rate limiter
# ════════════════════════════════════════════════════════════════════════════

class TestRateLimiter:
    def test_allows_under_limit(self):
        lim = _RateLimiter(max_requests=3, window_sec=60)
        assert lim.allow()
        assert lim.allow()
        assert lim.allow()

    def test_blocks_over_limit(self):
        lim = _RateLimiter(max_requests=2, window_sec=60)
        lim.allow()
        lim.allow()
        assert not lim.allow()

    def test_window_slides(self):
        import time as _t
        lim = _RateLimiter(max_requests=2, window_sec=0.02)
        lim.allow()
        lim.allow()
        assert not lim.allow()
        _t.sleep(0.05)
        assert lim.allow()

    def test_reset(self):
        lim = _RateLimiter(2, 60)
        lim.allow()
        lim.allow()
        lim.reset()
        assert lim.allow()


# ════════════════════════════════════════════════════════════════════════════
# wikipedia_search with mocked HTTP
# ════════════════════════════════════════════════════════════════════════════

_FAKE_WIKI_RESPONSE = {
    "query": {
        "search": [
            {
                "title": "Python (programming language)",
                "snippet": 'Python is a high-<span class="searchmatch">level</span> language',
            },
            {
                "title": "Python Software Foundation",
                "snippet": "The PSF is a non-profit organization",
            },
        ],
    }
}


class TestWikipediaSearch:
    def setup_method(self):
        # Reset rate limiter before each test
        swr._limiter.reset()

    def test_returns_parsed_results(self):
        with patch.object(swr, "_http_get_json", return_value=_FAKE_WIKI_RESPONSE):
            results = wikipedia_search("python", max_results=5, lang="en")
        assert len(results) == 2
        assert results[0].title == "Python (programming language)"
        assert "<span" not in results[0].snippet  # html stripped
        assert "Python_%28programming_language%29" in results[0].url
        assert results[0].source == "wikipedia_en"

    def test_respects_max_results(self):
        with patch.object(swr, "_http_get_json", return_value=_FAKE_WIKI_RESPONSE):
            results = wikipedia_search("python", max_results=1, lang="en")
        assert len(results) == 1

    def test_max_results_clamped(self):
        with patch.object(swr, "_http_get_json", return_value=_FAKE_WIKI_RESPONSE):
            results = wikipedia_search("python", max_results=50, lang="en")
        assert len(results) <= 10

    def test_http_error_returns_empty(self):
        def boom(*args, **kwargs):
            import urllib.error
            raise urllib.error.URLError("network down")
        with patch.object(swr, "_http_get_json", side_effect=boom):
            results = wikipedia_search("python", lang="en")
        assert results == []

    def test_url_encodes_query(self):
        captured = {}
        def fake(url, timeout=None):
            captured["url"] = url
            return _FAKE_WIKI_RESPONSE
        with patch.object(swr, "_http_get_json", side_effect=fake):
            wikipedia_search("hello world", lang="en")
        assert "hello%20world" in captured["url"]

    def test_language_switch(self):
        captured = {}
        def fake(url, timeout=None):
            captured["url"] = url
            return {"query": {"search": []}}
        with patch.object(swr, "_http_get_json", side_effect=fake):
            wikipedia_search("x", lang="es")
        assert "es.wikipedia.org" in captured["url"]


# ════════════════════════════════════════════════════════════════════════════
# search_web (unified)
# ════════════════════════════════════════════════════════════════════════════

class TestSearchWebUnified:
    def setup_method(self):
        swr._limiter.reset()

    def test_happy_path_returns_formatted_string(self):
        with patch.object(swr, "_http_get_json", return_value=_FAKE_WIKI_RESPONSE):
            text = search_web("python", max_results=2)
        assert "Found 2 results" in text
        assert "Python" in text
        assert "https://" in text

    def test_empty_query(self):
        assert "empty" in search_web("").lower()

    def test_no_results_returns_message(self):
        with patch.object(swr, "_http_get_json", return_value={"query": {"search": []}}):
            with patch.object(swr, "_http_get_text", return_value=""):
                text = search_web("zzzzz")
        assert "no results" in text.lower()

    def test_rate_limit_returns_error_message(self):
        def rate_limited(*a, **k):
            raise RateLimitExceeded("test limit")
        with patch.object(swr, "_http_get_json", side_effect=rate_limited):
            text = search_web("x")
        assert "rate limit" in text.lower()

    def test_structured_variant(self):
        with patch.object(swr, "_http_get_json", return_value=_FAKE_WIKI_RESPONSE):
            results = search_web_structured("python", max_results=2)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)
        assert len(results) == 2

    def test_fallback_to_english_if_spanish_empty(self):
        calls = []
        def fake(url, timeout=None):
            calls.append(url)
            if "es.wikipedia" in url:
                return {"query": {"search": []}}
            return _FAKE_WIKI_RESPONSE
        with patch.object(swr, "_http_get_json", side_effect=fake):
            text = search_web("python")
        assert "Found" in text
        # At least one es and one en call
        assert any("es.wikipedia" in c for c in calls)
        assert any("en.wikipedia" in c for c in calls)


# ════════════════════════════════════════════════════════════════════════════
# DuckDuckGo HTML fallback
# ════════════════════════════════════════════════════════════════════════════

_FAKE_DDG_HTML = '''
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpy">Python Info</a>
  <a class="result__snippet" href="x">A language for everything</a>
</div>
<div class="result">
  <a class="result__a" href="https://direct.example.com/">Direct URL</a>
  <a class="result__snippet" href="y">Second result</a>
</div>
'''


class TestDuckDuckGoHtml:
    def setup_method(self):
        swr._limiter.reset()

    def test_parses_results(self):
        with patch.object(swr, "_http_get_text", return_value=_FAKE_DDG_HTML):
            results = duckduckgo_html_search("python", max_results=5)
        assert len(results) == 2
        assert results[0].title == "Python Info"
        # URL unwrapped from the ddg redirect
        assert results[0].url.startswith("https://example.com/")
        assert results[1].url == "https://direct.example.com/"
