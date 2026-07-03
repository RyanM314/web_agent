"""
app/services/web_research.py
─────────────────────────────
FR3 — Real-time web research across trusted academic sources.

Uses DuckDuckGo as the search provider and filters results to prioritise
academic / trusted domains. Returns structured sources with relevance scores.

Search Query Condenser
──────────────────────
Before any external HTTP request is made, the raw user prompt is passed
through a lightweight Gemini call that extracts 3-4 concise, search-optimised
keywords.  This prevents long paragraphs from being dumped directly into
search URLs.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.exceptions import WebResearchError

try:
    from duckduckgo_search import DDGS
except ImportError:  # pragma: no cover
    DDGS = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ResearchSource:
    """A single source found during web research."""

    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0


# ── Public function ────────────────────────────────────────────────────────────

def perform_research(
    query: str,
    max_sources: int = 5,
) -> list[ResearchSource]:
    """
    Perform real-time web research for the given query.

    Before making any external request, the query is **condensed** into 3-4
    search-optimised keywords via a lightweight Gemini call.  This prevents
    raw multi-sentence prompts from being sent to the search engine.

    Parameters
    ----------
    query      : The raw search query (typically the lecturer's update request).
    max_sources: Maximum number of sources to return.

    Returns
    -------
    list[ResearchSource] — the research results with relevance scores.

    Raises
    ------
    WebResearchError if the search fails entirely.
    """
    settings = get_settings()
    academic_domains = _parse_academic_domains(settings.academic_domains)

    # ── Condense the raw prompt into search-optimised keywords ────────────
    condensed = _condense_search_query(query)
    logger.info(
        "Search query condensed: '%.80s' → '%.80s'",
        query,
        condensed,
    )

    try:
        raw_results = _search_duckduckgo(condensed, max_results=max_sources * 2)
    except Exception as exc:
        raise WebResearchError(f"DuckDuckGo search failed: {exc}") from exc

    if not raw_results:
        logger.warning(
            "Web research returned no results for condensed query: %s",
            condensed[:80],
        )
        return []

    # Score and rank results
    scored: list[ResearchSource] = []
    for item in raw_results:
        url = item.get("link", item.get("href", ""))
        title = item.get("title", "Untitled")
        snippet = item.get("snippet", item.get("body", ""))

        score = _compute_relevance_score(condensed, title, snippet, url, academic_domains)
        scored.append(
            ResearchSource(
                title=title,
                url=url,
                snippet=snippet[:500],
                relevance_score=round(score, 3),
            )
        )

    # Sort by relevance, take top-k
    scored.sort(key=lambda s: s.relevance_score, reverse=True)
    top = scored[:max_sources]

    logger.info(
        "Web research: %d raw → %d after scoring (query: %s)",
        len(raw_results),
        len(top),
        condensed[:60],
    )
    return top


# ── Internal helpers ───────────────────────────────────────────────────────────

def _condense_search_query(raw_prompt: str) -> str:
    """
    Pass the raw user prompt through a lightweight Gemini call to extract
    3-4 concise, search-optimised keywords.

    If the Gemini call fails (e.g. no API key, transient error) the function
    falls back to a heuristic extraction.

    Examples
    --------
    Input  : "Update the section on civics education and explain the role of the constitution and human rights."
    Output : "Civics Education constitution human rights"
    """
    import textwrap

    settings = get_settings()

    # If the prompt is already short (< 80 chars) use it as-is
    if len(raw_prompt.strip()) < 80:
        return raw_prompt.strip()

    # Try the OpenRouter condenser with a bare httpx.post (no SDK)
    try:
        import httpx as _httpx
        condenser_payload = {
            "model": settings.openrouter_generation_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Extract 3-4 concise, search-optimised keywords from the following text. "
                        f"Respond with ONLY the keywords, separated by spaces. Do NOT include "
                        f"any explanation, punctuation, or formatting.\n\n"
                        f"Text: {raw_prompt.strip()[:1000]}\n\nKeywords:"
                    ),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 128,
        }
        condenser_headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        condenser_resp = _httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=condenser_headers,
            json=condenser_payload,
            timeout=30.0,
        )
        if condenser_resp.is_success:
            condenser_data = condenser_resp.json()
            choices = condenser_data.get("choices", [])
            if choices:
                keywords = (choices[0].get("message", {}).get("content", "") or "").strip()
                if keywords and len(keywords) > 5:
                    logger.debug("Condenser produced: '%s'", keywords[:120])
                    return keywords
    except Exception as exc:
        logger.debug("Search query condenser failed, falling back: %s", exc)

    # ── Heuristic fallback ────────────────────────────────────────────────
    # Remove common filler words, take the first 8 significant terms
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all",
        "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "because", "but", "and", "or", "if",
        "while", "about", "up", "out", "also", "please", "update",
    }

    # Extract meaningful capitalised terms first (proper nouns, concepts)
    terms = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", raw_prompt)

    if terms:
        return " ".join(terms[:6])

    # Fallback: take significant lowercase words
    words = re.findall(r"\b[a-zA-Z]{4,}\b", raw_prompt.lower())
    filtered = [w for w in words if w not in stop_words]
    if filtered:
        return " ".join(filtered[:8])

    # Last resort: return first 100 chars trimmed
    return raw_prompt.strip()[:100].strip()


def _parse_academic_domains(domains_str: str) -> list[str]:
    """Parse the comma-separated academic domains config into a list."""
    return [d.strip().lower() for d in domains_str.split(",") if d.strip()]


def _search_duckduckgo(query: str, max_results: int = 10) -> list[dict]:
    """
    Perform a DuckDuckGo text search.

    Returns a list of dicts with keys: title, link, snippet.
    """
    if DDGS is None:
        raise WebResearchError(
            "duckduckgo-search is not installed. Run: pip install duckduckgo-search"
        )

    with DDGS() as ddgs:
        results = list(
            ddgs.text(
                keywords=query,
                max_results=max_results,
            )
        )
    return results


def _compute_relevance_score(
    query: str,
    title: str,
    snippet: str,
    url: str,
    academic_domains: list[str],
) -> float:
    """
    Compute a relevance score (0–1) for a search result.

    Boosts:
      - Domain matches an academic/trusted domain (+0.3)
      - Title or snippet contains key query terms (+0.1 per match)
      - URL is .edu (+0.2)
    """
    score = 0.0
    url_lower = url.lower()

    # Academic domain bonus
    for domain in academic_domains:
        if domain in url_lower:
            score += 0.3
            break

    # .edu TLD bonus
    if re.search(r"\.edu\b", url_lower):
        score += 0.2

    # Term overlap bonus
    query_terms = set(re.findall(r"\w+", query.lower()))
    title_terms = set(re.findall(r"\w+", title.lower()))
    snippet_terms = set(re.findall(r"\w+", snippet.lower()))

    title_overlap = len(query_terms & title_terms)
    snippet_overlap = len(query_terms & snippet_terms)

    score += min(title_overlap * 0.05, 0.3)
    score += min(snippet_overlap * 0.02, 0.2)

    # Base score for having results
    score = max(score, 0.1)

    return min(score, 1.0)


def _fetch_source_content(url: str, max_chars: int = 2000) -> str:
    """
    Fetch and extract plain text from a URL.

    Used for deeper verification of a source when needed.
    """
    try:
        import httpx
        from html.parser import HTMLParser
    except ImportError:
        logger.warning("httpx not available for content fetching.")
        return ""

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._text_parts: list[str] = []
            self._skip = False

        def handle_data(self, data: str) -> None:
            stripped = data.strip()
            if stripped and not self._skip:
                self._text_parts.append(stripped)

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag in ("script", "style", "noscript"):
                self._skip = True

        def handle_endtag(self, tag: str) -> None:
            if tag in ("script", "style", "noscript"):
                self._skip = False

        def get_text(self) -> str:
            return " ".join(self._text_parts)

    try:
        resp = httpx.get(url, timeout=10.0, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type:
            return ""
        extractor = TextExtractor()
        extractor.feed(resp.text)
        text = extractor.get_text()
        return text[:max_chars]
    except Exception as exc:
        logger.debug("Failed to fetch source content from %s: %s", url, exc)
        return ""