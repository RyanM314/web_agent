"""
app/core/openrouter_client.py
──────────────────────────────
Raw HTTP client for OpenRouter API (OpenAI-compatible endpoint).

Sends a bare JSON POST directly to:
    https://openrouter.ai/api/v1/chat/completions

No Google SDK wrappers — only ``httpx`` for the HTTP call.

Authentication
--------------
    Authorization: Bearer sk-or-v1-...

Request body (OpenAI / OpenRouter layout):
    {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "..."}]
    }

Token tracking
--------------
Every response logs ``prompt_tokens`` and ``completion_tokens`` at INFO level
for cost monitoring.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.core.exceptions import AgenticRagError

logger = logging.getLogger(__name__)

# ── Endpoint ────────────────────────────────────────────────────────────────────

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ── Public function ─────────────────────────────────────────────────────────────

def call_openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 4096,
    service_name: str = "openrouter_generate",
) -> str:
    """
    Send a chat-completion request to OpenRouter and return the generated text.

    Parameters
    ----------
    api_key      : OpenRouter API key (``sk-or-v1-...``).
    model        : OpenRouter model slug (e.g. ``openai/gpt-4o-mini``).
    messages     : List of ``{"role": …, "content": …}`` dicts.
    temperature  : Sampling temperature (0.0–1.0).
    max_tokens   : Maximum tokens in the response.
    service_name : Label for logging.

    Returns
    -------
    str — the generated text content.

    Raises
    ------
    AgenticRagError on network / API / auth / balance failures.
    """
    # ── Headers (exact format specified by OpenRouter docs) ─────────────────
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # ── Request body (standard OpenAI / OpenRouter layout) ──────────────────
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    key_prefix = api_key[:4] + "..." if len(api_key) > 4 else "(empty)"
    logger.debug(
        "OpenRouter POST [%s]: url=%s model=%s key_prefix=%s",
        service_name,
        _OPENROUTER_URL,
        model,
        key_prefix,
    )

    # ── Bare HTTP POST — no SDK wrappers ───────────────────────────────────
    try:
        resp = httpx.post(
            _OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=120.0,
        )
    except httpx.TimeoutException:
        raise AgenticRagError(
            f"OpenRouter request timed out for '{service_name}'."
        )
    except httpx.RequestError as exc:
        raise AgenticRagError(
            f"OpenRouter network error for '{service_name}': {exc}"
        )

    # ── Error handling ─────────────────────────────────────────────────────
    if resp.status_code == 401:
        raise AgenticRagError(
            "OpenRouter authentication failed. Check your OPENROUTER_API_KEY."
        )
    if resp.status_code == 402:
        raise AgenticRagError(
            "OpenRouter account balance is insufficient. "
            "Please top up at https://openrouter.ai/settings/credits."
        )
    if resp.status_code == 429:
        raise AgenticRagError(
            "OpenRouter rate limit hit. Please wait a moment and try again."
        )
    if resp.status_code == 404:
        detail = _try_extract_error(resp.text)
        raise AgenticRagError(
            f"OpenRouter returned 404 for model '{model}'. "
            f"Detail: {detail}"
        )
    if not resp.is_success:
        detail = _try_extract_error(resp.text)
        raise AgenticRagError(
            f"OpenRouter API error (HTTP {resp.status_code}): {detail}"
        )

    # ── Parse response ─────────────────────────────────────────────────────
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise AgenticRagError(
            f"OpenRouter returned non-JSON response: {resp.text[:500]}"
        ) from exc

    # ── Extract generated text ─────────────────────────────────────────────
    try:
        choices = data.get("choices", [])
        if not choices:
            raise AgenticRagError("OpenRouter returned no choices.")
        text = choices[0].get("message", {}).get("content", "")
        if not text:
            raise AgenticRagError("OpenRouter returned an empty response.")
    except (KeyError, IndexError, TypeError) as exc:
        raise AgenticRagError(
            f"Unexpected OpenRouter response structure: {exc}"
        ) from exc

    # ── Log token usage for cost tracking ──────────────────────────────────
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    logger.info(
        "OpenRouter [%s] — generated %d chars | "
        "prompt_tokens=%d  completion_tokens=%d  total_tokens=%d",
        service_name,
        len(text),
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
    )

    return text


# ── Internal helpers ───────────────────────────────────────────────────────────

def _try_extract_error(body_text: str) -> str:
    """Try to parse an error detail from an API error response."""
    try:
        body = json.loads(body_text)
        if "error" in body:
            err = body["error"]
            if isinstance(err, dict):
                return err.get("message", str(err))
            return str(err)
        return body_text[:500]
    except (json.JSONDecodeError, TypeError):
        return body_text[:500]