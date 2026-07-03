"""
app/services/embedding_service.py
──────────────────────────────────
Wraps Google Gemini's text-embedding endpoint.

• Batches chunks to stay within API rate limits.
• Retries transient errors with exponential back-off (tenacity).
• Returns a list of float vectors in the same order as the input texts.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from google import genai
from google.genai import types as genai_types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.gemini_client import create_gemini_client
from app.core.gemini_middleware import call_gemini

logger = logging.getLogger(__name__)

# Gemini embedding-001 accepts up to 100 texts per batch call.
_BATCH_SIZE = 50


_genai_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return a cached Gemini client configured with the API key or OAuth token."""
    global _genai_client
    if _genai_client is None:
        settings = get_settings()
        if not settings.google_api_key:
            raise EmbeddingError(
                "GOOGLE_API_KEY is not set. Add it to your .env file."
            )
        _genai_client = create_gemini_client(settings.google_api_key)
    return _genai_client


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    """Embed a single batch of texts; retried on transient errors."""
    client = _get_client()
    result = call_gemini(
        lambda: client.models.embed_content(
            model=model,
            contents=texts,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        ),
        service_name="embed_batch",
    )
    return [e.values for e in result.embeddings]


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """
    Embed an arbitrary number of texts using Gemini.

    Parameters
    ----------
    texts : sequence of strings (chunk texts)

    Returns
    -------
    list[list[float]] — one vector per input text, same order.

    Raises
    ------
    EmbeddingError on unrecoverable failure.
    """
    _get_client()   # validates API key early
    settings = get_settings()
    model = settings.embedding_model
    text_list = list(texts)

    if not text_list:
        return []

    all_vectors: list[list[float]] = []

    for batch_start in range(0, len(text_list), _BATCH_SIZE):
        batch = text_list[batch_start : batch_start + _BATCH_SIZE]
        logger.debug(
            "Embedding batch %d–%d of %d texts.",
            batch_start,
            batch_start + len(batch) - 1,
            len(text_list),
        )
        try:
            vectors = _embed_batch(batch, model)
            all_vectors.extend(vectors)
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding failed for batch starting at index {batch_start}: {exc}"
            ) from exc

        # Polite pause between batches to stay under rate limits
        if batch_start + _BATCH_SIZE < len(text_list):
            time.sleep(0.3)

    logger.info("Embedded %d texts → %d vectors.", len(text_list), len(all_vectors))
    return all_vectors
