"""
app/core/gemini_client.py
─────────────────────────
Factory that creates a ``genai.Client`` using the configured credential.

The modern ``google-genai`` SDK (≥2.0) accepts both Gemini API keys
(``AIza...``) and OAuth 2.0 access tokens (``AQ...``) via the ``api_key=``
parameter — no special handling is needed.
"""

from __future__ import annotations

import logging

from google import genai

logger = logging.getLogger(__name__)


def create_gemini_client(api_key: str) -> genai.Client:
    """
    Build a ``genai.Client`` that authenticates with the given credential.

    Parameters
    ----------
    api_key : str
        A Gemini API key (``AIza...``) **or** a Google OAuth 2.0 access
        token (``AQ...``).  The SDK natively supports both.

    Returns
    -------
    genai.Client
        Fully configured client ready for use.
    """
    key_prefix = api_key[:4] + "..." if len(api_key) > 4 else "(empty)"
    logger.info("Creating genai.Client with key prefix %s", key_prefix)
    return genai.Client(api_key=api_key)