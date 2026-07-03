"""
app/core/exceptions.py
──────────────────────
Domain-specific exception hierarchy.
FastAPI exception handlers in main.py convert these to proper HTTP responses.
"""

from __future__ import annotations


class RagAgentError(Exception):
    """Base error for the entire application."""


class UnsupportedFileTypeError(RagAgentError):
    """Raised when an uploaded file has a disallowed extension."""

    def __init__(self, extension: str, allowed: set[str]) -> None:
        self.extension = extension
        self.allowed = allowed
        super().__init__(
            f"File type '.{extension}' is not supported. "
            f"Allowed types: {', '.join(sorted(allowed))}."
        )


class FileTooLargeError(RagAgentError):
    """Raised when an uploaded file exceeds the configured size limit."""

    def __init__(self, size_bytes: int, limit_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"File size {size_bytes / 1024 / 1024:.1f} MB exceeds "
            f"limit of {limit_bytes / 1024 / 1024:.0f} MB."
        )


class DocumentParseError(RagAgentError):
    """Raised when text extraction from a document fails."""


class EmbeddingError(RagAgentError):
    """Raised when the embedding service returns an error."""


class VectorStoreError(RagAgentError):
    """Raised when Pinecone operations fail."""


class PromptProcessingError(RagAgentError):
    """Raised when the RAG prompt-generation pipeline fails."""


class WebResearchError(RagAgentError):
    """Raised when web research fails."""


class SelfCorrectionError(RagAgentError):
    """Raised when automatic self-correction fails."""


class SessionError(RagAgentError):
    """Raised when session operations fail."""


class AgenticRagError(RagAgentError):
    """Raised when the agentic RAG pipeline fails."""


class ExportError(RagAgentError):
    """Raised when document export (PDF/DOCX) fails."""


class RateLimitExceededError(RagAgentError):
    """
    Raised when a transient Gemini API rate-limit (HTTP 429) is hit.

    Unlike ``QuotaExhaustedError``, this error is **retryable** — the caller
    should pause and retry up to a few times.  It does **not** set the global
    quota-exhausted flag, so other concurrent requests are not blocked.

    The exception carries a ``retry_after_seconds`` field that the retry logic
    uses to determine the back-off delay.
    """

    def __init__(
        self,
        retry_after_seconds: float = 6.0,
        detail: str = "Gemini API rate limit hit. Pausing and retrying…",
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        self.detail = detail
        super().__init__(detail)


class QuotaExhaustedError(RagAgentError):
    """
    Raised when the Gemini API daily quota has been exhausted.

    The exception carries a ``retry_after_seconds`` field and a human-readable
    ``detail`` message. FastAPI exception handlers convert this to a 429
    HTTP response with ``quota_exhausted=True`` so the frontend can show the
    persistent countdown banner.
    """

    def __init__(
        self,
        retry_after_seconds: float = 0,
        detail: str = "API quota exhausted. Please try again later.",
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        self.detail = detail
        super().__init__(detail)