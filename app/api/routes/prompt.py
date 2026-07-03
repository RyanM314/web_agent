"""
app/api/routes/prompt.py
────────────────────────
FR2 — Prompt / Request Input

Endpoints
─────────
POST /api/v1/prompt   Submit a lecturer's update request → RAG → generated response
GET  /api/v1/prompt/health  Check the prompt pipeline health
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import EmbeddingError, PromptProcessingError, VectorStoreError
from app.models.schemas import HealthResponse, PromptRequest, PromptResponse
from app.services.prompt_service import process_prompt
from app.services.vector_store import health_check

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/prompt", tags=["FR2 — Prompt / Request Input"])


# ── POST /api/v1/prompt ────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=PromptResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit an update request for lecturer notes",
    description=(
        "Accepts a natural-language prompt describing what to update in the "
        "lecturer's notes. The system embeds the prompt, retrieves the most "
        "relevant chunks from Pinecone, builds a RAG prompt, and calls Gemini "
        "to generate the updated content. Optionally scope retrieval to a "
        "specific document via ``document_id``."
    ),
)
async def submit_prompt(
    request: PromptRequest,
) -> PromptResponse:
    """FR2: Submit prompt → retrieve context → generate updated content."""
    logger.info(
        "Prompt request received (len=%d, doc_id=%s, top_k=%d).",
        len(request.prompt),
        request.document_id or "all",
        request.top_k,
    )

    try:
        result = process_prompt(request)
    except PromptProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service error: {exc}",
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector database error: {exc}",
        ) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during prompt processing.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        ) from exc

    return result


# ── GET /api/v1/prompt/health ──────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check prompt pipeline health",
)
async def prompt_health() -> HealthResponse:
    """Verify Pinecone connectivity and embedding service availability."""
    pinecone_ok = health_check()

    # Light-weight embedding check
    embedding_status = "ok"
    try:
        from app.services.embedding_service import embed_texts
        embed_texts(["health check"])
    except Exception as exc:  # noqa: BLE001
        embedding_status = f"error: {exc}"

    # Light-weight generation check
    generation_status = "ok"
    try:
        from app.core.config import get_settings
        from app.core.gemini_client import create_gemini_client
        settings = get_settings()
        client = create_gemini_client(settings.google_api_key)
        # A minimal generation to verify the model responds
        response = client.models.generate_content(
            model=settings.embedding_generation_model,
            contents="Respond with exactly one word: ok",
        )
        if not response.text or response.text.strip().lower() != "ok":
            generation_status = f"unexpected response: {response.text}"
    except Exception as exc:  # noqa: BLE001
        generation_status = f"error: {exc}"

    overall_status = "healthy"
    if not pinecone_ok:
        overall_status = "degraded"
    if embedding_status != "ok" or generation_status != "ok":
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        app_name="",
        version="",
        pinecone_connected=pinecone_ok,
        embedding_service=f"embedding={embedding_status}, generation={generation_status}",
    )