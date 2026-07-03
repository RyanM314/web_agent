"""
app/api/routes/agentic_rag.py
──────────────────────────────
FR3 — Agentic RAG Processing

Endpoints
─────────
POST /api/v1/agentic-rag   Start a new agentic RAG session or submit a
                           follow-up prompt for iterative refinement
GET  /api/v1/agentic-rag/health  Check pipeline health
GET  /api/v1/agentic-rag/sessions  List active sessions
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import (
    AgenticRagError,
    EmbeddingError,
    QuotaExhaustedError,
    SessionError,
    VectorStoreError,
    WebResearchError,
)
from app.models.schemas import (
    AgenticRagRequest,
    AgenticRagResponse,
    HealthResponse,
)
from app.services.agentic_rag_service import process_agentic_rag
from app.services.session_manager import get_session_count
from app.services.vector_store import health_check

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/v1/agentic-rag",
    tags=["FR3 — Agentic RAG Processing"],
)


# ── POST /api/v1/agentic-rag ───────────────────────────────────────────────────

@router.post(
    "",
    response_model=AgenticRagResponse,
    status_code=status.HTTP_200_OK,
    summary="Process an agentic RAG update request",
    description=(
        "Starts a new agentic RAG session or continues an existing one for "
        "iterative refinement. The system autonomously:\n"
        "1. Retrieves relevant chunks from the vector store\n"
        "2. Performs real-time web research across trusted academic sources\n"
        "3. Cross-checks and verifies information\n"
        "4. Synthesises verified content via Gemini\n"
        "5. Delivers a formatted comparison of old vs new notes\n"
        "6. Applies automatic self-correction for length/style/research quality\n\n"
        "To start a new session, omit `session_id` and provide `document_id`.\n"
        "To refine further, send another prompt with the `session_id` from a "
        "previous response."
    ),
)
async def submit_agentic_rag(
    request: AgenticRagRequest,
) -> AgenticRagResponse:
    """FR3: Submit prompt → agentic RAG pipeline → updated notes + comparison."""
    logger.info(
        "Agentic RAG request: session=%s, prompt_len=%d, doc_id=%s, top_k=%d.",
        request.session_id or "new",
        len(request.prompt),
        request.document_id or "N/A",
        request.top_k,
    )

    try:
        result = process_agentic_rag(request)
    except AgenticRagError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except (EmbeddingError, VectorStoreError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except WebResearchError as exc:
        # Web research failures are non-fatal; log and continue
        logger.warning("Web research failed (non-fatal): %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Web research unavailable: {exc}",
        ) from exc
    except SessionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except QuotaExhaustedError:
        raise  # let the global exception handler in main.py return a proper 429 with quota payload
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during agentic RAG processing.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        ) from exc

    return result


# ── GET /api/v1/agentic-rag/health ─────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check agentic RAG pipeline health",
)
async def agentic_rag_health() -> HealthResponse:
    """Verify Pinecone connectivity and embedding service availability."""
    pinecone_ok = health_check()

    embedding_status = "ok"
    try:
        from app.services.embedding_service import embed_texts
        embed_texts(["health check"])
    except Exception as exc:  # noqa: BLE001
        embedding_status = f"error: {exc}"

    generation_status = "ok"
    try:
        import httpx as _httpx
        from app.core.config import get_settings
        import json
        settings = get_settings()
        health_payload = {
            "model": settings.openrouter_generation_model,
            "messages": [{"role": "user", "content": "Respond with exactly one word: ok"}],
            "max_tokens": 10,
            "temperature": 0.0,
        }
        health_headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        health_resp = _httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=health_headers,
            json=health_payload,
            timeout=30.0,
        )
        if health_resp.is_success:
            health_data = health_resp.json()
            choices = health_data.get("choices", [])
            text = (choices[0].get("message", {}).get("content", "") if choices else "").strip().lower()
            if text != "ok":
                generation_status = f"unexpected response: {text}"
        else:
            generation_status = f"OpenRouter returned HTTP {health_resp.status_code}"
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


# ── GET /api/v1/agentic-rag/sessions ───────────────────────────────────────────

@router.get(
    "/sessions",
    summary="List active agentic RAG sessions",
    description="Returns the count of currently active (non-expired) sessions.",
)
async def list_sessions() -> dict:
    """Return active session stats."""
    count = get_session_count()
    return {
        "status": "success",
        "active_sessions": count,
    }