"""
app/api/routes/upload.py
────────────────────────
FR1 — Outdated Notes Upload & Storage

Endpoints
─────────
POST /api/v1/upload          Upload a single file → parse → embed → Pinecone
GET  /api/v1/upload/health   Check Pinecone + embedding connectivity
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.exceptions import (
    DocumentParseError,
    EmbeddingError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    VectorStoreError,
)
from app.models.schemas import (
    ChunkInfo,
    HealthResponse,
    UploadResponse,
)
from app.services.upload_service import process_upload
from app.services.vector_store import delete_document, health_check

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/upload", tags=["FR1 — Upload & Storage"])


# ── POST /api/v1/upload ────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an outdated lecturer note file",
    description=(
        "Accepts a TXT, PDF, or DOCX file. "
        "The system parses it, splits it into chunks, embeds each chunk "
        "using Google Gemini, and stores the resulting vectors in Pinecone."
    ),
)
async def upload_notes(
    file: UploadFile = File(
        ...,
        description="Outdated lecturer notes file (TXT, PDF, or DOCX, max 50 MB).",
    ),
) -> UploadResponse:
    """FR1: Upload outdated notes → parse → embed → store in Pinecone."""
    settings = get_settings()

    logger.info(
        "Upload request: filename='%s' content_type='%s'",
        file.filename,
        file.content_type,
    )

    try:
        result = await process_upload(file)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except DocumentParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse document: {exc}",
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
        logger.exception("Unexpected error during upload.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        ) from exc

    chunk_infos = [
        ChunkInfo(
            chunk_index=chunk.index,
            char_count=len(chunk.text),
            vector_id=vid,
        )
        for chunk, vid in zip(result.chunks, result.vector_ids)
    ]

    return UploadResponse(
        status="success",
        message=(
            f"'{result.filename}' was successfully processed and stored. "
            f"{result.metadata.get('file_size_bytes', 0) // 1024} KB parsed into "
            f"{len(result.chunks)} chunks, all embedded and indexed in Pinecone."
        ),
        document_id=result.document_id,
        filename=result.filename,
        file_type=result.file_type,
        total_chunks=len(result.chunks),
        chunks=chunk_infos,
        uploaded_at=result.uploaded_at,
        metadata={
            "file_size_bytes": result.metadata.get("file_size_bytes"),
            "embedding_model": settings.embedding_model,
            "pinecone_index": settings.pinecone_index_name,
        },
    )


# ── DELETE /api/v1/upload/{document_id} ───────────────────────────────────────

@router.delete(
    "/{document_id}",
    summary="Delete a previously uploaded document",
    description="Removes all Pinecone vectors associated with the document_id.",
)
async def delete_notes(document_id: str) -> dict:
    """Remove a document's embeddings from Pinecone."""
    try:
        delete_document(document_id)
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return {
        "status": "success",
        "message": f"Document '{document_id}' deleted from vector store.",
        "document_id": document_id,
    }


# ── GET /api/v1/upload/health ──────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check upload pipeline health",
)
async def upload_health() -> HealthResponse:
    """Verify Pinecone connectivity and embedding service availability."""
    settings = get_settings()
    pinecone_ok = health_check()

    # Light-weight embedding check: embed a single short string
    embedding_status = "ok"
    try:
        from app.services.embedding_service import embed_texts

        embed_texts(["health check"])
    except Exception as exc:  # noqa: BLE001
        embedding_status = f"error: {exc}"

    return HealthResponse(
        status="healthy" if (pinecone_ok and embedding_status == "ok") else "degraded",
        app_name=settings.app_name,
        version=settings.app_version,
        pinecone_connected=pinecone_ok,
        embedding_service=embedding_status,
    )
