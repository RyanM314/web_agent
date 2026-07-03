"""
app/api/routes/export_notes.py
───────────────────────────────
FR4 — Output Generation Export

Endpoints
─────────
POST /api/v1/export/pdf    Export generated notes as a PDF file
POST /api/v1/export/docx   Export generated notes as a DOCX file
GET  /api/v1/export/formats  List available export formats
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Response, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import ExportError
from app.services.export_service import export_to_docx, export_to_pdf

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/export",
    tags=["FR4 — Output Generation Export"],
)


# ── Request body ───────────────────────────────────────────────────────────────

class ExportRequest(BaseModel):
    """Request body for exporting generated notes."""

    markdown_text: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="The full markdown content of the updated notes to export.",
    )
    title: str = Field(
        "Updated Notes",
        max_length=200,
        description="Document title displayed in the exported file.",
    )
    document_id: str | None = Field(
        None,
        description="Optional document ID to locate the original file for table/image preservation.",
    )


# ── POST /api/v1/export/pdf ────────────────────────────────────────────────────

@router.post(
    "/pdf",
    summary="Export notes as a PDF file",
    description=(
        "Accepts markdown text content (typically the ``full_updated_notes`` "
        "from an FR3 AgenticRagResponse) and returns a downloadable PDF file. "
        "The PDF preserves headings, bullet lists, numbered lists, code blocks, "
        "and basic formatting."
    ),
    responses={
        200: {
            "description": "PDF file downloaded successfully.",
            "content": {"application/pdf": {}},
        },
        422: {"description": "Validation error in request body."},
        500: {"description": "PDF generation failed."},
    },
)
async def export_pdf(request: ExportRequest) -> Response:
    """Export the generated notes as a downloadable PDF."""
    logger.info(
        "PDF export requested: title='%s', text_len=%d.",
        request.title,
        len(request.markdown_text),
    )

    original_path = _resolve_original_path(request.document_id)

    try:
        pdf_buf = export_to_pdf(
            markdown_text=request.markdown_text,
            title=request.title,
            original_document_path=original_path,
        )
    except ExportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    # Sanitise filename
    safe_title = _sanitise_filename(request.title)

    return Response(
        content=pdf_buf.read(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_title}.pdf"',
            "Content-Length": str(pdf_buf.getbuffer().nbytes),
        },
    )


# ── POST /api/v1/export/docx ───────────────────────────────────────────────────

@router.post(
    "/docx",
    summary="Export notes as a DOCX file",
    description=(
        "Accepts markdown text content (typically the ``full_updated_notes`` "
        "from an FR3 AgenticRagResponse) and returns a downloadable DOCX file. "
        "The DOCX preserves headings, bullet lists, numbered lists, code blocks, "
        "and basic formatting using python-docx."
    ),
    responses={
        200: {
            "description": "DOCX file downloaded successfully.",
            "content": {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {}
            },
        },
        422: {"description": "Validation error in request body."},
        500: {"description": "DOCX generation failed."},
    },
)
async def export_docx(request: ExportRequest) -> Response:
    """Export the generated notes as a downloadable DOCX file."""
    logger.info(
        "DOCX export requested: title='%s', text_len=%d.",
        request.title,
        len(request.markdown_text),
    )

    original_path = _resolve_original_path(request.document_id)

    try:
        docx_buf = export_to_docx(
            markdown_text=request.markdown_text,
            title=request.title,
            original_document_path=original_path,
        )
    except ExportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    safe_title = _sanitise_filename(request.title)

    return Response(
        content=docx_buf.read(),
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        headers={
            "Content-Disposition": f'attachment; filename="{safe_title}.docx"',
            "Content-Length": str(docx_buf.getbuffer().nbytes),
        },
    )


# ── GET /api/v1/export/formats ─────────────────────────────────────────────────

@router.get(
    "/formats",
    summary="List available export formats",
    description="Returns a list of supported export file formats.",
)
async def list_formats() -> dict:
    """Return the available export formats."""
    return {
        "status": "success",
        "formats": [
            {
                "format": "pdf",
                "mime_type": "application/pdf",
                "extension": ".pdf",
                "description": "Portable Document Format — suitable for printing and sharing.",
            },
            {
                "format": "docx",
                "mime_type": (
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                "extension": ".docx",
                "description": "Word Open XML format — editable in Microsoft Word, Google Docs, etc.",
            },
        ],
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_original_path(document_id: str | None) -> str | None:
    """
    Resolve the original uploaded file path from the document_id.

    Searches the uploads directory for a file starting with the document_id.
    Returns the full path if found, or None if not found.
    """
    if not document_id:
        return None

    settings = get_settings()
    uploads_dir = Path(settings.upload_dir)

    try:
        matches = list(uploads_dir.glob(f"{document_id}_*"))
        if matches:
            return str(matches[0])
    except Exception as exc:
        logger.debug("Could not resolve original path for document '%s': %s", document_id, exc)

    return None


def _sanitise_filename(title: str) -> str:
    """Remove characters that are problematic in filenames."""
    import re
    safe = re.sub(r'[<>:"/\\|?*]', "_", title)
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe[:100] or "updated_notes"