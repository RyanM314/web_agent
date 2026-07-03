"""
app/services/upload_service.py
───────────────────────────────
Orchestrates the complete FR1 pipeline:

  1. Validate file (type + size)
  2. Save to disk
  3. Parse document → plain text
  4. Split text → chunks
  5. Embed chunks → vectors (Gemini)
  6. Upsert vectors → Pinecone
  7. Return a structured result

This is the single entry-point called by the API route handler.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.core.config import get_settings
from app.core.exceptions import (
    FileTooLargeError,
    UnsupportedFileTypeError,
    EmbeddingError,
)
from app.services.chunker import Chunk, split_text
from app.services.document_parser import parse_document
from app.services.embedding_service import embed_texts
from app.services.vector_store import generate_document_id, upsert_chunks

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

class UploadResult:
    """Plain data container; converted to UploadResponse by the route."""

    def __init__(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        chunks: list[Chunk],
        vector_ids: list[str],
        saved_path: Path,
        metadata: dict[str, Any],
    ) -> None:
        self.document_id = document_id
        self.filename = filename
        self.file_type = file_type
        self.chunks = chunks
        self.vector_ids = vector_ids
        self.saved_path = saved_path
        self.metadata = metadata
        self.uploaded_at = datetime.now(timezone.utc)


# ── Public function ────────────────────────────────────────────────────────────

async def process_upload(upload_file: UploadFile) -> UploadResult:
    """
    Run the full FR1 pipeline for one uploaded file.

    Parameters
    ----------
    upload_file : FastAPI UploadFile object from the route.

    Returns
    -------
    UploadResult with all the information needed to build the response.

    Raises
    ------
    UnsupportedFileTypeError, FileTooLargeError — validation failures.
    DocumentParseError, EmbeddingError, VectorStoreError — pipeline failures.
    """
    settings = get_settings()

    # ── 1. Validate file type ──────────────────────────────────────────────────
    original_filename = upload_file.filename or "unknown"
    ext = Path(original_filename).suffix.lower().lstrip(".")

    if ext not in settings.allowed_ext_set:
        raise UnsupportedFileTypeError(ext, settings.allowed_ext_set)

    # ── 2. Read content + validate size ───────────────────────────────────────
    content = await upload_file.read()
    if len(content) > settings.max_file_size_bytes:
        raise FileTooLargeError(len(content), settings.max_file_size_bytes)

    # ── 3. Persist to disk ────────────────────────────────────────────────────
    document_id = generate_document_id()
    safe_name = f"{document_id}_{_sanitise(original_filename)}"
    save_path = settings.upload_path / safe_name

    save_path.write_bytes(content)
    logger.info("Saved upload: %s (%d bytes)", save_path, len(content))

    # ── 4. Parse document → text ──────────────────────────────────────────────
    text = parse_document(save_path)

    # ── 5. Chunk text ─────────────────────────────────────────────────────────
    base_metadata: dict[str, Any] = {
        "document_id": document_id,
        "filename": original_filename,
        "file_type": ext,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": len(content),
    }

    chunks = split_text(
        text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        metadata=base_metadata,
    )
    logger.info("Document '%s' split into %d chunks.", original_filename, len(chunks))

    # ── 6. Embed chunks ───────────────────────────────────────────────────────
    chunk_texts = [c.text for c in chunks]
    try:
        vectors = embed_texts(chunk_texts)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(str(exc)) from exc

    # ── 7. Upsert to Pinecone ─────────────────────────────────────────────────
    vector_ids = upsert_chunks(document_id, chunks, vectors, base_metadata)

    return UploadResult(
        document_id=document_id,
        filename=original_filename,
        file_type=ext,
        chunks=chunks,
        vector_ids=vector_ids,
        saved_path=save_path,
        metadata=base_metadata,
    )


# ── Internal helper ────────────────────────────────────────────────────────────

def _sanitise(filename: str) -> str:
    """Remove path traversal characters and limit length."""
    safe = Path(filename).name          # strip any directory component
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in safe)
    return safe[:120]                   # reasonable file-name length cap
