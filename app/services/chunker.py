"""
app/services/chunker.py
───────────────────────
Splits a long document string into overlapping chunks suitable for
embedding and vector-database storage.

Strategy: character-level sliding window with configurable size and overlap.
This is intentionally simple and format-agnostic; semantic chunking (split
on sentence / paragraph boundaries) can be plugged in here later without
changing the rest of the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """One piece of a document, ready to be embedded."""

    index: int          # zero-based position within its source document
    text: str           # the actual content
    char_start: int     # character offset in the original text
    char_end: int       # exclusive end offset
    metadata: dict = field(default_factory=dict)  # propagated to Pinecone


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Divide *text* into overlapping chunks.

    Parameters
    ----------
    text         : full document text (already normalised).
    chunk_size   : maximum characters per chunk.
    chunk_overlap: characters re-used between consecutive chunks.
    metadata     : dict merged into every Chunk's metadata field.

    Returns
    -------
    list[Chunk]  — at least one chunk even for very short texts.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
        )

    base_meta = dict(metadata or {})
    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:                     # skip blank windows
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                    metadata={**base_meta, "chunk_index": len(chunks)},
                )
            )

        if end == len(text):
            break
        start += step

    logger.debug(
        "Chunked %d chars → %d chunks (size=%d, overlap=%d).",
        len(text),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
