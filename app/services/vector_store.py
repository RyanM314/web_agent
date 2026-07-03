"""
app/services/vector_store.py
────────────────────────────
Manages all interactions with Pinecone:

• Lazy index creation (serverless, us-east-1 by default).
• Upsert vectors with rich metadata so documents can be filtered / deleted.
• Query by document_id for listing and deletion.
• Delete all vectors belonging to a document.

Every public function is synchronous and thread-safe (Pinecone client is
stateless under the hood).  Async wrappers can be added later with
asyncio.to_thread if needed.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pinecone import Pinecone, ServerlessSpec  # pip install pinecone

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.services.chunker import Chunk

logger = logging.getLogger(__name__)

# Module-level singleton — initialised on first use
_pinecone_client: Pinecone | None = None
_index = None


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_client() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        settings = get_settings()
        if not settings.pinecone_api_key:
            raise VectorStoreError(
                "PINECONE_API_KEY is not set. Add it to your .env file."
            )
        _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return _pinecone_client


def _get_index():
    """
    Return the Pinecone Index object, creating the index if it does not exist.

    Automatic dimension fix
    ───────────────────────
    If the existing Pinecone index has a different dimension than the configured
    embedding dimension (e.g. old index was 768 but the model now outputs 3072),
    the index is automatically deleted and recreated with the correct dimension.
    This prevents the "Vector dimension X does not match the dimension of the index Y" error.
    """
    global _index
    if _index is not None:
        return _index

    settings = get_settings()
    pc = _get_client()
    index_name = settings.pinecone_index_name
    expected_dim = settings.embedding_dimension  # from .env

    existing_indexes = {idx.name: idx for idx in pc.list_indexes()}

    # ── Auto-fix dimension mismatch ──────────────────────────────────────
    if index_name in existing_indexes:
        existing_idx = existing_indexes[index_name]
        actual_dim = existing_idx.dimension
        if actual_dim != expected_dim:
            logger.warning(
                "Dimension mismatch detected: Pinecone index '%s' has "
                "dimension %d but configured embedding dimension is %d. "
                "Recreating index with dimension=%d …",
                index_name, actual_dim, expected_dim, expected_dim,
            )
            pc.delete_index(index_name)
            # Remove from the local dict so the creation branch runs below
            del existing_indexes[index_name]
            logger.info("Old index '%s' deleted. Re-creating …", index_name)

    # ── Create index if it doesn't exist ─────────────────────────────────
    if index_name not in existing_indexes:
        logger.info(
            "Creating Pinecone index '%s' (dimension=%d, metric='cosine') …",
            index_name, expected_dim,
        )
        pc.create_index(
            name=index_name,
            dimension=expected_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.pinecone_environment,
            ),
        )
        logger.info("Index '%s' created.", index_name)
    else:
        logger.debug("Using existing Pinecone index '%s'.", index_name)

    _index = pc.Index(index_name)
    return _index


# ── Public API ─────────────────────────────────────────────────────────────────

def upsert_chunks(
    document_id: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
    doc_metadata: dict[str, Any],
) -> list[str]:
    """
    Store chunk vectors in Pinecone.

    Parameters
    ----------
    document_id  : UUID string grouping all chunks for a file.
    chunks       : list of Chunk objects (from chunker.split_text).
    vectors      : list of float vectors, same length and order as chunks.
    doc_metadata : file-level metadata (filename, file_type, …) merged
                   into every vector's metadata.

    Returns
    -------
    list[str] — the Pinecone vector IDs, one per chunk.
    """
    if len(chunks) != len(vectors):
        raise VectorStoreError(
            f"Mismatch: {len(chunks)} chunks but {len(vectors)} vectors."
        )

    index = _get_index()
    vector_ids: list[str] = []
    records: list[dict] = []

    for chunk, vector in zip(chunks, vectors):
        vector_id = f"{document_id}__chunk_{chunk.index:04d}"
        vector_ids.append(vector_id)
        records.append(
            {
                "id": vector_id,
                "values": vector,
                "metadata": {
                    **doc_metadata,
                    "document_id": document_id,
                    "chunk_index": chunk.index,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "text": chunk.text[:1_000],   # Pinecone metadata cap
                },
            }
        )

    # Pinecone recommends batches of ≤ 100 vectors
    _BATCH = 100
    for i in range(0, len(records), _BATCH):
        batch = records[i : i + _BATCH]
        try:
            index.upsert(vectors=batch)
            logger.debug("Upserted %d vectors (batch %d).", len(batch), i // _BATCH)
        except Exception as exc:
            raise VectorStoreError(
                f"Pinecone upsert failed at batch {i // _BATCH}: {exc}"
            ) from exc

    logger.info(
        "Stored %d vectors for document '%s' in Pinecone.", len(records), document_id
    )
    return vector_ids


def delete_document(document_id: str) -> int:
    """
    Delete all vectors belonging to a document.

    Returns
    -------
    int — number of vectors deleted (approximate, from fetch before delete).
    """
    index = _get_index()

    # Pinecone's delete-by-filter (serverless) uses metadata filter
    try:
        # List vector IDs by prefix then delete
        # Serverless indexes support delete with filter on metadata
        index.delete(filter={"document_id": {"$eq": document_id}})
        logger.info("Deleted vectors for document '%s'.", document_id)
        return -1   # Pinecone serverless delete doesn't return a count
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to delete vectors for document '{document_id}': {exc}"
        ) from exc


def health_check() -> bool:
    """Return True if Pinecone is reachable and the index exists."""
    try:
        _get_index()
        return True
    except Exception as exc:
        logger.warning("Pinecone health check failed: %s", exc)
        return False


def query(
    query_vector: list[float],
    top_k: int = 5,
    document_id: str | None = None,
    include_metadata: bool = True,
) -> list[dict]:
    """
    Search the Pinecone index for the most similar vectors.

    Parameters
    ----------
    query_vector : The embedding vector to search with.
    top_k        : Number of nearest neighbours to return.
    document_id  : Optional — narrow the search to a single document.
    include_metadata : Whether to include metadata in results.

    Returns
    -------
    list[dict] — each dict contains 'id', 'score', and optionally 'metadata'.
    """
    index = _get_index()

    filter_expr: dict | None = None
    if document_id is not None:
        filter_expr = {"document_id": {"$eq": document_id}}

    try:
        result = index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter_expr,
            include_metadata=include_metadata,
        )
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata if include_metadata else {},
            }
            for match in result.matches
        ]
    except Exception as exc:
        raise VectorStoreError(f"Pinecone query failed: {exc}") from exc


def list_document_ids() -> list[str]:
    """
    Fetch all unique document IDs from the vector store.

    Uses Pinecone's list endpoint with the document_id prefix pattern.
    This is a best-effort scan; very large indexes may need pagination.
    """
    index = _get_index()

    # Pinecone serverless supports listing vector IDs via the list endpoint.
    # We paginate through all IDs and extract unique document_id prefixes.
    doc_ids: set[str] = set()
    pagination_token: str | None = None

    try:
        while True:
            response = index.list_paginated(
                prefix="",
                limit=1000,
                pagination_token=pagination_token,
            )
            for vec_id in response.vectors or []:
                # vector_id format: {document_id}__chunk_{index}
                if "__chunk_" in vec_id.id:
                    doc_id = vec_id.id.split("__chunk_")[0]
                    doc_ids.add(doc_id)
            pagination_token = response.pagination.next
            if pagination_token is None:
                break
    except Exception:
        # list_paginated may not be available on all plans; fallback quietly
        logger.warning("Could not list document IDs from Pinecone (may be unsupported).")

    return sorted(doc_ids)


def recreate_index() -> None:
    """
    Delete the current Pinecone index and re-create it with the
    dimension and metric configured in settings (dimension=3072,
    metric='cosine').

    Call this once after switching to a new embedding model whose
    output dimension differs from the existing index dimension.
    """
    global _index
    settings = get_settings()
    pc = _get_client()
    index_name = settings.pinecone_index_name

    # 1. Delete the existing index if it exists
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        logger.warning("Deleting Pinecone index '%s' …", index_name)
        pc.delete_index(index_name)
        logger.info("Index '%s' deleted.", index_name)
    else:
        logger.info("Index '%s' does not exist yet — skipping delete.", index_name)

    # 2. Re-create with the current settings (dimension=3072, metric='cosine')
    logger.info(
        "Creating Pinecone index '%s' (dimension=%d, metric='cosine') …",
        index_name,
        settings.embedding_dimension,
    )
    pc.create_index(
        name=index_name,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=settings.pinecone_environment,
        ),
    )
    logger.info("Index '%s' re-created successfully.", index_name)

    # 3. Reset module-level _index so the next _get_index() call returns the new one
    _index = None


def generate_document_id() -> str:
    """Generate a unique document identifier."""
    return str(uuid.uuid4())
