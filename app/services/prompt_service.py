"""
app/services/prompt_service.py
───────────────────────────────
FR2 — RAG Prompt Processing Pipeline

Orchestrates the request-to-response flow:

  1. Embed the user's prompt using Gemini.
  2. Retrieve the most relevant chunks from Pinecone.
  3. Build a RAG prompt that includes the retrieved context.
  4. Send the prompt to Gemini for generation.
  5. Return the generated content + retrieval metadata.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError, PromptProcessingError, VectorStoreError
from app.models.schemas import PromptRequest, PromptResponse, RetrievedChunk
from app.services.embedding_service import embed_texts
from app.services.vector_store import query as vector_query

logger = logging.getLogger(__name__)

# ── System instruction template ────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = """\
You are an AI assistant that helps university lecturers update and modernise \
their course notes. You will be given:

1. **The lecturer's update request** — what they want to change or add.
2. **Relevant excerpts from their existing notes** — retrieved from a vector \
database of their uploaded documents.

Your task is to produce the **updated text** for the requested section(s). \
Follow these rules:

• **Accuracy** – Only include facts supported by the retrieved context. If the \
context does not contain enough information, state what is missing.
• **Clarity** – Write in a clear, academic style suitable for undergraduate or \
postgraduate lecture notes.
• **Structure** – Use markdown headings, bullet points, and code blocks where \
appropriate to make the notes easy to read.
• **Preservation** – Keep any existing content that does not need updating; \
only change what the lecturer asked to be updated.
• **Modernisation** – Where the lecturer's request implies an outdated concept, \
update it with current best-practices found in the retrieved context.
• **Format** – Return the full updated section so it can replace the original. \
Do not include meta-commentary like "I have updated the section…". Just output \
the updated notes directly.

Retrieved context from the lecturer's notes:
"""


# ── Public function ────────────────────────────────────────────────────────────

def process_prompt(request: PromptRequest) -> PromptResponse:
    """
    Run the full FR2 RAG pipeline.

    Parameters
    ----------
    request : PromptRequest
        The validated request from the API endpoint.

    Returns
    -------
    PromptResponse
        The LLM-generated updated content along with retrieval metadata.

    Raises
    ------
    PromptProcessingError
        On any failure in embedding, retrieval, or generation.
    """
    settings = get_settings()
    start_time = datetime.now(timezone.utc)

    # ── 1. Embed the prompt ────────────────────────────────────────────────────
    logger.info("Processing prompt (length=%d chars).", len(request.prompt))
    try:
        prompt_vectors = embed_texts([request.prompt])
    except EmbeddingError as exc:
        raise PromptProcessingError(f"Failed to embed prompt: {exc}") from exc

    if not prompt_vectors:
        raise PromptProcessingError("Embedding returned an empty result.")
    query_vector = prompt_vectors[0]

    # ── 2. Retrieve relevant chunks from Pinecone ──────────────────────────────
    try:
        raw_results = vector_query(
            query_vector=query_vector,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except VectorStoreError as exc:
        raise PromptProcessingError(f"Failed to query vector store: {exc}") from exc

    if not raw_results:
        logger.warning("No relevant chunks found for the given prompt.")
        # Still try to generate with an empty context — the LLM will handle it

    # ── 3. Build retrieved-chunk metadata ──────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk] = []
    context_parts: list[str] = []

    for rank, match in enumerate(raw_results, start=1):
        meta = match.get("metadata", {})
        chunk = RetrievedChunk(
            vector_id=match["id"],
            document_id=meta.get("document_id", "unknown"),
            filename=meta.get("filename", "unknown"),
            chunk_index=meta.get("chunk_index", -1),
            text=meta.get("text", ""),
            score=match.get("score", 0.0),
        )
        retrieved_chunks.append(chunk)
        context_parts.append(
            f"[Chunk {rank} — from '{chunk.filename}' "
            f"(document: {chunk.document_id})]\n{chunk.text}"
        )

    context_str = "\n\n".join(context_parts) if context_parts else "No relevant context was retrieved from the vector store."

    # ── 4. Build RAG prompt ────────────────────────────────────────────────────
    rag_prompt = (
        f"{_SYSTEM_INSTRUCTION}\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"Lecturer's update request:\n{request.prompt}"
    )

    # ── 5. Generate with Gemini ────────────────────────────────────────────────
    generated_response = _generate_with_gemini(rag_prompt, settings)

    # ── 6. Assemble response ───────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    message = (
        f"Prompt processed in {elapsed:.1f}s. "
        f"Retrieved {len(retrieved_chunks)} relevant chunk(s) "
        f"from {'document ' + request.document_id if request.document_id else 'all uploaded documents'}. "
        f"Generated response using {settings.embedding_generation_model}."
    )

    return PromptResponse(
        status="success",
        message=message,
        prompt=request.prompt,
        document_id=request.document_id,
        retrieved_chunks=retrieved_chunks,
        generated_response=generated_response,
        model_used=settings.embedding_generation_model,
        processed_at=start_time,
    )


# ── Internal helper — Gemini generation ────────────────────────────────────────

def _generate_with_gemini(rag_prompt: str, settings) -> str:
    """Send the RAG prompt to Gemini and return the generated text."""
    try:
        from google.genai import types as genai_types
        from app.core.gemini_client import create_gemini_client
        from app.core.gemini_middleware import call_gemini

        client = create_gemini_client(settings.google_api_key)

        response = call_gemini(
            lambda: client.models.generate_content(
                model=settings.embedding_generation_model,
                contents=rag_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=settings.generation_temperature,
                    max_output_tokens=settings.generation_max_output_tokens,
                ),
            ),
            service_name="prompt_generate",
        )

        generated = response.text
        if not generated:
            raise PromptProcessingError("Gemini returned an empty response.")

        logger.info(
            "Gemini generation complete (%d chars).",
            len(generated),
        )
        return generated

    except PromptProcessingError:
        raise
    except Exception as exc:
        raise PromptProcessingError(
            f"Gemini generation failed: {exc}"
        ) from exc
