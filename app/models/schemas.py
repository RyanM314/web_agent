"""
app/models/schemas.py
──────────────────────
Pydantic v2 schemas for all request/response bodies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


# ── Upload response ────────────────────────────────────────────────────────────

class ChunkInfo(BaseModel):
    chunk_index: int = Field(..., description="Zero-based position of this chunk")
    char_count: int = Field(..., description="Number of characters in the chunk")
    vector_id: str = Field(..., description="Pinecone vector ID for this chunk")


class UploadResponse(BaseModel):
    status: str = Field("success", description="'success' or 'error'")
    message: str
    document_id: str = Field(..., description="UUID that groups all chunks for this file")
    filename: str
    file_type: str
    total_chunks: int = Field(..., description="Number of chunks stored in Pinecone")
    chunks: list[ChunkInfo] = Field(default_factory=list)
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── List / status responses ────────────────────────────────────────────────────

class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_type: str
    total_chunks: int
    uploaded_at: str


class ListDocumentsResponse(BaseModel):
    status: str = "success"
    total_documents: int
    documents: list[DocumentSummary]


class DeleteResponse(BaseModel):
    status: str = "success"
    message: str
    document_id: str
    chunks_deleted: int


# ── Health check ───────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    pinecone_connected: bool
    embedding_service: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── FR2 — Prompt / Request Input ───────────────────────────────────────────────

class PromptRequest(BaseModel):
    """
    A lecturer's request to update one or more sections of their notes.

    If ``document_id`` is provided, the retrieval scope is narrowed to that
    specific document; otherwise it searches across **all** uploaded notes.
    """
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Natural-language request describing the update to make.",
        examples=[
            "Update the section on neural network backpropagation to include "
            "the newer AdamW optimiser and dropout regularisation."
        ],
    )
    document_id: str | None = Field(
        None,
        description=(
            "Optional UUID of a previously uploaded document. "
            "If omitted the system searches across all uploaded lecture notes."
        ),
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve from the vector store.",
    )


class RetrievedChunk(BaseModel):
    """A single chunk returned by the similarity search, shown for transparency."""

    vector_id: str = Field(..., description="Pinecone vector ID")
    document_id: str = Field(..., description="Source document UUID")
    filename: str = Field(..., description="Original upload filename")
    chunk_index: int = Field(..., description="Zero-based position in the source document")
    text: str = Field(..., description="The chunk text content")
    score: float = Field(..., description="Cosine-similarity score (0–1)")


class PromptResponse(BaseModel):
    """Response returned after the LLM processes the lecturer's update request."""

    model_config = {"protected_namespaces": ()}

    status: str = Field("success", description="'success' or 'error'")
    message: str = Field(
        ...,
        description="Human-readable summary of what was done.",
    )
    prompt: str = Field(..., description="The original prompt text")
    document_id: str | None = Field(
        None,
        description="Document UUID the prompt was scoped to (if provided).",
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Chunks retrieved from the vector store (top_k).",
    )
    generated_response: str = Field(
        ...,
        description="The LLM-generated updated content / answer.",
    )
    model_used: str = Field(
        ...,
        description="The Gemini model used for generation.",
    )
    processed_at: datetime = Field(default_factory=datetime.utcnow)


# ── FR3 — Agentic RAG Processing ───────────────────────────────────────────────

class AgenticRagRequest(BaseModel):
    """
    Request to start or continue an agentic RAG session.

    The system autonomously: retrieves relevant chunks from the vector store,
    performs real-time web research across trusted academic sources,
    cross-checks information, synthesises verified content, and delivers a
    formatted comparison of old vs new notes.

    If ``session_id`` is provided, the prompt is treated as an *iterative
    refinement* request within that existing session. Otherwise a new session
    is created.
    """
    session_id: str | None = Field(
        None,
        description="Existing session ID for iterative refinement. Omit to start a new session.",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Natural-language request describing the update to make.",
        examples=[
            "Update the section on neural network backpropagation to include "
            "the newer AdamW optimiser and dropout regularisation."
        ],
    )
    document_id: str | None = Field(
        None,
        description=(
            "Required for new sessions. UUID of a previously uploaded document "
            "containing the sections to update."
        ),
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve from the vector store.",
    )
    max_research_sources: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum number of web sources to consult during research.",
    )


class ResearchSource(BaseModel):
    """A web source consulted during the research phase."""

    title: str = Field(..., description="Page title")
    url: str = Field(..., description="Source URL")
    snippet: str = Field(..., description="Relevant text excerpt")
    relevance_score: float = Field(0.0, description="Estimated relevance (0–1)")


class SectionComparison(BaseModel):
    """Side-by-side comparison of an original section vs the updated version."""

    section_title: str = Field(
        ..., description="Heading or identifier for this section."
    )
    original_text: str = Field(
        ..., description="The original text from the uploaded notes."
    )
    updated_text: str = Field(
        ..., description="The newly generated updated text."
    )
    changes_summary: str = Field(
        ..., description="Human-readable summary of what changed and why."
    )


class SelfCorrectionReport(BaseModel):
    """Records any automatic corrections the system applied."""

    corrections_applied: list[str] = Field(
        default_factory=list,
        description="List of correction actions taken.",
    )
    length_original_sentences: int | None = Field(
        None, description="Sentence count in the original section."
    )
    length_updated_sentences: int | None = Field(
        None, description="Sentence count in the updated section."
    )
    research_sources_found: int = Field(
        0, description="Number of web sources successfully retrieved."
    )
    quality_score: float | None = Field(
        None, description="Overall quality score (0–1) after corrections."
    )


class AgenticRagResponse(BaseModel):
    """Full response from the agentic RAG pipeline."""

    model_config = {"protected_namespaces": ()}

    status: str = Field("success", description="'success' or 'error'")
    message: str = Field(
        ...,
        description="Human-readable summary of what was done.",
    )
    session_id: str = Field(
        ...,
        description="Session ID for iterative refinement. Send this back with follow-up prompts.",
    )
    prompt: str = Field(..., description="The original prompt text")
    document_id: str = Field(..., description="Document UUID processed.")
    filename: str = Field(
        "", description="Original filename of the document processed."
    )

    # Research
    research_sources: list[ResearchSource] = Field(
        default_factory=list,
        description="Sources consulted during real-time research.",
    )

    # Section comparisons (old vs new)
    section_comparisons: list[SectionComparison] = Field(
        default_factory=list,
        description="Side-by-side old vs new comparisons for each updated section.",
    )

    # Full generated output
    full_updated_notes: str = Field(
        ...,
        description="The complete updated version of the notes section(s).",
    )

    # Self-correction
    self_correction: SelfCorrectionReport = Field(
        default_factory=SelfCorrectionReport,
        description="Report of any automatic corrections applied.",
    )

    # Token usage & cost
    prompt_tokens: int = Field(
        0, description="Number of prompt tokens used.",
    )
    completion_tokens: int = Field(
        0, description="Number of completion tokens used.",
    )
    estimated_cost_usd: float = Field(
        0.0, description="Estimated cost in USD for this request.",
    )

    # Iteration metadata
    iteration_number: int = Field(
        1, description="Which iteration this is within the session."
    )
    total_iterations: int = Field(
        1, description="How many iterations have occurred in this session."
    )
    model_used: str = Field(
        ...,
        description="The OpenRouter model used for generation.",
    )
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Session history (for timeline view)
    session_history: list[dict] = Field(
        default_factory=list,
        description="Full iteration history for the session timeline.",
    )


# ── Error envelope (used by exception handlers) ────────────────────────────────

class ErrorResponse(BaseModel):
    status: str = "error"
    error_type: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)