"""
app/services/agentic_rag_service.py
────────────────────────────────────
FR3 — LangChain Agentic RAG Pipeline

The autonomous system receives a lecturer's request/prompt, processes it
through modular components:

  1. Retrieve relevant chunks from the vector store (Pinecone)
  2. Perform real-time web research across trusted academic sources
  3. Cross-check and verify information
  4. Synthesise verified content via the LLM (OpenRouter → supported OpenRouter model)
  5. Build old-vs-new section comparisons
  6. Apply self-correction (length, style, research quality)
  7. Support iterative refinement via sessions

Financial Safety Brakes
───────────────────────
  • MAX_AGENT_TURNS = 4 — hard cap on OpenRouter LLM calls per pipeline run
    (prevents infinite loops from burning balance).
  • Mandatory 3.5 s ``time.sleep`` before every OpenRouter request
    (paces API traffic and protects against burst charges).
  • Every response logs ``prompt_tokens`` / ``completion_tokens`` to stdout
    for cost tracking.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import re
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import (
    AgenticRagError,
    EmbeddingError,
    QuotaExhaustedError,
    VectorStoreError,
    WebResearchError,
)
from app.models.schemas import (
    AgenticRagRequest,
    AgenticRagResponse,
    ResearchSource as ResearchSourceSchema,
    RetrievedChunk,
    SectionComparison,
    SelfCorrectionReport,
)
from app.services.comparison_service import build_section_comparisons
from app.services.embedding_service import embed_texts
from app.services.document_parser import parse_document
from app.services.self_correction import (
    build_length_correction_prompt,
    build_style_correction_prompt,
    check_length_deviation,
    check_research_quality,
    check_style_concordance,
    compute_quality_score,
    count_sentences,
)
from app.services.session_manager import (
    SessionEntry,
    create_session,
    get_session,
    update_session,
)
from app.services.vector_store import query as vector_query
from app.services.web_research import ResearchSource, perform_research

logger = logging.getLogger(__name__)


def _merge_prior_with_comparisons(prior_full_notes: str, comparisons: list[SectionComparison], fallback: str | None = None) -> str:
    """
    Merge updated section(s) from `comparisons` into the `prior_full_notes` string.

    Strategy:
    - Try a direct substring replace of the original section text.
    - If not found, look for a heading matching the section title and replace
      the section content between that heading and the next heading.
    - If still not found and the comparison represents a new section, append it.
    - If prior_full_notes is empty, return `fallback` or the generated updated
      text as a best-effort behaviour.
    """
    if not prior_full_notes:
        return fallback or (comparisons[0].updated_text if comparisons else "")

    merged = prior_full_notes
    for comp in comparisons:
        orig = (comp.original_text or "").strip()
        updated = (comp.updated_text or "").strip()

        if not orig or orig == "(New section — not in original notes)":
            # Append new section at end using a sensible heading if not present
            merged += f"\n\n## {comp.section_title}\n\n{updated}\n"
            continue

        # 1) Direct substring replace (best effort)
        idx = merged.find(orig)
        if idx >= 0:
            merged = merged.replace(orig, updated, 1)
            continue

        # 2) Try to find heading and replace until next heading
        try:
            # match heading line with the section title
            pattern = re.compile(r"(^#{1,6}\s+" + re.escape(comp.section_title) + r"\s*$)([\s\S]*?)(?=^#{1,6}\s+|\Z)", re.MULTILINE)
            def _repl(m):
                return m.group(1) + "\n\n" + updated + "\n"

            new_merged, n = pattern.subn(_repl, merged)
            if n:
                merged = new_merged
                continue
        except re.error:
            # graceful fallback if regex fails for some titles
            pass

        # 3) Fallback: best-effort substring approximate replace by first 100 chars
        snippet = orig[:200]
        idx2 = merged.find(snippet)
        if idx2 >= 0:
            # attempt to replace a wider window around snippet
            start = max(0, idx2 - 50)
            end = min(len(merged), idx2 + len(orig) + 50)
            merged = merged[:start] + updated + merged[end:]
        else:
            # last resort: append updated section
            merged += f"\n\n## {comp.section_title}\n\n{updated}\n"

    return merged

# ── Financial safety brakes ─────────────────────────────────────────────────────

MAX_AGENT_TURNS = 4          # hard cap on OpenRouter LLM calls per pipeline run
PACING_DELAY_S = 3.5         # mandatory sleep BEFORE every OpenRouter request

# OpenRouter endpoint (single source of truth)
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Token usage tracking (accumulated across all LLM calls in one pipeline run)
_token_usage_log: list[dict] = []


# ── System instruction template ────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = """You are an expert academic writing assistant helping \
university lecturers update and modernise their course notes.

You will be given:
1. **The lecturer's update request** — what they want to change or add.
2. **Relevant excerpts from their existing notes** — retrieved from their uploaded documents.
3. **Web research results** — real-time information from trusted academic sources.

Your task is to produce the **fully updated text** for the requested section(s).

Follow these rules:
• **Accuracy** – Only include facts supported by the retrieved context OR the web research. \
If neither contains enough information, clearly state what is missing.
• **Clarity** – Write in a clear, formal academic style suitable for lecture notes.
• **Structure** – Use consistent markdown headings, bullet points, and code blocks \
matching the original document's formatting.
• **Preservation** – Keep any existing content that does not need updating.
• **Modernisation** – Where the lecturer's request implies an outdated concept, \
update it with current best-practices found in web research or retrieved context.
• **Format** – Return the full updated section so it can replace the original. \
Do NOT include meta-commentary like "I have updated the section…". \
Just output the updated notes directly.
• **Length** – Match the length (number of sentences) of the original section approximately.
• **Style** – Use exactly the same heading style, list style, and overall formatting as the original.

Retrieved context from the lecturer's notes:
"""


# ── Public function ────────────────────────────────────────────────────────────

def process_agentic_rag(request: AgenticRagRequest) -> AgenticRagResponse:
    """
    Run the full FR3 agentic RAG pipeline.

    If ``request.session_id`` is provided, it's an iterative refinement call
    within an existing session. Otherwise a new session is created.

    All generation calls use a raw ``httpx.post`` to OpenRouter — no Google
    SDK is involved.  Embeddings still use the Google Gemini SDK.
    """
    settings = get_settings()
    start_time = datetime.now(timezone.utc)
    
    # Reset token usage log for this pipeline run
    global _token_usage_log
    _token_usage_log = []

    # ── Session handling ───────────────────────────────────────────────────────
    is_iteration = request.session_id is not None
    if is_iteration:
        session = get_session(request.session_id)
        # Check max iterations
        if session.iteration_count >= settings.session_max_iterations:
            raise AgenticRagError(
                f"Session '{request.session_id}' has reached the maximum "
                f"of {settings.session_max_iterations} iterations."
            )
        # Use the session's document context
        document_id = session.document_id
        filename = session.filename
        original_chunk_texts = session.original_chunk_texts
        original_metadata = session.original_metadata
        prior_full_notes = session.current_full_notes
        conversation_history = _build_history_context(session)
        logger.info(
            "Iterative refinement: session '%s', iteration %d.",
            request.session_id,
            session.iteration_count + 1,
        )
    else:
        if not request.document_id:
            raise AgenticRagError(
                "document_id is required when starting a new session."
            )
        document_id = request.document_id
        filename = ""
        prior_full_notes = ""
        conversation_history = ""

        # Retrieve chunks from vector store
        try:
            prompt_vectors = embed_texts([request.prompt])
        except EmbeddingError as exc:
            raise AgenticRagError(f"Failed to embed prompt: {exc}") from exc

        if not prompt_vectors:
            raise AgenticRagError("Embedding returned an empty result.")

        try:
            raw_results = vector_query(
                query_vector=prompt_vectors[0],
                top_k=request.top_k,
                document_id=document_id,
            )
        except VectorStoreError as exc:
            raise AgenticRagError(f"Failed to query vector store: {exc}") from exc

        if not raw_results:
            raise AgenticRagError(
                f"No vectors found for document '{document_id}'. "
                "Make sure the document has been uploaded first (FR1)."
            )

        original_chunk_texts = []
        original_metadata = []
        for match in raw_results:
            meta = match.get("metadata", {})
            original_chunk_texts.append(meta.get("text", ""))
            original_metadata.append(meta)
            if not filename and meta.get("filename"):
                filename = meta["filename"]

        # Create the session
        session_id = create_session(
            document_id=document_id,
            filename=filename,
            original_chunk_texts=original_chunk_texts,
            original_metadata=original_metadata,
        )
        session = get_session(session_id)
        logger.info(
            "New agentic RAG session '%s' for document '%s'.",
            session_id,
            document_id,
        )
        # Try to load the original full document text from the uploads folder
        try:
            uploads_dir = settings.upload_path
            matches = list(Path(uploads_dir).glob(f"{document_id}_*"))
            if matches:
                # parse_document applies the same normalization used on upload
                prior_full_notes = parse_document(matches[0])
                logger.debug("Loaded original document text from %s", matches[0])
        except Exception as exc:  # non-fatal
            logger.debug("Could not load original file for merging: %s", exc)

    # ── 1. Perform web research ────────────────────────────────────────────────
    # The research function internally condenses the raw prompt into 3-4
    # search-optimised keywords before hitting the search engine.
    research_queries = _build_research_queries(request.prompt)
    all_sources: list[ResearchSource] = []
    for query in research_queries:
        try:
            sources = perform_research(
                query=query,
                max_sources=request.max_research_sources,
            )
            all_sources.extend(sources)
        except WebResearchError as exc:
            logger.warning("Web research query failed: %s", exc)

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique_sources: list[ResearchSource] = []
    for src in all_sources:
        if src.url not in seen_urls:
            seen_urls.add(src.url)
            unique_sources.append(src)
    unique_sources = unique_sources[: request.max_research_sources]

    # Check research quality (self-correction)
    research_ok, research_msg = check_research_quality(len(unique_sources))
    corrections_applied: list[str] = []
    if not research_ok:
        corrections_applied.append(research_msg)

    # ── 2. Generate updated content via OpenRouter ─────────────────────────────
    context_str = "\n\n---\n\n".join(
        f"[Chunk from '{s.get('metadata', {}).get('filename', 'unknown')}' "
        f"(section index {i})]\n{t}"
        for i, (t, s) in enumerate(
            zip(original_chunk_texts, original_metadata)
        )
    )

    research_str = "\n\n".join(
        f"[Source: {s.title}]({s.url})\n{s.snippet}"
        for s in unique_sources
    ) if unique_sources else "No web research results were retrieved."

    all_original_text = "\n\n".join(original_chunk_texts)

    # Build the RAG prompt
    rag_prompt_parts = [
        _SYSTEM_INSTRUCTION,
        context_str,
        f"\n\nWeb Research Results:\n{research_str}",
    ]

    if conversation_history:
        rag_prompt_parts.append(f"\n\nConversation History:\n{conversation_history}")

    if prior_full_notes:
        rag_prompt_parts.append(
            f"\n\nPreviously generated notes (for reference — please refine "
            f"based on the new request):\n{prior_full_notes[:3000]}"
        )

    rag_prompt_parts.append(
        f"\n\n---\n\nLecturer's update request:\n{request.prompt}"
    )

    if not research_ok:
        rag_prompt_parts.append(
            "\n\nNote: Web research was limited. Base the update primarily "
            "on the retrieved notes context and your general knowledge."
        )

    full_prompt = "\n\n".join(rag_prompt_parts)

    # ── Agent turn counter (MAX_AGENT_TURNS) ────────────────────────────────
    # Every OpenRouter call in the self-correction loop consumes a turn. If the
    # counter exceeds MAX_AGENT_TURNS, the pipeline raises a clear error so the
    # frontend can surface the message to the user before burning more budget.
    turn_counter = 0

    def _next_turn() -> None:
        """Increment turn counter and enforce the hard ceiling."""
        nonlocal turn_counter
        turn_counter += 1
        if turn_counter > MAX_AGENT_TURNS:
            raise AgenticRagError(
                f"The agent exceeded the maximum number of LLM turns "
                f"({MAX_AGENT_TURNS}). This usually happens when the output "
                f"requires repeated self-correction. "
                f"Please try a more specific prompt or simplify your request."
            )

    # Initial generation (turn 1 of MAX_AGENT_TURNS)
    _next_turn()
    _pace_request()  # mandatory 3.5 s sleep before any OpenRouter call
    generated = _openrouter_generate(full_prompt, settings)

    # ── 3. Self-correction loop ───────────────────────────────────────────────
    # 3a. Length deviation check
    length_ok, orig_sentences, upd_sentences, length_msg = check_length_deviation(
        original_chunk_texts, generated
    )
    if not length_ok:
        corrections_applied.append(length_msg)
        length_correction = build_length_correction_prompt(
            all_original_text, generated, orig_sentences
        )
        full_prompt += f"\n\n{length_correction}"
        _next_turn()
        _pace_request()
        generated = _openrouter_generate(full_prompt, settings)

        # Second check after correction
        length_ok2, _, upd_sentences2, length_msg2 = check_length_deviation(
            original_chunk_texts, generated
        )
        if not length_ok2:
            corrections_applied.append(
                f"Second length adjustment attempted: {length_msg2}"
            )

    # 3b. Style concordance check
    style_ok, style_msg, style_issues = check_style_concordance(
        original_chunk_texts, generated
    )
    if not style_ok:
        corrections_applied.append(style_msg)
        style_correction = build_style_correction_prompt(all_original_text)
        full_prompt += f"\n\n{style_correction}"
        _next_turn()
        _pace_request()
        generated = _openrouter_generate(full_prompt, settings)

    # 3c. Quality score
    quality_score = compute_quality_score(
        generated,
        len(unique_sources),
        corrections_applied,
    )

    # If quality too low, one more regeneration attempt
    if quality_score < settings.min_quality_score and settings.max_self_correction_rounds > 1:
        corrections_applied.append(
            f"Quality score {quality_score} below threshold "
            f"{settings.min_quality_score}. Regenerating with quality emphasis."
        )
        full_prompt += (
            "\n\nIMPORTANT: The previous version had quality issues. "
            "Please produce a high-quality, well-structured, comprehensive "
            "update with sufficient detail and clear academic language."
        )
        _next_turn()
        _pace_request()
        generated = _openrouter_generate(full_prompt, settings)
        quality_score = compute_quality_score(
            generated, len(unique_sources), corrections_applied
        )

    # ── 4. Build section comparisons ──────────────────────────────────────────
    comparisons = build_section_comparisons(
        original_chunk_texts, original_metadata, generated
    )

    # ── 5. Build self-correction report ────────────────────────────────────────
    correction_report = SelfCorrectionReport(
        corrections_applied=corrections_applied,
        length_original_sentences=orig_sentences if not is_iteration else None,
        length_updated_sentences=upd_sentences if not is_iteration else None,
        research_sources_found=len(unique_sources),
        quality_score=quality_score,
    )

    # ── 6. Save to session ────────────────────────────────────────────────────
    processed_at = datetime.now(timezone.utc).isoformat()
    entry = SessionEntry(
        prompt=request.prompt,
        response_text=generated,
        research_sources=[s.__dict__ for s in unique_sources],
        corrections=corrections_applied,
        comparisons=[c.model_dump() for c in comparisons],
    )
    # Merge the generated section(s) back into the complete original document
    merged_full_notes = _merge_prior_with_comparisons(prior_full_notes, comparisons, fallback=generated)
    update_session(
        session_id=session.session_id,
        entry=entry,
        full_notes=merged_full_notes,
    )
    
    # ── Compute token usage & cost estimate ──────────────────────────────────
    # Cost rates per 1K tokens for gpt-4o-mini (~$0.150/1K input, ~$0.600/1K output)
    PROMPT_COST_PER_1K = 0.000150   # $0.150 per 1M tokens → $0.000150 per 1K
    COMPLETION_COST_PER_1K = 0.000600
    total_prompt_tokens = sum(t.get("prompt_tokens", 0) for t in _token_usage_log)
    total_completion_tokens = sum(t.get("completion_tokens", 0) for t in _token_usage_log)
    estimated_cost = (total_prompt_tokens * PROMPT_COST_PER_1K / 1000) + \
                     (total_completion_tokens * COMPLETION_COST_PER_1K / 1000)
    
    # ── Build session history for timeline ────────────────────────────────────
    session_history = []
    for i, hist_entry in enumerate(session.history):
        session_history.append({
            "iteration": i + 1,
            "prompt": hist_entry.prompt[:200],
            "response_preview": hist_entry.response_text[:300],
            "corrections": hist_entry.corrections,
            "sources_count": len(hist_entry.research_sources),
        })

    # ── 7. Assemble response ──────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    session = get_session(session.session_id)

    message_parts = [
        f"Processed in {elapsed:.1f}s. "
        f"Iteration {session.iteration_count} of "
        f"session '{session.session_id}'.",
        f"Retrieved {len(original_chunk_texts)} chunk(s) from the vector store.",
    ]
    if unique_sources:
        message_parts.append(
            f"Consulted {len(unique_sources)} web research source(s)."
        )
    if corrections_applied:
        message_parts.append(
            f"Automatic corrections applied: {len(corrections_applied)}."
        )
    message_parts.append(
        f"Quality score: {quality_score:.2f}. "
        f"To refine further, send another prompt with session_id='{session.session_id}'."
    )

    return AgenticRagResponse(
        status="success",
        message=" ".join(message_parts),
        session_id=session.session_id,
        prompt=request.prompt,
        document_id=document_id,
        filename=filename,
        research_sources=[
            ResearchSourceSchema(
                title=s.title,
                url=s.url,
                snippet=s.snippet[:300],
                relevance_score=s.relevance_score,
            )
            for s in unique_sources
        ],
        section_comparisons=comparisons,
        full_updated_notes=session.current_full_notes,
        self_correction=correction_report,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        estimated_cost_usd=round(estimated_cost, 6),
        iteration_number=session.iteration_count,
        total_iterations=session.iteration_count,
        model_used=settings.openrouter_generation_model,
        processed_at=start_time,
        session_history=session_history,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _pace_request() -> None:
    """
    Mandatory pacing delay before every OpenRouter API call.

    Sleeps for ``PACING_DELAY_S`` (3.5 s) to space out consecutive requests
    and protect the OpenRouter balance from burst charges.
    """
    time.sleep(PACING_DELAY_S)


def _openrouter_generate(prompt: str, settings) -> str:
    """
    Single, unified, bulletproof HTTP text-generation call to OpenRouter.

    Uses a bare ``httpx.post`` — no Google SDK, no wrappers, no function
    calling agents.  The combined prompt string (system instruction +
    retrieved context + web research + user request) is sent as a single
    user message.

    Endpoint
    --------
    POST https://openrouter.ai/api/v1/chat/completions

    Headers
    -------
    Authorization: Bearer {OPENROUTER_API_KEY}
    Content-Type:  application/json

    Body (OpenAI / OpenRouter layout)
    ---------------------------------
    {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "..."}],
        "temperature": 0.3,
        "max_tokens": 4096
    }

    Response token usage is logged at INFO level for cost tracking.
    """
    api_key = settings.openrouter_api_key
    if not api_key:
        raise AgenticRagError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )

    # ── Headers ───────────────────────────────────────────────────────────────
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # ── Request body — standard OpenAI / OpenRouter format ────────────────────
    payload: dict[str, Any] = {
        "model": settings.openrouter_generation_model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": settings.generation_temperature,
        "max_tokens": settings.generation_max_output_tokens,
    }

    logger.debug(
        "OpenRouter POST: url=%s model=%s key_prefix=%s",
        _OPENROUTER_URL,
        payload["model"],
        api_key[:6] + "..." if len(api_key) > 6 else "(short)",
    )

    # ── Bare HTTP POST — zero SDK involvement ────────────────────────────────
    try:
        resp = httpx.post(
            _OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=120.0,
        )
    except httpx.TimeoutException:
        raise AgenticRagError(
            "OpenRouter request timed out after 120 seconds."
        )
    except httpx.RequestError as exc:
        raise AgenticRagError(
            f"OpenRouter network error: {exc}"
        )

    # ── Error handling ───────────────────────────────────────────────────────
    if not resp.is_success:
        detail = _try_extract_error(resp.text)
        if resp.status_code == 401:
            raise AgenticRagError(
                "OpenRouter authentication failed. Check your OPENROUTER_API_KEY."
            )
        if resp.status_code == 402:
            raise AgenticRagError(
                "OpenRouter account balance is insufficient. "
                "Please top up at https://openrouter.ai/settings/credits."
            )
        if resp.status_code == 429:
            raise AgenticRagError(
                "OpenRouter rate limit hit. Please wait a moment and try again."
            )
        raise AgenticRagError(
            f"OpenRouter API error (HTTP {resp.status_code}): {detail}"
        )

    # ── Parse JSON response ──────────────────────────────────────────────────
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise AgenticRagError(
            f"OpenRouter returned non-JSON response: {resp.text[:500]}"
        ) from exc

    # ── Extract generated text ───────────────────────────────────────────────
    try:
        choices = data.get("choices", [])
        if not choices:
            raise AgenticRagError("OpenRouter returned no choices.")
        text = choices[0].get("message", {}).get("content", "")
        if not text:
            raise AgenticRagError("OpenRouter returned an empty response.")
    except (KeyError, IndexError, TypeError) as exc:
        raise AgenticRagError(
            f"Unexpected OpenRouter response structure: {exc}. "
            f"Raw response: {json.dumps(data, indent=2)[:1000]}"
        ) from exc

    # ── Log token usage for cost tracking ────────────────────────────────────
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    logger.info(
        "OpenRouter — generated %d chars | "
        "prompt_tokens=%d  completion_tokens=%d  total_tokens=%d",
        len(text),
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
    )
    
    # Accumulate token usage
    global _token_usage_log
    _token_usage_log.append({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    })

    return text


def _try_extract_error(body_text: str) -> str:
    """Try to parse an error detail from an API error response body."""
    try:
        body = json.loads(body_text)
        if "error" in body:
            err = body["error"]
            if isinstance(err, dict):
                return err.get("message", str(err))
            return str(err)
        return body_text[:500]
    except (json.JSONDecodeError, TypeError):
        return body_text[:500]


def _build_research_queries(prompt: str) -> list[str]:
    """
    Build targeted web research queries from the lecturer's prompt.

    Extracts key technical terms and constructs 1–2 search queries.
    Note: the resulting queries are further condensed into 3-4 keywords
    inside ``perform_research`` via the search query condenser.
    """
    import re

    # Extract capitalised multi-word terms (likely technical concepts)
    terms = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", prompt)

    # Build primary query from the prompt itself
    primary = prompt[:200].strip()

    # Build secondary query from extracted terms
    if terms:
        secondary = " ".join(terms[:8])
    else:
        secondary = prompt[:150].strip()

    queries = [primary]
    if secondary and secondary != primary:
        queries.append(secondary)

    return queries


def _build_history_context(session) -> str:
    """Build a conversation history string from previous iterations."""
    if not session.history:
        return ""

    parts = ["Previous iterations:"]
    for i, entry in enumerate(session.history, start=1):
        parts.append(f"\nIteration {i}:")
        parts.append(f"  Request: {entry.prompt[:200]}")
        parts.append(f"  Response (first 500 chars): {entry.response_text[:500]}")
        if entry.corrections:
            parts.append(f"  Corrections: {'; '.join(entry.corrections)}")

    return "\n".join(parts)