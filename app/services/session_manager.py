"""
app/services/session_manager.py
────────────────────────────────
FR3 — Session management for iterative refinement.

Manages in-memory sessions that store:
  - The original document context (chunks + metadata)
  - The conversation history (prompts + responses)
  - The iteration count
  - The current state of the generated notes

Sessions are stored in a thread-safe dict with TTL-based eviction.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import SessionError
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class SessionEntry:
    """One iteration in a session's conversation history."""

    prompt: str
    response_text: str
    research_sources: list[dict] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    comparisons: list[dict] = field(default_factory=list)


@dataclass
class Session:
    """Holds all state for one agentic RAG session."""

    session_id: str
    document_id: str
    filename: str
    original_chunk_texts: list[str] = field(default_factory=list)
    original_metadata: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    iteration_count: int = 0
    history: list[SessionEntry] = field(default_factory=list)
    current_full_notes: str = ""
    processed_at: str = ""


# ── Thread-safe session store ──────────────────────────────────────────────────

_sessions: dict[str, Session] = {}
_lock = Lock()
_CLEANUP_INTERVAL: int = 300  # 5 minutes
_last_cleanup: float = time.time()


def create_session(
    document_id: str,
    filename: str,
    original_chunk_texts: list[str],
    original_metadata: list[dict],
) -> str:
    """
    Create a new session and return its unique ID.

    Parameters
    ----------
    document_id         : UUID of the document being updated.
    filename            : Original filename for display.
    original_chunk_texts: Text chunks from the vector store retrieval.
    original_metadata   : Corresponding metadata dicts.

    Returns
    -------
    str — the new session ID.
    """
    session_id = str(uuid.uuid4())
    session = Session(
        session_id=session_id,
        document_id=document_id,
        filename=filename,
        original_chunk_texts=original_chunk_texts,
        original_metadata=original_metadata,
    )

    with _lock:
        _sessions[session_id] = session
        _evict_expired()

    logger.info("Created session '%s' for document '%s'.", session_id, document_id)
    return session_id


def get_session(session_id: str) -> Session:
    """
    Retrieve an existing session.

    Raises SessionError if the session does not exist or has expired.
    """
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            raise SessionError(
                f"Session '{session_id}' not found. It may have expired."
            )
        settings = get_settings()
        age = time.time() - session.created_at
        max_age = settings.session_ttl_minutes * 60
        if age > max_age:
            del _sessions[session_id]
            raise SessionError(
                f"Session '{session_id}' has expired (TTL={settings.session_ttl_minutes} minutes)."
            )
        session.last_accessed = time.time()
        return session


def update_session(
    session_id: str,
    entry: SessionEntry,
    full_notes: str,
) -> Session:
    """
    Append an iteration entry and update the current notes in a session.

    Returns the updated Session.
    """
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            raise SessionError(f"Session '{session_id}' not found.")

        session.iteration_count += 1
        session.history.append(entry)
        session.current_full_notes = full_notes
        session.last_accessed = time.time()

        logger.info(
            "Session '%s' iteration %d updated.",
            session_id,
            session.iteration_count,
        )
        return session


def get_session_count() -> int:
    """Return the total number of active sessions."""
    with _lock:
        _evict_expired()
        return len(_sessions)


def _evict_expired() -> None:
    """Remove sessions that have exceeded their TTL."""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < _CLEANUP_INTERVAL:
        return

    settings = get_settings()
    max_age = settings.session_ttl_minutes * 60
    expired = [
        sid for sid, s in list(_sessions.items())
        if (now - s.created_at) > max_age
    ]
    for sid in expired:
        del _sessions[sid]
        logger.debug("Evicted expired session '%s'.", sid)

    _last_cleanup = now