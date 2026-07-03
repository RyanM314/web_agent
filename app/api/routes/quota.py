"""
app/api/routes/quota.py
────────────────────────
Quota status and control endpoints.

Endpoints
─────────
GET /api/v1/quota/status   Return current Gemini API quota state
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.core.quota_manager import get_quota_state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/quota", tags=["Quota Management"])


# ── GET /api/v1/quota/status ───────────────────────────────────────────────────

@router.get(
    "/status",
    summary="Check Gemini API quota status",
    description=(
        "Returns the current status of the Gemini API daily quota. "
        "If exhausted, includes the estimated reset time and a human-readable "
        "message. The frontend polls this endpoint to display quota notifications."
    ),
)
async def quota_status() -> dict:
    """Return structured quota status for the frontend."""
    qs = get_quota_state()
    info = qs.get_status_info()

    from datetime import datetime, timezone
    reset_dt = datetime.fromtimestamp(info["reset_at"], tz=timezone.utc)

    # Build a human-readable message
    if info["exhausted"]:
        reset_in = info["reset_in_seconds"]
        if reset_in > 3600:
            message = (
                f"You have reached your daily Gemini API quota; "
                f"please come back later. Your quota will reset "
                f"at {reset_dt.strftime('%H:%M UTC')} "
                f"(in approximately {reset_in // 3600}h {(reset_in % 3600) // 60}m)."
            )
        else:
            message = (
                f"You have reached your daily Gemini API quota; "
                f"please come back later. Resets in "
                f"{int(reset_in)} seconds."
            )
    else:
        message = "Gemini API quota is active."

    return {
        "status": "success",
        "quota_status": info["quota_status"],
        "exhausted": info["exhausted"],
        "reset_at_utc": reset_dt.isoformat(),
        "reset_in_seconds": info["reset_in_seconds"],
        "message": message,
        "last_error": info["last_error_detail"][:300] if info["last_error_detail"] else None,
    }