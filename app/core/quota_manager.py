"""
app/core/quota_manager.py
─────────────────────────
Quota state manager for Google Gemini API.

Tracks whether the daily quota has been exhausted and computes when the
next quota reset will occur (midnight UTC). Exposes a thread-safe singleton
that all service layers consult before making Gemini API calls.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Quota state constants ──────────────────────────────────────────────────────

QUOTA_EXHAUSTED_CODE = 429
QUOTA_STATUS_ACTIVE = "active"
QUOTA_STATUS_EXHAUSTED = "exhausted"
QUOTA_STATUS_UNKNOWN = "unknown"

# Gemini free-tier quota resets at midnight UTC.
# The error message may include a retry delay in seconds; we honour that,
# but also always compute the next midnight boundary.
_QUOTA_RESET_HOUR_UTC = 0
_QUOTA_RESET_MINUTE_UTC = 0


def _next_midnight_utc() -> float:
    """Return the Unix timestamp of the next midnight UTC."""
    now = datetime.now(timezone.utc)
    midnight = now.replace(
        hour=_QUOTA_RESET_HOUR_UTC,
        minute=_QUOTA_RESET_MINUTE_UTC,
        second=0,
        microsecond=0,
    )
    if midnight <= now:
        # Already past today's midnight; go to tomorrow
        import datetime as dt
        midnight += dt.timedelta(days=1)
    return midnight.timestamp()


# ── Thread-safe quota state ────────────────────────────────────────────────────

class QuotaState:
    """Thread-safe singleton that tracks Gemini API quota status."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: str = QUOTA_STATUS_ACTIVE
        self._exhausted_at: float | None = None       # Unix timestamp when quota was hit
        self._reset_at: float | None = None            # Unix timestamp when quota resets
        self._retry_after_seconds: float | None = None # From error message if provided
        self._last_error_detail: str | None = None     # Raw error detail for diagnostics

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def status(self) -> str:
        with self._lock:
            # Auto-recover if we've passed the reset time
            if self._status == QUOTA_STATUS_EXHAUSTED and self._reset_at is not None:
                if time.time() >= self._reset_at:
                    self._status = QUOTA_STATUS_ACTIVE
                    self._exhausted_at = None
                    self._reset_at = None
                    self._retry_after_seconds = None
                    self._last_error_detail = None
                    logger.info("Gemini quota has reset — resuming normal operation.")
            return self._status

    @property
    def exhausted_at(self) -> float | None:
        with self._lock:
            return self._exhausted_at

    @property
    def reset_at(self) -> float | None:
        with self._lock:
            if self._reset_at is not None:
                return self._reset_at
            return None

    @property
    def retry_after_seconds(self) -> float | None:
        with self._lock:
            return self._retry_after_seconds

    @property
    def last_error_detail(self) -> str | None:
        with self._lock:
            return self._last_error_detail

    def mark_exhausted(
        self,
        error_detail: str = "",
        retry_after: float | None = None,
    ) -> None:
        """
        Mark the quota as exhausted.

        Sets the reset time to the greater of:
        - Current time + retry_after (if provided via error headers)
        - Next midnight UTC (standard free-tier reset)
        """
        with self._lock:
            self._status = QUOTA_STATUS_EXHAUSTED
            self._exhausted_at = time.time()
            self._last_error_detail = error_detail

            now = time.time()
            midnight = _next_midnight_utc()

            if retry_after is not None and retry_after > 0:
                retry_ts = now + retry_after
                self._reset_at = min(retry_ts, midnight)
                self._retry_after_seconds = retry_after
            else:
                self._reset_at = midnight
                self._retry_after_seconds = midnight - now

            logger.warning(
                "Gemini quota exhausted. Reset at %s (in %.0f seconds).",
                datetime.fromtimestamp(self._reset_at, tz=timezone.utc).isoformat(),
                self._reset_at - now,
            )

    def is_exhausted(self) -> bool:
        """Check if quota is currently exhausted (auto-recovery check built in)."""
        return self.status == QUOTA_STATUS_EXHAUSTED

    def get_status_info(self) -> dict:
        """Return a dict with full quota status for the API response."""
        with self._lock:
            now = time.time()
            reset_ts = self._reset_at or _next_midnight_utc()
            return {
                "quota_status": self._status,
                "exhausted": self._status == QUOTA_STATUS_EXHAUSTED,
                "exhausted_at": self._exhausted_at,
                "reset_at": reset_ts,
                "reset_in_seconds": max(0.0, reset_ts - now),
                "retry_after_seconds": self._retry_after_seconds,
                "last_error_detail": self._last_error_detail,
            }


# Singleton
_quota_state = QuotaState()


def get_quota_state() -> QuotaState:
    """Return the application-wide quota state singleton."""
    return _quota_state