"""
app/core/gemini_middleware.py
─────────────────────────────
Gemini API middleware that intercepts all calls to detect quota exhaustion (HTTP 429).

Replaces direct ``google.genai.Client`` usage across the application with
a wrapper that:

1. Checks quota state **before** making the call (short-circuits if exhausted).
2. Wraps the actual API call and intercepts 429 / RESOURCE_EXHAUSTED responses.
3. **Rate-limit (transient) 429s** → raises ``RateLimitExceededError``
   (retryable; does NOT set the global quota-exhausted flag).
4. **Daily-quota (terminal) 429s** → raises ``QuotaExhaustedError``
   (marks the global exhausted flag; frontend shows countdown banner).
5. On repeated 429s after retries, escalates to daily-quota semantics.
"""

from __future__ import annotations

import logging
import re
import time
from functools import wraps
from typing import Any, Callable

from app.core.exceptions import QuotaExhaustedError, RateLimitExceededError
from app.core.quota_manager import get_quota_state

logger = logging.getLogger(__name__)

# ── Quota error patterns ───────────────────────────────────────────────────────

# Gemini returns 429 with the string "RESOURCE_EXHAUSTED" in the error body
_QUOTA_PATTERNS = [
    re.compile(r"429", re.IGNORECASE),
    re.compile(r"RESOURCE_EXHAUSTED", re.IGNORECASE),
    re.compile(r"quota exceeded", re.IGNORECASE),
    re.compile(r"rate.limit", re.IGNORECASE),
    re.compile(r"you exceeded your current quota", re.IGNORECASE),
    re.compile(r"quota.*free.tier", re.IGNORECASE),
    re.compile(r"RetryInfo.*retryDelay", re.IGNORECASE),
]

# Pattern to extract retry delay in seconds from error details
_RETRY_DELAY_PATTERN = re.compile(
    r"(?:retry_delay|retryDelay|retry in)\s*[\:\=]\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# If the retry delay is >= 60 seconds, this is likely a daily-quota exhaustion
# rather than a transient rate-limit.
_DAILY_QUOTA_RETRY_THRESHOLD_S = 60.0


# ── Quota check wrapper ────────────────────────────────────────────────────────

def check_quota_before_call(func: Callable) -> Callable:
    """
    Decorator that checks quota before allowing a Gemini call to proceed.

    Use on any function that makes a direct Gemini API call. If quota is
    exhausted, raises QuotaExhaustedError immediately without making the call.

    Usage::

        @check_quota_before_call
        def my_gemini_function(...):
            ...

    This decorator is automatically applied by ``call_gemini`` below.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        qs = get_quota_state()
        if qs.is_exhausted():
            info = qs.get_status_info()
            reset_in = info.get("reset_in_seconds", 0)
            raise QuotaExhaustedError(
                retry_after_seconds=reset_in,
                detail=(
                    "You have reached your daily Gemini API quota. "
                    "Please come back later. Your quota will reset "
                    f"in approximately {_format_duration(reset_in)}."
                ),
            )
        return func(*args, **kwargs)

    return wrapper


# ── Wrapped Gemini client call ─────────────────────────────────────────────────

def call_gemini(
    api_call: Callable[[], Any],
    service_name: str = "unknown",
) -> Any:
    """
    Execute a Gemini API call with full quota interception.

    Parameters
    ----------
    api_call : Callable that performs the actual Gemini API call.
    service_name : Human-readable label for logging (e.g. 'embed', 'generate').

    Returns
    -------
    The raw response from the API call.

    Raises
    ------
    RateLimitExceededError
        Transient 429 (RPM / short-lived rate limit).  **Does NOT** set the
        global quota-exhausted flag.  The caller should sleep and retry.

    QuotaExhaustedError
        Daily-quota exhaustion (terminal).  **Sets** the global quota-exhausted
        flag; the frontend will show the persistent countdown banner.
    """
    qs = get_quota_state()

    # ── Pre-call quota check ──────────────────────────────────────────────
    if qs.is_exhausted():
        info = qs.get_status_info()
        reset_in = info.get("reset_in_seconds", 0)
        raise QuotaExhaustedError(
            retry_after_seconds=reset_in,
            detail=(
                "You have reached your daily Gemini API quota. "
                "Please come back later. Your quota will reset "
                f"in approximately {_format_duration(reset_in)}."
            ),
        )

    # ── Execute the call ──────────────────────────────────────────────────
    try:
        result = api_call()
        return result

    except Exception as exc:
        error_str = str(exc)

        # ── Detect quota / rate-limit errors ──────────────────────────────
        if _is_quota_error(error_str):
            retry_after = _extract_retry_delay(error_str)

            # Transient rate-limit (short delay) — do NOT mark exhausted
            if _is_transient_rate_limit(error_str, retry_after):
                delay = retry_after if retry_after is not None else 6.0
                logger.info(
                    "Rate-limit transient on '%s': retry delay %.1f s. "
                    "Raising RateLimitExceededError (backoff will retry).",
                    service_name,
                    delay,
                )
                raise RateLimitExceededError(
                    retry_after_seconds=delay,
                    detail=(
                        "Gemini API rate limit hit. "
                        "Pausing and retrying in %.0f seconds…" % delay
                    ),
                ) from exc

            # Daily-quota exhaustion (long delay or explicit quota message)
            # Mark exhausted so further requests are blocked pre-call.
            qs.mark_exhausted(error_detail=error_str, retry_after=retry_after)
            reset_info = qs.get_status_info()
            logger.warning(
                "Daily quota exhausted on '%s'. Reset in %.0f s.",
                service_name,
                reset_info.get("reset_in_seconds", 0),
            )
            raise QuotaExhaustedError(
                retry_after_seconds=reset_info.get("reset_in_seconds", 0),
                detail=(
                    "You have reached your daily Gemini API quota. "
                    "Please come back later. Your quota will reset "
                    f"in approximately {_format_duration(reset_info.get('reset_in_seconds', 0))}."
                ),
            ) from exc

        # ── Other errors — re-raise as-is ─────────────────────────────────
        raise


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_quota_error(error_str: str) -> bool:
    """Check if an error string indicates a quota exhaustion."""
    for pattern in _QUOTA_PATTERNS:
        if pattern.search(error_str):
            return True
    return False


def _is_transient_rate_limit(
    error_str: str,
    retry_after: float | None,
) -> bool:
    """
    Determine whether a Gemini error is a *transient* rate-limit (retryable)
    versus a true daily-quota exhaustion.

    Heuristics
    ----------
    1. If ``retry_after`` is < 60 s → transient.
    2. If the error mentions ``rate.limit`` or ``RPM`` → transient.
    3. If the error mentions ``quota`` AND ``free.tier`` → daily-quota.
    4. If ``retry_after`` is None or >= 60 s → daily-quota.
    """
    lower = error_str.lower()

    # Explicit free-tier quota message → daily quota
    if "quota" in lower and "free" in lower:
        return False

    # Explicit rate-limit (not quota) → transient
    if "rate limit" in lower or "rate_limit" in lower or "rpm" in lower:
        return True

    # Use the retry-after value as the primary heuristic
    if retry_after is not None:
        return retry_after < _DAILY_QUOTA_RETRY_THRESHOLD_S

    # No retry-after info → conservative: treat as daily-quota
    return False


def _extract_retry_delay(error_str: str) -> float | None:
    """
    Extract the retry delay in seconds from a Gemini error message.

    Example matching text: "Please retry in 28.569324398s." or
    "retryDelay: '28s'" or similar.
    """
    match = _RETRY_DELAY_PATTERN.search(error_str)
    if match:
        return float(match.group(1))

    # Also try to find ISO duration like "28s" at end of messages
    duration_match = re.search(
        r"(\d+(?:\.\d+)?)\s*s(?:econds?)?", error_str, re.IGNORECASE
    )
    if duration_match:
        return float(duration_match.group(1))

    return None


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    if seconds <= 0:
        return "a few moments"
    if seconds < 60:
        return f"{int(seconds)} seconds"
    if seconds < 3600:
        return f"{int(seconds // 60)} minutes"
    if seconds < 86400:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m" if mins else f"{hours} hours"
    days = int(seconds // 86400)
    return f"{days} day(s)"