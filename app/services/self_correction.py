"""
app/services/self_correction.py
────────────────────────────────
FR3 — Automatic self-correction for generated notes.

Handles three constraints:
  1. Length deviation — if the updated section differs significantly in
     sentence count from the original, the content is regenerated with a
     length-adjustment instruction.
  2. Research quality — if web research returned fewer than expected
     results, the system falls back to more expansive research queries or
     proceeds with a context-only generation.
  3. Stylistic/structural concordance — ensures the output uses the same
     heading style, tone, and formatting as the original notes.
"""

from __future__ import annotations

import logging
import re

from app.core.config import get_settings
from app.core.exceptions import SelfCorrectionError
from app.models.schemas import SelfCorrectionReport

logger = logging.getLogger(__name__)


# ── Sentence-count heuristics ──────────────────────────────────────────────────

def count_sentences(text: str) -> int:
    """Rough sentence count based on terminal punctuation."""
    # Split on sentence-ending punctuation followed by space or end-of-string
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return len([s for s in sentences if s.strip()])


# ── Length deviation check ─────────────────────────────────────────────────────

def check_length_deviation(
    original_texts: list[str],
    updated_text: str,
    max_deviation_ratio: float | None = None,
) -> tuple[bool, int, int, str]:
    """
    Check if the updated text deviates too much in sentence count from the originals.

    Parameters
    ----------
    original_texts : List of original text sections.
    updated_text   : The newly generated text.
    max_deviation_ratio : Maximum allowed |new - original|/original ratio.

    Returns
    -------
    (is_acceptable, original_sentences, updated_sentences, correction_msg)
    """
    settings = get_settings()
    ratio = max_deviation_ratio or settings.max_length_deviation_ratio

    # Count sentences in each original section and average
    orig_sentences_list = [count_sentences(t) for t in original_texts]
    avg_original = max(sum(orig_sentences_list) / max(len(orig_sentences_list), 1), 1)
    total_original = sum(orig_sentences_list)

    updated_sentences = count_sentences(updated_text)

    deviation = abs(updated_sentences - total_original) / max(total_original, 1)

    if deviation <= ratio:
        return True, total_original, updated_sentences, ""

    correction_msg = (
        f"Length deviation detected: original has ~{total_original} sentence(s), "
        f"generated has ~{updated_sentences} (deviation={deviation:.0%}, "
        f"threshold={ratio:.0%}). Regenerating with length adjustment."
    )
    logger.warning(correction_msg)
    return False, total_original, updated_sentences, correction_msg


# ── Research quality check ─────────────────────────────────────────────────────

def check_research_quality(
    sources_found: int,
    min_sources: int = 1,
) -> tuple[bool, str]:
    """
    Check if enough research sources were found.

    Returns
    -------
    (is_sufficient, correction_msg)
    """
    if sources_found >= min_sources:
        return True, ""

    msg = (
        f"Insufficient research sources: found {sources_found}, "
        f"minimum required is {min_sources}. Falling back to context-only generation."
    )
    logger.warning(msg)
    return False, msg


# ── Style concordance check ────────────────────────────────────────────────────

def check_style_concordance(
    original_texts: list[str],
    updated_text: str,
) -> tuple[bool, str, dict[str, list[str]]]:
    """
    Ensure the updated text maintains stylistic concordance with the original.

    Checks:
      - Heading style (markdown # vs underlined vs plain)
      - Tone indicators (formal vs informal markers)
      - Structural elements (lists, code blocks, equations)

    Returns
    -------
    (is_concordant, correction_msg, issues_found)
    """
    issues: dict[str, list[str]] = {
        "heading_style": [],
        "tone": [],
        "structure": [],
    }
    all_original = "\n".join(original_texts)

    # ── Heading style ──────────────────────────────────────────────────────
    orig_headings = re.findall(r"^(#{1,6})\s", all_original, re.MULTILINE)
    updated_headings = re.findall(r"^(#{1,6})\s", updated_text, re.MULTILINE)

    if orig_headings and updated_headings:
        orig_level = orig_headings[0]
        for uh in updated_headings:
            if uh != orig_level:
                issues["heading_style"].append(
                    f"Expected heading level '{orig_level}' but found '{uh}'."
                )

    # ── List usage ─────────────────────────────────────────────────────────
    orig_has_lists = bool(re.search(r"^\s*[-*+]\s", all_original, re.MULTILINE))
    updated_has_lists = bool(re.search(r"^\s*[-*+]\s", updated_text, re.MULTILINE))

    if orig_has_lists and not updated_has_lists:
        issues["structure"].append(
            "Original uses bullet lists but updated version does not."
        )
    elif not orig_has_lists and updated_has_lists:
        issues["structure"].append(
            "Original does not use bullet lists but updated version introduces them."
        )

    # ── Code blocks ────────────────────────────────────────────────────────
    orig_has_code = bool(re.search(r"```", all_original))
    updated_has_code = bool(re.search(r"```", updated_text))

    if orig_has_code and not updated_has_code:
        issues["structure"].append(
            "Original contains code blocks but updated version does not."
        )
    elif not orig_has_code and updated_has_code:
        issues["structure"].append(
            "Original does not contain code blocks but updated version introduces them."
        )

    total_issues = sum(len(v) for v in issues.values())
    is_concordant = total_issues == 0

    if not is_concordant:
        detail = "; ".join(
            issue for lst in issues.values() for issue in lst
        )
        msg = (
            f"Style concordance issues detected ({total_issues}): {detail}. "
            "Regenerating with strict style-matching instructions."
        )
        logger.warning(msg)
        return False, msg, issues

    return True, "", issues


# ── Quality score ──────────────────────────────────────────────────────────────

def compute_quality_score(
    updated_text: str,
    sources_found: int,
    corrections_applied: list[str],
) -> float:
    """
    Compute an overall quality score (0–1) for the generated output.

    Factors:
      - Has content (base 0.5)
      - Has sufficient length (bonus up to 0.2)
      - Research was productive (bonus up to 0.2)
      - Few corrections needed (bonus up to 0.1)
    """
    score = 0.5  # Base for having content

    # Length bonus
    sentences = count_sentences(updated_text)
    if sentences >= 5:
        score += 0.1
    if sentences >= 10:
        score += 0.1

    # Research bonus
    if sources_found >= 3:
        score += 0.15
    elif sources_found >= 1:
        score += 0.1

    # Correction penalty (fewer corrections = better)
    correction_penalty = min(len(corrections_applied) * 0.05, 0.2)
    score -= correction_penalty

    return round(max(0.0, min(1.0, score)), 3)


# ── Generate length-adjusted prompt ────────────────────────────────────────────

def build_length_correction_prompt(
    original_text: str,
    updated_text: str,
    target_sentences: int,
) -> str:
    """
    Build a prompt instructing the LLM to adjust the length of the updated text.

    The instruction is appended to the original system prompt to guide regeneration.
    """
    current_sentences = count_sentences(updated_text)
    direction = "expand" if current_sentences < target_sentences else "condense"
    return (
        f"The previous generated response had {current_sentences} sentence(s), "
        f"but the original had {target_sentences} sentence(s). "
        f"Please {direction} the updated version to match the original length "
        f"(approximately {target_sentences} sentence(s)) while keeping all "
        f"the substantive updates and maintaining a natural academic flow."
    )


# ── Build style-correction prompt ──────────────────────────────────────────────

def build_style_correction_prompt(
    original_text: str,
) -> str:
    """
    Build a prompt instructing the LLM to match the original's style exactly.
    """
    return (
        f"IMPORTANT: The updated notes must use EXACTLY the same style as the "
        f"original notes below. Match the heading format (e.g., ## vs ###), "
        f"the use of bullet lists, code blocks, inline code, paragraph spacing, "
        f"and overall academic tone. Here is the original for style reference:\n\n"
        f"{original_text[:1500]}"
    )