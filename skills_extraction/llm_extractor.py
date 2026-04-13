"""
LLM skill extraction from job descriptions.

V2: whole-job extraction — one LLM call per job description.
V1 (deprecated): batched lines + optional candidates → structured mentions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .boilerplate import label_parsed_lines
from .candidate_mining import CandidateSpan
from .config import PipelineConfig
from .llm_backend import call_llm
from .llm_ollama import parse_json_loose
from .llm_vllm import call_vllm_direct
from .preprocessing import extract_description_fields, preprocess_description
from .prompts import (
    EXTRACTOR_SYSTEM,
    EXTRACTOR_USER_TEMPLATE,
    EXTRACTOR_V2_SYSTEM,
    EXTRACTOR_V2_USER_TEMPLATE,
)
from .schemas import ParsedLine
from .sectioning import segment_lines, split_inline_section_headers

logger = logging.getLogger(__name__)


def _line_by_id(lines: List[ParsedLine]) -> Dict[str, ParsedLine]:
    return {pl.line_id: pl for pl in lines}


def _all_substring_starts(haystack: str, needle: str) -> List[int]:
    if not needle:
        return []
    out: List[int] = []
    start = 0
    while True:
        j = haystack.find(needle, start)
        if j < 0:
            break
        out.append(j)
        start = j + 1
    return out


def _repair_span_offsets(
    line_text: str,
    span: str,
    proposed_cs: Optional[int],
    proposed_ce: Optional[int],
    evidence: str,
    pl_char_start: int,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Prefer model offsets when valid; else choose substring match nearest proposed start;
    else align via evidence substring; else first match.
    Returns (line_cs, line_ce, glob_cs, glob_ce).
    """
    if not span:
        return None, None, None, None

    if (
        isinstance(proposed_cs, int)
        and isinstance(proposed_ce, int)
        and 0 <= proposed_cs < proposed_ce <= len(line_text)
        and line_text[proposed_cs:proposed_ce] == span
    ):
        cs, ce = proposed_cs, proposed_ce
        return cs, ce, pl_char_start + cs, pl_char_start + ce

    positions = _all_substring_starts(line_text, span)
    if not positions:
        return None, None, None, None

    chosen = positions[0]
    if isinstance(proposed_cs, int) and proposed_cs >= 0:
        chosen = min(positions, key=lambda j: abs(j - proposed_cs))
    elif evidence and evidence.strip():
        ev_positions = _all_substring_starts(line_text, evidence.strip())
        if ev_positions:
            # Prefer span occurrence closest to any evidence anchor
            anchor = ev_positions[0]
            chosen = min(positions, key=lambda j: abs(j - anchor))

    cs, ce = chosen, chosen + len(span)
    return cs, ce, pl_char_start + cs, pl_char_start + ce


# ---------------------------------------------------------------------------
# V2: Whole-job extraction — one LLM call per job
# ---------------------------------------------------------------------------

def _build_source_lines(job_key: str, desc_raw: str) -> List[ParsedLine]:
    """Preprocess and segment a raw description into real ParsedLines."""
    pre = preprocess_description(desc_raw)
    norm = split_inline_section_headers(pre.description_normalized)
    lines = segment_lines(job_key, norm)
    label_parsed_lines(lines)
    return lines


def _anchor_span_to_line(
    span: str,
    context: str,
    section_hint: str,
    source_lines: List[ParsedLine],
    used_positions: Dict[str, List[Tuple[str, int]]],
) -> Optional[Tuple[ParsedLine, int, int]]:
    """Find the best real ParsedLine containing *span*.

    Strategy:
    1. If *context* is a true substring of a source line, prefer exact context match.
    2. Otherwise find all lines containing the span and pick the best one using
       section hint + deterministic position tracking to avoid repeated-span bugs.
    3. Returns None if no line contains the span.

    *used_positions* maps span -> list of already-used (line_id, pos) tuples
    so repeated occurrences bind to distinct source positions.
    """
    span_lower = span.lower()
    used_set = set(tuple(x) for x in used_positions.get(span, []))

    # Pass 1: exact context match (context is a true substring of line text)
    if context and context.strip():
        for pl in source_lines:
            if context in pl.text:
                pos = pl.text.find(span)
                if pos >= 0 and (pl.line_id, pos) not in used_set:
                    used_positions.setdefault(span, []).append((pl.line_id, pos))
                    return pl, pos, pos + len(span)

    # Pass 2: collect all lines containing the span
    candidates: List[Tuple[ParsedLine, int]] = []
    for pl in source_lines:
        starts = _all_substring_starts(pl.text, span)
        if not starts:
            # Case-insensitive fallback
            starts_lower = _all_substring_starts(pl.text.lower(), span_lower)
            if starts_lower:
                starts = starts_lower
        for s in starts:
            candidates.append((pl, s))

    if not candidates:
        return None

    # Sort deterministically: prefer section match, then by document order
    def _score(item: Tuple[ParsedLine, int]) -> Tuple[int, int, int]:
        pl, pos = item
        sec_match = 0 if pl.section.lower() == section_hint.lower() else 1
        return (sec_match, pl.line_index, pos)

    candidates.sort(key=_score)

    # Pick first candidate not already used for this span
    for pl, pos in candidates:
        if (pl.line_id, pos) not in used_set:
            used_positions.setdefault(span, []).append((pl.line_id, pos))
            return pl, pos, pos + len(span)

    # All positions used — fall back to first candidate (allow reuse)
    pl, pos = candidates[0]
    return pl, pos, pos + len(span)


def extract_mentions_for_job(
    cfg: PipelineConfig,
    job: Dict[str, Any],
    job_key: str,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract skill mentions from a full job description in a single LLM call.

    Returns a list of mention dicts, each with a ``_parsed_line`` (a real
    ParsedLine from the source text) so that stages 2-4 can consume them
    with accurate section, boilerplate_label, and char offsets.

    Args:
        cfg: Pipeline configuration.
        job: Raw job record (must contain a description field).
        job_key: Stable key for this job.
        model: Override model name (defaults to cfg.extractor_model).
        endpoint: If provided, call this vLLM endpoint directly.
    """
    model = model or cfg.extractor_model
    _title, desc_raw = extract_description_fields(job)
    if not desc_raw or not desc_raw.strip():
        return []

    user = EXTRACTOR_V2_USER_TEMPLATE.format(description=desc_raw)

    if endpoint:
        raw = call_vllm_direct(cfg, endpoint, model, EXTRACTOR_V2_SYSTEM, user, temperature=0.1)
    else:
        raw = call_llm(cfg, model, EXTRACTOR_V2_SYSTEM, user, temperature=0.1)

    try:
        data = parse_json_loose(raw)
    except Exception as e:
        logger.error("Extractor JSON parse failed for %s: %s", job_key, e)
        return []

    mentions = data.get("mentions") if isinstance(data, dict) else None
    if mentions is None and isinstance(data, list):
        mentions = data
    if not isinstance(mentions, list):
        return []

    # Build real parsed lines from the source text
    source_lines = _build_source_lines(job_key, desc_raw)
    # Track used positions per span to handle repeated spans deterministically
    used_positions: Dict[str, List[Tuple[str, int]]] = {}

    normalized: List[Dict[str, Any]] = []
    for m_idx, m in enumerate(mentions):
        if not isinstance(m, dict):
            continue
        span = m.get("skill_span") or ""
        if not span.strip():
            continue

        context = m.get("context") or ""
        section_hint = m.get("section") or "General"
        evidence = str(m.get("evidence", span) or "")
        confidence = m.get("confidence", m.get("span_confidence", 0.7))
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.7

        # Anchor to a real source line — skip if ungroundable
        anchor = _anchor_span_to_line(
            span, context, section_hint, source_lines, used_positions,
        )
        if anchor is None:
            logger.debug("Skipping ungroundable span %r in %s", span, job_key)
            continue

        pl, line_cs, line_ce = anchor
        glob_cs = pl.char_start + line_cs
        glob_ce = pl.char_start + line_ce

        # Classification fields from combined prompt
        requirement = str(m.get("requirement", "unclear")).lower()
        if requirement not in ("required", "optional", "unclear"):
            requirement = "unclear"
        hard_soft = str(m.get("hard_soft", "unknown")).lower()
        if hard_soft not in ("hard", "soft", "unknown"):
            hard_soft = "unknown"

        m["span_confidence"] = confidence
        m["evidence"] = evidence
        m["normalized_candidate"] = m.get("normalized_skill", span)
        m["is_skill"] = True  # if it's in the output, the model considers it a skill
        m["requirement_level"] = requirement
        m["hard_soft"] = hard_soft
        m["_glob_char_start"] = glob_cs
        m["_glob_char_end"] = glob_ce
        m["_line_char_start"] = line_cs
        m["_line_char_end"] = line_ce
        m["_offset_valid"] = pl.text[line_cs:line_ce] == span
        m["_parsed_line"] = pl
        normalized.append(m)

    return normalized


# ---------------------------------------------------------------------------
# V1: Batched extraction (DEPRECATED)
# ---------------------------------------------------------------------------

# extract_mentions_for_batch() — Part of the old batching approach (DEPRECATED).
# Receives a subset of a job's lines (typically 5) plus candidates from stage 0.
# Being replaced by a whole-job extraction function that sends the full description.
def extract_mentions_for_batch(
    cfg: PipelineConfig,
    batch_lines: List[ParsedLine],
    candidates: List[CandidateSpan],
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract mentions from a batch of lines. If endpoint is provided, call
    that vLLM endpoint directly (bypassing the shared endpoint pool)."""
    model = model or cfg.extractor_model
    cmap: Dict[str, List[CandidateSpan]] = {}
    for c in candidates:
        cmap.setdefault(c.line_id, []).append(c)

    lines_payload = []
    for pl in batch_lines:
        lines_payload.append(
            {
                "line_id": pl.line_id,
                "section_label": pl.section,
                "line_text": pl.text,
                "boilerplate_label": pl.boilerplate_label,
            }
        )
    cand_payload = []
    for pl in batch_lines:
        for c in cmap.get(pl.line_id, []):
            cand_payload.append(
                {
                    "line_id": c.line_id,
                    "candidate_text": c.candidate_text,
                    "rule_source": c.rule_source,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                }
            )

    user = EXTRACTOR_USER_TEMPLATE.format(
        candidates_json=json.dumps(cand_payload, ensure_ascii=False),
        lines_json=json.dumps(lines_payload, ensure_ascii=False),
    )

    if endpoint:
        raw = call_vllm_direct(cfg, endpoint, model, EXTRACTOR_SYSTEM, user, temperature=0.1)
    else:
        raw = call_llm(cfg, model, EXTRACTOR_SYSTEM, user, temperature=0.1)
    try:
        data = parse_json_loose(raw)
    except Exception as e:
        logger.error("Extractor JSON parse failed: %s", e)
        return []

    mentions = data.get("mentions") if isinstance(data, dict) else None
    if mentions is None and isinstance(data, list):
        mentions = data
    if not isinstance(mentions, list):
        return []

    by_id = _line_by_id(batch_lines)
    normalized: List[Dict[str, Any]] = []
    for m in mentions:
        if not isinstance(m, dict):
            continue
        lid = m.get("line_id")
        pl = by_id.get(lid)
        if not pl:
            continue
        line_text = pl.text
        span = m.get("skill_span") or ""
        cs_raw = m.get("char_start")
        ce_raw = m.get("char_end")
        proposed_cs = cs_raw if isinstance(cs_raw, int) else None
        proposed_ce = ce_raw if isinstance(ce_raw, int) else None
        evidence = str(m.get("evidence", span) or "")

        cs, ce, glob_cs, glob_ce = _repair_span_offsets(
            line_text, span, proposed_cs, proposed_ce, evidence, pl.char_start
        )
        if cs is None or ce is None:
            continue

        offset_valid = bool(span) and line_text[cs:ce] == span
        if "span_confidence" not in m and "confidence" in m:
            m["span_confidence"] = m.get("confidence")
        m["_glob_char_start"] = glob_cs
        m["_glob_char_end"] = glob_ce
        m["_line_char_start"] = cs
        m["_line_char_end"] = ce
        m["_offset_valid"] = offset_valid
        m["_parsed_line"] = pl
        normalized.append(m)
    return normalized


# batch_lines() — DEPRECATED
# Part of the old stage 0 → stage 1 batching approach that split each job's
# lines into groups of N (default 5) for separate LLM calls. This created
# ~4.5x more LLM calls than necessary (45K for 10K jobs), increased timeout
# failures, and lost cross-line context. Being replaced by sending full job
# descriptions in a single LLM call.
def batch_lines(lines: List[ParsedLine], max_per_batch: int) -> List[List[ParsedLine]]:
    out = []
    cur = []
    for pl in lines:
        if not pl.text.strip():
            continue
        cur.append(pl)
        if len(cur) >= max_per_batch:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out
