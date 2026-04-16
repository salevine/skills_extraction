"""
LLM skill extraction from job descriptions.

V2: whole-job extraction — one LLM call per job description.
V1 (deprecated): batched lines + optional candidates → structured mentions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .boilerplate import label_parsed_lines
from .candidate_mining import CandidateSpan
from .config import PipelineConfig
from .llm_backend import call_llm
from .llm_ollama import parse_json_loose
from .llm_vllm import call_vllm_direct, call_vllm_direct_with_failover
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

_LONG_DESCRIPTION_RETRY_MIN_CHARS = 6000
_LONG_DESCRIPTION_RETRY_MIN_LINES = 24
_EXTRACTOR_RETRY_MAX_CHUNK_CHARS = 5000
_EXTRACTOR_RETRY_MAX_CHUNK_LINES = 24
_EXTRACTOR_RETRY_MAX_SPLIT_DEPTH = 3


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


def _call_extractor_v2(
    cfg: PipelineConfig,
    model: str,
    description: str,
    job_key: str,
    endpoint: Optional[str] = None,
    all_endpoints: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    user = EXTRACTOR_V2_USER_TEMPLATE.format(description=description)

    if endpoint and all_endpoints:
        raw = call_vllm_direct_with_failover(
            cfg, endpoint, all_endpoints, model, EXTRACTOR_V2_SYSTEM, user, temperature=0.1,
        )
    elif endpoint:
        raw = call_vllm_direct(cfg, endpoint, model, EXTRACTOR_V2_SYSTEM, user, temperature=0.1)
    else:
        raw = call_llm(cfg, model, EXTRACTOR_V2_SYSTEM, user, temperature=0.1)

    try:
        data = parse_json_loose(raw)
    except Exception as e:
        raise RuntimeError(f"extractor_json_parse_failed: {job_key}: {e}") from e

    mentions = data.get("mentions") if isinstance(data, dict) else None
    if mentions is None and isinstance(data, list):
        mentions = data
    if not isinstance(mentions, list):
        raise RuntimeError(f"extractor_invalid_payload: {job_key}: missing mentions list")
    return mentions


def _normalize_v2_mentions(
    mentions: List[Dict[str, Any]],
    job_key: str,
    source_lines: List[ParsedLine],
    used_positions: Dict[str, List[Tuple[str, int]]],
) -> List[Dict[str, Any]]:
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

        anchor = _anchor_span_to_line(
            span, context, section_hint, source_lines, used_positions,
        )
        if anchor is None:
            logger.debug("Skipping ungroundable span %r in %s", span, job_key)
            continue

        pl, line_cs, line_ce = anchor
        glob_cs = pl.char_start + line_cs
        glob_ce = pl.char_start + line_ce

        requirement = str(m.get("requirement", "unclear")).lower()
        if requirement not in ("required", "optional", "unclear"):
            requirement = "unclear"
        hard_soft = str(m.get("hard_soft", "unknown")).lower()
        if hard_soft not in ("hard", "soft", "unknown"):
            hard_soft = "unknown"

        m["span_confidence"] = confidence
        m["evidence"] = evidence
        m["normalized_candidate"] = m.get("normalized_skill", span)
        m["is_skill"] = True
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


def _chunk_text_from_lines(normalized_text: str, chunk_lines: List[ParsedLine]) -> str:
    if not chunk_lines:
        return ""
    start = chunk_lines[0].char_start
    end = chunk_lines[-1].char_end
    return normalized_text[start:end]


def _chunk_lines_for_retry(lines: List[ParsedLine]) -> List[List[ParsedLine]]:
    chunks: List[List[ParsedLine]] = []
    current: List[ParsedLine] = []
    current_chars = 0

    for pl in lines:
        line_chars = max(1, pl.char_end - pl.char_start)
        next_chars = current_chars + line_chars + (1 if current else 0)
        start_new = bool(current) and (
            next_chars > _EXTRACTOR_RETRY_MAX_CHUNK_CHARS
            or len(current) >= _EXTRACTOR_RETRY_MAX_CHUNK_LINES
            or (
                pl.section.lower() != current[-1].section.lower()
                and current_chars >= (_EXTRACTOR_RETRY_MAX_CHUNK_CHARS // 2)
            )
        )
        if start_new:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(pl)
        current_chars += line_chars + (1 if current_chars else 0)

    if current:
        chunks.append(current)
    return chunks


def _is_retryable_extractor_error(err: Exception) -> bool:
    msg = str(err)
    return msg.startswith("extractor_json_parse_failed") or msg.startswith("extractor_invalid_payload")


def _split_text_for_retry(text: str) -> Optional[Tuple[str, str]]:
    if len(text) < 2:
        return None

    midpoint = len(text) // 2
    boundary_match = None
    boundary_distance = None
    for m in re.finditer(r"[.!?;\n]+\s+", text):
        pos = m.end()
        dist = abs(pos - midpoint)
        if boundary_distance is None or dist < boundary_distance:
            boundary_match = pos
            boundary_distance = dist

    split_at = boundary_match
    if split_at is None:
        window = 400
        lo = max(1, midpoint - window)
        hi = min(len(text) - 1, midpoint + window)
        left = text.rfind(" ", lo, midpoint)
        right = text.find(" ", midpoint, hi)
        candidates = [p for p in (left, right) if p not in (-1, 0)]
        if candidates:
            split_at = min(candidates, key=lambda p: abs(p - midpoint))

    if split_at is None:
        split_at = midpoint

    left_text = text[:split_at].strip()
    right_text = text[split_at:].strip()
    if not left_text or not right_text:
        return None
    return left_text, right_text


def _extract_mentions_recursive(
    cfg: PipelineConfig,
    model: str,
    description: str,
    job_key: str,
    source_lines: List[ParsedLine],
    used_positions: Dict[str, List[Tuple[str, int]]],
    endpoint: Optional[str] = None,
    all_endpoints: Optional[List[str]] = None,
    depth: int = 0,
) -> List[Dict[str, Any]]:
    try:
        mentions = _call_extractor_v2(
            cfg, model, description, job_key, endpoint=endpoint, all_endpoints=all_endpoints,
        )
    except RuntimeError as e:
        if (
            not _is_retryable_extractor_error(e)
            or depth >= _EXTRACTOR_RETRY_MAX_SPLIT_DEPTH
            or len(description) < (_EXTRACTOR_RETRY_MAX_CHUNK_CHARS // 2)
        ):
            raise
        split = _split_text_for_retry(description)
        if split is None:
            raise
        logger.debug(
            "Retrying extractor chunk for %s at depth %d after %s",
            job_key,
            depth + 1,
            e,
        )
        recovered: List[Dict[str, Any]] = []
        for chunk_text in split:
            recovered.extend(
                _extract_mentions_recursive(
                    cfg,
                    model,
                    chunk_text,
                    job_key,
                    source_lines,
                    used_positions,
                    endpoint=endpoint,
                    all_endpoints=all_endpoints,
                    depth=depth + 1,
                )
            )
        return recovered
    return _normalize_v2_mentions(mentions, job_key, source_lines, used_positions)


def _extract_mentions_chunked(
    cfg: PipelineConfig,
    model: str,
    normalized_text: str,
    job_key: str,
    source_lines: List[ParsedLine],
    used_positions: Dict[str, List[Tuple[str, int]]],
    endpoint: Optional[str] = None,
    all_endpoints: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    recovered: List[Dict[str, Any]] = []
    failures: List[str] = []
    chunks = _chunk_lines_for_retry(source_lines)

    for chunk_idx, chunk_lines in enumerate(chunks):
        chunk_text = _chunk_text_from_lines(normalized_text, chunk_lines)
        if not chunk_text.strip():
            continue
        try:
            recovered.extend(
                _extract_mentions_recursive(
                    cfg,
                    model,
                    chunk_text,
                    job_key,
                    source_lines,
                    used_positions,
                    endpoint=endpoint,
                    all_endpoints=all_endpoints,
                    depth=0,
                )
            )
        except RuntimeError as e:
            failures.append(str(e))
            logger.warning(
                "Chunked extractor retry failed for %s chunk %d/%d: %s",
                job_key,
                chunk_idx + 1,
                len(chunks),
                e,
            )

    if recovered:
        if failures:
            logger.warning(
                "Recovered partial extractor output for %s after chunk retries (%d failed chunk(s))",
                job_key,
                len(failures),
            )
        return recovered
    if failures:
        raise RuntimeError(failures[0])
    return []


def extract_mentions_for_job(
    cfg: PipelineConfig,
    job: Dict[str, Any],
    job_key: str,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    all_endpoints: Optional[List[str]] = None,
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
        endpoint: If provided, preferred vLLM endpoint.
        all_endpoints: If provided with endpoint, enables failover to other nodes.
    """
    model = model or cfg.extractor_model
    _title, desc_raw = extract_description_fields(job)
    if not desc_raw or not desc_raw.strip():
        return []

    pre = preprocess_description(desc_raw)
    normalized_text = split_inline_section_headers(pre.description_normalized)
    source_lines = segment_lines(job_key, normalized_text)
    label_parsed_lines(source_lines)
    used_positions: Dict[str, List[Tuple[str, int]]] = {}

    try:
        mentions = _call_extractor_v2(
            cfg, model, desc_raw, job_key, endpoint=endpoint, all_endpoints=all_endpoints,
        )
        return _normalize_v2_mentions(mentions, job_key, source_lines, used_positions)
    except RuntimeError as e:
        should_retry_chunked = (
            _is_retryable_extractor_error(e)
            and (
                len(desc_raw) >= _LONG_DESCRIPTION_RETRY_MIN_CHARS
                or len(source_lines) >= _LONG_DESCRIPTION_RETRY_MIN_LINES
            )
        )
        if not should_retry_chunked:
            raise
        logger.warning(
            "Extractor failed for long description %s; retrying with smaller chunks: %s",
            job_key,
            e,
        )
        return _extract_mentions_chunked(
            cfg,
            model,
            normalized_text,
            job_key,
            source_lines,
            used_positions,
            endpoint=endpoint,
            all_endpoints=all_endpoints,
        )


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
