"""
LLM span validation: batched lines + optional candidates → structured mentions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .candidate_mining import CandidateSpan
from .config import PipelineConfig
from .llm_ollama import call_ollama, parse_json_loose
from .prompts import EXTRACTOR_SYSTEM, EXTRACTOR_USER_TEMPLATE
from .schemas import ParsedLine

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


def extract_mentions_for_batch(
    cfg: PipelineConfig,
    batch_lines: List[ParsedLine],
    candidates: List[CandidateSpan],
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
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

    raw = call_ollama(cfg, model, EXTRACTOR_SYSTEM, user, temperature=0.1)
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
        m["_glob_char_start"] = glob_cs
        m["_glob_char_end"] = glob_ce
        m["_line_char_start"] = cs
        m["_line_char_end"] = ce
        m["_offset_valid"] = offset_valid
        m["_parsed_line"] = pl
        normalized.append(m)
    return normalized


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
