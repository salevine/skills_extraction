"""
Document quality / information quality scoring (multi-signal, not a single length threshold).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .schemas import QualityAssessment, QualityStatus


def _count_skillish_lines(lines: List[str]) -> int:
    pat = re.compile(
        r"\b(experience|proficient|proficiency|expertise|knowledge|familiar|skill|ability|"
        r"using|with|hands-on|background|exposure|must\s+have|preferred|years?)\b",
        re.I,
    )
    return sum(1 for L in lines if pat.search(L))


def _section_header_lines(lines: List[str]) -> int:
    c = 0
    for L in lines:
        s = L.strip()
        if 0 < len(s) < 80 and (s.endswith(":") or re.match(r"^[A-Z][A-Za-z\s/&]{2,50}$", s)):
            c += 1
    return c


def _duplicate_ratio(text: str) -> float:
    lines = [L.strip() for L in text.split("\n") if L.strip()]
    if len(lines) < 2:
        return 0.0
    uniq = len(set(lines))
    return 1.0 - (uniq / max(1, len(lines)))


def _boilerplate_ratio(lines: List[str]) -> float:
    from .boilerplate import classify_line_boilerplate
    from .schemas import ParsedLine

    if not lines:
        return 0.0
    bad = 0
    for i, t in enumerate(lines):
        pl = ParsedLine(
            line_id=f"_L{i}",
            section="body",
            text=t,
            char_start=0,
            char_end=len(t),
            boilerplate_label="uncertain",
            line_index=i,
        )
        lab = classify_line_boilerplate(pl)
        if lab in ("likely_boilerplate", "likely_benefits", "likely_legal", "likely_marketing"):
            bad += 1
    return bad / max(1, len(lines))


def assess_quality(
    description_raw: str,
    description_normalized: str,
    parsed_line_texts: List[str],
    complete_min_score: float = 0.45,
) -> QualityAssessment:
    reasons: List[str] = []
    features: Dict[str, Any] = {}

    nchar = len(description_normalized or "")
    nlines = len([L for L in parsed_line_texts if L.strip()])
    features["char_length"] = nchar
    features["non_empty_line_count"] = nlines
    features["section_header_like_lines"] = _section_header_lines(parsed_line_texts)
    features["skillish_line_count"] = _count_skillish_lines(parsed_line_texts)
    features["duplicate_line_ratio"] = _duplicate_ratio(description_normalized)
    features["boilerplate_line_ratio"] = _boilerplate_ratio(parsed_line_texts)

    # Null / malformed
    if description_raw is None or (isinstance(description_raw, str) and not description_raw.strip()):
        return QualityAssessment(
            status=QualityStatus.MALFORMED,
            quality_score=0.0,
            reasons=["missing or empty description"],
            features=features,
        )

    score = 0.0
    # Length contribution (cap)
    if nchar >= 800:
        score += 0.25
        reasons.append("substantial length")
    elif nchar >= 200:
        score += 0.15
        reasons.append("moderate length")
    elif nchar >= 50:
        score += 0.08
        reasons.append("short but non-empty")
    else:
        reasons.append("very short description")

    if nlines >= 5:
        score += 0.15
        reasons.append("multiple lines")
    elif nlines >= 1:
        score += 0.05

    if features["section_header_like_lines"] >= 1:
        score += 0.12
        reasons.append("section-like structure")

    if features["skillish_line_count"] >= 2:
        score += 0.22
        reasons.append("multiple skill-like lines")
    elif features["skillish_line_count"] == 1:
        score += 0.10
        reasons.append("some skill-like phrasing")

    dup = features["duplicate_line_ratio"]
    score -= 0.15 * min(dup, 1.0)
    if dup > 0.35:
        reasons.append("high duplicate line ratio")

    bp = features["boilerplate_line_ratio"]
    score -= 0.12 * min(bp, 1.0)
    if bp > 0.55:
        reasons.append("mostly boilerplate/benefits/legal lines")

    score = max(0.0, min(1.0, score))

    # Status
    if nchar < 30 or (nlines == 0 and nchar < 80):
        status = QualityStatus.TRUNCATED_OR_BROKEN
    elif bp > 0.65 and features["skillish_line_count"] == 0:
        status = QualityStatus.BOILERPLATE_HEAVY
    elif score < 0.2 and nchar < 120:
        status = QualityStatus.LOW_INFORMATION
    elif score >= complete_min_score:
        status = QualityStatus.COMPLETE
    elif score < complete_min_score:
        status = QualityStatus.LOW_INFORMATION
    else:
        status = QualityStatus.COMPLETE

    return QualityAssessment(status=status, quality_score=round(score, 4), reasons=reasons, features=features)
