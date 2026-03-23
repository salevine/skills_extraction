"""
Multi-signal final confidence (not model score alone).
"""

from __future__ import annotations

from typing import List, Optional


def rules_from_mention(rules_fired: List[str]) -> int:
    return len(rules_fired or [])


def compute_final_confidence(
    raw_model_confidence: float,
    verifier_confidence: Optional[float],
    section: str,
    rules_fired: List[str],
    boilerplate_label: str,
    offset_valid: bool,
    evidence_substring_of_line: bool,
    verifier_status: str = "skipped",
) -> float:
    s = float(raw_model_confidence)
    if verifier_status == "parse_failed":
        # Do not blend missing verifier output; penalize (not silent acceptance)
        s -= 0.18
    elif verifier_status == "error":
        s -= 0.12
    elif verifier_confidence is not None and verifier_status in ("accepted", "rejected"):
        s = 0.45 * s + 0.55 * float(verifier_confidence)

    sec_boost = {
        "requirements": 0.06,
        "qualifications": 0.05,
        "preferred": 0.03,
        "education": 0.03,
        "responsibilities": 0.02,
    }.get(section, 0.0)
    s += sec_boost

    s += min(0.1, rules_from_mention(rules_fired) * 0.025)

    if boilerplate_label in ("likely_legal", "likely_benefits", "likely_marketing", "likely_boilerplate"):
        s -= 0.14
    if boilerplate_label == "uncertain":
        s -= 0.04
    if not offset_valid:
        s -= 0.12
    if not evidence_substring_of_line:
        s -= 0.06

    return max(0.0, min(1.0, round(s, 4)))


def needs_verifier(
    raw_confidence: float,
    hard_soft: str,
    requirement_level: str,
    boilerplate_label: str,
    verify_low_below: float,
    verify_unknown_hs: bool,
    verify_unclear_req: bool,
    verify_uncertain_bp: bool,
) -> bool:
    if raw_confidence < verify_low_below:
        return True
    if verify_unknown_hs and hard_soft == "unknown":
        return True
    if verify_unclear_req and requirement_level == "unclear":
        return True
    if verify_uncertain_bp and boilerplate_label == "uncertain":
        return True
    return False
