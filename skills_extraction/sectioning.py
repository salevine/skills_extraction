"""
Section and line segmentation with stable line_ids and char offsets in normalized text.

Inserts newlines before common inline section headers so "Minimum Qualifications foo"
becomes two lines, improving section labels for real job text.
"""

from __future__ import annotations

import re
from typing import List

from .schemas import ParsedLine

# Lines that look like section headers (short, keyword-heavy)
_SECTION_PATTERNS = [
    (re.compile(r"^\s*(requirements?|required\s+qualifications?|minimum\s+qualifications?)\s*:?\s*$", re.I), "requirements"),
    (re.compile(r"^\s*(preferred\s+qualifications?|nice\s+to\s+have|bonus|plus)\s*:?\s*$", re.I), "preferred"),
    (re.compile(r"^\s*(qualifications?|skills?\s*(and|&)?\s*experience)\s*:?\s*$", re.I), "qualifications"),
    (re.compile(r"^\s*(responsibilities?|what\s+you\s*[\u2019']?ll\s+do|key\s+responsibilities)\s*:?\s*$", re.I), "responsibilities"),
    (re.compile(r"^\s*(about\s+(us|the\s+role|you))\s*:?\s*$", re.I), "about"),
    (re.compile(r"^\s*(benefits?|compensation|perks?)\s*:?\s*$", re.I), "benefits"),
    (re.compile(r"^\s*(equal\s+opportunity|eeoc|diversity|we\s+are\s+an\s+equal)\b", re.I), "legal"),
    (re.compile(r"^\s*(education|degree|certification)\s*:?\s*$", re.I), "education"),
    (re.compile(r"^\s*(overview|summary|position\s+summary)\s*:?\s*$", re.I), "overview"),
]

# Inline headers: insert newline before phrase when it follows non-newline text (mid-paragraph)
_INLINE_SECTION_MARKERS = [
    r"Minimum\s+Qualifications?",
    r"Preferred\s+Qualifications?",
    r"Required\s+Qualifications?",
    r"Basic\s+Qualifications?",
    r"Key\s+Responsibilities",
    r"Primary\s+Responsibilities",
    r"Job\s+Responsibilities",
    r"Responsibilities?\s*:",
    r"Duties\s*(?:and\s+Responsibilities)?\s*:",
    r"Education\s*(?:requirements?)?\s*:",
    r"Experience\s*(?:requirements?)?\s*:",
    r"Skills\s*(?:and\s+)?(?:Experience|Required)\s*:?",
    r"What\s+You[\u2019']?ll\s+Do",
    r"Qualifications?\s*:",
    r"Requirements?\s*:",
]


def split_inline_section_headers(text: str) -> str:
    """
    Break long paragraphs before recognized heading phrases so section detection
    can assign requirements vs body more reliably.
    """
    if not text:
        return text
    out = text
    for phrase in _INLINE_SECTION_MARKERS:
        out = re.sub(
            rf"(?<![\n\r])(?<=\S)\s+({phrase})",
            r"\n\1",
            out,
            flags=re.I,
        )
    # Colon-led inline: "Responsibilities: We are seeking" on one physical line → split after colon + space if tail is long
    out = re.sub(
        r"(?i)^(\s*(?:Key\s+)?Responsibilities?\s*:\s+)(.{60,})$",
        lambda m: m.group(1).rstrip() + "\n" + m.group(2).lstrip(),
        out,
        flags=re.MULTILINE,
    )
    return out


def detect_section_header(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    for rx, label in _SECTION_PATTERNS:
        if rx.match(stripped):
            return label
    # Line starts with heading-like phrase + colon + substantive tail (inline header)
    m = re.match(
        r"(?i)^\s*((?:minimum|required|preferred)\s+qualifications?|responsibilities?|"
        r"qualifications?|requirements?|education|experience)\s*:\s*\S",
        stripped,
    )
    if m:
        key = m.group(1).lower()
        if "preferred" in key:
            return "preferred"
        if "responsibilit" in key:
            return "responsibilities"
        if "education" in key:
            return "education"
        if "experience" in key and "qualification" not in key:
            return "qualifications"
        if "requirement" in key:
            return "requirements"
        if "qualification" in key:
            return "qualifications"
    if len(stripped) < 80 and stripped.endswith(":"):
        low = stripped.lower()
        for kw, lab in (
            ("requirement", "requirements"),
            ("qualification", "qualifications"),
            ("responsibilit", "responsibilities"),
            ("preferred", "preferred"),
            ("benefit", "benefits"),
        ):
            if kw in low:
                return lab
    return ""


def segment_lines(job_key: str, description_normalized: str) -> List[ParsedLine]:
    """
    Build ParsedLines from *final* normalized text (caller should run
    split_inline_section_headers first so offsets match stored description_normalized).
    """
    if not description_normalized:
        return []
    lines = description_normalized.split("\n")
    parsed: List[ParsedLine] = []
    cursor = 0
    current_section = "body"
    for i, line in enumerate(lines):
        start = cursor
        end = start + len(line)
        cursor = end + 1

        sec = detect_section_header(line)
        if sec:
            current_section = sec
        line_id = f"{job_key}_L{i+1:04d}"
        parsed.append(
            ParsedLine(
                line_id=line_id,
                section=current_section,
                text=line,
                char_start=start,
                char_end=end,
                boilerplate_label="uncertain",
                line_index=i,
            )
        )
    return parsed
