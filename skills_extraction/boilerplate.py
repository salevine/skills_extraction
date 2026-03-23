"""
Line-level boilerplate / benefits / legal / marketing classification.

Conservative default: promote to skills_relevant only with lexical skill/requirement evidence.
Requirement-leaning sections start as uncertain unless cues match.
"""

from __future__ import annotations

import re
from typing import List

from .schemas import BoilerplateLabel, ParsedLine

_LEGAL_CUES = re.compile(
    r"\b(equal\s+opportunity|eeoc|affirmative\s+action|accommodation|"
    r"disability|veteran\s+status|gender\s+identity|non-?discrimination|"
    r"applicants\s+will\s+receive|consideration\s+for\s+employment)\b",
    re.I,
)
_BENEFITS_CUES = re.compile(
    r"\b(health\s+insurance|401k|pto|paid\s+time\s+off|dental|vision|"
    r"stock\s+options|equity|bonus\s+structure|remote\s+work\s+benefits|"
    r"unlimited\s+vacation|gym\s+membership)\b",
    re.I,
)
_MARKETING_CUES = re.compile(
    r"\b(world-?class|fast-?paced|rockstar|ninja|family|join\s+our\s+amazing|"
    r"best\s+place\s+to\s+work|innovative\s+culture|passionate\s+team)\b",
    re.I,
)
_APPLY_CUES = re.compile(
    r"\b(apply\s+now|submit\s+your\s+resume|click\s+here\s+to\s+apply|"
    r"no\s+phone\s+calls|recruiting\s+agency)\b",
    re.I,
)

# Evidence that this line plausibly carries extractable skills/requirements
_SKILL_LEX = re.compile(
    r"\b(experience|experienced|proficien|expertise|knowledge|familiar|skill|ability|"
    r"hands-on|background|exposure|must\s+have|required\s*:|years?\s+of|bachelor|master|"
    r"ph\.?d\.?|certif|degree|programming|develop|engineer|technology|technologies|"
    r"software|stack|framework|platform|using\s+[A-Za-z0-9]|with\s+[A-Za-z0-9]{2,}|"
    r"proficient\s+in|knowledge\s+of)\b",
    re.I,
)

_SECTION_SKILL_FRIENDLY = frozenset(
    {"requirements", "qualifications", "preferred", "education", "responsibilities"}
)
_OVERVIEW_LIKE = frozenset({"overview", "about", "body"})


def classify_line_boilerplate(line: ParsedLine) -> str:
    text = line.text.strip()
    if not text:
        return BoilerplateLabel.LIKELY_BOILERPLATE.value
    sec = (line.section or "").lower()
    if sec == "legal":
        return BoilerplateLabel.LIKELY_LEGAL.value
    if sec == "benefits":
        return BoilerplateLabel.LIKELY_BENEFITS.value

    if _LEGAL_CUES.search(text):
        return BoilerplateLabel.LIKELY_LEGAL.value
    if _BENEFITS_CUES.search(text) and len(text) < 300:
        return BoilerplateLabel.LIKELY_BENEFITS.value
    if _APPLY_CUES.search(text):
        return BoilerplateLabel.LIKELY_BOILERPLATE.value
    if _MARKETING_CUES.search(text) and not re.search(
        r"\b(python|java|sql|aws|kubernetes|docker|agile|degree|years?\s+of)\b", text, re.I
    ):
        return BoilerplateLabel.LIKELY_MARKETING.value

    has_skill_lex = bool(_SKILL_LEX.search(text))

    # Overview / about / generic body: uncertain unless clear skill lex
    if sec in _OVERVIEW_LIKE or sec == "body":
        if has_skill_lex:
            return BoilerplateLabel.SKILLS_RELEVANT.value
        if len(text) > 400:
            return BoilerplateLabel.UNCERTAIN.value
        return BoilerplateLabel.UNCERTAIN.value

    # Requirement-leaning sections: still need lexical evidence for skills_relevant
    if sec in _SECTION_SKILL_FRIENDLY:
        if has_skill_lex:
            return BoilerplateLabel.SKILLS_RELEVANT.value
        # Short bullet-ish lines in these sections may list skills without verbs
        if len(text) < 200 and re.match(r"^\s*[•\-\*●◦]\s+\S", text):
            return BoilerplateLabel.UNCERTAIN.value
        return BoilerplateLabel.UNCERTAIN.value

    # Default: conservative
    if has_skill_lex:
        return BoilerplateLabel.SKILLS_RELEVANT.value
    if len(text) > 400:
        return BoilerplateLabel.UNCERTAIN.value
    return BoilerplateLabel.UNCERTAIN.value


def label_parsed_lines(lines: List[ParsedLine]) -> None:
    for pl in lines:
        pl.boilerplate_label = classify_line_boilerplate(pl)
