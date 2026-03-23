"""
Deterministic candidate span harvesting — precision-tuned: fewer false positives
than broad line-wide comma/prep mining; still recall-oriented for real requirement cues.
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple

from .schemas import CandidateSpan, ParsedLine

# "experience with X" style — capture tail list fragment
_EXP_WITH = re.compile(
    r"(?:^|\b)(?:experience|experienced|proficiency|proficient|expertise|knowledge|"
    r"familiarity|familiar|skilled|ability|background|exposure|hands-on\s+experience)"
    r"\s+(?:with|in)\s+([^.\n;]+)",
    re.I,
)
_MUST_HAVE = re.compile(
    r"(?:^|\b)(?:must\s+have|required|need\s+to\s+have|minimum)\s*[:\s]+([^.\n]+)",
    re.I,
)
_PREFERRED = re.compile(
    r"(?:^|\b)(?:preferred|nice\s+to\s+have|a\s+plus|bonus)\s*[:\s]+([^.\n]+)",
    re.I,
)
_STRONG_SKILLS = re.compile(r"\bstrong\s+([A-Za-z][A-Za-z0-9+.#\s\-]{2,60})\s+skills?\b", re.I)

# Stricter "tool-like" token: digit, dot-version, plus, hash, CamelCase multi-part, or 2+ chars + known tech shape
_TOOLISH = re.compile(
    r"^(?:[A-Za-z]+(?:\.[A-Za-z0-9]+)+|[A-Za-z0-9][A-Za-z0-9+.#\-]{1,39}|[A-Z][a-z]+[A-Z][A-Za-z0-9+.#\-]*)$"
)

# Comma / "and" list splitter inside a fragment
_SPLIT_LIST = re.compile(r",|\s+and\s+|\s*&\s*|\s*/\s*", re.I)

# Sections where comma-list mining is allowed (explicit requirement context)
_REQUIREMENT_SECTIONS = frozenset(
    {"requirements", "qualifications", "preferred", "education", "responsibilities"}
)

# Line must match this to allow comma_list_token outside those sections (or section gate)
_SKILL_CUES = re.compile(
    r"\b(experience|experienced|proficien|expertise|knowledge|familiar|skill|ability|"
    r"using|hands-on|background|exposure|must\s+have|required|preferred|years?\s+of|"
    r"technologies|tools?|software|stack|framework|platform|certified|certification|"
    r"degree\s+in|bachelor|master|ph\.?d)\b",
    re.I,
)

# Obvious non-skill tokens (verbs, generic nouns) — block as standalone candidates
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "our",
        "you",
        "all",
        "your",
        "this",
        "that",
        "with",
        "from",
        "have",
        "has",
        "are",
        "was",
        "were",
        "been",
        "being",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "not",
        "but",
        "or",
        "as",
        "an",
        "a",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "we",
        "they",
        "their",
        "us",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "if",
        "then",
        "than",
        "into",
        "over",
        "under",
        "about",
        "such",
        "other",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "some",
        "any",
        "no",
        "yes",
        "also",
        "only",
        "just",
        "even",
        "well",
        "very",
        "here",
        "there",
        "where",
        "work",
        "working",
        "team",
        "teams",
        "role",
        "job",
        "jobs",
        "position",
        "company",
        "companies",
        "organization",
        "career",
        "careers",
        "culture",
        "mission",
        "vision",
        "values",
        "patient",
        "patients",
        "customer",
        "customers",
        "client",
        "clients",
        "business",
        "services",
        "service",
        "solutions",
        "solution",
        "global",
        "national",
        "local",
        "world",
        "industry",
        "industries",
        "market",
        "markets",
        "growth",
        "opportunity",
        "opportunities",
        "environment",
        "environments",
        "science",
        "sciences",
        "research",
        "development",
        "product",
        "products",
        "project",
        "projects",
        "program",
        "programs",
        "process",
        "processes",
        "system",
        "systems",
        "data",
        "information",
        "content",
        "digital",
        "online",
        "new",
        "next",
        "best",
        "great",
        "strong",
        "excellent",
        "outstanding",
        "successful",
        "success",
        "leadership",
        "management",
        "communication",
        "skills",
        "ability",
        "abilities",
        "experience",
        "experiences",
        "year",
        "years",
        "day",
        "days",
        "time",
        "full",
        "part",
        "remote",
        "hybrid",
        "office",
        "home",
        "based",
        "location",
        "locations",
        "state",
        "states",
        "united",
        "equal",
        "employer",
        "employment",
        "applicants",
        "candidates",
        "people",
        "person",
        "individual",
        "individuals",
        "member",
        "members",
        "staff",
        "employee",
        "employees",
        "prevents",
        "prevent",
        "detects",
        "detect",
        "treats",
        "treat",
        "ensures",
        "ensure",
        "provides",
        "provide",
        "includes",
        "include",
        "supports",
        "support",
        "helps",
        "help",
        "makes",
        "make",
        "builds",
        "build",
        "creates",
        "create",
        "develops",
        "develop",
        "designs",
        "design",
        "implements",
        "implement",
        "manages",
        "manage",
        "leads",
        "lead",
        "works",
        "seeking",
        "looking",
        "join",
        "joining",
        "apply",
        "applying",
        "submit",
        "click",
        "visit",
        "website",
    }
)

# Restricted prep mining: only after these stronger heads (not bare "in|using")
_PREP_STRONG = re.compile(
    r"(?:^|\b)(?:experience|experienced|proficiency|proficient|expertise|knowledge|"
    r"familiarity|familiar|skilled|competent|competency|background|exposure|hands-on)\s+"
    r"(?:with|in)\s+([A-Za-z0-9][A-Za-z0-9+.#\s\-]{1,48}?)"
    r"(?:\s*[,.;]|$|\s+and\b)",
    re.I,
)


def _trim_span(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).strip(" \t.;:")


def _is_stop_token(tok: str) -> bool:
    t = tok.strip().lower()
    if len(t) < 2:
        return True
    if t in _STOPWORDS:
        return True
    if t.endswith("ing") and len(t) > 5 and t[:-3] in _STOPWORDS:
        return True
    if t.endswith("s") and len(t) > 4 and t[:-1] in _STOPWORDS:
        return True
    return False


def _token_toolish_enough(tok: str) -> bool:
    """Stricter filter for comma-list / weak extractions."""
    t = tok.strip()
    if len(t) < 2 or len(t) > 42:
        return False
    if _is_stop_token(t):
        return False
    # Must look tool-like: version number, +, #, or CamelCase tech, or all-caps acronym
    if re.search(r"[0-9+#.]", t):
        return True
    if re.match(r"^[A-Z]{2,}$", t) and len(t) <= 10:
        return True
    if _TOOLISH.match(t):
        return True
    # Allow short known stacks only if mixed case techy
    if re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", t):
        return True
    return False


def _allow_comma_list_mining(pl: ParsedLine) -> bool:
    sec = (pl.section or "").lower()
    if sec in _REQUIREMENT_SECTIONS:
        return True
    return bool(_SKILL_CUES.search(pl.text or ""))


def _offsets_in_line(full_line: str, fragment: str, line_base: int) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    frag = _trim_span(fragment)
    if not frag or len(frag) < 2:
        return out
    idx = full_line.find(frag)
    if idx >= 0:
        return [(frag, line_base + idx, line_base + idx + len(frag))]
    parts = [p for p in _SPLIT_LIST.split(frag) if p.strip()]
    for p in parts:
        p = _trim_span(p)
        if len(p) < 2:
            continue
        start = 0
        while True:
            j = full_line.find(p, start)
            if j < 0:
                break
            out.append((p, line_base + j, line_base + j + len(p)))
            start = j + 1
    return out


def mine_candidates_for_line(pl: ParsedLine, description_normalized: str) -> List[CandidateSpan]:
    spans: List[CandidateSpan] = []
    seen: Set[Tuple[int, int, str]] = set()
    line = pl.text
    base = pl.char_start

    def add(rule: str, frag: str) -> None:
        for text, cs, ce in _offsets_in_line(line, frag, base):
            key = (cs, ce, text)
            if key in seen:
                continue
            if _is_stop_token(text):
                continue
            seen.add(key)
            ctx_start = max(0, cs - base - 40)
            ctx_end = min(len(description_normalized), ce - base + 40 + len(line))
            ctx = description_normalized[pl.char_start + ctx_start : pl.char_start + min(len(line), ctx_end)]
            spans.append(
                CandidateSpan(
                    candidate_text=text,
                    line_id=pl.line_id,
                    char_start=cs,
                    char_end=ce,
                    rule_source=rule,
                    section=pl.section,
                    context_window=ctx[:200],
                )
            )

    for m in _EXP_WITH.finditer(line):
        add("experience_with_pattern", m.group(1))
    for m in _MUST_HAVE.finditer(line):
        add("must_have_pattern", m.group(1))
    for m in _PREFERRED.finditer(line):
        add("preferred_pattern", m.group(1))
    for m in _STRONG_SKILLS.finditer(line):
        add("strong_skills_pattern", m.group(1))

    # Restricted prep: only after strong heads (replaces noisy bare using|in)
    for m in _PREP_STRONG.finditer(line):
        frag = m.group(1).strip()
        if len(frag) >= 2 and not _is_stop_token(frag.split()[0] if frag.split() else frag):
            add("prep_phrase_strong", frag)

    # Comma-list tokens: only in requirement sections or lines with skill cues
    if _allow_comma_list_mining(pl) and "," in line and re.search(
        r"\b([A-Za-z][A-Za-z0-9+.#]{1,40})\s*,\s*([A-Za-z][A-Za-z0-9+.#]{1,40})", line
    ):
        for m in re.finditer(
            r"\b([A-Za-z][A-Za-z0-9+.#]{1,40})\s*(?:,|\s+and\s+|\s*/\s*)", line
        ):
            tok = m.group(1).strip()
            if _token_toolish_enough(tok):
                add("comma_list_token", tok)

    # Post-colon fragment: only when section or cues suggest requirements
    if len(spans) == 0 and len(line) > 15 and ":" in line:
        if _allow_comma_list_mining(pl) or pl.boilerplate_label == "skills_relevant":
            tail = line.split(":", 1)[-1].strip()
            if len(tail) > 3 and len(tail) < 120 and _SKILL_CUES.search(line):
                add("post_colon_fragment", tail[:120])

    return spans


def mine_all_candidates(lines: List[ParsedLine], description_normalized: str) -> List[CandidateSpan]:
    all_c: List[CandidateSpan] = []
    for pl in lines:
        if pl.boilerplate_label in ("likely_legal", "likely_benefits") and len(pl.text) < 200:
            continue
        all_c.extend(mine_candidates_for_line(pl, description_normalized))
    return all_c
