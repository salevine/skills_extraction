"""Configurable prompt templates for span-first extraction and verification."""

EXTRACTOR_SYSTEM = """You are extracting skill mentions from job description text.
Return ONLY valid JSON (no markdown). The user message contains structured input.
Rules:
- Work only from the provided lines and section labels.
- Do not infer from the job title.
- skill_span must be an exact substring of the line_text for that line_id.
- Do not deduplicate across mentions.
- Preserve unfamiliar tools, technologies, methods, or competencies if stated as capabilities.
- Exclude benefits, compensation, EEO, and pure employer branding unless a concrete skill appears.
"""

EXTRACTOR_USER_TEMPLATE = """For each line below, extract skill mentions as JSON array of objects.
Each object must have:
- line_id: string (must match input)
- skill_span: exact text from that line
- normalized_candidate: lightly cleaned (trim, collapse spaces)
- is_skill: boolean
- hard_soft: "hard" | "soft" | "unknown"
- requirement_level: "required" | "optional" | "unclear"
- char_start: integer offset from start of THAT line's text (0-based)
- char_end: exclusive end offset in that line
- evidence: short exact phrase from the line
- confidence: number 0.0-1.0
- reason: short string

Optional candidates (may help recall; validate each):
{candidates_json}

LINES (JSON array of {{line_id, section_label, line_text, boilerplate_label}}):
{lines_json}

Return JSON object with key "mentions" whose value is the array of all mentions from all lines.
If a line has no skills, omit entries for that line or use is_skill false only for rejected candidates.
Do not wrap in markdown code fences.
"""


VERIFIER_SYSTEM = """You validate skill mentions against a single job description line.
Return ONLY valid JSON, no markdown."""

VERIFIER_USER_TEMPLATE = """section_label: {section_label}
line_text: {line_text}
extracted_mention_json: {mention_json}

Return JSON with:
- is_valid_skill_mention: boolean
- corrected_hard_soft: "hard" | "soft" | "unknown"
- corrected_requirement_level: "required" | "optional" | "unclear"
- evidence: exact phrase from line_text
- confidence: 0.0-1.0
- notes: short string
"""
