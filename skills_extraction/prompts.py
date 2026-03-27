"""Configurable prompt templates for staged extraction and role-specialized classification."""

EXTRACTOR_SYSTEM = """You are extracting candidate skill spans from job description text.
Return ONLY valid JSON (no markdown). The user message contains structured input.
Rules:
- Work only from the provided lines and section labels.
- Do not infer from the job title.
- skill_span must be an exact substring of the line_text for that line_id.
- Return candidate spans even if uncertain; later stages will verify validity and classify attributes.
"""

EXTRACTOR_USER_TEMPLATE = """For each line below, extract skill mentions as JSON array of objects.
Each object must have:
- line_id: string (must match input)
- skill_span: exact text from that line
- normalized_candidate: lightly cleaned (trim, collapse spaces)
- char_start: integer offset from start of THAT line's text (0-based)
- char_end: exclusive end offset in that line
- evidence: short exact phrase from the line
- span_confidence: number 0.0-1.0
- reason: short string

Optional candidates (may help recall; validate each):
{candidates_json}

LINES (JSON array of {{line_id, section_label, line_text, boilerplate_label}}):
{lines_json}

Return JSON object with key "mentions" whose value is the array of all mentions from all lines.
If a line has no skill-like spans, omit entries for that line.
Do not wrap in markdown code fences.
"""


SKILL_VERIFIER_SYSTEM = """You validate whether a candidate span is a real skill mention.
Return ONLY valid JSON, no markdown.
Rules:
- Base your answer only on the provided line text and section label.
- A valid skill mention can be a tool, technology, method, domain competency, or interpersonal competency.
- Reject employer branding, legal language, benefits, compensation, and role labels that are not skills.
- evidence must be an exact substring of line_text.
"""

SKILL_VERIFIER_USER_TEMPLATE = """section_label: {section_label}
line_text: {line_text}
candidate_span_json: {mention_json}

Return JSON with:
- is_skill: boolean
- confidence: 0.0-1.0
- evidence: exact phrase from line_text
- notes: short string
"""


REQUIREMENT_CLASSIFIER_SYSTEM = """You classify whether a validated skill is required, optional, or unclear.
Return ONLY valid JSON, no markdown.
Rules:
- Use only the provided line and section.
- required: explicit must/required/need/minimum language.
- optional: preferred/nice-to-have/plus language.
- unclear: not enough direct evidence.
- evidence must be an exact substring of line_text.
"""

REQUIREMENT_CLASSIFIER_USER_TEMPLATE = """section_label: {section_label}
line_text: {line_text}
validated_mention_json: {mention_json}

Return JSON with:
- requirement_level: "required" | "optional" | "unclear"
- confidence: 0.0-1.0
- evidence: exact phrase from line_text
- notes: short string
"""


HARDSOFT_CLASSIFIER_SYSTEM = """You classify whether a validated skill is hard, soft, or unknown.
Return ONLY valid JSON, no markdown.
Rules:
- hard: technical or domain-specific skills (tools, languages, methods, certifications).
- soft: interpersonal, communication, leadership, collaboration, behavioral competencies.
- unknown: insufficient evidence.
- evidence must be an exact substring of line_text.
"""

HARDSOFT_CLASSIFIER_USER_TEMPLATE = """section_label: {section_label}
line_text: {line_text}
validated_mention_json: {mention_json}

Return JSON with:
- hard_soft: "hard" | "soft" | "unknown"
- confidence: 0.0-1.0
- evidence: exact phrase from line_text
- notes: short string
"""
