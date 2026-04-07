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


# ---------------------------------------------------------------------------
# V2 extractor: whole-job description in a single call (replaces batched approach)
# ---------------------------------------------------------------------------

EXTRACTOR_V2_SYSTEM = """You are extracting and classifying skill mentions from a job posting.
Return ONLY valid JSON (no markdown fences).
Rules:
- Extract every skill, technology, tool, method, certification, and competency mentioned.
- Include both hard skills (Python, SQL, AWS) and soft skills (communication, leadership).
- Do not extract job titles, company names, benefits, or legal language.
- For each mention, provide the exact text as it appears and the surrounding sentence for context.
- Identify which section of the posting the skill appears in (e.g. Requirements, Qualifications, Responsibilities, About, etc.). If unclear, use "General".
- Classify each mention as hard or soft skill, and whether it is required or optional.
- Only include mentions you believe are genuine skills or competencies.
"""

EXTRACTOR_V2_USER_TEMPLATE = """Extract and classify all skill mentions from this job description.

JOB DESCRIPTION:
{description}

Return a JSON object with key "mentions" containing an array of objects. Each object must have:
- skill_span: exact text from the description (verbatim substring)
- context: the full sentence or line containing the skill
- section: which section of the posting (e.g. "Requirements", "Qualifications", "Responsibilities", "About", "General")
- normalized_skill: lightly cleaned version (trim whitespace, collapse spaces)
- evidence: short phrase showing why this is a skill
- confidence: number 0.0-1.0
- requirement: "required" if explicitly must-have/required/minimum, "optional" if preferred/nice-to-have/plus, "unclear" if not enough evidence
- hard_soft: "hard" for technical/domain skills (tools, languages, methods, certifications), "soft" for interpersonal/behavioral competencies (communication, leadership, teamwork), "unknown" if unclear

If no skills are found, return {{"mentions": []}}.
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
