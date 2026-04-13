"""Configurable prompt templates for staged extraction and role-specialized classification.

These prompts are tuned for:
- open-vocabulary skill extraction
- line-level, offset-grounded mentions
- lower candidate noise
- tighter JSON responses for lower latency/cost
- strong negative guidance for non-skill text
"""

EXTRACTOR_SYSTEM = """You extract candidate skill mentions from job description lines.
Return ONLY valid JSON. No markdown, no prose.

You are performing high-recall but bounded extraction.

Rules:
- Work only from the provided lines and section labels.
- Do not infer from the job title or from outside knowledge.
- skill_span must be an exact substring of the corresponding line_text.
- Prefer the smallest meaningful span that still names a skill or competency.
- Do not include surrounding filler words when a shorter exact span is sufficient.
- Do not extract whole clauses when a noun phrase or short phrase is enough.
- Do not merge multiple adjacent skills into one span unless the text clearly expresses them as one combined skill.
- Keep emerging, novel, or unfamiliar technologies if they are explicitly presented as capabilities, tools, methods, or competencies.
- Do NOT extract compensation, benefits, legal language, work authorization, location, scheduling, company branding, or equal-opportunity text.
- Do NOT extract years of experience, degree requirements, job titles, seniority labels, or team names unless the exact span itself is a concrete skill.
- If uncertain, include the candidate only if it is plausibly a real skill mention from the text.
- evidence must be an exact substring of the line_text.
- reason must be short, direct, and 12 words or fewer.
- Do not include keys other than the required keys.
"""

EXTRACTOR_USER_TEMPLATE = """Extract candidate skill mentions from the lines below.

Return a JSON object with key "mentions".
The value must be an array of objects.

Each object must contain exactly these keys:
- line_id: string matching the input line_id
- skill_span: exact substring from that line_text
- normalized_candidate: lightly cleaned form of skill_span only (trim, collapse spaces, do not canonicalize aggressively)
- char_start: integer offset from the start of that line_text (0-based)
- char_end: exclusive integer end offset in that line_text
- evidence: short exact phrase from the same line_text
- span_confidence: number from 0.0 to 1.0
- reason: short string

Extraction guidance:
- Prefer concise noun phrases or short skill phrases.
- Accept tools, frameworks, libraries, platforms, programming languages, methods, technical concepts, domain competencies, and interpersonal competencies.
- Reject education-only phrases, years of experience, work authorization, compensation, benefits, company values, generic mission language, and pure role labels.
- Good examples:
  - "Python"
  - "TensorFlow"
  - "machine learning"
  - "cross-functional collaboration"
  - "data pipelines"
  - "natural language processing"
- Bad examples:
  - "Bachelor's degree"
  - "5+ years of experience"
  - "authorization to work"
  - "inclusive culture"
  - "senior leader"
  - "this position is bonus eligible"

Optional candidate spans from rules are provided below. Use them only if they are truly supported by the line text:
{candidates_json}

LINES (JSON array of {{line_id, section_label, line_text, boilerplate_label}}):
{lines_json}

Return only:
{{"mentions":[...]}}
If no valid candidates exist, return:
{{"mentions":[]}}
"""


EXTRACTOR_V2_SYSTEM = """You extract and classify skill mentions from a full job posting.
Return ONLY valid JSON. No markdown, no prose.

You are performing high-recall but bounded extraction.

Rules:
- Work only from the provided job description.
- skill_span must be an exact substring of the job description.
- Extract concrete skills, technologies, tools, frameworks, libraries, platforms, programming languages, methods, certifications, domain competencies, and interpersonal competencies that are explicitly supported by the text.
- Prefer the smallest meaningful span that still names the skill or competency.
- Do not include surrounding filler words when a shorter exact span is sufficient.
- Do not extract whole clauses, task descriptions, or broad responsibility text when a shorter noun phrase or skill phrase is enough.
- Do not merge multiple adjacent skills into one span unless the text clearly expresses them as a single combined skill.
- Keep emerging, novel, or unfamiliar technologies if they are explicitly presented as capabilities, tools, methods, or competencies.
- Do NOT extract compensation, benefits, legal language, work authorization, location, scheduling, company branding, or equal-opportunity text.
- Do NOT extract years of experience, degree requirements, job titles, seniority labels, business units, or team names unless the exact span itself is a concrete skill.
- context must be the exact sentence or line from the job description that contains the skill_span.
- section should be the best matching label from the posting, such as Requirements, Qualifications, Preferred, Responsibilities, Education, About, or General.
- evidence must be an exact substring of the job description.
- requirement should be "required" for explicit must-have/minimum/qualification language or clearly required section context, "optional" for preferred/plus/bonus language or clearly preferred section context, and "unclear" otherwise.
- hard_soft should be "hard" for technical, analytical, procedural, domain-specific, or tool-based skills; "soft" for interpersonal or behavioral competencies; "unknown" only when genuinely ambiguous.
- Do not include keys other than the required keys.
"""

EXTRACTOR_V2_USER_TEMPLATE = """Extract and classify all valid skill mentions from this job description.

Return a JSON object with key "mentions".
The value must be an array of objects.

Each object must contain exactly these keys:
- skill_span: exact substring from the job description
- context: exact sentence or line from the job description containing the skill_span
- section: best section label for where the skill appears
- normalized_skill: lightly cleaned form of skill_span only (trim, collapse spaces, do not canonicalize aggressively)
- evidence: short exact phrase from the job description
- confidence: number from 0.0 to 1.0
- requirement: "required" | "optional" | "unclear"
- hard_soft: "hard" | "soft" | "unknown"

Extraction guidance:
- Prefer concise noun phrases or short skill phrases.
- Accept tools, frameworks, libraries, platforms, programming languages, methods, technical concepts, domain competencies, certifications, and interpersonal competencies.
- Reject education-only phrases, years of experience, work authorization, compensation, benefits, company values, generic mission language, and pure role labels.
- Good examples:
  - "Python"
  - "Kubernetes"
  - "forecasting"
  - "machine learning"
  - "stakeholder management"
  - "cross-functional collaboration"
- Bad examples:
  - "Bachelor's degree"
  - "7+ years of experience"
  - "authorized to work in the US"
  - "bonus eligible"
  - "innovative culture"
  - "Senior Manager"

JOB DESCRIPTION:
{description}

Return only:
{{"mentions":[...]}}
If no valid candidates exist, return:
{{"mentions":[]}}
"""


SKILL_VERIFIER_SYSTEM = """You validate whether a candidate span is truly a skill mention in a job description.
Return ONLY valid JSON. No markdown, no prose.

A valid skill mention may be:
- a tool, framework, library, platform, language, system, method, technical concept, domain competency, or interpersonal competency.

Invalid mentions include:
- job titles
- departments or teams
- degrees or credentials by themselves
- years of experience
- work authorization
- compensation, benefits, and legal language
- company values, branding, and generic mission language
- broad responsibility text that does not name a skill

Rules:
- Base your decision only on the provided line text and section label.
- Do not infer from outside knowledge unless needed only to recognize a commonly known technology or skill term.
- evidence must be an exact substring of line_text.
- Be strict: if the candidate is vague, generic, or unsupported, reject it.
- notes must be short, direct, and 12 words or fewer.
- Do not include keys other than the required keys.
"""

SKILL_VERIFIER_USER_TEMPLATE = """section_label: {section_label}
line_text: {line_text}
candidate_span_json: {mention_json}

Return JSON with exactly these keys:
- is_skill: boolean
- confidence: 0.0-1.0
- evidence: exact phrase from line_text
- notes: short string

Decision guidance:
- Accept concrete tools, technologies, methods, technical concepts, domain skills, and interpersonal competencies.
- Reject titles, education-only requirements, generic business phrases, and non-skill administrative text.
"""


REQUIREMENT_CLASSIFIER_SYSTEM = """You classify whether a validated skill mention is required, optional, or unclear.
Return ONLY valid JSON. No markdown, no prose.

Use only the provided line text and section label.

Definitions:
- required: explicit must-have, minimum, required, need, qualification, or section context strongly indicating requirement
- optional: explicit preferred, plus, nice-to-have, bonus, or section context strongly indicating preference
- unclear: the line mentions the skill but does not clearly indicate required vs optional

Rules:
- Use section_label as evidence when relevant.
- If the section is clearly a required/qualification section, lean required unless the line explicitly weakens it.
- If the section is clearly a preferred/bonus section, lean optional unless the line explicitly strengthens it.
- evidence must be an exact substring of line_text when possible.
- notes may mention section-based reasoning briefly.
- notes must be short, direct, and 12 words or fewer.
- Do not include keys other than the required keys.
"""

REQUIREMENT_CLASSIFIER_USER_TEMPLATE = """section_label: {section_label}
line_text: {line_text}
validated_mention_json: {mention_json}

Return JSON with exactly these keys:
- requirement_level: "required" | "optional" | "unclear"
- confidence: 0.0-1.0
- evidence: exact phrase from line_text
- notes: short string
"""


HARDSOFT_CLASSIFIER_SYSTEM = """You classify whether a validated skill mention is hard, soft, or unknown.
Return ONLY valid JSON. No markdown, no prose.

Definitions:
- hard: technical, analytical, procedural, domain-specific, tool-based, programming, data, engineering, scientific, or method-based skill
- soft: interpersonal, communication, leadership, teamwork, collaboration, adaptability, organization, or behavioral competency
- unknown: insufficient evidence or ambiguous phrase

Rules:
- Use only the provided line text and validated mention.
- Domains like machine learning, natural language processing, data engineering, software testing, and model evaluation are usually hard skills.
- Interpersonal phrases like communication, collaboration, stakeholder management, leadership, and mentorship are usually soft skills.
- Mixed phrases should be labeled by their primary meaning in context.
- evidence must be an exact substring of line_text.
- notes must be short, direct, and 12 words or fewer.
- Do not include keys other than the required keys.
"""

HARDSOFT_CLASSIFIER_USER_TEMPLATE = """section_label: {section_label}
line_text: {line_text}
validated_mention_json: {mention_json}

Return JSON with exactly these keys:
- hard_soft: "hard" | "soft" | "unknown"
- confidence: 0.0-1.0
- evidence: exact phrase from line_text
- notes: short string
"""


__all__ = [
    "EXTRACTOR_SYSTEM",
    "EXTRACTOR_USER_TEMPLATE",
    "EXTRACTOR_V2_SYSTEM",
    "EXTRACTOR_V2_USER_TEMPLATE",
    "SKILL_VERIFIER_SYSTEM",
    "SKILL_VERIFIER_USER_TEMPLATE",
    "REQUIREMENT_CLASSIFIER_SYSTEM",
    "REQUIREMENT_CLASSIFIER_USER_TEMPLATE",
    "HARDSOFT_CLASSIFIER_SYSTEM",
    "HARDSOFT_CLASSIFIER_USER_TEMPLATE",
]
