"""
Explicit schemas for job records, lines, mentions, quality, and metadata.
All structures are JSON-serializable via `to_dict()` / `dataclasses.asdict` with enums as strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class QualityStatus(str, Enum):
    COMPLETE = "complete"
    LOW_INFORMATION = "low_information"
    TRUNCATED_OR_BROKEN = "truncated_or_broken"
    BOILERPLATE_HEAVY = "boilerplate_heavy"
    MALFORMED = "malformed"


class BoilerplateLabel(str, Enum):
    SKILLS_RELEVANT = "skills_relevant"
    LIKELY_BOILERPLATE = "likely_boilerplate"
    LIKELY_BENEFITS = "likely_benefits"
    LIKELY_LEGAL = "likely_legal"
    LIKELY_MARKETING = "likely_marketing"
    UNCERTAIN = "uncertain"


class HardSoft(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    UNKNOWN = "unknown"


class RequirementLevel(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    UNCLEAR = "unclear"


def _enum_val(x: Any) -> Any:
    if isinstance(x, Enum):
        return x.value
    return x


@dataclass
class QualityAssessment:
    status: QualityStatus
    quality_score: float
    reasons: List[str]
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ParsedLine:
    line_id: str
    section: str
    text: str
    char_start: int
    char_end: int
    boilerplate_label: str
    line_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateSpan:
    candidate_text: str
    line_id: str
    char_start: int
    char_end: int
    rule_source: str
    section: str
    context_window: str
    sentence_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillMention:
    mention_id: str
    line_id: str
    skill_span: str
    normalized_candidate: str
    is_skill: bool
    hard_soft: str
    requirement_level: str
    char_start: int
    char_end: int
    evidence: str
    raw_model_confidence: float
    final_confidence: float
    extractor_model: str
    rules_fired: List[str]
    verifier_status: str
    verifier_model: str
    verifier_confidence: Optional[float]
    verifier_notes: str
    created_at: str
    run_id: str
    requirement_model: str = ""
    requirement_status: str = "skipped"
    requirement_confidence: Optional[float] = None
    requirement_notes: str = ""
    hardsoft_model: str = ""
    hardsoft_status: str = "skipped"
    hardsoft_confidence: Optional[float] = None
    hardsoft_notes: str = ""
    pipeline_audit: Dict[str, Any] = field(default_factory=dict)
    section: str = ""
    source_line: str = ""
    # Multi-model consensus not implemented; fields reserved / documented
    support_count: int = 0
    models_that_found_it: List[str] = field(default_factory=list)
    model_disagreements: List[str] = field(default_factory=list)
    support_metadata: str = "single_extractor_pass; support fields not used for adjudication yet"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionMetadata:
    run_id: str
    pipeline_version: str
    extractor_model: str
    verifier_model: str
    job_key: str
    started_at: str
    completed_at: str = ""
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_parsed_line(obj: ParsedLine, reference: str) -> bool:
    if not obj.line_id or obj.char_start < 0 or obj.char_end < obj.char_start:
        return False
    if obj.char_end > len(reference):
        return False
    return reference[obj.char_start : obj.char_end] == obj.text


def validate_mention_offsets(text: str, start: int, end: int) -> bool:
    if start < 0 or end > len(text) or start > end:
        return False
    return True
