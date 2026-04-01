"""Requirement-level classifier for validated skill mentions."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .config import PipelineConfig
from .llm_ollama import call_llm, parse_json_loose
from .prompts import (
    REQUIREMENT_CLASSIFIER_SYSTEM,
    REQUIREMENT_CLASSIFIER_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


def classify_requirement_level(
    cfg: PipelineConfig,
    section_label: str,
    line_text: str,
    mention_dict: Dict[str, Any],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    model = model or cfg.requirement_model
    user = REQUIREMENT_CLASSIFIER_USER_TEMPLATE.format(
        section_label=section_label,
        line_text=line_text,
        mention_json=json.dumps(mention_dict, ensure_ascii=False),
    )
    raw = call_llm(
        cfg, model, REQUIREMENT_CLASSIFIER_SYSTEM, user, temperature=0.05, role="verifier"
    )
    try:
        data = parse_json_loose(raw)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning("Requirement classifier parse failed: %s", e)

    return {
        "requirement_level": mention_dict.get("requirement_level", "unclear"),
        "confidence": None,
        "evidence": mention_dict.get("evidence", ""),
        "notes": "requirement_classifier_parse_failed",
        "_parse_failed": True,
    }
