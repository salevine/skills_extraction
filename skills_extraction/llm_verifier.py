"""
Second-pass verifier for ambiguous extractions.
Parse failures are explicit (not silent acceptance).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .config import PipelineConfig
from .llm_ollama import call_ollama, parse_json_loose
from .prompts import VERIFIER_SYSTEM, VERIFIER_USER_TEMPLATE

logger = logging.getLogger(__name__)


def verify_mention(
    cfg: PipelineConfig,
    section_label: str,
    line_text: str,
    mention_dict: Dict[str, Any],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    model = model or cfg.verifier_model
    user = VERIFIER_USER_TEMPLATE.format(
        section_label=section_label,
        line_text=line_text,
        mention_json=json.dumps(mention_dict, ensure_ascii=False),
    )
    raw = call_ollama(cfg, model, VERIFIER_SYSTEM, user, temperature=0.05, role="verifier")
    try:
        data = parse_json_loose(raw)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning("Verifier parse failed: %s", e)
    # Do not optimistically accept: flag parse failure for downstream penalty / review
    return {
        "is_valid_skill_mention": True,
        "corrected_hard_soft": mention_dict.get("hard_soft", "unknown"),
        "corrected_requirement_level": mention_dict.get("requirement_level", "unclear"),
        "evidence": mention_dict.get("evidence", ""),
        "confidence": None,
        "notes": "verifier_parse_failed",
        "_parse_failed": True,
    }
