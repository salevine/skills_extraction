"""Second-pass skill validator for extracted spans."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .config import PipelineConfig
from .llm_ollama import call_llm, parse_json_loose
from .prompts import SKILL_VERIFIER_SYSTEM, SKILL_VERIFIER_USER_TEMPLATE

logger = logging.getLogger(__name__)


def verify_mention(
    cfg: PipelineConfig,
    section_label: str,
    line_text: str,
    mention_dict: Dict[str, Any],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    model = model or cfg.verifier_model
    user = SKILL_VERIFIER_USER_TEMPLATE.format(
        section_label=section_label,
        line_text=line_text,
        mention_json=json.dumps(mention_dict, ensure_ascii=False),
    )
    raw = call_llm(cfg, model, SKILL_VERIFIER_SYSTEM, user, temperature=0.05, role="verifier")
    try:
        data = parse_json_loose(raw)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning("Verifier parse failed: %s", e)
    # Do not optimistically accept: flag parse failure for downstream penalty / review
    return {
        "is_skill": True,
        "evidence": mention_dict.get("evidence", ""),
        "confidence": None,
        "notes": "verifier_parse_failed",
        "_parse_failed": True,
    }
