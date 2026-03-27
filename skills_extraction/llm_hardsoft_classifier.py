"""Hard/soft classifier for validated skill mentions."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .config import PipelineConfig
from .llm_ollama import call_ollama, parse_json_loose
from .prompts import HARDSOFT_CLASSIFIER_SYSTEM, HARDSOFT_CLASSIFIER_USER_TEMPLATE

logger = logging.getLogger(__name__)


def classify_hard_soft(
    cfg: PipelineConfig,
    section_label: str,
    line_text: str,
    mention_dict: Dict[str, Any],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    model = model or cfg.hardsoft_model
    user = HARDSOFT_CLASSIFIER_USER_TEMPLATE.format(
        section_label=section_label,
        line_text=line_text,
        mention_json=json.dumps(mention_dict, ensure_ascii=False),
    )
    raw = call_ollama(
        cfg, model, HARDSOFT_CLASSIFIER_SYSTEM, user, temperature=0.05, role="verifier"
    )
    try:
        data = parse_json_loose(raw)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning("Hard/soft classifier parse failed: %s", e)

    return {
        "hard_soft": mention_dict.get("hard_soft", "unknown"),
        "confidence": None,
        "evidence": mention_dict.get("evidence", ""),
        "notes": "hardsoft_classifier_parse_failed",
        "_parse_failed": True,
    }
