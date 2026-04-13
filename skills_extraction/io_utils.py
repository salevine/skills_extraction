"""
Load and save JSON; preserve original job records (append-only augmentation).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)


def load_jobs_json(path: Union[str, Path]) -> Tuple[List[Dict[str, Any]], str]:
    """Load jobs from JSON file. Returns (jobs_list, raw_path_str)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict):
        if "jobs" in data and isinstance(data["jobs"], list):
            jobs = data["jobs"]
        else:
            jobs = [data]
    elif isinstance(data, list):
        jobs = data
    else:
        raise ValueError(f"Unsupported JSON root type: {type(data)}")
    logger.info("Loaded %d job records from %s", len(jobs), p)
    return jobs, str(p.resolve())


def stable_job_key(job: Dict[str, Any], index: int) -> str:
    jid = job.get("id")
    if jid is not None and str(jid).strip():
        return f"J{str(jid).replace(' ', '_')[:80]}"
    return f"JIDX_{index:06d}"


def augment_job_record(original: Dict[str, Any], augmentation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge original job fields with pipeline augmentation.

    Uses a shallow copy of the original dict — safe because augmentation fully
    replaces every complex field (skill_mentions, extraction_metadata, etc.)
    rather than mutating nested structures in-place. Original job fields are
    typically flat scalars from JSON input.
    """
    out = dict(original)  # shallow copy — O(n) keys, no recursive descent
    out.update(augmentation)
    return out


def write_json(path: Union[str, Path], obj: Any, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    logger.info("Wrote JSON: %s", path)
