"""
Normalize line endings and whitespace for parsing while preserving raw description on the job.
Offsets for parsed structures are relative to `description_normalized`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedDescription:
    description_raw: str
    description_normalized: str


_WS_COLLAPSE = re.compile(r"[ \t]+")
_LEADING_LINE_WS = re.compile(r"^[ \t]+", re.MULTILINE)


def preprocess_description(raw: Optional[str]) -> PreprocessedDescription:
    if not raw or not isinstance(raw, str):
        return PreprocessedDescription(description_raw="", description_normalized="")
    description_raw = raw
    # Standardize line endings only; then trim trailing spaces per line minimally
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse horizontal whitespace within lines (tabs/spaces) — single space
    lines = []
    for line in s.split("\n"):
        line = _WS_COLLAPSE.sub(" ", line).strip()
        lines.append(line)
    normalized = "\n".join(lines)
    return PreprocessedDescription(
        description_raw=description_raw,
        description_normalized=normalized,
    )


def extract_description_fields(job: dict) -> Tuple[str, str]:
    """Return (title, description_raw) from heterogeneous job dict."""
    title = ""
    for k in (
        "JobTitle",
        "jobTitle",
        "job_title",
        "title_raw",
        "title_norm",
        "Title",
        "title",
        "position",
        "PositionTitle",
    ):
        v = job.get(k)
        if isinstance(v, str) and v.strip():
            title = v.strip()
            break
    desc = ""
    # `job_description` (snake_case) is DEMOTED to last resort. In some exports
    # (e.g. cs_jobs_export, where job_description_source == "onet") that field holds a
    # standardized O*NET occupational task statement, not the real ad: ~380 chars,
    # shared across many jobs, almost no soft-skill language. Extracting from it
    # produced a ~99.7% "hard" skew (run launch_20260627_114046). The scraped ad lives
    # in `description_raw`, so it is tried ahead of `job_description`.
    # The camelCase `JobDescription`/`jobDescription` stay first: in other datasets
    # those carry the real posting and have not shown the O*NET contamination — only
    # the snake_case key is known to be poisoned. Order is load-bearing; do not re-sort
    # (see test_description_priority).
    for k in (
        "JobDescription",
        "jobDescription",
        "description_raw",
        "Description",
        "description",
        "JobDesc",
        "jobDesc",
        "posting",
        "jobPosting",
        "job_description",
    ):
        v = job.get(k)
        if isinstance(v, str) and v.strip():
            desc = v.strip()
            if k == "job_description":
                # Reached only when no scraped/real-ad field was present. On datasets
                # like cs_jobs_export this means falling back to O*NET text, which
                # reproduces the hard-skew bug — make that observable.
                logger.warning(
                    "extract_description_fields fell back to 'job_description' "
                    "(may be O*NET-sourced, not the real posting) for job id=%s",
                    job.get("id"),
                )
            break
    # Do not substitute company/location as pseudo-description (distorts quality & extraction).
    return title, desc
