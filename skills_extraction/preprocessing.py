"""
Normalize line endings and whitespace for parsing while preserving raw description on the job.
Offsets for parsed structures are relative to `description_normalized`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple


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
    for k in (
        "JobDescription",
        "jobDescription",
        "job_description",
        "description_raw",
        "Description",
        "description",
        "JobDesc",
        "jobDesc",
        "posting",
        "jobPosting",
    ):
        v = job.get(k)
        if isinstance(v, str) and v.strip():
            desc = v.strip()
            break
    # Do not substitute company/location as pseudo-description (distorts quality & extraction).
    return title, desc
