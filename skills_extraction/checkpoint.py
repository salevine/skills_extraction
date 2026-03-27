"""
Checkpoint I/O for stage-first pipeline: incremental JSONL write/read,
serialize/deserialize ParsedLine & CandidateSpan, resume detection.

Checkpoint format (one JSON object per line):
  Line 1:    {"_meta": true, "run_id": "...", "stage": "...", "total_jobs": N, "started_at": "...", ...}
  Lines 2-N: {per-job or per-mention record}
  Last line: {"_complete": true, "completed_at": "...", "record_count": N}
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .candidate_mining import CandidateSpan
from .schemas import ParsedLine

logger = logging.getLogger(__name__)


def checkpoint_path(output_dir: Path, run_id: str, stage: str) -> Path:
    """Return path: output_dir/checkpoints/{run_id}_{stage}.jsonl"""
    return output_dir / "checkpoints" / f"{run_id}_{stage}.jsonl"


def checkpoint_complete(path: Path) -> bool:
    """Check whether a checkpoint file has a _complete trailer line."""
    if not path.exists():
        return False
    try:
        last_line = ""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if not last_line:
            return False
        obj = json.loads(last_line)
        return bool(obj.get("_complete"))
    except Exception:
        return False


def count_checkpoint_records(path: Path) -> int:
    """Count data records in a checkpoint (excludes _meta header and _complete footer)."""
    if not path.exists():
        return 0
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_meta") or obj.get("_complete"):
                    continue
                count += 1
    except Exception:
        return 0
    return count


def write_checkpoint_header(
    fh, run_id: str, stage: str, total: int, cfg_snapshot: Optional[Dict[str, Any]] = None
) -> None:
    """Write the metadata header line."""
    header = {
        "_meta": True,
        "run_id": run_id,
        "stage": stage,
        "total_jobs": total,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if cfg_snapshot:
        header["config"] = cfg_snapshot
    fh.write(json.dumps(header, ensure_ascii=False) + "\n")
    fh.flush()


def append_checkpoint_record(fh, record: Dict[str, Any]) -> None:
    """Write one data record and flush immediately."""
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()


def write_checkpoint_footer(fh, record_count: int) -> None:
    """Write the completion trailer."""
    footer = {
        "_complete": True,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "record_count": record_count,
    }
    fh.write(json.dumps(footer, ensure_ascii=False) + "\n")
    fh.flush()


def load_checkpoint(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load a checkpoint file.
    Returns (meta_dict, list_of_data_records).
    Raises FileNotFoundError if the file doesn't exist.
    """
    meta: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                meta = obj
            elif obj.get("_complete"):
                continue  # skip footer
            else:
                records.append(obj)
    return meta, records


# ---------------------------------------------------------------------------
# Serialization helpers for ParsedLine and CandidateSpan
# ---------------------------------------------------------------------------

def serialize_parsed_line(pl: ParsedLine) -> Dict[str, Any]:
    """Convert a ParsedLine to a plain dict for checkpoint storage."""
    return pl.to_dict()


def deserialize_parsed_line(d: Dict[str, Any]) -> ParsedLine:
    """Reconstruct a ParsedLine from a checkpoint dict."""
    return ParsedLine(**d)


def serialize_candidate(c: CandidateSpan) -> Dict[str, Any]:
    """Convert a CandidateSpan to a plain dict for checkpoint storage."""
    return c.to_dict()


def deserialize_candidate(d: Dict[str, Any]) -> CandidateSpan:
    """Reconstruct a CandidateSpan from a checkpoint dict."""
    return CandidateSpan(**d)


def serialize_mention(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize a raw mention dict for checkpoint storage.
    Converts _parsed_line (ParsedLine object) to _parsed_line_dict + _parsed_line_id.
    """
    out = {}
    for k, v in m.items():
        if k == "_parsed_line":
            if v is not None:
                out["_parsed_line_dict"] = serialize_parsed_line(v)
                out["_parsed_line_id"] = v.line_id
            continue
        out[k] = v
    return out


def deserialize_mention(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruct a mention dict from checkpoint data.
    Converts _parsed_line_dict back to _parsed_line (ParsedLine object).
    """
    out = {}
    for k, v in d.items():
        if k == "_parsed_line_dict":
            out["_parsed_line"] = deserialize_parsed_line(v)
            continue
        if k == "_parsed_line_id":
            continue  # consumed with _parsed_line_dict
        out[k] = v
    return out
