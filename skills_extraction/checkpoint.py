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
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .candidate_mining import CandidateSpan
from .schemas import ParsedLine

logger = logging.getLogger(__name__)


def compute_stage_fingerprint(
    stage: str,
    model: str,
    backend: str,
    prompt_texts: List[str],
    upstream_fingerprint: Optional[str] = None,
    pipeline_version: str = "",
) -> str:
    """Compute a short hash fingerprint for a pipeline stage.

    Inputs: model name, backend, system+user prompt templates, upstream
    stage fingerprint (for chaining), and pipeline version. Returns an
    8-char hex digest. A change in any input produces a different fingerprint,
    causing checkpoint invalidation on resume.
    """
    h = hashlib.sha256()
    h.update(f"stage={stage}\n".encode())
    h.update(f"model={model}\n".encode())
    h.update(f"backend={backend}\n".encode())
    h.update(f"version={pipeline_version}\n".encode())
    for pt in prompt_texts:
        h.update(pt.encode())
    if upstream_fingerprint:
        h.update(f"upstream={upstream_fingerprint}\n".encode())
    return h.hexdigest()[:16]


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
    fh, run_id: str, stage: str, total: int,
    cfg_snapshot: Optional[Dict[str, Any]] = None,
    fingerprint: Optional[str] = None,
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
    if fingerprint:
        header["fingerprint"] = fingerprint
    fh.write(json.dumps(header, ensure_ascii=False) + "\n")
    fh.flush()


_flush_counter: int = 0
_flush_interval: int = 1  # flush every N records; 1 = every record (safe default)


def set_flush_interval(n: int) -> None:
    """Set how often checkpoint writes are flushed to disk.

    n=1 (default): flush every record (safest, slightly slower).
    n>1: flush every n records (faster for large runs; at most n-1 records
    lost on crash, recoverable via resume).
    """
    global _flush_interval
    _flush_interval = max(1, n)


def append_checkpoint_record(fh, record: Dict[str, Any]) -> None:
    """Write one data record, flushing every _flush_interval records."""
    global _flush_counter
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    _flush_counter += 1
    if _flush_counter >= _flush_interval:
        fh.flush()
        _flush_counter = 0


def flush_checkpoint(fh) -> None:
    """Force-flush any buffered checkpoint data (call before footer/close)."""
    global _flush_counter
    fh.flush()
    _flush_counter = 0


def write_checkpoint_footer(fh, record_count: int) -> None:
    """Write the completion trailer (flushes any buffered records first)."""
    flush_checkpoint(fh)  # ensure all buffered data is on disk before trailer
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
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip().replace("\x00", "")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                logger.warning("Skipping corrupt line %d in %s", line_num, path)
                continue
            if obj.get("_meta"):
                meta = obj
            elif obj.get("_complete"):
                continue  # skip footer
            else:
                records.append(obj)
    if skipped:
        logger.warning("Skipped %d corrupt line(s) in %s", skipped, path)
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
    Stores only _parsed_line_id (not the full dict) when a line registry
    is available at the job-record level. Falls back to embedding the full
    _parsed_line_dict for backward compatibility with older checkpoints.
    """
    out = {}
    for k, v in m.items():
        if k == "_parsed_line":
            if v is not None:
                out["_parsed_line_id"] = v.line_id
            continue
        out[k] = v
    return out


def deserialize_mention(d: Dict[str, Any], line_registry: Optional[Dict[str, ParsedLine]] = None) -> Dict[str, Any]:
    """
    Reconstruct a mention dict from checkpoint data.

    If *line_registry* is provided, looks up _parsed_line by _parsed_line_id.
    Otherwise falls back to inline _parsed_line_dict (backward compat with
    older checkpoints that embedded the full dict per mention).
    """
    out = {}
    pl: Optional[ParsedLine] = None
    line_id: Optional[str] = None
    for k, v in d.items():
        if k == "_parsed_line_dict":
            # Legacy format: full dict embedded per mention
            pl = deserialize_parsed_line(v)
            continue
        if k == "_parsed_line_id":
            line_id = v
            continue
        out[k] = v
    # Prefer registry lookup; fall back to inline dict
    if line_id and line_registry and line_id in line_registry:
        out["_parsed_line"] = line_registry[line_id]
    elif pl is not None:
        out["_parsed_line"] = pl
    return out


def serialize_mentions_for_job(mentions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Serialize mentions for a job, deduplicating ParsedLine data.

    Returns (serialized_mentions, parsed_lines_registry) where the registry
    maps line_id -> serialized ParsedLine dict. Each mention stores only
    _parsed_line_id. Store both in the stage-1 checkpoint record.
    """
    registry: Dict[str, Dict[str, Any]] = {}
    serialized: List[Dict[str, Any]] = []
    for m in mentions:
        pl = m.get("_parsed_line")
        if pl is not None and pl.line_id not in registry:
            registry[pl.line_id] = serialize_parsed_line(pl)
        serialized.append(serialize_mention(m))
    return serialized, registry


def deserialize_mentions_for_job(
    mention_dicts: List[Dict[str, Any]],
    lines_registry_raw: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Deserialize mentions for a job, reconstituting ParsedLine from registry.

    *lines_registry_raw* maps line_id -> serialized ParsedLine dict.
    If absent (older checkpoint), falls back to per-mention _parsed_line_dict.
    """
    pl_cache: Dict[str, ParsedLine] = {}
    if lines_registry_raw:
        for lid, d in lines_registry_raw.items():
            pl_cache[lid] = deserialize_parsed_line(d)
    return [deserialize_mention(md, pl_cache if pl_cache else None) for md in mention_dicts]
