"""
Exports: augmented JSON, JSONL, CSV, and derived summary reports.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def write_augmented_jobs(path: Path, jobs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote augmented jobs: %s", path)


def write_mentions_jsonl(path: Path, jobs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for job in jobs:
            meta = job.get("extraction_metadata") or {}
            run_id = meta.get("run_id", "")
            job_id = job.get("id")
            title = (
                job.get("JobTitle")
                or job.get("title_raw")
                or job.get("title_norm")
                or job.get("title")
                or ""
            )
            for m in job.get("skill_mentions") or []:
                row = {
                    "run_id": run_id,
                    "job_id": job_id,
                    "line_id": m.get("line_id"),
                    "mention_id": m.get("mention_id"),
                    "title_raw": title,
                    "section": m.get("section", ""),
                    "source_line": m.get("source_line", ""),
                    "skill_span": m.get("skill_span"),
                    "normalized_candidate": m.get("normalized_candidate"),
                    "hard_soft": m.get("hard_soft"),
                    "requirement_level": m.get("requirement_level"),
                    "char_start": m.get("char_start"),
                    "char_end": m.get("char_end"),
                    "evidence": m.get("evidence"),
                    "raw_model_confidence": m.get("raw_model_confidence"),
                    "final_confidence": m.get("final_confidence"),
                    "extractor_model": m.get("extractor_model"),
                    "verifier_status": m.get("verifier_status"),
                    "rules_fired": m.get("rules_fired"),
                    "is_skill": m.get("is_skill"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote JSONL: %s", path)


def write_mentions_csv(path: Path, jobs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "job_id",
        "line_id",
        "mention_id",
        "title_raw",
        "section",
        "source_line",
        "skill_span",
        "normalized_candidate",
        "hard_soft",
        "requirement_level",
        "char_start",
        "char_end",
        "evidence",
        "raw_model_confidence",
        "final_confidence",
        "extractor_model",
        "verifier_status",
        "verifier_model",
        "rules_fired",
        "is_skill",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for job in jobs:
            meta = job.get("extraction_metadata") or {}
            run_id = meta.get("run_id", "")
            job_id = job.get("id")
            title = (
                job.get("JobTitle")
                or job.get("title_raw")
                or job.get("title_norm")
                or job.get("title")
                or ""
            )
            for m in job.get("skill_mentions") or []:
                r = {}
                for k in fieldnames:
                    if k in ("run_id", "job_id", "title_raw"):
                        continue
                    v = m.get(k)
                    r[k] = "" if v is None else v
                r["run_id"] = run_id
                r["job_id"] = job_id
                r["title_raw"] = title
                r["rules_fired"] = ";".join(m.get("rules_fired") or [])
                w.writerow(r)
    logger.info("Wrote CSV: %s", path)


def write_quality_report(path: Path, jobs: List[Dict[str, Any]]) -> None:
    ctr = Counter()
    for job in jobs:
        qa = job.get("quality_assessment") or {}
        ctr[qa.get("status", "unknown")] += 1
    lines = ["status,count"]
    for k, v in ctr.most_common():
        lines.append(f"{k},{v}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_frequency_report(path: Path, jobs: List[Dict[str, Any]], min_count: int = 1) -> None:
    c = Counter()
    for job in jobs:
        for m in job.get("skill_mentions") or []:
            if not m.get("is_skill"):
                continue
            key = (m.get("normalized_candidate") or m.get("skill_span") or "").strip()
            if key:
                c[key.lower()] += 1
    rows = [f"normalized_lower,count"] + [f"{k},{v}" for k, v in c.most_common() if v >= min_count]
    path.write_text("\n".join(rows), encoding="utf-8")


def write_low_confidence_review(path: Path, jobs: List[Dict[str, Any]], threshold: float = 0.55) -> None:
    out: List[Dict[str, Any]] = []
    for job in jobs:
        jid = job.get("id")
        for m in job.get("skill_mentions") or []:
            fc = float(m.get("final_confidence", 1))
            if fc < threshold or m.get("verifier_status") == "parse_failed":
                out.append({"job_id": jid, **m})
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
