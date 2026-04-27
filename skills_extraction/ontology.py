"""
Stage 6: Build a skills ontology from assembled pipeline output.

Pure aggregation — no LLM calls. Groups verified skill mentions by
normalized name, computes frequency/confidence/distribution stats,
and writes the ontology as JSON + CSV.

Can run as the final pipeline stage or standalone via --ontology-only.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import write_json
from .preprocessing import extract_description_fields

logger = logging.getLogger(__name__)

_STRIP_SUFFIXES = re.compile(
    r"\s+(skills?|proficiency|expertise|knowledge|experience|abilities|ability)$",
    re.IGNORECASE,
)
_COLLAPSE_WS = re.compile(r"\s+")


def _canonicalize(name: str) -> str:
    s = name.strip()
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = _COLLAPSE_WS.sub(" ", s)
    s = _STRIP_SUFFIXES.sub("", s)
    s = s.strip(" .,;:-/")
    return s


def _pick_display_name(canon: str, variants: set) -> str:
    if not variants:
        return canon
    for v in sorted(variants):
        if v[0].isupper() and v.lower() == canon:
            return v
    return sorted(variants)[0]


def build_ontology(
    augmented_jobs: List[Dict[str, Any]],
    run_id: str,
    min_mention_count: int = 1,
) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "variants": set(),
        "type_counts": Counter(),
        "requirement_counts": Counter(),
        "confidences": [],
        "job_ids": set(),
        "job_titles": Counter(),
    })

    for job in augmented_jobs:
        job_id = job.get("id", "")
        title = extract_description_fields(job)[0]
        for m in job.get("skill_mentions") or []:
            if not m.get("is_skill"):
                continue
            raw_span = (m.get("skill_span") or "").strip()
            normalized = (m.get("normalized_candidate") or raw_span).strip()
            if not normalized:
                continue

            canon = _canonicalize(normalized)
            if not canon:
                continue

            g = groups[canon]
            g["variants"].add(normalized)
            if raw_span and raw_span != normalized:
                g["variants"].add(raw_span)
            g["type_counts"][m.get("hard_soft", "unknown")] += 1
            g["requirement_counts"][m.get("requirement_level", "unclear")] += 1
            conf = m.get("final_confidence")
            if conf is not None:
                g["confidences"].append(float(conf))
            g["job_ids"].add(job_id)
            if title:
                g["job_titles"][title] += 1

    ontology: List[Dict[str, Any]] = []
    for canon, g in groups.items():
        mention_count = sum(g["type_counts"].values())
        if mention_count < min_mention_count:
            continue

        confs = g["confidences"]
        avg_conf = round(sum(confs) / len(confs), 4) if confs else None
        conf_min = round(min(confs), 4) if confs else None
        conf_max = round(max(confs), 4) if confs else None

        display_name = _pick_display_name(canon, g["variants"])
        display_lower = display_name.lower()
        variants_list = sorted(v for v in g["variants"] if v.lower() != display_lower)

        ontology.append({
            "canonical_skill": display_name,
            "canonical_key": canon,
            "variants": variants_list,
            "type": g["type_counts"].most_common(1)[0][0] if g["type_counts"] else "unknown",
            "type_distribution": dict(g["type_counts"]),
            "requirement_level": g["requirement_counts"].most_common(1)[0][0] if g["requirement_counts"] else "unclear",
            "requirement_distribution": dict(g["requirement_counts"]),
            "job_count": len(g["job_ids"]),
            "mention_count": mention_count,
            "avg_confidence": avg_conf,
            "confidence_range": [conf_min, conf_max],
            "common_contexts": [t for t, _ in g["job_titles"].most_common(10)],
            "run_id": run_id,
        })

    ontology.sort(key=lambda x: (-x["job_count"], -x["mention_count"]))
    return ontology


def write_ontology_json(path: Path, ontology: List[Dict[str, Any]]) -> None:
    write_json(path, ontology)
    logger.info("Wrote ontology JSON (%d skills): %s", len(ontology), path)


def write_ontology_csv(path: Path, ontology: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "canonical_skill", "canonical_key", "variants", "type",
        "type_distribution", "requirement_level", "requirement_distribution",
        "job_count", "mention_count", "avg_confidence",
        "confidence_min", "confidence_max", "common_contexts", "run_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for entry in ontology:
            cr = entry.get("confidence_range") or [None, None]
            w.writerow({
                "canonical_skill": entry["canonical_skill"],
                "canonical_key": entry["canonical_key"],
                "variants": "; ".join(entry.get("variants", [])),
                "type": entry["type"],
                "type_distribution": json.dumps(entry.get("type_distribution", {})),
                "requirement_level": entry["requirement_level"],
                "requirement_distribution": json.dumps(entry.get("requirement_distribution", {})),
                "job_count": entry["job_count"],
                "mention_count": entry["mention_count"],
                "avg_confidence": entry.get("avg_confidence", ""),
                "confidence_min": cr[0] or "",
                "confidence_max": cr[1] or "",
                "common_contexts": "; ".join(entry.get("common_contexts", [])),
                "run_id": entry["run_id"],
            })
    logger.info("Wrote ontology CSV (%d skills): %s", len(ontology), path)


def build_ontology_from_file(
    augmented_path: Path,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    min_mention_count: int = 1,
) -> List[Dict[str, Any]]:
    augmented_path = Path(augmented_path)
    if not augmented_path.exists():
        raise FileNotFoundError(f"Augmented file not found: {augmented_path}")

    jobs = json.loads(augmented_path.read_text(encoding="utf-8"))
    if isinstance(jobs, dict):
        jobs = jobs.get("jobs", jobs.get("data", [jobs]))

    if run_id is None:
        m = re.search(r"run_(\S+?)\.json", augmented_path.name)
        run_id = m.group(1) if m else "unknown"

    if output_dir is None:
        output_dir = augmented_path.parent

    ontology = build_ontology(jobs, run_id, min_mention_count=min_mention_count)

    json_path = output_dir / f"SkillsExtraction_ontology_run_{run_id}.json"
    csv_path = output_dir / f"SkillsExtraction_ontology_run_{run_id}.csv"
    write_ontology_json(json_path, ontology)
    write_ontology_csv(csv_path, ontology)

    print(f"Ontology: {len(ontology)} canonical skills from {len(jobs)} jobs")
    print(f"  JSON: {json_path.name}")
    print(f"  CSV:  {csv_path.name}")

    return ontology
