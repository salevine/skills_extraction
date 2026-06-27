"""
Stage 7: Job-title skill weights.

Groups verified skill mentions by normalized job title, maps each mention
to its canonical skill (via the ontology canonicalizer), deduplicates per
job posting, and assigns weighted points (hard=1.0, soft=0.5).

Accepts an optional title normalization lookup (from the NLP pipeline's
Job_Norm.py output) to collapse raw titles into ~2K standard titles.

Output: JSON keyed by normalized title, each with job_count and a skills dict
mapping canonical skill names to {points, type, posting_count}.

Can run as the final pipeline stage or standalone via --job-title-skills-only.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import write_json
from .ontology import _canonicalize, _pick_display_name
from .preprocessing import extract_description_fields

logger = logging.getLogger(__name__)

HARD_WEIGHT = 1.0
SOFT_WEIGHT = 0.5
DEFAULT_CONFIDENCE_FLOOR = 0.6


def load_title_normalization(path: Path) -> Dict[str, str]:
    """Load raw_title -> NormalizedTitle mapping from Job_Norm.py Excel output."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas and openpyxl are required to load title normalization Excel files")
    xls = pd.ExcelFile(path)
    if "Job_Titles" in xls.sheet_names:
        sheet = "Job_Titles"
    else:
        sheet = xls.sheet_names[0]
        logger.info(
            "Sheet 'Job_Titles' not found in %s; falling back to first sheet '%s'",
            path,
            sheet,
        )
    df = pd.read_excel(xls, sheet_name=sheet)
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        raw = row.get("Raw_title")
        norm = row.get("Normalized_title")
        if isinstance(raw, str) and isinstance(norm, str) and raw.strip() and norm.strip():
            mapping[raw.strip()] = norm.strip()
    logger.info("Loaded title normalization: %d mappings from %s", len(mapping), path)
    return mapping


def build_job_title_skills(
    augmented_jobs: List[Dict[str, Any]],
    run_id: str,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    title_norm_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build job-title-to-weighted-skills mapping from augmented pipeline output.

    Returns dict keyed by normalized title with structure:
        {
            "job_count": int,
            "skills": {
                "Python": {"points": 287.0, "type": "hard", "posting_count": 287},
                ...
            }
        }
    """
    canon_to_display: Dict[str, set] = defaultdict(set)
    title_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "job_ids": set(),
        "skill_postings": defaultdict(lambda: {
            "job_ids": set(),
            "type_votes": defaultdict(int),
        }),
    })
    unmapped_titles = 0

    for job in augmented_jobs:
        job_id = job.get("id", "")
        title_raw, _ = extract_description_fields(job)
        if not title_raw:
            continue

        if title_norm_map:
            title_norm = title_norm_map.get(title_raw.strip())
            if not title_norm:
                unmapped_titles += 1
                continue
        else:
            title_norm = title_raw.strip().lower()
        if not title_norm:
            continue

        td = title_data[title_norm]
        td["job_ids"].add(job_id)

        seen_skills_this_job: set = set()

        for m in job.get("skill_mentions") or []:
            if not m.get("is_skill"):
                continue
            conf = m.get("final_confidence")
            if conf is not None and conf < confidence_floor:
                continue

            normalized = (m.get("normalized_candidate") or m.get("skill_span") or "").strip()
            if not normalized:
                continue
            canon_key = _canonicalize(normalized)
            if not canon_key:
                continue

            canon_to_display[canon_key].add(normalized)
            hard_soft = m.get("hard_soft", "unknown")

            skill_key = (canon_key, hard_soft)
            if skill_key in seen_skills_this_job:
                continue
            seen_skills_this_job.add(skill_key)

            sp = td["skill_postings"][canon_key]
            sp["job_ids"].add(job_id)
            sp["type_votes"][hard_soft] += 1

    result: Dict[str, Any] = {}
    for title_norm in sorted(title_data):
        td = title_data[title_norm]
        job_count = len(td["job_ids"])
        skills: Dict[str, Any] = {}

        for canon_key, sp in td["skill_postings"].items():
            posting_count = len(sp["job_ids"])
            majority_type = max(sp["type_votes"], key=sp["type_votes"].get)
            weight = HARD_WEIGHT if majority_type == "hard" else SOFT_WEIGHT
            points = round(posting_count * weight, 1)
            display_name = _pick_display_name(canon_key, canon_to_display.get(canon_key, set()))

            skills[display_name] = {
                "points": points,
                "type": majority_type,
                "posting_count": posting_count,
            }

        skills = dict(sorted(skills.items(), key=lambda x: -x[1]["points"]))
        result[title_norm] = {
            "job_count": job_count,
            "skills": skills,
        }

    if unmapped_titles:
        logger.warning("Skipped %d jobs with titles not in normalization map", unmapped_titles)

    result = dict(sorted(result.items(), key=lambda x: -x[1]["job_count"]))
    return result


def write_job_title_skills_json(path: Path, data: Dict[str, Any]) -> None:
    write_json(path, data)
    title_count = len(data)
    skill_count = sum(len(v["skills"]) for v in data.values())
    logger.info(
        "Wrote job-title skills JSON (%d titles, %d skill entries): %s",
        title_count, skill_count, path,
    )


def build_job_title_skills_from_file(
    augmented_path: Path,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    title_norm_path: Optional[Path] = None,
) -> Dict[str, Any]:
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

    title_norm_map = None
    if title_norm_path:
        title_norm_map = load_title_normalization(Path(title_norm_path))

    data = build_job_title_skills(jobs, run_id, confidence_floor=confidence_floor, title_norm_map=title_norm_map)

    json_path = output_dir / f"SkillsExtraction_job_title_skills_run_{run_id}.json"
    write_job_title_skills_json(json_path, data)

    title_count = len(data)
    skill_count = sum(len(v["skills"]) for v in data.values())
    job_count = sum(v["job_count"] for v in data.values())
    print(f"Job-title skills: {title_count} titles, {skill_count} skill entries from {job_count} postings")
    print(f"  Confidence floor: {confidence_floor}")
    if title_norm_path:
        print(f"  Title normalization: {title_norm_path}")
    print(f"  JSON: {json_path.name}")

    return data
