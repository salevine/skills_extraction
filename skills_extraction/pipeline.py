"""
End-to-end pipeline: ingest → preprocess → section → quality → boilerplate →
candidates → LLM extract → verify → confidence → augment records.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .boilerplate import label_parsed_lines
from .candidate_mining import CandidateSpan, mine_all_candidates
from .confidence import compute_final_confidence, needs_verifier
from .config import PipelineConfig
from .exporters import (
    write_augmented_jobs,
    write_frequency_report,
    write_low_confidence_review,
    write_mentions_csv,
    write_mentions_jsonl,
    write_quality_report,
)
from .io_utils import augment_job_record, stable_job_key
from .llm_extractor import batch_lines, extract_mentions_for_batch
from .llm_verifier import verify_mention
from .preprocessing import extract_description_fields, preprocess_description
from .quality import assess_quality
from .run_stats import RunStats
from .schemas import ExtractionMetadata, QualityStatus, SkillMention
from .sectioning import segment_lines, split_inline_section_headers

logger = logging.getLogger(__name__)


def _matching_rules(
    candidates: List[CandidateSpan], line_id: str, g0: int, g1: int
) -> List[str]:
    rules: List[str] = []
    for c in candidates:
        if c.line_id != line_id:
            continue
        if (c.char_start <= g0 < c.char_end) or (c.char_start < g1 <= c.char_end) or (
            g0 <= c.char_start and g1 >= c.char_end
        ):
            rules.append(c.rule_source)
    return list(dict.fromkeys(rules)) if rules else ["llm_extractor"]


def process_single_job(
    job: Dict[str, Any],
    job_index: int,
    cfg: PipelineConfig,
    run_id: str,
    pipeline_started: str,
    progress_callback: Optional[Callable[[str, str, Any], None]] = None,
) -> Dict[str, Any]:
    job_key = stable_job_key(job, job_index)
    title, desc_raw = extract_description_fields(job)
    if progress_callback:
        progress_callback("start", (title or "Untitled")[:60])
    pre = preprocess_description(desc_raw)
    # Inline section breaks must be part of stored normalized text so char offsets match.
    norm = split_inline_section_headers(pre.description_normalized)

    lines = segment_lines(job_key, norm)
    label_parsed_lines(lines)

    line_texts = [pl.text for pl in lines]
    qa = assess_quality(desc_raw, norm, line_texts, cfg.quality_complete_min_score)
    candidates = mine_all_candidates(lines, norm)

    llm_lines = [
        pl
        for pl in lines
        if pl.text.strip()
        and pl.boilerplate_label not in ("likely_legal", "likely_benefits")
    ]

    skill_mentions: List[Dict[str, Any]] = []
    mention_counter = 0
    extractor_model_used = cfg.extractor_model

    skip_llm_this_job = cfg.skip_llm or (
        qa.status in (QualityStatus.TRUNCATED_OR_BROKEN, QualityStatus.MALFORMED)
        and len(norm) < 40
    )
    if skip_llm_this_job:
        if cfg.skip_llm:
            logger.info("Skipping LLM for job %s (--skip-llm)", job_key)
        else:
            logger.info("Skipping LLM for job %s (document too short / low quality)", job_key)
    elif not llm_lines:
        pass
    else:
        batches = batch_lines(llm_lines, cfg.extractor_batch_max_lines)
        raw_all: List[Dict[str, Any]] = []
        for batch_idx, batch in enumerate(batches):
            if progress_callback:
                progress_callback("extractor_batch", f"batch {batch_idx + 1}/{len(batches)}", {"batch_idx": batch_idx, "total_batches": len(batches)})
            try:
                part = extract_mentions_for_batch(cfg, batch, candidates, model=cfg.extractor_model)
                raw_all.extend(part)
            except Exception as e:
                logger.warning("Extractor batch failed (%s), trying fallback model", e)
                try:
                    part = extract_mentions_for_batch(
                        cfg, batch, candidates, model=cfg.fallback_model
                    )
                    extractor_model_used = cfg.fallback_model
                    raw_all.extend(part)
                except Exception as e2:
                    logger.error("Fallback extractor failed: %s", e2)

        for m in raw_all:
            pl = m.get("_parsed_line")
            if pl is None:
                continue
            if not m.get("is_skill", True):
                # Keep for audit with explicit false
                pass
            span = m.get("skill_span") or ""
            g0 = int(m.get("_glob_char_start", pl.char_start))
            g1 = int(m.get("_glob_char_end", pl.char_end))
            line_text = pl.text
            cs = int(m.get("_line_char_start", 0))
            ce = int(m.get("_line_char_end", len(line_text)))
            offset_valid = bool(m.get("_offset_valid")) or (line_text[cs:ce] == span if span else False)
            evidence = str(m.get("evidence", span))
            evidence_ok = evidence in line_text if evidence else False

            raw_conf = float(m.get("confidence", 0.7))
            hard_soft = str(m.get("hard_soft", "unknown")).lower()
            req_lvl = str(m.get("requirement_level", "unclear")).lower()
            rules = _matching_rules(candidates, pl.line_id, g0, g1)

            v_status = "skipped"
            v_model = ""
            v_conf: Optional[float] = None
            v_notes = ""
            use_verifier = cfg.verifier_enabled and needs_verifier(
                raw_conf,
                hard_soft,
                req_lvl,
                pl.boilerplate_label,
                cfg.verify_on_low_extractor_confidence,
                cfg.verify_on_unknown_hard_soft,
                cfg.verify_on_unclear_requirement,
                cfg.verify_on_uncertain_boilerplate,
            )
            mention_for_v = {
                "skill_span": span,
                "normalized_candidate": m.get("normalized_candidate", span),
                "hard_soft": hard_soft,
                "requirement_level": req_lvl,
                "confidence": raw_conf,
                "evidence": evidence,
            }
            if use_verifier:
                if progress_callback:
                    progress_callback("verifier", "verifying low-confidence mention", {})
                try:
                    vout = verify_mention(cfg, pl.section, line_text, mention_for_v)
                    v_model = cfg.verifier_model
                    v_notes = str(vout.get("notes", ""))
                    if vout.get("_parse_failed"):
                        v_status = "parse_failed"
                        v_conf = None
                    else:
                        v_status = (
                            "accepted"
                            if vout.get("is_valid_skill_mention", True)
                            else "rejected"
                        )
                        vc = vout.get("confidence")
                        if vc is None:
                            v_conf = raw_conf
                        else:
                            try:
                                v_conf = float(vc)
                            except (TypeError, ValueError):
                                v_conf = raw_conf
                        if vout.get("is_valid_skill_mention", True):
                            hard_soft = str(vout.get("corrected_hard_soft", hard_soft))
                            req_lvl = str(vout.get("corrected_requirement_level", req_lvl))
                        else:
                            m["is_skill"] = False
                except Exception as e:
                    v_status = "error"
                    v_notes = str(e)[:200]
                    v_model = cfg.verifier_model
                    v_conf = None

            final_c = compute_final_confidence(
                raw_model_confidence=raw_conf,
                verifier_confidence=v_conf,
                section=pl.section,
                rules_fired=rules,
                boilerplate_label=pl.boilerplate_label,
                offset_valid=offset_valid,
                evidence_substring_of_line=evidence_ok,
                verifier_status=v_status,
            )

            mention_counter += 1
            sm = SkillMention(
                mention_id=f"{job_key}_M{mention_counter:05d}",
                line_id=pl.line_id,
                skill_span=span,
                normalized_candidate=str(m.get("normalized_candidate", span)).strip(),
                is_skill=bool(m.get("is_skill", True)),
                hard_soft=hard_soft,
                requirement_level=req_lvl,
                char_start=g0,
                char_end=g1,
                evidence=evidence,
                raw_model_confidence=raw_conf,
                final_confidence=final_c,
                extractor_model=extractor_model_used,
                rules_fired=rules,
                verifier_status=v_status,
                verifier_model=v_model,
                verifier_confidence=v_conf,
                verifier_notes=v_notes,
                created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                run_id=run_id,
                section=pl.section,
                source_line=line_text,
                support_count=0,
                models_that_found_it=[],
                model_disagreements=[],
            )
            skill_mentions.append(sm.to_dict())

    completed = dt.datetime.now(dt.timezone.utc).isoformat()
    meta = ExtractionMetadata(
        run_id=run_id,
        pipeline_version=cfg.pipeline_version,
        extractor_model=extractor_model_used,
        verifier_model=cfg.verifier_model if cfg.verifier_enabled else "",
        job_key=job_key,
        started_at=pipeline_started,
        completed_at=completed,
        extra={"job_title_snapshot": title[:200]},
    )

    if progress_callback:
        progress_callback("done", f"{len(skill_mentions)} mentions", {"mentions": len(skill_mentions)})

    augmentation = {
        "description_raw": desc_raw,
        "description_normalized": norm,
        "quality_assessment": qa.to_dict(),
        "parsed_lines": [pl.to_dict() for pl in lines],
        "skill_candidates": [c.to_dict() for c in candidates],
        "skill_mentions": skill_mentions,
        "extraction_metadata": meta.to_dict(),
    }
    return augment_job_record(job, augmentation)


def run_pipeline(
    jobs: List[Dict[str, Any]],
    cfg: PipelineConfig,
    output_dir: Path,
    run_id: str,
    write_reports: bool = True,
    progress_callback: Optional[Callable[[int, int, str, str, Any], None]] = None,
    log_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Path], RunStats]:
    """
    Process all jobs and write artifacts under output_dir.
    progress_callback(job_idx, total_jobs, stage, detail, extra) is called during run.
    Returns (augmented_jobs, paths_dict, run_stats).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()

    stats = RunStats(
        run_id=run_id,
        started_at=started,
        pipeline_version=cfg.pipeline_version,
        extractor_model=cfg.extractor_model,
        verifier_model=cfg.verifier_model,
        fallback_model=cfg.fallback_model,
        ollama_base_url=cfg.ollama_base_url,
        skip_llm=cfg.skip_llm,
        batch_max_lines=cfg.extractor_batch_max_lines,
        verifier_enabled=cfg.verifier_enabled,
    )
    stats.jobs_total = len(jobs)

    def _timing_cb(model: str, elapsed: float, role: str) -> None:
        stats.record_llm(model, elapsed, role)

    original_timing = cfg.llm_timing_callback
    cfg.llm_timing_callback = _timing_cb

    augmented: List[Dict[str, Any]] = []
    for idx, job in enumerate(jobs):
        def _job_progress(s: str, d: str, e: Any = None) -> None:
            if progress_callback:
                progress_callback(idx, len(jobs), s, d, e)

        try:
            augmented.append(
                process_single_job(
                    job, idx, cfg, run_id, started, progress_callback=_job_progress
                )
            )
            stats.jobs_success += 1
        except Exception as e:
            logger.exception("Job %s failed: %s", idx, e)
            stats.jobs_failed += 1
            augmented.append(
                augment_job_record(
                    job,
                    {
                        "extraction_metadata": {
                            "run_id": run_id,
                            "pipeline_version": cfg.pipeline_version,
                            "error": str(e)[:500],
                            "job_index": idx,
                        }
                    },
                )
            )

    cfg.llm_timing_callback = original_timing

    stats.mentions_total = sum(
        len(j.get("skill_mentions") or [])
        for j in augmented
    )

    paths: Dict[str, Path] = {}
    paths["augmented_json"] = output_dir / f"SkillsExtraction_augmented_run_{run_id}.json"
    paths["mentions_jsonl"] = output_dir / f"SkillsExtraction_mentions_run_{run_id}.jsonl"
    paths["mentions_csv"] = output_dir / f"SkillsExtraction_mentions_run_{run_id}.csv"

    write_augmented_jobs(paths["augmented_json"], augmented)
    write_mentions_jsonl(paths["mentions_jsonl"], augmented)
    write_mentions_csv(paths["mentions_csv"], augmented)

    if write_reports:
        paths["quality_csv"] = output_dir / f"SkillsExtraction_quality_run_{run_id}.csv"
        paths["frequency_csv"] = output_dir / f"SkillsExtraction_skill_frequency_run_{run_id}.csv"
        paths["low_conf_json"] = output_dir / f"SkillsExtraction_low_confidence_run_{run_id}.json"
        write_quality_report(paths["quality_csv"], augmented)
        write_frequency_report(paths["frequency_csv"], augmented)
        write_low_confidence_review(paths["low_conf_json"], augmented, threshold=0.55)

    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(stats.format_for_log())
        except Exception as e:
            logger.warning("Could not append run summary to log: %s", e)

    paths["run_summary_json"] = output_dir / f"SkillsExtraction_run_summary_{run_id}.json"
    try:
        import json as _json
        paths["run_summary_json"].write_text(
            _json.dumps(stats.to_dict(), indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning("Could not write run summary JSON: %s", e)

    return augmented, paths, stats
