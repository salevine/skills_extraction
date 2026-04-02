"""
End-to-end pipeline: ingest -> preprocess -> section -> quality -> boilerplate ->
candidates -> LLM extract -> verify -> confidence -> augment records.

Stage-first architecture: all jobs pass through each stage before advancing to the
next, minimizing model swaps on the Ollama server (1 swap total instead of ~N*4).
"""

from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .boilerplate import label_parsed_lines
from .candidate_mining import CandidateSpan, mine_all_candidates
from .checkpoint import (
    append_checkpoint_record,
    checkpoint_complete,
    checkpoint_path,
    count_checkpoint_records,
    deserialize_candidate,
    deserialize_mention,
    deserialize_parsed_line,
    load_checkpoint,
    serialize_candidate,
    serialize_mention,
    serialize_parsed_line,
    write_checkpoint_footer,
    write_checkpoint_header,
)
from .confidence import compute_final_confidence
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
from .llm_hardsoft_classifier import classify_hard_soft
from .llm_requirement_classifier import classify_requirement_level
from .llm_verifier import verify_mention
from .preprocessing import extract_description_fields, preprocess_description
from .quality import assess_quality
from .run_stats import RunStats
from .schemas import ExtractionMetadata, ParsedLine, QualityStatus, SkillMention
from .sectioning import segment_lines, split_inline_section_headers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matching-rules helper (unchanged from original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Windowed parallel execution helper (vLLM only)
# ---------------------------------------------------------------------------

def _run_windowed(
    total: int,
    start_idx: int,
    window_size: int,
    process_fn: Callable[[int], Dict[str, Any]],
    progress_fn: Optional[Callable[[int, int], None]],
    fh,
    records: List[Dict[str, Any]],
) -> None:
    """
    Process items in windows of *window_size* using a thread pool.

    Within each window, all items are submitted concurrently. After the window
    completes, results are sorted by original index and written to the
    checkpoint file in order (main thread only). Progress callbacks also fire
    from the main thread during the ordered-write phase.

    Crash mid-window loses at most *window_size* items.
    """
    idx = start_idx
    while idx < total:
        window_end = min(idx + window_size, total)
        futures = {}
        with ThreadPoolExecutor(max_workers=window_size) as pool:
            for i in range(idx, window_end):
                futures[i] = pool.submit(process_fn, i)

        # Collect results in original order
        for i in range(idx, window_end):
            record = futures[i].result()  # re-raises worker exceptions
            append_checkpoint_record(fh, record)
            records.append(record)
            if progress_fn:
                progress_fn(i, total)

        idx = window_end


# ---------------------------------------------------------------------------
# Per-item processing functions for concurrent vLLM execution
# ---------------------------------------------------------------------------

def _process_extract_job(
    idx: int,
    stage0_data: List[Dict[str, Any]],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """Process a single job through Stage 1 extraction (thread-safe for vLLM)."""
    s0 = stage0_data[idx]
    job_key = s0["job_key"]
    skip_llm = s0["skip_llm"]

    extractor_model_used = cfg.extractor_model
    raw_mentions: List[Dict[str, Any]] = []

    if not skip_llm:
        all_lines = [deserialize_parsed_line(d) for d in s0["parsed_lines"]]
        llm_line_set = set(s0["llm_line_indices"])
        llm_lines = [pl for pl in all_lines if pl.line_index in llm_line_set]
        candidates = [deserialize_candidate(d) for d in s0["candidates"]]

        if llm_lines:
            batches = batch_lines(llm_lines, cfg.extractor_batch_max_lines)
            for batch in batches:
                try:
                    part = extract_mentions_for_batch(
                        cfg, batch, candidates, model=cfg.extractor_model
                    )
                    raw_mentions.extend(part)
                except Exception as e:
                    logger.warning("Extractor batch failed (%s), trying fallback", e)
                    try:
                        part = extract_mentions_for_batch(
                            cfg, batch, candidates, model=cfg.fallback_model
                        )
                        extractor_model_used = cfg.fallback_model
                        raw_mentions.extend(part)
                    except Exception as e2:
                        logger.error("Fallback extractor failed: %s", e2)

    return {
        "job_index": s0["job_index"],
        "job_key": job_key,
        "extractor_model_used": extractor_model_used,
        "raw_mentions": [serialize_mention(m) for m in raw_mentions],
    }


def _process_verify_item(
    mi: int,
    mention_tasks: List[Tuple],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """Process a single mention through Stage 2 verification (thread-safe for vLLM)."""
    job_index, job_key, mention_idx, m_dict, pl = mention_tasks[mi]

    verifier_output: Dict[str, Any]
    if not cfg.verifier_enabled:
        verifier_output = {"status": "skipped"}
    else:
        span = m_dict.get("skill_span", "")
        raw_conf = float(m_dict.get("span_confidence", m_dict.get("confidence", 0.7)))
        evidence = str(m_dict.get("evidence", span))
        mention_for_v = {
            "skill_span": span,
            "normalized_candidate": m_dict.get("normalized_candidate", span),
            "span_confidence": raw_conf,
            "evidence": evidence,
            "line_id": pl.line_id,
            "section": pl.section,
        }
        try:
            vout = verify_mention(cfg, pl.section, pl.text, mention_for_v)
            v_model = cfg.verifier_model
            v_notes = str(vout.get("notes", ""))
            if vout.get("_parse_failed"):
                verifier_output = {
                    "status": "parse_failed",
                    "model": v_model,
                    "is_skill": True,
                    "confidence": None,
                    "evidence": str(vout.get("evidence", evidence)),
                    "notes": v_notes,
                }
            else:
                is_skill = bool(vout.get("is_skill", True))
                vc = vout.get("confidence")
                if vc is None:
                    v_conf = raw_conf
                else:
                    try:
                        v_conf = float(vc)
                    except (TypeError, ValueError):
                        v_conf = raw_conf
                verifier_output = {
                    "status": "accepted" if is_skill else "rejected",
                    "model": v_model,
                    "is_skill": is_skill,
                    "confidence": v_conf,
                    "evidence": str(vout.get("evidence", evidence)),
                    "notes": v_notes,
                }
        except Exception as e:
            verifier_output = {
                "status": "error",
                "model": cfg.verifier_model,
                "is_skill": True,
                "confidence": None,
                "evidence": "",
                "notes": str(e)[:200],
            }

    return {
        "job_index": job_index,
        "job_key": job_key,
        "mention_idx": mention_idx,
        "verifier_output": verifier_output,
    }


def _process_requirement_item(
    mi: int,
    mention_tasks: List[Tuple],
    stage2_lookup: Dict[Tuple, Dict[str, Any]],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """Process a single mention through Stage 3 requirement classification (thread-safe for vLLM)."""
    job_index, job_key, mention_idx, m_dict, pl = mention_tasks[mi]
    vout = stage2_lookup.get((job_index, mention_idx), {"status": "skipped"})
    is_skill = _mention_is_skill(vout)

    requirement_output: Dict[str, Any]
    if not is_skill or not cfg.requirement_classifier_enabled:
        requirement_output = {"status": "skipped"}
    else:
        span = m_dict.get("skill_span", "")
        raw_conf = float(m_dict.get("span_confidence", m_dict.get("confidence", 0.7)))
        v_conf = vout.get("confidence")
        mention_for_c = {
            "skill_span": span,
            "normalized_candidate": m_dict.get("normalized_candidate", span),
            "is_skill": is_skill,
            "span_confidence": raw_conf,
            "skill_verifier_confidence": v_conf,
            "evidence": str(m_dict.get("evidence", span)),
        }
        try:
            req_out = classify_requirement_level(cfg, pl.section, pl.text, mention_for_c)
            req_model = cfg.requirement_model
            req_notes = str(req_out.get("notes", ""))
            if req_out.get("_parse_failed"):
                requirement_output = {
                    "status": "parse_failed",
                    "model": req_model,
                    "requirement_level": "unclear",
                    "confidence": None,
                    "evidence": str(req_out.get("evidence", "")),
                    "notes": req_notes,
                }
            else:
                req_lvl = str(req_out.get("requirement_level", "unclear")).lower()
                rc = req_out.get("confidence")
                req_conf = raw_conf if rc is None else float(rc)
                requirement_output = {
                    "status": "completed",
                    "model": req_model,
                    "requirement_level": req_lvl,
                    "confidence": req_conf,
                    "evidence": str(req_out.get("evidence", "")),
                    "notes": req_notes,
                }
        except Exception as e:
            requirement_output = {
                "status": "error",
                "model": cfg.requirement_model,
                "requirement_level": "unclear",
                "confidence": None,
                "evidence": "",
                "notes": str(e)[:200],
            }

    return {
        "job_index": job_index,
        "job_key": job_key,
        "mention_idx": mention_idx,
        "requirement_output": requirement_output,
    }


def _process_hardsoft_item(
    mi: int,
    mention_tasks: List[Tuple],
    stage2_lookup: Dict[Tuple, Dict[str, Any]],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """Process a single mention through Stage 4 hard/soft classification (thread-safe for vLLM)."""
    job_index, job_key, mention_idx, m_dict, pl = mention_tasks[mi]
    vout = stage2_lookup.get((job_index, mention_idx), {"status": "skipped"})
    is_skill = _mention_is_skill(vout)

    hardsoft_output: Dict[str, Any]
    if not is_skill or not cfg.hardsoft_classifier_enabled:
        hardsoft_output = {"status": "skipped"}
    else:
        span = m_dict.get("skill_span", "")
        raw_conf = float(m_dict.get("span_confidence", m_dict.get("confidence", 0.7)))
        v_conf = vout.get("confidence")
        mention_for_c = {
            "skill_span": span,
            "normalized_candidate": m_dict.get("normalized_candidate", span),
            "is_skill": is_skill,
            "span_confidence": raw_conf,
            "skill_verifier_confidence": v_conf,
            "evidence": str(m_dict.get("evidence", span)),
        }
        try:
            hs_out = classify_hard_soft(cfg, pl.section, pl.text, mention_for_c)
            hs_model = cfg.hardsoft_model
            hs_notes = str(hs_out.get("notes", ""))
            if hs_out.get("_parse_failed"):
                hardsoft_output = {
                    "status": "parse_failed",
                    "model": hs_model,
                    "hard_soft": "unknown",
                    "confidence": None,
                    "evidence": str(hs_out.get("evidence", "")),
                    "notes": hs_notes,
                }
            else:
                hard_soft = str(hs_out.get("hard_soft", "unknown")).lower()
                hc = hs_out.get("confidence")
                hs_conf = raw_conf if hc is None else float(hc)
                hardsoft_output = {
                    "status": "completed",
                    "model": hs_model,
                    "hard_soft": hard_soft,
                    "confidence": hs_conf,
                    "evidence": str(hs_out.get("evidence", "")),
                    "notes": hs_notes,
                }
        except Exception as e:
            hardsoft_output = {
                "status": "error",
                "model": cfg.hardsoft_model,
                "hard_soft": "unknown",
                "confidence": None,
                "evidence": "",
                "notes": str(e)[:200],
            }

    return {
        "job_index": job_index,
        "job_key": job_key,
        "mention_idx": mention_idx,
        "hardsoft_output": hardsoft_output,
    }


# ---------------------------------------------------------------------------
# Stage 0: Preprocess all jobs (no LLM)
# ---------------------------------------------------------------------------

def _run_stage0_preprocess(
    jobs: List[Dict[str, Any]],
    cfg: PipelineConfig,
    run_id: str,
    ckpt_dir: Path,
    progress_cb: Optional[Callable] = None,
    start_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Preprocess all jobs: normalize, section, label, assess quality, mine candidates."""
    path = checkpoint_path(ckpt_dir.parent, run_id, "stage0_preprocessed")
    records: List[Dict[str, Any]] = []

    # Load already-written records when resuming
    if start_idx > 0 and path.exists():
        _, records = load_checkpoint(path)
        records = records[:start_idx]

    mode = "a" if start_idx > 0 else "w"
    with open(path, mode, encoding="utf-8") as fh:
        if start_idx == 0:
            write_checkpoint_header(fh, run_id, "stage0_preprocessed", len(jobs))

        for idx in range(start_idx, len(jobs)):
            job = jobs[idx]
            job_key = stable_job_key(job, idx)
            title, desc_raw = extract_description_fields(job)

            if progress_cb:
                progress_cb(idx, len(jobs), "stage0_preprocess",
                            (title or "Untitled")[:60],
                            {"stage": "stage0_preprocess"})

            pre = preprocess_description(desc_raw)
            norm = split_inline_section_headers(pre.description_normalized)
            lines = segment_lines(job_key, norm)
            label_parsed_lines(lines)
            line_texts = [pl.text for pl in lines]
            qa = assess_quality(desc_raw, norm, line_texts, cfg.quality_complete_min_score)
            candidates = mine_all_candidates(lines, norm)

            llm_lines = [
                pl for pl in lines
                if pl.text.strip()
                and pl.boilerplate_label not in ("likely_legal", "likely_benefits")
            ]
            llm_line_indices = [pl.line_index for pl in llm_lines]

            skip_llm = cfg.skip_llm or (
                qa.status in (QualityStatus.TRUNCATED_OR_BROKEN, QualityStatus.MALFORMED)
                and len(norm) < 40
            )

            record = {
                "job_index": idx,
                "job_key": job_key,
                "title": title,
                "desc_raw": desc_raw,
                "desc_normalized": norm,
                "quality_assessment": qa.to_dict(),
                "parsed_lines": [serialize_parsed_line(pl) for pl in lines],
                "candidates": [serialize_candidate(c) for c in candidates],
                "skip_llm": skip_llm,
                "llm_line_indices": llm_line_indices,
            }
            append_checkpoint_record(fh, record)
            records.append(record)

        write_checkpoint_footer(fh, len(records))
    return records


# ---------------------------------------------------------------------------
# Stage 1: Extract all jobs (qwen3:14b stays loaded)
# ---------------------------------------------------------------------------

def _run_stage1_extract(
    stage0_data: List[Dict[str, Any]],
    cfg: PipelineConfig,
    run_id: str,
    ckpt_dir: Path,
    progress_cb: Optional[Callable] = None,
    stats: Optional[RunStats] = None,
    start_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Run LLM extraction on all jobs. qwen3:14b stays loaded for the entire stage."""
    path = checkpoint_path(ckpt_dir.parent, run_id, "stage1_extracted")
    records: List[Dict[str, Any]] = []

    if start_idx > 0 and path.exists():
        _, records = load_checkpoint(path)
        records = records[:start_idx]

    mode = "a" if start_idx > 0 else "w"
    with open(path, mode, encoding="utf-8") as fh:
        if start_idx == 0:
            write_checkpoint_header(fh, run_id, "stage1_extracted", len(stage0_data))

        if cfg.backend == "vllm":
            def _s1_progress(i: int, total: int) -> None:
                if progress_cb:
                    progress_cb(i, total, "stage1_extract",
                                f"extracting {stage0_data[i]['job_key']}",
                                {"stage": "stage1_extract"})

            _run_windowed(
                total=len(stage0_data),
                start_idx=start_idx,
                window_size=cfg.vllm_num_endpoints,
                process_fn=lambda idx: _process_extract_job(idx, stage0_data, cfg),
                progress_fn=_s1_progress,
                fh=fh,
                records=records,
            )
        else:
            for idx in range(start_idx, len(stage0_data)):
                s0 = stage0_data[idx]
                job_key = s0["job_key"]
                skip_llm = s0["skip_llm"]

                if progress_cb:
                    progress_cb(idx, len(stage0_data), "stage1_extract",
                                f"extracting {job_key}",
                                {"stage": "stage1_extract"})

                extractor_model_used = cfg.extractor_model
                raw_mentions: List[Dict[str, Any]] = []

                if not skip_llm:
                    all_lines = [deserialize_parsed_line(d) for d in s0["parsed_lines"]]
                    llm_line_set = set(s0["llm_line_indices"])
                    llm_lines = [pl for pl in all_lines if pl.line_index in llm_line_set]
                    candidates = [deserialize_candidate(d) for d in s0["candidates"]]

                    if llm_lines:
                        batches = batch_lines(llm_lines, cfg.extractor_batch_max_lines)
                        for batch_idx, batch in enumerate(batches):
                            if progress_cb:
                                progress_cb(idx, len(stage0_data), "stage1_extract",
                                            f"batch {batch_idx+1}/{len(batches)} for {job_key}",
                                            {"stage": "stage1_extract"})
                            try:
                                part = extract_mentions_for_batch(
                                    cfg, batch, candidates, model=cfg.extractor_model
                                )
                                raw_mentions.extend(part)
                            except Exception as e:
                                logger.warning("Extractor batch failed (%s), trying fallback", e)
                                try:
                                    part = extract_mentions_for_batch(
                                        cfg, batch, candidates, model=cfg.fallback_model
                                    )
                                    extractor_model_used = cfg.fallback_model
                                    raw_mentions.extend(part)
                                except Exception as e2:
                                    logger.error("Fallback extractor failed: %s", e2)

                record = {
                    "job_index": s0["job_index"],
                    "job_key": job_key,
                    "extractor_model_used": extractor_model_used,
                    "raw_mentions": [serialize_mention(m) for m in raw_mentions],
                }
                append_checkpoint_record(fh, record)
                records.append(record)

        write_checkpoint_footer(fh, len(records))
    return records


# ---------------------------------------------------------------------------
# Stage 2: Verify all mentions (mistral-nemo:12b loads once)
# ---------------------------------------------------------------------------

def _run_stage2_verify(
    stage0_data: List[Dict[str, Any]],
    stage1_data: List[Dict[str, Any]],
    cfg: PipelineConfig,
    run_id: str,
    ckpt_dir: Path,
    progress_cb: Optional[Callable] = None,
    stats: Optional[RunStats] = None,
    start_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Verify all extracted mentions. mistral-nemo:12b loads once, stays loaded."""
    path = checkpoint_path(ckpt_dir.parent, run_id, "stage2_verified")
    records: List[Dict[str, Any]] = []

    if start_idx > 0 and path.exists():
        _, records = load_checkpoint(path)
        records = records[:start_idx]

    # Build flat list of (job_index, job_key, mention_idx, mention, parsed_line)
    mention_tasks = _build_mention_task_list(stage0_data, stage1_data)
    total_mentions = len(mention_tasks)

    mode = "a" if start_idx > 0 else "w"
    with open(path, mode, encoding="utf-8") as fh:
        if start_idx == 0:
            write_checkpoint_header(fh, run_id, "stage2_verified", total_mentions)

        if cfg.backend == "vllm":
            def _s2_progress(i: int, total: int) -> None:
                if progress_cb:
                    progress_cb(i, total, "stage2_verify",
                                f"verifying mention {i+1}/{total}",
                                {"stage": "stage2_verify"})

            _run_windowed(
                total=total_mentions,
                start_idx=start_idx,
                window_size=cfg.vllm_num_endpoints,
                process_fn=lambda mi: _process_verify_item(mi, mention_tasks, cfg),
                progress_fn=_s2_progress,
                fh=fh,
                records=records,
            )
        else:
            for mi in range(start_idx, total_mentions):
                job_index, job_key, mention_idx, m_dict, pl = mention_tasks[mi]

                if progress_cb:
                    progress_cb(mi, total_mentions, "stage2_verify",
                                f"verifying mention {mi+1}/{total_mentions}",
                                {"stage": "stage2_verify"})

                verifier_output: Dict[str, Any]
                if not cfg.verifier_enabled:
                    verifier_output = {"status": "skipped"}
                else:
                    span = m_dict.get("skill_span", "")
                    raw_conf = float(m_dict.get("span_confidence", m_dict.get("confidence", 0.7)))
                    evidence = str(m_dict.get("evidence", span))
                    mention_for_v = {
                        "skill_span": span,
                        "normalized_candidate": m_dict.get("normalized_candidate", span),
                        "span_confidence": raw_conf,
                        "evidence": evidence,
                        "line_id": pl.line_id,
                        "section": pl.section,
                    }
                    try:
                        vout = verify_mention(cfg, pl.section, pl.text, mention_for_v)
                        v_model = cfg.verifier_model
                        v_notes = str(vout.get("notes", ""))
                        if vout.get("_parse_failed"):
                            verifier_output = {
                                "status": "parse_failed",
                                "model": v_model,
                                "is_skill": True,
                                "confidence": None,
                                "evidence": str(vout.get("evidence", evidence)),
                                "notes": v_notes,
                            }
                        else:
                            is_skill = bool(vout.get("is_skill", True))
                            vc = vout.get("confidence")
                            if vc is None:
                                v_conf = raw_conf
                            else:
                                try:
                                    v_conf = float(vc)
                                except (TypeError, ValueError):
                                    v_conf = raw_conf
                            verifier_output = {
                                "status": "accepted" if is_skill else "rejected",
                                "model": v_model,
                                "is_skill": is_skill,
                                "confidence": v_conf,
                                "evidence": str(vout.get("evidence", evidence)),
                                "notes": v_notes,
                            }
                    except Exception as e:
                        verifier_output = {
                            "status": "error",
                            "model": cfg.verifier_model,
                            "is_skill": True,
                            "confidence": None,
                            "evidence": "",
                            "notes": str(e)[:200],
                        }

                record = {
                    "job_index": job_index,
                    "job_key": job_key,
                    "mention_idx": mention_idx,
                    "verifier_output": verifier_output,
                }
                append_checkpoint_record(fh, record)
                records.append(record)

        write_checkpoint_footer(fh, len(records))
    return records


# ---------------------------------------------------------------------------
# Stage 3: Classify requirement level (mistral-nemo:12b still hot)
# ---------------------------------------------------------------------------

def _run_stage3_requirement(
    stage0_data: List[Dict[str, Any]],
    stage1_data: List[Dict[str, Any]],
    stage2_data: List[Dict[str, Any]],
    cfg: PipelineConfig,
    run_id: str,
    ckpt_dir: Path,
    progress_cb: Optional[Callable] = None,
    stats: Optional[RunStats] = None,
    start_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Classify requirement level for accepted mentions."""
    path = checkpoint_path(ckpt_dir.parent, run_id, "stage3_requirement")
    records: List[Dict[str, Any]] = []

    if start_idx > 0 and path.exists():
        _, records = load_checkpoint(path)
        records = records[:start_idx]

    mention_tasks = _build_mention_task_list(stage0_data, stage1_data)
    # Build verifier lookup: (job_index, mention_idx) -> verifier_output
    v_lookup = {(r["job_index"], r["mention_idx"]): r["verifier_output"] for r in stage2_data}
    total_mentions = len(mention_tasks)

    mode = "a" if start_idx > 0 else "w"
    with open(path, mode, encoding="utf-8") as fh:
        if start_idx == 0:
            write_checkpoint_header(fh, run_id, "stage3_requirement", total_mentions)

        if cfg.backend == "vllm":
            def _s3_progress(i: int, total: int) -> None:
                if progress_cb:
                    progress_cb(i, total, "stage3_requirement",
                                f"classifying requirement {i+1}/{total}",
                                {"stage": "stage3_requirement"})

            _run_windowed(
                total=total_mentions,
                start_idx=start_idx,
                window_size=cfg.vllm_num_endpoints,
                process_fn=lambda mi: _process_requirement_item(mi, mention_tasks, v_lookup, cfg),
                progress_fn=_s3_progress,
                fh=fh,
                records=records,
            )
        else:
            for mi in range(start_idx, total_mentions):
                job_index, job_key, mention_idx, m_dict, pl = mention_tasks[mi]
                vout = v_lookup.get((job_index, mention_idx), {"status": "skipped"})
                is_skill = _mention_is_skill(vout)

                if progress_cb:
                    progress_cb(mi, total_mentions, "stage3_requirement",
                                f"classifying requirement {mi+1}/{total_mentions}",
                                {"stage": "stage3_requirement"})

                requirement_output: Dict[str, Any]
                if not is_skill or not cfg.requirement_classifier_enabled:
                    requirement_output = {"status": "skipped"}
                else:
                    span = m_dict.get("skill_span", "")
                    raw_conf = float(m_dict.get("span_confidence", m_dict.get("confidence", 0.7)))
                    v_conf = vout.get("confidence")
                    mention_for_c = {
                        "skill_span": span,
                        "normalized_candidate": m_dict.get("normalized_candidate", span),
                        "is_skill": is_skill,
                        "span_confidence": raw_conf,
                        "skill_verifier_confidence": v_conf,
                        "evidence": str(m_dict.get("evidence", span)),
                    }
                    try:
                        req_out = classify_requirement_level(cfg, pl.section, pl.text, mention_for_c)
                        req_model = cfg.requirement_model
                        req_notes = str(req_out.get("notes", ""))
                        if req_out.get("_parse_failed"):
                            requirement_output = {
                                "status": "parse_failed",
                                "model": req_model,
                                "requirement_level": "unclear",
                                "confidence": None,
                                "evidence": str(req_out.get("evidence", "")),
                                "notes": req_notes,
                            }
                        else:
                            req_lvl = str(req_out.get("requirement_level", "unclear")).lower()
                            rc = req_out.get("confidence")
                            req_conf = raw_conf if rc is None else float(rc)
                            requirement_output = {
                                "status": "completed",
                                "model": req_model,
                                "requirement_level": req_lvl,
                                "confidence": req_conf,
                                "evidence": str(req_out.get("evidence", "")),
                                "notes": req_notes,
                            }
                    except Exception as e:
                        requirement_output = {
                            "status": "error",
                            "model": cfg.requirement_model,
                            "requirement_level": "unclear",
                            "confidence": None,
                            "evidence": "",
                            "notes": str(e)[:200],
                        }

                record = {
                    "job_index": job_index,
                    "job_key": job_key,
                    "mention_idx": mention_idx,
                    "requirement_output": requirement_output,
                }
                append_checkpoint_record(fh, record)
                records.append(record)

        write_checkpoint_footer(fh, len(records))
    return records


# ---------------------------------------------------------------------------
# Stage 4: Classify hard/soft (mistral-nemo:12b still hot)
# ---------------------------------------------------------------------------

def _run_stage4_hardsoft(
    stage0_data: List[Dict[str, Any]],
    stage1_data: List[Dict[str, Any]],
    stage2_data: List[Dict[str, Any]],
    cfg: PipelineConfig,
    run_id: str,
    ckpt_dir: Path,
    progress_cb: Optional[Callable] = None,
    stats: Optional[RunStats] = None,
    start_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Classify hard/soft for accepted mentions."""
    path = checkpoint_path(ckpt_dir.parent, run_id, "stage4_hardsoft")
    records: List[Dict[str, Any]] = []

    if start_idx > 0 and path.exists():
        _, records = load_checkpoint(path)
        records = records[:start_idx]

    mention_tasks = _build_mention_task_list(stage0_data, stage1_data)
    v_lookup = {(r["job_index"], r["mention_idx"]): r["verifier_output"] for r in stage2_data}
    total_mentions = len(mention_tasks)

    mode = "a" if start_idx > 0 else "w"
    with open(path, mode, encoding="utf-8") as fh:
        if start_idx == 0:
            write_checkpoint_header(fh, run_id, "stage4_hardsoft", total_mentions)

        if cfg.backend == "vllm":
            def _s4_progress(i: int, total: int) -> None:
                if progress_cb:
                    progress_cb(i, total, "stage4_hardsoft",
                                f"classifying hard/soft {i+1}/{total}",
                                {"stage": "stage4_hardsoft"})

            _run_windowed(
                total=total_mentions,
                start_idx=start_idx,
                window_size=cfg.vllm_num_endpoints,
                process_fn=lambda mi: _process_hardsoft_item(mi, mention_tasks, v_lookup, cfg),
                progress_fn=_s4_progress,
                fh=fh,
                records=records,
            )
        else:
            for mi in range(start_idx, total_mentions):
                job_index, job_key, mention_idx, m_dict, pl = mention_tasks[mi]
                vout = v_lookup.get((job_index, mention_idx), {"status": "skipped"})
                is_skill = _mention_is_skill(vout)

                if progress_cb:
                    progress_cb(mi, total_mentions, "stage4_hardsoft",
                                f"classifying hard/soft {mi+1}/{total_mentions}",
                                {"stage": "stage4_hardsoft"})

                hardsoft_output: Dict[str, Any]
                if not is_skill or not cfg.hardsoft_classifier_enabled:
                    hardsoft_output = {"status": "skipped"}
                else:
                    span = m_dict.get("skill_span", "")
                    raw_conf = float(m_dict.get("span_confidence", m_dict.get("confidence", 0.7)))
                    v_conf = vout.get("confidence")
                    mention_for_c = {
                        "skill_span": span,
                        "normalized_candidate": m_dict.get("normalized_candidate", span),
                        "is_skill": is_skill,
                        "span_confidence": raw_conf,
                        "skill_verifier_confidence": v_conf,
                        "evidence": str(m_dict.get("evidence", span)),
                    }
                    try:
                        hs_out = classify_hard_soft(cfg, pl.section, pl.text, mention_for_c)
                        hs_model = cfg.hardsoft_model
                        hs_notes = str(hs_out.get("notes", ""))
                        if hs_out.get("_parse_failed"):
                            hardsoft_output = {
                                "status": "parse_failed",
                                "model": hs_model,
                                "hard_soft": "unknown",
                                "confidence": None,
                                "evidence": str(hs_out.get("evidence", "")),
                                "notes": hs_notes,
                            }
                        else:
                            hard_soft = str(hs_out.get("hard_soft", "unknown")).lower()
                            hc = hs_out.get("confidence")
                            hs_conf = raw_conf if hc is None else float(hc)
                            hardsoft_output = {
                                "status": "completed",
                                "model": hs_model,
                                "hard_soft": hard_soft,
                                "confidence": hs_conf,
                                "evidence": str(hs_out.get("evidence", "")),
                                "notes": hs_notes,
                            }
                    except Exception as e:
                        hardsoft_output = {
                            "status": "error",
                            "model": cfg.hardsoft_model,
                            "hard_soft": "unknown",
                            "confidence": None,
                            "evidence": "",
                            "notes": str(e)[:200],
                        }

                record = {
                    "job_index": job_index,
                    "job_key": job_key,
                    "mention_idx": mention_idx,
                    "hardsoft_output": hardsoft_output,
                }
                append_checkpoint_record(fh, record)
                records.append(record)

        write_checkpoint_footer(fh, len(records))
    return records


# ---------------------------------------------------------------------------
# Stage 5: Assemble final output (no LLM)
# ---------------------------------------------------------------------------

def _run_stage5_assemble(
    jobs: List[Dict[str, Any]],
    stage0_data: List[Dict[str, Any]],
    stage1_data: List[Dict[str, Any]],
    stage2_data: List[Dict[str, Any]],
    stage3_data: List[Dict[str, Any]],
    stage4_data: List[Dict[str, Any]],
    cfg: PipelineConfig,
    run_id: str,
    started: str,
) -> List[Dict[str, Any]]:
    """Assemble all stage data into augmented job records (identical output to original pipeline)."""
    # Index stage2/3/4 by (job_index, mention_idx)
    v_lookup = {(r["job_index"], r["mention_idx"]): r["verifier_output"] for r in stage2_data}
    req_lookup = {(r["job_index"], r["mention_idx"]): r["requirement_output"] for r in stage3_data}
    hs_lookup = {(r["job_index"], r["mention_idx"]): r["hardsoft_output"] for r in stage4_data}

    # Index stage1 by job_index
    s1_by_job = {r["job_index"]: r for r in stage1_data}

    augmented: List[Dict[str, Any]] = []

    for s0 in stage0_data:
        job_idx = s0["job_index"]
        job_key = s0["job_key"]
        job = jobs[job_idx]
        s1 = s1_by_job.get(job_idx, {})

        lines = [deserialize_parsed_line(d) for d in s0["parsed_lines"]]
        candidates = [deserialize_candidate(d) for d in s0["candidates"]]
        extractor_model_used = s1.get("extractor_model_used", cfg.extractor_model)

        raw_mentions_ser = s1.get("raw_mentions", [])
        skill_mentions: List[Dict[str, Any]] = []
        mention_counter = 0
        stage_counters: Dict[str, int] = {
            "extractor_mentions": len(raw_mentions_ser),
            "skill_verifier_rejected": 0,
            "skill_verifier_parse_failed": 0,
            "requirement_parse_failed": 0,
            "hardsoft_parse_failed": 0,
            "stage_errors": 0,
        }

        for m_idx, m_ser in enumerate(raw_mentions_ser):
            m = deserialize_mention(m_ser)
            pl = m.get("_parsed_line")
            if pl is None:
                continue

            span = m.get("skill_span") or ""
            g0 = int(m.get("_glob_char_start", pl.char_start))
            g1 = int(m.get("_glob_char_end", pl.char_end))
            line_text = pl.text
            cs = int(m.get("_line_char_start", 0))
            ce = int(m.get("_line_char_end", len(line_text)))
            offset_valid = bool(m.get("_offset_valid")) or (
                line_text[cs:ce] == span if span else False
            )
            evidence = str(m.get("evidence", span))
            evidence_ok = evidence in line_text if evidence else False
            raw_conf = float(m.get("span_confidence", m.get("confidence", 0.7)))
            rules = _matching_rules(candidates, pl.line_id, g0, g1)

            # --- Verifier output ---
            vout = v_lookup.get((job_idx, m_idx), {"status": "skipped"})
            v_status = vout.get("status", "skipped")
            v_model = vout.get("model", "")
            v_conf = vout.get("confidence")
            v_notes = vout.get("notes", "")

            if v_status == "parse_failed":
                stage_counters["skill_verifier_parse_failed"] += 1
            if v_status == "rejected":
                stage_counters["skill_verifier_rejected"] += 1

            is_skill = _mention_is_skill(vout)

            # Build pipeline_audit.extractor
            pipeline_audit: Dict[str, Any] = {
                "extractor": {
                    "status": "completed",
                    "model": extractor_model_used,
                    "output": {
                        "skill_span": span,
                        "normalized_candidate": m.get("normalized_candidate", span),
                        "char_start": g0,
                        "char_end": g1,
                        "evidence": evidence,
                        "span_confidence": raw_conf,
                        "reason": str(m.get("reason", "")),
                    },
                },
                "skill_verifier": {"status": "skipped", "model": ""},
                "requirement_classifier": {"status": "skipped", "model": ""},
                "hardsoft_classifier": {"status": "skipped", "model": ""},
            }

            # Populate verifier audit
            if v_status != "skipped":
                pipeline_audit["skill_verifier"] = {
                    "status": v_status,
                    "model": v_model,
                }
                if v_status == "error":
                    pipeline_audit["skill_verifier"]["error"] = v_notes
                else:
                    pipeline_audit["skill_verifier"]["output"] = {
                        "is_skill": vout.get("is_skill", True),
                        "confidence": v_conf,
                        "evidence": vout.get("evidence", evidence),
                        "notes": v_notes,
                    }
                if v_status == "error":
                    stage_counters["stage_errors"] += 1

            # --- Requirement output ---
            rout = req_lookup.get((job_idx, m_idx), {"status": "skipped"})
            req_status = rout.get("status", "skipped")
            req_model = rout.get("model", "")
            req_conf = rout.get("confidence")
            req_notes = rout.get("notes", "")
            req_lvl = rout.get("requirement_level", "unclear")

            if req_status == "parse_failed":
                stage_counters["requirement_parse_failed"] += 1
            if req_status == "error":
                stage_counters["stage_errors"] += 1

            if req_status != "skipped":
                pipeline_audit["requirement_classifier"] = {
                    "status": req_status,
                    "model": req_model,
                }
                if req_status == "error":
                    pipeline_audit["requirement_classifier"]["error"] = req_notes
                else:
                    pipeline_audit["requirement_classifier"]["output"] = {
                        "requirement_level": req_lvl,
                        "confidence": req_conf,
                        "evidence": rout.get("evidence", evidence),
                        "notes": req_notes,
                    }

            # --- Hard/soft output ---
            hsout = hs_lookup.get((job_idx, m_idx), {"status": "skipped"})
            hs_status = hsout.get("status", "skipped")
            hs_model = hsout.get("model", "")
            hs_conf = hsout.get("confidence")
            hs_notes = hsout.get("notes", "")
            hard_soft = hsout.get("hard_soft", "unknown")

            if hs_status == "parse_failed":
                stage_counters["hardsoft_parse_failed"] += 1
            if hs_status == "error":
                stage_counters["stage_errors"] += 1

            if hs_status != "skipped":
                pipeline_audit["hardsoft_classifier"] = {
                    "status": hs_status,
                    "model": hs_model,
                }
                if hs_status == "error":
                    pipeline_audit["hardsoft_classifier"]["error"] = hs_notes
                else:
                    pipeline_audit["hardsoft_classifier"]["output"] = {
                        "hard_soft": hard_soft,
                        "confidence": hs_conf,
                        "evidence": hsout.get("evidence", evidence),
                        "notes": hs_notes,
                    }

            # --- Final confidence ---
            final_c = compute_final_confidence(
                raw_model_confidence=raw_conf,
                verifier_confidence=v_conf,
                requirement_confidence=req_conf,
                hardsoft_confidence=hs_conf,
                section=pl.section,
                rules_fired=rules,
                boilerplate_label=pl.boilerplate_label,
                offset_valid=offset_valid,
                evidence_substring_of_line=evidence_ok,
                verifier_status=v_status,
                requirement_status=req_status,
                hardsoft_status=hs_status,
            )

            mention_counter += 1
            sm = SkillMention(
                mention_id=f"{job_key}_M{mention_counter:05d}",
                line_id=pl.line_id,
                skill_span=span,
                normalized_candidate=str(m.get("normalized_candidate", span)).strip(),
                is_skill=is_skill,
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
                requirement_model=req_model,
                requirement_status=req_status,
                requirement_confidence=req_conf,
                requirement_notes=req_notes,
                hardsoft_model=hs_model,
                hardsoft_status=hs_status,
                hardsoft_confidence=hs_conf,
                hardsoft_notes=hs_notes,
                pipeline_audit=pipeline_audit,
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
        title = s0.get("title", "")
        meta = ExtractionMetadata(
            run_id=run_id,
            pipeline_version=cfg.pipeline_version,
            extractor_model=extractor_model_used,
            verifier_model=cfg.verifier_model if cfg.verifier_enabled else "",
            job_key=job_key,
            started_at=started,
            completed_at=completed,
            extra={
                "job_title_snapshot": title[:200],
                "requirement_model": cfg.requirement_model if cfg.requirement_classifier_enabled else "",
                "hardsoft_model": cfg.hardsoft_model if cfg.hardsoft_classifier_enabled else "",
            },
        )

        augmentation = {
            "description_raw": s0["desc_raw"],
            "description_normalized": s0["desc_normalized"],
            "quality_assessment": s0["quality_assessment"],
            "parsed_lines": s0["parsed_lines"],
            "skill_candidates": [serialize_candidate(c) for c in candidates],
            "skill_mentions": skill_mentions,
            "pipeline_stage_audit": {
                "stage_counters": stage_counters,
                "models": {
                    "extractor_model": extractor_model_used,
                    "skill_verifier_model": cfg.verifier_model if cfg.verifier_enabled else "",
                    "requirement_model": cfg.requirement_model if cfg.requirement_classifier_enabled else "",
                    "hardsoft_model": cfg.hardsoft_model if cfg.hardsoft_classifier_enabled else "",
                },
            },
            "extraction_metadata": meta.to_dict(),
        }
        augmented.append(augment_job_record(job, augmentation))

    return augmented


# ---------------------------------------------------------------------------
# Helper: build flat mention task list from stage0 + stage1
# ---------------------------------------------------------------------------

def _build_mention_task_list(
    stage0_data: List[Dict[str, Any]],
    stage1_data: List[Dict[str, Any]],
) -> List[Tuple[int, str, int, Dict[str, Any], ParsedLine]]:
    """
    Build a flat ordered list of (job_index, job_key, mention_idx, mention_dict, parsed_line)
    across all jobs, for stages 2-4 to iterate over.
    """
    s1_by_job = {r["job_index"]: r for r in stage1_data}
    tasks: List[Tuple[int, str, int, Dict[str, Any], ParsedLine]] = []
    for s0 in stage0_data:
        job_idx = s0["job_index"]
        job_key = s0["job_key"]
        s1 = s1_by_job.get(job_idx, {})
        for m_idx, m_ser in enumerate(s1.get("raw_mentions", [])):
            m = deserialize_mention(m_ser)
            pl = m.get("_parsed_line")
            if pl is not None:
                tasks.append((job_idx, job_key, m_idx, m, pl))
    return tasks


def _mention_is_skill(verifier_output: Dict[str, Any]) -> bool:
    """Determine if a mention is considered a skill based on verifier output."""
    status = verifier_output.get("status", "skipped")
    if status == "rejected":
        return False
    # For skipped, error, parse_failed, accepted: default True
    return bool(verifier_output.get("is_skill", True))


# ---------------------------------------------------------------------------
# Load-or-run stage orchestrator
# ---------------------------------------------------------------------------

def _load_or_run_stage(
    stage_name: str,
    ckpt_dir: Path,
    run_id: str,
    resume: bool,
    total_expected: int,
    run_fn: Callable,
    progress_cb: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    Check if a checkpoint exists for this stage:
    - Complete checkpoint + resume=True -> load from file
    - Partial checkpoint + resume=True -> resume from last record
    - Otherwise -> run from scratch
    """
    path = checkpoint_path(ckpt_dir.parent, run_id, stage_name)

    if resume and path.exists():
        if checkpoint_complete(path):
            meta, records = load_checkpoint(path)
            ckpt_total = meta.get("total_jobs", 0)
            if ckpt_total != total_expected:
                logger.warning(
                    "Checkpoint %s has %d records but expected %d jobs; ignoring checkpoint",
                    stage_name, ckpt_total, total_expected,
                )
            else:
                logger.info("Stage %s: loading complete checkpoint (%d records)", stage_name, len(records))
                return records

        # Partial checkpoint -- resume from last record
        existing = count_checkpoint_records(path)
        if existing > 0:
            logger.info("Stage %s: resuming from record %d", stage_name, existing)
            return run_fn(start_idx=existing)

    return run_fn(start_idx=0)


# ---------------------------------------------------------------------------
# process_single_job — backward-compatible entry point (unchanged logic)
# ---------------------------------------------------------------------------

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
    stage_counters: Dict[str, int] = {
        "extractor_mentions": 0,
        "skill_verifier_rejected": 0,
        "skill_verifier_parse_failed": 0,
        "requirement_parse_failed": 0,
        "hardsoft_parse_failed": 0,
        "stage_errors": 0,
    }

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

        stage_counters["extractor_mentions"] = len(raw_all)

        for m in raw_all:
            pl = m.get("_parsed_line")
            if pl is None:
                continue
            span = m.get("skill_span") or ""
            g0 = int(m.get("_glob_char_start", pl.char_start))
            g1 = int(m.get("_glob_char_end", pl.char_end))
            line_text = pl.text
            cs = int(m.get("_line_char_start", 0))
            ce = int(m.get("_line_char_end", len(line_text)))
            offset_valid = bool(m.get("_offset_valid")) or (line_text[cs:ce] == span if span else False)
            evidence = str(m.get("evidence", span))
            evidence_ok = evidence in line_text if evidence else False

            raw_conf = float(m.get("span_confidence", m.get("confidence", 0.7)))
            hard_soft = "unknown"
            req_lvl = "unclear"
            rules = _matching_rules(candidates, pl.line_id, g0, g1)

            v_status = "skipped"
            v_model = ""
            v_conf: Optional[float] = None
            v_notes = ""
            req_status = "skipped"
            req_model = ""
            req_conf: Optional[float] = None
            req_notes = ""
            hs_status = "skipped"
            hs_model = ""
            hs_conf: Optional[float] = None
            hs_notes = ""

            mention_for_v = {
                "skill_span": span,
                "normalized_candidate": m.get("normalized_candidate", span),
                "span_confidence": raw_conf,
                "evidence": evidence,
                "line_id": pl.line_id,
                "section": pl.section,
            }

            pipeline_audit: Dict[str, Any] = {
                "extractor": {
                    "status": "completed",
                    "model": extractor_model_used,
                    "output": {
                        "skill_span": span,
                        "normalized_candidate": m.get("normalized_candidate", span),
                        "char_start": g0,
                        "char_end": g1,
                        "evidence": evidence,
                        "span_confidence": raw_conf,
                        "reason": str(m.get("reason", "")),
                    },
                },
                "skill_verifier": {"status": "skipped", "model": ""},
                "requirement_classifier": {"status": "skipped", "model": ""},
                "hardsoft_classifier": {"status": "skipped", "model": ""},
            }

            if cfg.verifier_enabled:
                if progress_callback:
                    progress_callback("verifier", "validating extracted mention", {})
                try:
                    vout = verify_mention(cfg, pl.section, line_text, mention_for_v)
                    v_model = cfg.verifier_model
                    v_notes = str(vout.get("notes", ""))
                    if vout.get("_parse_failed"):
                        v_status = "parse_failed"
                        stage_counters["skill_verifier_parse_failed"] += 1
                        v_conf = None
                    else:
                        v_status = "accepted" if vout.get("is_skill", True) else "rejected"
                        vc = vout.get("confidence")
                        if vc is None:
                            v_conf = raw_conf
                        else:
                            try:
                                v_conf = float(vc)
                            except (TypeError, ValueError):
                                v_conf = raw_conf
                        if not vout.get("is_skill", True):
                            stage_counters["skill_verifier_rejected"] += 1
                            m["is_skill"] = False
                    pipeline_audit["skill_verifier"] = {
                        "status": v_status,
                        "model": v_model,
                        "output": {
                            "is_skill": bool(vout.get("is_skill", True)),
                            "confidence": v_conf,
                            "evidence": str(vout.get("evidence", evidence)),
                            "notes": v_notes,
                        },
                    }
                except Exception as e:
                    v_status = "error"
                    v_notes = str(e)[:200]
                    v_model = cfg.verifier_model
                    v_conf = None
                    stage_counters["stage_errors"] += 1
                    pipeline_audit["skill_verifier"] = {
                        "status": v_status,
                        "model": v_model,
                        "error": v_notes,
                    }

            is_skill = bool(m.get("is_skill", True))
            if v_status == "rejected":
                is_skill = False

            mention_for_classifiers = {
                "skill_span": span,
                "normalized_candidate": m.get("normalized_candidate", span),
                "is_skill": is_skill,
                "span_confidence": raw_conf,
                "skill_verifier_confidence": v_conf,
                "evidence": evidence,
            }

            if is_skill and cfg.requirement_classifier_enabled:
                if progress_callback:
                    progress_callback("classifier", "classifying requirement level", {})
                try:
                    req_out = classify_requirement_level(
                        cfg, pl.section, line_text, mention_for_classifiers
                    )
                    req_model = cfg.requirement_model
                    req_notes = str(req_out.get("notes", ""))
                    if req_out.get("_parse_failed"):
                        req_status = "parse_failed"
                        req_conf = None
                        stage_counters["requirement_parse_failed"] += 1
                    else:
                        req_status = "completed"
                        req_lvl = str(req_out.get("requirement_level", "unclear")).lower()
                        rc = req_out.get("confidence")
                        req_conf = raw_conf if rc is None else float(rc)
                    pipeline_audit["requirement_classifier"] = {
                        "status": req_status,
                        "model": req_model,
                        "output": {
                            "requirement_level": req_lvl,
                            "confidence": req_conf,
                            "evidence": str(req_out.get("evidence", evidence)),
                            "notes": req_notes,
                        },
                    }
                except Exception as e:
                    req_status = "error"
                    req_model = cfg.requirement_model
                    req_notes = str(e)[:200]
                    req_conf = None
                    stage_counters["stage_errors"] += 1
                    pipeline_audit["requirement_classifier"] = {
                        "status": req_status,
                        "model": req_model,
                        "error": req_notes,
                    }

            if is_skill and cfg.hardsoft_classifier_enabled:
                if progress_callback:
                    progress_callback("classifier", "classifying hard vs soft", {})
                try:
                    hs_out = classify_hard_soft(
                        cfg, pl.section, line_text, mention_for_classifiers
                    )
                    hs_model = cfg.hardsoft_model
                    hs_notes = str(hs_out.get("notes", ""))
                    if hs_out.get("_parse_failed"):
                        hs_status = "parse_failed"
                        hs_conf = None
                        stage_counters["hardsoft_parse_failed"] += 1
                    else:
                        hs_status = "completed"
                        hard_soft = str(hs_out.get("hard_soft", "unknown")).lower()
                        hc = hs_out.get("confidence")
                        hs_conf = raw_conf if hc is None else float(hc)
                    pipeline_audit["hardsoft_classifier"] = {
                        "status": hs_status,
                        "model": hs_model,
                        "output": {
                            "hard_soft": hard_soft,
                            "confidence": hs_conf,
                            "evidence": str(hs_out.get("evidence", evidence)),
                            "notes": hs_notes,
                        },
                    }
                except Exception as e:
                    hs_status = "error"
                    hs_model = cfg.hardsoft_model
                    hs_notes = str(e)[:200]
                    hs_conf = None
                    stage_counters["stage_errors"] += 1
                    pipeline_audit["hardsoft_classifier"] = {
                        "status": hs_status,
                        "model": hs_model,
                        "error": hs_notes,
                    }

            final_c = compute_final_confidence(
                raw_model_confidence=raw_conf,
                verifier_confidence=v_conf,
                requirement_confidence=req_conf,
                hardsoft_confidence=hs_conf,
                section=pl.section,
                rules_fired=rules,
                boilerplate_label=pl.boilerplate_label,
                offset_valid=offset_valid,
                evidence_substring_of_line=evidence_ok,
                verifier_status=v_status,
                requirement_status=req_status,
                hardsoft_status=hs_status,
            )

            mention_counter += 1
            sm = SkillMention(
                mention_id=f"{job_key}_M{mention_counter:05d}",
                line_id=pl.line_id,
                skill_span=span,
                normalized_candidate=str(m.get("normalized_candidate", span)).strip(),
                is_skill=is_skill,
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
                requirement_model=req_model,
                requirement_status=req_status,
                requirement_confidence=req_conf,
                requirement_notes=req_notes,
                hardsoft_model=hs_model,
                hardsoft_status=hs_status,
                hardsoft_confidence=hs_conf,
                hardsoft_notes=hs_notes,
                pipeline_audit=pipeline_audit,
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
        extra={
            "job_title_snapshot": title[:200],
            "requirement_model": cfg.requirement_model if cfg.requirement_classifier_enabled else "",
            "hardsoft_model": cfg.hardsoft_model if cfg.hardsoft_classifier_enabled else "",
        },
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
        "pipeline_stage_audit": {
            "stage_counters": stage_counters,
            "models": {
                "extractor_model": extractor_model_used,
                "skill_verifier_model": cfg.verifier_model if cfg.verifier_enabled else "",
                "requirement_model": cfg.requirement_model if cfg.requirement_classifier_enabled else "",
                "hardsoft_model": cfg.hardsoft_model if cfg.hardsoft_classifier_enabled else "",
            },
        },
        "extraction_metadata": meta.to_dict(),
    }
    return augment_job_record(job, augmentation)


# ---------------------------------------------------------------------------
# run_pipeline — stage-first architecture with checkpoints
# ---------------------------------------------------------------------------

def run_pipeline(
    jobs: List[Dict[str, Any]],
    cfg: PipelineConfig,
    output_dir: Path,
    run_id: str,
    write_reports: bool = True,
    progress_callback: Optional[Callable[[int, int, str, str, Any], None]] = None,
    log_path: Optional[Path] = None,
    resume: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Path], RunStats]:
    """
    Process all jobs stage-by-stage and write artifacts under output_dir.
    progress_callback(item_idx, total_items, stage, detail, extra) is called during run.
    Returns (augmented_jobs, paths_dict, run_stats).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()

    stats = RunStats(
        run_id=run_id,
        started_at=started,
        pipeline_version=cfg.pipeline_version,
        extractor_model=cfg.extractor_model,
        verifier_model=cfg.verifier_model,
        requirement_model=cfg.requirement_model,
        hardsoft_model=cfg.hardsoft_model,
        fallback_model=cfg.fallback_model,
        ollama_base_url=cfg.ollama_base_url,
        backend=cfg.backend,
        skip_llm=cfg.skip_llm,
        batch_max_lines=cfg.extractor_batch_max_lines,
        verifier_enabled=cfg.verifier_enabled,
        requirement_classifier_enabled=cfg.requirement_classifier_enabled,
        hardsoft_classifier_enabled=cfg.hardsoft_classifier_enabled,
    )
    stats.jobs_total = len(jobs)

    def _timing_cb(model: str, elapsed: float, role: str) -> None:
        stats.record_llm(model, elapsed, role)

    original_timing = cfg.llm_timing_callback
    cfg.llm_timing_callback = _timing_cb

    # --- Stage 0: Preprocess ---
    stats.record_stage_start("stage0_preprocessed")
    stage0_data = _load_or_run_stage(
        "stage0_preprocessed", ckpt_dir, run_id, resume, len(jobs),
        run_fn=lambda start_idx=0: _run_stage0_preprocess(
            jobs, cfg, run_id, ckpt_dir, progress_callback, start_idx=start_idx,
        ),
        progress_cb=progress_callback,
    )
    stats.record_stage_end("stage0_preprocessed")

    # --- Stage 1: Extract ---
    stats.record_stage_start("stage1_extracted")
    stage1_data = _load_or_run_stage(
        "stage1_extracted", ckpt_dir, run_id, resume, len(stage0_data),
        run_fn=lambda start_idx=0: _run_stage1_extract(
            stage0_data, cfg, run_id, ckpt_dir, progress_callback, stats, start_idx=start_idx,
        ),
        progress_cb=progress_callback,
    )
    stats.record_stage_end("stage1_extracted")

    # Count total mentions for stages 2-4
    total_mentions = sum(
        len(r.get("raw_mentions", [])) for r in stage1_data
    )

    # --- Stage 2: Verify ---
    stats.record_stage_start("stage2_verified")
    stage2_data = _load_or_run_stage(
        "stage2_verified", ckpt_dir, run_id, resume, total_mentions,
        run_fn=lambda start_idx=0: _run_stage2_verify(
            stage0_data, stage1_data, cfg, run_id, ckpt_dir, progress_callback, stats,
            start_idx=start_idx,
        ),
        progress_cb=progress_callback,
    )
    stats.record_stage_end("stage2_verified")

    # --- Stage 3: Requirement classification ---
    stats.record_stage_start("stage3_requirement")
    stage3_data = _load_or_run_stage(
        "stage3_requirement", ckpt_dir, run_id, resume, total_mentions,
        run_fn=lambda start_idx=0: _run_stage3_requirement(
            stage0_data, stage1_data, stage2_data, cfg, run_id, ckpt_dir, progress_callback,
            stats, start_idx=start_idx,
        ),
        progress_cb=progress_callback,
    )
    stats.record_stage_end("stage3_requirement")

    # --- Stage 4: Hard/soft classification ---
    stats.record_stage_start("stage4_hardsoft")
    stage4_data = _load_or_run_stage(
        "stage4_hardsoft", ckpt_dir, run_id, resume, total_mentions,
        run_fn=lambda start_idx=0: _run_stage4_hardsoft(
            stage0_data, stage1_data, stage2_data, cfg, run_id, ckpt_dir, progress_callback,
            stats, start_idx=start_idx,
        ),
        progress_cb=progress_callback,
    )
    stats.record_stage_end("stage4_hardsoft")

    # --- Stage 5: Assemble ---
    stats.record_stage_start("stage5_assemble")
    augmented = _run_stage5_assemble(
        jobs, stage0_data, stage1_data, stage2_data, stage3_data, stage4_data,
        cfg, run_id, started,
    )
    stats.record_stage_end("stage5_assemble")

    cfg.llm_timing_callback = original_timing

    stats.jobs_success = sum(
        1 for j in augmented
        if not j.get("extraction_metadata", {}).get("error")
    )
    stats.jobs_failed = stats.jobs_total - stats.jobs_success
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
