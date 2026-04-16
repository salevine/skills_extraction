"""
End-to-end pipeline: ingest -> preprocess -> section -> quality -> boilerplate ->
candidates -> LLM extract -> verify -> confidence -> augment records.

Stage-first architecture: all jobs pass through each stage before advancing to the
next, minimizing model swaps on the Ollama server (1 swap total instead of ~N*4).
"""

from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .boilerplate import label_parsed_lines
from .candidate_mining import CandidateSpan, mine_all_candidates
from .checkpoint import (
    append_checkpoint_record,
    checkpoint_complete,
    checkpoint_path,
    compute_stage_fingerprint,
    deserialize_candidate,
    deserialize_mentions_for_job,
    deserialize_parsed_line,
    load_checkpoint,
    serialize_candidate,
    serialize_mentions_for_job,
    serialize_parsed_line,
    write_checkpoint_footer,
    write_checkpoint_header,
)
from .confidence import compute_final_confidence
from .config import PipelineConfig
from .exporters import (
    write_augmented_jobs,
    write_frequency_report,
    write_job_skills_summary,
    write_low_confidence_review,
    write_mentions_csv,
    write_mentions_jsonl,
    write_quality_report,
)
from .io_utils import augment_job_record, stable_job_key
from .llm_extractor import batch_lines, extract_mentions_for_batch, extract_mentions_for_job
from .llm_hardsoft_classifier import classify_hard_soft
from .llm_requirement_classifier import classify_requirement_level
from .llm_verifier import verify_mention
from .preprocessing import extract_description_fields, preprocess_description
from .quality import assess_quality
from .run_stats import RunStats
from .schemas import ExtractionMetadata, ParsedLine, QualityStatus, SkillMention
from .sectioning import segment_lines, split_inline_section_headers

logger = logging.getLogger(__name__)

# Stage fingerprints — populated by run_pipeline(), read by stage functions
# when writing checkpoint headers. Maps stage_name -> fingerprint string.
_stage_fingerprints: Dict[str, str] = {}
_STAGE_RERUN_ALIASES: Dict[str, str] = {
    "stage1": "stage1_extracted",
    "stage1_extracted": "stage1_extracted",
    "stage2": "stage2_verified",
    "stage2_verified": "stage2_verified",
    "stage3": "stage3_requirement",
    "stage3_requirement": "stage3_requirement",
    "stage4": "stage4_hardsoft",
    "stage4_hardsoft": "stage4_hardsoft",
}
_STAGE_RERUN_ORDER: List[str] = [
    "stage1_extracted",
    "stage2_verified",
    "stage3_requirement",
    "stage4_hardsoft",
]


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
    return list(dict.fromkeys(rules))


# ---------------------------------------------------------------------------
# Windowed parallel execution helper (vLLM only)
# ---------------------------------------------------------------------------

class _WorkerError:
    """Sentinel wrapping a failed future for re-raise in the main thread."""
    __slots__ = ("exc",)
    def __init__(self, exc: BaseException):
        self.exc = exc


def _run_rolling(
    total: int,
    start_idx: int,
    max_workers: int,
    process_fn: Callable[[int], Dict[str, Any]],
    progress_fn: Optional[Callable[[int, int], None]],
    fh,
    records: List[Dict[str, Any]],
) -> None:
    """
    Process items using a long-lived rolling worker pool.

    A single ThreadPoolExecutor lives for the entire stage. Work is only
    submitted up to a bounded distance ahead of ``next_write`` so both
    in-flight futures and out-of-order completed results stay bounded.
    Results are written to the checkpoint file in original order as
    contiguous completed items become available.
    """
    backlog_limit = max_workers * 2
    pending: Dict[int, Any] = {}  # idx -> result or _WorkerError
    next_write = start_idx
    next_submit = start_idx

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx: Dict[Any, int] = {}

        def _submit_available() -> None:
            nonlocal next_submit
            while next_submit < total and (next_submit - next_write) < backlog_limit:
                future = pool.submit(process_fn, next_submit)
                future_to_idx[future] = next_submit
                next_submit += 1

        _submit_available()

        while next_write < total:
            if next_write in pending:
                result = pending.pop(next_write)
            else:
                done, _ = wait(tuple(future_to_idx.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    idx = future_to_idx.pop(future)
                    try:
                        pending[idx] = future.result()
                    except Exception as exc:
                        pending[idx] = _WorkerError(exc)
                result = pending.pop(next_write, None)
                if result is None:
                    continue

            if isinstance(result, _WorkerError):
                raise result.exc

            append_checkpoint_record(fh, result)
            records.append(result)
            if progress_fn:
                progress_fn(next_write, total)
            next_write += 1
            _submit_available()


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

    ser_mentions, lines_reg = serialize_mentions_for_job(raw_mentions)
    return {
        "job_index": s0["job_index"],
        "job_key": job_key,
        "extractor_model_used": extractor_model_used,
        "raw_mentions": ser_mentions,
        "_parsed_lines": lines_reg,
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
# Stage 0: Preprocess all jobs (no LLM) — DISABLED
# Retained for reference. This stage split descriptions into lines, labeled
# boilerplate/sections, mined regex candidates, and assessed quality. It
# enabled the 5-line batching approach in stage 1. Disabled because sending
# full job descriptions to the LLM in one call is simpler, more reliable
# (fewer HTTP calls), and preserves cross-line context.
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
# Stage 1: Extract all jobs — one LLM call per job (V2)
# ---------------------------------------------------------------------------

def _run_stage1_extract(
    jobs: List[Dict[str, Any]],
    cfg: PipelineConfig,
    run_id: str,
    ckpt_dir: Path,
    progress_cb: Optional[Callable] = None,
    stats: Optional[RunStats] = None,
    start_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Run LLM extraction on all jobs. One call per job, full description."""
    path = checkpoint_path(ckpt_dir.parent, run_id, "stage1_extracted")
    records: List[Dict[str, Any]] = []

    if start_idx > 0 and path.exists():
        _, records = load_checkpoint(path)
        records = records[:start_idx]

    mode = "a" if start_idx > 0 else "w"
    with open(path, mode, encoding="utf-8") as fh:
        if start_idx == 0:
            write_checkpoint_header(fh, run_id, "stage1_extracted", len(jobs),
                                       fingerprint=_stage_fingerprints.get("stage1_extracted"))

        if cfg.backend == "vllm":
            # vLLM parallel: one LLM call per job, bounded rolling submission
            # with endpoint failover and incremental checkpointing.
            endpoints = cfg.vllm_endpoints()
            completed_count = [0]
            failed_count = [0]
            total_jobs = len(jobs) - start_idx
            num_workers = len(endpoints)
            backlog_limit = num_workers * 2

            pending_results: Dict[int, Dict[str, Any]] = {}
            next_write_idx = start_idx
            next_submit_idx = start_idx

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                future_to_idx: Dict[Any, int] = {}

                def _submit_available() -> None:
                    nonlocal next_submit_idx
                    while (next_submit_idx < len(jobs)
                           and (next_submit_idx - next_write_idx) < backlog_limit):
                        _job = jobs[next_submit_idx]
                        _job_key = stable_job_key(_job, next_submit_idx)
                        ep = endpoints[next_submit_idx % num_workers]
                        f = pool.submit(
                            extract_mentions_for_job,
                            cfg, _job, _job_key, cfg.extractor_model, ep,
                            all_endpoints=endpoints,
                        )
                        future_to_idx[f] = next_submit_idx
                        next_submit_idx += 1

                _submit_available()

                while next_write_idx < len(jobs):
                    if next_write_idx in pending_results:
                        result = pending_results.pop(next_write_idx)
                    else:
                        done, _ = wait(tuple(future_to_idx.keys()), return_when=FIRST_COMPLETED)
                        for future in done:
                            idx = future_to_idx.pop(future)
                            try:
                                pending_results[idx] = {
                                    "raw_mentions": future.result(),
                                    "stage1_error": "",
                                }
                            except Exception as e:
                                logger.debug("vLLM extractor job %d failed: %s", idx, e)
                                pending_results[idx] = {
                                    "raw_mentions": [],
                                    "stage1_error": str(e)[:200],
                                }
                                failed_count[0] += 1
                            completed_count[0] += 1
                            if progress_cb:
                                fail_info = f" ({failed_count[0]} failed)" if failed_count[0] else ""
                                progress_cb(completed_count[0] - 1, total_jobs,
                                            "stage1_extract",
                                            f"job {completed_count[0]}/{total_jobs}{fail_info}",
                                            {"stage": "stage1_extract"})
                        result = pending_results.pop(next_write_idx, None)
                        if result is None:
                            continue

                    _job = jobs[next_write_idx]
                    _job_key = stable_job_key(_job, next_write_idx)
                    ser_mentions, lines_reg = serialize_mentions_for_job(result["raw_mentions"])
                    record = {
                        "job_index": next_write_idx,
                        "job_key": _job_key,
                        "extractor_model_used": cfg.extractor_model,
                        "raw_mentions": ser_mentions,
                        "_parsed_lines": lines_reg,
                    }
                    if result["stage1_error"]:
                        record["stage1_error"] = result["stage1_error"]
                    append_checkpoint_record(fh, record)
                    records.append(record)
                    next_write_idx += 1
                    _submit_available()

            if failed_count[0]:
                logger.warning("vLLM extraction: %d/%d jobs failed (timeouts/errors)",
                               failed_count[0], total_jobs)
        else:
            # Ollama / OpenRouter: sequential, one call per job
            for idx in range(start_idx, len(jobs)):
                job = jobs[idx]
                job_key = stable_job_key(job, idx)

                if progress_cb:
                    progress_cb(idx, len(jobs), "stage1_extract",
                                f"extracting {job_key}",
                                {"stage": "stage1_extract"})

                extractor_model_used = cfg.extractor_model
                stage1_error = ""
                try:
                    raw_mentions = extract_mentions_for_job(
                        cfg, job, job_key, model=cfg.extractor_model,
                    )
                except Exception as e:
                    logger.warning("Extractor failed for %s (%s), trying fallback", job_key, e)
                    try:
                        raw_mentions = extract_mentions_for_job(
                            cfg, job, job_key, model=cfg.fallback_model,
                        )
                        extractor_model_used = cfg.fallback_model
                    except Exception as e2:
                        logger.error("Fallback extractor failed for %s: %s", job_key, e2)
                        raw_mentions = []
                        stage1_error = str(e2)[:200]

                ser_mentions, lines_reg = serialize_mentions_for_job(raw_mentions)
                record = {
                    "job_index": idx,
                    "job_key": job_key,
                    "extractor_model_used": extractor_model_used,
                    "raw_mentions": ser_mentions,
                    "_parsed_lines": lines_reg,
                }
                if stage1_error:
                    record["stage1_error"] = stage1_error
                append_checkpoint_record(fh, record)
                records.append(record)

        write_checkpoint_footer(fh, len(records))
    return records


def _normalize_rerun_stage_name(rerun_from_stage: Optional[str]) -> Optional[str]:
    if not rerun_from_stage:
        return None
    key = rerun_from_stage.strip().lower()
    normalized = _STAGE_RERUN_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"Unsupported rerun stage: {rerun_from_stage}. "
            "Use one of: stage1, stage2, stage3, stage4."
        )
    return normalized


def _invalidate_checkpoints_from_stage(
    output_dir: Path,
    run_id: str,
    rerun_from_stage: Optional[str],
) -> List[Path]:
    normalized = _normalize_rerun_stage_name(rerun_from_stage)
    if normalized is None:
        return []

    invalidated: List[Path] = []
    start_idx = _STAGE_RERUN_ORDER.index(normalized)
    for stage_name in _STAGE_RERUN_ORDER[start_idx:]:
        path = checkpoint_path(output_dir, run_id, stage_name)
        if path.exists():
            path.unlink()
            invalidated.append(path)
    return invalidated


def _extract_stage1_record(
    job: Dict[str, Any],
    job_index: int,
    cfg: PipelineConfig,
    endpoint: Optional[str] = None,
    all_endpoints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    job_key = stable_job_key(job, job_index)
    raw_mentions = extract_mentions_for_job(
        cfg,
        job,
        job_key,
        model=cfg.extractor_model,
        endpoint=endpoint,
        all_endpoints=all_endpoints,
    )
    ser_mentions, lines_reg = serialize_mentions_for_job(raw_mentions)
    return {
        "job_index": job_index,
        "job_key": job_key,
        "extractor_model_used": cfg.extractor_model,
        "raw_mentions": ser_mentions,
        "_parsed_lines": lines_reg,
    }


def _rewrite_stage1_checkpoint(
    stage1_data: List[Dict[str, Any]],
    jobs_total: int,
    ckpt_dir: Path,
    run_id: str,
) -> None:
    path = checkpoint_path(ckpt_dir.parent, run_id, "stage1_extracted")
    ordered = sorted(stage1_data, key=lambda r: r.get("job_index", 0))
    with open(path, "w", encoding="utf-8") as fh:
        write_checkpoint_header(
            fh,
            run_id,
            "stage1_extracted",
            jobs_total,
            fingerprint=_stage_fingerprints.get("stage1_extracted"),
        )
        for record in ordered:
            append_checkpoint_record(fh, record)
        write_checkpoint_footer(fh, len(ordered))


def _retry_failed_stage1_records(
    jobs: List[Dict[str, Any]],
    stage1_data: List[Dict[str, Any]],
    cfg: PipelineConfig,
    run_id: str,
    ckpt_dir: Path,
    progress_cb: Optional[Callable] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    failed_indices = sorted(
        record["job_index"]
        for record in stage1_data
        if record.get("stage1_error")
    )
    if not failed_indices:
        return stage1_data, 0

    logger.info("Stage 1: retrying %d failed job(s) with current extractor", len(failed_indices))
    updated_records = {record["job_index"]: dict(record) for record in stage1_data}
    retried_total = len(failed_indices)

    if cfg.backend == "vllm":
        endpoints = cfg.vllm_endpoints()
        num_workers = max(1, len(endpoints))
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            future_to_idx: Dict[Any, int] = {}
            for job_index in failed_indices:
                endpoint = endpoints[job_index % num_workers]
                future = pool.submit(
                    _extract_stage1_record,
                    jobs[job_index],
                    job_index,
                    cfg,
                    endpoint,
                    endpoints,
                )
                future_to_idx[future] = job_index

            completed = 0
            for future in as_completed(future_to_idx):
                job_index = future_to_idx[future]
                completed += 1
                if progress_cb:
                    progress_cb(
                        completed - 1,
                        retried_total,
                        "stage1_retry",
                        f"retrying failed job {completed}/{retried_total}",
                        {"stage": "stage1_retry"},
                    )
                try:
                    updated_records[job_index] = future.result()
                except Exception as e:
                    logger.warning("Stage 1 retry failed for %s: %s", stable_job_key(jobs[job_index], job_index), e)
                    updated_records[job_index] = {
                        "job_index": job_index,
                        "job_key": stable_job_key(jobs[job_index], job_index),
                        "extractor_model_used": cfg.extractor_model,
                        "raw_mentions": [],
                        "_parsed_lines": {},
                        "stage1_error": str(e)[:200],
                    }
    else:
        for completed, job_index in enumerate(failed_indices, start=1):
            if progress_cb:
                progress_cb(
                    completed - 1,
                    retried_total,
                    "stage1_retry",
                    f"retrying failed job {completed}/{retried_total}",
                    {"stage": "stage1_retry"},
                )
            try:
                updated_records[job_index] = _extract_stage1_record(jobs[job_index], job_index, cfg)
            except Exception as e:
                logger.warning("Stage 1 retry failed for %s: %s", stable_job_key(jobs[job_index], job_index), e)
                updated_records[job_index] = {
                    "job_index": job_index,
                    "job_key": stable_job_key(jobs[job_index], job_index),
                    "extractor_model_used": cfg.extractor_model,
                    "raw_mentions": [],
                    "_parsed_lines": {},
                    "stage1_error": str(e)[:200],
                }

    refreshed = [updated_records[idx] for idx in sorted(updated_records)]
    _rewrite_stage1_checkpoint(refreshed, len(jobs), ckpt_dir, run_id)
    return refreshed, retried_total


# ---------------------------------------------------------------------------
# Stage 2: Verify all mentions (mistral-nemo:12b loads once)
# ---------------------------------------------------------------------------

def _run_stage2_verify(
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
    mention_tasks = _build_mention_task_list(stage1_data)
    total_mentions = len(mention_tasks)

    mode = "a" if start_idx > 0 else "w"
    with open(path, mode, encoding="utf-8") as fh:
        if start_idx == 0:
            write_checkpoint_header(fh, run_id, "stage2_verified", total_mentions,
                                       fingerprint=_stage_fingerprints.get("stage2_verified"))

        if cfg.backend == "vllm":
            def _s2_progress(i: int, total: int) -> None:
                if progress_cb:
                    progress_cb(i, total, "stage2_verify",
                                f"verifying mention {i+1}/{total}",
                                {"stage": "stage2_verify"})

            _run_rolling(
                total=total_mentions,
                start_idx=start_idx,
                max_workers=cfg.vllm_num_endpoints,
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

    mention_tasks = _build_mention_task_list(stage1_data)
    # Build verifier lookup: (job_index, mention_idx) -> verifier_output
    v_lookup = {(r["job_index"], r["mention_idx"]): r["verifier_output"] for r in stage2_data}
    total_mentions = len(mention_tasks)

    mode = "a" if start_idx > 0 else "w"
    with open(path, mode, encoding="utf-8") as fh:
        if start_idx == 0:
            write_checkpoint_header(fh, run_id, "stage3_requirement", total_mentions,
                                       fingerprint=_stage_fingerprints.get("stage3_requirement"))

        if cfg.backend == "vllm":
            def _s3_progress(i: int, total: int) -> None:
                if progress_cb:
                    progress_cb(i, total, "stage3_requirement",
                                f"classifying requirement {i+1}/{total}",
                                {"stage": "stage3_requirement"})

            _run_rolling(
                total=total_mentions,
                start_idx=start_idx,
                max_workers=cfg.vllm_num_endpoints,
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

    mention_tasks = _build_mention_task_list(stage1_data)
    v_lookup = {(r["job_index"], r["mention_idx"]): r["verifier_output"] for r in stage2_data}
    total_mentions = len(mention_tasks)

    mode = "a" if start_idx > 0 else "w"
    with open(path, mode, encoding="utf-8") as fh:
        if start_idx == 0:
            write_checkpoint_header(fh, run_id, "stage4_hardsoft", total_mentions,
                                       fingerprint=_stage_fingerprints.get("stage4_hardsoft"))

        if cfg.backend == "vllm":
            def _s4_progress(i: int, total: int) -> None:
                if progress_cb:
                    progress_cb(i, total, "stage4_hardsoft",
                                f"classifying hard/soft {i+1}/{total}",
                                {"stage": "stage4_hardsoft"})

            _run_rolling(
                total=total_mentions,
                start_idx=start_idx,
                max_workers=cfg.vllm_num_endpoints,
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
    stage1_data: List[Dict[str, Any]],
    stage2_data: List[Dict[str, Any]],
    stage3_data: List[Dict[str, Any]],
    stage4_data: List[Dict[str, Any]],
    cfg: PipelineConfig,
    run_id: str,
    started: str,
) -> List[Dict[str, Any]]:
    """Assemble all stage data into augmented job records."""
    # Index stage2/3/4 by (job_index, mention_idx)
    v_lookup = {(r["job_index"], r["mention_idx"]): r["verifier_output"] for r in stage2_data}
    req_lookup = {(r["job_index"], r["mention_idx"]): r["requirement_output"] for r in stage3_data}
    hs_lookup = {(r["job_index"], r["mention_idx"]): r["hardsoft_output"] for r in stage4_data}

    # Index stage1 by job_index
    s1_by_job = {r["job_index"]: r for r in stage1_data}

    augmented: List[Dict[str, Any]] = []

    for idx, job in enumerate(jobs):
        job_key = stable_job_key(job, idx)
        title, desc_raw = extract_description_fields(job)
        # Compute normalized text so exported offsets have a matching reference
        _pre = preprocess_description(desc_raw)
        desc_normalized = split_inline_section_headers(_pre.description_normalized)
        line_texts = desc_normalized.split("\n")
        qa = assess_quality(desc_raw, desc_normalized, line_texts, cfg.quality_complete_min_score)
        s1 = s1_by_job.get(idx, {})
        extractor_model_used = s1.get("extractor_model_used", cfg.extractor_model)
        job_error = str(s1.get("stage1_error", "") or "")
        if not s1:
            job_error = "missing_stage1_record"

        raw_mentions_ser = s1.get("raw_mentions", [])
        lines_reg_raw = s1.get("_parsed_lines")  # may be None for older checkpoints
        raw_mentions_deser = deserialize_mentions_for_job(raw_mentions_ser, lines_reg_raw)
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
        if job_error:
            stage_counters["stage_errors"] += 1

        for m_idx, m in enumerate(raw_mentions_deser):
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
            rules = []  # no synthetic rule boost; only real rule matches get credit

            # --- Verifier output ---
            # Stage 2 is authoritative when it ran. Extractor is_skill is
            # only used as fallback when stage 2 was skipped / not present.
            extractor_is_skill = m.get("is_skill")
            vout = v_lookup.get((idx, m_idx), {"status": "skipped"})
            v_status = vout.get("status", "skipped")
            v_model = vout.get("model", "")
            v_conf = vout.get("confidence")
            v_notes = vout.get("notes", "")

            if v_status == "parse_failed":
                stage_counters["skill_verifier_parse_failed"] += 1
            if v_status == "rejected":
                stage_counters["skill_verifier_rejected"] += 1

            if v_status in ("accepted", "rejected", "parse_failed", "error"):
                # Stage 2 actually ran — its result is authoritative
                is_skill = _mention_is_skill(vout)
            elif extractor_is_skill is not None:
                # Stage 2 did not run; fall back to extractor field
                is_skill = bool(extractor_is_skill)
                v_status = "extractor"
                v_conf = raw_conf
            else:
                is_skill = True  # default if neither ran

            # --- Requirement output ---
            # Stage 3 is authoritative when it ran. Extractor requirement_level
            # is only used as fallback when stage 3 was skipped / not present.
            extractor_req = m.get("requirement_level")
            rout = req_lookup.get((idx, m_idx), {"status": "skipped"})
            req_status = rout.get("status", "skipped")
            req_model = rout.get("model", "")
            req_conf = rout.get("confidence")
            req_notes = rout.get("notes", "")

            if req_status in ("completed", "parse_failed", "error"):
                # Stage 3 actually ran — use its result
                req_lvl = rout.get("requirement_level", "unclear")
                if req_status == "parse_failed":
                    stage_counters["requirement_parse_failed"] += 1
                if req_status == "error":
                    stage_counters["stage_errors"] += 1
            elif extractor_req and extractor_req in ("required", "optional", "unclear"):
                req_lvl = extractor_req
                req_status = "extractor"
                req_conf = raw_conf
            else:
                req_lvl = "unclear"

            # --- Hard/soft output ---
            # Stage 4 is authoritative when it ran. Extractor hard_soft
            # is only used as fallback when stage 4 was skipped / not present.
            extractor_hs = m.get("hard_soft")
            hsout = hs_lookup.get((idx, m_idx), {"status": "skipped"})
            hs_status = hsout.get("status", "skipped")
            hs_model = hsout.get("model", "")
            hs_conf = hsout.get("confidence")
            hs_notes = hsout.get("notes", "")

            if hs_status in ("completed", "parse_failed", "error"):
                # Stage 4 actually ran — use its result
                hard_soft = hsout.get("hard_soft", "unknown")
                if hs_status == "parse_failed":
                    stage_counters["hardsoft_parse_failed"] += 1
                if hs_status == "error":
                    stage_counters["stage_errors"] += 1
            elif extractor_hs and extractor_hs in ("hard", "soft", "unknown"):
                hard_soft = extractor_hs
                hs_status = "extractor"
                hs_conf = raw_conf
            else:
                hard_soft = "unknown"

            # Build pipeline_audit — preserve extractor's original classifications
            # separately from the final authoritative values chosen above.
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
                        "is_skill": bool(extractor_is_skill) if extractor_is_skill is not None else None,
                        "requirement_level": str(extractor_req) if extractor_req else None,
                        "hard_soft": str(extractor_hs) if extractor_hs else None,
                    },
                },
                "skill_verifier": {"status": v_status, "model": v_model},
                "requirement_classifier": {"status": req_status, "model": req_model},
                "hardsoft_classifier": {"status": hs_status, "model": hs_model},
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
        meta = ExtractionMetadata(
            run_id=run_id,
            pipeline_version=cfg.pipeline_version,
            extractor_model=extractor_model_used,
            verifier_model=cfg.verifier_model if cfg.verifier_enabled else "",
            job_key=job_key,
            started_at=started,
            completed_at=completed,
            error=job_error,
            extra={
                "job_title_snapshot": (title or "")[:200],
                "requirement_model": cfg.requirement_model if cfg.requirement_classifier_enabled else "",
                "hardsoft_model": cfg.hardsoft_model if cfg.hardsoft_classifier_enabled else "",
            },
        )

        augmentation = {
            "description_raw": desc_raw,
            "description_normalized": desc_normalized,
            "quality_assessment": qa.to_dict(),
            "skill_mentions": skill_mentions,
            "pipeline_stage_audit": {
                "stage_counters": stage_counters,
                "job_error": job_error,
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
# Helper: build flat mention task list from stage1 data
# ---------------------------------------------------------------------------

def _build_mention_task_list(
    stage1_data: List[Dict[str, Any]],
) -> List[Tuple[int, str, int, Dict[str, Any], ParsedLine]]:
    """
    Build a flat ordered list of (job_index, job_key, mention_idx, mention_dict, parsed_line)
    across all jobs, for stages 2-4 to iterate over.

    Each mention carries its own _parsed_line (created by extract_mentions_for_job),
    so this no longer needs stage0_data.
    """
    tasks: List[Tuple[int, str, int, Dict[str, Any], ParsedLine]] = []
    for s1 in stage1_data:
        job_idx = s1["job_index"]
        job_key = s1["job_key"]
        lines_reg_raw = s1.get("_parsed_lines")
        mentions = deserialize_mentions_for_job(s1.get("raw_mentions", []), lines_reg_raw)
        for m_idx, m in enumerate(mentions):
            pl = m.get("_parsed_line")
            if pl is not None:
                tasks.append((job_idx, job_key, m_idx, m, pl))
    return tasks


def _mention_is_skill(verifier_output: Dict[str, Any]) -> bool:
    """Determine if a mention is considered a skill based on verifier output.

    Returns True only for accepted or skipped mentions. Rejected, error,
    and parse_failed all return False to avoid wasting LLM calls on
    uncertain mentions in stages 3-4.
    """
    status = verifier_output.get("status", "skipped")
    if status in ("rejected", "error", "parse_failed"):
        return False
    # For skipped, accepted: default True
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
    fingerprint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Check if a checkpoint exists for this stage:
    - Complete checkpoint + resume=True + fingerprint matches -> load from file
    - Partial checkpoint + resume=True + fingerprint matches -> resume from last record
    - Fingerprint mismatch -> invalidate and rerun from scratch
    - Otherwise -> run from scratch
    """
    path = checkpoint_path(ckpt_dir.parent, run_id, stage_name)

    if resume and path.exists():
        if checkpoint_complete(path):
            meta, records = load_checkpoint(path)
            # Any current fingerprint requires a matching checkpoint fingerprint.
            ckpt_fp = meta.get("fingerprint")
            if fingerprint and ckpt_fp != fingerprint:
                logger.warning(
                    "Checkpoint %s fingerprint mismatch or missing fingerprint "
                    "(checkpoint=%s, current=%s); "
                    "config/prompts changed — rerunning from scratch",
                    stage_name, (ckpt_fp or "<missing>")[:8], fingerprint[:8],
                )
                return run_fn(start_idx=0)
            ckpt_total = meta.get("total_jobs", 0)
            if ckpt_total != total_expected:
                logger.warning(
                    "Checkpoint %s has %d records but expected %d jobs; rerunning from scratch",
                    stage_name, ckpt_total, total_expected,
                )
                return run_fn(start_idx=0)
            else:
                logger.info("Stage %s: loading complete checkpoint (%d records)", stage_name, len(records))
                return records

        # Partial checkpoint -- check fingerprint before resuming
        meta_partial, records_partial = load_checkpoint(path)
        ckpt_fp = meta_partial.get("fingerprint")
        if fingerprint and ckpt_fp != fingerprint:
            logger.warning(
                "Partial checkpoint %s fingerprint mismatch or missing fingerprint; "
                "rerunning from scratch",
                stage_name,
            )
            return run_fn(start_idx=0)

        ckpt_total = meta_partial.get("total_jobs", 0)
        if ckpt_total and ckpt_total != total_expected:
            logger.warning(
                "Partial checkpoint %s has %d records expected but current run expects %d; "
                "rerunning from scratch",
                stage_name, ckpt_total, total_expected,
            )
            return run_fn(start_idx=0)

        existing = len(records_partial)
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
    rerun_from_stage: Optional[str] = None,
    retry_stage1_errors: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Path], RunStats]:
    """
    Process all jobs stage-by-stage and write artifacts under output_dir.
    progress_callback(item_idx, total_items, stage, detail, extra) is called during run.
    Returns (augmented_jobs, paths_dict, run_stats).
    """
    from .checkpoint import set_flush_interval
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_flush_interval(cfg.checkpoint_flush_interval)
    started = dt.datetime.now(dt.timezone.utc).isoformat()

    invalidated = _invalidate_checkpoints_from_stage(output_dir, run_id, rerun_from_stage)
    if invalidated:
        logger.info(
            "Invalidated %d checkpoint(s) from %s onward for run %s",
            len(invalidated),
            _normalize_rerun_stage_name(rerun_from_stage),
            run_id,
        )

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

    # --- Stage 0: Preprocess (DISABLED) ---
    # Stage 0 split each job description into individual lines (ParsedLine objects),
    # labeled sections (e.g. "Requirements", "Qualifications"), classified boilerplate
    # (legal/EEO/benefits), mined regex-based candidate spans as LLM hints, and
    # assessed description quality. It produced structured data that stage 1 consumed
    # in 5-line batches.
    #
    # Going away because: the LLM is capable of extracting skills from the full job
    # description in a single call. The line-level preprocessing and batching added
    # complexity (45K LLM calls for 10K jobs instead of 10K), increased failure
    # surface (each HTTP call can timeout), and lost cross-line context within a job.
    # Boilerplate filtering and candidate mining are unnecessary — the model handles
    # raw text fine with 16K context.
    #
    # Stage 1 now works directly from raw jobs. Each mention carries a lightweight
    # ParsedLine with section/context so stages 2-4 work unchanged.
    #
    # stats.record_stage_start("stage0_preprocessed")
    # stage0_data = _load_or_run_stage(
    #     "stage0_preprocessed", ckpt_dir, run_id, resume, len(jobs),
    #     run_fn=lambda start_idx=0: _run_stage0_preprocess(
    #         jobs, cfg, run_id, ckpt_dir, progress_callback, start_idx=start_idx,
    #     ),
    #     progress_cb=progress_callback,
    # )
    # stats.record_stage_end("stage0_preprocessed")

    # Compute per-stage fingerprints for checkpoint invalidation
    from .prompts import (
        EXTRACTOR_V2_SYSTEM, EXTRACTOR_V2_USER_TEMPLATE,
        HARDSOFT_CLASSIFIER_SYSTEM, HARDSOFT_CLASSIFIER_USER_TEMPLATE,
        REQUIREMENT_CLASSIFIER_SYSTEM, REQUIREMENT_CLASSIFIER_USER_TEMPLATE,
        SKILL_VERIFIER_SYSTEM, SKILL_VERIFIER_USER_TEMPLATE,
    )
    fp_s1 = compute_stage_fingerprint(
        "stage1", cfg.extractor_model, cfg.backend,
        [EXTRACTOR_V2_SYSTEM, EXTRACTOR_V2_USER_TEMPLATE],
        pipeline_version=cfg.pipeline_version,
        extra_settings={"disable_thinking": cfg.disable_thinking},
    )
    fp_s2 = compute_stage_fingerprint(
        "stage2", cfg.verifier_model, cfg.backend,
        [SKILL_VERIFIER_SYSTEM, SKILL_VERIFIER_USER_TEMPLATE],
        upstream_fingerprint=fp_s1,
        pipeline_version=cfg.pipeline_version,
        extra_settings={"disable_thinking": cfg.disable_thinking},
    )
    fp_s3 = compute_stage_fingerprint(
        "stage3", cfg.requirement_model, cfg.backend,
        [REQUIREMENT_CLASSIFIER_SYSTEM, REQUIREMENT_CLASSIFIER_USER_TEMPLATE],
        upstream_fingerprint=fp_s2,
        pipeline_version=cfg.pipeline_version,
        extra_settings={"disable_thinking": cfg.disable_thinking},
    )
    fp_s4 = compute_stage_fingerprint(
        "stage4", cfg.hardsoft_model, cfg.backend,
        [HARDSOFT_CLASSIFIER_SYSTEM, HARDSOFT_CLASSIFIER_USER_TEMPLATE],
        upstream_fingerprint=fp_s2,
        pipeline_version=cfg.pipeline_version,
        extra_settings={"disable_thinking": cfg.disable_thinking},
    )

    # Store fingerprints so stage functions can embed them in checkpoint headers
    _stage_fingerprints["stage1_extracted"] = fp_s1
    _stage_fingerprints["stage2_verified"] = fp_s2
    _stage_fingerprints["stage3_requirement"] = fp_s3
    _stage_fingerprints["stage4_hardsoft"] = fp_s4

    # --- Stage 1: Extract (one LLM call per job) ---
    stats.record_stage_start("stage1_extracted")
    stage1_data = _load_or_run_stage(
        "stage1_extracted", ckpt_dir, run_id, resume, len(jobs),
        run_fn=lambda start_idx=0: _run_stage1_extract(
            jobs, cfg, run_id, ckpt_dir, progress_callback, stats, start_idx=start_idx,
        ),
        progress_cb=progress_callback,
        fingerprint=fp_s1,
    )
    if retry_stage1_errors:
        stage1_data, retried_count = _retry_failed_stage1_records(
            jobs,
            stage1_data,
            cfg,
            run_id,
            ckpt_dir,
            progress_callback,
        )
        if retried_count:
            invalidated.extend(_invalidate_checkpoints_from_stage(output_dir, run_id, "stage2"))
            logger.info(
                "Retried %d failed stage-1 job(s); invalidated downstream stage checkpoints",
                retried_count,
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
            stage1_data, cfg, run_id, ckpt_dir, progress_callback, stats,
            start_idx=start_idx,
        ),
        progress_cb=progress_callback,
        fingerprint=fp_s2,
    )
    stats.record_stage_end("stage2_verified")

    # --- Stage 3: Requirement classification ---
    stats.record_stage_start("stage3_requirement")
    stage3_data = _load_or_run_stage(
        "stage3_requirement", ckpt_dir, run_id, resume, total_mentions,
        run_fn=lambda start_idx=0: _run_stage3_requirement(
            stage1_data, stage2_data, cfg, run_id, ckpt_dir, progress_callback,
            stats, start_idx=start_idx,
        ),
        progress_cb=progress_callback,
        fingerprint=fp_s3,
    )
    stats.record_stage_end("stage3_requirement")

    # --- Stage 4: Hard/soft classification ---
    stats.record_stage_start("stage4_hardsoft")
    stage4_data = _load_or_run_stage(
        "stage4_hardsoft", ckpt_dir, run_id, resume, total_mentions,
        run_fn=lambda start_idx=0: _run_stage4_hardsoft(
            stage1_data, stage2_data, cfg, run_id, ckpt_dir, progress_callback,
            stats, start_idx=start_idx,
        ),
        progress_cb=progress_callback,
        fingerprint=fp_s4,
    )
    stats.record_stage_end("stage4_hardsoft")

    # --- Stage 5: Assemble ---
    stats.record_stage_start("stage5_assemble")
    augmented = _run_stage5_assemble(
        jobs, stage1_data, stage2_data, stage3_data, stage4_data,
        cfg, run_id, started,
    )
    stats.record_stage_end("stage5_assemble")

    cfg.llm_timing_callback = original_timing

    # A job is "successful" if it assembled without a pipeline execution error.
    # Zero mentions is valid. Stage-level extraction failures propagate into
    # extraction_metadata.error so they are not counted as successes.
    stats.jobs_success = sum(
        1 for j in augmented
        if not (j.get("extraction_metadata") or {}).get("error")
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

    write_augmented_jobs(paths["augmented_json"], augmented, pretty=cfg.export_pretty_json)
    write_mentions_jsonl(paths["mentions_jsonl"], augmented)
    write_mentions_csv(paths["mentions_csv"], augmented)

    if write_reports:
        paths["quality_csv"] = output_dir / f"SkillsExtraction_quality_run_{run_id}.csv"
        paths["frequency_csv"] = output_dir / f"SkillsExtraction_skill_frequency_run_{run_id}.csv"
        paths["low_conf_json"] = output_dir / f"SkillsExtraction_low_confidence_run_{run_id}.json"
        write_quality_report(paths["quality_csv"], augmented)
        write_frequency_report(paths["frequency_csv"], augmented)
        write_low_confidence_review(paths["low_conf_json"], augmented, threshold=0.55)

    paths["job_skills_csv"] = output_dir / f"SkillsExtraction_job_skills_run_{run_id}.csv"
    write_job_skills_summary(paths["job_skills_csv"], augmented)

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
