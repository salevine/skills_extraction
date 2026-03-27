"""
Run statistics: LLM timing, job counts, mention counts — for logs and papers.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RunStats:
    """Accumulates timing and counts for a pipeline run."""

    run_id: str
    started_at: str
    pipeline_version: str
    extractor_model: str
    verifier_model: str
    requirement_model: str
    hardsoft_model: str
    fallback_model: str
    ollama_base_url: str
    skip_llm: bool
    batch_max_lines: int
    verifier_enabled: bool
    requirement_classifier_enabled: bool
    hardsoft_classifier_enabled: bool

    # Counters
    jobs_total: int = 0
    jobs_success: int = 0
    jobs_failed: int = 0
    mentions_total: int = 0

    # LLM timing: model -> (calls, total_seconds)
    _extractor_by_model: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: {"calls": 0.0, "sec": 0.0}))
    _verifier_by_model: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: {"calls": 0.0, "sec": 0.0}))

    # Per-stage wall-clock timing
    _stage_timing: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def record_stage_start(self, stage: str) -> None:
        """Record the start time for a pipeline stage."""
        self._stage_timing[stage] = {
            "started_at": time.perf_counter(),
            "wall_sec": 0.0,
        }

    def record_stage_end(self, stage: str) -> None:
        """Record the end time for a pipeline stage and compute duration."""
        entry = self._stage_timing.get(stage)
        if entry and "started_at" in entry:
            entry["wall_sec"] = round(time.perf_counter() - entry["started_at"], 3)

    def record_llm(self, model: str, elapsed_sec: float, role: str = "extractor") -> None:
        """Record an LLM call. role is 'extractor' or 'verifier'."""
        if role == "extractor":
            self._extractor_by_model[model]["calls"] += 1
            self._extractor_by_model[model]["sec"] += elapsed_sec
        else:
            self._verifier_by_model[model]["calls"] += 1
            self._verifier_by_model[model]["sec"] += elapsed_sec

    def to_dict(self) -> Dict[str, Any]:
        completed = ""
        wall_sec = 0.0
        try:
            import datetime as dt
            completed = dt.datetime.now(dt.timezone.utc).isoformat()
            start_dt = dt.datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            end_dt = dt.datetime.fromisoformat(completed.replace("Z", "+00:00"))
            wall_sec = (end_dt - start_dt).total_seconds()
        except Exception:
            pass

        extractor_detail = []
        ext_total_calls = 0
        ext_total_sec = 0.0
        for model, d in self._extractor_by_model.items():
            c, s = d["calls"], d["sec"]
            extractor_detail.append({
                "model": model,
                "calls": int(c),
                "total_sec": round(s, 3),
                "sec_per_call_avg": round(s / c, 3) if c > 0 else 0,
            })
            ext_total_calls += int(c)
            ext_total_sec += s

        verifier_detail = []
        ver_total_calls = 0
        ver_total_sec = 0.0
        for model, d in self._verifier_by_model.items():
            c, s = d["calls"], d["sec"]
            verifier_detail.append({
                "model": model,
                "calls": int(c),
                "total_sec": round(s, 3),
                "sec_per_call_avg": round(s / c, 3) if c > 0 else 0,
            })
            ver_total_calls += int(c)
            ver_total_sec += s

        return {
            "run_id": self.run_id,
            "pipeline_version": self.pipeline_version,
            "started_at": self.started_at,
            "completed_at": completed,
            "wall_clock_sec": round(wall_sec, 2),
            "jobs": {
                "total": self.jobs_total,
                "success": self.jobs_success,
                "failed": self.jobs_failed,
            },
            "mentions_total": self.mentions_total,
            "config_snapshot": {
                "extractor_model": self.extractor_model,
                "verifier_model": self.verifier_model,
                "requirement_model": self.requirement_model,
                "hardsoft_model": self.hardsoft_model,
                "fallback_model": self.fallback_model,
                "ollama_base_url": self.ollama_base_url,
                "skip_llm": self.skip_llm,
                "batch_max_lines": self.batch_max_lines,
                "verifier_enabled": self.verifier_enabled,
                "requirement_classifier_enabled": self.requirement_classifier_enabled,
                "hardsoft_classifier_enabled": self.hardsoft_classifier_enabled,
            },
            "llm_timing": {
                "extractor": {
                    "total_calls": ext_total_calls,
                    "total_sec": round(ext_total_sec, 3),
                    "sec_per_call_avg": round(ext_total_sec / ext_total_calls, 3) if ext_total_calls > 0 else 0,
                    "by_model": extractor_detail,
                },
                "verifier": {
                    "total_calls": ver_total_calls,
                    "total_sec": round(ver_total_sec, 3),
                    "sec_per_call_avg": round(ver_total_sec / ver_total_calls, 3) if ver_total_calls > 0 else 0,
                    "by_model": verifier_detail,
                },
            },
            "stage_timing": {
                stage: {"wall_sec": info["wall_sec"]}
                for stage, info in self._stage_timing.items()
            },
        }

    def format_for_log(self) -> str:
        """Human-readable block for appending to log file."""
        d = self.to_dict()
        lines = [
            "",
            "=" * 70,
            "RUN SUMMARY (for reproducibility / papers)",
            "=" * 70,
            f"run_id: {d['run_id']}",
            f"pipeline_version: {d['pipeline_version']}",
            f"started_at: {d['started_at']}",
            f"completed_at: {d['completed_at']}",
            f"wall_clock_sec: {d['wall_clock_sec']}",
            "",
            "Jobs:",
            f"  total: {d['jobs']['total']}  success: {d['jobs']['success']}  failed: {d['jobs']['failed']}",
            f"mentions_total: {d['mentions_total']}",
            "",
            "Config:",
        ]
        for k, v in d["config_snapshot"].items():
            lines.append(f"  {k}: {v}")

        ext = d["llm_timing"]["extractor"]
        ver = d["llm_timing"]["verifier"]
        lines.extend([
            "",
            "LLM timing — extractor:",
            f"  total_calls: {ext['total_calls']}  total_sec: {ext['total_sec']}  sec_per_call_avg: {ext['sec_per_call_avg']}",
        ])
        for m in ext["by_model"]:
            lines.append(f"    {m['model']}: {m['calls']} calls, {m['total_sec']} sec, {m['sec_per_call_avg']} sec/call avg")

        lines.extend([
            "",
            "LLM timing — verifier:",
            f"  total_calls: {ver['total_calls']}  total_sec: {ver['total_sec']}  sec_per_call_avg: {ver['sec_per_call_avg']}",
        ])
        for m in ver["by_model"]:
            lines.append(f"    {m['model']}: {m['calls']} calls, {m['total_sec']} sec, {m['sec_per_call_avg']} sec/call avg")

        stage_t = d.get("stage_timing", {})
        if stage_t:
            lines.extend(["", "Stage timing:"])
            for stage, info in stage_t.items():
                lines.append(f"  {stage}: {info['wall_sec']}s")

        lines.append("")
        lines.append(json.dumps(d, indent=2))
        lines.append("=" * 70)
        return "\n".join(lines)
