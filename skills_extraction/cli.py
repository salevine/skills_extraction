# ============================================================================
# SKILLS EXTRACTION v2 — CLI (open-vocabulary, line-level, audit-first)
# ============================================================================
# Run from repository root:
#   python -m skills_extraction --input SampleJobs.json --output-dir ./skills_out
# Or use the root shim: python Runskills_extraction.py ...
# See README.md in this directory.
# ============================================================================

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from pathlib import Path

from .config import PipelineConfig, load_config_from_env, resolve_ollama_base_url
from .io_utils import load_jobs_json
from .pipeline import run_pipeline

logger = logging.getLogger(__name__)

_PROGRESS_START = time.perf_counter()


def _find_latest_run_id(ckpt_dir: Path) -> str:
    """Find the most recent run_id from checkpoint filenames."""
    if not ckpt_dir.exists():
        return ""
    import re
    run_ids: dict[str, float] = {}
    for f in ckpt_dir.glob("*.jsonl"):
        m = re.match(r"^(.+?)_stage\d+_\w+\.jsonl$", f.name)
        if m:
            rid = m.group(1)
            mtime = f.stat().st_mtime
            if rid not in run_ids or mtime > run_ids[rid]:
                run_ids[rid] = mtime
    if not run_ids:
        return ""
    return max(run_ids, key=run_ids.get)


def _format_eta(remaining_sec: float) -> str:
    """Format seconds as e.g. '2m 15s' or '45s'."""
    if remaining_sec <= 0 or not (remaining_sec < 86400):
        return "?m ?s"
    m = int(remaining_sec // 60)
    s = int(remaining_sec % 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _configure_logging(log_path: Path, quiet_console: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING if quiet_console else logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)


def generate_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="skills_extraction",
        description="Skills extraction v2: mention-level, line-aware, Ollama-backed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from repository root — the folder that contains `skills_extraction/`):
  python -m skills_extraction --input SampleJobs.json --output-dir ./skills_out --sample 10
  python -m skills_extraction -i jobs.json -o ./out --run-id 20260209_120000 --extractor-model qwen2.5:14b
  python -m skills_extraction -i jobs.json -o ./out --skip-llm
        """,
    )
    parser.add_argument("--input", "-i", default=str(Path("../jobs/SampleJobs.json")), help="Input JSON (list of jobs or {jobs: [...]}); default: ../jobs/SampleJobs.json")
    parser.add_argument("--output-dir", "-o", default="skills_extraction_output", help="Directory for outputs")
    parser.add_argument("--run-id", default="", help="Run id (default: timestamp YYYYMMDD_HHMMSS)")
    parser.add_argument("--local", action="store_true", help="Use Ollama at http://localhost:11434")
    parser.add_argument("--extractor-model", default="", help="Override extractor model tag (default: qwen2.5:14b or env)")
    parser.add_argument("--verifier-model", default="", help="Override verifier model tag")
    parser.add_argument("--requirement-model", default="", help="Override requirement classifier model tag")
    parser.add_argument("--hardsoft-model", default="", help="Override hard/soft classifier model tag")
    parser.add_argument("--fallback-model", default="", help="Override fallback model if extractor fails")
    parser.add_argument("--sample", type=int, default=0, help="Process only first N jobs after load")
    parser.add_argument("--no-verifier", action="store_true", help="Disable verifier pass")
    parser.add_argument("--no-requirement-classifier", action="store_true", help="Disable requirement classifier pass")
    parser.add_argument("--no-hardsoft-classifier", action="store_true", help="Disable hard/soft classifier pass")
    parser.add_argument("--skip-llm", action="store_true", help="Skip Ollama (structure + candidates only)")
    parser.add_argument("--no-reports", action="store_true", help="Skip derived CSV summary reports")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints; overwrite from scratch")
    parser.add_argument("--resume-latest", action="store_true", help="Resume the most recent run from checkpoints")
    parser.add_argument(
        "--rerun-from",
        choices=["stage1", "stage2", "stage3", "stage4"],
        default="",
        help="Delete checkpoints from this stage onward before resuming (for example: stage2 to reuse stage1 and rerun stages 2-4)",
    )
    parser.add_argument(
        "--retry-stage1-errors",
        action="store_true",
        help="When reusing an existing stage1 checkpoint, retry only records that still have stage1_error before downstream stages run",
    )
    parser.add_argument("--backend", choices=["ollama", "openrouter", "vllm"], default="", help="LLM backend (default: ollama, or SKILLS_BACKEND env)")
    parser.add_argument("--batch-lines", type=int, default=5, help="Max lines per extractor LLM call")
    parser.add_argument("--context-size", type=int, default=32768, help="Ollama num_ctx")
    parser.add_argument("--timeout", type=int, default=300, help="Ollama HTTP timeout in seconds (default: 300)")

    # vLLM backend options
    parser.add_argument("--vllm", action="store_true", help="Use vLLM backend instead of Ollama")
    parser.add_argument("--vllm-host", default="localhost", help="vLLM server hostname (default: localhost)")
    parser.add_argument("--vllm-base-port", type=int, default=8001, help="vLLM first endpoint port (default: 8001)")
    parser.add_argument("--vllm-num-endpoints", type=int, default=8, help="Number of vLLM endpoints (default: 8)")
    args = parser.parse_args()

    label = args.run_id.strip()
    if args.resume_latest:
        run_id = _find_latest_run_id(Path(args.output_dir) / "checkpoints")
        if not run_id:
            print("ERROR: --resume-latest: no checkpoint files found in", Path(args.output_dir) / "checkpoints")
            sys.exit(1)
        print(f"Resuming latest run: {run_id}")
    elif label and not args.no_resume:
        # When resuming with an explicit run-id, use it as-is so checkpoints match
        run_id = label
    else:
        ts = generate_run_id()
        run_id = f"{label}_{ts}" if label else ts
    out_dir = Path(args.output_dir)
    log_file = out_dir / f"SkillsExtraction_pipeline_run_{run_id}.log"
    _configure_logging(log_file)

    overrides = {
        "ollama_base_url": resolve_ollama_base_url(use_local=args.local),
        "ollama_context_size": args.context_size,
        "ollama_timeout_sec": args.timeout,
        "extractor_batch_max_lines": max(1, args.batch_lines),
        "verifier_enabled": not args.no_verifier,
        "requirement_classifier_enabled": not args.no_requirement_classifier,
        "hardsoft_classifier_enabled": not args.no_hardsoft_classifier,
        "skip_llm": args.skip_llm,
    }
    if args.extractor_model:
        overrides["extractor_model"] = args.extractor_model
    if args.verifier_model:
        overrides["verifier_model"] = args.verifier_model
    if args.fallback_model:
        overrides["fallback_model"] = args.fallback_model
    if args.requirement_model:
        overrides["requirement_model"] = args.requirement_model
    if args.hardsoft_model:
        overrides["hardsoft_model"] = args.hardsoft_model
    if args.backend:
        overrides["backend"] = args.backend

    if args.vllm:
        overrides["backend"] = "vllm"
        overrides["vllm_host"] = args.vllm_host
        overrides["vllm_base_port"] = args.vllm_base_port
        overrides["vllm_num_endpoints"] = args.vllm_num_endpoints

    cfg: PipelineConfig = load_config_from_env(overrides)

    jobs, src = load_jobs_json(args.input)
    if args.sample and args.sample > 0:
        jobs = jobs[: args.sample]

    print("=" * 70)
    print(f"Skills extraction v2 | run_id={run_id}")
    print(f"Input: {src} ({len(jobs)} jobs)")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Backend: {cfg.backend}")
    if cfg.backend == "vllm":
        endpoints = cfg.vllm_endpoints()
        print(f"vLLM: {len(endpoints)} endpoints")
        print(f"  {endpoints[0]} ... {endpoints[-1]}")
    elif cfg.backend == "openrouter":
        print(f"OpenRouter: {cfg.openrouter_base_url}")
    else:
        print(f"Ollama: {cfg.ollama_base_url}")
    print(
        "Extractor: "
        f"{cfg.extractor_model} | "
        f"Verifier: {cfg.verifier_model} | "
        f"Req: {cfg.requirement_model} | "
        f"HardSoft: {cfg.hardsoft_model} | "
        f"skip_llm={cfg.skip_llm}"
    )
    print(f"Checkpoints: {(out_dir / 'checkpoints').resolve()}")
    print(f"Resume: {not args.no_resume}")
    if args.rerun_from:
        print(f"Rerun from: {args.rerun_from}")
    if args.retry_stage1_errors:
        print("Retry stage1 errors: True")
    print(f"Log: {log_file}")
    print("=" * 70)
    print()

    global _PROGRESS_START
    _PROGRESS_START = time.perf_counter()
    _stage_starts: dict = {}
    _last_stage: list = [""]

    def _on_progress(job_idx: int, total: int, stage: str, detail: str, extra: object) -> None:
        now = time.perf_counter()

        # Detect stage transition — print newline and header
        stage_label = ""
        if isinstance(extra, dict):
            stage_label = extra.get("stage", "")
        current_stage = stage_label or stage
        if current_stage != _last_stage[0]:
            if _last_stage[0]:
                # Finish previous stage line
                print()
            _last_stage[0] = current_stage
            _stage_starts[current_stage] = now
            stage_display = current_stage.replace("_", " ").title()
            print(f"\n  {stage_display}", flush=True)

        # Calculate ETA from stage start
        stage_start = _stage_starts.get(current_stage, now)
        elapsed = now - stage_start
        pct = 100 * (job_idx + 1) / max(1, total)
        pct = min(99.9, pct)
        if job_idx + 1 < total and elapsed > 2:
            rate = (job_idx + 1) / elapsed
            remaining_sec = (total - job_idx - 1) / rate if rate > 0 else 0
            eta = _format_eta(remaining_sec)
        else:
            eta = "..."

        bar_w = 30
        filled = int(bar_w * pct / 100)
        bar = "█" * filled + "░" * (bar_w - filled)

        # Truncate detail to avoid wrapping
        max_detail = 40
        detail_safe = (detail or "").encode("ascii", errors="replace").decode("ascii")
        if len(detail_safe) > max_detail:
            detail_safe = detail_safe[:max_detail - 1] + "…"

        msg = f"  {bar} {pct:5.1f}% [{job_idx + 1:>{len(str(total))}}/{total}] ETA {eta:>8s}  {detail_safe}"
        # Pad with spaces to overwrite previous longer lines
        try:
            print(f"\r{msg:<100}", end="", flush=True)
        except UnicodeEncodeError:
            print(f"\r  {pct:.0f}% [{job_idx + 1}/{total}] ETA {eta}", end="", flush=True)

    _configure_logging(log_file, quiet_console=True)
    augmented, paths, stats = run_pipeline(
        jobs,
        cfg,
        out_dir,
        run_id,
        write_reports=not args.no_reports,
        progress_callback=_on_progress,
        log_path=log_file,
        resume=not args.no_resume,
        rerun_from_stage=args.rerun_from or None,
        retry_stage1_errors=args.retry_stage1_errors,
    )
    _configure_logging(log_file, quiet_console=False)

    wall = time.perf_counter() - _PROGRESS_START
    print("\r" + " " * 78 + "\r", end="")
    print("-" * 70)
    print("Done.")
    print(f"  Jobs: {stats.jobs_success} succeeded, {stats.jobs_failed} failed")
    print(f"  Mentions: {stats.mentions_total}")
    print(f"  Wall clock: {wall:.1f}s")
    print()
    print("Artifacts:")
    name_map = {
        "augmented_json": "Augmented jobs (JSON)",
        "mentions_jsonl": "Mentions (JSONL)",
        "mentions_csv": "Mentions (CSV)",
        "quality_csv": "Quality report",
        "frequency_csv": "Skill frequency",
        "low_conf_json": "Low-confidence queue",
        "run_summary_json": "Run summary (for papers)",
    }
    for k, p in paths.items():
        label = name_map.get(k, k)
        print(f"  {label}: {p.name}")
    print()
    rs_path = paths.get("run_summary_json")
    print(f"Run summary and full timing written to log" + (f" + {rs_path.name}" if rs_path else ""))
    print("-" * 70)
