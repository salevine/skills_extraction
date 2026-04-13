# Skills extraction — Changelog

Application code and docs live in this directory (`skills_extraction/`).  
This file is the canonical changelog for the skills-extraction project.

---

## [3.3.0] — 2026-04-13 — Stage authority, source grounding, vLLM throughput

### Stage-authority fix (correctness)

- **Problem:** Stage 5 assembly preferred extractor-supplied `is_skill`, `requirement_level`, and `hard_soft` over later verifier/classifier results. A stage-2 rejection could be silently overridden, producing `is_skill=True` with `verifier_status="extractor"` in final output.
- **Fix:** Stages 2-4 are now authoritative when they ran. A stage-2 rejection forces `is_skill=False`. Stage-3/4 `requirement_level` and `hard_soft` override extractor values when those stages completed, errored, or parse-failed. Extractor values are used only as fallback when the later stage was skipped.
- **Traceability:** Original extractor classifications (`is_skill`, `requirement_level`, `hard_soft`) are preserved in `pipeline_audit.extractor.output` for audit purposes.

### Whole-job mention grounding (correctness)

- **Problem:** `extract_mentions_for_job()` built synthetic `ParsedLine` objects from model-returned context, hardcoded `boilerplate_label="skills_relevant"`, and used `desc_raw.find(span)` for offsets. Repeated spans could bind to the wrong occurrence; downstream stages judged model-written context instead of source text; boilerplate penalties never fired.
- **Fix:** Mentions are now anchored to real `ParsedLine` objects from the source text. The raw description is preprocessed through `preprocess_description()` → `split_inline_section_headers()` → `segment_lines()` → `label_parsed_lines()`. Each mention is grounded by exact context match (preferred), section-hint-aware span search, or deterministic position tracking for repeated spans. Ungroundable spans are skipped.
- **Offset semantics:** `char_start`/`char_end` now index into `description_normalized` (the whitespace-normalized, section-split text). Stage 5 exports `description_normalized` alongside `description_raw` so consumers have a matching reference string.

### Rolling worker pool (throughput)

- **Problem:** `_run_windowed()` created a new `ThreadPoolExecutor` for every window and blocked at each boundary. One slow mention stalled the next batch.
- **Fix:** Replaced with `_run_rolling()` — a single long-lived `ThreadPoolExecutor` with bounded submission (`max_workers * 2` items ahead of the write head). Completed results are flushed in original order as contiguous items become available. No executor rebuild, no window boundary stalls.
- **Memory:** Submission is gated by `next_submit - next_write < backlog_limit`, so both in-flight futures and buffered out-of-order results stay bounded.

### Incremental stage-1 vLLM checkpointing (resilience)

- **Problem:** The vLLM stage-1 path buffered all results in memory and wrote checkpoints only after every future completed. A crash mid-stage forced a full rerun.
- **Fix:** Checkpoint records are now written incrementally as contiguous completed jobs become available. A crash loses only the incomplete tail, not the entire stage.

### vLLM client improvements (throughput)

- **Session reuse:** `llm_vllm.py` now uses per-thread `requests.Session` instances via `threading.local()`, reusing TCP connections within each worker thread without cross-thread contention.
- **Decoupled delay:** vLLM calls use a separate `vllm_per_call_delay_sec` config field (default `0.0`) instead of the shared `per_call_delay_sec` (still `0.25` for Ollama/OpenRouter). No artificial throttling on the vLLM path by default.

### Config changes

- **New field:** `PipelineConfig.vllm_per_call_delay_sec` (default `0.0`).
- **Unchanged:** `per_call_delay_sec` (default `0.25`) still applies to Ollama/OpenRouter.

### Tests

- `test_fixes.py`: 8 regression tests covering stage-authority override, classifier authority, repeated-span grounding, ungroundable span skipping, context-match preference, offset/normalized-text consistency, thread-local session isolation, and rolling pool memory boundedness.

---

## [3.2.2] — 2026-04-12 — Checkpoint resilience, status tooling, Windows deploy

### Checkpoint error handling

- **Problem:** A single corrupt or NUL-byte line in a checkpoint file caused `load_checkpoint()` to crash with `JSONDecodeError`, preventing pipeline resume even when 99%+ of the checkpoint was valid.
- **Fix:** `load_checkpoint()` now strips NUL bytes, wraps `json.loads()` in try/except, and logs a warning for each skipped line. The pipeline continues with all valid records.

### Operational tooling

- **`check_status.sh`:** Comprehensive run status report. Usage: `./check_status.sh` (all runs) or `./check_status.sh <RUN_ID>` (single run). Reports:
  - Per-stage status (COMPLETE/INCOMPLETE) with record counts and timestamps
  - Error counts per stage (`"status": "error"` records)
  - Models and backend config from checkpoint headers
  - Progress percentage for incomplete stages
  - Active process detection (`pgrep`)
  - Run summary (wall clock, mentions, LLM timing) from `run_summary.json` if available
  - Last 5 lines of the run's log file
- **`deploy.ps1`:** Windows PowerShell deployment script (replaces Mac-only `deploy.sh`). Uses rsync over SSH, supports `-s` flag to copy SampleJobs.json.
- **`deploy.bat`:** Windows cmd.exe alternative for environments without PowerShell.

---

## [3.2.1] — 2026-04-02 — vLLM deadlock fix, thinking suppression, server deployment

### Bug fix: Stage 1 vLLM deadlock

- **Root cause:** The v3.2.0 windowed execution parallelized at the **job level** — each thread processed an entire job, which required multiple extractor batches (one per 5 lines). With 7 endpoints and 7 concurrent jobs, all threads would check out an endpoint for their first batch, then block indefinitely waiting for a second endpoint that would never become available. This caused a silent deadlock where the process appeared alive but made zero progress.
- **Observed symptoms:** Stage 1 checkpoint file stayed at ~142 bytes (header only), no log output, no errors, GPU utilization at 0% on all cards. The process had to be killed externally.
- **Fix:** Replaced job-level windowed parallelism with **batch-level parallelism**. All batches across all jobs are flattened into a single list, then processed in windows of N (= number of endpoints). Each batch in a window is assigned a **dedicated endpoint URL** via the new `call_vllm_direct()` function, completely bypassing the shared endpoint pool. Results are reassembled per-job after all batches complete, preserving checkpoint ordering.
- **Why `call_vllm_direct()`:** The existing `call_vllm()` uses a module-level `queue.Queue` endpoint pool with blocking checkout. Even at the batch level, the pool's checkout/return cycle introduced contention under `ThreadPoolExecutor`. Direct endpoint assignment eliminates all shared state between threads.
- **Stages 2–4 unaffected:** These stages process individual mentions (one LLM call per item), so the original `_run_windowed()` with the endpoint pool works correctly — each thread checks out one endpoint, makes one call, and returns it.
- **Ollama/OpenRouter paths unchanged:** The sequential code path for non-vLLM backends was preserved exactly as-is.

### vLLM thinking-mode suppression

- **Problem:** Qwen3-14B on vLLM was generating extensive chain-of-thought reasoning (observed: ~400 reasoning tokens per call in the `reasoning_content` field) before producing the actual JSON output. This wasted GPU time and caused individual batches to take 30–80 seconds, with some hitting the 300-second timeout.
- **Observed impact:** A 20-job sample with 86 batches across 7 GPUs estimated ~70 minutes; only 3 of 8 GPUs were active at any given time because slow batches blocked each window.
- **Fix:** Added `"chat_template_kwargs": {"enable_thinking": false}` to the vLLM request payload in both `call_vllm_direct()` and `call_vllm()` when `cfg.disable_thinking` is `True` (which is the default). This is the vLLM equivalent of the Ollama `/no_think` suffix that was already in place since v3.2.0.
- **Expected speedup:** 3–5x per batch, reducing per-window time from ~60s to ~10–15s.

### Server deployment tooling

- **`deploy.sh`:** Rsync-based deployment script that copies the project to `stacey@titan3.cs.gsu.edu:~/skills_extraction/` and the sample jobs to `~/jobs/SampleJobs.json`. Excludes `.venv`, `__pycache__`, `.git`, and output directories.
- **`vLLM_run.sh`:** Server-side run script using `conda run -n skills` (avoids `conda init` requirement). Configured for 8 endpoints on ports 8000–8007 with `Qwen/Qwen3-14B` model name (matching the HuggingFace model ID that vLLM registers, not the Ollama tag format `qwen3:14b`).
- **`test_vllm.sh`:** Endpoint health checker that curls all 8 ports and reports HTTP status codes.
- **`test_simple.py`:** Minimal single-request test script for verifying vLLM chat/completions API connectivity.

### Deployment lessons learned

- **Model name mismatch:** vLLM registers models by HuggingFace ID (`Qwen/Qwen3-14B`), not Ollama-style tags (`qwen3:14b`). The `--extractor-model` flag must match exactly or all requests return 404.
- **Port configuration:** The `--vllm-num-endpoints` and `--vllm-base-port` flags are only applied when using the `--vllm` CLI flag, not `--backend vllm`. The latter sets the backend but does not trigger the vLLM-specific argument overrides in the CLI.
- **`conda run` buffering:** `conda run` buffers all stdout/stderr by default, making the pipeline appear to hang. The `--no-capture-output` flag is required to see real-time progress.
- **Server has 8x NVIDIA RTX A6000 GPUs** (49 GB VRAM each), with vLLM workers on ports 8000–8007.

---

## [3.2.0] — 2026-04-01 — Concurrent vLLM workers and multi-backend consolidation

### Multi-backend architecture

- **vLLM backend:** Added `llm_vllm.py` providing an OpenAI-compatible HTTP client for multi-GPU vLLM deployments. Endpoints are configured via `--vllm-host`, `--vllm-base-port`, and `--vllm-num-endpoints`; default is 8 endpoints on ports 8001–8008.
- **OpenRouter backend:** Added `llm_openrouter.py` for cloud-hosted model inference via the OpenRouter API. Configured via `OPENROUTER_API_KEY` env var.
- **Unified dispatcher:** `llm_backend.py` routes LLM calls to Ollama, vLLM, or OpenRouter based on `--backend` / `SKILLS_BACKEND`. All three share the same call signature (`cfg, model, system, user, temperature, role`), so pipeline stages are backend-agnostic.

### Concurrent vLLM workers

- **Endpoint pool:** Replaced `itertools.cycle` round-robin with a `queue.Queue` checkout/return pattern in `llm_vllm.py`. Endpoints are checked out before each request and returned immediately after completion (or failure, before retry sleep). This provides natural backpressure: if all 8 GPUs are busy, additional requests block until one becomes available, rather than blindly dispatching to an overloaded endpoint. Round-robin was insufficient because response times varied from 4.8s to 81.6s per batch during testing, meaning fast GPUs would idle while waiting for the cycle to return to them.
- **Windowed parallel execution:** Added `_run_windowed()` helper in `pipeline.py` that processes items in windows of N (= `vllm_num_endpoints`) using `ThreadPoolExecutor`. Within each window, all items are submitted concurrently to different GPU endpoints. After the window completes, results are sorted by original index and checkpoints are written in order from the main thread. This preserves checkpoint ordering guarantees while enabling parallelism. Crash mid-window loses at most N items.
- **Per-item processing functions:** Extracted `_process_extract_job()`, `_process_verify_item()`, `_process_requirement_item()`, and `_process_hardsoft_item()` as standalone functions suitable for concurrent execution. Each is stateless with respect to shared mutable state.
- **Thread-safe statistics:** Added `_stats_lock: threading.Lock` to `RunStats` and wrapped `record_llm()` in a lock, since the timing callback is invoked from worker threads during concurrent vLLM execution.
- **Backend-conditional branching:** Stages 1–4 now check `cfg.backend == "vllm"` and use windowed execution when true; Ollama and OpenRouter paths remain fully sequential and unchanged.
- **Progress callbacks:** Fire from the main thread during the ordered checkpoint-write phase, not from worker threads, preventing stdout interleaving.

### Operational hardening

- **Debug logging:** Added per-call tok/s logging in the vLLM client for throughput monitoring during multi-day extractions.
- **Thinking-mode suppression:** `disable_thinking` defaults to `True`, appending `/no_think` to Qwen3 prompts. This eliminates the extended reasoning output that was observed to waste 2–4x tokens per call without improving extraction quality.
- **Timeout flag:** `--timeout` CLI flag for configurable HTTP timeout (default 300s), addressing 502/503 errors during sustained multi-hour runs.
- **Run-ID resume fix:** Fixed run-id collision when resuming with the same label by appending timestamps to provided labels.

### Versioning

- Pipeline version bumped to `3.2.0`.

---

## [3.1.0] — 2026-03-27 — Stage-first execution with intermediate checkpoints

- **Stage-first pipeline:** `run_pipeline()` now processes ALL jobs through each stage before advancing to the next, replacing the previous job-at-a-time loop. Stages execute in order: preprocess (0) → extract (1) → verify (2) → requirement classify (3) → hard/soft classify (4) → assemble (5).
- **Model swap reduction:** Under the previous architecture, each job cycled through all four LLM stages, causing the Ollama server to swap between the extractor model (`qwen3:14b`) and the verifier/classifier model (`mistral-nemo:12b`) on every job — thousands of swaps for large corpora. The stage-first design reduces this to exactly one model transition per run (after extraction completes, the verifier model loads once and remains resident for stages 2–4). For a 10,000-job corpus, this eliminates an estimated 30,000+ model swap events.
- **Intermediate checkpoints:** Each stage writes incremental JSONL checkpoint files under `output_dir/checkpoints/{run_id}_{stage}.jsonl`. Checkpoints use a header/footer protocol: a `_meta` line records run context, data lines are flushed on each record, and a `_complete` trailer marks successful stage completion. This enables crash-recovery resume without re-executing completed work.
- **Resume-on-crash:** If a run is interrupted, restarting with the same `--run-id` automatically detects completed checkpoints (skips the stage entirely) and partial checkpoints (resumes from the last written record). The `--no-resume` flag forces a clean start.
- **New module `checkpoint.py`:** Encapsulates checkpoint path resolution, completeness detection, record counting, JSONL read/write, and serialization helpers for `ParsedLine`, `CandidateSpan`, and mention dicts (including `_parsed_line` object ↔ `_parsed_line_dict` conversion).
- **Per-stage timing:** `RunStats` now records wall-clock duration for each stage via `record_stage_start()`/`record_stage_end()`. Stage timing appears in both `run_summary.json` and the human-readable log block.
- **Run ID collision prevention:** When `--run-id` is provided, the CLI now appends a timestamp (`{label}_{YYYYMMDD_HHMMSS}`), preventing checkpoint collisions across runs with the same label.
- **HTTP connection pooling:** `llm_ollama.py` now uses a persistent `requests.Session` for all Ollama calls, eliminating per-call TCP connection setup. Estimated savings: ~0.3s per LLM call, or ~3–4 hours across a full 10,000-job extraction run.
- **Log noise reduction:** `urllib3.connectionpool` debug logging suppressed to WARNING level; pipeline logs now contain only stage-level events.
- **Progress display:** CLI progress bar now shows the current stage name (e.g., `stage1_extract | batch 3/6 for J15510`).
- **Backward compatibility:** `process_single_job()` is retained with unchanged behavior for callers that process jobs individually outside `run_pipeline()`.

---

## [3.0.0] — 2026-03-23 — Role-specialized staged pipeline + full audit trail

- **Pipeline redesign:** Extraction now runs as explicit stages: **span extractor → skill verifier → requirement classifier → hard/soft classifier**.
- **Role-specialized prompts:** Split prompt contracts by stage (candidate span detection, skill validity, requirement level, hard/soft class) for clearer responsibilities and lower prompt cross-talk.
- **New stage modules:** Added `llm_requirement_classifier.py` and `llm_hardsoft_classifier.py` and integrated them into `pipeline.py`.
- **Auditability (mention-level):** Each mention now includes `pipeline_audit` with per-stage `status`, `model`, and structured `output`/`error`, so failures are traceable to a specific stage.
- **Auditability (job-level):** Added `pipeline_stage_audit` with stage counters (`*_parse_failed`, rejected mentions, stage errors) and model snapshot for the job.
- **Schema extensions:** Added `requirement_*` and `hardsoft_*` fields to `SkillMention` (model, status, confidence, notes).
- **Confidence update:** `compute_final_confidence` now blends requirement/hard-soft confidences (when available) and applies penalties for classifier parse failures or errors.
- **Config/CLI expansion:** New config keys and CLI flags for stage models and toggles:
  - Models: `requirement_model`, `hardsoft_model`
  - Flags: `--requirement-model`, `--hardsoft-model`, `--no-requirement-classifier`, `--no-hardsoft-classifier`
  - Env vars: `SKILLS_REQUIREMENT_MODEL`, `SKILLS_HARDSOFT_MODEL`
- **Exports:** JSONL/CSV mention exports now include requirement/hard-soft stage statuses/models and serialized `pipeline_audit`.
- **Versioning:** Pipeline/package version bumped to `3.0.0`.

---

## [2.0.2] — 2026-03-23 — Precision / heuristics (review feedback)

- **Candidate mining:** Gated `comma_list_token` to requirement sections or skill-cue lines; stricter tool-shaped tokens; large **stoplist** for generic nouns/verbs; replaced noisy bare `using|in` mining with **prep_phrase_strong** (head must be experience/knowledge/etc.).
- **Sectioning:** **`split_inline_section_headers`** inserts newlines before common inline headings (Minimum Qualifications, Responsibilities:, etc.); **`detect_section_header`** handles colon-led inline headers; normalized text stored in jobs matches offsets.
- **Boilerplate:** More **conservative** — `skills_relevant` only with lexical skill/requirement evidence; overview/body default **uncertain**.
- **Descriptions:** Removed **company/location fallback** as pseudo-description; empty → quality **malformed** / clean skip.
- **Verifier:** JSON parse failure → **`verifier_status=parse_failed`**, **`verifier_confidence=null`**, confidence **penalty**; low-confidence export includes parse failures.
- **Offsets:** Extractor repair prefers **valid model offsets**, else match **nearest** to proposed start, else **evidence-aligned** match among occurrences.
- **Mentions:** **`support_count=0`**, empty **`models_that_found_it`**, **`support_metadata`** explains single-pass; **`verifier_confidence`** optional (null when parse failed).

---

## [2.0.1] — 2026-03-23 — Package layout

- All v2 application modules and CLI live under **`skills_extraction/`**.
- Run: **`python -m skills_extraction`** from the repository root, or **`python Runskills_extraction.py`** using the thin shim at repo root.
- Imports inside the package use **relative** imports.

---

## [2.0.0] — 2026-03-23 — Production pipeline redesign

The previous monolithic script (summary-first, whole-document LLM, Excel-centric v1) is **replaced** by this package. The repo root **`Runskills_extraction.py`** is a **shim** only.

### Objectives implemented

- **Open vocabulary** — no fixed skill list; patterns + LLM span validation only.
- **Per-line / evidence-first** — `parsed_lines` with `line_id`, section, offsets; mentions carry `source_line`, global `char_start`/`char_end` in `description_normalized`.
- **Audit trail** — `skill_candidates`, `skill_mentions`, `rules_fired`, extractor/verifier models, `verifier_status`, timestamps, `run_id`.
- **Append-only jobs** — `io_utils.augment_job_record` deep-copies and adds fields; originals preserved.
- **No early deduplication** — all candidate spans kept; mentions kept per model output (including `is_skill: false` when rejected).
- **Derived reports secondary** — optional quality / frequency / low-confidence exports; primary artifact is augmented JSON + JSONL/CSV per mention.

### Stages (see `pipeline.py`)

1. Ingestion + stable `job_key`  
2. Preprocessing (`description_raw` + `description_normalized`)  
3. Section + line segmentation  
4. Document quality scoring (`quality.py`)  
5. Boilerplate line labels (`boilerplate.py`)  
6. Candidate harvesting (`candidate_mining.py`) — recall-oriented rules  
7. LLM span validation in small batches (`llm_extractor.py`)  
8. Conditional verifier (`llm_verifier.py`)  
9. Multi-signal confidence (`confidence.py`)  
10. Exports: augmented JSON, JSONL, CSV + reports (`exporters.py`)

### CLI (v2)

- `--input`, `--output-dir`, `--run-id`, `--local`, `--sample`, `--skip-llm`, `--no-verifier`, `--no-reports`, model overrides, `--batch-lines`, `--context-size`.

### Artifacts

- `SkillsExtraction_augmented_run_{run_id}.json`  
- `SkillsExtraction_mentions_run_{run_id}.jsonl` / `.csv`  
- `SkillsExtraction_pipeline_run_{run_id}.log`  
- Optional: quality, skill frequency, low-confidence JSON reports  

### Example / schema

- `example_augmented_job_fragment.json`

### v1 note

The older Excel ensemble workflow lived in a large single-file **`Runskills_extraction.py`**. Restore from version control if needed.

---

## Historical — v1-era (2026-02-09)

See archived notes in git history for: Run ID / Skills_Row_ID Excel naming, Ollama `--local`, ensemble voting, `finalize_skills_results`, etc. (superseded by v2 mention-level JSON pipeline.)

---

## Notes

- Legacy copies under **`Skills/`** (`Run_skills_extraction.py`, etc.) are not kept in sync automatically.
