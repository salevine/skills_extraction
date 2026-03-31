# Skills extraction — Changelog

Application code and docs live in this directory (`skills_extraction/`).  
This file is the canonical changelog for the skills-extraction project.

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
