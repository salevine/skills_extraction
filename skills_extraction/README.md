# Skills extraction (v3) — full guide

Open-vocabulary, mention-level skill extraction with multi-backend LLM support (Ollama, vLLM, OpenRouter).
The **source of truth** is **mention-level** JSON (plus JSONL/CSV exports). Summary reports are **derived**, not primary.

This README is written to match the **current default code path** in `cli.py -> run_pipeline()`, not just the older design docs. See **`CHANGELOG.md`** for version history.

## Current status

- **Default execution path:** `run_pipeline()` runs **whole-job extraction** in stage 1, then mention-level verification / requirement / hard-soft passes in stages 2-4. Stages 2-4 are **authoritative** — their results override stage-1 extractor classifications in the final assembly.
- **Source grounding:** Stage-1 mentions are anchored to real `ParsedLine` objects from the preprocessed source text (sectioned, boilerplate-labeled). The preprocessing/sectioning/boilerplate modules are now exercised on the default hot path for grounding, not just the legacy `process_single_job()` path.
- **Scaled backend path:** vLLM is the primary production backend. Stage 1 uses direct endpoint assignment per job; stages 2-4 use a rolling worker pool with bounded submission for continuous throughput.
- **Offset reference text:** `char_start`/`char_end` on mentions index into `description_normalized` (whitespace-collapsed, section-split). Both `description_raw` and `description_normalized` are exported in the augmented output.

---

## What this actually does (read this first)

Job postings are messy. They mix real requirements with legal boilerplate, perks, and vague “we’re a family” fluff. They also don’t come with a neat list of skills—you have to *read* them. This application is a **pipeline** that takes a collection of job records (usually JSON), pulls out the **description text**, and tries to surface **what the employer is actually asking for** in terms of tools, technologies, methods, and competencies—without forcing everything into a fixed skill taxonomy up front.

**NLP side of the house.** The classical text-processing stack (description normalization, inline section splitting, line segmentation, boilerplate labeling) is now used on the default hot path for **mention grounding**: after stage 1 extracts spans from the full raw description, each mention is anchored to a real `ParsedLine` with accurate section labels, boilerplate classification, and character offsets into the normalized text. Quality scoring and candidate mining remain in the repo but are only exercised by the legacy `process_single_job()` path.

**Finding candidates before the LLM.** Deterministic candidate mining is still implemented in `candidate_mining.py`, but it is currently part of the **legacy / support path**, not the default `run_pipeline()` hot path.

**Where the large language model fits in.** In the current default flow, stage 1 does a **whole-job extraction** and returns exact spans plus model-supplied context / section / classification fields. Each mention is then grounded to a real source line. Stages 2-4 operate at the **mention level**: skill verification, requirement classification, and hard-vs-soft classification. Stages 2-4 are **authoritative** — if the verifier rejects a mention, it stays rejected regardless of what the extractor said. Extractor-supplied classifications are preserved in the audit trail but only used as fallback when a later stage was skipped.

**Confidence scores.** Each mention gets a final confidence between 0 and 1, blended from stage outputs and deterministic checks. Extractor + verifier remain the base blend (when verifier output is valid), and requirement / hard-soft classifier confidences are incorporated as secondary signals. Section boosts and penalties for parse failures, stage errors, invalid offsets, and weak evidence all apply. Because mentions are now grounded to real source lines with accurate `boilerplate_label` values, boilerplate penalties (legal, benefits, marketing) fire correctly on the default path.

*Example: high confidence.* A mention appears in a requirements-style context, survives verification, and receives strong agreement-level confidence from the later classifiers. That kind of staged agreement pushes the final score toward the top of the range.

*Example: low confidence.* A mention is rejected by the verifier, or survives only through parse failures / weak evidence with poor offsets. Those cases are explicitly penalized and drop into the review bucket.

**What you get out.** Every original job field is **preserved**; new fields are **appended** so you keep an audit trail. In the current default pipeline, the main appended outputs are `description_raw`, `description_normalized`, `skill_mentions`, `pipeline_stage_audit`, and `extraction_metadata`. Mention `char_start`/`char_end` offsets index into `description_normalized`. Mention exports (`JSONL` / `CSV`) and optional derived reports are written from that augmented output. The richer structural fields (`parsed_lines`, `skill_candidates`, `quality_assessment`) still exist in the codebase but are mainly produced by the legacy `process_single_job()` path.

Think of it as a **hybrid extraction system** for unstructured job ads: a whole-document LLM extraction pass grounded to real source structure, followed by staged mention-level adjudication where later stages are authoritative over the initial extractor.

---

## Why is `Runskills_extraction.py` in the repo root?

The **implementation** of this tool is entirely under **`skills_extraction/`** (this directory).

The file **`Runskills_extraction.py`** at the **repository root** (next to the `skills_extraction/` package folder) is only a **thin launcher (“shim”)**:

- It adds the repo root to `sys.path` so `import skills_extraction` works.
- It calls `skills_extraction.cli.main()`.

**Reasons to keep the shim:**

1. **Habit / docs** — Many runs already use `python Runskills_extraction.py ...`.
2. **Discoverability** — The script name sits next to `Run_ollama.py` and other top-level tools.
3. **No install step** — You don’t have to `pip install -e .` to run from a clone.

**Canonical / preferred invocation** (no extra file required):

```bash
cd /path/to/skills-extraction    # repository root (contains `skills_extraction/`)
python -m skills_extraction --input SampleJobs.json --output-dir ./skills_out
```

If you prefer **everything** under one folder only, you can delete the root shim and use **only** `python -m skills_extraction` from the repo root; the package does not depend on that file.

---

## Repository layout

```
skills-extraction/             # repository root
  Runskills_extraction.py      # optional shim → delegates to skills_extraction.cli
  deploy.sh                    # rsync code to vLLM server (-s to include sample data)
  vLLM_run.sh                  # server-side run script (conda + vLLM config)
  test_vllm.sh                 # curl-based endpoint health check
  test_simple.py               # minimal vLLM extraction test
  skills_extraction/           # all application code + this README
    __init__.py
    __main__.py                # enables: python -m skills_extraction
    cli.py                     # argparse + logging setup + progress display
    config.py                  # PipelineConfig, backend selection, env loading
    schemas.py
    pipeline.py                # stage orchestration, rolling parallel execution
    llm_backend.py             # unified LLM dispatcher
    llm_ollama.py              # Ollama client
    llm_vllm.py                # vLLM client (pool + direct endpoint modes)
    llm_openrouter.py          # OpenRouter client
    llm_extractor.py           # whole-job extractor + deprecated batched helpers
    checkpoint.py              # checkpoint save/load/resume
    ... (other modules)
    README.md                  # this file
    CHANGELOG.md               # version history for this tool
    example_augmented_job_fragment.json
```

---

## Requirements

- Python 3.10+ (uses `datetime.timezone`, type hints).
- **`requests`** — HTTP API for all backends (Ollama, vLLM, OpenRouter).
- **`python-dotenv`** — optional `.env.local` / `.env` for backend URLs and model overrides.

No fixed skill taxonomy or embedding service is required for the core pipeline.

---

## Configuration

### Backend selection

The pipeline supports three LLM backends, selected via `--backend` or `SKILLS_BACKEND` env var:

| Backend | Flag | Use case |
|---------|------|----------|
| **Ollama** | `--backend ollama` (default) | Single-GPU or remote Ollama server |
| **vLLM** | `--vllm` | Multi-GPU parallel inference (recommended for large runs) |
| **OpenRouter** | `--backend openrouter` | Cloud-hosted models via API |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `SKILLS_BACKEND` | `ollama`, `vllm`, or `openrouter` |
| `OLLAMA_BASE_URL` | e.g. `http://localhost:11434` |
| `OLLAMA_URL` | Base or full generate URL; `/api/generate` stripped if present |
| `OPENROUTER_API_KEY` | API key for OpenRouter backend |
| `SKILLS_EXTRACTOR_MODEL` | Default extractor model tag |
| `SKILLS_VERIFIER_MODEL` | Default verifier model tag |
| `SKILLS_REQUIREMENT_MODEL` | Default requirement classifier model tag |
| `SKILLS_HARDSOFT_MODEL` | Default hard/soft classifier model tag |

`.env.local` is loaded before `.env` (paths resolved from **repo root**, parent of `skills_extraction/`).

### Defaults (`config.PipelineConfig`)

- Extractor: `qwen3:14b`
- Verifier: `mistral-nemo:12b`
- Requirement classifier: `mistral-nemo:12b`
- Hard/soft classifier: `mistral-nemo:12b`
- Fallback if extractor batch fails: `llama3.1:8b`
- Remote Ollama default: `http://ollama.rs.gsu.edu` (override with `--local` → `http://localhost:11434`)
- Thinking-mode suppression: enabled by default (`disable_thinking: True`)
- vLLM per-call delay: `0.0` (no artificial throttling); Ollama/OpenRouter: `0.25s`

---

## CLI reference

Run from the **repository root** (the directory that contains the **`skills_extraction`** package folder):

```bash
python -m skills_extraction [OPTIONS]
```

### General options

| Option | Description |
|--------|-------------|
| `-i`, `--input` | **Required.** Path to JSON: array of jobs or `{ "jobs": [ ... ] }`. |
| `-o`, `--output-dir` | Output directory (default: `skills_extraction_output`). |
| `--run-id` | Optional run id; default `YYYYMMDD_HHMMSS`. |
| `--sample N` | Process only the first N jobs after load. |
| `--no-verifier` | Disable the verifier LLM pass. |
| `--no-requirement-classifier` | Disable requirement classifier pass. |
| `--no-hardsoft-classifier` | Disable hard/soft classifier pass. |
| `--skip-llm` | Legacy/debug flag. The richest no-LLM structural output is still on the older `process_single_job()` path; the default stage-first pipeline is centered on stage-1 extraction. |
| `--no-reports` | Skip derived quality/frequency/low-confidence reports. |
| `--no-resume` | Ignore existing checkpoints; overwrite from scratch. |
| `--batch-lines` | Max lines per extractor request (default `5`). |
| `--timeout` | HTTP timeout in seconds (default `300`). |

### Backend options

| Option | Description |
|--------|-------------|
| `--backend` | `ollama` (default), `openrouter`, or `vllm`. |
| `--local` | Use local Ollama at `http://localhost:11434`. |
| `--context-size` | Ollama `num_ctx` (default `32768`). |
| `--vllm` | Use vLLM backend (also applies `--vllm-*` overrides). |
| `--vllm-host` | vLLM server hostname (default: `localhost`). |
| `--vllm-base-port` | First endpoint port (default: `8001`). |
| `--vllm-num-endpoints` | Number of vLLM endpoints/GPUs (default: `8`). |

### Model overrides

| Option | Description |
|--------|-------------|
| `--extractor-model` | Model tag for extraction (default: `qwen3:14b`). |
| `--verifier-model` | Model tag for verification (default: `mistral-nemo:12b`). |
| `--requirement-model` | Model tag for requirement classifier. |
| `--hardsoft-model` | Model tag for hard/soft classifier. |
| `--fallback-model` | Used if a batch fails with the primary extractor. |

**Important:** When using vLLM, model names must match the HuggingFace model ID registered by the vLLM server (e.g., `Qwen/Qwen3-14B`), not the Ollama tag format (`qwen3:14b`). Use `curl http://localhost:PORT/v1/models` to check.

### Examples

```bash
# Ollama (local)
python -m skills_extraction -i jobs.json -o ./out --local --sample 10

# Ollama (remote)
python -m skills_extraction -i jobs.json -o ./out

# vLLM with 8 GPUs, extraction only
python -m skills_extraction -i jobs.json -o ./out \
  --vllm --vllm-host localhost --vllm-base-port 8000 --vllm-num-endpoints 8 \
  --extractor-model "Qwen/Qwen3-14B" \
  --no-verifier --no-requirement-classifier --no-hardsoft-classifier

# Structure only (no LLM)
python -m skills_extraction -i jobs.json -o ./out --skip-llm
```

**Shim (same options):**

```bash
python Runskills_extraction.py -i SampleJobs.json -o ./out --sample 10
```

---

## Pipeline stages (default `run_pipeline()` path)

1. **Ingest** — Load JSON; preserve original keys; stable `job_key`.
2. **Stage 1 extract** — Send each full job description to the extractor in one LLM call. Mentions are grounded to real `ParsedLine` objects from the preprocessed source text.
3. **Stage 2 verify** — Validate each extracted mention as skill / not-skill. **Authoritative** — overrides extractor `is_skill`.
4. **Stage 3 requirement** — Classify accepted mentions as `required|optional|unclear`. **Authoritative** — overrides extractor `requirement_level`.
5. **Stage 4 hard/soft** — Classify accepted mentions as `hard|soft|unknown`. **Authoritative** — overrides extractor `hard_soft`.
6. **Stage 5 assemble** — Merge per-stage outputs into final mention rows plus job-level audit metadata. Extractor classifications preserved in audit trail.
7. **Export** — Augmented JSON, JSONL, CSV, optional reports, and `run_summary.json`.

**Legacy path still in code:** preprocessing, sectioning, boilerplate labeling, quality scoring, and candidate mining remain implemented and are still exercised by `process_single_job()`, but stage 0 is currently disabled in the default stage-first orchestration.

---

## Output artifacts

Written under `--output-dir`:

| File | Contents |
|------|-----------|
| `SkillsExtraction_augmented_run_{run_id}.json` | Full jobs with appended fields (see below). |
| `SkillsExtraction_mentions_run_{run_id}.jsonl` | One JSON object per mention (long format). |
| `SkillsExtraction_mentions_run_{run_id}.csv` | Same as JSONL, tabular. |
| `SkillsExtraction_pipeline_run_{run_id}.log` | Pipeline log. |
| `SkillsExtraction_quality_run_{run_id}.csv` | Unless `--no-reports`. |
| `SkillsExtraction_skill_frequency_run_{run_id}.csv` | Unless `--no-reports`. |
| `SkillsExtraction_low_confidence_run_{run_id}.json` | Unless `--no-reports` (includes `parse_failed`). |
| `SkillsExtraction_run_summary_{run_id}.json` | Timing, models, job counts (CLI runs). |

Under the current default stage-first flow, the mention exports and run summary are the primary artifacts. The augmented JSON now includes `description_normalized` and source-grounded mention offsets. Reports that depend on per-line structural metadata (parsed lines, candidates) are still thinner than in the legacy single-job path.

---

## Augmented job record (fields added)

Original fields are **never removed**. In the current default stage-first pipeline, new fields include:

| Field | Role |
|-------|------|
| `description_raw` | Raw job description copied into the augmentation block. |
| `description_normalized` | Whitespace-normalized, section-split text. Mention `char_start`/`char_end` index into this string. |
| `skill_mentions` | Mention-level staged outputs (`verifier_*`, `requirement_*`, `hardsoft_*`) and per-mention `pipeline_audit` with per-stage status/model/output/error. Extractor's original classifications are preserved in `pipeline_audit.extractor.output` for traceability. |
| `pipeline_stage_audit` | Job-level counters for stage failures/rejections and model snapshot for audit/debug. |
| `extraction_metadata` | `run_id`, `pipeline_version`, models, timestamps, `job_key`. |

The richer structural fields below still exist in the legacy single-job path and helper modules, but are **not currently appended by default `run_pipeline()`**:

- `quality_assessment`
- `parsed_lines`
- `skill_candidates`

**Example:** see `example_augmented_job_fragment.json` in this directory. That fragment reflects the older richer augmentation shape; the current default output includes `description_normalized` but not parsed lines or candidates.

---

## Server deployment (vLLM)

The pipeline can be deployed to a remote server with vLLM workers for multi-GPU extraction. The repo includes deployment scripts tested on an 8x NVIDIA RTX A6000 cluster.

### Setup

```bash
# Deploy code to server (first time: add -s to include SampleJobs.json)
./deploy.sh -s

# Subsequent deploys (code only)
./deploy.sh
```

The deploy script:
- Rsyncs the project to `~/skills_extraction/` on the server
- With `-s`: copies `SampleJobs.json` to `~/jobs/` (the default input path)
- Excludes `.venv`, `__pycache__`, `.git`, and output directories

### Running on the server

```bash
# Run via the pre-configured script
bash ~/skills_extraction/vLLM_run.sh

# Or manually
conda run --no-capture-output -n skills python -m skills_extraction \
  --input ../jobs/SampleJobs.json --output-dir ./out \
  --vllm --vllm-host localhost --vllm-base-port 8000 --vllm-num-endpoints 8 \
  --extractor-model "Qwen/Qwen3-14B" \
  --no-verifier --no-requirement-classifier --no-hardsoft-classifier \
  --sample 20
```

### Testing endpoints

```bash
# Test all endpoints
bash ~/skills_extraction/test_vllm.sh

# Test a single endpoint
curl http://localhost:8000/v1/models

# Quick extraction test
python ~/skills_extraction/test_simple.py
```

### Conda environment

The server uses `conda run` (avoids needing `conda init`). First-time setup:

```bash
conda create -n skills python=3.11 -y
conda run -n skills pip install -r requirements.txt
```

### Deployment notes

- **`conda run` buffering:** Always use `--no-capture-output` or stdout is buffered until process exit, making the pipeline appear to hang.
- **Model names:** vLLM registers models by HuggingFace ID (`Qwen/Qwen3-14B`), not Ollama tags (`qwen3:14b`). Use `curl localhost:PORT/v1/models` to check.
- **`--vllm` vs `--backend vllm`:** Use `--vllm` to enable vLLM — this flag triggers the port/host/endpoint overrides. `--backend vllm` alone sets the backend but ignores `--vllm-*` arguments.
- **Checkpoints are per-run-id:** Multiple runs can share an output directory without collision.

### Performance observations (8x RTX A6000, Qwen3-14B)

| Configuration | 20 jobs (86 batches) | Avg per batch |
|--------------|---------------------|---------------|
| Thinking ON, 7 endpoints | ~50 min | 107.9s |
| Thinking OFF, 8 endpoints | TBD | TBD |

Thinking-mode suppression (`disable_thinking: True`, the default) sends `chat_template_kwargs: {"enable_thinking": false}` in vLLM requests, eliminating ~400 reasoning tokens per call. Isolated test: 19.7s → 1.2s for a simple prompt.

---

## Module map

| Module | Role |
|--------|------|
| `config.py` | `PipelineConfig`, backend selection, env loading |
| `schemas.py` | Dataclasses / enums for lines, candidates, mentions, quality |
| `io_utils.py` | JSON load, `stable_job_key`, append-only merge |
| `preprocessing.py` | Raw vs normalized description |
| `sectioning.py` | Lines + section labels |
| `boilerplate.py` | Line-level boilerplate classification |
| `quality.py` | Document quality scoring |
| `candidate_mining.py` | Pattern-based candidate spans for the legacy/support path |
| `llm_backend.py` | Unified dispatcher — routes to Ollama / vLLM / OpenRouter |
| `llm_ollama.py` | Ollama HTTP client + JSON repair |
| `llm_vllm.py` | vLLM OpenAI-compatible client, endpoint pool, `call_vllm_direct()`, per-thread session reuse |
| `llm_openrouter.py` | OpenRouter HTTP client |
| `prompts.py` | Stage-specific prompt text (extract/verify/classify) |
| `llm_extractor.py` | Whole-job extraction plus deprecated batched extraction helpers |
| `llm_verifier.py` | Skill verifier stage |
| `llm_requirement_classifier.py` | Requirement classifier stage |
| `llm_hardsoft_classifier.py` | Hard/soft classifier stage |
| `confidence.py` | Final confidence blending |
| `checkpoint.py` | Checkpoint save/load, resume logic, serialization |
| `exporters.py` | JSON / JSONL / CSV / reports |
| `pipeline.py` | Orchestration, rolling worker pool, incremental checkpointing |
| `run_stats.py` | Run timing / LLM stats for logs and `run_summary` JSON |

---

## Programmatic use

```python
from pathlib import Path
from skills_extraction import PipelineConfig, load_config_from_env, run_pipeline

cfg = load_config_from_env({"skip_llm": True})
jobs = [...]  # list of dicts
augmented, paths, stats = run_pipeline(jobs, cfg, Path("./out"), run_id="20260101_120000")
```

---

## Design principles (short)

- **Open vocabulary** — No required skill taxonomy for detection.
- **No early dedupe** — Mention-level rows preserved; aggregate later if needed.
- **No clustering as primary detector** — DBSCAN/embeddings are out of scope for extraction.
- **Evidence + offsets** — Mentions trace back to `description_normalized` and `line_id`.

---

## Changelog

See **`CHANGELOG.md`** in this directory.

---

## Suggested tests

- `test_preprocess_maps_newlines` — CRLF → `\n`, empty description.
- `test_segment_line_ids_stable` — line count vs newlines.
- `test_candidate_offsets_in_range` — candidates within document length.
- `test_confidence_bounds` — final confidence in `[0, 1]`.
- `test_augment_preserves_original_keys` — no dropped `id` / title fields.

---

## Future work

- **Sentence segmentation inside long lines** — keep `line_id` as storage unit; add optional `sentence_id` and extract on sentences for huge paragraphs.
- Embedding similarity for **canonicalization** (post-hoc only).
- Active-learning queue from low-confidence export.
- Optional **spaCy** / **NLTK** behind a feature flag for richer chunking / POS-aware candidates.
