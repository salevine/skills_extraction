# Skills extraction (v3) — full guide

Open-vocabulary, **line-level**, **audit-first** skill mention extraction using Ollama.  
The **source of truth** is **mention-level** JSON (plus JSONL/CSV exports). Summary reports are **derived**, not primary.

**Current release: 3.0.0** (see **`CHANGELOG.md`** for full history). Highlights: **role-specialized staged pipeline** (span extraction → skill verifier → requirement classifier → hard/soft classifier), **mention-level `pipeline_audit`**, **job-level `pipeline_stage_audit`**, new classifier model/toggle CLI options, and confidence blending that includes secondary classifier outputs.

---

## What this actually does (read this first)

Job postings are messy. They mix real requirements with legal boilerplate, perks, and vague “we’re a family” fluff. They also don’t come with a neat list of skills—you have to *read* them. This application is a **pipeline** that takes a collection of job records (usually JSON), pulls out the **description text**, and tries to surface **what the employer is actually asking for** in terms of tools, technologies, methods, and competencies—without forcing everything into a fixed skill taxonomy up front.

**NLP side of the house.** The system doesn’t just “send the whole blob to the model and hope.” It does a fair bit of classical, transparent text work first. Descriptions are **normalized** (consistent newlines, a stable string the rest of the pipeline agrees on) so that **character offsets** mean something—you can point back to exactly where a mention lived in the text. The document is **split into lines** and given simple **section cues** (things like requirements vs. narrative) so later stages know *where* in the posting a span appeared. There’s a **document-quality pass** that looks at signals like repetition, how “skill-ish” the lines feel, and how much looks like filler; the goal isn’t to be preachy about bad ads, but to **flag** thin or broken text so you don’t over-trust extractions from garbage inputs. Each line also gets a **boilerplate label**—legal, benefits, marketing, **uncertain** (default for narrative), or **skills_relevant** only when **lexical** skill/requirement cues appear—so fewer generic body lines flood the extractor.

**Finding candidates before the LLM.** **Deterministic pattern mining** (experience / must-have / preferred / strong “skills” phrases, etc.) plus **gated** list-like tokens: comma-separated “tool-ish” candidates run mainly in **requirement-leaning sections** or lines that already look skill-related, with a **stoplist** for generic words. Goal: **enough recall** for real requirements without stuffing the prompt with narrative junk.

**Where the large language model fits in.** The LLM flow is now **role-specialized by stage**. First, the extractor proposes candidate spans only (with evidence and `span_confidence`). Second, a skill verifier decides whether each span is truly a skill mention. Third and fourth stages classify accepted mentions as `required|optional|unclear` and `hard|soft|unknown`. This keeps each stage narrow, improves auditability, and makes failures easier to localize. Parse failures are recorded stage-by-stage (`parse_failed`) with penalties in final confidence and visibility in exports.

**Confidence scores.** Each mention gets a final confidence between 0 and 1, blended from multiple stage outputs and deterministic checks. Extractor + verifier remain the base blend (when verifier output is valid), and requirement/hard-soft classifier confidences are incorporated as secondary signals. Section boosts, rule-support boosts, and penalties for boilerplate context, invalid offsets/evidence, and stage parse failures/errors still apply.

*Example: high confidence.* A line reads “Required: 5+ years experience with Python and SQL.” The span “Python” appears in the “requirements” section, was mined by both the “experience with” and “must have” patterns, and the model returns it with 0.88. Boilerplate is “skills_relevant”, offsets are valid, evidence matches. Final score: 0.88 + 0.06 (section) + 0.05 (two rules) → **0.99**.

*Example: low confidence.* A line reads “We are an equal opportunity employer.” The model, overeager, suggests “employer” as a skill with 0.62. No patterns fired (it’s LLM-only), the line is labeled “likely_legal”, and the span’s offsets are slightly off. Final score: 0.62 − 0.14 (boilerplate) − 0.12 (bad offsets) → **0.36**.

**What you get out.** Every original job field is **preserved**; new fields are **appended** so you keep an audit trail: normalized text, per-line parses, candidates, mentions with offsets, quality metadata, and run info. From there you can export **JSONL/CSV** for stats or modeling, plus optional **summary reports** (quality overview, skill-ish frequency tables, a queue of low-confidence items for human spot checks). You can also run with **`--skip-llm`** to exercise the whole structural and mining stack without calling Ollama—handy for debugging or when you only want segmentation and candidates.

Think of it as a **hybrid extraction system** for unstructured job ads: linguistic preprocessing and line-level segmentation, rule-based candidate generation, and generative span labeling with optional verification and multi-factor confidence, aimed at reproducible, mention-level outputs suitable for labor-market or skills research—without pretending the model is infallible or that every posting is equally informative.

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
  skills_extraction/           # all application code + this README
    __init__.py
    __main__.py                # enables: python -m skills_extraction
    cli.py                     # argparse + logging setup
    config.py
    schemas.py
    pipeline.py
    ... (other modules)
    README.md                  # this file
    CHANGELOG.md               # version history for this tool
    example_augmented_job_fragment.json
```

---

## Requirements

- Python 3.10+ (uses `datetime.timezone`, type hints).
- **`requests`** — Ollama HTTP API.
- **`python-dotenv`** — optional `.env.local` / `.env` for Ollama URL and model overrides.

No fixed skill taxonomy or embedding service is required for the core pipeline.

---

## Configuration

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OLLAMA_BASE_URL` | e.g. `http://localhost:11434` |
| `OLLAMA_URL` | Base or full generate URL; `/api/generate` stripped if present |
| `SKILLS_EXTRACTOR_MODEL` | Default extractor model tag |
| `SKILLS_VERIFIER_MODEL` | Default verifier model tag |
| `SKILLS_REQUIREMENT_MODEL` | Default requirement classifier model tag |
| `SKILLS_HARDSOFT_MODEL` | Default hard/soft classifier model tag |

`.env.local` is loaded before `.env` (paths resolved from **repo root**, parent of `skills_extraction/`).

### Defaults (`config.PipelineConfig`)

- Extractor: `qwen2.5:14b`
- Verifier: `mistral-nemo:12b`
- Requirement classifier: `mistral-nemo:12b`
- Hard/soft classifier: `mistral-nemo:12b`
- Fallback if extractor batch fails: `llama3.1:8b`
- Remote Ollama default: `http://ollama.rs.gsu.edu` (override with `--local` → `http://localhost:11434`)

---

## CLI reference

Run from the **repository root** (the directory that contains the **`skills_extraction`** package folder):

```bash
python -m skills_extraction [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-i`, `--input` | **Required.** Path to JSON: array of jobs or `{ "jobs": [ ... ] }`. |
| `-o`, `--output-dir` | Output directory (default: `skills_extraction_output`). |
| `--run-id` | Optional run id; default `YYYYMMDD_HHMMSS`. |
| `--local` | Use local Ollama at `http://localhost:11434`. |
| `--extractor-model` | Ollama model tag for extraction. |
| `--verifier-model` | Ollama model tag for verification. |
| `--requirement-model` | Ollama model tag for requirement classifier. |
| `--hardsoft-model` | Ollama model tag for hard/soft classifier. |
| `--fallback-model` | Used if a batch fails with the primary extractor. |
| `--sample N` | Process only the first N jobs after load. |
| `--no-verifier` | Disable the second LLM pass. |
| `--no-requirement-classifier` | Disable requirement classifier pass. |
| `--no-hardsoft-classifier` | Disable hard/soft classifier pass. |
| `--skip-llm` | No Ollama calls: structure, `parsed_lines`, `skill_candidates`, quality only. |
| `--no-reports` | Skip derived quality/frequency/low-confidence reports. |
| `--batch-lines` | Max lines per extractor request (default `5`). |
| `--context-size` | Ollama `num_ctx` (default `32768`). |

**Shim (same options):**

```bash
python Runskills_extraction.py -i SampleJobs.json -o ./out --sample 10
```

---

## Pipeline stages (high level)

1. **Ingest** — Load JSON; preserve original keys; stable `job_key`.
2. **Preprocess** — Keep `description_raw`; build `description_normalized` (includes **inline section header splits** so offsets match stored text).
3. **Section + lines** — `line_id`, section heuristics (standalone + colon-led headers), char ranges.
4. **Quality** — Multi-signal score + status; **missing description** → malformed (no company/location stand-in).
5. **Boilerplate** — Per-line label; **conservative** promotion to `skills_relevant`.
6. **Candidate mining** — Rule/pattern spans; **precision-tuned** list/prep rules (not line-wide noise).
7. **LLM span extractor** — Batched lines + candidates → candidate span mentions; **offset repair** (valid model span → nearest match → evidence-aligned).
8. **LLM skill verifier** — Validates each candidate as skill/not-skill; parse failures flagged explicitly.
9. **LLM requirement classifier** — Assigns `required|optional|unclear` for validated mentions.
10. **LLM hard/soft classifier** — Assigns `hard|soft|unknown` for validated mentions.
11. **Confidence** — Combine extractor/verifier/classifier confidence signals plus section/rule/boilerplate/offset checks.
12. **Export** — Augmented JSON, JSONL, CSV, optional reports (with stage audit fields).

Details live in the module table below and in source docstrings.

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

---

## Augmented job record (fields added)

Original fields are **never removed**. New fields include:

| Field | Role |
|-------|------|
| `description_normalized` | Reference string for global `char_start` / `char_end`. |
| `quality_assessment` | `status`, `quality_score`, `reasons`, `features`. |
| `parsed_lines` | Line-level audit: `line_id`, `section`, `text`, offsets, `boilerplate_label`. |
| `skill_candidates` | All mined spans (no early dedupe). |
| `skill_mentions` | Mention-level staged outputs (`verifier_*`, `requirement_*`, `hardsoft_*`) and per-mention `pipeline_audit` with per-stage status/model/output/error. |
| `pipeline_stage_audit` | Job-level counters for stage failures/rejections and model snapshot for audit/debug. |
| `extraction_metadata` | `run_id`, `pipeline_version`, models, timestamps, `job_key`. |

**Example:** see `example_augmented_job_fragment.json` in this directory.

---

## Module map

| Module | Role |
|--------|------|
| `config.py` | `PipelineConfig`, Ollama URL resolution |
| `schemas.py` | Dataclasses / enums for lines, candidates, mentions, quality |
| `io_utils.py` | JSON load, `stable_job_key`, append-only merge |
| `preprocessing.py` | Raw vs normalized description |
| `sectioning.py` | Lines + section labels |
| `boilerplate.py` | Line-level boilerplate classification |
| `quality.py` | Document quality scoring |
| `candidate_mining.py` | Pattern-based candidate spans |
| `llm_ollama.py` | HTTP + JSON repair |
| `prompts.py` | Stage-specific prompt text (extract/verify/classify) |
| `llm_extractor.py` | Batched span extraction |
| `llm_verifier.py` | Skill verifier stage |
| `llm_requirement_classifier.py` | Requirement classifier stage |
| `llm_hardsoft_classifier.py` | Hard/soft classifier stage |
| `confidence.py` | Final confidence blending |
| `exporters.py` | JSON / JSONL / CSV / reports |
| `pipeline.py` | Orchestration |
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
