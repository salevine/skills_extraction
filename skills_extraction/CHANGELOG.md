# Skills extraction — Changelog

Application code and docs live in this directory (`skills_extraction/`).  
This file is the canonical changelog for the skills-extraction project.

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
