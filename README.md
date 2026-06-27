# Skills Extraction Pipeline

Open-vocabulary, mention-level **skill extraction** from job postings with multi-backend LLM support (**Ollama**, **vLLM**, **OpenRouter**). Produces a structured **skills ontology** with canonical names, variants, type classifications, requirement levels, and frequency statistics.

## Quick start

```bash
cd skills-extraction
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
# optional editable install: pip install -e .

copy .env.example .env.local    # optional: set OLLAMA_BASE_URL, models

python -m skills_extraction -i your_jobs.json -o ./out --sample 5
```

## Pipeline stages

| Stage | Name | What it does | LLM? |
|-------|------|--------------|------|
| 1 | **Extract** | Whole-job extraction of skill mentions with spans | Yes |
| 2 | **Verify** | Validate each mention as skill / not-skill (authoritative) | Yes |
| 3 | **Requirement** | Classify as required / optional / unclear (authoritative) | Yes |
| 4 | **Hard/Soft** | Classify as hard / soft / unknown (authoritative) | Yes |
| 5 | **Assemble** | Merge per-stage outputs, compute final confidence | No |
| 6 | **Ontology** | Aggregate into canonical skills with variants and stats | No |

## Running individual stages

### Full pipeline (all stages)

```bash
python -m skills_extraction -i jobs.json -o ./out --vllm
```

### Interactive launcher (titan3)

`launch.sh` walks you through a run, then launches the full two-phase pipeline
(Qwen extraction → swap to Mistral-Nemo for verify/classify) under `nohup` so it
survives disconnect. It prompts for:

- **Input file** — a scrollable numbered menu of the `.json` files in
  `../data_files` (pick by number, or choose "Enter a custom path"), plus
  **sample size**
- **Output directory** for result files + checkpoints (default `../data_files`)
- **Log directory** for the nohup run log, the generated runner script, and the
  pipeline's own internal log (passed through as `--log-dir`); default `./logs`
- **Run id** — new (timestamped) or resume an existing one (with optional
  `--rerun-from <stage>` or `--retry-stage1-errors`)
- **Endpoints / base port** and the **extractor / verifier model** names

It then prints a summary of every setting and the exact paths, asks for
confirmation, and launches in the background.

```bash
~/startQwen 8        # load extraction model first (servers are only verified, not started)
./launch.sh          # answer prompts, review summary, confirm — then it runs in the background
```

**Locations are independent.** Input file, output directory, and log directory
are each chosen separately, so different programs can own different locations.
Result files and checkpoints go to the output dir; the run log and the
auto-generated runner script stay in the log dir — logs never mix with data.

On resume it skips the extraction phase when stage 1 is already complete for the
run id. Follow progress with `tail -f <log dir>/run_<id>_<ts>.log` or
`SKILLS_OUT_DIR=<output dir> ./check_status.sh <id>` (the launcher passes
`SKILLS_OUT_DIR` automatically so its end-of-run status finds checkpoints
wherever you put them).

### Rerun from a specific stage (keep earlier results)

```bash
# Keep stage 1, rerun stages 2-4 (then 5-6 automatically)
python -m skills_extraction -i jobs.json -o ./out --run-id YOUR_RUN_ID --rerun-from stage2

# Retry only stage 1 failures, then continue
python -m skills_extraction -i jobs.json -o ./out --run-id YOUR_RUN_ID --retry-stage1-errors
```

### Skip LLM stages selectively

```bash
# Extraction only (no verification or classification)
python -m skills_extraction -i jobs.json -o ./out \
  --no-verifier --no-requirement-classifier --no-hardsoft-classifier
```

### Build ontology standalone (stage 6 only)

No LLM needed — works from an existing augmented JSON file:

```bash
python -m skills_extraction --ontology-only ./out/SkillsExtraction_augmented_run_RUNID.json -o ./out
```

## Output

The pipeline produces these artifacts per run:

| File | Contents |
|------|----------|
| `*_augmented_run_{id}.json` | Full jobs with appended extraction fields |
| `*_mentions_run_{id}.jsonl` | One JSON object per mention (long format) |
| `*_mentions_run_{id}.csv` | Same as JSONL, tabular |
| `*_ontology_run_{id}.json` | Skills ontology — canonical skills with variants, stats |
| `*_ontology_run_{id}.csv` | Same ontology, tabular |
| `*_quality_run_{id}.csv` | Quality report |
| `*_skill_frequency_run_{id}.csv` | Frequency counts |
| `*_low_confidence_run_{id}.json` | Mentions needing review |
| `*_run_summary_{id}.json` | Timing, models, job counts |

### Ontology format

The ontology JSON is an array of canonical skill entries:

```json
[
  {
    "canonical_skill": "Python",
    "canonical_key": "python",
    "variants": ["Python experience", "Python skills"],
    "type": "hard",
    "type_distribution": {"hard": 1443, "soft": 1},
    "requirement_level": "required",
    "requirement_distribution": {"required": 1249, "optional": 191, "unclear": 4},
    "job_count": 1333,
    "mention_count": 1444,
    "avg_confidence": 0.9572,
    "confidence_range": [0.7425, 1.0],
    "common_contexts": ["Software Engineer", "Machine Learning Engineer", "..."],
    "run_id": "full_20260413_203931"
  }
]
```

## Documentation

- **Full guide, CLI reference, data model:** [`skills_extraction/README.md`](skills_extraction/README.md)
- **Version history:** [`skills_extraction/CHANGELOG.md`](skills_extraction/CHANGELOG.md)

## Layout

| Path | Purpose |
|------|---------|
| `skills_extraction/` | Python package (pipeline, CLI, exporters) |
| `Runskills_extraction.py` | Optional launcher shim (same as `python -m skills_extraction`) |
| `requirements.txt` | Runtime dependencies |
| `deploy.sh` | Rsync code to vLLM server (`-s` to include sample data) |
| `launch.sh` | Interactive launcher (titan3): prompts for input/output/log dirs + settings, runs two-phase pipeline under `nohup` |
| `check_status.sh` | Per-run checkpoint/progress status; honors `SKILLS_OUT_DIR` to find results outside `./out` |
| `vLLM_run.sh` | Server-side run script (conda + vLLM config) |
| `test_vllm.sh` | Curl-based endpoint health check |

## Requirements

- Python 3.10+
- One configured backend: **Ollama**, **vLLM**, or **OpenRouter**

## License

Use your institution's default or add a `LICENSE` file when you publish the GitHub repo.
