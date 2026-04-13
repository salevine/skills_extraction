# Skills Extraction Pipeline

## Project overview
NLP pipeline that extracts skill mentions from job posting descriptions using LLMs. Stages: preprocessing (stage 0), extraction (stage 1), verification, requirement classification, hard/soft classification, aggregation. Currently focused on stage 1 extraction at scale.

## Architecture
- **Pipeline:** `skills_extraction/pipeline.py` — stage-first orchestration, checkpoint-based resume
- **vLLM backend:** `skills_extraction/llm_vllm.py` — `call_vllm_direct()` for thread-safe concurrent calls
- **Extractor:** `skills_extraction/llm_extractor.py` — `extract_mentions_for_batch()` with optional `endpoint` param
- **Boilerplate:** `skills_extraction/boilerplate.py` — rule-based line classifier (no LLM)
- **CLI:** `skills_extraction/cli.py` — progress display, logging config
- **Config:** `skills_extraction/config.py` — `PipelineConfig` dataclass

## Key design decisions
- vLLM uses **batch-level parallelism** with `ThreadPoolExecutor` + `as_completed` (NOT job-level — that deadlocks)
- Each thread gets a dedicated endpoint via round-robin (`endpoints[bi % len(endpoints)]`)
- `call_vllm_direct()` bypasses the shared endpoint pool to avoid contention
- Thinking mode is **disabled by default** (`disable_thinking: True`) — 2.2x faster, extracts 12% more mentions
- vLLM thinking suppression: `chat_template_kwargs: {"enable_thinking": false}` in payload
- Ollama thinking suppression: `/no_think` suffix appended to prompts
- Model name for vLLM is HuggingFace ID (`Qwen/Qwen3-14B`), NOT Ollama tag (`qwen3:14b`)

## Deployment
- Local dev on Mac, deploy to titan3 via `./deploy.sh` (add `-s` to also copy SampleJobs.json)
- Server run: `vLLM_run.sh` handles conda env setup and launches pipeline
- Use `nohup` for unattended runs, output to `run.log`
- Test endpoints: `test_vllm.sh` curls all 8 ports

## Current state (2026-04-02)
- v3.2.1 deployed
- Full 10K run in progress on titan3 (45,010 batches, 8 GPUs, ~50hr)
- Progress UI fix for timeout warnings is ready locally but NOT yet deployed (don't redeploy mid-run)
- After run completes: collect run_summary.json, update README perf table, update NOTES

## Files to keep out of git
- `CHAPTER_*.md`, `NOTES_*.md` — research notes (in .gitignore)
- `out/` — pipeline output directory
- Logs (`*.log`)
