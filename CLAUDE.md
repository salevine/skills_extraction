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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **skills_extraction** (1099 symbols, 1932 relationships, 94 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/skills_extraction/context` | Codebase overview, check index freshness |
| `gitnexus://repo/skills_extraction/clusters` | All functional areas |
| `gitnexus://repo/skills_extraction/processes` | All execution flows |
| `gitnexus://repo/skills_extraction/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
