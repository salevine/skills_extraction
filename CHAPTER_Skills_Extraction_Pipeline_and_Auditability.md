# CHAPTER X

## Skills Extraction Pipeline and Auditability

### X.1 Motivation and Design Position

Skill extraction in this dissertation is treated as an infrastructure problem rather than a single-model prediction task. Job descriptions are noisy, inconsistently structured, and shaped by employer-specific writing conventions. If extraction is implemented as a monolithic black box, downstream role weighting and course-career alignment become difficult to interpret, difficult to refresh, and difficult to trust. For this reason, the extraction workflow is designed around explicit intermediate stages, retained artifacts, and versioned execution metadata.

This chapter describes the extraction subsystem used to produce workforce-side skill signals from job postings. The design follows three constraints that are consistent with the broader dissertation:

1. **Open vocabulary over fixed ontology lock-in.** The system must remain responsive to emerging tools and practices.
2. **Traceability over opaque convenience.** Every reported skill mention should be inspectable in context.
3. **Refreshability over one-time optimization.** The extraction process must support periodic reruns with explicit version boundaries.

The contribution is therefore architectural. The goal is not to claim perfect extraction accuracy from free text, but to provide a scalable, auditable, and reviewable process that produces useful workforce representations for alignment.

### X.2 Role-Specialized Pipeline Overview

The extraction workflow operates on normalized job-description text and executes a role-specialized sequence of language-model stages. Instead of asking one prompt to perform all tasks, the system separates responsibilities into four stages:

1. **Span Extractor**: proposes candidate skill spans as exact substrings of line-level text.
2. **Skill Verifier**: determines whether each candidate is a valid skill mention.
3. **Requirement Classifier**: labels accepted mentions as required, optional, or unclear.
4. **Hard/Soft Classifier**: labels accepted mentions as hard, soft, or unknown.

This staged design addresses a practical failure mode in monolithic extraction: when one prompt must both find spans and assign multiple attributes, errors become entangled and difficult to diagnose. By decomposing the task, each stage can be inspected and tuned independently.

The extraction pipeline is embedded in a larger preprocessing flow that performs:

- text normalization,
- line segmentation and section labeling,
- lightweight boilerplate detection,
- deterministic candidate mining prior to LLM calls.

These deterministic steps are intentionally retained because they improve controllability and reduce prompt burden. The language model is used where semantic flexibility is needed, not as a replacement for all preprocessing logic.

### X.3 Input Preparation and Structural Context

The pipeline uses line-level processing as its primary unit of extraction. This choice provides a balance between context and specificity. Full-document prompting can blur local evidence boundaries, while overly short windows can miss qualification context. Line-level segmentation, with section labels, provides enough structure to support phrase extraction while preserving precise character offsets.

For each posting, the system stores:

- raw description text,
- normalized description text (reference string for offsets),
- parsed lines with `line_id`, section label, and line offsets.

Candidate mining then proposes potential spans using deterministic patterns (for example, requirement phrases and tool-shaped list tokens). These candidates are not treated as final outputs; they serve as recall-oriented hints to the extractor. This preserves flexibility while reducing random drift in model output.

### X.4 Stage 1: Span Extraction

The first stage is intentionally narrow: identify candidate skill spans and return exact line-level offsets plus evidence snippets. The extractor does not decide final validity and does not carry full responsibility for attribute labels. This is a deliberate separation of concerns.

At this stage, the system enforces substring alignment. If model offsets are inconsistent, a repair routine attempts deterministic correction against the original line text. This includes nearest-match fallback and evidence-anchored matching when multiple occurrences exist. Mentions that cannot be grounded to exact text are excluded from progression.

This stage prioritizes controlled recall, with the understanding that later stages will filter and refine. In operational terms, this avoids prematurely discarding plausible emerging skills while still enforcing strict text-grounding constraints.

### X.5 Stage 2: Skill Verification

The second stage evaluates whether a candidate span is a genuine skill mention. This stage exists because not every extracted phrase is useful for role-skill modeling. Job postings include organizational branding, legal language, compensation language, and role descriptors that can superficially resemble skill statements.

The verifier receives:

- section label,
- source line text,
- candidate span and normalized candidate form,
- span-level confidence and evidence.

It returns a binary validity decision (`is_skill`) with confidence and notes. Parse failures are not silently accepted. Instead, parse-failure status is explicitly recorded and carried into downstream confidence logic. This policy is important for dissertation-level reproducibility because it avoids hidden failure paths.

### X.6 Stage 3 and Stage 4: Attribute Classification

Only mentions accepted by the verifier proceed to attribute classification:

- **Requirement classifier**: required, optional, or unclear.
- **Hard/soft classifier**: hard, soft, or unknown.

In many postings, requirement intent is explicit through section headers or lexical cues (for example, "required qualifications," "preferred"). In others, signals are mixed. The `unclear` state is preserved for this reason; forcing a binary decision where textual evidence is ambiguous would create brittle role weighting.

Similarly, hard/soft distinctions are operational rather than ontological. Many competencies in contemporary computing are hybrid. The pipeline therefore includes `unknown` as a valid outcome and records classifier confidence rather than forcing categorical certainty.

### X.7 Mention-Level Auditability

A central output of this implementation is mention-level audit state. Every extracted mention stores a `pipeline_audit` object containing stage-by-stage status and outputs. This includes, for each stage:

- execution status (`completed`, `accepted`, `rejected`, `parse_failed`, `error`, or `skipped` as applicable),
- model identifier used for that stage,
- structured output fields,
- error text when execution fails.

This representation enables post hoc analysis of failure location. For example, a low-confidence mention can be traced to:

- unstable span extraction,
- verifier rejection,
- classifier parse failure,
- or deterministic penalties (boilerplate context, invalid offsets, weak evidence grounding).

Without this artifact retention, "bad outputs" are visible but not diagnosable. With retention, each result can be decomposed into stage contributions.

### X.8 Job-Level Auditability and Run Metadata

In addition to mention-level records, each job output includes a `pipeline_stage_audit` summary. This contains stage counters such as:

- number of extractor mentions produced,
- verifier rejections,
- parse failures by stage,
- stage-level execution errors.

Each run also records model configuration snapshots and timestamps. At minimum, this includes extractor, verifier, requirement-classifier, and hard/soft-classifier model identifiers. These metadata are required for refresh-cycle interpretation: if role-level skill distributions shift across runs, analysis can distinguish labor-market change from configuration change.

This dissertation treats such metadata as first-class research artifacts, not operational logs. They are required to satisfy traceability-oriented research questions and to support reproducibility claims within scope.

### X.9 Confidence Composition and Error Sensitivity

Final mention confidence is not treated as a single model score. The pipeline combines:

- span-stage confidence,
- verifier confidence (when valid),
- secondary classifier confidences (when valid),
- deterministic context signals (section placement, rule support),
- penalties for failure conditions (parse failures, execution errors, invalid offset/evidence grounding, boilerplate context).

This blended strategy reflects a practical thesis: extraction reliability emerges from agreement across heterogeneous signals rather than any one component. It also supports robust low-confidence review workflows, where uncertain mentions can be queued for targeted inspection rather than silently promoted to role-level aggregates.

### X.10 Exported Artifacts and Review Surfaces

The pipeline writes three primary artifact classes:

1. **Augmented job-level JSON** containing preprocessing outputs, candidates, mention records, and audit metadata.
2. **Mention-level JSONL** for long-form analysis and reproducible slicing.
3. **Mention-level CSV** for lightweight inspection and integration with external analysis tooling.

Optional reports (for example, low-confidence queues and frequency summaries) are treated as derived views, not source-of-truth data. The source of truth remains mention-level structured records with retained stage audit fields.

This distinction matters for dissertation rigor. Summary reports are useful for communication, but only retained intermediate artifacts allow reconstruction and decomposition of alignment inputs.

### X.11 Integration with Role-Skill Weighting

The extraction subsystem feeds the role-skill weighting model by providing normalized skill mentions with requirement labels. Required versus optional distinctions are retained because they materially affect role-level weighting behavior. At the same time, the extraction layer does not claim perfect semantic disambiguation. Instead, it provides confidence-bearing evidence units that can be aggregated with transparent assumptions.

The staged design also supports controlled thresholding. For example, downstream aggregation can:

- include only mentions above a confidence threshold,
- compare weighted outcomes with and without `unclear` requirement labels,
- quantify sensitivity to verifier parse failures.

These controls are important for stability analyses and refresh-readiness evaluation.

### X.12 Stage-First Execution Architecture

#### Observations from Pilot Runs

The logical stage decomposition described in Sections X.4–X.6 is necessary for auditability, but its execution order has material implications for operational feasibility at corpus scale. The initial implementation processed each job through all four LLM stages before advancing to the next job. A pilot run of five postings against the Ollama server exposed the cost structure of this approach and motivated a structural revision.

The five-job pilot completed in approximately 20 minutes (1,198 seconds wall clock) and produced 267 skill mentions. Per-stage timing revealed a sharply asymmetric cost profile:

| Stage | Wall clock | LLM calls | Avg per call |
|-------|-----------|-----------|-------------|
| Preprocessing (no LLM) | 0.04s | — | — |
| Extraction (`qwen3:14b`) | 705.6s | 19 batch calls | 36.9s |
| Verification (`mistral-nemo:12b`) | 258.9s | 267 calls | 0.77s |
| Requirement classification | 117.5s | 267 calls | — |
| Hard/soft classification | 116.2s | 267 calls | — |
| Assembly (no LLM) | 0.01s | — | — |

Two observations were immediately relevant. First, the extraction stage dominated wall-clock time, consuming 59% of the total despite producing only 19 LLM calls. This reflected the computational cost of the larger extractor model and the batch-level prompting strategy. Second, the verifier and classifier stages were individually fast (under one second per mention) but collectively numerous: 267 mentions multiplied across three classification stages produced over 800 LLM calls to the same model.

Under the job-at-a-time execution model, each job boundary would trigger a model swap between `qwen3:14b` (extractor) and `mistral-nemo:12b` (verifier and classifiers). In the Ollama deployment used for this dissertation, only one model resides in GPU memory at a time. Loading a model incurs 10–30 seconds of latency depending on model size and server state. For a single job with four LLM stages involving two distinct models, this means at least one swap per job. For five jobs, the pilot run incurred at minimum five swap events.

Extrapolating to the target corpus of 10,000 postings, job-at-a-time execution would produce over 10,000 model swap events — conservatively 30,000 or more when accounting for the multiple transitions between extraction and verification within each job. At 10–30 seconds per swap, this would add between 80 and 250 hours of non-productive loading latency to the run.

A second observation from HTTP-level logging revealed that each LLM call was opening a new TCP connection to the Ollama server, performing the request, and closing the connection. The connection setup overhead was approximately 0.3 seconds per call. Across the estimated 40,000+ LLM calls for a full corpus run, this represented an additional 3–4 hours of avoidable network latency.

#### Architectural Response

These observations led to a restructuring of the pipeline loop from job-first to stage-first execution. The revised architecture processes all jobs through each stage as a complete batch before advancing to the next stage:

1. **Stage 0** (preprocessing): all 10,000 jobs are normalized, segmented, and scored without any LLM involvement.
2. **Stage 1** (extraction): all jobs pass through the extractor. The extraction model (`qwen3:14b`) loads once and remains resident in GPU memory for the entire stage.
3. **Stages 2–4** (verification, requirement classification, hard/soft classification): all mentions from all jobs pass through each classification stage sequentially. The verifier/classifier model (`mistral-nemo:12b`) loads once at the start of Stage 2 and remains resident through Stage 4.
4. **Stage 5** (assembly): all stage outputs are combined into the final augmented records without LLM involvement.

The result is exactly one model transition per run (from extractor to verifier, between Stages 1 and 2), regardless of corpus size. This eliminates the dominant source of non-productive latency identified in the pilot.

To address the per-call TCP overhead, the HTTP client was changed from stateless `requests.post()` calls to a persistent `requests.Session` that maintains connection keep-alive across all LLM calls within a run. This eliminates repeated TCP handshake and connection setup.

This restructuring does not alter the logical stage decomposition. Each mention still passes through the same sequence of decisions, receives the same audit fields, and produces the same output schema. The change is purely in execution scheduling.

#### Intermediate Checkpoints and Crash Recovery

Stage-first execution introduces a new failure exposure: if a run is interrupted during Stage 3, the work completed in Stages 0–2 would be lost without intermediate persistence. For a full corpus run expected to span many hours, this risk is operationally unacceptable. To address this, the pipeline writes incremental checkpoint files at each stage boundary.

Checkpoints use a JSONL format with a structured protocol:

- A metadata header line records the run identifier, stage name, total expected records, and start timestamp.
- Data lines are appended and flushed one at a time as each job or mention completes processing.
- A completion footer line marks successful stage termination and records the final count.

On restart, the pipeline inspects existing checkpoints for the given run identifier. A checkpoint with a valid footer is treated as complete and the stage is skipped entirely. A checkpoint without a footer (indicating interruption) is resumed from the last successfully written record. This enables crash recovery without re-executing work that has already been persisted.

The checkpoint format is intentionally human-readable. Each line is a self-contained JSON object. Serialization of internal objects (`ParsedLine`, `CandidateSpan`, mention dictionaries with attached line references) is handled through explicit `to_dict()`/reconstruction patterns rather than opaque binary formats. This is consistent with the broader traceability commitment: intermediate pipeline state should be inspectable, not just recoverable.

#### Projected Performance Impact

Based on the pilot timing data, the following projections apply to a 10,000-job corpus:

- **Model swap elimination.** Job-at-a-time execution would produce an estimated 10,000–30,000 swap events at 10–30 seconds each: 80–250 hours of loading latency. Stage-first execution reduces this to a single swap event of 10–30 seconds.
- **Connection pooling.** At ~40,000 LLM calls with 0.3 seconds of per-call TCP overhead, persistent sessions save approximately 3–4 hours.
- **Net effect.** The combined improvements convert an extraction task that would span multiple weeks under the original architecture into one that completes within the wall-clock budget of a single extended session, limited primarily by actual model inference time rather than infrastructure overhead.

These projections are conservative in that they assume constant swap cost. In practice, swap latency can increase under memory pressure or when the server has evicted cached model weights, making the actual savings potentially larger.

#### Per-Stage Timing and Run Provenance

The stage-first architecture also enables per-stage wall-clock timing, which is recorded in the run summary alongside per-model LLM call statistics. This provides operational visibility into where time is spent and enables direct comparison with the pilot observations. For the dissertation, it enables reporting of extraction throughput broken down by pipeline phase, which is relevant for refresh-cycle planning and infrastructure sizing discussions.

Run identifiers now include a mandatory timestamp component, even when a human-readable label is provided (for example, `full_10k_20260327_153936`). This prevents checkpoint collisions when the same label is reused across runs, which is important for iterative development and parameter sweeps.

### X.13 Operational Challenges During the 10,000-Posting Extraction

The stage-first architecture described in Section X.12 was designed to make corpus-scale extraction operationally feasible. In practice, the first full-corpus run of 10,000 postings exposed several classes of infrastructure challenges that are relevant both to the engineering record and to the broader argument about extraction reproducibility.

#### Server Stability and Transient Failures

The extraction run against the university Ollama deployment spanned multiple days. During this period, the server intermittently returned HTTP 502 and 503 errors, typically correlated with periods of concurrent usage by other research groups. The pipeline's retry logic (up to three attempts per LLM call with exponential backoff) absorbed most of these transient failures, but sustained outage windows of 10–30 minutes occasionally triggered retry exhaustion. In these cases, the checkpoint-resume mechanism allowed the run to restart from the last persisted record without re-executing completed work.

A separate class of failure involved zombie processes from prior interrupted runs writing concurrently to checkpoint files. When two processes target the same checkpoint path, the interleaved JSONL lines produce structurally invalid files that cannot be resumed. This was resolved operationally by enforcing single-writer access through timestamped run identifiers, but it highlighted a limitation of the file-based checkpoint design: it assumes exclusive access to the checkpoint directory for a given run.

#### Throughput Characterization

Per-call debug logging was added during the full run to characterize actual throughput. The Ollama server consistently delivered 107–114 tokens per second across both the extractor (`qwen3:14b`) and verifier/classifier (`mistral-nemo:12b`) models. However, wall-clock time per LLM call varied substantially: from 4.8 seconds for short verification calls to 81.6 seconds for large extraction batches. This variation reflects prompt length and generation length differences rather than server instability.

The logged data confirmed that model inference time, not network or server overhead, was the dominant cost. This observation was important for infrastructure sizing: adding more concurrent connections to a single-GPU deployment would not improve throughput, because the GPU is the bottleneck. Meaningful acceleration requires distributing inference across multiple GPUs.

#### Thinking-Mode Token Waste

The Qwen3 model family supports an extended reasoning mode in which the model produces a `<think>...</think>` block before its answer. Per-call debug logging during the full extraction run revealed that this thinking output was not merely present but dominant. The `eval_count` field reported by the Ollama server — which counts all tokens generated, including those in `<think>` blocks that are stripped before the response is returned — consistently exceeded the visible response size by a factor of 2–7x. Representative examples from the production log illustrate the scale of the waste:

| Wall clock | Response chars | Eval tokens (total generated) | Thinking overhead |
|-----------|---------------|-------------------------------|-------------------|
| 88.7s | 2,371 | 8,929 | 3.8x |
| 87.5s | 1,474 | 8,823 | 6.0x |
| 81.8s | 3,492 | 8,181 | 2.3x |
| 58.4s | 803 | 5,959 | 7.4x |

In the most extreme cases, the model generated nearly 9,000 tokens to produce a response containing fewer than 1,500 characters. At the observed throughput of 107–108 tok/s for these longer generations, the thinking tokens alone consumed 50–75 seconds of GPU time per call. For a span-extraction task where the model is asked to return structured JSON containing exact substrings, this intermediate reasoning provided no measurable improvement in extraction quality.

The pipeline was updated to default to `disable_thinking=True`, which appends a `/no_think` suffix to suppress extended reasoning. This change was deployed mid-run at record 6,218 of 10,000 (approximately 62% through Stage 1 extraction) by stopping the running process and restarting it against the updated code. The checkpoint-resume mechanism allowed the restart to continue from the last persisted record with no data loss.

The effect was immediately visible in the debug log. The first calls after the restart showed a fundamentally different token profile:

| Wall clock | Response chars | Eval tokens (total generated) | Thinking overhead |
|-----------|---------------|-------------------------------|-------------------|
| 33.3s | 4,389 | 3,351 | 0.8x |
| 26.6s | 3,787 | 2,697 | 0.7x |
| 35.8s | 1,670 | 3,691 | 2.2x |

With thinking suppressed via the prompt suffix, the eval token count dropped to approximately 1:1 with the response size in some cases, and wall-clock times fell from the 60–90 second range to 27–36 seconds. However, continued monitoring revealed that the prompt-level suppression was unreliable: approximately half of subsequent calls still exhibited 2–8x thinking overhead. The `/no_think` suffix is a soft instruction that Qwen3 follows on a per-turn basis; the model's dual-mode architecture can re-engage reasoning when prompt content triggers it, regardless of the suffix.

This observation led to a second mid-run intervention. Ollama's API supports a runtime-level `think` parameter that controls thinking mode at the inference engine level rather than through prompt content. The pipeline's Ollama client was switched from the `/api/generate` endpoint to `/api/chat` with `"think": false` set in each request payload. This API-level control operates below the prompt layer and provides a deterministic guarantee that no thinking tokens are generated, rather than a probabilistic suppression.

The effect was immediate and complete. After the second restart at record 6,226, every call showed eval token counts strictly proportional to response length at the expected tokenizer ratio (~0.3 tokens per character), with zero thinking overhead:

| Wall clock | Response chars | Eval tokens (total generated) | Tokens/char ratio |
|-----------|---------------|-------------------------------|-------------------|
| 6.7s | 2,399 | 657 | 0.27 |
| 9.6s | 3,222 | 968 | 0.30 |
| 14.4s | 5,136 | 1,496 | 0.29 |
| 5.9s | 1,978 | 584 | 0.30 |
| 1.3s | 350 | 100 | 0.29 |

Wall-clock times dropped from the original 60–90 second range to 1–15 seconds per call — a 5–10x reduction compared to the thinking-active baseline. The throughput in tokens per second actually increased slightly (from 107–110 to 113–117 tok/s), likely because shorter generations encounter less memory-bandwidth pressure.

The progression across the three phases of the run — unconstrained thinking, prompt-level suppression, and API-level suppression — constitutes an unplanned but informative ablation. It demonstrates that for structured extraction tasks, extended reasoning provides no measurable benefit while imposing severe latency costs, and that the mechanism of suppression matters: prompt-level hints are insufficient, while API-level controls are effective.

These mid-run interventions are themselves methodological artifacts worth noting. The fact that the pipeline could be stopped and restarted twice with configuration changes at the 62% and 62.3% marks, with checkpoint-based resume preserving all prior work each time, validates the crash-recovery design described in Section X.12. The `disable_thinking` setting is recorded in the configuration snapshot so that future runs can be compared explicitly. For all subsequent runs, API-level thinking suppression is the default.

#### Infrastructure Sizing

Based on the timing data from the full run, single-GPU extraction of the 10,000-posting corpus required approximately 4–5 days of continuous processing with thinking mode active (see log file `SkillsExtraction_pipeline_run_full_10k_20260327_185602.log`). The first mid-run deployment of prompt-level thinking suppression at the 62% mark reduced per-call latency by approximately 2x for a subset of calls. The second deployment of API-level suppression at the 62.3% mark reduced per-call latency by 5–10x uniformly. For a full run with API-level thinking suppression active from the start, the projected single-GPU timeline is approximately 18–24 hours — a 5–6x improvement over the thinking-active baseline.

This timeline, while improved, would still be impractical for the refresh cycles envisioned in the dissertation's maintenance model. The timing breakdown confirmed that extraction (Stage 1) consumed the majority of wall-clock time, with verification and classification (Stages 2–4) collectively faster due to the lighter model and shorter prompts.

These observations motivated the multi-GPU parallelism design described in Section X.14.

### X.14 Multi-GPU Parallelism via vLLM

#### Motivation

The timing data from Section X.13 established that single-GPU extraction requires 4–5 days for the full corpus. For a system intended to support periodic refresh cycles, this timeline creates a practical constraint: each refresh run occupies the infrastructure for nearly a week, limiting the frequency and responsiveness of updates. Additionally, the 4.8–81.6 second variance in per-call wall-clock time suggested that a round-robin endpoint-selection strategy would be suboptimal, as fast endpoints would idle while waiting for slow calls to complete on other GPUs.

#### Architecture: Endpoint Pool and Windowed Execution

The multi-GPU design uses vLLM (a high-throughput LLM serving framework) deployed across 8 GPU endpoints, each exposing an OpenAI-compatible HTTP API on a distinct port. The pipeline communicates with these endpoints through a queue-based endpoint pool rather than round-robin selection.

The endpoint pool uses a `queue.Queue` with checkout/return semantics. When a worker thread needs to make an LLM call, it checks out an endpoint from the pool (blocking if all are in use). After the call completes — whether successfully or with an error — the endpoint is returned to the pool before any retry backoff sleep, ensuring that endpoints are not held idle during wait periods. This design provides natural backpressure: if all 8 GPUs are saturated, additional work blocks at the queue rather than overloading endpoints.

Execution uses a windowed concurrency model. Items are processed in windows of N (equal to the number of GPU endpoints). Within each window, all items are submitted concurrently to a thread pool, with each thread checking out its own endpoint from the queue. After all items in a window complete, the main thread sorts results by their original index and writes them to the checkpoint file in order. This preserves the checkpoint ordering guarantee required for crash recovery while enabling parallelism within each window.

Progress callbacks fire from the main thread during the ordered-write phase, not from worker threads. This prevents interleaved stdout output during concurrent execution.

#### Thread Safety

Several design properties ensure thread safety without complex synchronization:

- Each `call_vllm()` invocation uses a stateless `requests.post()` call with no shared HTTP session, eliminating connection-state races.
- The pipeline configuration (`cfg`) is read-only during execution; no worker thread modifies shared configuration.
- Checkpoint file handles are only accessed from the main thread during the ordered-write phase.
- The `RunStats.record_llm()` method, which is invoked via the timing callback from worker threads, is protected by a `threading.Lock` to prevent counter corruption.

#### Scope and Backend Independence

The windowed execution path activates only when `cfg.backend == "vllm"`. The Ollama and OpenRouter backends retain their existing sequential execution paths with no code changes. This ensures that the concurrent design does not introduce regression risk for the primary Ollama-based workflow that produced the initial extraction results.

#### Projected Performance

With 8 GPU endpoints operating concurrently, the theoretical speedup is 8x, reducing the full-corpus extraction timeline from 4–5 days to approximately 12–18 hours. In practice, the speedup will be moderated by load imbalance within windows (the window completes only when its slowest item finishes) and by the sequential checkpoint-write overhead between windows. However, even a 5–6x realized speedup would reduce the extraction timeline to under 24 hours, making weekly refresh cycles operationally feasible.

### X.15 Boundaries, Limitations, and Validity Notes

Several limitations remain explicit.

First, extraction from free text is inherently uncertain. No stage decomposition removes ambiguity in weakly written postings. Second, model and prompt sensitivity remain real risks; this is why versions and run metadata are retained. Third, the pipeline prioritizes auditability and stability over maximal throughput, although the stage-first execution architecture (Section X.12) substantially reduces the operational cost of this trade-off by eliminating redundant model loading.

Fourth, checkpoint-based resume assumes deterministic preprocessing. If input data or configuration changes between a crash and a restart under the same run identifier, the resumed stages may operate on inconsistent assumptions. The pipeline mitigates this by recording total record counts in checkpoint headers and refusing to resume when counts diverge, but it does not perform content-level validation of resumed state. For dissertation purposes, this is acceptable because runs are executed against fixed input snapshots.

Fifth, the operational challenges documented in Section X.13 — transient server errors, checkpoint corruption from concurrent writers, and multi-day execution timelines — are not unique to this implementation. They reflect the general cost structure of running large-scale LLM inference against shared infrastructure. The multi-GPU parallelism described in Section X.14 addresses the throughput constraint but introduces its own trade-off: windowed execution can lose up to N items on crash (where N is the number of concurrent endpoints), compared to at most one item under sequential execution.

These are accepted trade-offs in a dissertation context focused on explainable infrastructure for student-facing systems. The objective is not to maximize benchmark precision on a static corpus, but to create a refreshable process where outputs remain inspectable and governance remains feasible.

### X.16 Implications for Dissertation Evaluation

The staged extraction architecture directly supports the evaluation logic defined elsewhere in this dissertation, particularly traceability-oriented analysis. For selected course-career pairs, alignment signals can be decomposed back to:

- role-level weighted skills,
- canonical skill groupings,
- posting-level mentions,
- and source line evidence with stage-level audit history.

This decomposition is only possible because intermediate extraction artifacts are retained by design. The stage-first checkpoint architecture further strengthens this property: intermediate state is not merely retained in memory during execution, but persisted to inspectable JSONL files at each stage boundary. These checkpoint files serve as both recovery mechanisms and additional audit surfaces.

In this sense, auditability is not an implementation convenience; it is part of the methodological argument of the dissertation.

### X.17 Summary

This chapter has described a role-specialized, open-vocabulary skill extraction pipeline designed for traceability, refreshability, and integration into a student-facing alignment framework. The key design decisions are explicit stage separation with retained audit artifacts at both mention and job levels, stage-first execution scheduling with intermediate checkpoints to enable corpus-scale processing within feasible time budgets, and multi-GPU parallelism via windowed concurrent execution to reduce refresh-cycle timelines from days to hours. Together, these enable practical calibration, transparent failure analysis, crash-recoverable execution, and reproducible refresh-cycle comparisons.

Within the scope of this dissertation, extraction quality is evaluated not only by what skills are found, but by whether those skills can be justified, reconstructed, and reviewed through preserved intermediate evidence. That standard is central to deploying machine-assisted alignment in educational contexts where interpretability is a requirement rather than a preference.

