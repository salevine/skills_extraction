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

### X.13 Boundaries, Limitations, and Validity Notes

Several limitations remain explicit.

First, extraction from free text is inherently uncertain. No stage decomposition removes ambiguity in weakly written postings. Second, model and prompt sensitivity remain real risks; this is why versions and run metadata are retained. Third, the pipeline prioritizes auditability and stability over maximal throughput, although the stage-first execution architecture (Section X.12) substantially reduces the operational cost of this trade-off by eliminating redundant model loading.

Fourth, checkpoint-based resume assumes deterministic preprocessing. If input data or configuration changes between a crash and a restart under the same run identifier, the resumed stages may operate on inconsistent assumptions. The pipeline mitigates this by recording total record counts in checkpoint headers and refusing to resume when counts diverge, but it does not perform content-level validation of resumed state. For dissertation purposes, this is acceptable because runs are executed against fixed input snapshots.

These are accepted trade-offs in a dissertation context focused on explainable infrastructure for student-facing systems. The objective is not to maximize benchmark precision on a static corpus, but to create a refreshable process where outputs remain inspectable and governance remains feasible.

### X.14 Implications for Dissertation Evaluation

The staged extraction architecture directly supports the evaluation logic defined elsewhere in this dissertation, particularly traceability-oriented analysis. For selected course-career pairs, alignment signals can be decomposed back to:

- role-level weighted skills,
- canonical skill groupings,
- posting-level mentions,
- and source line evidence with stage-level audit history.

This decomposition is only possible because intermediate extraction artifacts are retained by design. The stage-first checkpoint architecture further strengthens this property: intermediate state is not merely retained in memory during execution, but persisted to inspectable JSONL files at each stage boundary. These checkpoint files serve as both recovery mechanisms and additional audit surfaces.

In this sense, auditability is not an implementation convenience; it is part of the methodological argument of the dissertation.

### X.15 Summary

This chapter has described a role-specialized, open-vocabulary skill extraction pipeline designed for traceability, refreshability, and integration into a student-facing alignment framework. The key design decisions are explicit stage separation with retained audit artifacts at both mention and job levels, and stage-first execution scheduling with intermediate checkpoints to enable corpus-scale processing within feasible time budgets. Together, these enable practical calibration, transparent failure analysis, crash-recoverable execution, and reproducible refresh-cycle comparisons.

Within the scope of this dissertation, extraction quality is evaluated not only by what skills are found, but by whether those skills can be justified, reconstructed, and reviewed through preserved intermediate evidence. That standard is central to deploying machine-assisted alignment in educational contexts where interpretability is a requirement rather than a preference.

