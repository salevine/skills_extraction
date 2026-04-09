# Future Enhancement: Reducing LLM Dependency in Skills Extraction

**Status:** Idea / Future Planning  
**Created:** 2026-04-06  
**Context:** The current pipeline uses large LLMs (Qwen3-14B, Mistral-Nemo-12B) at
runtime for every job posting. A full 10K-job run takes ~50 hours on 8 GPUs across
5 pipeline stages. This document explores two approaches to dramatically reduce that
cost while preserving extraction quality.

---

## The Core Insight

The current pipeline already produces high-quality labeled data as a byproduct of
every run. Each completed run generates:

- **Job descriptions** with preprocessed, section-tagged lines (Stage 0)
- **Skill mentions** with character-level spans, evidence text, and context windows (Stage 1)
- **Verification judgments** — is_skill, confidence, evidence (Stage 2)
- **Requirement classifications** — required / preferred / optional (Stage 3)
- **Hard/soft classifications** (Stage 4)

This is a supervised training set. The LLM has already done the expensive cognitive
work — we can transfer that knowledge into cheaper systems.

---

## Plan A: Distillation into a Small NER Model

### Concept

Fine-tune a small transformer model to perform token-level skill extraction (NER),
trained on the LLM's output. The small model replaces Stages 1-2 entirely and runs
on CPU at thousands of postings per minute.

### Why This Works

- **NER is a solved problem architecturally.** Token classification with transformers
  is mature, well-tooled, and fast at inference.
- **Your data is already span-annotated.** `CandidateSpan` records include
  `char_start`, `char_end`, `candidate_text`, `context_window`, and `section` —
  this maps directly to BIO-tagged NER training data.
- **The verification stage provides quality labels.** Mentions where `is_skill=True`
  and `final_confidence > 0.7` are high-quality positives. Mentions where
  `is_skill=False` are hard negatives (the model thought they were skills, the
  verifier disagreed — these are the most valuable negative examples).

### Step-by-Step Implementation

#### Step 1: Build the Training Data Converter

Write a script that reads Stage 1 + Stage 2 checkpoints and produces NER training
data in standard format (CoNLL or HuggingFace `datasets` format).

For each job posting:
1. Take the preprocessed text from Stage 0
2. For each verified mention (`is_skill=True`, `final_confidence >= 0.7`):
   - Use `char_start` / `char_end` to mark the span
   - Label tokens as `B-SKILL` (first token), `I-SKILL` (continuation), `O` (outside)
3. Optionally add sub-labels: `B-SKILL_HARD`, `B-SKILL_SOFT`, `B-SKILL_REQ`,
   `B-SKILL_PREF` to capture Stage 3-4 classifications in a single pass

**Data quality filters:**
- Exclude mentions with `verifier_status == "parse_failed"` or `confidence is None`
- Exclude mentions where extractor and verifier disagreed (low agreement = noisy label)
- Consider upsampling rare/novel skills to avoid the model learning only common ones

#### Step 2: Select and Fine-Tune a Base Model

**Recommended models (ranked by suitability):**

| Model | Parameters | Why | Tradeoff |
|-------|-----------|-----|----------|
| **microsoft/deberta-v3-base** | 86M | Best-in-class for token classification at this size. DeBERTa's disentangled attention handles positional relationships well, which matters for span detection in semi-structured text like job postings. | English-only, which is fine for your dataset |
| **google/bert-base-uncased** | 110M | Most battle-tested NER base model. Enormous ecosystem of examples, tutorials, and tooling. Slightly worse than DeBERTa but simpler to debug. | Older architecture, slightly lower ceiling |
| **Qwen/Qwen2.5-0.5B** | 500M | Decoder model, so it can do generative extraction (output skill list) rather than just tagging. Shares architecture lineage with your Qwen3-14B extractor, so distillation transfer may be more effective. | 5x larger than BERT, slower inference, needs GPU for reasonable speed |
| **numind/NuNER-v2.0** | 86M | Already pre-trained for NER tasks across diverse entity types. May need less fine-tuning data to reach good performance on skills specifically. | Less widely used, smaller community |

**Recommendation:** Start with **DeBERTa-v3-base**. It's the best balance of quality,
speed, and tooling. You can train it on a single GPU in under an hour and run
inference on CPU.

**Training approach:**
```
- Framework: HuggingFace Transformers + Trainer API
- Task: TokenClassification (BIO tagging)
- Training data: ~10K job postings × ~38 mentions avg = ~380K labeled spans
- Train/val/test split: 80/10/10 by job posting (not by mention, to avoid leakage)
- Epochs: 3-5 (with early stopping on validation F1)
- Learning rate: 2e-5 (standard for fine-tuning)
- Batch size: 16-32
- Max sequence length: 512 tokens (most job posting sections fit)
```

#### Step 3: Evaluate Against LLM Baseline

Run both systems on a held-out test set and compare:

- **Span-level F1:** Does the small model find the same skill mentions?
- **Exact match rate:** How often do spans match exactly vs. partially?
- **Novel skill recall:** Specifically measure performance on skills that appear
  fewer than 5 times in training data — this is where the model is most likely to
  fail.
- **Speed:** Measure postings-per-second on CPU vs. GPU LLM inference.

**Target:** Span-level F1 >= 0.90 compared to the LLM extractor. Below 0.85,
the quality loss is likely too high to justify.

#### Step 4: Deploy

The distilled model is a single ~350MB file that runs on CPU with the HuggingFace
pipeline API. No vLLM, no GPU allocation, no endpoint management. Inference is
~1000 postings/minute on a modern CPU.

### Limitations

- **Cannot discover truly novel skills.** If a skill term never appeared in training
  data, the model has no signal for it. This is the fundamental tradeoff.
- **Degradation over time.** As new technologies and skills emerge (e.g., "vibe coding"
  in 2025, "MCP development" in 2026), the model's recall will decay.
- **Multi-word and compositional skills are harder.** "Experience with CI/CD pipelines
  using GitHub Actions" contains at least 2-3 skills depending on granularity.
  NER models handle this less flexibly than generative LLMs.

---

## Plan B: Hybrid — Small Model + LLM Escalation

### Concept

Use the distilled model from Plan A as a fast first pass, but add an LLM-based
second pass for uncertain cases. This preserves the LLM's ability to discover novel
skills while cutting LLM compute by 80-90%.

### Architecture

```
Job Posting
    │
    ▼
┌──────────────┐
│  Small Model  │  (CPU, ~1ms per posting)
│  (DeBERTa)   │
└──────┬───────┘
       │
       ├── High confidence (>0.85) ──────► Accept directly
       │
       ├── Low confidence (<0.3) ─────────► Reject directly
       │
       └── Uncertain (0.3 - 0.85) ───────► LLM Review
                                               │
                                               ▼
                                         ┌──────────┐
                                         │  Qwen3   │  (GPU, ~2s per posting)
                                         │  14B     │
                                         └──────────┘
```

### Why This Works

- **Most mentions are easy.** "Python", "project management", "SQL" — the small
  model handles these with high confidence. Only ambiguous or novel cases need
  the LLM.
- **The LLM sees fewer items.** Instead of processing every mention in every posting,
  it only reviews the uncertain fraction. In practice, this is typically 10-20%
  of mentions.
- **Novel skill discovery is preserved.** You can also send a random sample of
  "high confidence rejections" to the LLM periodically to catch false negatives —
  skills the small model is confidently wrong about.

### Implementation Steps

#### Step 1: Train the Small Model (Same as Plan A, Steps 1-3)

The only change: train the model to also output a **calibrated confidence score**.
Use temperature scaling or Platt scaling on the softmax logits so that when the
model says 0.9, it really is right 90% of the time.

#### Step 2: Build the Escalation Router

Add a component to the pipeline that:
1. Runs the small model on all postings
2. Collects mentions where max token probability is below the confidence threshold
3. Batches uncertain postings and sends them to the LLM (reusing your existing
   vLLM infrastructure)
4. Merges results

This fits naturally into your existing stage-first architecture — Stage 1 becomes
"fast extraction" and a new Stage 1b becomes "LLM review of uncertain cases."

#### Step 3: Tune the Confidence Thresholds

The accept/reject thresholds control the quality-speed tradeoff:
- **Conservative** (accept > 0.95, reject < 0.1): ~40% goes to LLM, high quality
- **Balanced** (accept > 0.85, reject < 0.3): ~15% goes to LLM, good quality
- **Aggressive** (accept > 0.7, reject < 0.5): ~5% goes to LLM, some quality loss

Tune these on your validation set by measuring F1 at each threshold setting.

#### Step 4: Recall Drift Monitoring

Every 6 months (or whenever you ingest a new dataset):
1. Run the LLM extractor on a random sample of 500 postings
2. Compare LLM mentions against small model mentions
3. Measure recall drift: `(LLM_mentions - SmallModel_mentions) / LLM_mentions`
4. If drift > 10%, retrain the small model on the new LLM-labeled data

This is your "LLM refresh" cycle — the LLM becomes a periodic calibration tool
rather than the runtime engine.

### Model Recommendations for the Hybrid Approach

For the **escalation LLM**, you have flexibility since it's handling a small volume:

| Model | When to Use |
|-------|------------|
| **Qwen3-14B (current)** | Best quality, already integrated. Use if you have GPU access. |
| **Qwen2.5-7B-Instruct** | Half the VRAM, 80% of the quality. Good if GPU budget is tight. |
| **Mistral-Nemo-12B** | Already deployed for your verifier. Could reuse for escalation to simplify infrastructure. |
| **Cloud API (Claude/GPT)** | If volume is low enough (<1000 postings/month), a cloud API call is cheaper than maintaining GPU infrastructure. At ~15% escalation rate on 10K postings, that's ~1500 API calls — a few dollars. |

---

## Comparison of Approaches

| Dimension | Plan A (Full Distillation) | Plan B (Hybrid) |
|-----------|---------------------------|-----------------|
| **LLM dependency at runtime** | None | Reduced 80-90% |
| **Novel skill discovery** | None until retrain | Preserved via escalation |
| **Infrastructure** | CPU only | CPU + occasional GPU |
| **Speed** | ~1000 postings/min | ~500 postings/min (with escalation) |
| **Quality ceiling** | Limited by training data | Near-LLM quality |
| **Complexity** | Simple (one model) | Moderate (routing logic, two models) |
| **Best for** | Stable, well-known skill domains | Evolving domains, research use |

---

## Recommended Approach

**Start with Plan A, evolve to Plan B.**

1. Build the training data converter first — this is useful regardless of which
   plan you pursue, and it forces you to deeply understand your labeled data.
2. Train a DeBERTa model and measure F1 against the LLM baseline.
3. If F1 >= 0.90: Plan A may be sufficient. Deploy it and monitor.
4. If F1 is 0.80-0.90: The gap is in novel/ambiguous skills. Add the LLM escalation
   path (Plan B) to close it.
5. If F1 < 0.80: The task may be too nuanced for a small NER model. Consider a
   larger distilled model (Qwen2.5-0.5B generative) or stick with the LLM pipeline
   but optimize throughput instead.

The 6-month retrain cycle works for either plan. Each retrain uses the latest LLM
run as training data, so the small model gradually absorbs new skills as they emerge
in postings.

---

## Prerequisites

- **Completed 10K run with all stages.** You need verified, classified mentions
  to build quality training data. (In progress as of 2026-04-02.)
- **HuggingFace Transformers + datasets installed.** Standard pip packages.
- **~1 hour of GPU time for training.** A single consumer GPU (RTX 3090 or similar)
  is sufficient. Training can also run on Google Colab.
- **Evaluation framework.** Script to compute span-level precision/recall/F1
  comparing two extraction runs.

---

## Estimated Effort

| Phase | Work | Time Estimate |
|-------|------|---------------|
| Training data converter | Script to read checkpoints, emit BIO-tagged data | 1-2 days |
| Model training + eval | Fine-tune DeBERTa, measure F1 | 1-2 days |
| Pipeline integration (Plan A) | Replace Stage 1 LLM call with model inference | 1 day |
| Escalation router (Plan B) | Confidence routing, LLM fallback, merge logic | 2-3 days |
| Drift monitoring | Sample-and-compare script, threshold tuning | 1 day |

Total: ~1 week for Plan A, ~2 weeks for Plan B.
