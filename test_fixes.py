"""Regression tests for pipeline fixes (2026-04-13).

Tests:
1. Stage-authority: stage-2 rejection overrides extractor is_skill
2. Mention grounding: repeated spans anchor to distinct lines
3. Mention grounding: ungroundable spans are skipped

Run: python3 test_fixes.py
"""

import sys
import types


# ---------------------------------------------------------------------------
# Test 1: Stage-authority bug — stage-2 rejection must win
# ---------------------------------------------------------------------------

def test_stage5_authority():
    """If stage 2 rejected a mention, final is_skill must be False,
    even when the extractor wrote is_skill=True."""
    from skills_extraction.pipeline import _run_stage5_assemble, _mention_is_skill
    from skills_extraction.config import PipelineConfig
    from skills_extraction.schemas import ParsedLine
    from skills_extraction.checkpoint import serialize_mentions_for_job

    cfg = PipelineConfig(
        skip_llm=True,
        verifier_enabled=True,
        requirement_classifier_enabled=True,
        hardsoft_classifier_enabled=True,
    )

    # Simulate a job with one mention where extractor says is_skill=True
    pl = ParsedLine(
        line_id="testjob_L0001",
        section="requirements",
        text="Must have 5+ years experience with Python and SQL.",
        char_start=0,
        char_end=50,
        boilerplate_label="skills_relevant",
        line_index=0,
    )
    mention = {
        "skill_span": "Python",
        "normalized_candidate": "Python",
        "span_confidence": 0.85,
        "evidence": "5+ years experience with Python",
        "is_skill": True,           # extractor says True
        "requirement_level": "required",
        "hard_soft": "hard",
        "_glob_char_start": 38,
        "_glob_char_end": 44,
        "_line_char_start": 38,
        "_line_char_end": 44,
        "_offset_valid": True,
        "_parsed_line": pl,
    }

    # Build stage1_data with the mention
    ser_mentions, lines_reg = serialize_mentions_for_job([mention])
    jobs = [{"JobTitle": "Test", "Description": pl.text}]
    stage1_data = [{
        "job_index": 0,
        "job_key": "testjob",
        "extractor_model_used": "test-model",
        "raw_mentions": ser_mentions,
        "_parsed_lines": lines_reg,
    }]

    # Stage 2 REJECTED the mention
    stage2_data = [{
        "job_index": 0,
        "job_key": "testjob",
        "mention_idx": 0,
        "verifier_output": {
            "status": "rejected",
            "model": "verifier-model",
            "is_skill": False,
            "confidence": 0.9,
            "evidence": "This is a duty, not a skill",
            "notes": "rejected by verifier",
        },
    }]

    # Stages 3-4 skipped (since rejected)
    stage3_data = [{
        "job_index": 0, "job_key": "testjob", "mention_idx": 0,
        "requirement_output": {"status": "skipped"},
    }]
    stage4_data = [{
        "job_index": 0, "job_key": "testjob", "mention_idx": 0,
        "hardsoft_output": {"status": "skipped"},
    }]

    augmented = _run_stage5_assemble(
        jobs, stage1_data, stage2_data, stage3_data, stage4_data,
        cfg, "test_run", "2026-01-01T00:00:00Z",
    )

    assert len(augmented) == 1
    mentions = augmented[0]["skill_mentions"]
    assert len(mentions) == 1

    m = mentions[0]
    # THE KEY ASSERTION: stage-2 rejection must override extractor is_skill
    assert m["is_skill"] is False, (
        f"FAIL: is_skill={m['is_skill']}; expected False (stage-2 rejected)"
    )
    assert m["verifier_status"] == "rejected", (
        f"FAIL: verifier_status={m['verifier_status']}; expected 'rejected'"
    )
    # Audit should preserve extractor's original is_skill for traceability
    ext_audit = m["pipeline_audit"]["extractor"]["output"]
    assert ext_audit["is_skill"] is True, (
        f"FAIL: audit extractor is_skill should be True (original value)"
    )
    print("PASS: test_stage5_authority")


# ---------------------------------------------------------------------------
# Test 2: Stage-authority — stage-3/4 results win over extractor
# ---------------------------------------------------------------------------

def test_stage5_classifier_authority():
    """If stage 3 completed with 'optional' and extractor said 'required',
    final requirement_level must be 'optional'."""
    from skills_extraction.pipeline import _run_stage5_assemble
    from skills_extraction.config import PipelineConfig
    from skills_extraction.schemas import ParsedLine
    from skills_extraction.checkpoint import serialize_mentions_for_job

    cfg = PipelineConfig(
        skip_llm=True,
        verifier_enabled=True,
        requirement_classifier_enabled=True,
        hardsoft_classifier_enabled=True,
    )

    pl = ParsedLine(
        line_id="testjob2_L0001",
        section="preferred",
        text="Familiarity with Docker and Kubernetes is a plus.",
        char_start=0, char_end=48,
        boilerplate_label="skills_relevant",
        line_index=0,
    )
    mention = {
        "skill_span": "Docker",
        "normalized_candidate": "Docker",
        "span_confidence": 0.80,
        "evidence": "Familiarity with Docker",
        "is_skill": True,
        "requirement_level": "required",  # extractor says required
        "hard_soft": "soft",              # extractor says soft
        "_glob_char_start": 22, "_glob_char_end": 28,
        "_line_char_start": 22, "_line_char_end": 28,
        "_offset_valid": True,
        "_parsed_line": pl,
    }

    ser_mentions, lines_reg = serialize_mentions_for_job([mention])
    jobs = [{"JobTitle": "Test2", "Description": pl.text}]
    stage1_data = [{
        "job_index": 0, "job_key": "testjob2",
        "extractor_model_used": "test-model",
        "raw_mentions": ser_mentions,
        "_parsed_lines": lines_reg,
    }]
    stage2_data = [{
        "job_index": 0, "job_key": "testjob2", "mention_idx": 0,
        "verifier_output": {
            "status": "accepted", "model": "v", "is_skill": True,
            "confidence": 0.88, "evidence": "", "notes": "",
        },
    }]
    stage3_data = [{
        "job_index": 0, "job_key": "testjob2", "mention_idx": 0,
        "requirement_output": {
            "status": "completed", "model": "r",
            "requirement_level": "optional",  # stage 3 says optional
            "confidence": 0.85, "evidence": "", "notes": "",
        },
    }]
    stage4_data = [{
        "job_index": 0, "job_key": "testjob2", "mention_idx": 0,
        "hardsoft_output": {
            "status": "completed", "model": "h",
            "hard_soft": "hard",  # stage 4 says hard
            "confidence": 0.90, "evidence": "", "notes": "",
        },
    }]

    augmented = _run_stage5_assemble(
        jobs, stage1_data, stage2_data, stage3_data, stage4_data,
        cfg, "test_run2", "2026-01-01T00:00:00Z",
    )
    m = augmented[0]["skill_mentions"][0]
    assert m["requirement_level"] == "optional", (
        f"FAIL: requirement_level={m['requirement_level']}; expected 'optional' (stage-3)"
    )
    assert m["hard_soft"] == "hard", (
        f"FAIL: hard_soft={m['hard_soft']}; expected 'hard' (stage-4)"
    )
    print("PASS: test_stage5_classifier_authority")


# ---------------------------------------------------------------------------
# Test 3: Mention grounding — repeated spans get distinct anchors
# ---------------------------------------------------------------------------

def test_repeated_span_grounding():
    """Two mentions of the same span should anchor to different source lines."""
    from skills_extraction.llm_extractor import _anchor_span_to_line, _build_source_lines
    from skills_extraction.schemas import ParsedLine

    # Simulate two source lines that both contain "Python"
    lines = [
        ParsedLine(line_id="j_L0001", section="requirements", text="Must know Python.",
                    char_start=0, char_end=17, boilerplate_label="skills_relevant", line_index=0),
        ParsedLine(line_id="j_L0002", section="preferred", text="Python experience preferred.",
                    char_start=18, char_end=46, boilerplate_label="skills_relevant", line_index=1),
    ]

    used = {}
    # First anchor
    r1 = _anchor_span_to_line("Python", "", "requirements", lines, used)
    assert r1 is not None, "FAIL: first anchor should not be None"

    # Second anchor — should get a different position
    r2 = _anchor_span_to_line("Python", "", "preferred", lines, used)
    assert r2 is not None, "FAIL: second anchor should not be None"

    pl1, cs1, ce1 = r1
    pl2, cs2, ce2 = r2

    # They should be on different lines (or at least different positions)
    assert (pl1.line_id, cs1) != (pl2.line_id, cs2), (
        f"FAIL: both anchors are identical: ({pl1.line_id}, {cs1})"
    )
    print("PASS: test_repeated_span_grounding")


# ---------------------------------------------------------------------------
# Test 4: Mention grounding — ungroundable span is skipped
# ---------------------------------------------------------------------------

def test_ungroundable_span_skipped():
    """A span that doesn't appear in any source line returns None."""
    from skills_extraction.llm_extractor import _anchor_span_to_line
    from skills_extraction.schemas import ParsedLine

    lines = [
        ParsedLine(line_id="j_L0001", section="body", text="We need a Java developer.",
                    char_start=0, char_end=25, boilerplate_label="skills_relevant", line_index=0),
    ]

    used = {}
    result = _anchor_span_to_line("HallucinatedSkill", "", "body", lines, used)
    assert result is None, f"FAIL: ungroundable span should return None, got {result}"
    print("PASS: test_ungroundable_span_skipped")


# ---------------------------------------------------------------------------
# Test 5: Context-based grounding preferred
# ---------------------------------------------------------------------------

def test_context_match_preferred():
    """When model context matches a real line, that line is used even if
    another line also contains the span."""
    from skills_extraction.llm_extractor import _anchor_span_to_line
    from skills_extraction.schemas import ParsedLine

    lines = [
        ParsedLine(line_id="j_L0001", section="body", text="Use SQL for reports.",
                    char_start=0, char_end=20, boilerplate_label="uncertain", line_index=0),
        ParsedLine(line_id="j_L0002", section="requirements", text="3+ years SQL experience required.",
                    char_start=21, char_end=53, boilerplate_label="skills_relevant", line_index=1),
    ]

    used = {}
    # Context matches line 2
    result = _anchor_span_to_line("SQL", "3+ years SQL experience required.", "requirements", lines, used)
    assert result is not None
    pl, cs, ce = result
    assert pl.line_id == "j_L0002", f"FAIL: expected L0002, got {pl.line_id}"
    print("PASS: test_context_match_preferred")


# ---------------------------------------------------------------------------
# Test 6: Offsets index into description_normalized, not description_raw
# ---------------------------------------------------------------------------

def test_offsets_match_normalized():
    """Stage-5 output must include description_normalized, and char_start/char_end
    from a grounded mention must correctly index into it."""
    from skills_extraction.pipeline import _run_stage5_assemble
    from skills_extraction.config import PipelineConfig
    from skills_extraction.schemas import ParsedLine
    from skills_extraction.checkpoint import serialize_mentions_for_job
    from skills_extraction.preprocessing import preprocess_description
    from skills_extraction.sectioning import split_inline_section_headers

    cfg = PipelineConfig(
        skip_llm=True,
        verifier_enabled=False,
        requirement_classifier_enabled=False,
        hardsoft_classifier_enabled=False,
    )

    # Raw text with extra whitespace that normalization will collapse
    raw = "Must   have  experience  with  Python  and  SQL."
    pre = preprocess_description(raw)
    norm = split_inline_section_headers(pre.description_normalized)
    # norm collapses to "Must have experience with Python and SQL."

    # Find Python in normalized text
    py_start = norm.find("Python")
    assert py_start >= 0
    py_end = py_start + len("Python")

    pl = ParsedLine(
        line_id="offset_test_L0001", section="requirements",
        text=norm, char_start=0, char_end=len(norm),
        boilerplate_label="skills_relevant", line_index=0,
    )

    mention = {
        "skill_span": "Python",
        "normalized_candidate": "Python",
        "span_confidence": 0.85,
        "evidence": "experience with Python",
        "is_skill": True,
        "requirement_level": "required",
        "hard_soft": "hard",
        "_glob_char_start": py_start,
        "_glob_char_end": py_end,
        "_line_char_start": py_start,
        "_line_char_end": py_end,
        "_offset_valid": True,
        "_parsed_line": pl,
    }

    ser_mentions, lines_reg = serialize_mentions_for_job([mention])
    jobs = [{"JobTitle": "Offset Test", "Description": raw}]
    stage1_data = [{
        "job_index": 0, "job_key": "offset_test",
        "extractor_model_used": "test-model",
        "raw_mentions": ser_mentions,
        "_parsed_lines": lines_reg,
    }]
    stage2_data = [{"job_index": 0, "job_key": "offset_test", "mention_idx": 0,
                    "verifier_output": {"status": "skipped"}}]
    stage3_data = [{"job_index": 0, "job_key": "offset_test", "mention_idx": 0,
                    "requirement_output": {"status": "skipped"}}]
    stage4_data = [{"job_index": 0, "job_key": "offset_test", "mention_idx": 0,
                    "hardsoft_output": {"status": "skipped"}}]

    augmented = _run_stage5_assemble(
        jobs, stage1_data, stage2_data, stage3_data, stage4_data,
        cfg, "offset_test_run", "2026-01-01T00:00:00Z",
    )

    job_out = augmented[0]
    # description_normalized must be present
    assert "description_normalized" in job_out, (
        "FAIL: description_normalized not in output"
    )
    desc_norm = job_out["description_normalized"]
    m = job_out["skill_mentions"][0]
    cs, ce = m["char_start"], m["char_end"]
    sliced = desc_norm[cs:ce]
    assert sliced == "Python", (
        f"FAIL: desc_normalized[{cs}:{ce}] = {sliced!r}, expected 'Python'"
    )
    print("PASS: test_offsets_match_normalized")


# ---------------------------------------------------------------------------
# Test 7: Thread-local sessions are distinct per thread
# ---------------------------------------------------------------------------

def test_thread_local_sessions():
    """Concurrent threads must each get their own requests.Session instance."""
    import threading
    from skills_extraction.llm_vllm import _get_session

    sessions = {}
    barrier = threading.Barrier(2)

    def _capture(name):
        s = _get_session()
        sessions[name] = id(s)
        barrier.wait()  # keep both threads alive so objects can't be reused

    t1 = threading.Thread(target=_capture, args=("t1",))
    t2 = threading.Thread(target=_capture, args=("t2",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert sessions["t1"] != sessions["t2"], (
        f"FAIL: both threads got the same session id: {sessions['t1']}"
    )
    print("PASS: test_thread_local_sessions")


# ---------------------------------------------------------------------------
# Test 8: Qwen vLLM payload applies both no-think controls
# ---------------------------------------------------------------------------

def test_vllm_qwen_no_think_payload():
    """Qwen on vLLM should get both the request kwarg and the prompt-level
    /no_think suffix when disable_thinking is enabled."""
    from skills_extraction.config import PipelineConfig
    from skills_extraction.llm_vllm import _build_vllm_payload

    cfg = PipelineConfig(backend="vllm", disable_thinking=True)
    payload = _build_vllm_payload(
        cfg,
        "Qwen/Qwen3-14B",
        "Return JSON only.",
        "Extract skills from this description.",
        0.1,
    )

    assert payload.get("chat_template_kwargs") == {"enable_thinking": False}, (
        f"FAIL: missing Qwen chat_template_kwargs: {payload}"
    )
    user_text = payload["messages"][1]["content"]
    assert user_text.endswith("\n/no_think"), (
        f"FAIL: missing /no_think suffix in user payload: {user_text!r}"
    )

    payload_non_qwen = _build_vllm_payload(
        cfg,
        "mistralai/Mistral-Nemo-Instruct-2407",
        "Return JSON only.",
        "Extract skills from this description.",
        0.1,
    )
    assert "chat_template_kwargs" not in payload_non_qwen, (
        "FAIL: non-Qwen model should not receive Qwen thinking kwargs"
    )
    assert not payload_non_qwen["messages"][1]["content"].endswith("/no_think"), (
        "FAIL: non-Qwen model should not receive /no_think suffix"
    )
    print("PASS: test_vllm_qwen_no_think_payload")


# ---------------------------------------------------------------------------
# Test 9: Leading think blocks are stripped before downstream parsing
# ---------------------------------------------------------------------------

def test_vllm_strip_leading_think_block():
    """If vLLM still emits a leading <think> block, strip it so JSON parsing
    consumes the final answer content."""
    from skills_extraction.llm_vllm import _strip_leading_think_block

    raw = "<think>\nlong reasoning\n</think>\n{\"mentions\":[]}"
    cleaned = _strip_leading_think_block(raw)
    assert cleaned == "{\"mentions\":[]}", (
        f"FAIL: think block not stripped correctly: {cleaned!r}"
    )
    print("PASS: test_vllm_strip_leading_think_block")


# ---------------------------------------------------------------------------
# Test 10: _run_rolling bounds in-flight futures
# ---------------------------------------------------------------------------

def test_rolling_bounded():
    """_run_rolling must not start work arbitrarily far ahead of next_write."""
    import threading
    import time
    import skills_extraction.pipeline as pipeline_mod
    from skills_extraction.pipeline import _run_rolling

    max_workers = 2
    total = 20
    bound = max_workers * 2
    started = []
    release_first = threading.Event()
    failure = []
    lock = threading.Lock()

    def _process(i):
        with lock:
            started.append(i)
        if i == 0:
            if not release_first.wait(timeout=5):
                raise AssertionError("FAIL: timed out waiting to release first item")
        else:
            time.sleep(0.01)
        return {"idx": i}

    records = []
    class FakeFH:
        def write(self, data): pass
        def flush(self): pass

    def _run():
        try:
            _run_rolling(
                total=total,
                start_idx=0,
                max_workers=max_workers,
                process_fn=_process,
                progress_fn=None,
                fh=FakeFH(),
                records=records,
            )
        except Exception as exc:
            failure.append(exc)

    _orig = pipeline_mod.append_checkpoint_record
    pipeline_mod.append_checkpoint_record = lambda fh, rec: None
    try:
        runner = threading.Thread(target=_run)
        runner.start()

        deadline = time.time() + 2.0
        while time.time() < deadline:
            with lock:
                started_count = len(started)
            if started_count >= bound:
                break
            time.sleep(0.01)

        # Keep item 0 blocked briefly; if scheduling is unbounded, the free
        # worker will continue starting jobs far beyond the allowed backlog.
        time.sleep(0.1)
        with lock:
            started_before_release = len(started)

        assert started_before_release <= bound, (
            f"FAIL: started {started_before_release} items before idx 0 completed; "
            f"expected at most {bound}"
        )

        release_first.set()
        runner.join(timeout=5)
        assert not runner.is_alive(), "FAIL: _run_rolling did not finish"
    finally:
        release_first.set()
        pipeline_mod.append_checkpoint_record = _orig

    if failure:
        raise failure[0]

    assert len(records) == total, f"FAIL: expected {total} records, got {len(records)}"
    print("PASS: test_rolling_bounded")


# ---------------------------------------------------------------------------
# Test 11: Classifier role strings are distinct from verifier
# ---------------------------------------------------------------------------

def test_classifier_role_strings():
    """Stage 3/4 classifiers must pass distinct role names through call_llm,
    not reuse 'verifier', so timing stats can distinguish them."""
    import ast
    from pathlib import Path

    pkg = Path("skills_extraction")
    req_src = (pkg / "llm_requirement_classifier.py").read_text()
    hs_src = (pkg / "llm_hardsoft_classifier.py").read_text()

    assert 'role="requirement_classifier"' in req_src, (
        "FAIL: llm_requirement_classifier still uses wrong role string"
    )
    assert 'role="hardsoft_classifier"' in hs_src, (
        "FAIL: llm_hardsoft_classifier still uses wrong role string"
    )
    assert 'role="verifier"' not in req_src, (
        "FAIL: llm_requirement_classifier still contains role='verifier'"
    )
    assert 'role="verifier"' not in hs_src, (
        "FAIL: llm_hardsoft_classifier still contains role='verifier'"
    )
    print("PASS: test_classifier_role_strings")


# ---------------------------------------------------------------------------
# Test 12: Zero-mention job counts as success, not failure
# ---------------------------------------------------------------------------

def test_zero_mention_job_is_success():
    """A job that produces zero skill mentions should still count as
    jobs_success (the pipeline completed normally), not jobs_failed."""
    from skills_extraction.pipeline import _run_stage5_assemble
    from skills_extraction.config import PipelineConfig

    cfg = PipelineConfig(
        skip_llm=True,
        verifier_enabled=False,
        requirement_classifier_enabled=False,
        hardsoft_classifier_enabled=False,
    )

    # Job with a description that yields zero mentions
    jobs = [{"JobTitle": "Empty Test", "Description": "This is a test job with no skills."}]
    stage1_data = [{
        "job_index": 0, "job_key": "zero_mention_test",
        "extractor_model_used": "test-model",
        "raw_mentions": [],  # zero mentions
    }]
    stage2_data = []
    stage3_data = []
    stage4_data = []

    augmented = _run_stage5_assemble(
        jobs, stage1_data, stage2_data, stage3_data, stage4_data,
        cfg, "zero_test_run", "2026-01-01T00:00:00Z",
    )

    assert len(augmented) == 1, f"FAIL: expected 1 augmented job, got {len(augmented)}"
    job_out = augmented[0]
    assert job_out.get("skill_mentions") == [], (
        f"FAIL: expected empty skill_mentions, got {job_out.get('skill_mentions')}"
    )
    assert not (job_out.get("extraction_metadata") or {}).get("error"), (
        f"FAIL: zero-mention job should not carry an error, got {job_out.get('extraction_metadata')}"
    )

    # Simulate the stats accounting from run_pipeline
    jobs_success = sum(
        1 for j in augmented
        if not (j.get("extraction_metadata") or {}).get("error")
    )
    jobs_failed = len(jobs) - jobs_success
    assert jobs_success == 1, f"FAIL: zero-mention job should be success, got {jobs_success}"
    assert jobs_failed == 0, f"FAIL: zero-mention job should not be failed, got {jobs_failed}"
    print("PASS: test_zero_mention_job_is_success")


# ---------------------------------------------------------------------------
# Test 13: Stage-1 extraction failure counts as failed job
# ---------------------------------------------------------------------------

def test_stage1_error_job_is_failed():
    """A job with an explicit stage-1 extraction error should assemble, but it
    must be counted as failed and carry the error into extraction_metadata."""
    from skills_extraction.pipeline import _run_stage5_assemble
    from skills_extraction.config import PipelineConfig

    cfg = PipelineConfig(
        skip_llm=True,
        verifier_enabled=False,
        requirement_classifier_enabled=False,
        hardsoft_classifier_enabled=False,
    )

    jobs = [{"JobTitle": "Failed Extract", "Description": "Python preferred."}]
    stage1_data = [{
        "job_index": 0,
        "job_key": "stage1_failed_job",
        "extractor_model_used": "test-model",
        "raw_mentions": [],
        "stage1_error": "extractor_json_parse_failed: stage1_failed_job: bad payload",
    }]

    augmented = _run_stage5_assemble(
        jobs, stage1_data, [], [], [],
        cfg, "failed_job_run", "2026-01-01T00:00:00Z",
    )

    assert len(augmented) == 1, f"FAIL: expected 1 augmented job, got {len(augmented)}"
    job_out = augmented[0]
    error = (job_out.get("extraction_metadata") or {}).get("error")
    assert error, "FAIL: stage1 failure should propagate into extraction_metadata.error"

    jobs_success = sum(
        1 for j in augmented
        if not (j.get("extraction_metadata") or {}).get("error")
    )
    jobs_failed = len(jobs) - jobs_success
    assert jobs_success == 0, f"FAIL: stage1-failed job should not be success, got {jobs_success}"
    assert jobs_failed == 1, f"FAIL: stage1-failed job should count as failed, got {jobs_failed}"
    print("PASS: test_stage1_error_job_is_failed")


# ---------------------------------------------------------------------------
# Test 14: ParsedLine dedup in checkpoint serialization
# ---------------------------------------------------------------------------

def test_checkpoint_parsed_line_dedup():
    """serialize_mentions_for_job should deduplicate ParsedLine data into a
    registry, and deserialize_mentions_for_job should reconstruct correctly."""
    from skills_extraction.schemas import ParsedLine
    from skills_extraction.checkpoint import (
        serialize_mentions_for_job, deserialize_mentions_for_job,
    )

    pl = ParsedLine(
        line_id="dedup_L0001", section="requirements",
        text="Python and SQL required.", char_start=0, char_end=24,
        boilerplate_label="skills_relevant", line_index=0,
    )
    mentions = [
        {"skill_span": "Python", "_parsed_line": pl, "_glob_char_start": 0, "_glob_char_end": 6},
        {"skill_span": "SQL", "_parsed_line": pl, "_glob_char_start": 11, "_glob_char_end": 14},
    ]

    ser, registry = serialize_mentions_for_job(mentions)

    # Registry should have exactly 1 entry (both mentions share the same line)
    assert len(registry) == 1, f"FAIL: expected 1 registry entry, got {len(registry)}"
    assert "dedup_L0001" in registry

    # Serialized mentions should NOT have _parsed_line_dict (only _parsed_line_id)
    for s in ser:
        assert "_parsed_line_dict" not in s, "FAIL: serialized mention still has full _parsed_line_dict"
        assert "_parsed_line_id" in s

    # Deserialize and check reconstruction
    restored = deserialize_mentions_for_job(ser, registry)
    assert len(restored) == 2
    for r in restored:
        assert "_parsed_line" in r
        assert r["_parsed_line"].line_id == "dedup_L0001"
        assert r["_parsed_line"].text == "Python and SQL required."

    print("PASS: test_checkpoint_parsed_line_dedup")


# ---------------------------------------------------------------------------
# Test 15: Backward compat — old checkpoints with inline _parsed_line_dict
# ---------------------------------------------------------------------------

def test_checkpoint_backward_compat():
    """Older checkpoints that embed _parsed_line_dict per mention (no registry)
    must still deserialize correctly."""
    from skills_extraction.schemas import ParsedLine
    from skills_extraction.checkpoint import deserialize_mentions_for_job

    # Simulate old format: _parsed_line_dict inline, no registry
    old_format_mention = {
        "skill_span": "Java",
        "_parsed_line_dict": {
            "line_id": "old_L0001", "section": "body",
            "text": "Java required.", "char_start": 0, "char_end": 14,
            "boilerplate_label": "skills_relevant", "line_index": 0,
        },
        "_parsed_line_id": "old_L0001",
    }

    restored = deserialize_mentions_for_job([old_format_mention], None)
    assert len(restored) == 1
    m = restored[0]
    assert "_parsed_line" in m
    assert m["_parsed_line"].line_id == "old_L0001"
    assert m["_parsed_line"].text == "Java required."
    print("PASS: test_checkpoint_backward_compat")


# ---------------------------------------------------------------------------
# Test 16: Batched flushing
# ---------------------------------------------------------------------------

def test_batched_flush():
    """With flush_interval > 1, flushes happen every N records, not every record."""
    import io
    from skills_extraction.checkpoint import (
        set_flush_interval, append_checkpoint_record, flush_checkpoint,
    )

    flush_count = [0]
    class CountingWriter(io.StringIO):
        def flush(self):
            flush_count[0] += 1
            super().flush()

    fh = CountingWriter()
    set_flush_interval(4)
    flush_count[0] = 0

    try:
        for i in range(10):
            append_checkpoint_record(fh, {"idx": i})

        # With interval=4 and 10 records: expect floor(10/4) = 2 flushes
        assert flush_count[0] == 2, f"FAIL: expected 2 flushes, got {flush_count[0]}"

        # Force flush should add one more
        flush_checkpoint(fh)
        assert flush_count[0] == 3, f"FAIL: expected 3 after force flush, got {flush_count[0]}"
    finally:
        set_flush_interval(1)  # restore default

    print("PASS: test_batched_flush")


# ---------------------------------------------------------------------------
# Test 17: Fingerprint mismatch invalidates checkpoint
# ---------------------------------------------------------------------------

def test_fingerprint_mismatch_invalidates():
    """_load_or_run_stage should rerun from scratch when fingerprint mismatches."""
    import tempfile, shutil
    from pathlib import Path
    from skills_extraction.pipeline import _load_or_run_stage
    from skills_extraction.checkpoint import (
        write_checkpoint_header, write_checkpoint_footer, append_checkpoint_record,
    )

    tmpdir = Path(tempfile.mkdtemp())
    try:
        ckpt_dir = tmpdir / "checkpoints"
        ckpt_dir.mkdir()

        # Write a complete checkpoint with fingerprint "OLD_FP"
        from skills_extraction.checkpoint import checkpoint_path
        path = checkpoint_path(tmpdir, "test_run", "test_stage")
        with open(path, "w") as fh:
            write_checkpoint_header(fh, "test_run", "test_stage", 2, fingerprint="OLD_FP_1234567890")
            append_checkpoint_record(fh, {"idx": 0, "data": "old"})
            append_checkpoint_record(fh, {"idx": 1, "data": "old"})
            write_checkpoint_footer(fh, 2)

        run_count = [0]
        def _run_fn(start_idx=0):
            run_count[0] += 1
            return [{"idx": 0, "data": "new"}, {"idx": 1, "data": "new"}]

        # Same fingerprint — should load from checkpoint, not rerun
        result = _load_or_run_stage(
            "test_stage", ckpt_dir, "test_run", resume=True, total_expected=2,
            run_fn=_run_fn, fingerprint="OLD_FP_1234567890",
        )
        assert run_count[0] == 0, f"FAIL: should have loaded checkpoint, but ran {run_count[0]} times"
        assert result[0]["data"] == "old"

        # Different fingerprint — should invalidate and rerun
        result = _load_or_run_stage(
            "test_stage", ckpt_dir, "test_run", resume=True, total_expected=2,
            run_fn=_run_fn, fingerprint="NEW_FP_0987654321",
        )
        assert run_count[0] == 1, f"FAIL: should have rerun, but ran {run_count[0]} times"
        assert result[0]["data"] == "new"

        print("PASS: test_fingerprint_mismatch_invalidates")
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Test 18: Missing fingerprint invalidates older checkpoint
# ---------------------------------------------------------------------------

def test_missing_fingerprint_invalidates():
    """Older checkpoints without a fingerprint must be invalidated when the
    current stage requires one."""
    import tempfile, shutil
    from pathlib import Path
    from skills_extraction.pipeline import _load_or_run_stage
    from skills_extraction.checkpoint import (
        append_checkpoint_record, checkpoint_path, write_checkpoint_footer, write_checkpoint_header,
    )

    tmpdir = Path(tempfile.mkdtemp())
    try:
        ckpt_dir = tmpdir / "checkpoints"
        ckpt_dir.mkdir()
        path = checkpoint_path(tmpdir, "test_run", "test_stage")
        with open(path, "w") as fh:
            write_checkpoint_header(fh, "test_run", "test_stage", 1)
            append_checkpoint_record(fh, {"idx": 0, "data": "old"})
            write_checkpoint_footer(fh, 1)

        run_count = [0]
        def _run_fn(start_idx=0):
            run_count[0] += 1
            return [{"idx": 0, "data": "new"}]

        result = _load_or_run_stage(
            "test_stage", ckpt_dir, "test_run", resume=True, total_expected=1,
            run_fn=_run_fn, fingerprint="FP_PRESENT_123456",
        )
        assert run_count[0] == 1, "FAIL: missing fingerprint should invalidate old checkpoint"
        assert result[0]["data"] == "new"
        print("PASS: test_missing_fingerprint_invalidates")
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Test 19: Checkpoint total mismatch invalidates complete and partial resume
# ---------------------------------------------------------------------------

def test_checkpoint_total_mismatch_invalidates():
    """A checkpoint for a different total job count must rerun from scratch
    instead of loading or resuming from the wrong index."""
    import tempfile, shutil
    from pathlib import Path
    from skills_extraction.pipeline import _load_or_run_stage
    from skills_extraction.checkpoint import (
        append_checkpoint_record, checkpoint_path, write_checkpoint_footer, write_checkpoint_header,
    )

    tmpdir = Path(tempfile.mkdtemp())
    try:
        ckpt_dir = tmpdir / "checkpoints"
        ckpt_dir.mkdir()
        path = checkpoint_path(tmpdir, "test_run", "test_stage")

        # Complete checkpoint with wrong total
        with open(path, "w") as fh:
            write_checkpoint_header(fh, "test_run", "test_stage", 2, fingerprint="FP1")
            append_checkpoint_record(fh, {"idx": 0, "data": "old"})
            append_checkpoint_record(fh, {"idx": 1, "data": "old"})
            write_checkpoint_footer(fh, 2)

        run_args = []
        def _run_complete(start_idx=0):
            run_args.append(start_idx)
            return [{"idx": 0, "data": "new"}]

        result = _load_or_run_stage(
            "test_stage", ckpt_dir, "test_run", resume=True, total_expected=3,
            run_fn=_run_complete, fingerprint="FP1",
        )
        assert run_args == [0], f"FAIL: complete total mismatch should rerun from 0, got {run_args}"
        assert result[0]["data"] == "new"

        # Partial checkpoint with wrong total
        with open(path, "w") as fh:
            write_checkpoint_header(fh, "test_run", "test_stage", 2, fingerprint="FP1")
            append_checkpoint_record(fh, {"idx": 0, "data": "partial"})

        run_args.clear()
        def _run_partial(start_idx=0):
            run_args.append(start_idx)
            return [{"idx": 0, "data": "new"}]

        result = _load_or_run_stage(
            "test_stage", ckpt_dir, "test_run", resume=True, total_expected=3,
            run_fn=_run_partial, fingerprint="FP1",
        )
        assert run_args == [0], f"FAIL: partial total mismatch should rerun from 0, got {run_args}"
        assert result[0]["data"] == "new"
        print("PASS: test_checkpoint_total_mismatch_invalidates")
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Test 20: Fingerprint changes when model, prompt, or config changes
# ---------------------------------------------------------------------------

def test_fingerprint_changes_with_model():
    """Changing the model name, prompt text, or config input must produce a
    different fingerprint."""
    from skills_extraction.checkpoint import compute_stage_fingerprint

    fp1 = compute_stage_fingerprint("stage1", "qwen3:14b", "vllm", ["prompt text"])
    fp2 = compute_stage_fingerprint("stage1", "qwen3:32b", "vllm", ["prompt text"])
    fp3 = compute_stage_fingerprint("stage1", "qwen3:14b", "vllm", ["different prompt"])
    fp4 = compute_stage_fingerprint("stage2", "mistral-nemo:12b", "vllm", ["system", "user v1"])
    fp5 = compute_stage_fingerprint("stage2", "mistral-nemo:12b", "vllm", ["system", "user v2"])
    fp6 = compute_stage_fingerprint(
        "stage1", "qwen3:14b", "vllm", ["prompt text"],
        extra_settings={"disable_thinking": True},
    )
    fp7 = compute_stage_fingerprint(
        "stage1", "qwen3:14b", "vllm", ["prompt text"],
        extra_settings={"disable_thinking": False},
    )

    assert fp1 != fp2, "FAIL: different models should produce different fingerprints"
    assert fp1 != fp3, "FAIL: different prompts should produce different fingerprints"
    assert fp4 != fp5, "FAIL: different user templates should produce different fingerprints"
    assert fp6 != fp7, "FAIL: different config settings should produce different fingerprints"
    # Same inputs should be deterministic
    fp1b = compute_stage_fingerprint("stage1", "qwen3:14b", "vllm", ["prompt text"])
    assert fp1 == fp1b, "FAIL: same inputs should produce same fingerprint"

    print("PASS: test_fingerprint_changes_with_model")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_tests = [
        ("test_stage5_authority", test_stage5_authority),
        ("test_stage5_classifier_authority", test_stage5_classifier_authority),
        ("test_repeated_span_grounding", test_repeated_span_grounding),
        ("test_ungroundable_span_skipped", test_ungroundable_span_skipped),
        ("test_context_match_preferred", test_context_match_preferred),
        ("test_offsets_match_normalized", test_offsets_match_normalized),
        ("test_thread_local_sessions", test_thread_local_sessions),
        ("test_vllm_qwen_no_think_payload", test_vllm_qwen_no_think_payload),
        ("test_vllm_strip_leading_think_block", test_vllm_strip_leading_think_block),
        ("test_rolling_bounded", test_rolling_bounded),
        ("test_classifier_role_strings", test_classifier_role_strings),
        ("test_zero_mention_job_is_success", test_zero_mention_job_is_success),
        ("test_stage1_error_job_is_failed", test_stage1_error_job_is_failed),
        ("test_checkpoint_parsed_line_dedup", test_checkpoint_parsed_line_dedup),
        ("test_checkpoint_backward_compat", test_checkpoint_backward_compat),
        ("test_batched_flush", test_batched_flush),
        ("test_fingerprint_mismatch_invalidates", test_fingerprint_mismatch_invalidates),
        ("test_missing_fingerprint_invalidates", test_missing_fingerprint_invalidates),
        ("test_checkpoint_total_mismatch_invalidates", test_checkpoint_total_mismatch_invalidates),
        ("test_fingerprint_changes_with_model", test_fingerprint_changes_with_model),
    ]

    failures = 0
    for name, fn in all_tests:
        try:
            fn()
        except Exception as e:
            print(f"FAIL: {name}: {e}")
            failures += 1

    if failures:
        print(f"\n{failures} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"\nAll {len(all_tests)} tests passed.")
        sys.exit(0)
