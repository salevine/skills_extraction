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
    from skills_extraction.checkpoint import serialize_mention

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
    jobs = [{"JobTitle": "Test", "Description": pl.text}]
    stage1_data = [{
        "job_index": 0,
        "job_key": "testjob",
        "extractor_model_used": "test-model",
        "raw_mentions": [serialize_mention(mention)],
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
    from skills_extraction.checkpoint import serialize_mention

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

    jobs = [{"JobTitle": "Test2", "Description": pl.text}]
    stage1_data = [{
        "job_index": 0, "job_key": "testjob2",
        "extractor_model_used": "test-model",
        "raw_mentions": [serialize_mention(mention)],
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
    from skills_extraction.checkpoint import serialize_mention
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

    jobs = [{"JobTitle": "Offset Test", "Description": raw}]
    stage1_data = [{
        "job_index": 0, "job_key": "offset_test",
        "extractor_model_used": "test-model",
        "raw_mentions": [serialize_mention(mention)],
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
# Test 8: _run_rolling bounds in-flight futures
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
        ("test_rolling_bounded", test_rolling_bounded),
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
