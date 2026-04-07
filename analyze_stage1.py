#!/usr/bin/env python3
import json, re, sys
from collections import Counter, defaultdict
from statistics import median, mean

STAGE1 = 'D:/PhD/skills-extraction/full_run_output/checkpoints/full_10k_20260327_185602_stage1_extracted.jsonl'
LOGFILE = 'D:/PhD/skills-extraction/full_run_output/SkillsExtraction_pipeline_run_full_10k_20260327_185602.log'
STAGE2 = 'D:/PhD/skills-extraction/full_run_output/checkpoints/full_10k_20260327_185602_stage2_verified.jsonl'
SEP = chr(61) * 80
SUBSEP = chr(45) * 60

print(SEP)
print("STAGE 1 EXTRACTION ANALYSIS")
print(SEP)
print()

total_jobs = 0
total_mentions = 0
mentions_per_job = []
skill_counter = Counter()
all_confidences = []
header = None
footer = None

with open(STAGE1, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [WARN] Could not parse line {line_num}")
            continue
        if "job_index" not in rec:
            if line_num <= 2:
                header = rec
            else:
                footer = rec
            continue
        total_jobs += 1
        raw_mentions = rec.get("raw_mentions", [])
        n = len(raw_mentions)
        total_mentions += n
        mentions_per_job.append(n)
        for m in raw_mentions:
            nc = m.get("normalized_candidate", "").strip()
            if nc:
                skill_counter[nc.lower()] += 1
            conf = m.get("span_confidence")
            if conf is not None:
                try:
                    all_confidences.append(float(conf))
                except (ValueError, TypeError):
                    pass
        if total_jobs % 2000 == 0:
            print(f"  ... processed {total_jobs} jobs ({total_mentions} mentions)")

print(f"  Done. {total_jobs} jobs, {total_mentions} mentions.")
print()

print(SUBSEP)
print("TABLE 1: Stage 1 Extraction Summary")
print(SUBSEP)
avg_m = total_mentions / total_jobs if total_jobs else 0
med_m = median(mentions_per_job) if mentions_per_job else 0
min_m = min(mentions_per_job) if mentions_per_job else 0
max_m = max(mentions_per_job) if mentions_per_job else 0
zero_jobs = sum(1 for x in mentions_per_job if x == 0)
unique_skills = len(skill_counter)
for label, val in [
    ("Total jobs processed", f"{total_jobs:,}"),
    ("Total mentions extracted", f"{total_mentions:,}"),
    ("Avg mentions per job", f"{avg_m:.2f}"),
    ("Median mentions per job", f"{med_m:.1f}"),
    ("Min mentions per job", str(min_m)),
    ("Max mentions per job", str(max_m)),
    ("Jobs with 0 mentions", f"{zero_jobs:,}"),
    ("Unique normalized_candidate values", f"{unique_skills:,}"),
]:
    print(f"  {label:<40s} {val:>12s}")
print()
if header:
    print(f"  Header: {json.dumps(header, default=str)[:200]}")
if footer:
    print(f"  Footer: {json.dumps(footer, default=str)[:200]}")
print()

print(SUBSEP)
print("TABLE 2: Top 30 Most Common Skills")
print(SUBSEP)
hdr2 = f"  {'#':<4s} {'Skill':<50s} {'Count':>8s} {'%':>7s}"
print(hdr2)
print("  " + "-" * (len(hdr2) - 2))
for rank, (skill, count) in enumerate(skill_counter.most_common(30), 1):
    pct = 100.0 * count / total_mentions if total_mentions else 0
    print(f"  {rank:<4d} {skill:<50s} {count:>8,d} {pct:>5.2f}%")
print()

print(SUBSEP)
print("TABLE 3: Mention Distribution (mentions per job)")
print(SUBSEP)
buckets_def = [
    ("0", 0, 0),
    ("1-10", 1, 10),
    ("11-20", 11, 20),
    ("21-30", 21, 30),
    ("31-50", 31, 50),
    ("51-100", 51, 100),
    ("100+", 101, 99999),
]
print(f"  {'Bucket':<12s} {'Jobs':>8s} {'%':>7s}  Bar")
print("  " + "-" * 70)
for label, lo, hi in buckets_def:
    count = sum(1 for x in mentions_per_job if lo <= x <= hi)
    pct = 100.0 * count / total_jobs if total_jobs else 0
    bar = "#" * int(pct / 2)
    print(f"  {label:<12s} {count:>8,d} {pct:>6.2f}%  {bar}")
print()

print(SUBSEP)
print("TABLE 4: Confidence Distribution (span_confidence)")
print(SUBSEP)
if all_confidences:
    avg_c = mean(all_confidences)
    med_c = median(all_confidences)
    min_c = min(all_confidences)
    max_c = max(all_confidences)
    print(f"  {'Mentions with confidence':<40s} {len(all_confidences):>12,d}")
    print(f"  {'Average':<40s} {avg_c:>12.4f}")
    print(f"  {'Median':<40s} {med_c:>12.4f}")
    print(f"  {'Min':<40s} {min_c:>12.4f}")
    print(f"  {'Max':<40s} {max_c:>12.4f}")
    print()
    cb = defaultdict(int)
    for c in all_confidences:
        if c >= 1.0:
            cb[10] += 1
        else:
            cb[int(c * 10)] += 1
    print(f"  {'Range':<14s} {'Count':>8s} {'%':>7s}  Bar")
    print("  " + "-" * 70)
    for b in range(11):
        if b < 10:
            lbl = f"[{b/10:.1f}, {(b+1)/10:.1f})"
        else:
            lbl = "[1.0]"
        cnt = cb.get(b, 0)
        pct = 100.0 * cnt / len(all_confidences) if all_confidences else 0
        bar = "#" * max(1, int(pct/2)) if cnt > 0 else ""
        print(f"  {lbl:<14s} {cnt:>8,d} {pct:>6.2f}%  {bar}")
else:
    print("  No confidence values found.")
print()
# ============================================================
# LOG TIMING
# ============================================================
print(SEP)
print("LOG TIMING ANALYSIS")
print(SEP)
print()

llm_pat = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2},\d+ - DEBUG - .*?LLM .*? \[.*?\] ([\d.]+)s \| "
    r"prompt=(\d+) chars, resp=(\d+) chars, eval_tokens=(\d+), ([\d.]+) tok/s"
)
wall_times = []
eval_tokens_list = []
toks_list = []
prompt_chars_list = []
resp_chars_list = []
calls_by_date = defaultdict(list)
warning_count = 0
error_count = 0
retry_count = 0
timeout_count = 0

with open(LOGFILE, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if " - WARNING - " in line:
            warning_count += 1
            ll = line.lower()
            if "timeout" in ll:
                timeout_count += 1
            if "retry" in ll or "retrying" in ll:
                retry_count += 1
        if " - ERROR - " in line:
            error_count += 1
            ll = line.lower()
            if "timeout" in ll:
                timeout_count += 1
        m = llm_pat.match(line)
        if m:
            ds = m.group(1)
            wt = float(m.group(2))
            pc = int(m.group(3))
            rc = int(m.group(4))
            et = int(m.group(5))
            ts = float(m.group(6))
            wall_times.append(wt)
            eval_tokens_list.append(et)
            toks_list.append(ts)
            prompt_chars_list.append(pc)
            resp_chars_list.append(rc)
            calls_by_date[ds].append((wt, ts, et))

print(SUBSEP)
print("TABLE 5: Log Timing Analysis")
print(SUBSEP)
total_llm = len(wall_times)
total_wall = sum(wall_times)
total_et = sum(eval_tokens_list)
print(f"  {'Total LLM calls':<40s} {total_llm:>12,d}")
print(f"  {'Total wall-clock time (seconds)':<40s} {total_wall:>12,.1f}")
print(f"  {'Total wall-clock time (hours)':<40s} {total_wall/3600:>12,.2f}")
print(f"  {'Total eval tokens':<40s} {total_et:>12,d}")
print()
if wall_times:
    print(f"  {'Avg wall time per call (s)':<40s} {mean(wall_times):>12.2f}")
    print(f"  {'Median wall time per call (s)':<40s} {median(wall_times):>12.2f}")
    print(f"  {'Min wall time per call (s)':<40s} {min(wall_times):>12.2f}")
    print(f"  {'Max wall time per call (s)':<40s} {max(wall_times):>12.2f}")
    print()
    print(f"  {'Avg tok/s':<40s} {mean(toks_list):>12.1f}")
    print(f"  {'Median tok/s':<40s} {median(toks_list):>12.1f}")
    print(f"  {'Min tok/s':<40s} {min(toks_list):>12.1f}")
    print(f"  {'Max tok/s':<40s} {max(toks_list):>12.1f}")
    print()
    print(f"  {'Avg prompt chars':<40s} {mean(prompt_chars_list):>12.0f}")
    print(f"  {'Avg response chars':<40s} {mean(resp_chars_list):>12.0f}")
    print()
print(f"  {'WARNING lines in log':<40s} {warning_count:>12,d}")
print(f"  {'ERROR lines in log':<40s} {error_count:>12,d}")
print(f"  {'Timeout-related lines':<40s} {timeout_count:>12,d}")
print(f"  {'Retry-related lines':<40s} {retry_count:>12,d}")
print()
if calls_by_date:
    print("  Breakdown by date:")
    hdr = f"  {'Date':<14s} {'Calls':>8s} {'Tot hrs':>8s} {'Avg wt':>10s} {'Med wt':>10s} {'Avg t/s':>10s} {'Med t/s':>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for date in sorted(calls_by_date.keys()):
        entries = calls_by_date[date]
        wts = [e[0] for e in entries]
        tss = [e[1] for e in entries]
        n = len(entries)
        tot_h = sum(wts) / 3600
        print(f"  {date:<14s} {n:>8,d} {tot_h:>8.2f} {mean(wts):>10.2f} {median(wts):>10.2f} {mean(tss):>10.1f} {median(tss):>10.1f}")
print()
# ============================================================
# STAGE 2
# ============================================================
print(SEP)
print("STAGE 2 PROGRESS")
print(SEP)
print()

total_lines_s2 = 0
first_lines = []
last_lines = []
all_fields = set()
s2_records = 0
s2_header = None
s2_footer = None

with open(STAGE2, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        total_lines_s2 += 1
        line = line.strip()
        if not line:
            continue
        if i < 5:
            first_lines.append(line)
        last_lines.append(line)
        if len(last_lines) > 5:
            last_lines.pop(0)
        try:
            rec = json.loads(line)
            all_fields.update(rec.keys())
            if i == 0 and "job_index" not in rec:
                s2_header = rec
                continue
            if ("completed" in rec and "job_index" not in rec) or ("total_records" in rec and "job_index" not in rec):
                s2_footer = rec
                continue
            s2_records += 1
        except json.JSONDecodeError:
            pass

print(f"  Total lines in file:   {total_lines_s2:,d}")
print(f"  Data records:          {s2_records:,d}")
print(f"  Fields found:          {sorted(all_fields)}")
print()
if s2_header:
    print(f"  Header: {json.dumps(s2_header, default=str)[:300]}")
if s2_footer:
    print(f"  Footer: {json.dumps(s2_footer, default=str)[:300]}")
print()
print("  First 3 lines (truncated to 250 chars):")
for idx, ln in enumerate(first_lines[:3]):
    print(f"    [{idx}] {ln[:250]}")
print()
print("  Last 3 lines (truncated to 250 chars):")
for idx, ln in enumerate(last_lines[-3:]):
    print(f"    [{idx}] {ln[:250]}")
print()

s2_sample_fields = Counter()
s2_verdict_counter = Counter()
s2_count = 0

with open(STAGE2, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if i == 0 and "job_index" not in rec:
            continue
        if "completed" in rec and "job_index" not in rec:
            continue
        if "total_records" in rec and "job_index" not in rec:
            continue
        s2_count += 1
        for k in rec.keys():
            s2_sample_fields[k] += 1
        for vf in ["verdict", "verification_result", "verified", "is_valid", "keep", "status", "label", "is_skill"]:
            if vf in rec:
                s2_verdict_counter[f"{vf}={rec[vf]}"] += 1

print(f"  Stage 2 record count: {s2_count:,d}")
print()
print("  Field frequency across all records:")
for field, cnt in s2_sample_fields.most_common():
    pct = 100 * cnt / s2_count if s2_count else 0
    print(f"    {field:<40s} {cnt:>8,d} ({pct:.1f}%)")

if s2_verdict_counter:
    print()
    print("  Verdict/status distribution:")
    for lbl, cnt in s2_verdict_counter.most_common(20):
        pct = 100 * cnt / s2_count if s2_count else 0
        print(f"    {lbl:<40s} {cnt:>8,d} ({pct:.1f}%)")

print()
print(SEP)
print("ANALYSIS COMPLETE")
print(SEP)
