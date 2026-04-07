import re
from collections import defaultdict
from datetime import datetime, timedelta

LOG_FILE = r'D:\PhD\skills-extraction\full_run_output\SkillsExtraction_pipeline_run_full_10k_20260327_185602.log'
pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - DEBUG - .* - LLM (\S+) \[(\w+)\] ([\d.]+)s \| prompt=(\d+) chars, resp=(\d+) chars, eval_tokens=(\d+), ([\d.]+) tok/s \| total_duration=([\d.]+)s'

records = []
with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as fh:
    for line in fh:
        m = re.search(pattern, line)
        if m:
            dt = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            records.append({
                'datetime': dt, 'model': m.group(2), 'role': m.group(3),
                'wall_time': float(m.group(4)), 'prompt_chars': int(m.group(5)),
                'resp_chars': int(m.group(6)), 'eval_tokens': int(m.group(7)),
                'tok_s': float(m.group(8)), 'total_duration': float(m.group(9)),
            })

print(f'Parsed {len(records)} LLM call records from log file.')
print()
if not records:
    print('No records found!')
    exit()

# ============================================================
# 1. OVERALL STATS
# ============================================================
total_calls = len(records)
total_wall = sum(r['wall_time'] for r in records)
total_eval_tokens = sum(r['eval_tokens'] for r in records)
total_prompt_chars = sum(r['prompt_chars'] for r in records)
total_resp_chars = sum(r['resp_chars'] for r in records)
avg_tok_s = sum(r['tok_s'] for r in records) / total_calls
first_dt = records[0]['datetime']
last_dt = records[-1]['datetime']
elapsed = (last_dt - first_dt).total_seconds()

sep = '=' * 80
dash = '-'
print(sep)
print('  OVERALL STATS')
print(sep)
print(f'  Total LLM calls:       {total_calls:>10,}')
print(f'  Total wall time:       {total_wall:>10,.1f} s  ({total_wall/3600:.1f} hours)')
print(f'  Total eval tokens:     {total_eval_tokens:>10,}')
print(f'  Total prompt chars:    {total_prompt_chars:>10,}')
print(f'  Total resp chars:      {total_resp_chars:>10,}')
print(f'  Avg tok/s:             {avg_tok_s:>10.1f}')
print(f'  Time span:             {first_dt} -> {last_dt}')
print(f'  Elapsed:               {elapsed/3600:.1f} hours')
if elapsed > 0:
    print(f'  Avg calls/hour:        {total_calls / (elapsed/3600):.1f}')
print()

# ============================================================
# 2. PHASE ANALYSIS
# ============================================================
print(sep)
print('  PHASE ANALYSIS')
print(sep)

for r in records:
    r['resp_eval_ratio'] = r['resp_chars'] / max(r['eval_tokens'], 1)

hourly_windows = defaultdict(list)
for r in records:
    key = r['datetime'].replace(minute=0, second=0)
    hourly_windows[key].append(r)

print()
print('  Hourly summary (to identify phases):')
print()
fmt_h = '  {:<22s} {:>6s} {:>10s} {:>9s} {:>14s} {:>12s}'
print(fmt_h.format('Hour', 'Calls', 'Avg tok/s', 'Avg wall', 'Avg resp/eval', 'Avg eval_tok'))
print('  ' + dash*22 + ' ' + dash*6 + ' ' + dash*10 + ' ' + dash*9 + ' ' + dash*14 + ' ' + dash*12)

sorted_hours = sorted(hourly_windows.keys())
for hour in sorted_hours:
    recs = hourly_windows[hour]
    n = len(recs)
    avg_ts = sum(r['tok_s'] for r in recs) / n
    avg_wt = sum(r['wall_time'] for r in recs) / n
    avg_ratio = sum(r['resp_eval_ratio'] for r in recs) / n
    avg_et = sum(r['eval_tokens'] for r in recs) / n
    h_str = hour.strftime('%Y-%m-%d %H:%M')
    print(f'  {h_str:>22s} {n:>6d} {avg_ts:>10.1f} {avg_wt:>9.1f}s {avg_ratio:>14.2f} {avg_et:>12.0f}')

dates_seen = sorted(set(r['datetime'].date() for r in records))
date_str = ', '.join(str(d) for d in dates_seen)
print(f'')
print(f'  Dates in log: {date_str}')

by_date = defaultdict(list)
for r in records:
    by_date[r['datetime'].date()].append(r)

print()
fmt_d = '  {:<14s} {:>7s} {:>10s} {:>9s} {:>10s} {:>9s}'
print(fmt_d.format('Date', 'Calls', 'Avg tok/s', 'Avg wall', 'Avg ratio', 'Avg eval'))
print('  ' + dash*14 + ' ' + dash*7 + ' ' + dash*10 + ' ' + dash*9 + ' ' + dash*10 + ' ' + dash*9)
for d in sorted(by_date.keys()):
    recs = by_date[d]
    n = len(recs)
    avg_ts = sum(r['tok_s'] for r in recs) / n
    avg_wt = sum(r['wall_time'] for r in recs) / n
    avg_ratio = sum(r['resp_eval_ratio'] for r in recs) / n
    avg_et = sum(r['eval_tokens'] for r in recs) / n
    print(f'  {str(d):<14s} {n:>7d} {avg_ts:>10.1f} {avg_wt:>9.1f}s {avg_ratio:>10.2f} {avg_et:>9.0f}')

# Auto-detect phases via rolling window transitions
window_size = 50
transitions = []
for i in range(window_size, len(records) - window_size):
    before_avg_ts = sum(r['tok_s'] for r in records[i-window_size:i]) / window_size
    after_avg_ts = sum(r['tok_s'] for r in records[i:i+window_size]) / window_size
    if after_avg_ts - before_avg_ts > 15 and before_avg_ts < 100:
        transitions.append((i, records[i]['datetime'], before_avg_ts, after_avg_ts))

if transitions:
    deduped = [transitions[0]]
    for t in transitions[1:]:
        if t[0] - deduped[-1][0] > 100:
            deduped.append(t)
    transitions = deduped
    print(f'')
    print(f'  Detected {len(transitions)} tok/s-based phase transition(s):')
    for idx, (pos, dt, before, after) in enumerate(transitions):
        print(f'    Transition {idx+1} at record #{pos} ({dt}): avg tok/s {before:.1f} -> {after:.1f}')

ratio_transitions = []
for i in range(window_size, len(records) - window_size):
    before_avg_r = sum(r['resp_eval_ratio'] for r in records[i-window_size:i]) / window_size
    after_avg_r = sum(r['resp_eval_ratio'] for r in records[i:i+window_size]) / window_size
    if before_avg_r - after_avg_r > 2.0 and before_avg_r > 5:
        ratio_transitions.append((i, records[i]['datetime'], before_avg_r, after_avg_r))

if ratio_transitions:
    deduped_r = [ratio_transitions[0]]
    for t in ratio_transitions[1:]:
        if t[0] - deduped_r[-1][0] > 100:
            deduped_r.append(t)
    ratio_transitions = deduped_r
    print(f'')
    print(f'  Ratio-based transitions (resp_chars/eval_tokens drop):')
    for idx, (pos, dt, before, after) in enumerate(ratio_transitions):
        print(f'    Transition {idx+1} at record #{pos} ({dt}): ratio {before:.2f} -> {after:.2f}')

all_transitions_sorted = sorted(
    [(t[0], t[1]) for t in transitions] + [(t[0], t[1]) for t in ratio_transitions],
    key=lambda x: x[0]
)
if all_transitions_sorted:
    final_transitions = [all_transitions_sorted[0]]
    for t in all_transitions_sorted[1:]:
        if t[0] - final_transitions[-1][0] > 200:
            final_transitions.append(t)
else:
    final_transitions = []

print(f'')
print(f'  Using {len(final_transitions)} transition point(s) to define phases.')

phase_records = {1: [], 2: [], 3: []}
if len(final_transitions) == 0:
    phase_records[1] = records
elif len(final_transitions) == 1:
    cutoff = final_transitions[0][0]
    phase_records[1] = records[:cutoff]
    phase_records[3] = records[cutoff:]
else:
    cutoff1 = final_transitions[0][0]
    cutoff2 = final_transitions[1][0]
    phase_records[1] = records[:cutoff1]
    phase_records[2] = records[cutoff1:cutoff2]
    phase_records[3] = records[cutoff2:]

phase_labels = {
    1: 'Phase 1 (thinking active)',
    2: 'Phase 2 (prompt /no_think)',
    3: 'Phase 3 (API think:false)',
}

print()
fmt_p = '  {:<32s} {:>7s} {:>8s} {:>10s} {:>10s} {:>9s} {:>10s}'
print(fmt_p.format('Phase', 'Calls', 'Wall(h)', 'Eval tok', 'Avg tok/s', 'Avg wall', 'Avg ratio'))
print('  ' + dash*32 + ' ' + dash*7 + ' ' + dash*8 + ' ' + dash*10 + ' ' + dash*10 + ' ' + dash*9 + ' ' + dash*10)

for phase_id in [1, 2, 3]:
    recs = phase_records[phase_id]
    if not recs:
        continue
    n = len(recs)
    tw = sum(r['wall_time'] for r in recs)
    te = sum(r['eval_tokens'] for r in recs)
    avg_ts = sum(r['tok_s'] for r in recs) / n
    avg_wt = sum(r['wall_time'] for r in recs) / n
    avg_ratio = sum(r['resp_eval_ratio'] for r in recs) / n
    label = phase_labels.get(phase_id, f'Phase {phase_id}')
    print(f'  {label:<32s} {n:>7,d} {tw/3600:>8.1f} {te:>10,} {avg_ts:>10.1f} {avg_wt:>9.1f}s {avg_ratio:>10.2f}')
    dt0 = recs[0]['datetime']
    dt1 = recs[-1]['datetime']
    print(f'    Time range: {dt0} -> {dt1}')

print()

# ============================================================
# 3. BY ROLE
# ============================================================
print(sep)
print('  BY ROLE')
print(sep)

by_role = defaultdict(list)
for r in records:
    by_role[r['role']].append(r)

print()
fmt_r = '  {:<16s} {:>8s} {:>9s} {:>9s} {:>10s} {:>11s} {:>9s} {:>12s}'
print(fmt_r.format('Role', 'Count', 'Avg wall', 'Avg eval', 'Avg tok/s', 'Avg prompt', 'Avg resp', 'Tot wall(h)'))
print('  ' + dash*16 + ' ' + dash*8 + ' ' + dash*9 + ' ' + dash*9 + ' ' + dash*10 + ' ' + dash*11 + ' ' + dash*9 + ' ' + dash*12)

for role in sorted(by_role.keys()):
    recs = by_role[role]
    n = len(recs)
    avg_wt = sum(r['wall_time'] for r in recs) / n
    avg_et = sum(r['eval_tokens'] for r in recs) / n
    avg_ts = sum(r['tok_s'] for r in recs) / n
    avg_pc = sum(r['prompt_chars'] for r in recs) / n
    avg_rc = sum(r['resp_chars'] for r in recs) / n
    tot_wt = sum(r['wall_time'] for r in recs)
    print(f'  {role:<16s} {n:>8,d} {avg_wt:>9.1f}s {avg_et:>9.0f} {avg_ts:>10.1f} {avg_pc:>11.0f} {avg_rc:>9.0f} {tot_wt/3600:>12.1f}')

print()

# ============================================================
# 4. HOURLY THROUGHPUT (last 24 hours)
# ============================================================
print(sep)
print('  HOURLY THROUGHPUT (last 24 hours)')
print(sep)

cutoff_24h = last_dt - timedelta(hours=24)
recent_records = [r for r in records if r['datetime'] >= cutoff_24h]

if recent_records:
    hourly_recent = defaultdict(list)
    for r in recent_records:
        key = r['datetime'].replace(minute=0, second=0)
        hourly_recent[key].append(r)

    print()
    fmt_hr = '  {:<22s} {:>7s} {:>10s} {:>9s} {:>12s}'
    print(fmt_hr.format('Hour', 'Calls', 'Avg tok/s', 'Avg wall', 'Eval tokens'))
    print('  ' + dash*22 + ' ' + dash*7 + ' ' + dash*10 + ' ' + dash*9 + ' ' + dash*12)

    for hour in sorted(hourly_recent.keys()):
        recs = hourly_recent[hour]
        n = len(recs)
        avg_ts = sum(r['tok_s'] for r in recs) / n
        avg_wt = sum(r['wall_time'] for r in recs) / n
        tot_et = sum(r['eval_tokens'] for r in recs)
        h_str = hour.strftime('%Y-%m-%d %H:%M')
        print(f'  {h_str:>22s} {n:>7d} {avg_ts:>10.1f} {avg_wt:>9.1f}s {tot_et:>12,}')

    total_recent = len(recent_records)
    hours_span = (recent_records[-1]['datetime'] - recent_records[0]['datetime']).total_seconds() / 3600
    if hours_span > 0:
        print(f'')
        print(f'  Effective throughput: {total_recent / hours_span:.1f} calls/hour over {hours_span:.1f} hours')
else:
    print('  No records in last 24 hours.')

print()

# ============================================================
# 5. CURRENT STATS (last 100 calls)
# ============================================================
print(sep)
print('  CURRENT STATS (last 100 calls)')
print(sep)

last100 = records[-100:]
n = len(last100)
avg_wt = sum(r['wall_time'] for r in last100) / n
avg_et = sum(r['eval_tokens'] for r in last100) / n
avg_ts = sum(r['tok_s'] for r in last100) / n
min_wt = min(r['wall_time'] for r in last100)
max_wt = max(r['wall_time'] for r in last100)
min_ts = min(r['tok_s'] for r in last100)
max_ts = max(r['tok_s'] for r in last100)
avg_pc = sum(r['prompt_chars'] for r in last100) / n
avg_rc = sum(r['resp_chars'] for r in last100) / n

dt0 = last100[0]['datetime']
dt1 = last100[-1]['datetime']
print()
print(f'  Time range:          {dt0} -> {dt1}')
print(f'  Avg wall time:       {avg_wt:>8.1f}s')
print(f'  Min wall time:       {min_wt:>8.1f}s')
print(f'  Max wall time:       {max_wt:>8.1f}s')
print(f'  Avg eval tokens:     {avg_et:>8.0f}')
print(f'  Avg tok/s:           {avg_ts:>8.1f}')
print(f'  Min tok/s:           {min_ts:>8.1f}')
print(f'  Max tok/s:           {max_ts:>8.1f}')
print(f'  Avg prompt chars:    {avg_pc:>8.0f}')
print(f'  Avg resp chars:      {avg_rc:>8.0f}')

print()
print('  By role (last 100):')
by_role_100 = defaultdict(list)
for r in last100:
    by_role_100[r['role']].append(r)

fmt_r100 = '  {:<16s} {:>6s} {:>9s} {:>10s} {:>9s}'
print(fmt_r100.format('Role', 'Count', 'Avg wall', 'Avg tok/s', 'Avg eval'))
print('  ' + dash*16 + ' ' + dash*6 + ' ' + dash*9 + ' ' + dash*10 + ' ' + dash*9)
for role in sorted(by_role_100.keys()):
    recs = by_role_100[role]
    rn = len(recs)
    rwt = sum(r['wall_time'] for r in recs) / rn
    rts = sum(r['tok_s'] for r in recs) / rn
    ret = sum(r['eval_tokens'] for r in recs) / rn
    print(f'  {role:<16s} {rn:>6d} {rwt:>9.1f}s {rts:>10.1f} {ret:>9.0f}')

print()
print(sep)
print('  ANALYSIS COMPLETE')
print(sep)
