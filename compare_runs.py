#!/usr/bin/env python3
import json, statistics
from collections import Counter

print('=' * 80)
print('LOADING DATA...')
print('=' * 80)

ollama_data = {}
ollama_skills = []

with open('D:/PhD/skills-extraction/full_run_output/checkpoints/full_10k_20260327_185602_stage1_extracted.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('_meta') or obj.get('_complete') or '_footer' in obj or 'job_key' not in obj:
            continue
        jk = obj['job_key']
        mentions = obj.get('raw_mentions', [])
        ollama_data[jk] = len(mentions)
        for m in mentions:
            nc = m.get('normalized_candidate', m.get('skill_span', '')).lower().strip()
            if nc:
                ollama_skills.append(nc)

print(f'Ollama: {len(ollama_data)} jobs, {sum(ollama_data.values())} mentions')

vllm_data = {}
vllm_skills = []

with open('D:/PhD/titan3/SkillsExtraction_augmented_run_20260402_214027.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for job in data:
    jk = job['extraction_metadata']['job_key']
    mentions = job.get('skill_mentions', [])
    vllm_data[jk] = len(mentions)
    for m in mentions:
        nc = m.get('normalized_candidate', m.get('skill_span', '')).lower().strip()
        if nc:
            vllm_skills.append(nc)
del data

print(f'vLLM:   {len(vllm_data)} jobs, {sum(vllm_data.values())} mentions')

common_keys = set(ollama_data.keys()) & set(vllm_data.keys())
only_o = set(ollama_data.keys()) - set(vllm_data.keys())
only_v = set(vllm_data.keys()) - set(ollama_data.keys())
print(f'Common: {len(common_keys)}, Only-Ollama: {len(only_o)}, Only-vLLM: {len(only_v)}')


# ===== 1. HIGH-LEVEL COMPARISON =====
print()
print('=' * 80)
print('1. HIGH-LEVEL COMPARISON TABLE')
print('=' * 80)

def calc_stats(dd):
    counts = sorted(dd.values())
    n = len(counts)
    return (sum(counts), statistics.mean(counts), statistics.median(counts),
            min(counts), max(counts), sum(1 for c in counts if c == 0),
            statistics.stdev(counts), counts[n//4], counts[3*n//4])

ot, oa, om, omn, omx, oz, osd, o25, o75 = calc_stats(ollama_data)
vt, va, vm, vmn, vmx, vz, vsd, v25, v75 = calc_stats(vllm_data)
ou = len(set(ollama_skills))
vu = len(set(vllm_skills))

hdr = f'{"Metric":<30} {"Ollama (qwen3:14b)":>22} {"vLLM (Qwen3-14B)":>22} {"Difference":>15}'
print(hdr)
print('-' * len(hdr))
rows = [
    ('Total jobs',           len(ollama_data),  len(vllm_data)),
    ('Total mentions',       ot, vt),
    ('Avg mentions/job',     round(oa,2), round(va,2)),
    ('Stdev mentions/job',   round(osd,2), round(vsd,2)),
    ('Median mentions/job',  om, vm),
    ('25th percentile',      o25, v25),
    ('75th percentile',      o75, v75),
    ('Min mentions/job',     omn, vmn),
    ('Max mentions/job',     omx, vmx),
    ('Jobs with 0 mentions', oz, vz),
    ('Unique skill strings', ou, vu),
]
for label, oval, vval in rows:
    diff = vval - oval
    if isinstance(oval, float):
        print(f'{label:<30} {oval:>22} {vval:>22} {diff:>+15.2f}')
    else:
        print(f'{label:<30} {oval:>22} {vval:>22} {diff:>+15}')


# ===== 2. PER-JOB DIFFERENCE =====
print()
print('=' * 80)
print('2. PER-JOB DIFFERENCE ANALYSIS')
print('=' * 80)

exact = vmore = omore = 0
diffs = []
for jk in common_keys:
    oc, vc = ollama_data[jk], vllm_data[jk]
    d = vc - oc
    diffs.append((jk, oc, vc, abs(d), d))
    if d == 0: exact += 1
    elif d > 0: vmore += 1
    else: omore += 1

n = len(common_keys)
print(f'Among {n} common jobs:')
print(f'  Exact count match:        {exact:>6} ({100*exact/n:.1f}%)')
print(f'  vLLM has more mentions:   {vmore:>6} ({100*vmore/n:.1f}%)')
print(f'  Ollama has more mentions: {omore:>6} ({100*omore/n:.1f}%)')

sd = [d[4] for d in diffs]
print(f'  Signed diff (vLLM - Ollama) stats:')
print(f'    Mean:   {statistics.mean(sd):+.2f}')
print(f'    Median: {statistics.median(sd):+.1f}')
print(f'    Stdev:  {statistics.stdev(sd):.2f}')

buckets = Counter()
for d in sd:
    if d == 0: buckets['     0'] += 1
    elif 1<=d<=5: buckets['  +1 to  +5'] += 1
    elif 6<=d<=10: buckets['  +6 to +10'] += 1
    elif 11<=d<=20: buckets[' +11 to +20'] += 1
    elif 21<=d<=50: buckets[' +21 to +50'] += 1
    elif d>50: buckets[' +51 or more'] += 1
    elif -5<=d<=-1: buckets['  -5 to  -1'] += 1
    elif -10<=d<=-6: buckets[' -10 to  -6'] += 1
    elif -20<=d<=-11: buckets[' -20 to -11'] += 1
    elif -50<=d<=-21: buckets[' -50 to -21'] += 1
    else: buckets['-51 or less'] += 1

print(f'  Distribution of per-job difference (vLLM - Ollama):')
for b in ['-51 or less',' -50 to -21',' -20 to -11',' -10 to  -6','  -5 to  -1',
          '     0','  +1 to  +5','  +6 to +10',' +11 to +20',' +21 to +50',' +51 or more']:
    c = buckets.get(b, 0)
    bar = '#' * (c // 20)
    print(f'    {b}: {c:>5}  {bar}')

diffs_sorted = sorted(diffs, key=lambda x: x[3], reverse=True)
print(f'  Top 20 jobs with largest absolute difference:')
print(f'  {"Job Key":<12} {"Ollama":>8} {"vLLM":>8} {"Diff(v-o)":>12}')
print(f'  {"-"*12} {"-"*8} {"-"*8} {"-"*12}')
for jk, oc, vc, ad, sd2 in diffs_sorted[:20]:
    print(f'  {jk:<12} {oc:>8} {vc:>8} {sd2:>+12}')


# ===== 3. ZERO-MENTION ANALYSIS =====
print()
print('=' * 80)
print('3. ZERO-MENTION ANALYSIS')
print('=' * 80)

z_o_nz_v = [(jk, ollama_data[jk], vllm_data[jk]) for jk in common_keys if ollama_data[jk]==0 and vllm_data[jk]>0]
z_v_nz_o = [(jk, ollama_data[jk], vllm_data[jk]) for jk in common_keys if vllm_data[jk]==0 and ollama_data[jk]>0]
z_both = [(jk,) for jk in common_keys if ollama_data[jk]==0 and vllm_data[jk]==0]

print(f'  Zero in BOTH runs:                     {len(z_both):>5}')
print(f'  Zero in Ollama but >0 in vLLM:         {len(z_o_nz_v):>5}')
print(f'  Zero in vLLM but >0 in Ollama:         {len(z_v_nz_o):>5}')

if z_o_nz_v:
    print(f'  Jobs with 0 Ollama but >0 vLLM ({len(z_o_nz_v)} jobs, top 30):')
    print(f'  {"Job Key":<12} {"Ollama":>8} {"vLLM":>8}')
    print(f'  {"-"*12} {"-"*8} {"-"*8}')
    for jk, oc, vc in sorted(z_o_nz_v, key=lambda x:-x[2])[:30]:
        print(f'  {jk:<12} {oc:>8} {vc:>8}')

if z_v_nz_o:
    print(f'  Jobs with 0 vLLM but >0 Ollama ({len(z_v_nz_o)} jobs, top 30):')
    print(f'  {"Job Key":<12} {"Ollama":>8} {"vLLM":>8}')
    print(f'  {"-"*12} {"-"*8} {"-"*8}')
    for jk, oc, vc in sorted(z_v_nz_o, key=lambda x:-x[1])[:30]:
        print(f'  {jk:<12} {oc:>8} {vc:>8}')


# ===== 4. SKILL OVERLAP =====
print()
print('=' * 80)
print('4. SKILL OVERLAP ANALYSIS (Top 100 skills by frequency)')
print('=' * 80)

oc = Counter(ollama_skills)
vc = Counter(vllm_skills)

o100 = set(s for s,_ in oc.most_common(100))
v100 = set(s for s,_ in vc.most_common(100))
both = o100 & v100
only_o100 = o100 - v100
only_v100 = v100 - o100

print(f'  In BOTH top-100:         {len(both)}')
print(f'  Only in Ollama top-100:  {len(only_o100)}')
print(f'  Only in vLLM top-100:    {len(only_v100)}')

print(f'  Top 30 shared skills (sorted by combined freq):')
print(f'  {"Rk":>4} {"Skill":<40} {"Ollama":>8} {"vLLM":>8} {"Ratio":>8}')
print(f'  {"-"*4} {"-"*40} {"-"*8} {"-"*8} {"-"*8}')
shared = sorted(both, key=lambda s: -(oc[s]+vc[s]))
for r, s in enumerate(shared[:30], 1):
    ratio = vc[s]/oc[s] if oc[s]>0 else 999
    print(f'  {r:>4} {s:<40} {oc[s]:>8} {vc[s]:>8} {ratio:>8.2f}')

if only_o100:
    print(f'  Skills ONLY in Ollama top-100:')
    print(f'  {"Skill":<40} {"O-freq":>8} {"V-freq":>8} {"V-rank":>8}')
    print(f'  {"-"*40} {"-"*8} {"-"*8} {"-"*8}')
    vrm = {s:r for r,(s,_) in enumerate(vc.most_common(),1)}
    for s in sorted(only_o100, key=lambda x:-oc[x]):
        print(f'  {s:<40} {oc[s]:>8} {vc.get(s,0):>8} {vrm.get(s,"N/A"):>8}')

if only_v100:
    print(f'  Skills ONLY in vLLM top-100:')
    print(f'  {"Skill":<40} {"V-freq":>8} {"O-freq":>8} {"O-rank":>8}')
    print(f'  {"-"*40} {"-"*8} {"-"*8} {"-"*8}')
    orm = {s:r for r,(s,_) in enumerate(oc.most_common(),1)}
    for s in sorted(only_v100, key=lambda x:-vc[x]):
        print(f'  {s:<40} {vc[s]:>8} {oc.get(s,0):>8} {orm.get(s,"N/A"):>8}')


# ===== 5. TIMEOUT CORRELATION =====
print()
print('=' * 80)
print('5. vLLM TIMEOUT / FAILED BATCH ANALYSIS')
print('=' * 80)

failed_batches = [655, 798, 1077, 1612, 2028, 2710, 3295, 7290, 8053, 8364, 8592, 9184, 18698]
print(f'  Failed batch numbers (13 total): {failed_batches}')

try:
    with open('D:/PhD/titan3/SkillsExtraction_run_summary_20260402_214027.json', 'r', encoding='utf-8') as f:
        summary = json.load(f)
    print(f'  vLLM Run Summary:')
    for k in ['total_jobs','total_mentions','total_unique_skills','total_zero_mention_jobs','pipeline_version']:
        if k in summary:
            print(f'    {k}: {summary[k]}')
except Exception as e:
    print(f'  Could not load summary: {e}')

print(f'  --- Potential Timeout Impact ---')
vdef = [(jk,ollama_data[jk],vllm_data[jk],ollama_data[jk]-vllm_data[jk])
        for jk in common_keys if ollama_data[jk]>vllm_data[jk] and (ollama_data[jk]-vllm_data[jk])>=5]
vdef.sort(key=lambda x:-x[3])

odef = [(jk,ollama_data[jk],vllm_data[jk],vllm_data[jk]-ollama_data[jk])
        for jk in common_keys if vllm_data[jk]>ollama_data[jk] and (vllm_data[jk]-ollama_data[jk])>=5]
odef.sort(key=lambda x:-x[3])

print(f'  Jobs where Ollama has >=5 more: {len(vdef)} (total deficit: {sum(x[3] for x in vdef)})')
print(f'  Jobs where vLLM has >=5 more:   {len(odef)} (total surplus: {sum(x[3] for x in odef)})')

if vdef:
    print(f'  Top 20 jobs with most missing vLLM mentions:')
    print(f'  {"Job Key":<12} {"Ollama":>8} {"vLLM":>8} {"Deficit":>10}')
    print(f'  {"-"*12} {"-"*8} {"-"*8} {"-"*10}')
    for jk,occ,vcc,d in vdef[:20]:
        print(f'  {jk:<12} {occ:>8} {vcc:>8} {d:>10}')

td_v = sum(max(0,ollama_data[jk]-vllm_data[jk]) for jk in common_keys)
td_o = sum(max(0,vllm_data[jk]-ollama_data[jk]) for jk in common_keys)
net = sum(vllm_data.values()) - sum(ollama_data.values())
osum = sum(ollama_data.values())

z_v_nz_o2 = sum(1 for jk in common_keys if vllm_data[jk]==0 and ollama_data[jk]>0)
z_o_nz_v2 = sum(1 for jk in common_keys if ollama_data[jk]==0 and vllm_data[jk]>0)

print(f'  Summary:')
print(f'    13 failed batches in vLLM run')
print(f'    {z_v_nz_o2} jobs: 0 vLLM but >0 Ollama')
print(f'    {z_o_nz_v2} jobs: 0 Ollama but >0 vLLM')
print(f'    Total deficit (Ollama>vLLM):  {td_v}')
print(f'    Total surplus (vLLM>Ollama):  {td_o}')
print(f'    Net (vLLM - Ollama):          {net:+d}')
print(f'    Pct increase vLLM over Ollama:{100*net/osum:+.1f}%')
print(f'  Note: Ollama failures used fallback (OpenRouter), so likely still produced results.')

print()
print('=' * 80)
print('ANALYSIS COMPLETE')
print('=' * 80)
