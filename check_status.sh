#!/bin/bash
# Check status of all pipeline runs (or a single run) in the checkpoints directory
# Usage: ./check_status.sh [RUN_ID]
#   ./check_status.sh                  — show all runs
#   ./check_status.sh 20260402_214027  — show only that run
set -euo pipefail

BASE_DIR="$HOME/skills_extraction"
CHECKPOINT_DIR="$BASE_DIR/out/checkpoints"
OUT_DIR="$BASE_DIR/out"
FILTER_RUN="${1:-}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

STAGES=("stage0_preprocessed" "stage1_extracted" "stage2_verified" "stage3_requirement" "stage4_hardsoft")

# Collect unique run IDs
if [ -n "$FILTER_RUN" ]; then
    # Verify at least one checkpoint exists for this run
    if ! ls "$CHECKPOINT_DIR"/${FILTER_RUN}_stage*.jsonl &>/dev/null; then
        echo "No checkpoints found for run $FILTER_RUN"
        exit 1
    fi
    run_ids="$FILTER_RUN"
else
    run_ids=$(ls "$CHECKPOINT_DIR"/*.jsonl 2>/dev/null | xargs -n1 basename | sed 's/_stage.*//' | sort -u)
fi

if [ -z "$run_ids" ]; then
    echo "No checkpoint files found."
    exit 0
fi

# Check if a pipeline process is running
check_active_process() {
    local rid="$1"
    local pids
    pids=$(pgrep -f "run-id.*$rid" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  PROCESS: RUNNING (pid: $pids)"
    else
        echo "  PROCESS: not running"
    fi
}

# Read models from a checkpoint header
show_models() {
    local rid="$1"
    # Find any checkpoint to read config from
    local any_file
    any_file=$(ls "$CHECKPOINT_DIR"/${rid}_stage*.jsonl 2>/dev/null | head -1)
    if [ -z "$any_file" ]; then return; fi
    python3 -c "
import json, sys
with open('$any_file') as f:
    header = json.loads(f.readline())
cfg = header.get('config', {})
if not cfg:
    sys.exit(0)
ext = cfg.get('extractor_model', '?')
ver = cfg.get('verifier_model', '?')
req = cfg.get('requirement_model', '?')
hs  = cfg.get('hardsoft_model', '?')
be  = cfg.get('backend', '?')
ep  = cfg.get('vllm_num_endpoints', '?')
print(f'  MODELS: extractor={ext}  verifier={ver}  req={req}  hardsoft={hs}')
print(f'  BACKEND: {be}  endpoints={ep}')
" 2>/dev/null || true
}

# Count errors in a checkpoint file
count_errors() {
    local f="$1"
    grep -c '"status": "error"' "$f" 2>/dev/null || echo "0"
}

# Get expected record count from previous completed stage
get_expected() {
    local rid="$1"
    local stage_idx="$2"
    if [ "$stage_idx" -le 1 ]; then
        # Stages 0-1 are per-job; get total_jobs from header
        local any_file
        any_file=$(ls "$CHECKPOINT_DIR"/${rid}_stage*.jsonl 2>/dev/null | head -1)
        if [ -n "$any_file" ]; then
            python3 -c "
import json
with open('$any_file') as f:
    header = json.loads(f.readline())
print(header.get('total_jobs', '?'))
" 2>/dev/null || echo "?"
        else
            echo "?"
        fi
    else
        # Stages 2-4 use mention count from stage 1
        local s1="$CHECKPOINT_DIR/${rid}_stage1_extracted.jsonl"
        if [ -f "$s1" ]; then
            local last
            last=$(tail -1 "$s1")
            echo "$last" | python3 -c "
import json, sys
obj = json.load(sys.stdin)
if obj.get('_complete'):
    print(obj.get('record_count', '?'))
else:
    print('?')
" 2>/dev/null || echo "?"
        else
            echo "?"
        fi
    fi
}

# Show run summary if it exists
show_summary() {
    local rid="$1"
    local summary
    summary=$(find "$OUT_DIR" -maxdepth 1 -name "*summary*${rid}*" -type f 2>/dev/null | head -1)
    if [ -z "$summary" ]; then return; fi
    python3 -c "
import json
with open('$summary') as f:
    d = json.load(f)
wall = d.get('wall_clock_sec', 0)
hrs = int(wall // 3600)
mins = int((wall % 3600) // 60)
jobs = d.get('jobs', {})
mentions = d.get('mentions_total', '?')
ext = d.get('llm_timing', {}).get('extractor', {})
ver = d.get('llm_timing', {}).get('verifier', {})
print(f'  SUMMARY: {hrs}h {mins}m wall clock | {mentions} mentions | jobs: {jobs.get(\"success\",\"?\")}/{jobs.get(\"total\",\"?\")} success')
if ext.get('total_calls', 0) > 0:
    print(f'  LLM CALLS: extractor={ext[\"total_calls\"]} ({ext[\"sec_per_call_avg\"]}s/call)  verifier={ver.get(\"total_calls\",0)} ({ver.get(\"sec_per_call_avg\",0)}s/call)')
st = d.get('stage_timing', {})
if st:
    parts = []
    for s, info in st.items():
        w = info.get('wall_sec', 0)
        sh = int(w // 3600)
        sm = int((w % 3600) // 60)
        parts.append(f'{s}={sh}h{sm}m')
    print(f'  STAGE TIME: {\"  \".join(parts)}')
" 2>/dev/null || true
}

# Show last few log lines
show_log_tail() {
    local rid="$1"
    local logfile
    logfile=$(find "$OUT_DIR" -maxdepth 1 -name "*pipeline_run*${rid}*" -name "*.log" -type f 2>/dev/null | head -1)
    if [ -z "$logfile" ]; then
        # Also check base dir
        logfile=$(find "$BASE_DIR" -maxdepth 1 -name "*${rid}*.log" -type f 2>/dev/null | head -1)
    fi
    if [ -z "$logfile" ]; then return; fi
    echo "  LOG: $(basename "$logfile")"
    echo "  --- last 5 lines ---"
    tail -5 "$logfile" | sed 's/^/  | /'
}

# Main loop
for rid in $run_ids; do
    echo "========================================"
    echo "  Run: $rid"
    echo "========================================"

    check_active_process "$rid"
    show_models "$rid"
    echo ""

    prev_count=""
    for i in "${!STAGES[@]}"; do
        stage="${STAGES[$i]}"
        f="$CHECKPOINT_DIR/${rid}_${stage}.jsonl"

        if [ ! -f "$f" ]; then
            echo "  $stage: -- missing --"
            continue
        fi

        lines=$(wc -l < "$f")
        last=$(tail -1 "$f")
        errors=$(count_errors "$f")

        if echo "$last" | python3 -c "import sys,json; exit(0 if json.load(sys.stdin).get('_complete') else 1)" 2>/dev/null; then
            count=$(echo "$last" | python3 -c "import sys,json; print(json.load(sys.stdin).get('record_count','?'))")
            started=$(head -1 "$f" | python3 -c "import sys,json; print(json.load(sys.stdin).get('started_at','?')[:19])")
            finished=$(echo "$last" | python3 -c "import sys,json; print(json.load(sys.stdin).get('completed_at','?')[:19])")
            err_str=""
            if [ "$errors" -gt 0 ] 2>/dev/null; then err_str="  errors=$errors"; fi
            echo "  $stage: COMPLETE  records=$count  started=$started  finished=$finished$err_str"
            prev_count="$count"
        else
            data_lines=$((lines - 1))  # subtract header
            started=$(head -1 "$f" | python3 -c "import sys,json; print(json.load(sys.stdin).get('started_at','?')[:19])" 2>/dev/null || echo "?")
            expected=$(get_expected "$rid" "$i")
            pct=""
            if [ "$expected" != "?" ] && [ "$expected" -gt 0 ] 2>/dev/null; then
                pct=" ($(( data_lines * 100 / expected ))%)"
            fi
            err_str=""
            if [ "$errors" -gt 0 ] 2>/dev/null; then err_str="  errors=$errors"; fi
            echo "  $stage: INCOMPLETE  records=$data_lines/$expected$pct  started=$started$err_str"
        fi
    done

    echo ""
    show_summary "$rid"
    show_log_tail "$rid"
    echo ""
done
