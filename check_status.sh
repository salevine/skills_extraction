#!/bin/bash
# Check status of all pipeline runs in the checkpoints directory
set -euo pipefail

CHECKPOINT_DIR="${1:-$HOME/skills_extraction/out/checkpoints}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

echo "Checkpoint dir: $CHECKPOINT_DIR"
echo ""

# Collect unique run IDs
run_ids=$(ls "$CHECKPOINT_DIR"/*.jsonl 2>/dev/null | xargs -n1 basename | sed 's/_stage.*//' | sort -u)

if [ -z "$run_ids" ]; then
    echo "No checkpoint files found."
    exit 0
fi

STAGES=("stage0_preprocessed" "stage1_extracted" "stage2_verified" "stage3_requirement" "stage4_hardsoft")

for rid in $run_ids; do
    echo "=== Run: $rid ==="
    for stage in "${STAGES[@]}"; do
        f="$CHECKPOINT_DIR/${rid}_${stage}.jsonl"
        if [ ! -f "$f" ]; then
            echo "  $stage: missing"
            continue
        fi
        lines=$(wc -l < "$f")
        last=$(tail -1 "$f")
        if echo "$last" | python3 -c "import sys,json; exit(0 if json.load(sys.stdin).get('_complete') else 1)" 2>/dev/null; then
            count=$(echo "$last" | python3 -c "import sys,json; print(json.load(sys.stdin).get('record_count','?'))")
            started=$(head -1 "$f" | python3 -c "import sys,json; print(json.load(sys.stdin).get('started_at','?')[:19])")
            finished=$(echo "$last" | python3 -c "import sys,json; print(json.load(sys.stdin).get('completed_at','?')[:19])")
            echo "  $stage: COMPLETE  records=$count  started=$started  finished=$finished"
        else
            data_lines=$((lines - 1))  # subtract header
            started=$(head -1 "$f" | python3 -c "import sys,json; print(json.load(sys.stdin).get('started_at','?')[:19])" 2>/dev/null || echo "?")
            echo "  $stage: INCOMPLETE  lines=$data_lines  started=$started"
        fi
    done
    echo ""
done
