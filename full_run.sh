#!/bin/bash
# Full pipeline run: Qwen extraction (stages 0-1), swap to Mistral (stages 2-4)
# Usage (on titan3):
#   nohup ./full_run.sh > run.log 2>&1 &
#   tail -f run.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Full pipeline run — $(date) ==="

# Pick a single run id up front and carry it through both phases.
# Allow override via env for scripted restarts if needed.
RUN_ID="${RUN_ID:-full_$(date +%Y%m%d_%H%M%S)}"
echo "=== Run ID: $RUN_ID ==="

# ------------------------------------------------------------------
# Stage 1: Extract with Qwen (assumes Qwen vLLM servers already running)
# Start them first if needed:  ~/startQwen 8
# ------------------------------------------------------------------
echo ""
echo "=== Checking Qwen vLLM servers ==="
for port in 8000 8001 8002 8003 8004 8005 8006 8007; do
    if ! curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
        echo "ERROR: vLLM not responding on port $port. Start Qwen first: ~/startQwen 8"
        exit 1
    fi
done
MODEL=$(curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "Model on port 8000: $MODEL"

echo ""
echo "=== Stages 0-1: Extraction (Qwen/Qwen3-14B) ==="
conda run --no-capture-output -n skills python -m skills_extraction \
  --input ../jobs/SampleJobs.json \
  --output-dir ./out \
  --run-id "$RUN_ID" \
  --vllm \
  --vllm-host localhost \
  --vllm-base-port 8000 \
  --vllm-num-endpoints 8 \
  --extractor-model "Qwen/Qwen3-14B" \
  --no-verifier \
  --no-requirement-classifier \
  --no-hardsoft-classifier

echo ""
echo "=== Stage 1 complete. Run ID: $RUN_ID ==="
echo "=== Swapping to Mistral-Nemo ==="

# ------------------------------------------------------------------
# Swap models
# ------------------------------------------------------------------
"$SCRIPT_DIR/stopModel"
echo "Waiting for Qwen to shut down..."
sleep 10

"$SCRIPT_DIR/startMistral" 8
echo "Waiting for Mistral-Nemo to load..."
# Poll until ready
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "Mistral-Nemo ready after ${i}0 seconds"
        break
    fi
    sleep 10
done

# Verify all 8 endpoints
for port in 8000 8001 8002 8003 8004 8005 8006 8007; do
    if ! curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
        echo "ERROR: Mistral not responding on port $port"
        exit 1
    fi
done

# ------------------------------------------------------------------
# Stages 2-4: Verify + classify with Mistral-Nemo
# ------------------------------------------------------------------
echo ""
echo "=== Stages 2-4: Verify + Classify (Mistral-Nemo) — resuming run $RUN_ID ==="
conda run --no-capture-output -n skills python -m skills_extraction \
  --input ../jobs/SampleJobs.json \
  --output-dir ./out \
  --vllm \
  --vllm-host localhost \
  --run-id "$RUN_ID" \
  --vllm-base-port 8000 \
  --vllm-num-endpoints 8 \
  --extractor-model "Qwen/Qwen3-14B" \
  --verifier-model "mistralai/Mistral-Nemo-Instruct-2407" \
  --requirement-model "mistralai/Mistral-Nemo-Instruct-2407" \
  --hardsoft-model "mistralai/Mistral-Nemo-Instruct-2407"

echo ""
echo "=== Pipeline complete — $(date) ==="
echo "=== Run ID: $RUN_ID ==="

# Show status
"$SCRIPT_DIR/check_status.sh" "$RUN_ID"
