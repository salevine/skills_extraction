#!/bin/bash
set -euo pipefail

cd ~/skills_extraction

# Set this to the run_id from your interrupted run
# Find it with: ls ~/skills_extraction/out/checkpoints/
RUN_ID="${RUN_ID:?Set RUN_ID env var, e.g. RUN_ID=20260408_143022 ./resumeMistral.sh}"

# Resume stages 2-4 with Mistral-Nemo on GPUs 4-7 (ports 8004-8007)
# Assumes Mistral vLLM servers already running:
#   ~/startMistral 4 8004 4

echo "=== Resuming pipeline with Mistral-Nemo on ports 8004-8007 ==="

conda run --no-capture-output -n skills python -m skills_extraction \
  --input ../jobs/SampleJobs.json \
  --output-dir ./out \
  --vllm \
  --vllm-host localhost \
  --run-id "$RUN_ID" \
  --vllm-base-port 8004 \
  --vllm-num-endpoints 4 \
  --extractor-model "Qwen/Qwen3-14B" \
  --verifier-model "mistralai/Mistral-Nemo-Instruct-2407" \
  --requirement-model "mistralai/Mistral-Nemo-Instruct-2407" \
  --hardsoft-model "mistralai/Mistral-Nemo-Instruct-2407"

echo "=== Pipeline complete ==="
