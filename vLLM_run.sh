#!/bin/bash
cd ~/skills_extraction

# Create conda env if it doesn't exist
if ! conda info --envs | grep -q "skills"; then
  conda create -n skills python=3.11 -y
fi

# Install deps
conda run --no-capture-output -n skills pip install -r requirements.txt

# --- Stage 1: Extract with Qwen (assumes Qwen vLLM servers already running) ---
echo "=== Stage 1: Extraction (Qwen/Qwen3-14B) ==="
conda run --no-capture-output -n skills python -m skills_extraction \
  --input ../jobs/SampleJobs.json \
  --output-dir ./out \
  --vllm \
  --vllm-host localhost \
  --vllm-base-port 8000 \
  --vllm-num-endpoints 8 \
  --extractor-model "Qwen/Qwen3-14B" \
  --no-verifier \
  --no-requirement-classifier \
  --no-hardsoft-classifier

echo "=== Stage 1 complete. Swapping models... ==="

# --- Swap: stop Qwen, start Mistral-Nemo ---
./stopModel
sleep 5
./startMistral 8

# Wait for ALL Mistral-Nemo endpoints to be ready (retry with backoff)
PORTS="8000 8001 8002 8003 8004 8005 8006 8007"
MAX_RETRIES=12
RETRY_DELAY=15

echo "Waiting for Mistral-Nemo to load on all ports..."
sleep 30  # initial pause for model loading

for port in $PORTS; do
  attempt=0
  while [ $attempt -lt $MAX_RETRIES ]; do
    status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/v1/models)
    if [ "$status" = "200" ]; then
      echo "  Port $port: ready"
      break
    fi
    attempt=$((attempt + 1))
    echo "  Port $port: not ready (attempt $attempt/$MAX_RETRIES), waiting ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
  done
  if [ $attempt -eq $MAX_RETRIES ]; then
    echo "ERROR: Port $port failed to respond after $MAX_RETRIES attempts. Aborting."
    exit 1
  fi
done

echo "All endpoints ready."

# --- Stages 2-4: Verify + classify with Mistral-Nemo (resume picks up from stage 2) ---
echo "=== Stages 2-4: Verify + Classify (Mistral-Nemo) ==="
conda run --no-capture-output -n skills python -m skills_extraction \
  --input ../jobs/SampleJobs.json \
  --output-dir ./out \
  --vllm \
  --vllm-host localhost \
  --vllm-base-port 8000 \
  --vllm-num-endpoints 8 \
  --extractor-model "Qwen/Qwen3-14B" \
  --verifier-model "mistralai/Mistral-Nemo-Instruct-2407" \
  --requirement-model "mistralai/Mistral-Nemo-Instruct-2407" \
  --hardsoft-model "mistralai/Mistral-Nemo-Instruct-2407"

echo "=== Pipeline complete ==="

# Clean up
./stopModel
