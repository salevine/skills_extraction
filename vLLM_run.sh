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

# Wait for Mistral-Nemo to be ready
echo "Waiting for Mistral-Nemo to load..."
sleep 60

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
