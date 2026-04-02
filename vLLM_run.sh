#!/bin/bash
cd ~/skills_extraction

# Create conda env if it doesn't exist
if ! conda info --envs | grep -q "skills"; then
  conda create -n skills python=3.11 -y
fi

# Install deps and run using conda run (avoids conda init requirement)
conda run --no-capture-output -n skills pip install -r requirements.txt

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
  --no-hardsoft-classifier \
  --sample 5
