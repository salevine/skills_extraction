#!/bin/bash
SERVER="stacey@titan3.cs.gsu.edu"

# Create necessary directories on the server
ssh "$SERVER" "mkdir -p ~/skills_extraction ~/jobs"

# Copy the skills_extraction project
rsync -avz \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude '.pytest_cache' \
  --exclude '.mypy_cache' \
  --exclude '.ruff_cache' \
  --exclude '.DS_Store' \
  --exclude 'out/' \
  --exclude 'skills_extraction_output' \
  --exclude '_skills_test_out' \
  --exclude '*.log' \
  /Users/stacey/Documents/GitHub/skills_extraction/ \
  "$SERVER":~/skills_extraction/

# Copy SampleJobs.json only with -s flag
if [ "$1" = "-s" ]; then
  echo "Copying SampleJobs.json..."
  scp /Users/stacey/Documents/GitHub/NLP/SampleJobs.json \
    "$SERVER":~/jobs/SampleJobs.json
fi
