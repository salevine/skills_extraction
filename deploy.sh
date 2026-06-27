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
  --exclude '.claude/' \
  --exclude '.gitnexus/' \
  --exclude 'CLAUDE.md' \
  --exclude 'AGENTS.md' \
  --exclude 'README.md' \
  --exclude 'skills_extraction/README.md' \
  --exclude '.gitignore' \
  --exclude '.github' \
  --exclude 'test_payload.py' \
  --exclude 'test_fixes.py' \
  --exclude 'test_simple.py' \
  --exclude 'deploy.sh' \
  --exclude 'deploy.bat' \
  --exclude 'deploy.ps1' \
  --exclude 'analyze_log.py' \
  --exclude 'analyze_stage1.py' \
  --exclude 'compare_runs.py' \
  --exclude 'show_job.py' \
  --exclude 'archive_old_runs.py' \
  --exclude 'clone_stage1_checkpoint.py' \
  --exclude 'prompts_updated.py' \
  --exclude 'Runskills_extraction.py' \
  --exclude 'script_part1.b64' \
  --exclude 'CHAPTER_*' \
  --exclude 'NOTES_*' \
  --exclude 'FUTURE_*' \
  --exclude 'full_run_output' \
  --exclude 'skills_extraction/CHANGELOG.md' \
  --exclude 'skills_extraction/README.md' \
  --exclude 'skills_extraction/prompts_backup_*' \
  --exclude 'out/' \
  --exclude 'skills_extraction_output' \
  --exclude '_skills_test_out' \
  --exclude '*.log' \
  --exclude '*.xlsx' \
  /Users/stacey/Documents/GitHub/skills_extraction/ \
  "$SERVER":~/skills_extraction/

# Copy SampleJobs.json only with -s flag
if [ "$1" = "-s" ]; then
  echo "Copying SampleJobs.json..."
  scp /Users/stacey/Documents/GitHub/NLP/SampleJobs.json \
    "$SERVER":~/jobs/SampleJobs.json
fi
