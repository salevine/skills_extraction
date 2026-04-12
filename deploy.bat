@echo off
setlocal

set SERVER=stacey@titan3.cs.gsu.edu
set LOCAL=D:\PhD\skills-extraction
set REMOTE=~/skills_extraction

:: Create necessary directories on the server
ssh %SERVER% "mkdir -p %REMOTE% ~/jobs"

:: Copy the skills_extraction project
rsync -avz ^
  --exclude ".venv" ^
  --exclude "__pycache__" ^
  --exclude ".git" ^
  --exclude "skills_extraction_output" ^
  --exclude "_skills_test_out" ^
  --exclude "*.log" ^
  --exclude "out" ^
  --exclude "nul" ^
  --exclude "full_run_output" ^
  %LOCAL%/ %SERVER%:%REMOTE%/

:: Copy SampleJobs.json only with -s flag
if "%~1"=="-s" (
  echo Copying SampleJobs.json...
  scp D:\PhD\NLP\SampleJobs.json %SERVER%:~/jobs/SampleJobs.json
)

echo Deploy complete.
