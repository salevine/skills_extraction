$Server = "stacey@titan3.cs.gsu.edu"
$Local  = "D:/PhD/skills-extraction"
$Remote = "~/skills_extraction"

# Create necessary directories on the server
ssh $Server "mkdir -p $Remote ~/jobs"

# Copy the skills_extraction project
rsync -avz `
  --exclude '.venv' `
  --exclude '__pycache__' `
  --exclude '.git' `
  --exclude 'skills_extraction_output' `
  --exclude '_skills_test_out' `
  --exclude '*.log' `
  --exclude 'out' `
  --exclude 'nul' `
  --exclude 'full_run_output' `
  "$Local/" "${Server}:${Remote}/"

# Copy SampleJobs.json with -s flag
if ($args -contains "-s") {
    Write-Host "Copying SampleJobs.json..."
    scp "D:/PhD/jobs/SampleJobs.json" "${Server}:~/jobs/SampleJobs.json"
}

Write-Host "Deploy complete."
