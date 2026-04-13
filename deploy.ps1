$Server = "stacey@titan3.cs.gsu.edu"
$Local  = "D:/PhD/skills-extraction"
$Remote = "~/skills_extraction"

# Create necessary directories on the server
ssh $Server "mkdir -p $Remote ~/jobs"

# Deploy Python package: remove old, copy fresh, clean pycache
Write-Host "Deploying Python package..."
ssh $Server "rm -rf $Remote/skills_extraction && mkdir -p $Remote/skills_extraction"
scp -r "$Local/skills_extraction/" "${Server}:${Remote}/"
ssh $Server "rm -rf $Remote/skills_extraction/__pycache__"

# Verify deployment
Write-Host "Verifying..."
$lineCount = ssh $Server "wc -l < $Remote/skills_extraction/pipeline.py"
Write-Host "  pipeline.py: $lineCount lines"

# Copy top-level scripts
Write-Host "Deploying scripts..."
$scripts = @(
    "check_status.sh",
    "resumeMistral.sh",
    "vLLM_run.sh",
    "test_vllm.sh",
    "full_run.sh",
    "setup.py",
    "pyproject.toml"
)
foreach ($f in $scripts) {
    $path = "$Local/$f"
    if (Test-Path $path) {
        scp $path "${Server}:${Remote}/$f"
    }
}

# Copy SampleJobs.json with -s flag
if ($args -contains "-s") {
    Write-Host "Copying SampleJobs.json..."
    scp "D:/PhD/jobs/SampleJobs.json" "${Server}:~/jobs/SampleJobs.json"
}

Write-Host "Deploy complete."
