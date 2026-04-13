$Server = "stacey@titan3.cs.gsu.edu"
$Local  = "D:/PhD/skills-extraction"
$Remote = "~/skills_extraction"

# Create necessary directories on the server
ssh $Server "mkdir -p $Remote/skills_extraction ~/jobs"

# Copy Python package
scp -r "$Local/skills_extraction/" "${Server}:${Remote}/skills_extraction/"

# Copy top-level scripts
$scripts = @(
    "check_status.sh",
    "resumeMistral.sh",
    "vLLM_run.sh",
    "test_vllm.sh",
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
