param (
    [string]$commitMessage
)

if (-not $commitMessage) {
    Write-Host "Usage: .\run_versioning_apps.ps1 <commit_message>"
    exit 1
}

try {
    # Run the Python script with the commit message
    python version_apps.py $commitMessage
} catch {
    Write-Host "An error occurred: $_"
    exit 1
}