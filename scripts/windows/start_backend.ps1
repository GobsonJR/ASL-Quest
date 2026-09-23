<#
.SYNOPSIS
    Starts the ASL-Quest FastAPI backend (same command as README.md).
#>

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

$venvPython = Join-Path $RepoRoot "venv311\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "venv311 not found. Run .\scripts\windows\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# backend/settings.py only reads os.environ, so load .env (if present) into the
# process environment via uvicorn's --env-file (python-dotenv ships with
# uvicorn[standard]). Variables already set in the shell take precedence.
$uvicornArgs = @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000")
if (Test-Path (Join-Path $RepoRoot ".env")) {
    $uvicornArgs += @("--env-file", ".env")
}

& $venvPython @uvicornArgs
