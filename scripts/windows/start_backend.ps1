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

& $venvPython -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
