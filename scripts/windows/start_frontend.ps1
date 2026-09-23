<#
.SYNOPSIS
    Starts the ASL-Quest Vite frontend dev server (same command as README.md).
#>

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$FrontendDir = Join-Path $RepoRoot "frontend"

if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
    Write-Host "frontend\node_modules not found. Run .\scripts\windows\setup.ps1 first (or 'npm install' in frontend\)." -ForegroundColor Red
    exit 1
}

Push-Location $FrontendDir
try {
    & npm run dev
} finally {
    Pop-Location
}
