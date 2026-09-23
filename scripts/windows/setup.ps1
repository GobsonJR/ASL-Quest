<#
.SYNOPSIS
    One-time ASL-Quest environment bootstrap for a Windows laptop.

.DESCRIPTION
    Checks prerequisites, creates the Python virtual environment (venv311),
    installs backend and frontend dependencies, and reports what's still
    needed (model files, database, .env) without silently making decisions
    for you. Safe to re-run  -  every step is idempotent.

    Does NOT: download or copy model checkpoints, run database migrations,
    or start either server. See deployment\README.md and SETUP_WINDOWS.md
    for those steps, or pass -RunMigrations / -SeedNativeSigns to opt in.

.PARAMETER RunMigrations
    Also run `alembic upgrade head` against the local SQLite database after
    installing dependencies (creates/updates all tables; does not touch
    existing rows).

.PARAMETER SeedNativeSigns
    Also run scripts\seed_native_signs_native_10.py after migrations
    (idempotent  -  skips glosses that already exist).

.EXAMPLE
    .\scripts\windows\setup.ps1
    .\scripts\windows\setup.ps1 -RunMigrations -SeedNativeSigns
#>

[CmdletBinding()]
param(
    [switch]$RunMigrations,
    [switch]$SeedNativeSigns
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

function Write-Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Write-Ok($text) { Write-Host "  [OK] $text" -ForegroundColor Green }
function Write-Warn($text) { Write-Host "  [!]  $text" -ForegroundColor Yellow }
function Write-Fail($text) { Write-Host "  [FAIL] $text" -ForegroundColor Red }

# --- 1. Check Python -------------------------------------------------------
Write-Step "Checking for Python 3.11"
$pythonExe = $null
try {
    $pyVersionOutput = & py -3.11 --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pythonExe = "py"
        Write-Ok "Found $pyVersionOutput via the 'py' launcher"
    }
} catch {}
if (-not $pythonExe) {
    Write-Fail "Python 3.11 not found via 'py -3.11'. Install Python 3.11 from https://www.python.org/downloads/ (or via winget: 'winget install Python.Python.3.11') and re-run this script."
    exit 1
}

# --- 2. Check Node/npm -------------------------------------------------------
Write-Step "Checking for Node.js and npm"
try {
    $nodeVersion = & node --version
    $npmVersion = & npm --version
    Write-Ok "Node $nodeVersion, npm $npmVersion"
} catch {
    Write-Fail "Node.js/npm not found on PATH. Install Node.js (LTS) from https://nodejs.org/ and re-run this script."
    exit 1
}

# --- 3. Create venv311 -------------------------------------------------------
Write-Step "Python virtual environment (venv311)"
$venvPython = Join-Path $RepoRoot "venv311\Scripts\python.exe"
if (Test-Path $venvPython) {
    Write-Ok "venv311 already exists at $venvPython"
} else {
    Write-Host "  Creating venv311 ..."
    & py -3.11 -m venv (Join-Path $RepoRoot "venv311")
    if (-not (Test-Path $venvPython)) {
        Write-Fail "venv311 creation failed."
        exit 1
    }
    Write-Ok "Created venv311"
}

# --- 4. Install backend dependencies -----------------------------------------
Write-Step "Installing backend dependencies (requirements.txt)"
& $venvPython -m pip install --upgrade pip | Out-Null
& $venvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Fail "pip install failed  -  see output above."
    exit 1
}
Write-Ok "Backend dependencies installed"

# --- 5. Install frontend dependencies -----------------------------------------
Write-Step "Installing frontend dependencies (npm install)"
Push-Location (Join-Path $RepoRoot "frontend")
try {
    & npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "npm install failed  -  see output above."
        exit 1
    }
    Write-Ok "Frontend dependencies installed"
} finally {
    Pop-Location
}

# --- 6/7/8. Validate models, database, env vars -------------------------------
Write-Step "Validating model files, database, and environment (read-only)"
& (Join-Path $PSScriptRoot "validate_setup.ps1")

# --- Optional: migrations / native signs seed ---------------------------------
if ($RunMigrations) {
    Write-Step "Running database migrations (alembic upgrade head)"
    & $venvPython -m alembic upgrade head
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "alembic upgrade head failed  -  see output above."
        exit 1
    }
    Write-Ok "Migrations applied"
} else {
    Write-Warn "Skipped migrations (pass -RunMigrations to apply them, or run 'venv311\Scripts\python.exe -m alembic upgrade head' yourself)"
}

if ($SeedNativeSigns) {
    Write-Step "Seeding native_signs catalog (idempotent)"
    & $venvPython (Join-Path $RepoRoot "scripts\seed_native_signs_native_10.py")
} else {
    Write-Warn "Skipped native_signs seeding (pass -SeedNativeSigns, or run 'venv311\Scripts\python.exe scripts\seed_native_signs_native_10.py' yourself)"
}

# --- Next steps ---------------------------------------------------------------
Write-Step "Setup complete. Next steps"
Write-Host "  1. Copy .env.example to .env and fill in real values (never commit .env)."
Write-Host "  2. Copy the 4 required model files (see deployment\README.md), or run:"
Write-Host "       .\scripts\windows\install_deployment_assets.ps1"
Write-Host "  3. Start the backend:  .\scripts\windows\start_backend.ps1"
Write-Host "  4. Start the frontend: .\scripts\windows\start_frontend.ps1"
