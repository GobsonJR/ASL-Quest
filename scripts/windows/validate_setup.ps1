<#
.SYNOPSIS
    Read-only validation of an ASL-Quest checkout: model files, database
    migration state, and required environment variables.

.DESCRIPTION
    Never installs anything, never modifies the database, never prints
    secret values (only whether each variable is set). Safe to run at any
    time, including on the real data\asl_quest.db.

.EXAMPLE
    .\scripts\windows\validate_setup.ps1
#>

$ErrorActionPreference = "Continue"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

function Write-Ok($text) { Write-Host "  [OK] $text" -ForegroundColor Green }
function Write-Warn($text) { Write-Host "  [!]  $text" -ForegroundColor Yellow }
function Write-Fail($text) { Write-Host "  [MISSING] $text" -ForegroundColor Red }

$problems = 0

# --- Model files --------------------------------------------------------------
Write-Host "`n-- Model files (required at runtime) --" -ForegroundColor Cyan
$requiredModels = @(
    @{ Path = "models\best_model.pth"; For = "A-Z recognition" },
    @{ Path = "models\hand_landmarker.task"; For = "A-Z hand cropping (MediaPipe)" },
    @{ Path = "models\pretrained\asl_citizen_i3d\ASL_citizen_I3D_weights.pt"; For = "Native Signs (I3D backbone)" },
    @{ Path = "outputs\asl_citizen_native_10\experiments\i3d_frozen\checkpoints\head_best.pt"; For = "Native Signs (trained head)" }
)
foreach ($model in $requiredModels) {
    $fullPath = Join-Path $RepoRoot $model.Path
    if (Test-Path $fullPath) {
        $sizeMb = [math]::Round((Get-Item $fullPath).Length / 1MB, 1)
        Write-Ok "$($model.Path) ($sizeMb MB)  -  $($model.For)"
    } else {
        Write-Fail "$($model.Path)  -  $($model.For). See deployment\README.md."
        $problems++
    }
}

# --- Database + migrations ------------------------------------------------------
Write-Host "`n-- Database --" -ForegroundColor Cyan
$dbPath = Join-Path $RepoRoot "data\asl_quest.db"
if (Test-Path $dbPath) {
    Write-Ok "data\asl_quest.db exists"
    $venvPython = Join-Path $RepoRoot "venv311\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $current = & $venvPython -m alembic current 2>$null
        if ($current) {
            Write-Host "  Current revision: $($current -join ' ')"
        } else {
            Write-Warn "Could not read current Alembic revision  -  has it been migrated yet? Run 'venv311\Scripts\python.exe -m alembic upgrade head'."
            $problems++
        }
    } else {
        Write-Warn "venv311 not found yet  -  run setup.ps1 first to check the migration state."
    }
} else {
    Write-Warn "data\asl_quest.db does not exist yet. Either run 'venv311\Scripts\python.exe -m alembic upgrade head' to create a fresh migrated database, or copy an existing one into deployment\database\  -  see SETUP_WINDOWS.md for the recommendation."
}

# --- Environment variables -----------------------------------------------------
Write-Host "`n-- Environment variables (presence only, never values) --" -ForegroundColor Cyan
$envVars = @(
    @{ Name = "DATABASE_URL"; Required = $false; Note = "unset = local SQLite dev database (fine for a single-laptop demo)" },
    @{ Name = "ASL_QUEST_SECRET_KEY"; Required = $true; Note = "JWT signing key  -  set a real value outside pure local dev" },
    @{ Name = "ASL_QUEST_TOKEN_EXPIRE_MINUTES"; Required = $false; Note = "unset = 10080 (7 days)" },
    @{ Name = "ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL"; Required = $false; Note = "unset = no auto-admin bootstrap" },
    @{ Name = "ASL_QUEST_CORS_ORIGINS"; Required = $false; Note = "unset = http://127.0.0.1:5173,http://localhost:5173" },
    @{ Name = "OPENROUTER_API_KEY"; Required = $false; Note = "unset = assistant replies with a graceful setup message instead of answering" },
    @{ Name = "CHATBOT_MODEL"; Required = $false; Note = "unset = openrouter/free" }
)
foreach ($v in $envVars) {
    $value = [Environment]::GetEnvironmentVariable($v.Name)
    if ($value) {
        Write-Ok "$($v.Name) is set"
    } elseif ($v.Required) {
        Write-Warn "$($v.Name) is not set  -  $($v.Note)"
    } else {
        Write-Host "  [ - ] $($v.Name) not set ($($v.Note))"
    }
}
$envFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Warn ".env does not exist  -  copy .env.example to .env and fill in real values. (Environment variables set another way, e.g. in your shell, still work without it.)"
}

Write-Host ""
if ($problems -gt 0) {
    Write-Host "$problems item(s) need attention before all features will work." -ForegroundColor Yellow
} else {
    Write-Host "All required runtime assets found." -ForegroundColor Green
}
