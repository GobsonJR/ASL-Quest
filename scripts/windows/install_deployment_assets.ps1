<#
.SYNOPSIS
    Moves manually-copied runtime assets from deployment\ into their real
    destination paths.

.DESCRIPTION
    Matches files dropped in deployment\models\ by filename against the
    known model checkpoints (see deployment\README.md) and moves each one
    to its real path, creating destination folders as needed. Never
    overwrites an existing destination file unless -Force is passed.

    Also handles deployment\database\asl_quest.db -> data\asl_quest.db if
    present, with the same overwrite protection.

.PARAMETER Force
    Overwrite existing destination files/database instead of skipping them.

.EXAMPLE
    .\scripts\windows\install_deployment_assets.ps1
    .\scripts\windows\install_deployment_assets.ps1 -Force
#>

[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

function Write-Ok($text) { Write-Host "  [OK] $text" -ForegroundColor Green }
function Write-Skip($text) { Write-Host "  [SKIP] $text" -ForegroundColor Yellow }

$modelMap = @{
    "best_model.pth"              = "models\best_model.pth"
    "hand_landmarker.task"        = "models\hand_landmarker.task"
    "ASL_citizen_I3D_weights.pt"  = "models\pretrained\asl_citizen_i3d\ASL_citizen_I3D_weights.pt"
    "head_best.pt"                = "outputs\asl_citizen_native_10\experiments\i3d_frozen\checkpoints\head_best.pt"
}

Write-Host "-- Model files --" -ForegroundColor Cyan
$stagingDir = Join-Path $RepoRoot "deployment\models"
$found = $false
foreach ($file in Get-ChildItem -Path $stagingDir -File -ErrorAction SilentlyContinue) {
    if ($file.Name -eq ".gitkeep") { continue }
    $found = $true
    if (-not $modelMap.ContainsKey($file.Name)) {
        Write-Host "  [?] $($file.Name)  -  unrecognized filename, left in place. See deployment\README.md for expected names." -ForegroundColor Yellow
        continue
    }
    $destination = Join-Path $RepoRoot $modelMap[$file.Name]
    $destinationDir = Split-Path $destination -Parent
    if (-not (Test-Path $destinationDir)) { New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null }
    if ((Test-Path $destination) -and (-not $Force)) {
        Write-Skip "$($file.Name)  -  $destination already exists (pass -Force to overwrite)"
        continue
    }
    Move-Item -Path $file.FullName -Destination $destination -Force
    Write-Ok "$($file.Name) -> $($modelMap[$file.Name])"
}
if (-not $found) {
    Write-Host "  (deployment\models\ is empty  -  nothing to install)"
}

Write-Host "`n-- Database (optional) --" -ForegroundColor Cyan
$dbSource = Join-Path $RepoRoot "deployment\database\asl_quest.db"
if (Test-Path $dbSource) {
    $dbDestination = Join-Path $RepoRoot "data\asl_quest.db"
    $destExists = (Test-Path $dbDestination) -and ((Get-Item $dbDestination).Length -gt 0)
    if ($destExists -and (-not $Force)) {
        Write-Skip "data\asl_quest.db already exists and is non-empty (pass -Force to overwrite  -  this discards its current contents)"
    } else {
        New-Item -ItemType Directory -Path (Join-Path $RepoRoot "data") -Force | Out-Null
        Move-Item -Path $dbSource -Destination $dbDestination -Force
        Write-Ok "asl_quest.db -> data\asl_quest.db"
    }
} else {
    Write-Host "  (deployment\database\asl_quest.db not present  -  fine if you're starting from a fresh migrated database)"
}

Write-Host "`nRun .\scripts\windows\validate_setup.ps1 to confirm everything is now in place." -ForegroundColor Cyan
