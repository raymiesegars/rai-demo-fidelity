# Download all MuseTalk model weights into vendor/MuseTalk/models
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv\Scripts"
$Models = Join-Path $Root "vendor\MuseTalk\models"

if (-not (Test-Path $Venv)) {
    Write-Error "Run setup.ps1 first to create the venv."
}

$Py = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location $Root
& $Py download_weights.py
