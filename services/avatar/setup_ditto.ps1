# Setup Ditto talking-head for local bench.
# Run from services\avatar:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_ditto.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Vendor = Join-Path $Root "vendor\ditto-talkinghead"
$Models = Join-Path $Root "models\ditto"
$Venv = Join-Path $Root ".venv_ditto"

Write-Host "==> Ditto setup"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "vendor") | Out-Null
New-Item -ItemType Directory -Force -Path $Models | Out-Null

if (-not (Test-Path (Join-Path $Vendor ".git"))) {
  Write-Host "Cloning antgroup/ditto-talkinghead ..."
  git clone --depth 1 https://github.com/antgroup/ditto-talkinghead.git $Vendor
} else {
  Write-Host "Vendor already present: $Vendor"
}

Write-Host "Applying Windows Ditto patches (numpy blend fallback) ..."
& (Join-Path $Root ".venv\Scripts\python.exe") (Join-Path $Root "scripts\patch_ditto_windows.py") $Vendor
# also fine if main venv missing — use ditto venv after creation

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
  Write-Host "Creating .venv_ditto (Python 3.10 recommended) ..."
  py -3.10 -m venv $Venv
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.10 not found, trying 3.11 ..."
    py -3.11 -m venv $Venv
  }
}
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Failed to create .venv_ditto" }

& $Py -m pip install -U pip
Write-Host "Installing torch cu124 + ditto deps (this takes a while) ..."
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
# onnxruntime (CPU) is more reliable on Windows than onnxruntime-gpu (needs CUDA 13+).
# Torch still uses CUDA for the heavy Ditto models.
& $Py -m pip install librosa tqdm filetype imageio opencv-python-headless opencv-python scikit-image cython imageio-ffmpeg numpy soundfile onnxruntime einops omegaconf scipy "mediapipe==0.10.14"
if ($LASTEXITCODE -ne 0) { throw "deps install failed" }

Write-Host "Re-applying Windows Ditto patches (ffmpeg via imageio-ffmpeg) ..."
& $Py (Join-Path $Root "scripts\patch_ditto_windows.py") $Vendor

$PytorchModels = Join-Path $Models "ditto_pytorch\models"
if (-not (Test-Path $PytorchModels)) {
  Write-Host "Downloading Ditto checkpoints from HuggingFace (needs git-lfs, multi-GB) ..."
  git lfs install
  if (-not (Test-Path (Join-Path $Models ".git"))) {
    git clone https://huggingface.co/digital-avatar/ditto-talkinghead $Models
  } else {
    Push-Location $Models
    git lfs pull
    Pop-Location
  }
} else {
  Write-Host "Checkpoints present under $Models"
}

$Cfg = Join-Path $Models "ditto_cfg\v0.4_hubert_cfg_pytorch.pkl"
$Data = Join-Path $Models "ditto_pytorch"
if (-not (Test-Path $Cfg)) {
  Write-Host "WARNING: missing cfg file. Run: git -C models\ditto lfs pull"
  Write-Host "Expected: $Cfg"
}

# Persist env hints into .env (append if missing)
$EnvFile = Join-Path $Root ".env"
$lines = @(
  "DITTO_PYTHON=$Py",
  "DITTO_DATA_ROOT=$Data",
  "DITTO_CFG=$Cfg"
)
if (-not (Test-Path $EnvFile)) { New-Item -ItemType File -Path $EnvFile | Out-Null }
$existing = Get-Content $EnvFile -ErrorAction SilentlyContinue
foreach ($line in $lines) {
  $key = ($line -split "=", 2)[0]
  if (-not ($existing | Where-Object { $_ -like "$key=*" })) {
    Add-Content -Path $EnvFile -Value $line
  }
}

Write-Host ""
Write-Host "SUCCESS. Next:"
Write-Host "  1) Ctrl+C any running avatar server"
Write-Host "  2) .\run_server.ps1 -Backend ditto"
Write-Host "  3) Analytics -> Run 10-dialogue test -> score"
Write-Host "DITTO_PYTHON=$Py"
