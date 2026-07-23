# Setup MuseTalk 1.5 for local fidelity benches (warm realtime path).
# Run from services\avatar:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_musetalk.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Vendor = Join-Path $Root "vendor\MuseTalk"
$Venv = Join-Path $Root ".venv_musetalk"

Write-Host "==> MuseTalk 1.5 setup"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "vendor") | Out-Null

if (-not (Test-Path (Join-Path $Vendor ".git"))) {
  Write-Host "Cloning TMElyralab/MuseTalk ..."
  git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git $Vendor
} else {
  Write-Host "Vendor already present: $Vendor"
}

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
  Write-Host "Creating .venv_musetalk (Python 3.10 preferred) ..."
  py -3.10 -m venv $Venv
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.10 not found, trying 3.11 ..."
    py -3.11 -m venv $Venv
  }
}
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Failed to create .venv_musetalk" }

& $Py -m pip install -U pip
Write-Host "Installing torch cu124 + MuseTalk deps (long) ..."
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

$Req = Join-Path $Vendor "requirements.txt"
if (Test-Path $Req) {
  & $Py -m pip install -r $Req
}
& $Py -m pip install soundfile opencv-python-headless imageio imageio-ffmpeg huggingface_hub omegaconf transformers accelerate einops librosa
Write-Host "Installing MMLab packages (mmengine/mmcv/mmdet/mmpose) ..."
& $Py -m pip install -U openmim
& $Py -m mim install mmengine
& $Py -m mim install "mmcv==2.1.0"
& $Py -m mim install "mmdet==3.1.0"
& $Py -m mim install "mmpose==1.1.0"

$Models = Join-Path $Vendor "models"
$V15 = Join-Path $Models "musetalkV15\unet.pth"
if (-not (Test-Path $V15)) {
  Write-Host "Downloading MuseTalk weights from HuggingFace (multi-GB) ..."
  New-Item -ItemType Directory -Force -Path $Models | Out-Null
  $dl = Join-Path $Root "uploads\_musetalk_work\download_weights.py"
  New-Item -ItemType Directory -Force -Path (Split-Path $dl) | Out-Null
  @"
from huggingface_hub import snapshot_download
import os
base = r'''$Models'''
os.makedirs(base, exist_ok=True)
snapshot_download('TMElyralab/MuseTalk', local_dir=base, local_dir_use_symlinks=False)
print('HF MuseTalk snapshot done')
for repo, dest_name in [
    ('stabilityai/sd-vae-ft-mse', 'sd-vae'),
    ('openai/whisper-tiny', 'whisper'),
]:
    dest = os.path.join(base, dest_name)
    os.makedirs(dest, exist_ok=True)
    snapshot_download(repo, local_dir=dest, local_dir_use_symlinks=False)
    print('downloaded', repo)
"@ | Set-Content -Path $dl -Encoding UTF8
  & $Py $dl
} else {
  Write-Host "MuseTalk 1.5 weights present: $V15"
}

if (-not (Test-Path $V15)) {
  Write-Host "WARNING: still missing $V15 - check HF download layout under models/musetalkV15/"
}

$EnvFile = Join-Path $Root ".env"
$lines = @(
  "MUSETALK_PYTHON=$Py",
  "MUSETALK_ROOT=$Vendor",
  "MUSETALK_VERSION=v15",
  "MUSETALK_FP16=1",
  "MUSETALK_BATCH_SIZE=8"
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
Write-Host "DONE. Next steps:"
Write-Host "  Ctrl+C any running avatar server"
Write-Host "  .\run_server.ps1 -Backend musetalk"
Write-Host "  Demo: Full image, short /say, then Analytics Capture + score"
Write-Host "MUSETALK_PYTHON=$Py"
Write-Host "Tip: MUSETALK_FP16=1 (default). Set =0 only if you have spare VRAM and want max quality."
