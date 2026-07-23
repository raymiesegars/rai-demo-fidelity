# Setup Sonic (CVPR 2025) for local bench — quality-first defaults.
# Run from services\avatar:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_sonic.ps1
#
# Downloads ~15GB+ (SVD-XT + Sonic + Whisper + RIFE). Needs a roomy GPU
# (paper tested 32GB; 24GB works at 512px / fp16).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Vendor = Join-Path $Root "vendor\Sonic"
$Venv = Join-Path $Root ".venv_sonic"
$Ckpt = Join-Path $Vendor "checkpoints"

Write-Host "==> Sonic setup"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "vendor") | Out-Null

if (-not (Test-Path (Join-Path $Vendor ".git"))) {
  Write-Host "Cloning jixiaozhong/Sonic ..."
  git clone --depth 1 https://github.com/jixiaozhong/Sonic.git $Vendor
} else {
  Write-Host "Vendor already present: $Vendor"
}

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
  Write-Host "Creating .venv_sonic (Python 3.11 preferred) ..."
  py -3.11 -m venv $Venv
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.11 not found, trying 3.10 ..."
    py -3.10 -m venv $Venv
  }
}
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Failed to create .venv_sonic" }

& $Py -m pip install -U pip setuptools wheel
Write-Host "Installing torch cu124 ..."
& $Py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

Write-Host "Installing Sonic deps ..."
# Flexible pins: official reqs are older; keep API-compatible packages.
& $Py -m pip install `
  "diffusers==0.29.0" `
  "transformers==4.43.2" `
  "accelerate" `
  "omegaconf==2.3.0" `
  "einops==0.7.0" `
  "librosa>=0.10" `
  "imageio" `
  "imageio-ffmpeg" `
  "opencv-python-headless<4.12" `
  "numpy<2" `
  "soundfile" `
  "tqdm" `
  "huggingface_hub[cli]" `
  "safetensors" `
  "Pillow"
if ($LASTEXITCODE -ne 0) { throw "deps install failed" }

New-Item -ItemType Directory -Force -Path $Ckpt | Out-Null

function Ensure-Hf([string]$Repo, [string]$LocalDir, [string]$Marker) {
  $mark = Join-Path $LocalDir $Marker
  if (Test-Path $mark) {
    Write-Host "OK: $LocalDir ($Marker)"
    return
  }
  Write-Host "Downloading $Repo → $LocalDir (large; be patient) ..."
  New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
  $env:HF_REPO = $Repo
  $env:HF_DIR = $LocalDir
  & $Py -c "from huggingface_hub import snapshot_download; import os; snapshot_download(os.environ['HF_REPO'], local_dir=os.environ['HF_DIR']); print('downloaded', os.environ['HF_REPO'])"
  if ($LASTEXITCODE -ne 0) { throw "Download failed for $Repo" }
  if (-not (Test-Path $mark)) { throw "Download incomplete for $Repo (missing $Marker)" }
}

# Sonic weights + RIFE + YOLO face
Ensure-Hf "LeonJoe13/Sonic" $Ckpt "Sonic\unet.pth"
# SVD-XT backbone (large)
Ensure-Hf "stabilityai/stable-video-diffusion-img2vid-xt" (Join-Path $Ckpt "stable-video-diffusion-img2vid-xt") "unet\diffusion_pytorch_model.fp16.safetensors"
# Whisper tiny audio encoder
Ensure-Hf "openai/whisper-tiny" (Join-Path $Ckpt "whisper-tiny") "model.safetensors"

# Sanity layout
$need = @(
  "Sonic\unet.pth",
  "Sonic\audio2token.pth",
  "Sonic\audio2bucket.pth",
  "RIFE\flownet.pkl",
  "yoloface_v5m.pt",
  "stable-video-diffusion-img2vid-xt\unet\diffusion_pytorch_model.fp16.safetensors",
  "whisper-tiny\model.safetensors"
)
foreach ($rel in $need) {
  $p = Join-Path $Ckpt $rel
  if (-not (Test-Path $p)) { throw "Missing checkpoint: $p" }
  Write-Host "OK $rel"
}

$EnvFile = Join-Path $Root ".env"
$lines = @(
  "SONIC_PYTHON=$Py",
  "SONIC_ROOT=$Vendor",
  "SONIC_MIN_RES=512",
  "SONIC_STEPS=10",
  "SONIC_DYNAMIC_SCALE=1.0",
  "SONIC_EXPAND_RATIO=0.5",
  "SONIC_RIFE=1"
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
Write-Host "DONE. Next:"
Write-Host "  .\run_server.ps1 -Backend sonic"
Write-Host "Quality knobs: SONIC_STEPS=10 (fast bench) or 25 (official, very slow); SONIC_RIFE=1"
Write-Host "License: Sonic is CC BY-NC-SA - non-commercial research/demo only."
Write-Host "SONIC_PYTHON=$Py"
