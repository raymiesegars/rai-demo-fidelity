# Setup EchoMimicV3-Flash (Ant Group) — quality generative peer to FlashHead.
# 8-step flash-pro path; ~12GB+ VRAM; tested on RTX 4090.
# Run from services\avatar:
#   .\setup_echomimic.ps1
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Vendor = Join-Path $Root "vendor\echomimic_v3"
$Venv = Join-Path $Root ".venv_echomimic"
$Flash = Join-Path $Vendor "flash"

Write-Host "==> EchoMimicV3-Flash setup"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "vendor") | Out-Null

if (-not (Test-Path (Join-Path $Vendor ".git"))) {
  Write-Host "Cloning antgroup/echomimic_v3 ..."
  git clone --depth 1 https://github.com/antgroup/echomimic_v3.git $Vendor
} else {
  Write-Host "Vendor present: $Vendor"
}

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
  Write-Host "Creating .venv_echomimic ..."
  $created = $false
  foreach ($ver in @("3.11", "3.10", "3.12")) {
    Write-Host "  trying py -$ver ..."
    & py "-$ver" -m venv $Venv 2>$null
    if ((Test-Path (Join-Path $Venv "Scripts\python.exe"))) { $created = $true; break }
  }
  if (-not $created) {
    Write-Host "  trying python ..."
    & python -m venv $Venv
  }
}
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Failed to create .venv_echomimic (need Python 3.11+ on PATH)" }

& $Py -m pip install -U pip setuptools wheel
Write-Host "Installing torch cu124 ..."
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

Write-Host "Installing EchoMimic deps (skipping pinned tensorflow) ..."
$ErrorActionPreference = "Continue"
& $Py -m pip install `
  Pillow einops safetensors timm tomesd `
  torchdiffeq torchsde decord datasets `
  "numpy<2" scikit-image opencv-python omegaconf SentencePiece `
  albumentations "imageio[ffmpeg]" `
  ftfy func_timeout onnxruntime `
  "accelerate>=0.25.0" `
  "diffusers>=0.30.1" `
  "transformers>=4.46.2" `
  "moviepy==2.2.1" `
  librosa soundfile pyloudnorm `
  tqdm pyyaml
if ($LASTEXITCODE -ne 0) { throw "deps install failed" }

# retina-face pulls tensorflow - optional; face mask not required for flash-pro
& $Py -m pip install mmgp 2>&1 | Select-Object -Last 10

New-Item -ItemType Directory -Force -Path $Flash | Out-Null
# Newer huggingface_hub dropped the [cli] extra; install base package only.
& $Py -m pip install -q "huggingface_hub" 2>&1 | Out-Null
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

function Ensure-Hf([string]$Repo, [string]$Dest) {
  if ((Test-Path $Dest) -and (Get-ChildItem $Dest -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count -gt 5) {
    Write-Host "OK: $Dest"
    return
  }
  Write-Host "HF download $Repo -> $Dest"
  New-Item -ItemType Directory -Force -Path $Dest | Out-Null
  & $Py -c "from huggingface_hub import snapshot_download; snapshot_download(r'$Repo', local_dir=r'$Dest')"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: download failed for $Repo - run hf auth login then re-run setup"
  }
}

Ensure-Hf "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP" (Join-Path $Flash "Wan2.1-Fun-V1.1-1.3B-InP")
Ensure-Hf "TencentGameMate/chinese-wav2vec2-base" (Join-Path $Flash "chinese-wav2vec2-base")

$Xform = Join-Path $Flash "transformer"
New-Item -ItemType Directory -Force -Path $Xform | Out-Null
$XformFile = Join-Path $Xform "diffusion_pytorch_model.safetensors"
if (-not (Test-Path $XformFile)) {
  Write-Host "Downloading EchoMimicV3 flash-pro transformer weights ..."
  $Tmp = Join-Path $Vendor "_hf_flash_tmp"
  New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
  & $Py -c "from huggingface_hub import snapshot_download; snapshot_download('BadToBest/EchoMimicV3', local_dir=r'$Tmp', allow_patterns=['echomimicv3-flash-pro/*'])"
  $Cand = Join-Path $Tmp "echomimicv3-flash-pro\transformer\diffusion_pytorch_model.safetensors"
  if (-not (Test-Path $Cand)) {
    $found = Get-ChildItem $Tmp -Recurse -Filter "diffusion_pytorch_model.safetensors" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $Cand = $found.FullName }
  }
  if ($Cand -and (Test-Path $Cand)) {
    Copy-Item $Cand $XformFile -Force
    Write-Host "Transformer -> $XformFile"
  } else {
    Write-Host "WARNING: flash-pro transformer not found - check HF layout under $Tmp"
  }
} else {
  Write-Host "OK: $XformFile"
}

$EnvFile = Join-Path $Root ".env"
$lines = @(
  "ECHOMIMIC_PYTHON=$Py",
  "ECHOMIMIC_ROOT=$Vendor",
  "ECHOMIMIC_FLASH_DIR=$Flash",
  "ECHOMIMIC_STEPS=8",
  "ECHOMIMIC_SIZE=512",
  "ECHOMIMIC_PROMPT=A person is speaking naturally to the camera."
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
Write-Host "  .\run_server.ps1 -Backend echomimic -KillExisting"
Write-Host "ECHOMIMIC_PYTHON=$Py"
Write-Host "Tip: ECHOMIMIC_SIZE=768 for sharper (more VRAM); default 512 for bench latency."
