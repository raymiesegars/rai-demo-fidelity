# Setup LiveTalk (GAIR-NLP) for local bench — clip/utterance path.
# Official env is Linux + flash-attn; Windows is best-effort (SDPA fallback).
# Run from services\avatar:
#   .\setup_livetalk.ps1
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Vendor = Join-Path $Root "vendor\LiveTalk"
$Venv = Join-Path $Root ".venv_livetalk"
$Ckpt = Join-Path $Vendor "pretrained_checkpoints"

Write-Host "==> LiveTalk setup"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "vendor") | Out-Null

if (-not (Test-Path (Join-Path $Vendor ".git"))) {
  Write-Host "Cloning GAIR-NLP/LiveTalk ..."
  git clone --depth 1 https://github.com/GAIR-NLP/LiveTalk.git $Vendor
} else {
  Write-Host "Vendor present: $Vendor"
}

$Omni = Join-Path $Vendor "OmniAvatar"
if (-not (Test-Path (Join-Path $Omni ".git"))) {
  Write-Host "Cloning Omni-Avatar/OmniAvatar into LiveTalk ..."
  git clone --depth 1 https://github.com/Omni-Avatar/OmniAvatar.git $Omni
}

Write-Host "Applying OmniAvatar patches ..."
$Patch = Join-Path $Vendor "utils\Omniavatarpatch"
Copy-Item (Join-Path $Patch "model_config.py") (Join-Path $Omni "OmniAvatar\configs\model_config.py") -Force
Copy-Item (Join-Path $Patch "wan_video_dit.py") (Join-Path $Omni "OmniAvatar\models\wan_video_dit.py") -Force
Copy-Item (Join-Path $Patch "wan_video.py") (Join-Path $Omni "OmniAvatar\wan_video.py") -Force
Copy-Item (Join-Path $Patch "wav2vec.py") (Join-Path $Omni "OmniAvatar\models\wav2vec.py") -Force
Copy-Item (Join-Path $Patch "args_config.py") (Join-Path $Omni "OmniAvatar\utils\args_config.py") -Force
Copy-Item (Join-Path $Patch "flow_match.py") (Join-Path $Omni "OmniAvatar\schedulers\flow_match.py") -Force

# Upstream ships a broken import (scripts.inference missing) — keep our shim.
$ShimSrc = Join-Path $Root "scripts\livetalk_inference_shim.py"
$ShimDst = Join-Path $Vendor "scripts\inference.py"
if (Test-Path $ShimSrc) {
  Copy-Item $ShimSrc $ShimDst -Force
  Write-Host "Installed LiveTalk scripts/inference.py shim"
}

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
  Write-Host "Creating .venv_livetalk ..."
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
if (-not (Test-Path $Py)) { throw "Failed to create .venv_livetalk (need Python 3.11+ on PATH)" }

& $Py -m pip install -U pip setuptools wheel
Write-Host "Installing torch cu124 ..."
& $Py -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

Write-Host "Installing LiveTalk deps (skipping Linux-only flash-attn / tensorrt) ..."
& $Py -m pip install `
  "opencv-python>=4.9" `
  "diffusers==0.31.0" `
  "transformers>=4.49" `
  accelerate tqdm imageio imageio-ffmpeg `
  numpy easydict ftfy omegaconf einops `
  "av==13.1.0" opencv-python-headless `
  open_clip_torch starlette pydantic==2.10.6 `
  scikit-image huggingface_hub dominate `
  onnx onnxruntime peft==0.15.1 `
  "librosa==0.10.2.post1" scipy `
  soundfile pyyaml pillow
if ($LASTEXITCODE -ne 0) { throw "deps install failed" }

# Optional CLIP from OpenAI (best-effort)
& $Py -m pip install "git+https://github.com/openai/CLIP.git" 2>&1 | Select-Object -Last 15

Push-Location $Vendor
& $Py setup.py develop 2>&1 | Select-Object -Last 20
Pop-Location

New-Item -ItemType Directory -Force -Path $Ckpt | Out-Null
Write-Host "Downloading checkpoints (multi-GB; needs huggingface-cli / HF login) ..."
& $Py -m pip install -q "huggingface_hub" 2>&1 | Out-Null
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

function Ensure-Hf([string]$Repo, [string]$Dest) {
  if (Test-Path $Dest) {
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

Ensure-Hf "Wan-AI/Wan2.1-T2V-1.3B" (Join-Path $Ckpt "Wan2.1-T2V-1.3B")
Ensure-Hf "GAIR/LiveTalk-1.3B-V0.1" (Join-Path $Ckpt "LiveTalk-1.3B-V0.1")
Ensure-Hf "facebook/wav2vec2-base-960h" (Join-Path $Ckpt "wav2vec2")

$EnvFile = Join-Path $Root ".env"
$lines = @(
  "LIVETALK_PYTHON=$Py",
  "LIVETALK_ROOT=$Vendor",
  "LIVETALK_PROMPT=A realistic video of a person speaking directly to the camera with natural expressions and precise lip sync."
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
Write-Host "  .\run_server.ps1 -Backend livetalk -KillExisting"
Write-Host "  Note: official LiveTalk targets Linux; Windows uses SDPA (no flash-attn)."
Write-Host "LIVETALK_PYTHON=$Py"
