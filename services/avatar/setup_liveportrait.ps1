# Setup FasterLivePortrait + JoyVASA for audio-driven still animation.
# Run from services\avatar:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_liveportrait.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Vendor = Join-Path $Root "vendor\FasterLivePortrait"
$Venv = Join-Path $Root ".venv_liveportrait"
$Ckpt = Join-Path $Vendor "checkpoints"

Write-Host "==> LivePortrait (FasterLivePortrait + JoyVASA) setup"
Write-Host "    Expect multi-GB downloads and 15-40 minutes."
New-Item -ItemType Directory -Force -Path (Join-Path $Root "vendor") | Out-Null

if (-not (Test-Path (Join-Path $Vendor ".git"))) {
  Write-Host "Cloning warmshao/FasterLivePortrait ..."
  git clone --depth 1 https://github.com/warmshao/FasterLivePortrait.git $Vendor
} else {
  Write-Host "Vendor already present: $Vendor"
}

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
  Write-Host "Creating .venv_liveportrait (Python 3.10 preferred) ..."
  py -3.10 -m venv $Venv
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.10 not found, trying 3.11 ..."
    py -3.11 -m venv $Venv
  }
}
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Failed to create .venv_liveportrait" }

& $Py -m pip install -U pip setuptools wheel
Write-Host "Installing torch cu124 ..."
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

$Req = Join-Path $Vendor "requirements.txt"
if (Test-Path $Req) {
  Write-Host "Installing FasterLivePortrait requirements ..."
  & $Py -m pip install -r $Req
}

# Pin onnxruntime-gpu for CUDA 12 (1.27+ wants CUDA 13).
# Quote carefully — PowerShell treats < as an operator.
Write-Host "Installing onnxruntime-gpu + FasterLivePortrait extras ..."
& $Py -m pip install 'onnxruntime-gpu>=1.18,<1.27' omegaconf huggingface_hub `
  transformers accelerate librosa soundfile opencv-python-headless `
  imageio imageio-ffmpeg 'numpy<2' scikit-image tqdm pyyaml einops `
  torchgeometry torchaudio ffmpeg-python munch mediapipe onnx insightface pykalman
if ($LASTEXITCODE -ne 0) { throw "deps install failed" }
# Keep numpy on 1.x after mediapipe/insightface may pull 2.x.
# insightface often installs CPU onnxruntime and shadows the GPU build — fix that.
& $Py -m pip install 'numpy<2' -q
& $Py -m pip uninstall -y onnxruntime 2>$null
& $Py -m pip install 'onnxruntime-gpu==1.20.1' -q
# CUDA EP needs these DLLs on PATH (worker prepends them at runtime).
& $Py -m pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 `
  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 `
  nvidia-nvjitlink-cu12 -q

Write-Host "Applying FLP compatibility patches ..."
& $Py (Join-Path $Root "scripts\patch_flp_compat.py") $Vendor

New-Item -ItemType Directory -Force -Path $Ckpt | Out-Null

$Dl = Join-Path $Root "scripts\_download_liveportrait.py"
@'
"""Download LivePortrait / JoyVASA / HuBERT checkpoints into FLP checkpoints/."""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ckpt = Path(sys.argv[1]).resolve()
ckpt.mkdir(parents=True, exist_ok=True)

print("Downloading LivePortrait ONNX (~2GB)...")
snapshot_download(repo_id="warmshao/FasterLivePortrait", local_dir=str(ckpt))

print("Downloading JoyVASA...")
snapshot_download(repo_id="jdh-algo/JoyVASA", local_dir=str(ckpt / "JoyVASA"))

print("Downloading chinese-hubert-base...")
snapshot_download(
    repo_id="TencentGameMate/chinese-hubert-base",
    local_dir=str(ckpt / "chinese-hubert-base"),
)

motion_dir = ckpt / "JoyVASA" / "motion_generator"
expected = motion_dir / "motion_generator_hubert_chinese.pt"
if not expected.is_file():
    candidates = [
        "motion_generator_hubert_chinese.pt",
        "iter_0020000.pt",
        "motion_generator.pt",
    ]
    found = None
    for name in candidates:
        p = motion_dir / name
        if p.is_file():
            found = p
            break
    if found is None:
        pts = list(motion_dir.glob("*.pt")) if motion_dir.is_dir() else []
        found = pts[0] if pts else None
    if found is not None and found != expected:
        shutil.copy2(found, expected)
        print(f"Mapped JoyVASA weight: {found.name} -> {expected.name}")

need = [
    ckpt / "liveportrait_onnx" / "warping_spade.onnx",
    expected,
    ckpt / "JoyVASA" / "motion_template" / "motion_template.pkl",
    ckpt / "chinese-hubert-base" / "config.json",
]
missing = [str(p) for p in need if not p.is_file()]
for p in need:
    print(("OK  " if p.is_file() else "MISSING  ") + str(p))
if missing:
    raise SystemExit(f"Setup incomplete — missing {len(missing)} file(s)")
print("All required checkpoint files present.")
'@ | Set-Content -Path $Dl -Encoding UTF8

Write-Host "Downloading checkpoints (long) ..."
& $Py $Dl $Ckpt
if ($LASTEXITCODE -ne 0) { throw "checkpoint download failed" }

Write-Host ""
Write-Host "=============================================="
Write-Host " SUCCESS - LivePortrait ready"
Write-Host "=============================================="
Write-Host "Run:"
Write-Host "  .\run_server.ps1 -Backend liveportrait"
Write-Host "UI: http://127.0.0.1:8100  (Full image framing)"
