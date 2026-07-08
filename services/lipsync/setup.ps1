# One-time setup: venv + PyTorch + MuseTalk deps + ffmpeg
# Run from: services\lipsync
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = "C:\Users\raymi\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path $Python)) {
    $Python = (Get-Command py -ErrorAction SilentlyContinue).Source
    if ($Python) { $Python = "py -3.11" }
}

Write-Host "==> Creating venv..."
if (-not (Test-Path ".venv")) {
    & $Python -m venv .venv
}

$VenvPy = ".\.venv\Scripts\python.exe"
& $VenvPy -m pip install --upgrade pip setuptools wheel

Write-Host "==> Installing PyTorch 2.0.1 (CUDA 11.8)..."
& $VenvPy -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

Write-Host "==> Installing MuseTalk requirements..."
& $VenvPy -m pip install -r vendor\MuseTalk\requirements.txt

Write-Host "==> Installing MMLab (mmcv, mmdet, mmpose)..."
& $VenvPy -m pip install --no-cache-dir -U openmim
& $VenvPy -m mim install mmengine
& $VenvPy -m mim install "mmcv==2.0.1"
& $VenvPy -m mim install "mmdet==3.1.0"
& $VenvPy -m pip install chumpy --no-build-isolation
& $VenvPy -m mim install "mmpose==1.1.0"
& $VenvPy -m pip install "numpy==1.23.5"

Write-Host "==> Installing ffmpeg (if missing)..."
$FfmpegDir = Join-Path $Root "tools\ffmpeg"
if (-not (Get-ChildItem -Path $FfmpegDir -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue)) {
    New-Item -ItemType Directory -Force -Path $FfmpegDir | Out-Null
    $zip = "$env:TEMP\ffmpeg-win.zip"
    Invoke-WebRequest -Uri "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $FfmpegDir -Force
}

Write-Host "==> Verifying CUDA..."
& $VenvPy -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

Write-Host ""
Write-Host "Setup complete. Next steps:"
Write-Host "  1. Copy alan-loop.mp4 to services\lipsync\assets\alan-loop.mp4"
Write-Host "  2. .\download_weights.ps1"
Write-Host "  3. .\run_prepare_avatar.ps1"
Write-Host "  4. .\run_test_clip.ps1"
