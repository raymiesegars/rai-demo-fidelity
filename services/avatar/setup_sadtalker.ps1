# Setup SadTalker for local bench (offline portrait animation).
# Run from services\avatar:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_sadtalker.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Vendor = Join-Path $Root "vendor\SadTalker"
$Venv = Join-Path $Root ".venv_sadtalker"
$CkptDir = Join-Path $Vendor "checkpoints"
$GfpDir = Join-Path $Vendor "gfpgan\weights"
$Rel = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"

Write-Host "==> SadTalker setup"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "vendor") | Out-Null

if (-not (Test-Path (Join-Path $Vendor ".git"))) {
  Write-Host "Cloning OpenTalker/SadTalker ..."
  git clone --depth 1 https://github.com/OpenTalker/SadTalker.git $Vendor
} else {
  Write-Host "Vendor already present: $Vendor"
}

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
  Write-Host "Creating .venv_sadtalker (Python 3.10 preferred) ..."
  py -3.10 -m venv $Venv
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.10 not found, trying 3.11 ..."
    py -3.11 -m venv $Venv
  }
}
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Failed to create .venv_sadtalker" }

& $Py -m pip install -U pip setuptools wheel
Write-Host "Installing torch cu124 ..."
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

Write-Host "Installing SadTalker deps (flexible pins for Windows) ..."
# Avoid ancient pins that break on modern pip/torch; keep the functional set.
& $Py -m pip install `
  "numpy<2" `
  "opencv-python-headless<4.12" `
  imageio imageio-ffmpeg `
  "librosa>=0.9" `
  numba resampy pydub scipy `
  kornia tqdm yacs pyyaml joblib `
  "scikit-image>=0.19" `
  safetensors `
  soundfile `
  "mediapipe==0.10.14"
if ($LASTEXITCODE -ne 0) { throw "core deps failed" }

# Face align / optional enhancer — best-effort (basicsr is flaky on Windows).
& $Py -m pip install facexlib gfpgan 2>&1 | Select-Object -Last 20
& $Py -m pip install "av" 2>&1 | Select-Object -Last 10
# gfpgan/basicsr often pulls numpy 2 + breaks opencv — re-pin.
& $Py -m pip install "numpy<2" "opencv-python<4.12" "opencv-python-headless<4.12" -q
# Patch basicsr for torchvision>=0.17 (functional_tensor moved).
$Deg = Join-Path $Venv "Lib\site-packages\basicsr\data\degradations.py"
if (Test-Path $Deg) {
  $c = Get-Content $Deg -Raw
  $c2 = $c -replace 'from torchvision\.transforms\.functional_tensor import rgb_to_grayscale','from torchvision.transforms.functional import rgb_to_grayscale'
  if ($c -ne $c2) { Set-Content -Path $Deg -Value $c2 -NoNewline; Write-Host "Patched basicsr degradations for modern torchvision" }
}

function Get-FileIfMissing([string]$Url, [string]$Out) {
  if ((Test-Path $Out) -and ((Get-Item $Out).Length -gt 1MB)) {
    Write-Host "OK: $Out"
    return
  }
  New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
  Write-Host "Downloading $(Split-Path $Out -Leaf) ..."
  curl.exe -L --fail --retry 3 -o $Out $Url
  if ($LASTEXITCODE -ne 0) { throw "download failed: $Url" }
}

New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null
New-Item -ItemType Directory -Force -Path $GfpDir | Out-Null

Get-FileIfMissing "$Rel/SadTalker_V0.0.2_256.safetensors" (Join-Path $CkptDir "SadTalker_V0.0.2_256.safetensors")
Get-FileIfMissing "$Rel/SadTalker_V0.0.2_512.safetensors" (Join-Path $CkptDir "SadTalker_V0.0.2_512.safetensors")
Get-FileIfMissing "$Rel/mapping_00109-model.pth.tar" (Join-Path $CkptDir "mapping_00109-model.pth.tar")
Get-FileIfMissing "$Rel/mapping_00229-model.pth.tar" (Join-Path $CkptDir "mapping_00229-model.pth.tar")

# Face detection / alignment used by CropAndExtract (even without GFPGAN enhance)
Get-FileIfMissing "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth" (Join-Path $GfpDir "alignment_WFLW_4HG.pth")
Get-FileIfMissing "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" (Join-Path $GfpDir "detection_Resnet50_Final.pth")
Get-FileIfMissing "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" (Join-Path $GfpDir "parsing_parsenet.pth")
# Optional enhancer weight (only if SADTALKER_ENHANCER=gfpgan)
Get-FileIfMissing "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" (Join-Path $GfpDir "GFPGANv1.4.pth")

$EnvFile = Join-Path $Root ".env"
$lines = @(
  "SADTALKER_PYTHON=$Py",
  "SADTALKER_ROOT=$Vendor",
  "SADTALKER_SIZE=256",
  "SADTALKER_PREPROCESS=full",
  "SADTALKER_STILL=1",
  "SADTALKER_BATCH_SIZE=2"
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
Write-Host "  Ctrl+C any running avatar server"
Write-Host "  .\run_server.ps1 -Backend sadtalker"
Write-Host "  Demo: Full image, short /say, then Analytics Capture + score"
Write-Host "SADTALKER_PYTHON=$Py"
Write-Host "Tip: SADTALKER_SIZE=512 for sharper faces (slower). Enhancer off by default."
