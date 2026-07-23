# Setup Wav2Lip for local bench (still-image + locked composite).
# Run from services\avatar:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_wav2lip.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Vendor = Join-Path $Root "vendor\Wav2Lip"
$CkptDir = Join-Path $Root "models"
$Ckpt = Join-Path $CkptDir "wav2lip_gan.pth"
$S3fd = Join-Path $Vendor "face_detection\detection\sfd\s3fd.pth"
$Py = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "==> Wav2Lip setup"
if (-not (Test-Path $Py)) { throw "Create services/avatar/.venv first (see README quick start)" }

New-Item -ItemType Directory -Force -Path (Join-Path $Root "vendor") | Out-Null
New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null

if (-not (Test-Path (Join-Path $Vendor ".git"))) {
  Write-Host "Cloning Rudrabha/Wav2Lip ..."
  git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git $Vendor
} else {
  Write-Host "Vendor already present: $Vendor"
}

Write-Host "Patching Wav2Lip audio.py for modern librosa ..."
& $Py (Join-Path $Root "scripts\patch_wav2lip_librosa.py") $Vendor

Write-Host "Installing Wav2Lip Python deps into .venv ..."
& $Py -m pip install "librosa>=0.10" opencv-python soundfile tqdm numba huggingface_hub -q
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

if (-not (Test-Path $Ckpt) -or (Get-Item $Ckpt).Length -lt 100MB) {
  Write-Host "Downloading wav2lip_gan.pth from Nekochu/Wav2Lip (~436 MB) ..."
  $env:CKPT_OUT = $Ckpt
  & $Py -c "from huggingface_hub import hf_hub_download; import shutil, os; p=hf_hub_download('Nekochu/Wav2Lip','wav2lip_gan.pth'); shutil.copy2(p, os.environ['CKPT_OUT']); print('saved', os.environ['CKPT_OUT'], os.path.getsize(os.environ['CKPT_OUT']))"
  if ($LASTEXITCODE -ne 0) { throw "Checkpoint download failed" }
} else {
  Write-Host "Checkpoint OK: $Ckpt"
}

# s3fd is optional here — our backend uses mediapipe for face boxes.
# Skip quietly if absent (old script tried Nekochu/s3fd.pth which 404s).
if (Test-Path $S3fd) {
  Write-Host "s3fd present: $S3fd"
} else {
  Write-Host "s3fd.pth not required (mediapipe face detect) — skipping"
}

$env:CKPT_OUT = $Ckpt
& $Py -c "import torch, os; p=os.environ['CKPT_OUT']; obj=torch.load(p, map_location='cpu', weights_only=False); assert isinstance(obj, dict) and 'state_dict' in obj, 'bad checkpoint'; print('checkpoint valid:', p)"
if ($LASTEXITCODE -ne 0) { throw "Checkpoint validation failed" }

Write-Host ""
Write-Host "SUCCESS. Next steps:"
Write-Host "  1) Free port 8100 if something is already using it"
Write-Host "  2) .\run_server.ps1 -Backend wav2lip"
Write-Host "  3) http://127.0.0.1:8100 → Demo → Full image → talk"
Write-Host "  4) Analytics → Capture live stats → score fidelity"
