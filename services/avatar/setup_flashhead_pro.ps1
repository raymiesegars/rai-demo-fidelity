# Download SoulX-FlashHead Pro weights (Model_Pro + Wan VAE).
# Lite (Model_Lite + VAE_LTX) is assumed already present from the original setup.
# Run from services\avatar:
#   .\setup_flashhead_pro.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
  throw "Missing $Py — create the main avatar .venv first (FlashHead Lite setup)."
}

$Dest = Join-Path $Root "models\SoulX-FlashHead-1_3B"
$Pro = Join-Path $Dest "Model_Pro"
$Vae = Join-Path $Dest "VAE_Wan\Wan2.1_VAE.pth"

Write-Host "==> FlashHead Pro weights -> $Dest"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
& $Py -m pip install -q "huggingface_hub" 2>&1 | Out-Null

$needPro = -not (Test-Path (Join-Path $Pro "config.json"))
$needVae = -not (Test-Path $Vae)
if (-not $needPro -and -not $needVae) {
  Write-Host "OK: Model_Pro and VAE_Wan already present."
  Write-Host "Start with: .\run_server.ps1 -Backend flashhead-pro -KillExisting"
  exit 0
}

Write-Host "Downloading Model_Pro + VAE_Wan from Soul-AILab/SoulX-FlashHead-1_3B ..."
& $Py -c @"
from huggingface_hub import snapshot_download
snapshot_download(
    'Soul-AILab/SoulX-FlashHead-1_3B',
    local_dir=r'$Dest',
    allow_patterns=['Model_Pro/*', 'VAE_Wan/*'],
)
print('download complete')
"@
if ($LASTEXITCODE -ne 0) {
  throw "HF download failed. Try: .\.venv\Scripts\huggingface-cli.exe login"
}

if (-not (Test-Path (Join-Path $Pro "config.json"))) {
  throw "Model_Pro/config.json still missing after download"
}
if (-not (Test-Path $Vae)) {
  throw "VAE_Wan/Wan2.1_VAE.pth still missing after download"
}

Write-Host ""
Write-Host "Pro weights ready."
Write-Host "  .\run_server.ps1 -Backend flashhead-pro -KillExisting"
Write-Host "Expect ~11 FPS on a 4090 (better look than Lite, not chat-realtime)."
