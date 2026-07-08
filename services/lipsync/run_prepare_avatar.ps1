# Prepare Alan avatar for MuseTalk real-time inference (one-time, ~2-5 min)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$MuseTalk = Join-Path $Root "vendor\MuseTalk"
$Ffmpeg = Get-ChildItem -Path (Join-Path $Root "tools\ffmpeg") -Recurse -Filter ffmpeg.exe | Select-Object -First 1
$AlanVideo = Join-Path $Root "assets\alan-loop.mp4"

if (-not (Test-Path $AlanVideo)) {
    Write-Error @"
alan-loop.mp4 not found at:
  $AlanVideo

Copy it from RunPod:
  /workspace/rai-demo-fidelity/services/avatar-worker/assets/alan-loop.mp4
  or /workspace/assets/alan-loop.mp4

RunPod web terminal:
  # if you can access the file, use RunPod 'Connect' file browser or:
  base64 /workspace/assets/alan-loop.mp4 | head
"@
}

$env:Path = "$(Join-Path $Root '.venv\Scripts');$($Ffmpeg.DirectoryName);$env:Path"
Set-Location $MuseTalk

# Convert loop to 25fps if needed (MuseTalk trains at 25fps)
$Fps25 = Join-Path $Root "assets\alan-loop-25fps.mp4"
if (-not (Test-Path $Fps25)) {
    Write-Host "==> Converting alan-loop to 25fps..."
    ffmpeg -y -i $AlanVideo -r 25 -c:v libx264 -pix_fmt yuv420p $Fps25
}

# Patch config to use 25fps video
$ConfigContent = @"
alan:
  preparation: true
  bbox_shift: 5
  video_path: "$($Fps25 -replace '\\','/')"
  audio_clips:
    sample: "$(Join-Path $Root 'assets\sample.wav' -replace '\\','/')"
"@
$ConfigPath = Join-Path $Root "configs\alan_realtime.yaml"
$ConfigContent | Set-Content -Path $ConfigPath -Encoding UTF8

Write-Host "==> Preparing avatar (face detect + latents)..."
python -m scripts.realtime_inference `
  --inference_config "$ConfigPath" `
  --result_dir results\realtime `
  --unet_model_path models\musetalkV15\unet.pth `
  --unet_config models\musetalkV15\musetalk.json `
  --version v15 `
  --fps 25 `
  --ffmpeg_path "$($Ffmpeg.DirectoryName)"

Write-Host ""
Write-Host "Avatar prepared. Set preparation: false in configs\alan_realtime.yaml for next runs."
