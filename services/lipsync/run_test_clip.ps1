# Generate a test lip-sync clip (requires prepared avatar + sample audio)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$MuseTalk = Join-Path $Root "vendor\MuseTalk"
$Ffmpeg = Get-ChildItem -Path (Join-Path $Root "tools\ffmpeg") -Recurse -Filter ffmpeg.exe | Select-Object -First 1
$ConfigPath = Join-Path $Root "configs\alan_realtime.yaml"
$SampleWav = Join-Path $Root "assets\sample.wav"

# Use MuseTalk bundled sample if we don't have one yet
if (-not (Test-Path $SampleWav)) {
    $Bundled = Join-Path $MuseTalk "data\audio\eng.wav"
    if (Test-Path $Bundled) {
        Copy-Item $Bundled $SampleWav
    } else {
        Write-Error "No sample.wav in assets/. Add any short speech WAV to assets\sample.wav"
    }
}

# Ensure preparation is false for inference-only run
(Get-Content $ConfigPath) -replace 'preparation: true', 'preparation: false' | Set-Content $ConfigPath

$env:Path = "$(Join-Path $Root '.venv\Scripts');$($Ffmpeg.DirectoryName);$env:Path"
Set-Location $MuseTalk

Write-Host "==> Running MuseTalk inference..."
python -m scripts.realtime_inference `
  --inference_config "$ConfigPath" `
  --result_dir results\realtime `
  --unet_model_path models\musetalkV15\unet.pth `
  --unet_config models\musetalkV15\musetalk.json `
  --version v15 `
  --fps 25 `
  --ffmpeg_path "$($Ffmpeg.DirectoryName)"

$Out = Join-Path $MuseTalk "results\v15\avatars\alan\vid_output\sample.mp4"
if (Test-Path $Out) {
    $Dest = Join-Path $Root "results\test_sample.mp4"
    New-Item -ItemType Directory -Force -Path (Split-Path $Dest) | Out-Null
    Copy-Item $Out $Dest -Force
    Write-Host "Saved: $Dest"
} else {
    Write-Host "Check output under: $MuseTalk\results\v15\avatars\alan\vid_output\"
}
