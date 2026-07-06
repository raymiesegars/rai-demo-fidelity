#!/bin/bash
# Run once on RunPod pod to enable AVATAR_MODE=wav2lip
set -euo pipefail

WAV2LIP_ROOT="${WAV2LIP_ROOT:-/workspace/Wav2Lip}"

echo "==> Installing Wav2Lip to $WAV2LIP_ROOT"
if [ ! -d "$WAV2LIP_ROOT" ]; then
  git clone https://github.com/Rudrabha/Wav2Lip.git "$WAV2LIP_ROOT"
fi

cd "$WAV2LIP_ROOT"
pip install -q -r requirements.txt
pip install -q librosa==0.10.1

mkdir -p checkpoints
if [ ! -f checkpoints/wav2lip_gan.pth ]; then
  echo "==> Downloading Wav2Lip GAN checkpoint (~150MB)…"
  wget -q -O checkpoints/wav2lip_gan.pth \
    "https://github.com/Rudrabha/Wav2Lip/releases/download/v0.1/wav2lip_gan.pth" \
    || wget -q -O checkpoints/wav2lip_gan.pth \
    "https://huggingface.co/spaces/akhaliq/Wav2Lip/resolve/main/wav2lip_gan.pth"
fi

echo "==> Wav2Lip ready. Set AVATAR_MODE=wav2lip and restart python main.py"
