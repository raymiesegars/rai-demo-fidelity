#!/bin/bash
# Run once on RunPod pod to enable AVATAR_MODE=wav2lip
# Wav2Lip's upstream requirements.txt targets Python 3.6 — we install modern deps instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAV2LIP_ROOT="${WAV2LIP_ROOT:-/workspace/Wav2Lip}"

echo "==> Installing Wav2Lip to $WAV2LIP_ROOT"
if [ ! -d "$WAV2LIP_ROOT" ]; then
  git clone https://github.com/Rudrabha/Wav2Lip.git "$WAV2LIP_ROOT"
fi

cd "$WAV2LIP_ROOT"
mkdir -p temp checkpoints

echo "==> Installing Python 3.12-compatible dependencies"
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "    Installing PyTorch with CUDA…"
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
  echo "    PyTorch + CUDA already available"
fi

pip install -q "librosa>=0.10,<0.11" tqdm numba scipy opencv-python

echo "==> Applying compatibility patches for Python 3.12 / torch 2.x"
python "$SCRIPT_DIR/patch_wav2lip.py" "$WAV2LIP_ROOT"

if [ ! -f checkpoints/wav2lip_gan.pth ]; then
  echo "==> Downloading Wav2Lip GAN checkpoint (~150MB)…"
  wget -q -O checkpoints/wav2lip_gan.pth \
    "https://github.com/Rudrabha/Wav2Lip/releases/download/v0.1/wav2lip_gan.pth" \
    || wget -q -O checkpoints/wav2lip_gan.pth \
    "https://huggingface.co/spaces/akhaliq/Wav2Lip/resolve/main/wav2lip_gan.pth"
fi

echo "==> Verifying Wav2Lip install…"
python -c "
from pathlib import Path
root = Path('$WAV2LIP_ROOT')
assert (root / 'inference.py').is_file()
assert (root / 'checkpoints/wav2lip_gan.pth').is_file()
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
"

echo "==> Wav2Lip ready. Set AVATAR_MODE=wav2lip and restart: python main.py"
