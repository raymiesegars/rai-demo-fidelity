#!/bin/bash
# Run once on RunPod pod to enable AVATAR_MODE=wav2lip
# Wav2Lip's upstream requirements.txt targets Python 3.6 — we install modern deps instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAV2LIP_ROOT="${WAV2LIP_ROOT:-/workspace/Wav2Lip}"
CHECKPOINT="${WAV2LIP_ROOT}/checkpoints/wav2lip_gan.pth"
# HuggingFace checkpoint with state_dict is ~436MB; Google Drive ~140MB is TorchScript (wrong).
MIN_BYTES=100000000

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

pip install -q "librosa>=0.10,<0.11" tqdm numba scipy opencv-python huggingface_hub

echo "==> Applying compatibility patches for Python 3.12 / torch 2.x"
python "$SCRIPT_DIR/patch_wav2lip.py" "$WAV2LIP_ROOT"

verify_checkpoint() {
  python -c "
from pathlib import Path
import sys
p = Path('$CHECKPOINT')
if not p.is_file() or p.stat().st_size < $MIN_BYTES:
    print('bad size:', p.stat().st_size if p.is_file() else 'missing')
    sys.exit(1)
import torch
obj = torch.load(str(p), map_location='cpu', weights_only=False)
if not (isinstance(obj, dict) and 'state_dict' in obj):
    print('wrong format:', type(obj).__name__, '(need dict with state_dict — NOT Google Drive TorchScript)')
    sys.exit(1)
print('checkpoint ok:', p.stat().st_size, 'bytes, state_dict present')
"
}

download_checkpoint() {
  echo "==> Downloading Wav2Lip GAN checkpoint (~436MB, must have state_dict)…"
  rm -f "$CHECKPOINT"

  local urls=(
    "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth"
    "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth"
  )

  for url in "${urls[@]}"; do
    echo "    Trying $url"
    if curl -fL --retry 2 --retry-delay 3 --progress-bar -o "$CHECKPOINT" "$url"; then
      if verify_checkpoint 2>/dev/null; then
        return 0
      fi
      echo "    Downloaded file failed validation — trying next mirror…"
    fi
    rm -f "$CHECKPOINT"
  done

  echo "    Trying huggingface-cli…"
  huggingface-cli download Nekochu/Wav2Lip wav2lip_gan.pth \
    --local-dir "$(dirname "$CHECKPOINT")" --local-dir-use-symlinks False
  if [ -f "$(dirname "$CHECKPOINT")/wav2lip_gan.pth" ]; then
    mv -f "$(dirname "$CHECKPOINT")/wav2lip_gan.pth" "$CHECKPOINT"
    verify_checkpoint
    return 0
  fi

  echo "ERROR: Could not download a valid wav2lip_gan.pth with state_dict."
  echo "Do NOT use the Google Drive file — it is TorchScript and breaks inference."
  exit 1
}

if ! verify_checkpoint 2>/dev/null; then
  echo "    Checkpoint missing or wrong format — downloading…"
  download_checkpoint
fi

echo "==> Verifying Wav2Lip install…"
python -c "
from pathlib import Path
root = Path('$WAV2LIP_ROOT')
assert (root / 'inference.py').is_file()
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
"

echo "==> Wav2Lip ready. Set AVATAR_MODE=wav2lip and restart: python main.py"
