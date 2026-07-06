#!/bin/bash
# Run once on RunPod pod to enable AVATAR_MODE=wav2lip
# Wav2Lip's upstream requirements.txt targets Python 3.6 — we install modern deps instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAV2LIP_ROOT="${WAV2LIP_ROOT:-/workspace/Wav2Lip}"
CHECKPOINT="${WAV2LIP_ROOT}/checkpoints/wav2lip_gan.pth"
# Official release was ~139MB; HF mirrors are often ~436MB
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

pip install -q "librosa>=0.10,<0.11" tqdm numba scipy opencv-python

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
torch.load(str(p), map_location='cpu', weights_only=False)
print('checkpoint ok:', p.stat().st_size, 'bytes')
"
}

download_checkpoint() {
  echo "==> Downloading Wav2Lip GAN checkpoint (100–450MB)…"
  rm -f "$CHECKPOINT"

  local urls=(
    "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth"
    "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth"
  )

  for url in "${urls[@]}"; do
    echo "    Trying $url"
    if curl -fL --retry 2 --retry-delay 3 --progress-bar -o "$CHECKPOINT" "$url"; then
      return 0
    fi
    rm -f "$CHECKPOINT"
  done

  echo "    curl failed — trying gdown (Google Drive)…"
  pip install -q gdown
  gdown "https://drive.google.com/uc?id=15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk" \
    -O "$CHECKPOINT" \
    || {
      gdown --folder "https://drive.google.com/drive/folders/1I-0dNLfFOSFwrfqjNa-SXuwaURHE5K4k" \
        -O /tmp/wav2lip_ckpt
      cp /tmp/wav2lip_ckpt/wav2lip_gan.pth "$CHECKPOINT"
    }
}

if ! verify_checkpoint 2>/dev/null; then
  echo "    Checkpoint missing or corrupt — downloading…"
  download_checkpoint
  verify_checkpoint
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
