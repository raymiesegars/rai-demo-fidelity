#!/usr/bin/env bash
# FasterLivePortrait + JoyVASA — the path to real lip sync on Alan's loop video.
# Run once on RunPod (RTX 4090). Expect ~15–30 min for model downloads + TRT conversion.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLP_ROOT="${FLP_ROOT:-/workspace/FasterLivePortrait}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$FLP_ROOT/checkpoints}"

echo "==> FasterLivePortrait + JoyVASA setup"
echo "    Install root: $FLP_ROOT"

if [[ ! -d "$FLP_ROOT" ]]; then
  git clone https://github.com/warmshao/FasterLivePortrait.git "$FLP_ROOT"
fi

cd "$FLP_ROOT"
pip install -q -r requirements.txt
pip install -q -U "huggingface_hub[cli]" onnxruntime-gpu

mkdir -p "$CHECKPOINT_DIR"

echo "==> Downloading LivePortrait ONNX checkpoints (~2GB)…"
huggingface-cli download warmshao/FasterLivePortrait --local-dir "$CHECKPOINT_DIR"

echo "==> Downloading JoyVASA + Chinese-HuBERT…"
huggingface-cli download jdh-algo/JoyVASA --local-dir "$CHECKPOINT_DIR/JoyVASA"
huggingface-cli download TencentGameMate/chinese-hubert-base --local-dir "$CHECKPOINT_DIR/chinese-hubert-base"

if command -v trtexec &>/dev/null || [[ -n "${TENSORRT_HOME:-}" ]]; then
  echo "==> TensorRT found — converting ONNX to TRT (recommended for real-time)…"
  if [[ -f scripts/all_onnx2trt.sh ]]; then
    bash scripts/all_onnx2trt.sh || echo "WARN: TRT conversion failed — try onnx mode first"
  fi
else
  echo "==> TensorRT not found — you can use onnx mode in webui (slower)."
  echo "    For production lip sync, install TensorRT 8.x and re-run this script."
fi

cat <<EOF

==> Models downloaded.

Next steps (manual test in FasterLivePortrait web UI):
  cd $FLP_ROOT
  python webui.py --mode onnx

In the UI:
  - Source: your alan-loop.mp4 (or a still from the loop)
  - Drive: Audio tab + agent TTS wav file
  - Animation region: mouth / lips only (important for video sources)

When that looks good, we wire AVATAR_MODE=liveportrait in avatar-worker (coming soon).

EOF
