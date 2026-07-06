#!/usr/bin/env bash
# FasterLivePortrait + JoyVASA — real lip sync on Alan's loop video.
# Run once on RunPod (RTX 4090). Expect ~15–30 min for downloads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLP_ROOT="${FLP_ROOT:-/workspace/FasterLivePortrait}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$FLP_ROOT/checkpoints}"

hf_download() {
  if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$@"
  elif command -v hf &>/dev/null; then
    hf download "$@"
  else
    python -m huggingface_hub.cli.huggingface_cli download "$@"
  fi
}

echo "==> FasterLivePortrait + JoyVASA setup"
echo "    Install root: $FLP_ROOT"
echo "    (This takes 15–30 minutes — downloads are large.)"
echo ""

if [[ ! -d "$FLP_ROOT" ]]; then
  echo "==> Cloning FasterLivePortrait…"
  git clone https://github.com/warmshao/FasterLivePortrait.git "$FLP_ROOT"
else
  echo "==> FasterLivePortrait already cloned"
fi

cd "$FLP_ROOT"

echo "==> Installing Python dependencies (2–5 min, you will see pip output)…"
pip install -r requirements.txt
pip install -U huggingface_hub onnxruntime-gpu
# Ensure CLI is on PATH (newer hub versions use 'hf' instead of extras)
if ! command -v huggingface-cli &>/dev/null && ! command -v hf &>/dev/null; then
  pip install -U "huggingface_hub[cli]" || pip install -U huggingface_hub
fi

mkdir -p "$CHECKPOINT_DIR"

echo ""
echo "==> Downloading LivePortrait ONNX checkpoints (~2GB)…"
hf_download warmshao/FasterLivePortrait --local-dir "$CHECKPOINT_DIR"

echo ""
echo "==> Downloading JoyVASA…"
hf_download jdh-algo/JoyVASA --local-dir "$CHECKPOINT_DIR/JoyVASA"

echo ""
echo "==> Downloading Chinese-HuBERT…"
hf_download TencentGameMate/chinese-hubert-base --local-dir "$CHECKPOINT_DIR/chinese-hubert-base"

if command -v trtexec &>/dev/null || [[ -n "${TENSORRT_HOME:-}" ]]; then
  echo ""
  echo "==> TensorRT found — converting ONNX to TRT (optional, can take a while)…"
  if [[ -f scripts/all_onnx2trt.sh ]]; then
    bash scripts/all_onnx2trt.sh || echo "WARN: TRT conversion failed — onnx mode still works"
  fi
else
  echo ""
  echo "==> TensorRT not found — skipping TRT conversion (onnx mode is fine)."
fi

echo ""
echo "==> Done! Models are in $CHECKPOINT_DIR"
echo ""
echo "Next:"
echo "  cd /workspace/rai-demo-fidelity/services/avatar-worker"
echo "  bash apply_liveportrait_env.sh"
echo "  export \$(grep -v '^#' .env | xargs)"
echo "  python main.py"
