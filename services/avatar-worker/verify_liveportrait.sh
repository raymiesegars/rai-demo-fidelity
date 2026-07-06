#!/usr/bin/env bash
# Exit 0 if LivePortrait models are present for avatar-worker.
set -euo pipefail

FLP_ROOT="${FLP_ROOT:-/workspace/FasterLivePortrait}"
CKPT="${CHECKPOINT_DIR:-$FLP_ROOT/checkpoints}"

need() {
  local path="$1"
  if [[ -f "$path" ]]; then
    echo "  OK  $path"
  else
    echo "  MISSING  $path"
    missing=1
  fi
}

missing=0
echo "Checking LivePortrait install at $FLP_ROOT …"
need "$CKPT/liveportrait_onnx/warping_spade.onnx"
need "$CKPT/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt"
need "$CKPT/chinese-hubert-base/config.json"

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi
echo "All required model files found."
