#!/usr/bin/env bash
# Exit 0 if LivePortrait models are present for avatar-worker.
set -euo pipefail

FLP_ROOT="${FLP_ROOT:-/workspace/FasterLivePortrait}"
CKPT="${CHECKPOINT_DIR:-$FLP_ROOT/checkpoints}"

find_one() {
  local pattern="$1"
  local found
  found="$(find "$CKPT" -path "$pattern" -type f 2>/dev/null | head -1)"
  if [[ -n "$found" ]]; then
    echo "  OK  $found"
  else
    echo "  MISSING  $pattern (under $CKPT)"
    missing=1
  fi
}

missing=0
echo "Checking LivePortrait install at $FLP_ROOT …"
find_one "*/liveportrait_onnx/warping_spade.onnx"
find_one "*/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt"
find_one "*/chinese-hubert-base/config.json"

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi
echo "All required model files found."
