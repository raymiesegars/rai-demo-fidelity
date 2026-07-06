#!/usr/bin/env bash
# One-shot: fix onnxruntime, apply env, preflight, start worker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> [1/4] Fix onnxruntime for CUDA 12…"
bash fix_onnx_cuda.sh

echo "==> [2/4] Patch FasterLivePortrait for RunPod…"
python patch_flp_compat.py "${FLP_ROOT:-/workspace/FasterLivePortrait}"

echo "==> [3/4] Apply LivePortrait .env…"
bash apply_liveportrait_env.sh

echo "==> [4/4] Preflight + start worker…"
set -a
# shellcheck disable=SC1091
source <(grep -v '^#' .env | sed 's/^/export /')
set +a
python -c "from liveportrait_engine import preflight_liveportrait; preflight_liveportrait()"
python main.py
