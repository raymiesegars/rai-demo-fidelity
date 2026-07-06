#!/usr/bin/env bash
# Fix: onnxruntime-gpu 1.27+ needs libcudart.so.13; RunPod torch cu128 has CUDA 12.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
TARGET_VER="1.20.2"

echo "=============================================="
echo " Fixing onnxruntime for CUDA 12 (RunPod)"
echo " Python: $($PYTHON --version 2>&1)"
echo " Target: onnxruntime-gpu==${TARGET_VER}"
echo "=============================================="

echo "==> Current install (if any)…"
$PYTHON -m pip show onnxruntime-gpu 2>/dev/null || echo "(not installed)"

echo "==> Removing broken onnxruntime packages…"
$PYTHON -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true

echo "==> Installing onnxruntime-gpu==${TARGET_VER}…"
$PYTHON -m pip install --force-reinstall --no-cache-dir "onnxruntime-gpu==${TARGET_VER}"

echo "==> Pinning numpy for numba/torch compatibility…"
$PYTHON -m pip install "numpy>=1.22,<2.5"

echo "==> Verifying import + CUDA provider…"
$PYTHON -c "
import sys
import onnxruntime as ort

ver = ort.__version__
loc = ort.__file__
print('onnxruntime', ver)
print('path', loc)
print('device', ort.get_device())
providers = ort.get_available_providers()
print('providers', providers)

major, minor = (int(x) for x in ver.split('.')[:2])
if (major, minor) >= (1, 27):
    print('ERROR: still on onnxruntime >= 1.27 (needs CUDA 13)', file=sys.stderr)
    sys.exit(1)
if 'CUDAExecutionProvider' not in providers:
    print('ERROR: CUDAExecutionProvider missing', file=sys.stderr)
    sys.exit(1)
print('OK')
"

echo ""
echo "=============================================="
echo " SUCCESS — restart the avatar worker:"
echo "   cd $SCRIPT_DIR"
echo "   bash start_liveportrait.sh"
echo "=============================================="
