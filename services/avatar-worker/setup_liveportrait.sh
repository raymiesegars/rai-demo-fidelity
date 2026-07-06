#!/usr/bin/env bash
# FasterLivePortrait + JoyVASA — real lip sync on Alan's loop video.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLP_ROOT="${FLP_ROOT:-/workspace/FasterLivePortrait}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$FLP_ROOT/checkpoints}"

hf_download_repo() {
  local repo="$1"
  local dest="$2"
  if command -v hf &>/dev/null; then
    hf download "$repo" --local-dir "$dest"
  else
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${repo}', local_dir='${dest}')"
  fi
}

echo "=============================================="
echo " FasterLivePortrait + JoyVASA setup"
echo " Expect 15–30 minutes. Do not close this tab."
echo "=============================================="
echo ""

if [[ ! -d "$FLP_ROOT" ]]; then
  echo "==> [1/5] Cloning FasterLivePortrait…"
  git clone https://github.com/warmshao/FasterLivePortrait.git "$FLP_ROOT"
else
  echo "==> [1/5] FasterLivePortrait already cloned"
fi

cd "$FLP_ROOT"

echo "==> [2/5] Installing Python packages (2–5 min)…"
pip install -r requirements.txt
pip install -U huggingface_hub onnxruntime-gpu omegaconf

mkdir -p "$CHECKPOINT_DIR"

echo ""
echo "==> [3/5] Downloading LivePortrait ONNX (~2GB)…"
hf_download_repo warmshao/FasterLivePortrait "$CHECKPOINT_DIR"

echo ""
echo "==> [4/5] Downloading JoyVASA…"
hf_download_repo jdh-algo/JoyVASA "$CHECKPOINT_DIR/JoyVASA"

echo ""
echo "==> [5/5] Downloading Chinese-HuBERT…"
hf_download_repo TencentGameMate/chinese-hubert-base "$CHECKPOINT_DIR/chinese-hubert-base"

echo ""
echo "==> Verifying required files…"
bash "$SCRIPT_DIR/verify_liveportrait.sh" || {
  echo "ERROR: Setup incomplete — some model files are missing."
  exit 1
}

echo ""
echo "=============================================="
echo " SUCCESS — models ready at $CHECKPOINT_DIR"
echo "=============================================="
echo ""
echo "Now run these commands ONE AT A TIME:"
echo "  cd /workspace/rai-demo-fidelity"
echo "  git pull"
echo "  cd services/avatar-worker"
echo "  bash apply_liveportrait_env.sh"
echo "  export \$(grep -v '^#' .env | xargs)"
echo "  python main.py"
echo ""
echo "You MUST see: Mouth drive: liveportrait"
echo "If you see 'Avatar mode: mock' lip sync is NOT on."
