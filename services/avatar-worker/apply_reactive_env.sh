#!/usr/bin/env bash
# Recommended demo: ping-pong loop + instant speech-reactive mouth (no LivePortrait).
set -euo pipefail
cd "$(dirname "$0")"
ENV_FILE=".env"

if [[ ! -f "$ENV_FILE" ]]; then
  cp .env.example "$ENV_FILE"
fi

set_kv() {
  local key="$1" val="$2"
  if grep -q "^${key}=" "$ENV_FILE"; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
  else
    echo "${key}=${val}" >> "$ENV_FILE"
  fi
}

set_kv AVATAR_MODE reactive
set_kv MOUTH_DRIVE reactive
set_kv ENERGY_MOUTH 0
set_kv ANIMATION_ENERGY_THRESHOLD 8
set_kv MOUTH_SENSITIVITY 350
set_kv MOUTH_OPEN_AMOUNT 1.15
set_kv MOUTH_UPPER_LIFT 0.08
set_kv MOUTH_LOWER_DROP 0.30
set_kv MOUTH_CORNER_MOVE 0.55
set_kv MOUTH_ATTACK 0.68
set_kv MOUTH_DECAY 0.38
set_kv MOUTH_SILENCE_CUTOFF_SEC 0.10
set_kv MOUTH_LIP_LINE 0.40
set_kv MOUTH_RECT_X_LEFT 0.24
set_kv MOUTH_RECT_X_RIGHT 0.76
set_kv MOUTH_RECT_TOP 0.68
set_kv MOUTH_RECT_BOTTOM 0.92
set_kv MOUTH_RECT_Y_SHIFT 0.0
set_kv MOUTH_DEBUG 0

for old_key in MOUTH_MAX_STRETCH LIP_RECT_X_LEFT LIP_RECT_X_RIGHT LIP_RECT_TOP \
  LIP_RECT_BOTTOM LIP_RECT_Y_SHIFT MOUTH_SEC_PER_CHAR MOUTH_UTTERANCE_PAD_SEC; do
  sed -i "/^${old_key}=/d" "$ENV_FILE" 2>/dev/null || true
done

echo "Updated $ENV_FILE for speech-reactive mode:"
grep -E '^(AVATAR_MODE|MOUTH_|ENERGY_|ANIMATION_)' "$ENV_FILE"
echo ""
echo "Start: export \$(grep -v '^#' .env | xargs) && python main.py"
echo "Debug: export MOUTH_DEBUG=1 — green=ROI, cyan=lip line, red/blue=upper/lower lip"
