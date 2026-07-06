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
set_kv MOUTH_SENSITIVITY 400
set_kv MOUTH_MAX_STRETCH 0.20
set_kv MOUTH_LIP_GAP 0.35
set_kv MOUTH_RECT_LEFT 0.36
set_kv MOUTH_RECT_RIGHT 0.64
set_kv MOUTH_RECT_TOP 0.70
set_kv MOUTH_RECT_BOTTOM 0.84
set_kv MOUTH_RECT_Y_SHIFT 0.0

echo "Updated $ENV_FILE for speech-reactive mode:"
grep -E '^(AVATAR_MODE|MOUTH_|ENERGY_|ANIMATION_)' "$ENV_FILE"
echo ""
echo "Start: export \$(grep -v '^#' .env | xargs) && python main.py"
echo "Expect: Mouth drive: reactive — instant audio-synced mouth on loop"
