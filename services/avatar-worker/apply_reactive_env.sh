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
set_kv MOUTH_SEC_PER_CHAR 0.068
set_kv MOUTH_UTTERANCE_PAD_SEC 0.55
set_kv MOUTH_SPEAKER_EXTEND_SEC 0.55
set_kv MOUTH_AUDIO_EXTEND_SEC 0.30
set_kv LIP_RECT_Y_SHIFT -0.04
set_kv MOUTH_DEBUG 0

for old_key in MOUTH_BRIGHTNESS MOUTH_MIN_DROP_PX MOUTH_RECT_LEFT MOUTH_RECT_RIGHT \
  MOUTH_RECT_TOP MOUTH_RECT_BOTTOM MOUTH_RECT_Y_SHIFT MOUTH_WARMTH MOUTH_JAW_DROP \
  MOUTH_RECT_WIDTH MOUTH_LIP_GAP; do
  if [[ "$old_key" == "MOUTH_RECT_Y_SHIFT" ]]; then continue; fi
  sed -i "/^${old_key}=/d" "$ENV_FILE" 2>/dev/null || true
done

echo "Updated $ENV_FILE for speech-reactive mode:"
grep -E '^(AVATAR_MODE|MOUTH_|ENERGY_|ANIMATION_|LIP_RECT_)' "$ENV_FILE"
echo ""
echo "Start: export \$(grep -v '^#' .env | xargs) && python main.py"
echo "Tip: export MOUTH_DEBUG=1 to outline the lip mask"
