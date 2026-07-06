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
set_kv ANIMATION_ENERGY_THRESHOLD 20
set_kv MOUTH_SENSITIVITY 900
set_kv MOUTH_MAX_STRETCH 0.22

echo "Updated $ENV_FILE for speech-reactive mode:"
grep -E '^(AVATAR_MODE|MOUTH_|ENERGY_|ANIMATION_)' "$ENV_FILE"
echo ""
echo "Start: export \$(grep -v '^#' .env | xargs) && python main.py"
echo "Expect: Mouth drive: reactive — instant audio-synced mouth on loop"
