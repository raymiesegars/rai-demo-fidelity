#!/usr/bin/env bash
# Use LivePortrait lip sync when models are installed; falls back to idle loop otherwise.
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

set_kv AVATAR_MODE liveportrait
set_kv MOUTH_DRIVE idle
set_kv FLP_ROOT /workspace/FasterLivePortrait
set_kv FLP_ANIMATION_REGION exp

echo "Updated $ENV_FILE for LivePortrait mode:"
grep -E '^(AVATAR_MODE|MOUTH_DRIVE|FLP_)=' "$ENV_FILE"
echo ""
echo "Install models first: bash setup_liveportrait.sh"
