#!/usr/bin/env bash
# Updates avatar-worker .env for lip-patch composite mode (safe to re-run).
set -euo pipefail
cd "$(dirname "$0")"
ENV_FILE=".env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "No .env found — copying from .env.example (add your LiveKit keys!)"
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

set_kv AVATAR_MODE wav2lip
set_kv MOUTH_DRIVE composite
set_kv LIP_SYNC_FIRST_CHUNK_SEC 0.2
set_kv LIP_SYNC_CHUNK_SEC 0.4
set_kv LIP_PATCH_STALE_SEC 0.25
set_kv LIP_RECT_Y_SHIFT -0.04
set_kv WAV2LIP_PLAYBACK 0

echo "Updated $ENV_FILE for lip composite mode:"
grep -E '^(AVATAR_MODE|MOUTH_DRIVE|LIP_SYNC_|WAV2LIP_PLAYBACK)=' "$ENV_FILE"
