#!/usr/bin/env bash
# Safe demo defaults: ping-pong idle loop, no broken mouth hacks.
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

set_kv AVATAR_MODE mock
set_kv MOUTH_DRIVE idle

echo "Updated $ENV_FILE for clean idle-loop demo:"
grep -E '^(AVATAR_MODE|MOUTH_DRIVE)=' "$ENV_FILE"
echo ""
echo "Alan will loop naturally while the agent speaks. Lip sync comes next (see README.md)."
