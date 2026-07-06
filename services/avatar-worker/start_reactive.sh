#!/usr/bin/env bash
# Recommended demo: ping-pong loop + instant speech-reactive mouth (no ML per frame).
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f speech_reactive.py ]]; then
  echo "Missing speech_reactive.py — git pull or copy from repo first."
  exit 1
fi
if ! grep -q SpeechReactiveMouth main.py 2>/dev/null; then
  echo "main.py is outdated (no reactive mode). Run: git pull"
  exit 1
fi

bash apply_reactive_env.sh
set -a
# shellcheck disable=SC1091
source .env
set +a

echo "Starting avatar worker (reactive mouth)…"
exec python main.py
