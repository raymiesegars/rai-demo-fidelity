# GPU lip-sync upgrade (RunPod)

## Quick start — Wav2Lip (recommended for demo)

On your RunPod SSH terminal:

```bash
cd /workspace/rai-demo-fidelity/services/avatar-worker
git pull
bash setup_wav2lip.sh
```

Update `.env`:

```bash
AVATAR_MODE=wav2lip
TARGET_FPS=25
WAV2LIP_ROOT=/workspace/Wav2Lip
WAV2LIP_CHECKPOINT=/workspace/Wav2Lip/checkpoints/wav2lip_gan.pth
```

Restart the avatar worker:

```bash
export $(grep -v '^#' .env | xargs)
python main.py
```

You should see: `Wav2Lip lip sync enabled` and `Subscribed to agent audio for lip sync`.

When Alan speaks, logs will show:
- `Lip-syncing X.XXs of agent audio…`
- `Queued N lip-synced frames for playback`

**Note:** First lip-sync per utterance takes ~2–5 seconds on a 4090 (batch inference). Idle loop plays while processing, then lip-synced frames play.

---

## How it works

1. Avatar worker streams ping-pong idle loop
2. Subscribes to the **agent's TTS audio track** in the LiveKit room
3. When agent finishes a sentence, runs **Wav2Lip** on `alan-loop.mp4` + that audio
4. Plays lip-synced frames, then returns to idle loop

---

## Premium path — FasterLivePortrait + JoyVASA

For real-time lip-only drive (no batch delay), install FasterLivePortrait + JoyVASA:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token $HUGGINGFACE_TOKEN

git clone https://github.com/warmshao/FasterLivePortrait.git /workspace/FasterLivePortrait
cd /workspace/FasterLivePortrait
pip install -r requirements.txt
huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints
huggingface-cli download jdh-algo/JoyVASA --local-dir ./checkpoints/JoyVASA
```

Use **lip-only** animation region when source is `alan-loop.mp4`.

---

## Modes

| AVATAR_MODE | Behavior |
|-------------|----------|
| `mock` | Idle loop only — no lip sync |
| `wav2lip` | Idle loop + Wav2Lip on each agent utterance |
