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
MOUTH_DRIVE=composite
LIP_SYNC_CHUNK_SEC=1.0
TARGET_FPS=25
WAV2LIP_ROOT=/workspace/Wav2Lip
WAV2LIP_CHECKPOINT=/workspace/Wav2Lip/checkpoints/wav2lip_gan.pth
```

Restart the avatar worker:

```bash
export $(grep -v '^#' .env | xargs)
python main.py
```

You should see: `Mouth drive: composite` and `Wav2Lip engine ready`.

When Alan speaks, logs will show:
- `Wav2Lip produced N lip patches (static anchor)`
- `Queued N lip patches`

**Note:** Wav2Lip runs on a single anchor still (~0.5–2s per audio chunk on a 4090). Only a **small lip crop** is blended onto the moving idle loop — not a giant face box pulse.

---

## How it works

1. Avatar worker streams ping-pong idle loop (body/head motion preserved)
2. Subscribes to the **agent's TTS audio track** in the LiveKit room
3. Buffers audio in ~1s chunks, runs **Wav2Lip** on a cached anchor frame + that audio
4. Extracts tight lip patches from Wav2Lip output and **feather-blends** them onto the current idle frame using per-frame face boxes

Set `MOUTH_DRIVE=idle` to disable lip blending (loop only while agent speaks).

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
| `wav2lip` | Idle loop + Wav2Lip lip patches composited during speech (`MOUTH_DRIVE=composite`) |
