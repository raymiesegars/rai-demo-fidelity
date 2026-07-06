# Avatar worker (RunPod GPU)

## Quick start — recommended demo (speech-reactive mouth)

Ping-pong loop + **instant** mouth movement synced to agent audio. No LivePortrait, no Wav2Lip, no per-frame ML — runs at full FPS on any GPU.

```bash
cd /workspace/rai-demo-fidelity/services/avatar-worker
git pull   # or copy speech_reactive.py + main.py from your PC
bash start_reactive.sh
```

Look for:

```
Per-frame face boxes ready (114 frames)
Mouth drive: reactive — instant audio-synced mouth on loop
Avatar mode: reactive — instant speech-synced mouth on loop
```

**Requires** cached face boxes at `/workspace/Wav2Lip/temp/alan_face_boxes.json` (already on RunPod from earlier setup).

### Tune mouth strength

| Variable | Default | Effect |
|----------|---------|--------|
| `MOUTH_SENSITIVITY` | `900` | Lower = more open on quiet speech |
| `MOUTH_MAX_STRETCH` | `0.22` | How far the mouth opens |
| `ANIMATION_ENERGY_THRESHOLD` | `20` | Ignore noise below this RMS |
| `LIP_RECT_Y_SHIFT` | `-0.04` | Nudge mouth overlay up/down |

---

## Idle loop only (no mouth animation)

```bash
bash apply_demo_env.sh   # sets AVATAR_MODE=mock, MOUTH_DRIVE=idle
export $(grep -v '^#' .env | xargs)
python main.py
```

---

## LivePortrait + JoyVASA (experimental — not real-time on current RunPod)

Wav2Lip lip patches **do not work** on a moving full-body loop. LivePortrait drives the face mesh but needs **~3–13s per frame** on our ONNX+CPU setup — not viable for live conversation.

See `setup_liveportrait.sh` and `start_liveportrait.sh` if you want to experiment later (e.g. TensorRT Docker).

---

## Modes

| AVATAR_MODE | Behavior |
|-------------|----------|
| `reactive` | **Recommended** — audio energy → subtle mouth stretch on loop |
| `mock` | Idle loop only |
| `liveportrait` | JoyVASA + LivePortrait (slow; falls back to reactive if models missing) |
| `wav2lip` | Legacy composite (not recommended on moving loop) |

## Env scripts

| Script | Purpose |
|--------|---------|
| `start_reactive.sh` | **Recommended** — env + start worker |
| `apply_reactive_env.sh` | Set reactive env vars only |
| `apply_demo_env.sh` | Idle loop only |
| `apply_liveportrait_env.sh` | LivePortrait (experimental) |
| `setup_liveportrait.sh` | Download FLP + JoyVASA models |

## How reactive mode works

1. Agent TTS audio arrives over LiveKit.
2. RMS energy is smoothed (attack/decay).
3. Each loop frame uses its cached face box to stretch/blend the lip region — synced in the same frame, zero inference lag.

This is not Hollywood lip sync, but it reads as “talking” during conversation and **always keeps up** with the agent.
