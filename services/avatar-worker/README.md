# Avatar worker (RunPod GPU)

Streams Alan's ping-pong idle loop to LiveKit and (eventually) drives the mouth from agent TTS audio.

## Recommended demo today — idle loop only

Wav2Lip lip-patch compositing **does not work** on a moving full-body loop (delay, misalignment, floating mouth). Use clean idle mode until LivePortrait is wired in.

```bash
cd /workspace/rai-demo-fidelity/services/avatar-worker
git pull
bash apply_demo_env.sh
export $(grep -v '^#' .env | xargs)
python main.py
```

You should see: `AVATAR_MODE=mock` — Alan loops naturally while the agent speaks. Voice + chat still work.

---

## Why Wav2Lip composite failed

| Approach | Problem |
|----------|---------|
| Mouth warp / pulse | Giant box scaling, not real lips |
| Wav2Lip full-loop batch | Seconds late, blurry |
| Wav2Lip static + lip patch | ~1s lag, patch misaligned on moving head, looks like a second mouth |

**Root issue:** batch lip models trained on a **still face** cannot be pasted onto a **ping-pong body loop** and look real.

---

## Real solution — FasterLivePortrait + JoyVASA

This is what actually targets your goal: **audio-driven facial motion rendered back into the video**, with **lip-only region** when the source is a video loop (per FasterLivePortrait docs).

### One-time setup on RunPod

```bash
cd /workspace/rai-demo-fidelity/services/avatar-worker
bash setup_liveportrait.sh
```

### Manual quality check (before we automate)

```bash
cd /workspace/FasterLivePortrait
python webui.py --mode onnx
# Open http://localhost:9870 — use Alan's loop as source, drive with a TTS wav, lip-only region
```

### What we will build next (`AVATAR_MODE=liveportrait`)

1. JoyVASA turns agent audio → face motion coefficients in real time  
2. LivePortrait warps the **current loop frame** (lip region only)  
3. No batch delay, no floating patches — same pose as the idle loop  

`AVATAR_MODE=liveportrait` is not automated yet; `setup_liveportrait.sh` + web UI test is the current milestone.

---

## Modes

| AVATAR_MODE | Behavior |
|-------------|----------|
| `mock` | **Recommended** — ping-pong idle loop, agent voice only |
| `wav2lip` | Legacy — only if `MOUTH_DRIVE=composite` (not recommended) |
| `liveportrait` | Coming soon — JoyVASA + FasterLivePortrait |

| MOUTH_DRIVE | Behavior |
|-------------|----------|
| `idle` | **Recommended** — no mouth hack |
| `composite` | Deprecated — Wav2Lip patches (broken on loop video) |

---

## Env files

| Script | What it does |
|--------|----------------|
| `bash apply_demo_env.sh` | Sets `AVATAR_MODE=mock` + `MOUTH_DRIVE=idle` |
| `bash apply_lip_composite_env.sh` | Legacy Wav2Lip composite (do not use for demo) |
