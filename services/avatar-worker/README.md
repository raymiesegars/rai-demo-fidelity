# Avatar worker (RunPod GPU)

## Quick start — demo today (no lip sync)

Clean ping-pong loop + agent voice. Use this if LivePortrait is not installed yet.

```bash
cd /workspace/rai-demo-fidelity/services/avatar-worker
git pull
bash apply_demo_env.sh
export $(grep -v '^#' .env | xargs)
python main.py
```

---

## Lip sync — LivePortrait + JoyVASA (recommended path)

Wav2Lip lip patches **do not work** on a moving full-body loop. This stack drives the **actual face mesh** on each loop frame.

### Step 1 — Install models (once, ~15–30 min)

```bash
cd /workspace/rai-demo-fidelity/services/avatar-worker
bash setup_liveportrait.sh
```

### Step 2 — Enable LivePortrait mode

```bash
bash apply_liveportrait_env.sh
export $(grep -v '^#' .env | xargs)
python main.py
```

Look for:

```
LivePortrait engine found at /workspace/FasterLivePortrait
Mouth drive: liveportrait — JoyVASA + LivePortrait
```

If models are missing, the worker falls back to idle loop automatically.

### Optional — test quality in FLP web UI first

```bash
cd /workspace/FasterLivePortrait
python webui.py --mode onnx
```

Source = `alan-loop.mp4`, Drive = Audio, Animation region = **exp** (mouth/expression only).

---

## Modes

| AVATAR_MODE | Behavior |
|-------------|----------|
| `mock` | Idle loop only — safe demo default |
| `liveportrait` | JoyVASA audio → LivePortrait face warp on loop frames |
| `wav2lip` | Legacy — only with `MOUTH_DRIVE=composite` (not recommended) |

## Env scripts

| Script | Purpose |
|--------|---------|
| `apply_demo_env.sh` | Idle loop demo |
| `apply_liveportrait_env.sh` | LivePortrait lip sync |
| `setup_liveportrait.sh` | Download FLP + JoyVASA models |

## Env vars (LivePortrait)

| Variable | Default | Meaning |
|----------|---------|---------|
| `FLP_ROOT` | `/workspace/FasterLivePortrait` | FasterLivePortrait install |
| `FLP_CFG` | `configs/onnx_infer.yaml` | Inference config |
| `FLP_ANIMATION_REGION` | `exp` | `exp` = mouth/expression only on video source |
