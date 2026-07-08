# Local MuseTalk lip-sync stack (RTX 4090)

Generative mouth video from audio using **MuseTalk 1.5** — replaces the speech-reactive warp hack.

## Prerequisites

- Windows + **NVIDIA RTX 4090** (or similar, 8GB+ VRAM)
- Python 3.11
- `alan-loop.mp4` in `assets/` (copy from RunPod)

## Quick start

```powershell
cd services\lipsync

# 1. One-time setup (venv, PyTorch, deps, ffmpeg)
.\setup.ps1

# 2. Download model weights (~5-8 GB, one-time)
.\download_weights.ps1

# 3. Copy Alan video from RunPod to:
#    services\lipsync\assets\alan-loop.mp4

# 4. Prepare avatar (face detect + latents, ~2-5 min)
.\run_prepare_avatar.ps1

# 5. Test with sample audio
.\run_test_clip.ps1
```

## Copy alan-loop.mp4 from RunPod

**Option A — RunPod file browser:** Connect → file browser → download `alan-loop.mp4`

**Option B — RunPod web terminal + base64** (small files only; loop may be large)

**Option C — SCP** (once SSH key is set up on RunPod)

Place file at: `services\lipsync\assets\alan-loop.mp4`

## Local live demo (coming next)

```powershell
.\run_local_demo.ps1
```

Opens a browser UI: type chat → OpenAI + Cartesia TTS → MuseTalk lip-sync video.

## Architecture

```
User text → OpenAI LLM → Cartesia TTS → audio.wav
                                              ↓
                         MuseTalk (prepared Alan avatar) → lip-synced MP4
                                              ↓
                                    Browser video player
```

## Scalability path

| Phase | Where | Notes |
|-------|-------|-------|
| **Now** | Local 4090 | Fast iteration, no RunPod/LiveKit |
| **Demo** | RunPod 4090 | Same MuseTalk stack in Docker |
| **Prod** | GPU per session | MuseTalk realtime ~30fps; or SoulX-FlashHead upgrade |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA not available` | Reinstall PyTorch cu118 in `.venv` |
| `mmcv` install fails | Use `mim install mmcv==2.0.1` |
| `ffmpeg not found` | Run `setup.ps1` or add `tools\ffmpeg\...\bin` to PATH |
| Avatar prep fails | Ensure `alan-loop.mp4` exists; try `bbox_shift: 0` or `10` in config |
| Mouth too open/closed | Tune `bbox_shift` in `configs/alan_realtime.yaml` |

## Env vars (local demo)

Copy from `services/agent/.env`:

```
OPENAI_API_KEY=...
CARTESIA_API_KEY=...
CARTESIA_VOICE_ID=df89f42f-f285-4613-adbf-14eedcec4c9e
```
