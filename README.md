# Talking-head model bench

Local harness for comparing open-source talking-head backends on a single GPU
(RTX 4090-class). Swap models with `AVATAR_BACKEND`, run the same Demo UI, and
record metrics into `services/avatar/bench/results/`.

A password-protected **Analytics** site (read-only charts + matrix) lives in
[`apps/analytics`](apps/analytics) for Vercel deployment. The interactive Demo
stays local only.

## Repository layout

```
apps/analytics/          Password-gated comparison site (Vercel)
services/avatar/         Local demo server, backends, bench data, setup scripts
  server/                FastAPI + Demo UI
  bench/                 models.json + results/*.json
  docs/                  Bench protocol + local setup
  setup_*.ps1            Per-model installers
  run_server.ps1         Start Demo on http://127.0.0.1:8100
```

## Prerequisites

- Windows 10/11 (primary target), NVIDIA GPU with recent drivers
- Python 3.11 (`py -3.11`), Git, Git LFS (`git lfs install`) for some models
- CUDA-capable PyTorch (install commands below use cu124 wheels)
- API keys for chat Demo: OpenAI + a TTS provider (see `.env.example`)

## Quick start — FlashHead Lite (default)

```powershell
cd services\avatar
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead vendor\SoulX-FlashHead
.\.venv\Scripts\python.exe -m pip install -q huggingface_hub
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Soul-AILab/SoulX-FlashHead-1_3B', local_dir='models/SoulX-FlashHead-1_3B', allow_patterns=['Model_Lite/*','VAE_LTX/*']); snapshot_download('facebook/wav2vec2-base-960h', local_dir='models/wav2vec2-base-960h')"

copy .env.example .env
# Edit .env: OPENAI_API_KEY, TTS keys (or TTS_PROVIDER=edge for free local TTS)

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_server.ps1
```

Open http://127.0.0.1:8100 — Demo tab for the live avatar, Analytics tab for
the local comparison matrix.

## Swap backends

```powershell
cd services\avatar
.\run_server.ps1 -Backend flashhead -KillExisting
.\run_server.ps1 -Backend wav2lip -KillExisting
.\run_server.ps1 -Backend flashhead-pro -KillExisting
```

| Backend | Setup | Notes |
|---|---|---|
| `flashhead` | Quick start above | Continuous stream (Lite) — primary chat candidate |
| `flashhead-pro` | `.\setup_flashhead_pro.ps1` | Same stack, Pro weights; slower than realtime on one 4090 |
| `wav2lip` | `.\setup_wav2lip.ps1` | Classic lipsync crop |
| `musetalk` | `.\setup_musetalk.ps1` | Latent lipsync (own venv) |
| `sadtalker` | `.\setup_sadtalker.ps1` | Offline 3DMM portrait |
| `sonic` | `.\setup_sonic.ps1` | SVD img2vid (non-commercial license) |
| `liveportrait` | `.\setup_liveportrait.ps1` | Warp + JoyVASA |
| `livetalk` | `.\setup_livetalk.ps1` | Distilled Wan; Linux-oriented |
| `echomimic` | `.\setup_echomimic.ps1` | EchoMimicV3-Flash clips |
| `ditto` | `.\setup_ditto.ps1` | Motion diffusion; TRT path not required for clip bench |

Full step-by-step: **[services/avatar/docs/LOCAL_SETUP.md](services/avatar/docs/LOCAL_SETUP.md)**.  
Bench protocol and scoring: **[services/avatar/docs/MODEL_BENCH.md](services/avatar/docs/MODEL_BENCH.md)**.

## Capture scores

1. Start a backend, use **Full image** framing on Demo.
2. Talk through a few turns (or run the 10-dialogue bench from Analytics).
3. Analytics → **Capture live stats → results**, then fill manual fidelity scores.
4. Results are written to `services/avatar/bench/results/<id>.json`.

CLI:

```powershell
cd services\avatar
.\.venv\Scripts\python.exe -m bench.harness --from-stats
.\.venv\Scripts\python.exe -m bench.score flashhead --fidelity 7 --uncanny 7 --composite 7
```

## Deploy Analytics to Vercel

See **[apps/analytics/README.md](apps/analytics/README.md)**.

Summary:

1. Root Directory = `apps/analytics`
2. Env: `SITE_PASSWORD`, `AUTH_SECRET`
3. Redeploy after updating `bench/results/*.json` so build syncs the matrix

## Environment

Copy `services/avatar/.env.example` → `services/avatar/.env`. Do not commit `.env`.
Weights (`models/`), vendored repos (`vendor/`), and virtualenvs are gitignored.
