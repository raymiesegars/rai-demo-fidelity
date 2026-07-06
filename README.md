# Patient Fidelity Demo

High-fidelity patient avatar demo: text chat with Alan, Cartesia TTS voice, lip-synced video over LiveKit.

## Architecture

```
Browser (Next.js)  ──text──►  LiveKit Room  ◄──  Python Agent (OpenAI + Cartesia TTS)
       ▲                              │
       └── patient video track ───────┘  Avatar Worker (ping-pong alan-loop.mp4)
```

## Prerequisites

- Node.js 20+
- Python 3.11+
- LiveKit Cloud project (configured)
- RunPod RTX 4090 pod for avatar video (or run avatar worker locally for testing)

## Quick start (local)

### 1. Web app

```powershell
cd apps/web
npm install
npm run dev
```

Open http://localhost:3000

### 2. Voice agent

```powershell
cd services/agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Copy .env.example to .env and fill keys (or use provided .env)
python main.py dev
```

### 3. Avatar worker (patient video)

```powershell
cd services/avatar-worker
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Start order:** avatar worker → agent → web app → click **Start session**

## RunPod deployment (avatar worker)

1. SSH into your pod:
   ```powershell
   ssh gzstarnr13wpz3-6441140d@ssh.runpod.io -i $env:USERPROFILE\.ssh\id_ed25519
   ```

2. Clone/copy this repo to `/workspace`

3. Install deps and run:
   ```bash
   cd services/avatar-worker
   pip install -r requirements.txt
   pip install -U "huggingface_hub[cli]"
   export $(grep -v '^#' .env | xargs)
   python main.py
   ```

4. For GPU lip-sync upgrade, see `services/avatar-worker/README.md`

### Docker (optional)

```bash
cd services/avatar-worker
docker build -t patient-avatar .
docker run --env-file .env --gpus all patient-avatar
```

## Environment files

| File | Purpose |
|------|---------|
| `apps/web/.env.local` | LiveKit URL + API keys for token route |
| `services/agent/.env` | OpenAI, Cartesia, LiveKit |
| `services/avatar-worker/.env` | LiveKit, loop path, GPU mode |

Copy from `.env.example` files if starting fresh. **Never commit `.env` files.**

## Cost estimate

~**$0.12 per 10-minute session** at $0.69/hr RunPod 4090 (GPU dominates).

The web UI shows a live session cost meter.

## Patient asset

`alan-loop.mp4` at repo root — ping-pong idle loop for Alan.

## Security

Rotate any API keys that were shared in chat. Keys belong only in local `.env` files.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No patient video | Ensure avatar worker is running and joined `patient-demo` room |
| No voice response | Ensure agent is running (`python main.py dev`) |
| Agent not joining | Check LiveKit credentials; agent name is `patient-agent` |
| Token error | Verify `apps/web/.env.local` has API key + secret |
| `huggingface-cli not found` on RunPod | `pip install -U "huggingface_hub[cli]"` |
