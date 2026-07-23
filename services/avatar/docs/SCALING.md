# Scaling notes

From one local RTX 4090 to a multi-session fleet. The FlashHead Lite engine
emits one continuous 25 fps stream per session (idle and speech share the same
generation loop). Clip-based backends are not the production concurrency path.

Measured on the local 4090 (this repo, `smoke_test.py` / `e2e_test.py`):

| Metric | Value |
|---|---|
| Chunk generation (24 frames / 0.96 s) | ~410 ms steady state (~730 ms first chunk) |
| Realtime headroom | ~2.3× ⇒ **2–3 sessions per 4090** (vendor claims 3) |
| Avatar prep from a new image | ~5–15 s (one-time per session) |
| First spoken word after user message | ~2–4 s (LLM + TTS + 0.6 s stream lead) |
| VRAM per engine | ~8 GB ⇒ fits 2 engines per 24 GB card |

## Architecture (local = production)

The engine (`server/engine.py`) is transport-agnostic: it takes an image + a
rolling audio buffer and emits frame chunks through a sink callback. Local demo
and production differ **only in the sink**:

```
                    ┌────────────── session worker (1 GPU slot) ──────────────┐
 user audio/text →  │ LLM (OpenAI) → TTS (ElevenLabs Flash WS) → FlashHead    │
                    │                                    engine │ frame sink  │
                    └───────────────────────────────────────────┼─────────────┘
        local demo:  WebSocket → JPEG/PCM chunks → browser canvas+WebAudio
        production:  LiveKit video+audio track → WebRTC → browser <video>
```

- **Local (this repo):** FastAPI + WebSocket, JPEG frames, WebAudio-clock sync.
- **Production:** the same engine publishes raw frames/PCM into a LiveKit room
  via `livekit-agents` (VideoSource/AudioSource). ~50 lines of adapter code.

Session isolation is inherent: one engine instance = one user = one scenario
(own persona prompt, own image, own conversation history).

## Deployment topology

```
Browser ↔ LiveKit Cloud (WebRTC, rooms, TURN)
              ↕ dispatch
        Session workers on RunPod GPU fleet
        (each worker: N engine slots, N = 2-3 per 4090-class GPU)
              ↕
        OpenAI / ElevenLabs APIs
```

1. **Web tier** (Vercel or any static host): token endpoint + UI. Stateless.
2. **Orchestrator** (small always-on service): assigns sessions to workers,
   scales the RunPod pool on queue depth, handles reconnects.
3. **GPU workers** (RunPod): Docker image with weights baked in. Each worker
   process pins one engine per GPU slot; a 4090 hosts 2–3 slots.
4. **LiveKit Cloud**: transport + presence. No GPU work.

## Capacity & cost at 7,000 concurrent

GPU sizing (the dominant cost):

| GPUs | Sessions/GPU | GPUs needed | RunPod rate | $/hr peak |
|---|---|---|---|---|
| RTX 4090 (Secure pod) | 3 | 2,334 | ~$0.44/hr | **~$1,030/hr** |
| RTX 4090 (Serverless active) | 3 | 2,334 | ~$0.76/hr | ~$1,770/hr |
| L40S (Serverless active) | 4–5* | ~1,550 | ~$1.33/hr | ~$2,060/hr |

\* L40S/A100 fit more engines by VRAM but win mainly on ops (availability,
reliability); 4090-class Secure pods are the cheapest per session.

Per-session-hour, all services:

| Item | Rate | $/session-hr |
|---|---|---|
| GPU (4090 pod ÷ 3 sessions) | $0.44/hr | $0.147 |
| LiveKit agent minutes | $0.01/min | $0.60 |
| LiveKit participant WebRTC | $0.0004/min | $0.024 |
| ElevenLabs Flash (~40% talk, ~9k chars) | $0.05/1k chars | $0.45 |
| OpenAI gpt-4o-mini (~40 turns) | ~$0.008/turn | $0.32 |
| **Total** | | **≈ $1.54/session-hr** |

**7,000 concurrent for one hour ≈ $10.8k.** If 7,000 is *daily actives* with
10-minute sessions (≈300–700 peak concurrent), costs drop to **$180–$400/hr
peak** and roughly **$18–40k/month** — LiveKit agent minutes and TTS, not GPU,
become the biggest lines. Negotiate both at volume (LiveKit enterprise,
ElevenLabs enterprise ~30–50% off list).

### Cost levers, in order of impact
1. **Sessions per GPU** — SageAttention + torch.compile (Linux/Triton) pushes
   the Lite model toward its 96 fps ceiling ⇒ 3 sessions safely, maybe 4.
2. **Scale-to-zero** — serverless flex workers for the long tail; keep a warm
   active pool sized to P50 traffic (weights baked into the image; FlashBoot
   cold start ~30–60 s with model load).
3. **Idle throttling** — when nobody is watching (tab hidden), pause generation
   entirely; the engine already stops when no clients are connected.
4. **LLM/TTS** — persona prompts are short; gpt-4o-mini + Flash v2.5 are already
   the cheap tier. Local Llama on the same GPUs is possible but competes for
   VRAM with engine slots — not worth it below ~2k concurrent.

## ElevenLabs in production

- Use the **WebSocket streaming API** (`eleven_flash_v2_5`, `pcm_16000`,
  `optimize_streaming_latency=3`): TTFB ~100–150 ms NA/EU. The local demo uses
  the HTTP endpoint per sentence (~300–500 ms) — the WS swap saves ~0.3 s of
  time-to-first-word and streams audio into the engine buffer as it arrives.
- Feed LLM sentence chunks directly into the TTS socket (`flush: true` at turn
  end) — this pipeline already chunks by sentence, so it's a drop-in change in
  `clients.py`.
- Watch: concurrency limits per plan tier (enterprise for thousands of parallel
  streams), APAC TTFB (150–200 ms), and voice-clone licensing for any real
  person's face/voice pairing.

## Failure & edge cases

| Case | Handling |
|---|---|
| GPU worker dies mid-session | Orchestrator reassigns; avatar re-preps from stored image (~10 s); LiveKit room survives |
| Bad upload (no face, side profile) | Face-crop raises → 422 with clear message (already implemented) |
| Generation falls behind realtime | Engine resyncs its schedule clock; client drops stale frames (already implemented) |
| TTS/LLM outage | Engine keeps generating idle motion — the avatar never freezes; surface a text error |
| User interrupts mid-reply | Clear engine speech buffer (one method call) — video transitions back to idle within one chunk (~1 s) |
| Long silence / tab hidden | No clients connected ⇒ generation pauses ⇒ GPU slot is effectively free (enables oversubscription) |

## Migration checklist (local → production)

1. Dockerfile: CUDA 12.4 base, bake `models/` into the image, Linux gets real
   xfuser/flash-attn/SageAttention (delete the Windows compat shim path).
2. Replace the WS sink with a LiveKit publisher (`livekit-agents`, VideoSource
   512×512@25fps + AudioSource 16 kHz).
3. Swap TTS to ElevenLabs WebSocket streaming.
4. Orchestrator: session→worker assignment + RunPod autoscaling API.
5. Load test one worker: 3 sessions × 30 min, watch `busy_ratio` < 0.85.
