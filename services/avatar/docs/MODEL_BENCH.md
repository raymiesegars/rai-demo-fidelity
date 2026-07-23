# Model bench protocol

Compare talking-head models in isolation (image + audio → video), then store
scores in `bench/results/*.json`. The Demo Analytics tab and the Vercel site
both read that folder.

## One process, one backend

| Approach | Verdict |
|---|---|
| Branch per model | Results scatter; hard to compare |
| All models loaded at once | Exceeds one 24 GB GPU |
| `AVATAR_BACKEND` swap + restart | Correct — one model in VRAM, shared UI |

```powershell
cd services\avatar
.\run_server.ps1 -Backend flashhead
.\run_server.ps1 -Backend ditto -KillExisting
```

Installers and per-model notes: [LOCAL_SETUP.md](LOCAL_SETUP.md).

## Categories

| Category | How measured |
|---|---|
| Image animation vs video gen | Catalog metadata (`modality`) |
| Fidelity / uncanny valley | Manual 1–10 after fixed clip set |
| Concurrency / scalability | From `busy_ratio` + VRAM → sessions/GPU |
| Latency / conversation speed | gen_ms, realtime factor, optional TTFW |
| Hosting flexibility | Curated ops score in results JSON |
| Cost | `$0.44/hr ÷ sessions_per_gpu` (GPU only) |

LiveKit, nginx, Redis, and agent fleets are out of scope for these benches so
model differences are not mixed with transport costs.

## Framing rule

Use **Full image** on Demo for fair comparison. Face-only backends must paste
into a locked face rectangle (no per-frame redetect). Close-up mode shows the
raw model frame and is not the production composite path.

## Cost formula

```
busy_ratio     = GPU_seconds / speech_seconds
eff_sessions   = min(0.85 / busy_ratio, VRAM_cap)
$/sess-hr      = 0.44 / eff_sessions
```

Fractional `eff_sessions` below 1 means slower than realtime.

## CLI

```powershell
.\.venv\Scripts\python.exe -m bench.harness --from-stats
.\.venv\Scripts\python.exe -m bench.score flashhead --fidelity 7 --uncanny 6 --composite 8
```

## Publishing Analytics

After updating `bench/results/*.json`:

```powershell
cd ..\..\apps\analytics
npm run sync-data
```

Commit `apps/analytics/data/comparison.json` (or let Vercel build run `prebuild`)
and redeploy the analytics app.
