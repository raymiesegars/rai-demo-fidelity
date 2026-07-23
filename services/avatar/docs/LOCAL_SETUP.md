# Local setup — all backends

Run every command from `services\avatar` unless noted. If PowerShell blocks
scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Or: `powershell -ExecutionPolicy Bypass -File .\setup_wav2lip.ps1`

## Shared Demo environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Fill `OPENAI_API_KEY`. For TTS without a paid key: `TTS_PROVIDER=edge` in `.env`.

Start / restart:

```powershell
.\run_server.ps1 -Backend <id> -KillExisting
```

Demo UI: http://127.0.0.1:8100  
Use **Full image** framing so face-only models composite into a locked rect.

---

## flashhead (Lite) — default

```powershell
git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead vendor\SoulX-FlashHead
.\.venv\Scripts\python.exe -m pip install -q huggingface_hub
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Soul-AILab/SoulX-FlashHead-1_3B', local_dir=r'models/SoulX-FlashHead-1_3B', allow_patterns=['Model_Lite/*','VAE_LTX/*']); snapshot_download('facebook/wav2vec2-base-960h', local_dir=r'models/wav2vec2-base-960h')"
.\run_server.ps1 -Backend flashhead
```

Continuous generative stream with idle motion. Primary realtime candidate on one 4090.

## flashhead-pro

Requires Lite weights already present, then:

```powershell
.\setup_flashhead_pro.ps1
.\run_server.ps1 -Backend flashhead-pro -KillExisting
```

Downloads `Model_Pro` + `VAE_Wan`. Higher detail; about 0.2× realtime on a 4090.
Idle generative motion is disabled by default (static hold between replies).

## wav2lip

```powershell
.\setup_wav2lip.ps1
.\run_server.ps1 -Backend wav2lip -KillExisting
```

Classic lipsync on a face crop; static between utterances.

## musetalk

```powershell
.\setup_musetalk.ps1
.\run_server.ps1 -Backend musetalk -KillExisting
```

Uses `.venv_musetalk`. Clip infer streamed into the shared player.

## sadtalker

```powershell
.\setup_sadtalker.ps1
.\run_server.ps1 -Backend sadtalker -KillExisting
```

Offline 3DMM portrait baseline (not conversation-realtime).

## sonic

```powershell
.\setup_sonic.ps1
.\run_server.ps1 -Backend sonic -KillExisting
```

Stable Video Diffusion img2vid. License is CC BY-NC-SA (non-commercial). Slow.

## liveportrait

```powershell
.\setup_liveportrait.ps1
.\run_server.ps1 -Backend liveportrait -KillExisting
```

FasterLivePortrait ONNX + JoyVASA. First warm can take minutes; wait until ready.

## livetalk

```powershell
.\setup_livetalk.ps1
.\run_server.ps1 -Backend livetalk -KillExisting
```

Distilled Wan 1.3B. Official path is Linux + flash-attn; Windows uses SDPA.

## echomimic

```powershell
.\setup_echomimic.ps1
.\run_server.ps1 -Backend echomimic -KillExisting
```

EchoMimicV3-Flash. Long warm load; utterance clips, not a continuous idle stream.

## ditto

```powershell
.\setup_ditto.ps1
.\run_server.ps1 -Backend ditto -KillExisting
```

Requires git-lfs for weights. Open PyTorch path is clip-based; paper realtime uses TensorRT.

---

## Recording bench results

1. Restart with the target `-Backend`.
2. Demo → Full image → short conversation (or Analytics → **Run 10-dialogue test**).
3. Analytics → **Capture live stats → results**.
4. Fill fidelity / uncanny / composite → **Save scores**.

Results: `bench/results/<backend>.json`. Catalog: `bench/models.json`.

```powershell
.\.venv\Scripts\python.exe -m bench.score <backend> --fidelity 6 --uncanny 5 --composite 7 --lips 6 --hosting 5
```

Sync the Vercel analytics snapshot after editing results:

```powershell
cd ..\..\apps\analytics
npm run sync-data
```
