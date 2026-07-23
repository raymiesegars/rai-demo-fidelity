"""Automated local bench: push speech audio, measure gen/VRAM, write results JSON.

Usage (server already running with desired AVATAR_BACKEND):
  cd services/avatar
  .\.venv\Scripts\python.exe -m bench.harness --from-stats
  .\.venv\Scripts\python.exe -m bench.harness --say "Hello, this is a latency probe."

Engine-only (implemented backends, no HTTP):
  .\.venv\Scripts\python.exe -m bench.harness --engine --image uploads/default.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT))

from bench.store import (  # noqa: E402
    GPU_USD_PER_HR,
    load_result,
    merge_live_stats,
    save_result,
    sessions_per_gpu,
)


def active_backend_id() -> str:
    return (os.environ.get("AVATAR_BACKEND") or "flashhead").strip().lower()


def _http_json(url: str, data: dict | None = None, method: str = "GET") -> dict:
    body = None
    headers: dict[str, str] = {}
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
        method = "POST" if method == "GET" else method
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def snapshot_from_server(base: str = "http://127.0.0.1:8100") -> dict:
    stats = _http_json(f"{base}/stats")
    health = _http_json(f"{base}/healthz")
    backend = health.get("backend_id") or active_backend_id()
    return merge_live_stats(backend, stats)


def probe_say(text: str, base: str = "http://127.0.0.1:8100") -> dict:
    t0 = time.time()
    out = _http_json(f"{base}/say", {"text": text})
    wall_ms = round((time.time() - t0) * 1000)
    time.sleep(max(2.0, float(out.get("audio_s") or 2) + 1.5))
    snap = snapshot_from_server(base)
    auto = snap.setdefault("automated", {})
    if auto.get("ttfw_ms_avg") is None:
        auto["ttfw_ms_avg"] = wall_ms
    auto["notes"] = (auto.get("notes") or "") + f" | say_wall_ms={wall_ms}"
    return save_result(snap["model_id"], snap)


def engine_bench(image: str, wav: str | None, seconds: float = 3.0) -> dict:
    import numpy as np
    import torch

    os.chdir(ROOT / "server")
    from backends.registry import create_backend

    eng = create_backend()
    bid = getattr(eng, "backend_id", active_backend_id())
    if bid != "flashhead" and type(eng).__name__ == "StubBackend":
        raise SystemExit(
            f"Engine bench requires an implemented backend; {bid} is still a stub."
        )

    t0 = time.time()
    eng.prepare_avatar(image, framing=1.0)
    prep_ms = round((time.time() - t0) * 1000)

    if wav and Path(wav).exists():
        try:
            import soundfile as sf
            audio, sr = sf.read(wav, dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            audio = audio.astype(np.float32)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Failed to load wav: {e}") from e
    else:
        n = int(16000 * seconds)
        t = np.arange(n, dtype=np.float32) / 16000.0
        audio = (
            0.08
            * np.sin(2 * np.pi * 180 * t)
            * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        ).astype(np.float32)

    eng.push_speech(audio)

    chunks: list[dict] = []
    done = threading.Event()

    def sink(chunk: dict) -> None:
        chunks.append(chunk)
        if len(chunks) >= 4:
            done.set()

    eng.set_frame_sink(sink)
    eng.start()
    done.wait(timeout=45)

    gen = [c["gen_ms"] for c in chunks if c.get("gen_ms") is not None]
    avg = round(sum(gen) / len(gen)) if gen else None
    chunk_s = float(eng.chunk_seconds)
    busy = (avg / 1000 / chunk_s) if avg else None
    rtf = (chunk_s * 1000 / avg) if avg else None

    vram_u = vram_t = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        vram_u = round((total - free) / 1e9, 2)
        vram_t = round(total / 1e9, 2)

    spg = sessions_per_gpu(busy, vram_u, vram_t)
    existing = load_result(bid) or {
        "model_id": bid,
        "manual": {"fidelity": {}, "hosting": {}},
        "clips": [],
        "modality": {},
        "hardware": {},
    }
    existing["automated"] = {
        **(existing.get("automated") or {}),
        "prep_ms": prep_ms,
        "chunk_frames": round(chunk_s * 25),
        "chunk_seconds": chunk_s,
        "gen_ms_avg": avg,
        "realtime_factor": round(rtf, 2) if rtf else None,
        "busy_ratio": round(busy, 3) if busy else None,
        "sessions_per_gpu": spg,
        "vram_used_gb": vram_u,
        "gpu_usd_per_hr": GPU_USD_PER_HR,
        "usd_per_session_hour_gpu": round(GPU_USD_PER_HR / spg, 3) if spg else None,
        "notes": f"engine_bench n_chunks={len(chunks)}",
    }
    existing["hardware"] = {
        **(existing.get("hardware") or {}),
        "vram_gb": vram_t,
        "gpu": "cuda" if torch.cuda.is_available() else "cpu",
    }
    existing["status"] = "partial"
    return save_result(bid, existing)


def main() -> None:
    ap = argparse.ArgumentParser(description="Avatar model bench harness")
    ap.add_argument("--base", default="http://127.0.0.1:8100")
    ap.add_argument("--from-stats", action="store_true")
    ap.add_argument("--say", type=str, default=None)
    ap.add_argument("--engine", action="store_true")
    ap.add_argument("--image", type=str, default=str(ROOT / "uploads" / "default.png"))
    ap.add_argument("--wav", type=str, default=None)
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    if args.engine:
        result = engine_bench(args.image, args.wav, args.seconds)
    elif args.say:
        result = probe_say(args.say, args.base)
    else:
        result = snapshot_from_server(args.base)
    print(json.dumps(result.get("automated", result), indent=2))


if __name__ == "__main__":
    main()
