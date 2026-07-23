"""Live avatar server: any portrait image + voice -> continuous talking video.

Endpoints
  POST /avatar            upload portrait image, prepares FlashHead avatar
  WS   /stream            binary frame/audio stream (one continuous session)
  WS   /chat              text in -> streamed reply text + speech into engine
  GET  /                  web UI

Stream protocol (server -> client, binary):
  [1 byte type][payload]
    0x01 = JPEG frame        (payload: jpeg bytes)
    0x02 = PCM16 audio chunk (payload: 0.96 s of 16 kHz mono pcm16)
  JSON text messages carry metadata: {"type":"chunk","seq":N,"speaking":bool,...}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("avatar.app")

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent / "web"
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)
DEFAULT_FRAMING = 1.0

# server/ + services/avatar/ on path (backends + bench package)
for _p in (Path(__file__).resolve().parent, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PERSONA = """You are Alan, a patient in a medical intake demo.
You are calm and cooperative and speak in short natural sentences (1-3 per reply).
Keep your first sentence short (4-8 words) so you can start answering quickly.
You have mild lower-back pain for about two weeks, worse when bending.
Answer the clinician's questions directly. Never break character or mention being an AI."""


def _load_env() -> None:
    for p in (ROOT / ".env", ROOT.parent / "agent" / ".env", ROOT.parent / "lipsync" / ".env"):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

app = FastAPI(title="Live Avatar")

engine = None
chat_client = None
tts_client = None

# Fan-out of generated chunks to connected /stream sockets.
_stream_queues: set[asyncio.Queue] = set()
_loop: asyncio.AbstractEventLoop | None = None


def _frame_sink(chunk: dict) -> None:
    if _loop is None:
        return
    for q in list(_stream_queues):
        _loop.call_soon_threadsafe(q.put_nowait, chunk)


@app.on_event("startup")
async def _startup() -> None:
    global engine, chat_client, tts_client, _loop
    _loop = asyncio.get_running_loop()

    from backends.registry import active_backend_id, create_backend
    from clients import ChatClient, TTSClient

    chat_client = ChatClient(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        system_prompt=PERSONA,
    )
    tts_client = TTSClient()

    def _load_engine():
        global engine
        bid = active_backend_id()
        logger.info("Loading avatar backend: %s", bid)
        eng = create_backend()
        eng.set_frame_sink(_frame_sink)
        # Always prefer the Jordan Blake bench portrait as the default face.
        default = ROOT / "uploads" / "default.png"
        jordan = ROOT / "assets" / "jordan-blake-still.png"
        repo_jordan = ROOT.parent.parent / "jordan-blake-still.png"
        import shutil
        src = jordan if jordan.is_file() else (repo_jordan if repo_jordan.is_file() else None)
        if src is not None:
            try:
                shutil.copy2(src, default)
                logger.info("Default portrait set from %s", src)
            except OSError:
                logger.exception("could not copy default portrait")
        if not default.exists():
            vendor_example = ROOT / "vendor" / "SoulX-FlashHead" / "examples" / "girl.png"
            if vendor_example.exists():
                shutil.copy2(vendor_example, default)
        # Stubs skip prepare; implemented backends warm a default face when present.
        if default.exists() and type(eng).__name__ != "StubBackend":
            try:
                eng.prepare_avatar(str(default), framing=DEFAULT_FRAMING)
            except Exception:  # noqa: BLE001
                logger.exception("default avatar prep failed (continuing)")
        eng.start()
        if type(eng).__name__ != "StubBackend":
            tts_client.warmup()
        engine = eng
        logger.info(
            "Backend live: %s (%s)",
            getattr(eng, "backend_id", bid),
            "ready" if eng.prepared else "stub/not-prepared",
        )

    await asyncio.get_running_loop().run_in_executor(None, _load_engine)


@app.on_event("shutdown")
async def _shutdown() -> None:
    eng = engine
    if eng is not None and hasattr(eng, "stop"):
        await asyncio.get_running_loop().run_in_executor(None, eng.stop)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        str(WEB_DIR / "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/healthz")
def healthz() -> dict:
    from backends.registry import active_backend_id, list_backends

    err = getattr(engine, "_error", None) if engine is not None else None
    warm_err = getattr(engine, "_warm_error", None) if engine is not None else None
    warm_ready = True
    if engine is not None and hasattr(engine, "_worker"):
        w = getattr(engine, "_worker", None)
        warm_ready = bool(
            getattr(w, "_prepared_source", None)
            and getattr(w, "_proc", None) is not None
            and getattr(w, "_proc").poll() is None
        ) if w is not None else True
    return {
        "ready": engine is not None and bool(getattr(engine, "prepared", False)),
        "backend_id": getattr(engine, "backend_id", None) or active_backend_id(),
        "backend_name": getattr(engine, "backend_name", None),
        "backends": list_backends(),
        "busy_ratio": getattr(engine, "busy_ratio", None),
        "tts": getattr(tts_client, "provider", None),
        "error": err or warm_err,
        "stub": type(engine).__name__ == "StubBackend" if engine is not None else False,
        "worker_ready": warm_ready if type(engine).__name__ != "StubBackend" else False,
    }


@app.get("/bench/models")
def bench_models() -> dict:
    from backends.registry import list_backends
    from bench.store import load_catalog

    cat = load_catalog()
    active = {b["id"]: b for b in list_backends()}
    for m in cat["models"]:
        m["runtime"] = active.get(m["id"], {})
    return cat


@app.get("/bench/comparison")
def bench_comparison() -> dict:
    from bench.store import comparison_matrix

    return comparison_matrix()


@app.get("/bench/results/{model_id}")
def bench_result(model_id: str) -> JSONResponse:
    from bench.store import load_catalog, load_result

    known = {m["id"] for m in load_catalog()["models"]}
    if model_id not in known:
        return JSONResponse({"error": "unknown model"}, status_code=404)
    data = load_result(model_id)
    if data is None:
        return JSONResponse({"error": "no results yet"}, status_code=404)
    return JSONResponse(data)


class ScoreBody(BaseModel):
    fidelity: dict | None = None
    hosting: dict | None = None
    status: str | None = None


@app.post("/bench/results/{model_id}/scores")
def bench_scores(model_id: str, body: ScoreBody) -> JSONResponse:
    """Save manual fidelity/hosting scores from the Analytics UI."""
    from bench.store import load_catalog, load_result, save_result

    known = {m["id"] for m in load_catalog()["models"]}
    if model_id not in known:
        return JSONResponse({"error": "unknown model"}, status_code=404)
    data = load_result(model_id) or {
        "model_id": model_id,
        "status": "partial",
        "automated": {},
        "manual": {"fidelity": {}, "hosting": {}},
        "clips": [],
    }
    manual = data.setdefault("manual", {})
    if body.fidelity:
        fid = manual.setdefault("fidelity", {})
        fid.update({k: v for k, v in body.fidelity.items() if v is not None})
    if body.hosting:
        host = manual.setdefault("hosting", {})
        host.update({k: v for k, v in body.hosting.items() if v is not None})
    if body.status:
        data["status"] = body.status
    elif (manual.get("fidelity") or {}).get("overall") is not None:
        data["status"] = "reviewed" if (data.get("automated") or {}).get("gen_ms_avg") else "partial"
    return JSONResponse(save_result(model_id, data))


@app.get("/bench/dialogues")
def bench_dialogues() -> dict:
    """Preset clinician prompts for the standard model comparison run."""
    path = ROOT / "bench" / "dialogues.json"
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/bench/reset-portrait")
async def bench_reset_portrait() -> JSONResponse:
    """Re-load Jordan Blake as the active avatar (Full image framing)."""
    if engine is None:
        return JSONResponse({"error": "engine not loaded"}, status_code=503)
    if type(engine).__name__ == "StubBackend":
        return JSONResponse({"error": "backend is a stub"}, status_code=503)
    jordan = ROOT / "assets" / "jordan-blake-still.png"
    default = ROOT / "uploads" / "default.png"
    if not jordan.is_file():
        return JSONResponse({"error": "jordan-blake-still.png missing under assets/"}, status_code=404)
    import shutil
    shutil.copy2(jordan, default)
    t0 = time.time()
    try:
        meta = await asyncio.get_running_loop().run_in_executor(
            None, lambda: engine.prepare_avatar(str(default), 1.0)
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("reset portrait failed")
        return JSONResponse({"error": str(e)}, status_code=422)
    return JSONResponse({"ok": True, "prep_ms": round((time.time() - t0) * 1000), **meta})


@app.post("/bench/results/{model_id}/clear")
def bench_clear(model_id: str) -> JSONResponse:
    """Wipe automated + manual scores for a failed/invalid run."""
    from bench.store import clear_result, load_catalog

    known = {m["id"] for m in load_catalog()["models"]}
    if model_id not in known:
        return JSONResponse({"error": "unknown model"}, status_code=404)
    return JSONResponse(clear_result(model_id))


@app.post("/bench/capture")
def bench_capture() -> JSONResponse:
    """Snapshot live /stats into bench/results/<active_backend>.json."""
    from backends.registry import active_backend_id
    from bench.store import merge_live_stats

    if engine is None:
        return JSONResponse({"error": "engine not loaded"}, status_code=503)
    bid = getattr(engine, "backend_id", None) or active_backend_id()
    snap = merge_live_stats(bid, stats())
    return JSONResponse(snap)


# Turn-level latency records appended by the chat pipeline.
_turn_stats: list = []


@app.get("/stats")
def stats() -> dict:
    if engine is None:
        return {"ready": False}
    import torch

    with engine.stats_lock:
        gen_ms = list(engine.gen_ms_history)
        chunks_total = engine.chunks_total
        chunks_speaking = engine.chunks_speaking
    vram_used = vram_total = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        vram_used = round((total - free) / 1e9, 2)
        vram_total = round(total / 1e9, 2)
    avg_gen = sum(gen_ms) / len(gen_ms) if gen_ms else 0
    metrics_mode = getattr(engine, "metrics_mode", "chunk")
    if metrics_mode == "utterance":
        # gen_ms = full utterance render; busy_ratio = render_s / audio_s
        busy = float(getattr(engine, "busy_ratio", 0) or 0)
        # Prefer average busy from recent renders when we can: busy ≈ avg_gen/audio
        # engine.busy_ratio is last utterance; still the right order of magnitude.
        realtime = round(1.0 / busy, 2) if busy > 0 else None
        busy_out = round(busy, 3)
    else:
        busy_out = round(avg_gen / 1000 / engine.chunk_seconds, 3) if avg_gen else 0
        realtime = round(engine.chunk_seconds * 1000 / avg_gen, 2) if avg_gen else None
    return {
        "ready": engine.prepared,
        "backend_id": getattr(engine, "backend_id", None),
        "backend_name": getattr(engine, "backend_name", None),
        "metrics_mode": metrics_mode,
        "uptime_s": round(time.time() - engine.started_at),
        "chunk_seconds": engine.chunk_seconds,
        "gen_ms_recent": gen_ms[-120:],
        "gen_ms_avg": round(avg_gen),
        "busy_ratio": busy_out,
        "realtime_factor": realtime,
        "chunks_total": chunks_total,
        "chunks_speaking": chunks_speaking,
        "vram_used_gb": vram_used,
        "vram_total_gb": vram_total,
        "turns": _turn_stats[-40:],
        "tts_provider": getattr(tts_client, "provider", None),
        "viewers": len(_stream_queues),
    }


@app.post("/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    framing: float = Form(DEFAULT_FRAMING),
) -> JSONResponse:
    if engine is None:
        return JSONResponse({"error": "engine loading"}, status_code=503)
    suffix = Path(file.filename or "img.png").suffix.lower() or ".png"
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        return JSONResponse({"error": "unsupported image type"}, status_code=400)
    framing = max(0.0, min(1.0, framing))
    dest = UPLOADS / f"{uuid.uuid4().hex[:10]}{suffix}"
    dest.write_bytes(await file.read())

    t0 = time.time()
    try:
        meta = await asyncio.get_running_loop().run_in_executor(
            None, lambda: engine.prepare_avatar(str(dest), framing)
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("avatar prep failed")
        return JSONResponse({"error": f"could not prepare avatar: {e}"}, status_code=422)
    return JSONResponse({
        "ok": True,
        "prep_ms": round((time.time() - t0) * 1000),
        **meta,
    })


class ReframeBody(BaseModel):
    framing: float = DEFAULT_FRAMING


@app.post("/avatar/reframe")
async def reframe_avatar(body: ReframeBody) -> JSONResponse:
    if engine is None or not engine.prepared:
        return JSONResponse({"error": "avatar not ready"}, status_code=503)
    framing = max(0.0, min(1.0, body.framing))
    t0 = time.time()
    try:
        meta = await asyncio.get_running_loop().run_in_executor(
            None, lambda: engine.reframe_avatar(framing)
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("avatar reframe failed")
        return JSONResponse({"error": f"could not reframe avatar: {e}"}, status_code=422)
    return JSONResponse({
        "ok": True,
        "prep_ms": round((time.time() - t0) * 1000),
        **meta,
    })


@app.get("/avatar/session")
def avatar_session() -> JSONResponse:
    if engine is None or not engine.prepared:
        return JSONResponse({"ready": False}, status_code=503)
    return JSONResponse({"ready": True, **engine.session_info()})


@app.get("/avatar/source")
def avatar_source() -> FileResponse:
    if engine is None or not engine._source_image:
        return JSONResponse({"error": "no source image"}, status_code=404)
    path = Path(engine._source_image)
    if not path.exists():
        return JSONResponse({"error": "source missing"}, status_code=404)
    return FileResponse(
        str(path),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.websocket("/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    if engine and engine.prepared:
        await ws.send_json({"type": "session", **engine.session_info()})
    q: asyncio.Queue = asyncio.Queue(maxsize=8)
    _stream_queues.add(q)
    if engine:
        engine.client_connected()
    try:
        while True:
            chunk = await q.get()
            await ws.send_json({
                "type": "chunk", "seq": chunk["seq"], "speaking": chunk["speaking"],
                "speech_s": chunk["speech_s"],
                "frames": len(chunk["jpegs"]), "gen_ms": chunk["gen_ms"],
            })
            await ws.send_bytes(b"\x02" + chunk["audio_pcm16"])
            for jpg in chunk["jpegs"]:
                await ws.send_bytes(b"\x01" + jpg)
    except WebSocketDisconnect:
        pass
    finally:
        _stream_queues.discard(q)
        if engine:
            engine.client_disconnected()


def _speak_pipeline(text: str, history: list[dict], out_q: asyncio.Queue) -> None:
    """LLM sentences -> TTS -> push into engine speech buffer.

    Emits `speech_text` (sentence + audio duration) as each sentence's audio is
    pushed, so the client can reveal captions in sync with playback.
    """
    assert _loop is not None

    def emit(msg: dict) -> None:
        _loop.call_soon_threadsafe(out_q.put_nowait, msg)

    parts: list[str] = []
    t0 = time.time()
    llm_first_ms = None
    first_audio_ms = None
    try:
        # Clip models (Sonic/MuseTalk/…) start rendering as soon as speech stops
        # growing. Pushing per-sentence caused early cutoffs (only first clause).
        # Coalesce TTS into one utterance for those backends; FlashHead Lite still
        # streams sentence-by-sentence for lower TTFW. Pro smooth-buffer needs the
        # full utterance so it doesn't flush between sentences.
        coalesce = hasattr(engine, "render_utterance") or bool(
            getattr(engine, "smooth_buffer", False)
        )
        pending_audio: list[np.ndarray] = []
        for sentence in chat_client.stream_sentences(history, text):
            if llm_first_ms is None:
                llm_first_ms = round((time.time() - t0) * 1000)
            parts.append(sentence)
            audio = tts_client.synthesize(sentence)
            if first_audio_ms is None:
                first_audio_ms = round((time.time() - t0) * 1000)
                emit({"type": "first_audio_ms", "ms": first_audio_ms})
            emit({"type": "speech_text", "text": sentence,
                  "dur": round(len(audio) / 16000, 3)})
            if coalesce:
                pending_audio.append(np.asarray(audio, dtype=np.float32))
            else:
                engine.push_speech(audio)
        if coalesce and pending_audio:
            engine.push_speech(np.concatenate(pending_audio))
        # Hold "done" until lipsync is queued so the client can reveal
        # message + audio + video together (no text-only / audio-only gap).
        wait = getattr(engine, "wait_for_playback_or_error", None)
        if callable(wait):
            err = wait(1800.0)
            if err:
                emit({"type": "error", "message": f"Avatar render failed: {err[:800]}"})
                return
        emit({"type": "av_ready"})
        emit({"type": "done", "reply": " ".join(parts).strip()})
        _turn_stats.append({
            "t": round(time.time()),
            "llm_first_ms": llm_first_ms,
            "first_audio_ms": first_audio_ms,
            "sentences": len(parts),
            "chars": sum(len(p) for p in parts),
        })
    except Exception as e:  # noqa: BLE001
        logger.exception("chat pipeline failed")
        emit({"type": "error", "message": str(e)})


class SayBody(BaseModel):
    text: str


@app.post("/say")
async def say(body: SayBody) -> JSONResponse:
    """Speak arbitrary text directly (bypasses the LLM). Demo/test helper."""
    if engine is None or not engine.prepared:
        return JSONResponse({"error": "avatar not ready"}, status_code=503)
    t0 = time.time()
    try:
        audio = await asyncio.get_running_loop().run_in_executor(
            None, tts_client.synthesize, body.text
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("tts failed")
        return JSONResponse({"error": f"tts failed: {e}"}, status_code=502)
    engine.push_speech(audio)
    wait = getattr(engine, "wait_for_playback_or_error", None)
    if callable(wait):
        err = await asyncio.get_running_loop().run_in_executor(None, wait, 600.0)
        if err:
            return JSONResponse({"error": f"Avatar render failed: {err[:800]}"}, status_code=502)
    return JSONResponse({
        "ok": True,
        "tts_ms": round((time.time() - t0) * 1000),
        "audio_s": round(len(audio) / 16000, 2),
        "av_ready": True,
    })


@app.websocket("/chat")
async def ws_chat(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            text = (data.get("text") or "").strip()
            if not text:
                await ws.send_json({"type": "error", "message": "empty text"})
                continue
            if engine is None or not engine.prepared:
                await ws.send_json({"type": "error", "message": "avatar not ready"})
                continue
            history = data.get("history") or []
            out_q: asyncio.Queue = asyncio.Queue()
            asyncio.get_running_loop().run_in_executor(
                None, _speak_pipeline, text, history, out_q
            )
            while True:
                msg = await out_q.get()
                await ws.send_json(msg)
                if msg["type"] in ("done", "error"):
                    break
    except WebSocketDisconnect:
        return
