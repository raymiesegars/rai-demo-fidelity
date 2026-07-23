"""Persistent Ditto worker — loads StreamSDK once, serves many utterances.

Protocol (JSON lines on the real stdout; all library prints go to stderr):
  → {"cmd":"ping"}
  ← {"ok":true,"cmd":"pong"}
  → {"cmd":"infer","audio_path":"...","source_path":"...","output_path":"...",
      "sampling_timesteps":10}
  ← {"ok":true,"cmd":"infer","ms":4200,"video_path":"..."}
  → {"cmd":"quit"}
  ← {"ok":true,"cmd":"bye"}

Env:
  DITTO_DATA_ROOT, DITTO_CFG
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def _reply(msg: dict) -> None:
    sys.__stdout__.write(json.dumps(msg) + "\n")
    sys.__stdout__.flush()


def main() -> int:
    # Keep JSON responses clean — tqdm / print / warnings → stderr
    sys.stdout = sys.stderr

    vendor = Path(__file__).resolve().parents[1] / "vendor" / "ditto-talkinghead"
    if not vendor.is_dir():
        _reply({"ok": False, "error": f"vendor missing: {vendor}"})
        return 1
    sys.path.insert(0, str(vendor))
    os.chdir(str(vendor))

    data_root = os.environ.get("DITTO_DATA_ROOT", "")
    cfg_pkl = os.environ.get("DITTO_CFG", "")
    if not data_root or not cfg_pkl:
        _reply({"ok": False, "error": "DITTO_DATA_ROOT and DITTO_CFG required"})
        return 1

    t0 = time.time()
    try:
        from inference import run
        from stream_pipeline_offline import StreamSDK

        SDK = StreamSDK(cfg_pkl, data_root)
    except Exception as e:
        _reply({"ok": False, "error": f"SDK load failed: {e}", "trace": traceback.format_exc()[-2000:]})
        return 1
    load_s = round(time.time() - t0, 2)
    default_steps = int(os.environ.get("DITTO_SAMPLING_TIMESTEPS", "10"))
    _reply({"ok": True, "cmd": "ready", "load_s": load_s, "sampling_timesteps": default_steps})

    cached_source: str | None = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _reply({"ok": False, "error": f"bad json: {e}"})
            continue

        cmd = req.get("cmd")
        if cmd == "ping":
            _reply({"ok": True, "cmd": "pong"})
            continue
        if cmd == "quit":
            _reply({"ok": True, "cmd": "bye"})
            return 0
        if cmd != "infer":
            _reply({"ok": False, "error": f"unknown cmd: {cmd}"})
            continue

        audio_path = req.get("audio_path")
        source_path = req.get("source_path")
        output_path = req.get("output_path")
        steps = int(req.get("sampling_timesteps", default_steps))
        if not audio_path or not source_path or not output_path:
            _reply({"ok": False, "error": "infer requires audio_path, source_path, output_path"})
            continue

        t1 = time.time()
        try:
            # Paper realtime path uses 10 denoising steps (quality ≈ 50).
            more = {"setup_kwargs": {"sampling_timesteps": steps}}
            run(SDK, audio_path, source_path, output_path, more)
            video = output_path if Path(output_path).is_file() else output_path + ".tmp.mp4"
            if not Path(video).is_file():
                raise RuntimeError("no output video written")
            cached_source = source_path
            _reply({
                "ok": True,
                "cmd": "infer",
                "ms": round((time.time() - t1) * 1000),
                "video_path": video,
                "sampling_timesteps": steps,
                "source_cached": cached_source == source_path,
            })
        except Exception as e:
            _reply({
                "ok": False,
                "cmd": "infer",
                "error": str(e),
                "trace": traceback.format_exc()[-2500:],
                "ms": round((time.time() - t1) * 1000),
            })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
