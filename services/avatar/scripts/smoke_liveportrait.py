"""Smoke-test LivePortrait worker: start → prepare → short infer."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "FasterLivePortrait"
PY = ROOT / ".venv_liveportrait" / "Scripts" / "python.exe"
WORKER = ROOT / "scripts" / "liveportrait_worker.py"
SRC = ROOT / "uploads" / "default.png"
if not SRC.is_file():
    SRC = ROOT / "assets" / "jordan-blake-still.png"


def main() -> int:
    if not PY.is_file():
        print("missing venv", PY)
        return 1
    if not WORKER.is_file():
        print("missing worker", WORKER)
        return 1
    if not SRC.is_file():
        print("missing source", SRC)
        return 1

    env = os.environ.copy()
    env["FLP_ROOT"] = str(VENDOR)
    env["PYTHONUNBUFFERED"] = "1"
    env["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
    env["PATH"] = str(PY.parent) + os.pathsep + env.get("PATH", "")

    print("starting worker…", flush=True)
    proc = subprocess.Popen(
        [str(PY), str(WORKER)],
        cwd=str(VENDOR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    def drain() -> None:
        assert proc.stderr
        for line in proc.stderr:
            print("ERR:", line.rstrip(), flush=True)

    threading.Thread(target=drain, daemon=True).start()

    def read(timeout: float) -> dict:
        box: dict = {}

        def _r() -> None:
            try:
                assert proc.stdout
                box["line"] = proc.stdout.readline()
            except BaseException as e:  # noqa: BLE001
                box["err"] = e

        t = threading.Thread(target=_r, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise TimeoutError(f"no response in {timeout}s (poll={proc.poll()})")
        if "err" in box:
            raise box["err"]
        line = box.get("line") or ""
        if not line:
            raise RuntimeError(f"empty response (poll={proc.poll()})")
        return json.loads(line)

    try:
        ready = read(300)
        print("READY:", ready, flush=True)
        if not ready.get("ok"):
            return 1

        work = ROOT / "uploads" / "_lp_smoke"
        work.mkdir(parents=True, exist_ok=True)
        proc.stdin.write(json.dumps({"cmd": "prepare", "source_path": str(SRC)}) + "\n")
        proc.stdin.flush()
        print("PREP:", read(120), flush=True)

        # 0.6s silence wav
        import numpy as np
        import soundfile as sf

        wav = work / "t.wav"
        sf.write(str(wav), np.zeros(9600, dtype=np.float32), 16000)
        out = work / "out.mp4"
        proc.stdin.write(json.dumps({
            "cmd": "infer",
            "audio_path": str(wav),
            "output_path": str(out),
        }) + "\n")
        proc.stdin.flush()
        print("INFER:", read(300), flush=True)
        print("video exists:", out.is_file(), out.stat().st_size if out.is_file() else 0)
        return 0
    except Exception as e:
        print("FAIL:", e, flush=True)
        print("poll:", proc.poll(), flush=True)
        return 1
    finally:
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=5)
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
