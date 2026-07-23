"""SadTalker worker smoke test."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(os.environ.get("SADTALKER_PYTHON", ROOT / ".venv_sadtalker" / "Scripts" / "python.exe"))
WORKER = ROOT / "scripts" / "sadtalker_worker.py"
VENDOR = Path(os.environ.get("SADTALKER_ROOT", ROOT / "vendor" / "SadTalker"))
IMG = ROOT / "assets" / "jordan-blake-still.png"
WORK = ROOT / "uploads" / "_sadtalker_work"
WAV = WORK / "smoke.wav"
OUT = WORK / "smoke.mp4"


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    if not WAV.is_file():
        import numpy as np
        import soundfile as sf

        t = np.linspace(0, 1.2, 19200, endpoint=False)
        sf.write(str(WAV), (np.sin(2 * np.pi * 220 * t) * 0.2).astype("float32"), 16000)

    OUT.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update({
        "SADTALKER_ROOT": str(VENDOR),
        "SADTALKER_SIZE": "256",
        "SADTALKER_PREPROCESS": "full",
        "SADTALKER_STILL": "1",
        "PYTHONUNBUFFERED": "1",
    })
    p = subprocess.Popen(
        [str(PY), str(WORKER)],
        cwd=str(VENDOR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )
    assert p.stdin and p.stdout

    def req(msg: dict) -> dict:
        p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        line = p.stdout.readline()
        if not line:
            err = p.stderr.read()
            raise RuntimeError(f"worker died\n{err[-3000:]}")
        return json.loads(line)

    ready = json.loads(p.stdout.readline())
    print("READY:", ready)
    if not ready.get("ok"):
        print(p.stderr.read()[-3000:])
        return 1

    prep = req({"cmd": "prepare", "source_path": str(IMG)})
    print("PREPARE:", {k: prep.get(k) for k in ("ok", "ms", "error")})
    if not prep.get("ok"):
        print(prep.get("trace", "")[-2000:])
        return 1

    inf = req({"cmd": "infer", "audio_path": str(WAV), "output_path": str(OUT)})
    print("INFER:", {k: inf.get(k) for k in ("ok", "ms", "video_path", "error")})
    if not inf.get("ok"):
        print(inf.get("trace", "")[-2500:])
        return 1

    req({"cmd": "quit"})
    p.wait(timeout=30)
    if not OUT.is_file() or OUT.stat().st_size < 1000:
        print("SMOKE FAIL missing video")
        return 1
    print("SMOKE OK", OUT.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
