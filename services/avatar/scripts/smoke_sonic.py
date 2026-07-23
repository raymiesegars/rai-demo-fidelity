"""Sonic worker smoke test (short audio)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(os.environ.get("SONIC_PYTHON", ROOT / ".venv_sonic" / "Scripts" / "python.exe"))
WORKER = ROOT / "scripts" / "sonic_worker.py"
VENDOR = Path(os.environ.get("SONIC_ROOT", ROOT / "vendor" / "Sonic"))
IMG = ROOT / "assets" / "jordan-blake-still.png"
WORK = ROOT / "uploads" / "_sonic_work"
WAV = WORK / "smoke.wav"
OUT = WORK / "smoke.mp4"


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    if not WAV.is_file():
        import numpy as np
        import soundfile as sf

        t = np.linspace(0, 1.0, 16000, endpoint=False)
        sf.write(str(WAV), (np.sin(2 * np.pi * 220 * t) * 0.2).astype("float32"), 16000)

    OUT.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update({
        "SONIC_ROOT": str(VENDOR),
        "SONIC_MIN_RES": "512",
        "SONIC_STEPS": "25",
        "SONIC_RIFE": "1",
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

    def req(msg: dict, timeout_hint: str = "") -> dict:
        p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        print(f"waiting {msg.get('cmd')} {timeout_hint}…", flush=True)
        line = p.stdout.readline()
        if not line:
            err = p.stderr.read()
            raise RuntimeError(f"worker died\n{err[-4000:]}")
        return json.loads(line)

    ready = json.loads(p.stdout.readline())
    print("READY:", ready, flush=True)
    if not ready.get("ok"):
        print(p.stderr.read()[-4000:])
        return 1

    prep = req({"cmd": "prepare", "source_path": str(IMG)})
    print("PREPARE:", {k: prep.get(k) for k in ("ok", "ms", "crop_bbox", "error")}, flush=True)
    if not prep.get("ok"):
        print(prep.get("trace", "")[-2000:])
        return 1

    inf = req({"cmd": "infer", "audio_path": str(WAV), "output_path": str(OUT)}, "(may take minutes)")
    print("INFER:", {k: inf.get(k) for k in ("ok", "ms", "video_path", "error")}, flush=True)
    if not inf.get("ok"):
        print(inf.get("trace", "")[-3000:])
        return 1

    req({"cmd": "quit"})
    p.wait(timeout=60)
    if not OUT.is_file() or OUT.stat().st_size < 1000:
        print("SMOKE FAIL")
        return 1
    print("SMOKE OK", OUT.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
