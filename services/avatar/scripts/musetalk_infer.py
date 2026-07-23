"""Thin MuseTalk CLI wrapper used by backends/musetalk.py.

Tries a few known MuseTalk entry patterns so minor upstream layout changes
don't break the bench. Edit COMMANDS below if your checkout differs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vendor", required=True)
    args = ap.parse_args()

    vendor = Path(args.vendor)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "result.mp4"

    sys.path.insert(0, str(vendor))
    os.chdir(vendor)

    # Attempt 1: scripts/inference.py style (varies by MuseTalk version)
    candidates = []
    inf = vendor / "scripts" / "inference.py"
    if inf.is_file():
        candidates.append([
            sys.executable, str(inf),
            "--audio_path", args.audio,
            "--source", args.image,
            "--result_dir", str(out_dir),
        ])
        candidates.append([
            sys.executable, str(inf),
            "--audio", args.audio,
            "--image", args.image,
            "--outfile", str(out_mp4),
        ])
    rt = vendor / "scripts" / "realtime_inference.py"
    if rt.is_file():
        # Often needs a yaml config; create a minimal one
        cfg = out_dir / "bench.yaml"
        cfg.write_text(
            f"task_0:\n  audio_path: {Path(args.audio).as_posix()}\n"
            f"  video_path: {Path(args.image).as_posix()}\n"
            f"  bbox_shift: 0\n",
            encoding="utf-8",
        )
        candidates.append([
            sys.executable, "-m", "scripts.realtime_inference",
            "--inference_config", str(cfg),
            "--result_dir", str(out_dir),
            "--preparation", "True",
        ])

    errors = []
    for cmd in candidates:
        print("try:", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, cwd=str(vendor))
        if p.returncode == 0 and (list(out_dir.rglob("*.mp4")) or list(out_dir.rglob("*.png"))):
            return 0
        errors.append(f"cmd failed rc={p.returncode}: {' '.join(cmd)}")

    print("All MuseTalk invoke patterns failed:", file=sys.stderr)
    for e in errors:
        print(" ", e, file=sys.stderr)
    print(
        "Open vendor README and adjust scripts/musetalk_infer.py COMMANDS.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
