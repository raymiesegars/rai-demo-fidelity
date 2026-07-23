"""CLI to write manual fidelity / hosting scores into bench/results/<model>.json.

  .\.venv\Scripts\python.exe -m bench.score flashhead --fidelity 7 --uncanny 6 --composite 8
  .\.venv\Scripts\python.exe -m bench.score ditto --hosting 7 --notes "TRT build painful on Win"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.store import load_catalog, load_result, save_result  # noqa: E402

FID_KEYS = (
    "identity", "skin_teeth", "lip_sync", "blinks_idle",
    "head_motion", "composite_stability", "uncanny_valley", "overall",
)
HOST_KEYS = (
    "self_host", "windows_ok", "deps_fragility", "license_ok", "ops_ease", "overall",
)


def main() -> None:
    known = [m["id"] for m in load_catalog()["models"]]
    ap = argparse.ArgumentParser(description="Record manual model scores")
    ap.add_argument("model_id", choices=known)
    ap.add_argument("--fidelity", type=float, default=None, help="overall fidelity 1-10")
    ap.add_argument("--uncanny", type=float, default=None, help="uncanny valley 1-10 (10=natural)")
    ap.add_argument("--composite", type=float, default=None, help="composite stability 1-10")
    ap.add_argument("--identity", type=float, default=None)
    ap.add_argument("--lips", type=float, default=None)
    ap.add_argument("--hosting", type=float, default=None, help="overall hosting 1-10")
    ap.add_argument("--notes", type=str, default=None)
    ap.add_argument("--host-notes", type=str, default=None)
    args = ap.parse_args()

    data = load_result(args.model_id) or {
        "model_id": args.model_id,
        "status": "partial",
        "automated": {},
        "manual": {"fidelity": {}, "hosting": {}},
        "clips": [],
    }
    manual = data.setdefault("manual", {})
    fid = manual.setdefault("fidelity", {})
    host = manual.setdefault("hosting", {})

    if args.fidelity is not None:
        fid["overall"] = args.fidelity
    if args.uncanny is not None:
        fid["uncanny_valley"] = args.uncanny
    if args.composite is not None:
        fid["composite_stability"] = args.composite
    if args.identity is not None:
        fid["identity"] = args.identity
    if args.lips is not None:
        fid["lip_sync"] = args.lips
    if args.notes is not None:
        fid["notes"] = args.notes
    if args.hosting is not None:
        host["overall"] = args.hosting
    if args.host_notes is not None:
        host["notes"] = args.host_notes

    if fid.get("overall") is not None and data.get("status") in (None, "empty", "partial"):
        # stay partial until automated also filled; mark reviewed if overall set
        data["status"] = "reviewed" if data.get("automated", {}).get("gen_ms_avg") else "partial"

    saved = save_result(args.model_id, data)
    print(json.dumps(saved["manual"], indent=2))


if __name__ == "__main__":
    main()
