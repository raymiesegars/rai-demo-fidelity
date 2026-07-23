"""Persistent Sonic worker — load SVD/Sonic once, crop prepare once, infer many.

Quality path (matches official demo + mild upgrades):
  - face crop with expand_ratio=0.5
  - min_resolution=512, inference_steps=25 (override via env)
  - RIFE frame interpolation ON
  - ffmpeg mux CRF 15 / preset slow

Protocol (JSONL on real stdout):
  → {"cmd":"prepare","source_path":"..."}
  ← {"ok":true,"cmd":"prepare","ms":...,"crop_bbox":[x1,y1,x2,y2]}
  → {"cmd":"infer","audio_path":"...","output_path":"..."}
  ← {"ok":true,"cmd":"infer","ms":...,"video_path":"..."}
  → {"cmd":"quit"}
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path


def _reply(msg: dict) -> None:
    sys.__stdout__.write(json.dumps(msg) + "\n")
    sys.__stdout__.flush()


def main() -> int:
    sys.stdout = sys.stderr

    vendor = Path(os.environ.get("SONIC_ROOT", "")).resolve()
    if not vendor.is_dir():
        vendor = Path(__file__).resolve().parents[1] / "vendor" / "Sonic"
    if not vendor.is_dir():
        _reply({"ok": False, "error": f"Sonic vendor missing: {vendor}"})
        return 1

    sys.path.insert(0, str(vendor))
    os.chdir(str(vendor))

    try:
        import imageio_ffmpeg

        ff = Path(imageio_ffmpeg.get_ffmpeg_exe()).parent
        os.environ["PATH"] = str(ff) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    min_res = int(os.environ.get("SONIC_MIN_RES", "512"))
    steps = int(os.environ.get("SONIC_STEPS", "10"))
    dynamic_scale = float(os.environ.get("SONIC_DYNAMIC_SCALE", "1.0"))
    expand_ratio = float(os.environ.get("SONIC_EXPAND_RATIO", "0.5"))
    use_rife = os.environ.get("SONIC_RIFE", "1").strip() not in ("0", "false", "False")
    seed = int(os.environ.get("SONIC_SEED", "72589"))
    device_id = int(os.environ.get("SONIC_DEVICE", "0"))

    ckpt = vendor / "checkpoints" / "Sonic" / "unet.pth"
    svd = vendor / "checkpoints" / "stable-video-diffusion-img2vid-xt"
    if not ckpt.is_file() or not svd.is_dir():
        _reply({
            "ok": False,
            "error": (
                f"Sonic checkpoints incomplete.\n  {ckpt}\n  {svd}\n"
                "Run: .\\setup_sonic.ps1"
            ),
        })
        return 1

    t0 = time.time()
    try:
        from sonic import Sonic

        pipe = Sonic(device_id=device_id, enable_interpolate_frame=use_rife)
        # Optional quality overrides on loaded yaml config
        pipe.config.num_inference_steps = steps
        pipe.config.motion_bucket_scale = dynamic_scale
        if os.environ.get("SONIC_AUDIO_GUIDANCE"):
            pipe.config.audio_guidance_scale = float(os.environ["SONIC_AUDIO_GUIDANCE"])
        if os.environ.get("SONIC_APPEARANCE_GUIDANCE"):
            g = float(os.environ["SONIC_APPEARANCE_GUIDANCE"])
            pipe.config.min_appearance_guidance_scale = g
            pipe.config.max_appearance_guidance_scale = g
        pipe.config.seed = seed
    except Exception as e:
        _reply({
            "ok": False,
            "error": f"model load failed: {e}",
            "trace": traceback.format_exc()[-2500:],
        })
        return 1

    load_s = round(time.time() - t0, 2)
    _reply({
        "ok": True,
        "cmd": "ready",
        "load_s": load_s,
        "min_res": min_res,
        "steps": steps,
        "rife": use_rife,
        "dynamic_scale": dynamic_scale,
        "expand_ratio": expand_ratio,
    })
    print(f"Sonic worker ready in {load_s}s", flush=True)

    prepared: dict | None = None
    work_root = vendor / "results" / "bench_avatar"
    work_root.mkdir(parents=True, exist_ok=True)

    def prepare(source_path: str) -> list[int]:
        nonlocal prepared
        src = Path(source_path)
        avatar_dir = work_root / "jordan"
        avatar_dir.mkdir(parents=True, exist_ok=True)
        crop_path = avatar_dir / "face_crop.png"

        face_info = pipe.preprocess(str(src), expand_ratio=expand_ratio)
        if face_info.get("face_num", 0) <= 0 or not face_info.get("crop_bbox"):
            raise RuntimeError("Sonic face detector found no face in source image")
        bbox = [int(v) for v in face_info["crop_bbox"]]
        pipe.crop_image(str(src), str(crop_path), bbox)
        prepared = {
            "source": str(src),
            "crop": str(crop_path),
            "crop_bbox": bbox,
        }
        print(f"prepare: crop_bbox={bbox}", flush=True)
        return bbox

    def infer(audio_path: str, output_path: str) -> str:
        if prepared is None:
            raise RuntimeError("call prepare first")
        out_mp4 = Path(output_path)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        if out_mp4.is_file():
            out_mp4.unlink()

        rc = pipe.process(
            prepared["crop"],
            str(audio_path),
            str(out_mp4),
            min_resolution=min_res,
            inference_steps=steps,
            dynamic_scale=dynamic_scale,
            keep_resolution=False,
            seed=seed,
        )
        if rc != 0 or not out_mp4.is_file():
            raise RuntimeError(f"Sonic process failed (rc={rc})")
        return str(out_mp4)

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
        if cmd == "prepare":
            t1 = time.time()
            try:
                bbox = prepare(req["source_path"])
                _reply({
                    "ok": True,
                    "cmd": "prepare",
                    "ms": round((time.time() - t1) * 1000),
                    "crop_bbox": bbox,
                })
            except Exception as e:
                _reply({
                    "ok": False,
                    "cmd": "prepare",
                    "error": str(e),
                    "trace": traceback.format_exc()[-2000:],
                })
            continue
        if cmd == "infer":
            t1 = time.time()
            try:
                video = infer(req["audio_path"], req["output_path"])
                _reply({
                    "ok": True,
                    "cmd": "infer",
                    "ms": round((time.time() - t1) * 1000),
                    "video_path": video,
                    "crop_bbox": (prepared or {}).get("crop_bbox"),
                })
            except Exception as e:
                _reply({
                    "ok": False,
                    "cmd": "infer",
                    "error": str(e),
                    "trace": traceback.format_exc()[-2500:],
                    "ms": round((time.time() - t1) * 1000),
                })
            continue
        _reply({"ok": False, "error": f"unknown cmd: {cmd}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
