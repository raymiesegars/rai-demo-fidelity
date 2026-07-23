"""Persistent SadTalker worker — load models once, prepare face once, infer many.

Protocol (JSONL on real stdout; library prints → stderr):
  → {"cmd":"prepare","source_path":"..."}
  ← {"ok":true,"cmd":"prepare","ms":...}
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


def _patch_torch_load() -> None:
    """PyTorch 2.6+ defaults weights_only=True; SadTalker checkpoints need False."""
    try:
        import torch

        _orig = torch.load

        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig(*args, **kwargs)

        torch.load = _load  # type: ignore[assignment]
    except Exception:
        pass


def _patch_numpy_aliases() -> None:
    """SadTalker still uses removed np.float / np.int aliases."""
    import numpy as np

    for name, alias in (
        ("float", float),
        ("int", int),
        ("bool", bool),
        ("complex", complex),
        ("object", object),
        ("str", str),
    ):
        if not hasattr(np, name):
            setattr(np, name, alias)


def main() -> int:
    sys.stdout = sys.stderr
    _patch_numpy_aliases()
    _patch_torch_load()

    vendor = Path(os.environ.get("SADTALKER_ROOT", "")).resolve()
    if not vendor.is_dir():
        vendor = Path(__file__).resolve().parents[1] / "vendor" / "SadTalker"
    if not vendor.is_dir():
        _reply({"ok": False, "error": f"SadTalker vendor missing: {vendor}"})
        return 1

    sys.path.insert(0, str(vendor))
    os.chdir(str(vendor))

    try:
        import imageio_ffmpeg

        ff = Path(imageio_ffmpeg.get_ffmpeg_exe()).parent
        os.environ["PATH"] = str(ff) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    size = int(os.environ.get("SADTALKER_SIZE", "256"))
    preprocess = os.environ.get("SADTALKER_PREPROCESS", "full").strip() or "full"
    still = os.environ.get("SADTALKER_STILL", "1").strip() not in ("0", "false", "False")
    batch_size = int(os.environ.get("SADTALKER_BATCH_SIZE", "2"))
    pose_style = int(os.environ.get("SADTALKER_POSE_STYLE", "0"))
    expression_scale = float(os.environ.get("SADTALKER_EXPRESSION_SCALE", "1.0"))
    enhancer = os.environ.get("SADTALKER_ENHANCER", "").strip() or None
    checkpoint_dir = Path(os.environ.get("SADTALKER_CHECKPOINT_DIR", vendor / "checkpoints"))

    ckpt_256 = checkpoint_dir / "SadTalker_V0.0.2_256.safetensors"
    ckpt_size = checkpoint_dir / f"SadTalker_V0.0.2_{size}.safetensors"
    if not ckpt_size.is_file():
        _reply({
            "ok": False,
            "error": (
                f"SadTalker checkpoint missing: {ckpt_size}\n"
                "Run: .\\setup_sadtalker.ps1"
            ),
        })
        return 1
    if size == 512 and not ckpt_size.is_file() and ckpt_256.is_file():
        size = 256

    t0 = time.time()
    try:
        import torch
        from src.utils.preprocess import CropAndExtract
        from src.test_audio2coeff import Audio2Coeff
        from src.facerender.animate import AnimateFromCoeff
        from src.generate_batch import get_data
        from src.generate_facerender_batch import get_facerender_data
        from src.utils.init_path import init_path

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading SadTalker on {device} size={size} preprocess={preprocess} …", flush=True)
        paths = init_path(
            str(checkpoint_dir),
            str(vendor / "src" / "config"),
            size,
            False,
            preprocess,
        )
        preprocess_model = CropAndExtract(paths, device)
        audio_to_coeff = Audio2Coeff(paths, device)
        animate_from_coeff = AnimateFromCoeff(paths, device)
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
        "size": size,
        "preprocess": preprocess,
        "still": still,
        "device": device,
        "enhancer": enhancer,
    })
    print(f"SadTalker worker ready in {load_s}s", flush=True)

    prepared: dict | None = None
    work_root = vendor / "results" / "bench_avatar"
    work_root.mkdir(parents=True, exist_ok=True)

    def prepare(source_path: str) -> None:
        nonlocal prepared
        src = Path(source_path)
        stamp = f"{src.stat().st_mtime_ns}_{src.stat().st_size}_{size}_{preprocess}"
        avatar_dir = work_root / "jordan"
        stamp_path = avatar_dir / "source_stamp.txt"
        first_frame_dir = avatar_dir / "first_frame_dir"

        if (
            avatar_dir.is_dir()
            and stamp_path.is_file()
            and stamp_path.read_text(encoding="utf-8") == stamp
            and (avatar_dir / "first_coeff.mat").is_file()
        ):
            first_coeff = avatar_dir / "first_coeff.mat"
            # Reconstruct crop paths from previous prepare layout
            crop_pic = first_frame_dir / "first_frame.png"
            # Prefer files produced by CropAndExtract
            for p in first_frame_dir.glob("*.png"):
                crop_pic = p
                break
            # crop_info is pickled alongside
            import pickle

            info_path = avatar_dir / "crop_info.pkl"
            if not info_path.is_file():
                raise RuntimeError("cached prepare incomplete — delete results/bench_avatar/jordan")
            with open(info_path, "rb") as f:
                crop_info = pickle.load(f)
            prepared = {
                "first_coeff": str(first_coeff),
                "crop_pic": str(crop_pic),
                "crop_info": crop_info,
                "source": str(src),
            }
            print("prepare: loaded cache", flush=True)
            return

        if avatar_dir.is_dir():
            shutil.rmtree(avatar_dir)
        first_frame_dir.mkdir(parents=True)

        print("3DMM Extraction for source image …", flush=True)
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
            str(src),
            str(first_frame_dir),
            preprocess,
            source_image_flag=True,
            pic_size=size,
        )
        if first_coeff_path is None:
            raise RuntimeError("SadTalker could not extract 3DMM coeffs from source image")

        # Normalize cache paths
        first_coeff = avatar_dir / "first_coeff.mat"
        shutil.copy2(first_coeff_path, first_coeff)
        import pickle

        with open(avatar_dir / "crop_info.pkl", "wb") as f:
            pickle.dump(crop_info, f)
        stamp_path.write_text(stamp, encoding="utf-8")

        prepared = {
            "first_coeff": str(first_coeff),
            "crop_pic": str(crop_pic_path),
            "crop_info": crop_info,
            "source": str(src),
        }
        print(f"prepare: ok crop={crop_pic_path}", flush=True)

    def infer(audio_path: str, output_path: str) -> str:
        if prepared is None:
            raise RuntimeError("call prepare first")

        audio_path = str(audio_path)
        out_mp4 = Path(output_path)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        save_dir = work_root / f"utt_{int(time.time() * 1000)}"
        if save_dir.is_dir():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True)

        batch = get_data(
            prepared["first_coeff"],
            audio_path,
            device,
            None,
            still=still,
        )
        coeff_path = audio_to_coeff.generate(batch, str(save_dir), pose_style, None)

        data = get_facerender_data(
            coeff_path,
            prepared["crop_pic"],
            prepared["first_coeff"],
            audio_path,
            batch_size,
            None,
            None,
            None,
            expression_scale=expression_scale,
            still_mode=still,
            preprocess=preprocess,
            size=size,
        )
        result = animate_from_coeff.generate(
            data,
            str(save_dir),
            prepared["source"],
            prepared["crop_info"],
            enhancer=enhancer,
            background_enhancer=None,
            preprocess=preprocess,
            img_size=size,
        )
        # inference.py moves result to save_dir+'.mp4'; animate returns path inside save_dir
        result_path = Path(result)
        if not result_path.is_file():
            # sometimes named save_dir.mp4 sibling
            sibling = Path(str(save_dir) + ".mp4")
            if sibling.is_file():
                result_path = sibling
            else:
                raise RuntimeError(f"SadTalker produced no video at {result}")

        shutil.copy2(result_path, out_mp4)
        try:
            shutil.rmtree(save_dir)
        except Exception:
            pass
        if Path(str(save_dir) + ".mp4").is_file():
            try:
                Path(str(save_dir) + ".mp4").unlink()
            except Exception:
                pass
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
                prepare(req["source_path"])
                _reply({"ok": True, "cmd": "prepare", "ms": round((time.time() - t1) * 1000)})
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
