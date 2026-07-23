"""Persistent FasterLivePortrait + JoyVASA worker.

Loads ONNX LivePortrait + JoyVASA once, prepares a still once, renders many
audio utterances to mp4 (pasteback onto full image).

Protocol (JSONL on real stdout; library prints → stderr):
  → {"cmd":"prepare","source_path":"..."}
  ← {"ok":true,"cmd":"prepare","ms":...}
  → {"cmd":"infer","audio_path":"...","output_path":"..."}
  ← {"ok":true,"cmd":"infer","ms":...,"video_path":"...","n_frames":N}
  → {"cmd":"quit"}
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


def _patch_torch_load() -> None:
    try:
        import torch

        _orig = torch.load

        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig(*args, **kwargs)

        torch.load = _load  # type: ignore[assignment]
    except Exception:
        pass


def main() -> int:
    sys.stdout = sys.stderr
    _patch_torch_load()
    os.environ.setdefault("TRANSFORMERS_ATTN_IMPLEMENTATION", "eager")

    # Put pip nvidia-* DLL dirs on PATH so onnxruntime CUDA EP can load.
    try:
        import glob
        import site

        bins: list[str] = []
        for sp in site.getsitepackages():
            for name in (
                "cudnn", "cublas", "cuda_runtime", "cufft",
                "curand", "cusolver", "cusparse", "nvjitlink",
            ):
                bins.extend(glob.glob(os.path.join(sp, "nvidia", name, "bin")))
        if bins:
            os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    vendor = Path(os.environ.get("FLP_ROOT", "")).resolve()
    if not vendor.is_dir():
        vendor = Path(__file__).resolve().parents[1] / "vendor" / "FasterLivePortrait"
    if not vendor.is_dir():
        _reply({"ok": False, "error": f"FasterLivePortrait vendor missing: {vendor}"})
        return 1

    # Prefer scripts next to this worker, then vendor-local patch.
    root = Path(__file__).resolve().parents[1]
    patch_py = root / "scripts" / "patch_flp_compat.py"
    if patch_py.is_file():
        sys.path.insert(0, str(patch_py.parent))
        from patch_flp_compat import apply_patches

        apply_patches(vendor)

    sys.path.insert(0, str(vendor))
    os.chdir(str(vendor))

    try:
        import imageio_ffmpeg

        ff = Path(imageio_ffmpeg.get_ffmpeg_exe()).parent
        os.environ["PATH"] = str(ff) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    cfg_name = os.environ.get("FLP_CFG", "configs/onnx_infer.yaml")
    anim_region = os.environ.get("FLP_ANIMATION_REGION", "exp")
    cfg_scale = float(os.environ.get("FLP_CFG_SCALE", "1.2"))
    fps = int(os.environ.get("LIVEPORTRAIT_FPS", "25"))

    onnx = vendor / "checkpoints" / "liveportrait_onnx" / "warping_spade.onnx"
    joy = (
        vendor
        / "checkpoints"
        / "JoyVASA"
        / "motion_generator"
        / "motion_generator_hubert_chinese.pt"
    )
    hubert = vendor / "checkpoints" / "chinese-hubert-base" / "config.json"
    if not onnx.is_file() or not joy.is_file() or not hubert.is_file():
        _reply({
            "ok": False,
            "error": (
                f"LivePortrait checkpoints incomplete.\n  {onnx}\n  {joy}\n  {hubert}\n"
                "Run: .\\setup_liveportrait.ps1"
            ),
        })
        return 1

    t0 = time.time()
    try:
        from omegaconf import OmegaConf
        from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
        from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline
        import cv2
        import numpy as np
    except Exception as e:
        _reply({"ok": False, "error": f"Import failed: {e}", "trace": traceback.format_exc()})
        return 1

    cfg_path = vendor / cfg_name
    if not cfg_path.is_file():
        _reply({"ok": False, "error": f"Config missing: {cfg_path}"})
        return 1

    cfg = OmegaConf.load(str(cfg_path))
    cfg.infer_params.flag_pasteback = True
    cfg.infer_params.animation_region = anim_region
    cfg.infer_params.flag_relative_motion = True
    cfg.infer_params.flag_stitching = True
    cfg.infer_params.cfg_scale = cfg_scale

    for key in ("models", "animal_models"):
        if key not in cfg:
            continue
        for model in cfg[key].values():
            if hasattr(model, "model_path"):
                mp = model.model_path
                if isinstance(mp, str) and not Path(mp).is_absolute():
                    model.model_path = str(vendor / mp.lstrip("./"))
                elif isinstance(mp, list):
                    model.model_path = [str(vendor / p.lstrip("./")) for p in mp]

    jm = cfg.joyvasa_models
    for field in ("motion_model_path", "audio_model_path", "motion_template_path"):
        val = getattr(jm, field)
        if val and not Path(val).is_absolute():
            setattr(jm, field, str(vendor / str(val).lstrip("./")))

    try:
        pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
        joyvasa = JoyVASAAudio2MotionPipeline(
            motion_model_path=jm.motion_model_path,
            audio_model_path=jm.audio_model_path,
            motion_template_path=jm.motion_template_path,
            cfg_mode=cfg.infer_params.cfg_mode,
            cfg_scale=float(cfg.infer_params.cfg_scale),
        )
    except Exception as e:
        _reply({"ok": False, "error": f"Load failed: {e}", "trace": traceback.format_exc()})
        return 1

    _reply({
        "ok": True,
        "cmd": "ready",
        "load_s": round(time.time() - t0, 1),
        "cfg": cfg_name,
        "animation_region": anim_region,
    })

    prepared_key: str | None = None

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
        if cmd == "quit":
            _reply({"ok": True, "cmd": "quit"})
            return 0

        if cmd == "prepare":
            try:
                src = Path(req["source_path"]).resolve()
                if not src.is_file():
                    raise FileNotFoundError(str(src))
                key = str(src)
                if prepared_key == key:
                    _reply({"ok": True, "cmd": "prepare", "ms": 0, "cached": True})
                    continue
                t1 = time.time()
                ok = pipe.prepare_source(str(src), realtime=True)
                if not ok:
                    raise RuntimeError(f"No face detected in source: {src}")
                prepared_key = key
                _reply({
                    "ok": True,
                    "cmd": "prepare",
                    "ms": int((time.time() - t1) * 1000),
                    "src_frames": len(pipe.src_imgs),
                })
            except Exception as e:
                prepared_key = None
                _reply({
                    "ok": False,
                    "cmd": "prepare",
                    "error": str(e),
                    "trace": traceback.format_exc(),
                })
            continue

        if cmd == "infer":
            try:
                if prepared_key is None:
                    raise RuntimeError("Call prepare first")
                audio_path = Path(req["audio_path"]).resolve()
                output_path = Path(req["output_path"]).resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not audio_path.is_file():
                    raise FileNotFoundError(str(audio_path))

                # Face-crop only (realtime=True skips slow full-frame pasteback).
                # Server composites the crop onto the locked Jordan still.
                face_only = os.environ.get("LIVEPORTRAIT_FACE_ONLY", "1").strip() not in (
                    "0", "false", "False",
                )
                max_frames = int(os.environ.get("LIVEPORTRAIT_MAX_FRAMES", "0") or "0")

                t1 = time.time()
                motion = joyvasa.gen_motion_sequence(str(audio_path))
                n_frames = int(motion["n_frames"])
                if n_frames <= 0:
                    raise RuntimeError("JoyVASA produced 0 frames")

                # Optional downsample for long utterances (keeps lips roughly in sync
                # by stretching fewer frames across the audio at playback time).
                indices = list(range(n_frames))
                if max_frames > 0 and n_frames > max_frames:
                    step = max(1, int(round(n_frames / max_frames)))
                    indices = list(range(0, n_frames, step))
                    print(
                        f"LivePortrait: capping {n_frames} → {len(indices)} frames (step={step})",
                        flush=True,
                    )

                src_n = len(pipe.src_imgs)
                frames: list = []
                for fi, i in enumerate(indices):
                    if fi % 10 == 0 or fi == len(indices) - 1:
                        print(
                            f"LivePortrait render {fi + 1}/{len(indices)} "
                            f"(motion {i}/{n_frames}) face_only={face_only}",
                            flush=True,
                        )
                    idx = i % src_n
                    eyes = (
                        motion["c_eyes_lst"][i]
                        if motion.get("c_eyes_lst") and len(motion["c_eyes_lst"]) > i
                        else None
                    )
                    lip = (
                        motion["c_lip_lst"][i]
                        if motion.get("c_lip_lst") and len(motion["c_lip_lst"]) > i
                        else None
                    )
                    dri = [motion["motion"][i], eyes, lip]
                    out_crop, out_pstbk = pipe.run_with_pkl(
                        dri,
                        pipe.src_imgs[idx],
                        pipe.src_infos[idx],
                        first_frame=(fi == 0),
                        # realtime=True → skip paste_back_pytorch (very slow on stills)
                        realtime=face_only,
                    )
                    out_org = out_crop if face_only else out_pstbk
                    if out_org is None:
                        continue
                    if hasattr(out_org, "cpu"):
                        bgr = out_org.cpu().numpy()
                    else:
                        bgr = np.asarray(out_org)
                    if bgr.dtype != np.uint8:
                        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
                    # FLP crops/pasteback are RGB
                    if bgr.ndim == 3 and bgr.shape[2] == 3:
                        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
                    frames.append(bgr)

                if not frames:
                    raise RuntimeError("LivePortrait produced no frames")

                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"VideoWriter failed: {output_path}")
                for fr in frames:
                    if fr.shape[0] != h or fr.shape[1] != w:
                        fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR)
                    writer.write(fr)
                writer.release()

                _reply({
                    "ok": True,
                    "cmd": "infer",
                    "ms": int((time.time() - t1) * 1000),
                    "video_path": str(output_path),
                    "n_frames": len(frames),
                    "motion_frames": n_frames,
                    "face_only": face_only,
                })
            except Exception as e:
                _reply({
                    "ok": False,
                    "cmd": "infer",
                    "error": str(e),
                    "trace": traceback.format_exc(),
                })
            continue

        _reply({"ok": False, "error": f"unknown cmd: {cmd}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
