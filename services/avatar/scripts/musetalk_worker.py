"""Persistent MuseTalk 1.5 worker — load UNet once, prepare avatar once, infer many.

Uses the realtime Avatar path (prep latents/masks ahead of time) so each utterance
only runs Whisper + UNet + VAE decode — closest to MuseTalk's intended speed.

Protocol (JSONL on real stdout; library prints → stderr):
  → {"cmd":"prepare","source_path":"...","bbox_shift":0}
  ← {"ok":true,"cmd":"prepare","ms":...}
  → {"cmd":"infer","audio_path":"...","output_path":"..."}
  ← {"ok":true,"cmd":"infer","ms":...,"video_path":"..."}
  → {"cmd":"quit"}
"""

from __future__ import annotations

import copy
import glob
import json
import os
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np


def _reply(msg: dict) -> None:
    sys.__stdout__.write(json.dumps(msg) + "\n")
    sys.__stdout__.flush()


def _face_bbox_mediapipe(frame_bgr: np.ndarray, bbox_shift: int = 0) -> list[int]:
    """Match MuseTalk's DWPose landmark bbox using MediaPipe Face Mesh.

    Official MuseTalk (preprocessing.get_landmark_and_bbox):
      half_face = face_landmark[29]   # mid-nose
      half_face_dist = chin_y - half_face_y
      upper_bond = half_face_y - half_face_dist
      box = (min_x, upper_bond, max_x, chin_y)

    Critical: do NOT force a square crop. Expanding to square on a wide face
    pulled y1 near the top of the frame so the mouth was a tiny smear in the
    256² latent — that was the extreme blur.
    """
    import mediapipe as mp

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as mesh:
        res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as det:
            dres = det.process(rgb)
        if not dres.detections:
            raise RuntimeError("MediaPipe found no face in MuseTalk source image")
        box = dres.detections[0].location_data.relative_bounding_box
        x1 = int(max(0, box.xmin * w))
        y1 = int(max(0, box.ymin * h))
        x2 = int(min(w, (box.xmin + box.width) * w))
        y2 = int(min(h, (box.ymin + box.height) * h))
        # Detector boxes are forehead→chin; pull upper bond down like MuseTalk.
        mid = y1 + int((y2 - y1) * 0.42)
        half = y2 - mid
        y1 = max(0, mid - half + bbox_shift)
        return [x1, y1, x2, y2]

    pts = np.array(
        [[lm.x * w, lm.y * h] for lm in res.multi_face_landmarks[0].landmark],
        dtype=np.float32,
    )
    # Face oval for left/right extent (closest to DWPose face kps x-range)
    oval = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]
    face = pts[oval]
    # MediaPipe: 1 ≈ nose tip (≈ DWPose face landmark 29–30), 152 = chin
    half_face = pts[1].copy()
    half_face[1] = half_face[1] + float(bbox_shift)
    chin_y = float(np.max(face[:, 1]))
    half_face_dist = chin_y - float(half_face[1])
    upper_bond = max(0.0, float(half_face[1]) - half_face_dist)
    x1 = int(np.min(face[:, 0]))
    x2 = int(np.max(face[:, 0]))
    y1 = int(upper_bond)
    y2 = int(chin_y)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"invalid face box {[x1, y1, x2, y2]}")
    return [x1, y1, x2, y2]


def _unsharp_bgr(img: np.ndarray, amount: float = 0.55, sigma: float = 1.0) -> np.ndarray:
    """Recover a bit of mouth edge after VAE decode (256² → face size)."""
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = img.astype(np.float32) + amount * (img.astype(np.float32) - blur.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _read_imgs(img_list: list[str]) -> list[np.ndarray]:
    """Local copy of MuseTalk read_imgs — avoids importing preprocessing (mmpose)."""
    frames = []
    for p in img_list:
        im = cv2.imread(p)
        if im is not None:
            frames.append(im)
    return frames


def main() -> int:
    sys.stdout = sys.stderr

    vendor = Path(os.environ.get("MUSETALK_ROOT", "")).resolve()
    if not vendor.is_dir():
        vendor = Path(__file__).resolve().parents[1] / "vendor" / "MuseTalk"
    if not vendor.is_dir():
        _reply({"ok": False, "error": f"MuseTalk vendor missing: {vendor}"})
        return 1

    sys.path.insert(0, str(vendor))
    os.chdir(str(vendor))

    # Prefer imageio-ffmpeg on PATH
    try:
        import imageio_ffmpeg

        ff = Path(imageio_ffmpeg.get_ffmpeg_exe()).parent
        os.environ["PATH"] = str(ff) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    version = os.environ.get("MUSETALK_VERSION", "v15")
    # FP16 is required for realtime-ish VRAM/speed on a single 24GB card.
    # Quality comes from crop + lossless frame handoff, not fp32.
    use_fp16 = os.environ.get("MUSETALK_FP16", "1").strip() not in ("0", "false", "False")
    batch_size = int(os.environ.get("MUSETALK_BATCH_SIZE", "8"))
    fps = int(os.environ.get("MUSETALK_FPS", "25"))
    extra_margin = int(os.environ.get("MUSETALK_EXTRA_MARGIN", "10"))
    parsing_mode = os.environ.get("MUSETALK_PARSING_MODE", "jaw")
    bbox_shift_default = int(os.environ.get("MUSETALK_BBOX_SHIFT", "0"))
    # Tighter blend than MuseTalk defaults: less soft mush around the mouth.
    blend_expand = float(os.environ.get("MUSETALK_BLEND_EXPAND", "1.25"))
    upper_boundary = float(os.environ.get("MUSETALK_UPPER_BOUNDARY", "0.58"))
    sharpen = float(os.environ.get("MUSETALK_SHARPEN", "0.55"))
    # Include quality knobs in avatar cache key so toggling FP16/crop rebuilds.
    quality_tag = (
        f"fp16={int(use_fp16)}_margin={extra_margin}_mode={parsing_mode}"
        f"_mesh=3_exp={blend_expand}_ub={upper_boundary}"
    )

    if version == "v15":
        unet_path = vendor / "models" / "musetalkV15" / "unet.pth"
        unet_cfg = vendor / "models" / "musetalkV15" / "musetalk.json"
    else:
        unet_path = vendor / "models" / "musetalk" / "pytorch_model.bin"
        unet_cfg = vendor / "models" / "musetalk" / "musetalk.json"
    whisper_dir = vendor / "models" / "whisper"

    if not unet_path.is_file() or not unet_cfg.is_file():
        _reply({
            "ok": False,
            "error": (
                f"MuseTalk {version} weights missing.\n"
                f"  expected: {unet_path}\n"
                f"  expected: {unet_cfg}\n"
                "Run: .\\setup_musetalk.ps1"
            ),
        })
        return 1

    t0 = time.time()
    try:
        import torch
        from transformers import WhisperModel
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.blending import get_image_blending, get_image_prepare_material
        from musetalk.utils.face_parsing import FaceParsing
        from musetalk.utils.utils import datagen, load_all_model
        # NOTE: do NOT import musetalk.utils.preprocessing — it hard-requires mmpose.

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"loading MuseTalk models on {device} …", flush=True)
        vae, unet, pe = load_all_model(
            unet_model_path=str(unet_path),
            vae_type="sd-vae",
            unet_config=str(unet_cfg),
            device=device,
        )
        timesteps = torch.tensor([0], device=device)
        if use_fp16 and device.type == "cuda":
            pe = pe.half()
            vae.vae = vae.vae.half()
            unet.model = unet.model.half()
        pe = pe.to(device)
        vae.vae = vae.vae.to(device)
        unet.model = unet.model.to(device)

        audio_processor = AudioProcessor(feature_extractor_path=str(whisper_dir))
        weight_dtype = unet.model.dtype
        whisper = WhisperModel.from_pretrained(str(whisper_dir))
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)

        if version == "v15":
            fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        else:
            fp = FaceParsing()
    except Exception as e:
        _reply({"ok": False, "error": f"model load failed: {e}", "trace": traceback.format_exc()[-2500:]})
        return 1

    load_s = round(time.time() - t0, 2)
    _reply({
        "ok": True,
        "cmd": "ready",
        "load_s": load_s,
        "version": version,
        "fp16": use_fp16,
        "batch_size": batch_size,
        "device": str(device),
    })
    print(f"MuseTalk worker ready in {load_s}s", flush=True)

    # Avatar state
    avatar: dict | None = None
    work_root = vendor / "results" / "bench_avatar"
    work_root.mkdir(parents=True, exist_ok=True)

    def prepare(source_path: str, bbox_shift: int) -> None:
        nonlocal avatar
        src = Path(source_path)
        avatar_dir = work_root / "jordan"
        # Always rebuild when source changes (hash by mtime+size)
        stamp = f"{src.stat().st_mtime_ns}_{src.stat().st_size}_{bbox_shift}_{version}_{quality_tag}"
        stamp_path = avatar_dir / "source_stamp.txt"
        if avatar_dir.is_dir() and stamp_path.is_file() and stamp_path.read_text(encoding="utf-8") == stamp:
            # reload from disk
            pass
        else:
            if avatar_dir.is_dir():
                shutil.rmtree(avatar_dir)
            full_imgs = avatar_dir / "full_imgs"
            mask_dir = avatar_dir / "mask"
            full_imgs.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            # Still image → single-frame cycle (MuseTalk accepts image as video_path)
            img = cv2.imread(str(src))
            if img is None:
                raise RuntimeError(f"could not read source image: {src}")
            cv2.imwrite(str(full_imgs / "00000000.png"), img)

            input_img_list = sorted(glob.glob(str(full_imgs / "*.png")))
            frame_list = [cv2.imread(p) for p in input_img_list]
            frame_list = [f for f in frame_list if f is not None]
            if not frame_list:
                raise RuntimeError("no frames for MuseTalk prepare")
            # Prefer DWPose path when mmpose is installed; else MediaPipe (Windows).
            try:
                from musetalk.utils.preprocessing import get_landmark_and_bbox

                coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
            except Exception as e:
                print(f"DWPose unavailable ({e}); using MediaPipe face crop", flush=True)
                coord_list = [_face_bbox_mediapipe(frame_list[0], bbox_shift)]

            input_latent_list = []
            coord_placeholder = (0.0, 0.0, 0.0, 0.0)
            for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                if version == "v15":
                    y2 = min(y2 + extra_margin, frame.shape[0])
                    coord_list[idx] = [x1, y1, x2, y2]
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                input_latent_list.append(vae.get_latents_for_unet(crop))
            print(f"face box={[int(v) for v in coord_list[0]]} on {frame_list[0].shape[1]}x{frame_list[0].shape[0]}", flush=True)

            if not input_latent_list:
                raise RuntimeError("MuseTalk prepare: no valid face latents")

            frame_cycle = frame_list + frame_list[::-1]
            coord_cycle = coord_list + coord_list[::-1]
            # Latents must match coord cycle length (still image → duplicate).
            while len(input_latent_list) < len(coord_cycle):
                input_latent_list = input_latent_list + input_latent_list[::-1]
            latent_cycle = input_latent_list[: len(coord_cycle)]
            mask_coords = []
            masks = []
            mode = parsing_mode if version == "v15" else "raw"
            for i, frame in enumerate(frame_cycle):
                cv2.imwrite(str(full_imgs / f"{i:08d}.png"), frame)
                x1, y1, x2, y2 = [int(v) for v in coord_cycle[i]]
                mask, crop_box = get_image_prepare_material(
                    frame,
                    [x1, y1, x2, y2],
                    fp=fp,
                    mode=mode,
                    upper_boundary_ratio=upper_boundary,
                    expand=blend_expand,
                )
                # MuseTalk's default 0.1*size gaussian is extremely soft on HD
                # stills — re-harden then feather lightly so lips stay crisp.
                if mask is not None and mask.size:
                    hard = (mask > 96).astype(np.uint8) * 255
                    k = max(3, int(0.03 * max(mask.shape[:2])) | 1)
                    mask = cv2.GaussianBlur(hard, (k, k), 0)
                cv2.imwrite(str(mask_dir / f"{i:08d}.png"), mask)
                mask_coords.append(crop_box)
                masks.append(mask)

            # Debug: save the face crop fed into the 256 VAE so we can spot bad boxes.
            try:
                x1, y1, x2, y2 = [int(v) for v in coord_cycle[0]]
                dbg = frame_cycle[0][y1:y2, x1:x2]
                cv2.imwrite(str(avatar_dir / "debug_face_crop.png"), dbg)
                cv2.imwrite(
                    str(avatar_dir / "debug_face_256.png"),
                    cv2.resize(dbg, (256, 256), interpolation=cv2.INTER_LANCZOS4),
                )
            except Exception:
                pass

            with open(avatar_dir / "coords.pkl", "wb") as f:
                pickle.dump(coord_cycle, f)
            with open(avatar_dir / "mask_coords.pkl", "wb") as f:
                pickle.dump(mask_coords, f)
            torch.save(latent_cycle, avatar_dir / "latents.pt")
            stamp_path.write_text(stamp, encoding="utf-8")

        # Load into memory
        try:
            latent_cycle = torch.load(avatar_dir / "latents.pt", weights_only=False)
        except TypeError:
            latent_cycle = torch.load(avatar_dir / "latents.pt")
        with open(avatar_dir / "coords.pkl", "rb") as f:
            coord_cycle = pickle.load(f)
        with open(avatar_dir / "mask_coords.pkl", "rb") as f:
            mask_coords = pickle.load(f)
        imgs = sorted(glob.glob(str(avatar_dir / "full_imgs" / "*.png")))
        masks_p = sorted(glob.glob(str(avatar_dir / "mask" / "*.png")))
        frame_cycle = _read_imgs(imgs)
        masks = _read_imgs(masks_p)
        avatar = {
            "dir": avatar_dir,
            "latents": latent_cycle,
            "coords": coord_cycle,
            "frames": frame_cycle,
            "masks": masks,
            "mask_coords": mask_coords,
        }

    def infer(audio_path: str, output_path: str) -> str:
        if avatar is None:
            raise RuntimeError("call prepare first")
        audio_path = str(audio_path)
        out_mp4 = Path(output_path)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = avatar["dir"] / "tmp"
        if tmp_dir.is_dir():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        whisper_input_features, librosa_length = audio_processor.get_audio_feature(
            audio_path, weight_dtype=weight_dtype
        )
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        video_num = len(whisper_chunks)
        print(f"infer: {video_num} frames, batch={batch_size}, fp16={use_fp16}", flush=True)
        recon_frames: list = []
        gen = datagen(whisper_chunks, avatar["latents"], batch_size)
        n_done = 0
        for whisper_batch, latent_batch in gen:
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
            pred = unet.model(
                latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            pred = pred.to(device=device, dtype=vae.vae.dtype)
            recon_frames.extend(vae.decode_latents(pred))
            n_done = min(len(recon_frames), video_num)
            if n_done == video_num or n_done % max(batch_size * 2, 16) == 0:
                print(f"infer: unet {n_done}/{video_num}", flush=True)

        frames_out: list[np.ndarray] = []
        for i, res_frame in enumerate(recon_frames[:video_num]):
            bbox = avatar["coords"][i % len(avatar["coords"])]
            ori = copy.deepcopy(avatar["frames"][i % len(avatar["frames"])])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(
                    res_frame.astype(np.uint8),
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                res_frame = _unsharp_bgr(res_frame, amount=sharpen)
            except Exception:
                continue
            mask = avatar["masks"][i % len(avatar["masks"])]
            mask_box = avatar["mask_coords"][i % len(avatar["mask_coords"])]
            combine = get_image_blending(ori, res_frame, bbox, mask, mask_box)
            frames_out.append(combine)

        if not frames_out:
            raise RuntimeError("MuseTalk produced 0 frames")

        # Single .npy stack — lossless + far faster than PNG-per-frame or mp4v.
        out_npy = Path(str(out_mp4) + ".npy")
        stack = np.stack(frames_out, axis=0)
        np.save(str(out_npy), stack)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"infer: wrote {len(frames_out)} frames → {out_npy.name}", flush=True)
        return str(out_npy)

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
                prepare(req["source_path"], int(req.get("bbox_shift", bbox_shift_default)))
                _reply({"ok": True, "cmd": "prepare", "ms": round((time.time() - t1) * 1000)})
            except Exception as e:
                _reply({"ok": False, "cmd": "prepare", "error": str(e), "trace": traceback.format_exc()[-2000:]})
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
