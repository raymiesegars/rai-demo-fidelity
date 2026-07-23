"""LiveTalk warm JSONL worker (image + audio → mp4).

Loads CausalInferencePipeline once, then serves prepare/infer/quit.
Run from LiveTalk vendor root with LIVETALK_ROOT set.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml


def _reply(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _snap_duration(seconds: float) -> int:
    """LiveTalk wants video_duration = 3n+2 (5, 8, 11, …)."""
    s = max(2.0, float(seconds))
    n = max(0, int(round((s - 2.0) / 3.0)))
    return 3 * n + 2


def main() -> int:
    root = Path(os.environ.get("LIVETALK_ROOT", ".")).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "OmniAvatar"))

    cfg_path = root / "configs" / "causal_inference.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Absolute checkpoint paths
    for key in ("text_encoder_path", "dit_path", "vae_path", "wav2vec_path"):
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str((root / p).resolve())

    example_img = root / "examples" / "inference" / "example1.jpg"
    example_wav = root / "examples" / "inference" / "example1.wav"
    if not example_img.is_file():
        # Fallback: any jpg under examples
        imgs = list((root / "examples").rglob("*.jpg")) + list((root / "examples").rglob("*.png"))
        if imgs:
            example_img = imgs[0]
    cfg["image_path"] = str(example_img) if example_img.is_file() else str(root)
    cfg["audio_path"] = str(example_wav) if example_wav.is_file() else str(root)
    cfg["output_path"] = str(root / "_boot_out.mp4")
    cfg["video_duration"] = 5
    cfg.setdefault(
        "prompt",
        os.environ.get(
            "LIVETALK_PROMPT",
            "A realistic video of a person speaking directly to the camera.",
        ),
    )

    boot_yaml = root / "_worker_boot.yaml"
    with open(boot_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    sys.argv = ["livetalk_worker", "--config", str(boot_yaml)]

    t0 = time.time()
    try:
        import torch
        from scripts.inference_example import CausalInferencePipeline
        from OmniAvatar.utils.args_config import args as lt_args

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = CausalInferencePipeline.from_pretrained(args=lt_args, device=device)
        dtype = torch.bfloat16 if getattr(lt_args, "dtype", "bf16") == "bf16" else torch.float16
    except Exception as e:
        _reply({"ok": False, "cmd": "ready", "error": str(e), "trace": traceback.format_exc()})
        return 1

    prepared_image: str | None = None
    _reply({
        "ok": True,
        "cmd": "ready",
        "load_s": round(time.time() - t0, 1),
        "device": str(device),
        "fps": int(getattr(lt_args, "fps", 16)),
    })

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
                prepared_image = str(src)
                _reply({"ok": True, "cmd": "prepare", "source": prepared_image})
            except Exception as e:
                _reply({
                    "ok": False,
                    "cmd": "prepare",
                    "error": str(e),
                    "trace": traceback.format_exc(),
                })
            continue

        if cmd == "infer":
            try:
                if not prepared_image:
                    raise RuntimeError("Call prepare first")
                audio_path = Path(req["audio_path"]).resolve()
                output_path = Path(req["output_path"]).resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not audio_path.is_file():
                    raise FileNotFoundError(str(audio_path))

                import librosa
                import soundfile as sf

                audio, sr = librosa.load(str(audio_path), sr=16000)
                dur = len(audio) / float(sr)
                video_duration = _snap_duration(dur)
                # Cap long utterances for bench latency
                max_dur = int(os.environ.get("LIVETALK_MAX_DURATION", "8") or "8")
                if video_duration > max_dur:
                    # snap max_dur down to 3n+2
                    video_duration = _snap_duration(float(max_dur))
                    audio = audio[: int(video_duration * sr)]
                    trim = output_path.parent / f"_trim_{output_path.stem}.wav"
                    sf.write(str(trim), audio, 16000)
                    audio_path = trim

                fps = int(getattr(lt_args, "fps", 16))
                num_frames = (video_duration * fps + 4) // 4
                prompt = req.get("prompt") or getattr(lt_args, "prompt", "") or cfg["prompt"]

                t1 = time.time()
                noise = torch.randn(
                    [1, num_frames, 16, 64, 64], device=device, dtype=dtype
                )
                video = pipeline(
                    noise=noise,
                    text_prompts=prompt,
                    image_path=prepared_image,
                    audio_path=str(audio_path),
                    initial_latent=None,
                    return_latents=False,
                )
                # video: [B,T,C,H,W] float 0..1
                frames = (video.squeeze(0).permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(
                    np.uint8
                )
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"VideoWriter failed: {output_path}")
                for fr in frames:
                    bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                    writer.write(bgr)
                writer.release()

                _reply({
                    "ok": True,
                    "cmd": "infer",
                    "ms": int((time.time() - t1) * 1000),
                    "video_path": str(output_path),
                    "n_frames": len(frames),
                    "video_duration": video_duration,
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
