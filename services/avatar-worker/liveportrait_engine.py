"""FasterLivePortrait + JoyVASA — audio-driven mouth on the loop video."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("liveportrait")

SAVE_WAV = None  # set after import from lip_sync


def _save_wav(path: str, samples: np.ndarray, rate: int = 48000) -> None:
    from lip_sync import save_wav_int16

    save_wav_int16(path, samples, rate)


class LivePortraitEngine:
    """JoyVASA motion + LivePortrait render on each loop frame."""

    def __init__(self, loop_video_path: str) -> None:
        self.loop_video_path = str(Path(loop_video_path).resolve())
        self.flp_root = Path(os.environ.get("FLP_ROOT", "/workspace/FasterLivePortrait"))
        self._pipe = None
        self._joyvasa = None
        self._lock = threading.Lock()
        self._frame_count = 0

    def is_ready(self) -> bool:
        onnx = self.flp_root / "checkpoints/liveportrait_onnx/warping_spade.onnx"
        joy = (
            self.flp_root
            / "checkpoints/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt"
        )
        hubert = self.flp_root / "checkpoints/chinese-hubert-base/config.json"
        return onnx.is_file() and joy.is_file() and hubert.is_file()

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return

        if not self.is_ready():
            raise RuntimeError(
                f"FasterLivePortrait not installed at {self.flp_root}. "
                "Run: bash setup_liveportrait.sh"
            )

        root = str(self.flp_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        os.chdir(root)

        from omegaconf import OmegaConf
        from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
        from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline

        cfg_name = os.environ.get("FLP_CFG", "configs/onnx_infer.yaml")
        cfg_path = self.flp_root / cfg_name
        cfg = OmegaConf.load(str(cfg_path))
        cfg.infer_params.flag_pasteback = True
        cfg.infer_params.animation_region = os.environ.get("FLP_ANIMATION_REGION", "exp")
        cfg.infer_params.flag_relative_motion = True
        cfg.infer_params.flag_stitching = True

        # Resolve checkpoint paths relative to FLP root.
        for key in ("models", "animal_models"):
            if key not in cfg:
                continue
            for model in cfg[key].values():
                if hasattr(model, "model_path"):
                    mp = model.model_path
                    if isinstance(mp, str) and not Path(mp).is_absolute():
                        model.model_path = str(self.flp_root / mp.lstrip("./"))
                    elif isinstance(mp, list):
                        model.model_path = [
                            str(self.flp_root / p.lstrip("./")) for p in mp
                        ]
        jm = cfg.joyvasa_models
        for field in ("motion_model_path", "audio_model_path", "motion_template_path"):
            val = getattr(jm, field)
            if val and not Path(val).is_absolute():
                setattr(jm, field, str(self.flp_root / str(val).lstrip("./")))

        logger.info("Loading LivePortrait pipeline (onnx)…")
        self._pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
        if not self._pipe.prepare_source(self.loop_video_path, realtime=True):
            raise RuntimeError(f"No face detected in loop video: {self.loop_video_path}")

        self._joyvasa = JoyVASAAudio2MotionPipeline(
            motion_model_path=jm.motion_model_path,
            audio_model_path=jm.audio_model_path,
            motion_template_path=jm.motion_template_path,
            cfg_mode=cfg.infer_params.cfg_mode,
            cfg_scale=float(cfg.infer_params.cfg_scale),
        )
        logger.info(
            "LivePortrait ready — %d source frames, animation_region=%s",
            len(self._pipe.src_imgs),
            cfg.infer_params.animation_region,
        )

    def render_wav(
        self, wav_path: str, start_frame_idx: int, reset_motion: bool = False
    ) -> list[np.ndarray]:
        """Return RGBA full frames for this audio chunk."""
        with self._lock:
            self._ensure_loaded()
            assert self._pipe is not None and self._joyvasa is not None
            t0 = time.monotonic()
            motion = self._joyvasa.gen_motion_sequence(wav_path)
            n_frames = int(motion["n_frames"])
            if n_frames == 0:
                return []

            src_n = len(self._pipe.src_imgs)
            out: list[np.ndarray] = []
            for i in range(n_frames):
                idx = (start_frame_idx + i) % src_n
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
                first = reset_motion and i == 0
                _, out_org = self._pipe.run_with_pkl(
                    dri,
                    self._pipe.src_imgs[idx],
                    self._pipe.src_infos[idx],
                    first_frame=first,
                    realtime=True,
                )
                if out_org is None:
                    continue
                if isinstance(out_org, np.ndarray):
                    bgr = out_org
                else:
                    bgr = out_org.cpu().numpy() if hasattr(out_org, "cpu") else np.asarray(out_org)
                if bgr.dtype != np.uint8:
                    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
                if bgr.ndim == 3 and bgr.shape[2] == 3:
                    rgba = cv2.cvtColor(bgr, cv2.COLOR_RGB2RGBA)
                else:
                    rgba = bgr
                out.append(rgba)

            logger.info(
                "LivePortrait rendered %d frames in %.2fs",
                len(out),
                time.monotonic() - t0,
            )
            return out
