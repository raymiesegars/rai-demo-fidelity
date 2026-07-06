"""Wav2Lip batch lip-sync for patient loop + agent audio."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("lip-sync")

CHECKPOINT_MIN_BYTES = 100_000_000


def save_wav_int16(path: str, samples: np.ndarray, sample_rate: int = 48000) -> None:
    samples = np.asarray(samples, dtype=np.int16).flatten()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def load_video_frames_rgba(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
    cap.release()
    return frames


def checkpoint_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size < CHECKPOINT_MIN_BYTES:
        return False
    try:
        import torch

        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        return isinstance(obj, dict) and "state_dict" in obj
    except Exception:
        return False


def detect_face_box(video_path: str, wav2lip_root: Path) -> tuple[int, int, int, int]:
    """Detect face once at startup — skips ~15s face-det pass on every utterance."""
    root = str(wav2lip_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    import torch
    import face_detection  # noqa: Wav2Lip local package

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame from {video_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D,
        flip_input=False,
        device=device,
    )
    try:
        batch = np.asarray([frame])
        rect = detector.get_detections_for_batch(batch)[0]
    finally:
        del detector

    if rect is None:
        raise RuntimeError("Face not detected in patient loop — check alan-loop.mp4")

    x1, y1, x2, y2 = rect
    pady1, pady2, padx1, padx2 = 0, 10, 0, 0
    y1 = max(0, int(y1) - pady1)
    y2 = min(frame.shape[0], int(y2) + pady2)
    x1 = max(0, int(x1) - padx1)
    x2 = min(frame.shape[1], int(x2) + padx2)
    logger.info("Cached face box for loop: top=%d bottom=%d left=%d right=%d", y1, y2, x1, x2)
    return y1, y2, x1, x2


class Wav2LipEngine:
    """Runs Wav2Lip inference via cloned repo on RunPod."""

    def __init__(
        self,
        loop_video_path: str,
        wav2lip_root: str | None = None,
        checkpoint: str | None = None,
    ) -> None:
        self.loop_video_path = str(Path(loop_video_path).resolve())
        self.wav2lip_root = Path(
            wav2lip_root or os.environ.get("WAV2LIP_ROOT", "/workspace/Wav2Lip")
        )
        self.checkpoint = Path(
            checkpoint
            or os.environ.get(
                "WAV2LIP_CHECKPOINT",
                str(self.wav2lip_root / "checkpoints" / "wav2lip_gan.pth"),
            )
        )
        self.inference_py = self.wav2lip_root / "inference.py"
        self.resize_factor = int(os.environ.get("WAV2LIP_RESIZE_FACTOR", "2"))
        self._face_box: tuple[int, int, int, int] | None = None

        if self.is_ready():
            self._face_box = detect_face_box(self.loop_video_path, self.wav2lip_root)

    def is_ready(self) -> bool:
        return (
            self.inference_py.is_file()
            and checkpoint_is_valid(self.checkpoint)
            and Path(self.loop_video_path).is_file()
        )

    def sync_utterance(self, audio_wav: str) -> list[np.ndarray]:
        if not self.is_ready() or self._face_box is None:
            raise RuntimeError("Wav2Lip not ready or face box not cached")

        y1, y2, x1, x2 = self._face_box
        with tempfile.TemporaryDirectory() as tmp:
            out_mp4 = str(Path(tmp) / "lipsync.mp4")
            cmd = [
                "python",
                str(self.inference_py),
                "--checkpoint_path",
                str(self.checkpoint),
                "--face",
                self.loop_video_path,
                "--audio",
                audio_wav,
                "--outfile",
                out_mp4,
                "--fps",
                os.environ.get("TARGET_FPS", "25"),
                "--resize_factor",
                str(self.resize_factor),
                "--box",
                str(y1),
                str(y2),
                str(x1),
                str(x2),
                "--wav2lip_batch_size",
                "128",
            ]
            result = subprocess.run(
                cmd,
                cwd=str(self.wav2lip_root),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error("Wav2Lip stderr: %s", result.stderr[-2000:])
                raise RuntimeError(f"Wav2Lip failed: {result.stderr[-500:]}")

            if not Path(out_mp4).is_file():
                raise RuntimeError("Wav2Lip produced no output video")

            frames = load_video_frames_rgba(out_mp4)
            logger.info("Wav2Lip produced %d frames", len(frames))
            return frames
