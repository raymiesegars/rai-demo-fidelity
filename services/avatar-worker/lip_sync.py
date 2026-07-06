"""Wav2Lip batch lip-sync for patient loop + agent audio."""

from __future__ import annotations

import json
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


def ensure_wav2lip_patched(wav2lip_root: Path) -> None:
    """Apply and verify Wav2Lip patches (idempotent)."""
    inference = wav2lip_root / "inference.py"
    if not inference.is_file():
        raise RuntimeError(f"Wav2Lip not found at {wav2lip_root}")
    text = inference.read_text(encoding="utf-8")
    if "--boxes_file" in text and "args.boxes_file" in text:
        return
    patch_script = Path(__file__).resolve().parent / "patch_wav2lip.py"
    result = subprocess.run(
        [sys.executable, str(patch_script), str(wav2lip_root)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("patch_wav2lip failed: %s", result.stderr)
        raise RuntimeError("Failed to patch Wav2Lip")
    text = inference.read_text(encoding="utf-8")
    if "--boxes_file" not in text or "args.boxes_file" not in text:
        raise RuntimeError("Wav2Lip boxes_file patch missing after patch_wav2lip.py")


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


def _import_face_detector(wav2lip_root: Path):
    root = str(wav2lip_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import torch
    import face_detection  # noqa: Wav2Lip local package

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D,
        flip_input=False,
        device=device,
    )
    return detector, torch


def _smooth_boxes(boxes: np.ndarray, window: int = 5) -> np.ndarray:
    smoothed = boxes.copy()
    for i in range(len(smoothed)):
        if i + window > len(smoothed):
            chunk = smoothed[len(smoothed) - window :]
        else:
            chunk = smoothed[i : i + window]
        smoothed[i] = np.mean(chunk, axis=0)
    return smoothed


def compute_per_frame_boxes(video_path: str, wav2lip_root: Path) -> list[list[int]]:
    """Detect face on every loop frame once — required for idle animations."""
    detector, _torch = _import_face_detector(wav2lip_root)

    frames_bgr: list[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()
    if not frames_bgr:
        raise RuntimeError(f"No frames in {video_path}")

    batch_size = 16
    predictions: list = []
    try:
        for i in range(0, len(frames_bgr), batch_size):
            batch = np.asarray(frames_bgr[i : i + batch_size])
            predictions.extend(detector.get_detections_for_batch(batch))
    finally:
        del detector

    pady1, pady2, padx1, padx2 = 0, 10, 0, 0
    boxes: list[list[int]] = []
    for rect, image in zip(predictions, frames_bgr):
        if rect is None:
            raise RuntimeError("Face not detected in patient loop frame")
        y1 = max(0, int(rect[1]) - pady1)
        y2 = min(image.shape[0], int(rect[3]) + pady2)
        x1 = max(0, int(rect[0]) - padx1)
        x2 = min(image.shape[1], int(rect[2]) + padx2)
        boxes.append([y1, y2, x1, x2])

    smoothed = _smooth_boxes(np.array(boxes, dtype=float)).astype(int).tolist()
    logger.info("Cached %d per-frame face boxes for loop video", len(smoothed))
    return smoothed


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
        self._boxes_file: Path | None = None

        if self.is_ready():
            ensure_wav2lip_patched(self.wav2lip_root)
            boxes_path = self.wav2lip_root / "temp" / "alan_face_boxes.json"
            if boxes_path.is_file():
                with boxes_path.open(encoding="utf-8") as f:
                    cached = json.load(f)
                if len(cached) > 0:
                    logger.info("Loaded %d cached face boxes", len(cached))
                    self._boxes_file = boxes_path
            if self._boxes_file is None:
                boxes = compute_per_frame_boxes(self.loop_video_path, self.wav2lip_root)
                boxes_path.parent.mkdir(parents=True, exist_ok=True)
                boxes_path.write_text(json.dumps(boxes), encoding="utf-8")
                self._boxes_file = boxes_path

    def is_ready(self) -> bool:
        return (
            self.inference_py.is_file()
            and checkpoint_is_valid(self.checkpoint)
            and Path(self.loop_video_path).is_file()
        )

    def sync_utterance(self, audio_wav: str) -> list[np.ndarray]:
        if not self.is_ready() or self._boxes_file is None:
            raise RuntimeError("Wav2Lip not ready or face boxes not cached")

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
                "--boxes_file",
                str(self._boxes_file),
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
