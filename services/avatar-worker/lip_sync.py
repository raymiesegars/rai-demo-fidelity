"""Wav2Lip batch lip-sync for patient loop + agent audio."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("lip-sync")

# Official wav2lip_gan.pth is ~139 MB; partial downloads are usually a few KB.
CHECKPOINT_MIN_BYTES = 130_000_000


def save_wav_int16(path: str, samples: np.ndarray, sample_rate: int = 48000) -> None:
    """Save mono int16 PCM to WAV."""
    samples = np.asarray(samples, dtype=np.int16).flatten()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def load_video_frames_bgr(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def load_video_frames_rgba(path: str) -> list[np.ndarray]:
    return [
        cv2.cvtColor(f, cv2.COLOR_BGR2RGBA)
        for f in load_video_frames_bgr(path)
    ]


def checkpoint_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    if size < CHECKPOINT_MIN_BYTES:
        logger.warning("Checkpoint too small (%d bytes) — likely corrupt download", size)
        return False
    try:
        import torch

        torch.load(str(path), map_location="cpu", weights_only=False)
        return True
    except Exception as exc:
        logger.warning("Checkpoint failed torch.load: %s", exc)
        return False


class Wav2LipEngine:
    """Runs Wav2Lip inference via cloned repo on RunPod."""

    def __init__(
        self,
        loop_video_path: str,
        wav2lip_root: str | None = None,
        checkpoint: str | None = None,
    ) -> None:
        # Must be absolute — Wav2Lip subprocess runs with cwd=WAV2LIP_ROOT.
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

    def is_ready(self) -> bool:
        return (
            self.inference_py.is_file()
            and checkpoint_is_valid(self.checkpoint)
            and Path(self.loop_video_path).is_file()
        )

    def sync_utterance(self, audio_wav: str) -> list[np.ndarray]:
        """Return lip-synced frames as RGBA numpy arrays."""
        if not self.is_ready():
            raise RuntimeError(
                "Wav2Lip not ready. Re-run: bash setup_wav2lip.sh "
                "(checkpoint may be corrupt — delete checkpoints/wav2lip_gan.pth first). "
                f"inference={self.inference_py.is_file()}, "
                f"checkpoint={checkpoint_is_valid(self.checkpoint)}"
            )

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
                "--pads",
                "0",
                "10",
                "0",
                "0",
                "--face_det_batch_size",
                "8",
                "--wav2lip_batch_size",
                "64",
            ]
            logger.info(
                "Running Wav2Lip lip sync (face=%s, audio=%s)…",
                self.loop_video_path,
                audio_wav,
            )
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
