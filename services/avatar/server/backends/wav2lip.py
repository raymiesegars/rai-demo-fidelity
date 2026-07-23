"""In-process Wav2Lip backend for still-image + locked full-frame composite.

Requires:
  services/avatar/vendor/Wav2Lip/          (git clone)
  services/avatar/models/wav2lip_gan.pth   (HF download)

Setup:  .\\setup_wav2lip.ps1
Run:    .\\run_server.ps1 -Backend wav2lip
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from backends.common import UtteranceStreamEngine

logger = logging.getLogger("avatar.wav2lip")


def _find_root(root: Path) -> Path:
    env = os.environ.get("WAV2LIP_ROOT")
    if env:
        return Path(env)
    return root / "vendor" / "Wav2Lip"


def _find_checkpoint(root: Path, wav2lip_root: Path) -> Path:
    env = os.environ.get("WAV2LIP_CHECKPOINT")
    if env:
        return Path(env)
    for p in (
        root / "models" / "wav2lip_gan.pth",
        wav2lip_root / "checkpoints" / "wav2lip_gan.pth",
    ):
        if p.is_file():
            return p
    return root / "models" / "wav2lip_gan.pth"


def _checkpoint_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 100_000_000:
        return False
    try:
        import torch
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        return isinstance(obj, dict) and "state_dict" in obj
    except Exception:
        return False


class Wav2LipBackend(UtteranceStreamEngine):
    backend_id = "wav2lip"
    backend_name = "Wav2Lip"

    def __init__(self, root: Path) -> None:
        super().__init__(chunk_frames=12, fps=25)
        self.root = root
        self.wav2lip_root = _find_root(root)
        self.checkpoint = _find_checkpoint(root, self.wav2lip_root)
        self._model = None
        self._device = "cpu"
        self._face_96: np.ndarray | None = None
        self._box_xyxy: list[int] | None = None  # on full source image

        if not (self.wav2lip_root / "models.py").is_file() and not (
            self.wav2lip_root / "models" / "__init__.py"
        ).is_file():
            # Wav2Lip layout: models/wav2lip.py or models.py depending on fork
            pass
        if not _checkpoint_ok(self.checkpoint):
            raise RuntimeError(
                f"Wav2Lip checkpoint missing/invalid: {self.checkpoint}\n"
                f"Run: .\\setup_wav2lip.ps1"
            )
        if not self.wav2lip_root.is_dir():
            raise RuntimeError(
                f"Wav2Lip repo not found at {self.wav2lip_root}\n"
                f"Run: .\\setup_wav2lip.ps1"
            )
        self._load_model()

    def _load_model(self) -> None:
        import torch

        root = str(self.wav2lip_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        # Official repo: from models import Wav2Lip
        from models import Wav2Lip  # type: ignore

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Wav2Lip()
        ckpt = torch.load(str(self.checkpoint), map_location=self._device, weights_only=False)
        state = ckpt["state_dict"]
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
        self._model = model.to(self._device).eval()
        logger.info("Wav2Lip model loaded on %s from %s", self._device, self.checkpoint)

    def on_prepare(self) -> None:
        assert self._source_bgr is not None and self._face_box_abs is not None
        x1, y1, x2, y2 = self._face_box_abs
        face = self._source_bgr[y1:y2, x1:x2]
        self._face_96 = cv2.resize(face, (96, 96))
        self._box_xyxy = [x1, y1, x2, y2]

    def _mel_chunks(self, wav_f32: np.ndarray) -> list[np.ndarray]:
        root = str(self.wav2lip_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        import audio as w2l_audio  # type: ignore

        # Wav2Lip audio.load_wav expects a path; use mel from array via temp or
        # call melspectrogram on raw samples (load_wav just reads + resamples).
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            sf.write(path, wav_f32, 16000)
            wav = w2l_audio.load_wav(path, 16000)
            mel = w2l_audio.melspectrogram(wav)
        finally:
            Path(path).unlink(missing_ok=True)

        if np.isnan(mel).any():
            raise ValueError("Mel contains nan")
        mel_step = 16
        fps = float(self.fps)
        chunks = []
        mult = 80.0 / fps
        i = 0
        while True:
            start = int(i * mult)
            if start + mel_step > mel.shape[1]:
                chunks.append(mel[:, mel.shape[1] - mel_step :])
                break
            chunks.append(mel[:, start : start + mel_step])
            i += 1
        return chunks

    def _run_batch(self, faces: list[np.ndarray], mels: list[np.ndarray]) -> np.ndarray:
        import torch

        assert self._model is not None
        img = np.asarray(faces, dtype=np.float32)
        mel = np.asarray(mels, dtype=np.float32)
        masked = img.copy()
        masked[:, 48:, :] = 0
        img_in = np.concatenate((masked, img), axis=3) / 255.0
        mel_in = mel.reshape(len(mel), mel.shape[1], mel.shape[2], 1)
        img_t = torch.FloatTensor(np.transpose(img_in, (0, 3, 1, 2))).to(self._device)
        mel_t = torch.FloatTensor(np.transpose(mel_in, (0, 3, 1, 2))).to(self._device)
        with torch.no_grad():
            pred = self._model(mel_t, img_t)
        return pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

    def render_utterance(self, audio_f32_16k: np.ndarray) -> list[np.ndarray]:
        assert self._face_96 is not None and self._box_xyxy is not None
        assert self._source_bgr is not None
        mels = self._mel_chunks(audio_f32_16k)
        batch = int(os.environ.get("WAV2LIP_BATCH_SIZE", "128"))
        preds: list[np.ndarray] = []
        faces, mel_batch = [], []
        for m in mels:
            faces.append(self._face_96.copy())
            mel_batch.append(m)
            if len(faces) >= batch:
                preds.extend(list(self._run_batch(faces, mel_batch)))
                faces, mel_batch = [], []
        if faces:
            preds.extend(list(self._run_batch(faces, mel_batch)))

        x1, y1, x2, y2 = self._box_xyxy
        out_frames = []
        for p in preds:
            canvas = self._source_bgr.copy()
            face = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            # Soft paste face region (locked box — no tracking)
            roi = canvas[y1:y2, x1:x2].astype(np.float32)
            src = face.astype(np.float32)
            # Simple edge feather
            h, w = face.shape[:2]
            mask = np.ones((h, w), dtype=np.float32)
            f = max(2, min(h, w) // 16)
            for i in range(f):
                a = (i + 1) / (f + 1)
                mask[i, :] *= a
                mask[-(i + 1), :] *= a
                mask[:, i] *= a
                mask[:, -(i + 1)] *= a
            alpha = mask[..., None]
            canvas[y1:y2, x1:x2] = np.clip(src * alpha + roi * (1 - alpha), 0, 255).astype(np.uint8)
            out_frames.append(canvas)
        return out_frames


def create(root: Path):
    return Wav2LipBackend(root)
