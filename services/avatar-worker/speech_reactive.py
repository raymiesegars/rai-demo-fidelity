"""Instant speech-reactive mouth on the idle loop — no ML inference per frame."""

from __future__ import annotations

import os

import cv2
import numpy as np

from composite import lip_rect


class SpeechReactiveMouth:
    """Audio-synced subtle mouth opening on Alan's face during agent speech."""

    def __init__(self) -> None:
        self._openness = 0.0
        self._attack = float(os.environ.get("MOUTH_ATTACK", "0.55"))
        self._decay = float(os.environ.get("MOUTH_DECAY", "0.18"))
        self._max_stretch = float(os.environ.get("MOUTH_MAX_STRETCH", "0.22"))
        self._threshold = float(os.environ.get("ANIMATION_ENERGY_THRESHOLD", "20"))
        self._sensitivity = float(os.environ.get("MOUTH_SENSITIVITY", "900"))
        self._warmth = float(os.environ.get("MOUTH_WARMTH", "0.06"))

    @property
    def openness(self) -> float:
        return self._openness

    def update(self, energy: float) -> float:
        target = min(1.0, max(0.0, (energy - self._threshold) / self._sensitivity))
        rate = self._attack if target > self._openness else self._decay
        self._openness += (target - self._openness) * rate
        return self._openness

    def apply(self, frame: np.ndarray, box: list[int] | None) -> np.ndarray:
        if box is None or self._openness < 0.03:
            return frame

        ly1, ly2, lx1, lx2 = lip_rect(box)
        ly1 = max(0, ly1)
        lx1 = max(0, lx1)
        ly2 = min(frame.shape[0], ly2)
        lx2 = min(frame.shape[1], lx2)
        w, h = lx2 - lx1, ly2 - ly1
        if w < 12 or h < 6:
            return frame

        out = frame.copy()
        roi = out[ly1:ly2, lx1:lx2, :3].astype(np.float32)
        stretch = 1.0 + self._openness * self._max_stretch
        new_h = max(h + 1, int(h * stretch))
        warped = cv2.resize(roi, (w, new_h), interpolation=cv2.INTER_LINEAR)

        # Anchor stretch toward chin (expand downward).
        pad_top = max(0, (new_h - h) * 2 // 5)
        src_y1 = pad_top
        src_y2 = min(new_h, pad_top + h)
        mouth = warped[src_y1:src_y2, :, :]

        mask = _mouth_mask(h, w)
        if self._warmth > 0:
            mouth = mouth * (1.0 + self._warmth * self._openness)

        dst = out[ly1:ly2, lx1:lx2, :3].astype(np.float32)
        blended = mouth * mask[:, :, np.newaxis] + dst * (1.0 - mask[:, :, np.newaxis])
        out[ly1:ly2, lx1:lx2, :3] = np.clip(blended, 0, 255).astype(np.uint8)
        return out


def _mouth_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    axes = (max(2, w // 2 - 2), max(2, h // 2 - 1))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
    blur = max(3, min(h, w) // 4) | 1
    return np.clip(cv2.GaussianBlur(mask, (blur, blur), 0), 0.0, 1.0)
