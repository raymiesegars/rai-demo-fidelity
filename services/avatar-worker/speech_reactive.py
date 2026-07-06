"""Instant speech-reactive mouth on the idle loop — no ML inference per frame."""

from __future__ import annotations

import math
import os
import time

import cv2
import numpy as np


def reactive_mouth_rect(box: list[int]) -> tuple[int, int, int, int]:
    """Tight crop on the lips only."""
    y1, y2, x1, x2 = box
    fh, fw = y2 - y1, x2 - x1
    y_shift = float(os.environ.get("MOUTH_RECT_Y_SHIFT", "0.0"))
    x_left = float(os.environ.get("MOUTH_RECT_LEFT", "0.36"))
    x_right = float(os.environ.get("MOUTH_RECT_RIGHT", "0.64"))
    y_top = float(os.environ.get("MOUTH_RECT_TOP", "0.70")) + y_shift
    y_bot = float(os.environ.get("MOUTH_RECT_BOTTOM", "0.84")) + y_shift

    ly1 = y1 + int(fh * min(y_top, y_bot - 0.04))
    ly2 = y1 + int(fh * max(y_bot, y_top + 0.04))
    lx1 = x1 + int(fw * x_left)
    lx2 = x1 + int(fw * x_right)
    return ly1, ly2, lx1, lx2


class SpeechReactiveMouth:
    """Subtle lip motion synced to agent speech."""

    def __init__(self) -> None:
        self._openness = 0.0
        self._attack = float(os.environ.get("MOUTH_ATTACK", "0.55"))
        self._decay = float(os.environ.get("MOUTH_DECAY", "0.22"))
        self._max_stretch = float(os.environ.get("MOUTH_MAX_STRETCH", "0.26"))
        self._threshold = float(os.environ.get("ANIMATION_ENERGY_THRESHOLD", "8"))
        self._sensitivity = float(os.environ.get("MOUTH_SENSITIVITY", "400"))
        self._gap = float(os.environ.get("MOUTH_LIP_GAP", "0.35"))
        self._utterance_until = 0.0
        self._speaker_until = 0.0

    @property
    def openness(self) -> float:
        return self._openness

    def reset(self) -> None:
        self._openness = 0.0
        self._utterance_until = 0.0
        self._speaker_until = 0.0

    def start_utterance(self, duration_sec: float) -> None:
        self._utterance_until = max(
            self._utterance_until,
            time.monotonic() + max(0.6, duration_sec),
        )

    def note_active_speaker(self) -> None:
        self._speaker_until = time.monotonic() + 0.45

    def update(self, energy: float) -> float:
        now = time.monotonic()
        if now < self._utterance_until:
            target = 0.50 + 0.35 * (0.5 + 0.5 * math.sin(now * 13.0))
        elif now < self._speaker_until:
            target = 0.62
        else:
            target = min(1.0, max(0.0, (energy - self._threshold) / self._sensitivity))
        rate = self._attack if target > self._openness else self._decay
        self._openness += (target - self._openness) * rate
        return self._openness

    def apply(self, frame: np.ndarray, box: list[int] | None) -> np.ndarray:
        if box is None or self._openness < 0.02:
            return frame

        ly1, ly2, lx1, lx2 = reactive_mouth_rect(box)
        ly1 = max(0, ly1)
        lx1 = max(0, lx1)
        ly2 = min(frame.shape[0], ly2)
        lx2 = min(frame.shape[1], lx2)
        w, h = lx2 - lx1, ly2 - ly1
        if w < 10 or h < 6:
            return frame

        out = frame.copy()
        roi = out[ly1:ly2, lx1:lx2, :3].astype(np.float32)
        o = self._openness

        seam = max(1, int(h * 0.42))
        upper = roi[:seam].copy()
        lower = roi[seam:].copy()
        lower_h = lower.shape[0]
        if lower_h < 2:
            return frame

        gap_px = max(0, int(h * o * self._gap * 0.10))
        stretch = 1.0 + o * self._max_stretch
        new_lower_h = max(2, int(lower_h * stretch))
        lower_warp = cv2.resize(lower, (w, new_lower_h), interpolation=cv2.INTER_LINEAR)

        patch = roi.copy()
        patch[:seam] = upper
        drop_start = seam + gap_px
        drop_end = min(h, drop_start + new_lower_h)
        fit_h = drop_end - drop_start
        if fit_h > 0:
            patch[drop_start:drop_end] = cv2.resize(lower_warp, (w, fit_h))

        if o > 0.18:
            cx, cy = w // 2, seam + max(1, gap_px // 2)
            ax = max(1, int(w * 0.09 * o))
            ay = max(1, int(h * 0.05 * o))
            shadow = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(shadow, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
            shadow = cv2.GaussianBlur(shadow, (5, 5), 0) * o * 0.22
            dark = np.array([40, 28, 32], dtype=np.float32)
            patch = patch * (1.0 - shadow[:, :, np.newaxis]) + dark * shadow[:, :, np.newaxis]

        mask = _lip_mask(h, w, seam)
        dst = out[ly1:ly2, lx1:lx2, :3].astype(np.float32)
        blended = patch * mask[:, :, np.newaxis] + dst * (1.0 - mask[:, :, np.newaxis])
        out[ly1:ly2, lx1:lx2, :3] = np.clip(blended, 0, 255).astype(np.uint8)
        return out


def _lip_mask(h: int, w: int, seam: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    cx = w // 2
    cv2.ellipse(
        mask,
        (cx, max(2, seam // 2)),
        (max(3, w // 2 - 3), max(2, seam // 2)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    cv2.ellipse(
        mask,
        (cx, min(h - 2, seam + (h - seam) // 2)),
        (max(3, w // 2 - 2), max(2, (h - seam) // 2)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    blur = max(3, min(h, w) // 5) | 1
    return np.clip(cv2.GaussianBlur(mask, (blur, blur), 0), 0.0, 1.0)
