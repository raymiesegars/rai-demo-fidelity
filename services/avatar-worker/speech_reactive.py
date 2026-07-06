"""Instant speech-reactive mouth on the idle loop — no ML inference per frame."""

from __future__ import annotations

import logging
import math
import os
import time

import cv2
import numpy as np

logger = logging.getLogger("avatar-worker")


def reactive_mouth_rect(box: list[int]) -> tuple[int, int, int, int]:
    """Mouth ROI inside the face box — tuned for full-body Alan loop."""
    y1, y2, x1, x2 = box
    fh, fw = y2 - y1, x2 - x1
    y_shift = float(os.environ.get("MOUTH_RECT_Y_SHIFT", "0.02"))
    y_top = float(os.environ.get("MOUTH_RECT_TOP", "0.62")) + y_shift
    y_bot = float(os.environ.get("MOUTH_RECT_BOTTOM", "0.88")) + y_shift
    x_left = float(os.environ.get("MOUTH_RECT_LEFT", "0.32"))
    x_right = float(os.environ.get("MOUTH_RECT_RIGHT", "0.68"))

    ly1 = y1 + int(fh * min(y_top, y_bot - 0.05))
    ly2 = y1 + int(fh * max(y_bot, y_top + 0.05))
    lx1 = x1 + int(fw * x_left)
    lx2 = x1 + int(fw * x_right)
    return ly1, ly2, lx1, lx2


class SpeechReactiveMouth:
    """Visible lip motion synced to agent speech."""

    def __init__(self) -> None:
        self._openness = 0.0
        self._attack = float(os.environ.get("MOUTH_ATTACK", "0.60"))
        self._decay = float(os.environ.get("MOUTH_DECAY", "0.18"))
        self._max_stretch = float(os.environ.get("MOUTH_MAX_STRETCH", "0.35"))
        self._threshold = float(os.environ.get("ANIMATION_ENERGY_THRESHOLD", "8"))
        self._sensitivity = float(os.environ.get("MOUTH_SENSITIVITY", "400"))
        self._brightness = float(os.environ.get("MOUTH_BRIGHTNESS", "0.30"))
        self._min_drop_px = int(os.environ.get("MOUTH_MIN_DROP_PX", "3"))
        self._utterance_until = 0.0
        self._speaker_until = 0.0
        self._logged_roi = False

    @property
    def openness(self) -> float:
        return self._openness

    def reset(self) -> None:
        self._openness = 0.0
        self._utterance_until = 0.0
        self._speaker_until = 0.0
        self._logged_roi = False

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
            target = 0.55 + 0.40 * (0.5 + 0.5 * math.sin(now * 13.0))
        elif now < self._speaker_until:
            target = 0.70
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
        if w < 8 or h < 5:
            return frame

        if self._openness > 0.4 and not self._logged_roi:
            logger.info(
                "Mouth ROI %dx%d px at (%d,%d) face_box=%s open=%.2f",
                w,
                h,
                lx1,
                ly1,
                box,
                self._openness,
            )
            self._logged_roi = True

        o = self._openness
        out = frame.copy()
        orig = out[ly1:ly2, lx1:lx2, :3].astype(np.float32)
        patch = orig.copy()

        # Brightness pulse — visible even on compressed video.
        patch = np.clip(patch * (1.0 + self._brightness * o), 0, 255)

        # Lower lip drops by real pixels (not sub-pixel stretch).
        seam = max(1, int(h * 0.38))
        drop = max(self._min_drop_px, int(h * 0.22 * o))
        upper = patch[:seam].copy()
        lower = patch[seam:].copy()
        lower_h = lower.shape[0]

        if lower_h >= 2:
            warped = cv2.resize(
                lower,
                (w, max(2, int(lower_h * (1.0 + o * self._max_stretch)))),
                interpolation=cv2.INTER_LINEAR,
            )
            result = patch.copy()
            result[:seam] = upper
            gap_end = min(h, seam + drop)
            if gap_end > seam:
                result[seam:gap_end] = upper[-1:] * 0.75
            dest_end = min(h, seam + drop + warped.shape[0])
            fit_h = dest_end - (seam + drop)
            if fit_h > 0:
                result[seam + drop : seam + drop + fit_h] = cv2.resize(
                    warped, (w, fit_h)
                )
            patch = result

        mask = _lip_mask(h, w)
        blended = patch * mask[:, :, np.newaxis] + orig * (1.0 - mask[:, :, np.newaxis])
        out[ly1:ly2, lx1:lx2, :3] = np.clip(blended, 0, 255).astype(np.uint8)

        if os.environ.get("MOUTH_DEBUG", "0") == "1":
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)

        return out


def _lip_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    cv2.ellipse(mask, (cx, cy), (max(3, w // 2 - 2), max(3, h // 2 - 2)), 0, 0, 360, 1.0, -1)
    blur = max(3, min(h, w) // 4) | 1
    return np.clip(cv2.GaussianBlur(mask, (blur, blur), 0), 0.0, 1.0)
