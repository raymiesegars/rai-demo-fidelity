"""Instant speech-reactive mouth on the idle loop — no ML inference per frame."""

from __future__ import annotations

import logging
import math
import os
import time

import cv2
import numpy as np

from composite import lip_rect

logger = logging.getLogger("avatar-worker")


class SpeechReactiveMouth:
    """Lip-shaped motion synced to agent speech — no bright spotlight."""

    def __init__(self) -> None:
        self._openness = 0.0
        self._attack = float(os.environ.get("MOUTH_ATTACK", "0.52"))
        self._decay = float(os.environ.get("MOUTH_DECAY", "0.14"))
        self._max_stretch = float(os.environ.get("MOUTH_MAX_STRETCH", "0.20"))
        self._threshold = float(os.environ.get("ANIMATION_ENERGY_THRESHOLD", "8"))
        self._sensitivity = float(os.environ.get("MOUTH_SENSITIVITY", "400"))
        self._utterance_until = 0.0
        self._logged_roi = False

    @property
    def openness(self) -> float:
        return self._openness

    def reset(self) -> None:
        self._openness = 0.0
        self._utterance_until = 0.0
        self._logged_roi = False

    def extend_speech(self, duration_sec: float) -> None:
        self._utterance_until = max(
            self._utterance_until,
            time.monotonic() + max(0.25, duration_sec),
        )

    def start_utterance(self, duration_sec: float) -> None:
        self.extend_speech(duration_sec)

    def note_active_speaker(self) -> None:
        self.extend_speech(float(os.environ.get("MOUTH_SPEAKER_EXTEND_SEC", "0.55")))

    def update(self, energy: float) -> float:
        now = time.monotonic()
        if energy >= self._threshold:
            self.extend_speech(float(os.environ.get("MOUTH_AUDIO_EXTEND_SEC", "0.30")))

        if now < self._utterance_until:
            # Slower oscillation reads more like syllables than a buzzer.
            target = 0.38 + 0.32 * (0.5 + 0.5 * math.sin(now * 10.5))
        else:
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
        if w < 8 or h < 5:
            return frame

        if self._openness > 0.35 and not self._logged_roi:
            logger.info(
                "Mouth ROI %dx%d px at (%d,%d) open=%.2f",
                w,
                h,
                lx1,
                ly1,
                self._openness,
            )
            self._logged_roi = True

        o = self._openness
        out = frame.copy()
        orig = out[ly1:ly2, lx1:lx2, :3].astype(np.float32)
        patch = orig.copy()

        # Lower lip only — upper lip stays fixed.
        seam = max(1, int(h * 0.44))
        drop = max(1, int(h * 0.14 * o))
        lower = patch[seam:].copy()
        lower_h = lower.shape[0]

        if lower_h >= 2:
            warped_h = max(2, int(lower_h * (1.0 + o * self._max_stretch)))
            warped = cv2.resize(lower, (w, warped_h), interpolation=cv2.INTER_LINEAR)
            result = patch.copy()
            result[:seam] = patch[:seam]
            fit_end = min(h, seam + drop + warped_h)
            fit_h = fit_end - (seam + drop)
            if fit_h > 0:
                result[seam + drop : seam + drop + fit_h] = cv2.resize(
                    warped, (w, fit_h)
                )
            patch = result

        mask = _mouth_mask(h, w)
        blended = patch * mask[:, :, np.newaxis] + orig * (1.0 - mask[:, :, np.newaxis])
        out[ly1:ly2, lx1:lx2, :3] = np.clip(blended, 0, 255).astype(np.uint8)

        if os.environ.get("MOUTH_DEBUG", "0") == "1":
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (0, 255, 0), 1)
            cx, cy = w // 2, int(h * 0.55)
            cv2.ellipse(
                out,
                (lx1 + cx, ly1 + cy),
                (max(3, int(w * 0.44)), max(2, int(h * 0.32))),
                0,
                0,
                360,
                (0, 0, 255),
                1,
            )

        return out


def _mouth_mask(h: int, w: int) -> np.ndarray:
    """Wide flat lip mask — not a bright circle."""
    mask = np.zeros((h, w), dtype=np.float32)
    cx = w // 2
    cy = int(h * 0.55)
    axes = (max(4, int(w * 0.44)), max(2, int(h * 0.32)))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
    blur = max(3, min(h, w) // 6) | 1
    return np.clip(cv2.GaussianBlur(mask, (blur, blur), 0), 0.0, 1.0)
