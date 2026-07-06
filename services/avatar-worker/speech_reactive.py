"""Instant speech-reactive mouth on the idle loop — no ML inference per frame."""

from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np

logger = logging.getLogger("avatar-worker")


def mouth_roi(box: list[int]) -> tuple[int, int, int, int]:
    """Mouth region: upper + lower lip only (stops above the chin)."""
    y1, y2, x1, x2 = box
    fh, fw = y2 - y1, x2 - x1
    y_shift = float(os.environ.get("MOUTH_RECT_Y_SHIFT", "0.0"))
    x_left = float(os.environ.get("MOUTH_RECT_X_LEFT", "0.24"))
    x_right = float(os.environ.get("MOUTH_RECT_X_RIGHT", "0.76"))
    top = float(os.environ.get("MOUTH_RECT_TOP", "0.68")) + y_shift
    bottom = float(os.environ.get("MOUTH_RECT_BOTTOM", "0.84")) + y_shift

    ly1 = y1 + int(fh * min(top, bottom - 0.06))
    ly2 = y1 + int(fh * max(bottom, top + 0.06))
    lx1 = x1 + int(fw * x_left)
    lx2 = x1 + int(fw * x_right)
    return ly1, ly2, lx1, lx2


class SpeechReactiveMouth:
    """Lip motion driven by live audio energy — stops when voice stops."""

    def __init__(self) -> None:
        self._openness = 0.0
        self._smooth_energy = 0.0
        self._last_voice_at = 0.0
        self._attack = float(os.environ.get("MOUTH_ATTACK", "0.68"))
        self._decay = float(os.environ.get("MOUTH_DECAY", "0.38"))
        self._open_amount = float(os.environ.get("MOUTH_OPEN_AMOUNT", "1.15"))
        self._upper_lift = float(os.environ.get("MOUTH_UPPER_LIFT", "0.10"))
        self._lower_drop = float(os.environ.get("MOUTH_LOWER_DROP", "0.36"))
        self._corner_move = float(os.environ.get("MOUTH_CORNER_MOVE", "0.55"))
        self._upper_depth = float(os.environ.get("MOUTH_UPPER_DEPTH", "0.38"))
        self._lower_depth = float(os.environ.get("MOUTH_LOWER_DEPTH", "0.28"))
        self._threshold = float(os.environ.get("ANIMATION_ENERGY_THRESHOLD", "8"))
        self._sensitivity = float(os.environ.get("MOUTH_SENSITIVITY", "350"))
        self._silence_cutoff = float(os.environ.get("MOUTH_SILENCE_CUTOFF_SEC", "0.10"))
        self._lip_line = float(os.environ.get("MOUTH_LIP_LINE", "0.44"))
        self._logged_roi = False

    @property
    def openness(self) -> float:
        return self._openness

    def reset(self) -> None:
        self._openness = 0.0
        self._smooth_energy = 0.0
        self._last_voice_at = 0.0
        self._logged_roi = False

    def arm_reply(self) -> None:
        self._last_voice_at = time.monotonic()

    def note_active_speaker(self) -> None:
        self._last_voice_at = time.monotonic()

    def update(self, energy: float) -> float:
        now = time.monotonic()
        self._smooth_energy = self._smooth_energy * 0.5 + energy * 0.5

        if self._smooth_energy >= self._threshold:
            self._last_voice_at = now
            norm = min(1.0, (self._smooth_energy - self._threshold) / self._sensitivity)
            target = 0.40 + 0.72 * norm
        elif now - self._last_voice_at < self._silence_cutoff:
            target = self._openness * 0.55
        else:
            target = 0.0

        rate = self._attack if target > self._openness else self._decay
        self._openness += (target - self._openness) * rate
        return self._openness

    def apply(self, frame: np.ndarray, box: list[int] | None) -> np.ndarray:
        if box is None or self._openness < 0.04:
            return frame

        ly1, ly2, lx1, lx2 = mouth_roi(box)
        ly1 = max(0, ly1)
        lx1 = max(0, lx1)
        ly2 = min(frame.shape[0], ly2)
        lx2 = min(frame.shape[1], lx2)
        w, h = lx2 - lx1, ly2 - ly1
        if w < 10 or h < 8:
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

        o = self._openness * self._open_amount
        out = frame.copy()
        orig = out[ly1:ly2, lx1:lx2, :3].astype(np.float32)
        warped = _warp_mouth_open(
            orig,
            o,
            self._lip_line,
            self._upper_lift,
            self._lower_drop,
            self._corner_move,
            self._upper_depth,
            self._lower_depth,
        )
        mask = _full_mouth_mask(h, w)
        blended = warped * mask[:, :, np.newaxis] + orig * (1.0 - mask[:, :, np.newaxis])
        out[ly1:ly2, lx1:lx2, :3] = np.clip(blended, 0, 255).astype(np.uint8)

        if os.environ.get("MOUTH_DEBUG", "0") == "1":
            lip_y = ly1 + int(h * self._lip_line)
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (0, 255, 0), 1)
            cv2.line(out, (lx1, lip_y), (lx2, lip_y), (0, 255, 255), 1)
            _draw_debug_lips(out, lx1, ly1, w, h)

        return out


def _warp_mouth_open(
    roi: np.ndarray,
    openness: float,
    lip_line_frac: float,
    upper_frac: float,
    lower_frac: float,
    corner_move: float,
    upper_depth: float,
    lower_depth: float,
) -> np.ndarray:
    """Displace upper/lower lip only — fade to zero before the chin."""
    h, w = roi.shape[:2]
    lip_y = lip_line_frac * h
    upper_lift = max(0.5, h * upper_frac * openness)
    lower_drop = max(1.0, h * lower_frac * openness)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) * 0.5
    edge = np.clip(np.abs(xx - cx) / (cx + 0.5), 0.0, 1.0)
    falloff = corner_move + (1.0 - corner_move) * (1.0 - edge)

    above = yy < lip_y
    below = ~above
    t_above = np.clip((lip_y - yy) / max(1.0, lip_y), 0.0, 1.0)
    t_below = np.clip((yy - lip_y) / max(1.0, h * lower_depth), 0.0, 1.0)

    upper_fade = np.clip(1.0 - (lip_y - yy) / max(1.0, h * upper_depth), 0.0, 1.0)
    lower_fade = np.clip(1.0 - (yy - lip_y) / max(1.0, h * lower_depth), 0.0, 1.0)

    disp = np.zeros_like(yy)
    disp[above] = -upper_lift * falloff[above] * t_above[above] * upper_fade[above]
    disp[below] = lower_drop * falloff[below] * (1.0 - t_below[below]) * lower_fade[below]

    map_y = yy + disp
    map_x = xx.copy()

    return cv2.remap(
        roi,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _full_mouth_mask(h: int, w: int) -> np.ndarray:
    """Upper + lower lip only — no chin in the blend zone."""
    mask = np.zeros((h, w), dtype=np.float32)

    cv2.ellipse(
        mask,
        (w // 2, int(h * 0.34)),
        (max(4, int(w * 0.46)), max(3, int(h * 0.22))),
        0,
        0,
        360,
        1.0,
        -1,
    )
    cv2.ellipse(
        mask,
        (w // 2, int(h * 0.58)),
        (max(4, int(w * 0.47)), max(3, int(h * 0.20))),
        0,
        0,
        360,
        1.0,
        -1,
    )
    y1, y2 = int(h * 0.14), int(h * 0.72)
    x1, x2 = int(w * 0.06), int(w * 0.94)
    mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], 0.90)

    blur = max(3, min(h, w) // 8) | 1
    return np.clip(cv2.GaussianBlur(mask, (blur, blur), 0), 0.0, 1.0)


def _draw_debug_lips(out: np.ndarray, ox: int, oy: int, w: int, h: int) -> None:
    cv2.ellipse(
        out,
        (ox + w // 2, oy + int(h * 0.34)),
        (max(3, int(w * 0.46)), max(2, int(h * 0.22))),
        0,
        0,
        360,
        (255, 0, 0),
        1,
    )
    cv2.ellipse(
        out,
        (ox + w // 2, oy + int(h * 0.58)),
        (max(3, int(w * 0.47)), max(2, int(h * 0.20))),
        0,
        0,
        360,
        (0, 0, 255),
        1,
    )
