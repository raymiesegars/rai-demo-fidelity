"""Instant speech-reactive mouth on the idle loop — no ML inference per frame."""

from __future__ import annotations

import os

import cv2
import numpy as np


def reactive_mouth_rect(box: list[int]) -> tuple[int, int, int, int]:
    """Wide lower-face ROI centered on the mouth (not the tiny Wav2Lip lip crop)."""
    y1, y2, x1, x2 = box
    fh, fw = y2 - y1, x2 - x1
    y_shift = float(os.environ.get("MOUTH_RECT_Y_SHIFT", "0.08"))
    width_frac = float(os.environ.get("MOUTH_RECT_WIDTH", "0.78"))
    y_top = float(os.environ.get("MOUTH_RECT_TOP", "0.50")) + y_shift
    y_bot = float(os.environ.get("MOUTH_RECT_BOTTOM", "0.96")) + y_shift

    cx = x1 + fw // 2
    half_w = max(8, int(fw * width_frac / 2))
    ly1 = y1 + int(fh * min(y_top, y_bot - 0.1))
    ly2 = y1 + int(fh * max(y_bot, y_top + 0.1))
    return ly1, ly2, cx - half_w, cx + half_w


class SpeechReactiveMouth:
    """Audio-synced jaw drop + mouth opening on Alan's face during agent speech."""

    def __init__(self) -> None:
        self._openness = 0.0
        self._attack = float(os.environ.get("MOUTH_ATTACK", "0.60"))
        self._decay = float(os.environ.get("MOUTH_DECAY", "0.20"))
        self._max_stretch = float(os.environ.get("MOUTH_MAX_STRETCH", "0.50"))
        self._threshold = float(os.environ.get("ANIMATION_ENERGY_THRESHOLD", "8"))
        self._sensitivity = float(os.environ.get("MOUTH_SENSITIVITY", "400"))
        self._jaw_drop = float(os.environ.get("MOUTH_JAW_DROP", "0.40"))
        self._width_open = float(os.environ.get("MOUTH_WIDTH_OPEN", "0.14"))
        self._cavity = float(os.environ.get("MOUTH_CAVITY", "0.55"))

    @property
    def openness(self) -> float:
        return self._openness

    def reset(self) -> None:
        self._openness = 0.0

    def update(self, energy: float) -> float:
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
        if w < 16 or h < 12:
            return frame

        out = frame.copy()
        roi = out[ly1:ly2, lx1:lx2, :3].astype(np.float32)
        o = self._openness

        scale_y = 1.0 + o * self._max_stretch
        scale_x = 1.0 + o * self._width_open
        new_h = max(h + 1, int(h * scale_y))
        new_w = max(w + 1, int(w * scale_x))
        warped = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pin upper lip — expand jaw downward from ~32% down the ROI.
        anchor_frac = 0.32
        anchor_dst = int(h * anchor_frac)
        anchor_src = int(new_h * anchor_frac)
        y0 = anchor_dst - anchor_src
        x0 = (new_w - w) // 2
        y1s = max(0, y0)
        y2s = min(new_h, y0 + h)
        x1s = max(0, x0)
        x2s = min(new_w, x0 + w)
        dst_y1 = y1s - y0
        dst_y2 = dst_y1 + (y2s - y1s)
        dst_x1 = x1s - x0
        dst_x2 = dst_x1 + (x2s - x1s)

        patch = roi.copy()
        src = warped[y1s:y2s, x1s:x2s]
        if src.shape[0] > 0 and src.shape[1] > 0:
            patch[dst_y1:dst_y2, dst_x1:dst_x2] = src

        # Lower-lip jaw drop (shift bottom half down).
        split = int(h * 0.42)
        drop_px = int(o * h * self._jaw_drop)
        if drop_px > 0 and split < h - 2:
            lower = patch[split:].copy()
            patch[split:] = patch[split]  # smear seam
            end = min(h, split + drop_px + lower.shape[0])
            lower_h = end - split - drop_px
            if lower_h > 0:
                lower_resized = cv2.resize(lower, (w, lower_h))
                patch[split + drop_px : split + drop_px + lower_h] = lower_resized

        # Dark mouth interior when open.
        if o > 0.10:
            cx, cy = w // 2, int(h * 0.58)
            ax = max(2, int(w * 0.22 * o))
            ay = max(2, int(h * 0.14 * o))
            cavity = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(cavity, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
            cavity = cv2.GaussianBlur(cavity, (9, 9), 0) * o * self._cavity
            dark = np.array([28, 18, 22], dtype=np.float32)
            patch = patch * (1.0 - cavity[:, :, np.newaxis]) + dark * cavity[:, :, np.newaxis]

        mask = _mouth_mask(h, w)
        dst = out[ly1:ly2, lx1:lx2, :3].astype(np.float32)
        blended = patch * mask[:, :, np.newaxis] + dst * (1.0 - mask[:, :, np.newaxis])
        out[ly1:ly2, lx1:lx2, :3] = np.clip(blended, 0, 255).astype(np.uint8)
        return out


def _mouth_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, int(h * 0.55)
    axes = (max(4, w // 2 - 4), max(4, int(h * 0.46)))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
    blur = max(5, min(h, w) // 3) | 1
    return np.clip(cv2.GaussianBlur(mask, (blur, blur), 0), 0.0, 1.0)
