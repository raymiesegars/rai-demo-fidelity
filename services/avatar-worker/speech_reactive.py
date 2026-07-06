"""Instant speech-reactive mouth on the idle loop — no ML inference per frame."""

from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np

logger = logging.getLogger("avatar-worker")


def mouth_roi(box: list[int]) -> tuple[int, int, int, int]:
    """Mouth region inside the face box — tall enough for both lips, chin trimmed in warp/mask."""
    y1, y2, x1, x2 = box
    fh, fw = y2 - y1, x2 - x1
    y_shift = float(os.environ.get("MOUTH_RECT_Y_SHIFT", "0.0"))
    x_left = float(os.environ.get("MOUTH_RECT_X_LEFT", "0.24"))
    x_right = float(os.environ.get("MOUTH_RECT_X_RIGHT", "0.76"))
    top = float(os.environ.get("MOUTH_RECT_TOP", "0.68")) + y_shift
    bottom = float(os.environ.get("MOUTH_RECT_BOTTOM", "0.90")) + y_shift

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
        self._attack = float(os.environ.get("MOUTH_ATTACK", "0.42"))
        self._decay = float(os.environ.get("MOUTH_DECAY", "0.28"))
        self._open_amount = float(os.environ.get("MOUTH_OPEN_AMOUNT", "0.95"))
        self._upper_lift = float(os.environ.get("MOUTH_UPPER_LIFT", "0.07"))
        self._lower_drop = float(os.environ.get("MOUTH_LOWER_DROP", "0.24"))
        self._corner_move = float(os.environ.get("MOUTH_CORNER_MOVE", "0.62"))
        self._chin_damp_start = float(os.environ.get("MOUTH_CHIN_DAMP_START", "0.78"))
        self._gap_shade = float(os.environ.get("MOUTH_GAP_SHADE", "0.22"))
        self._threshold = float(os.environ.get("ANIMATION_ENERGY_THRESHOLD", "8"))
        self._sensitivity = float(os.environ.get("MOUTH_SENSITIVITY", "350"))
        self._silence_cutoff = float(os.environ.get("MOUTH_SILENCE_CUTOFF_SEC", "0.10"))
        self._lip_line = float(os.environ.get("MOUTH_LIP_LINE", "0.42"))
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
        # Extra smoothing reduces the twitchy look from frame-to-frame energy spikes.
        self._smooth_energy = self._smooth_energy * 0.72 + energy * 0.28

        if self._smooth_energy >= self._threshold:
            self._last_voice_at = now
            norm = min(1.0, (self._smooth_energy - self._threshold) / self._sensitivity)
            target = 0.32 + 0.58 * norm
        elif now - self._last_voice_at < self._silence_cutoff:
            target = self._openness * 0.62
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
            self._upper_lift,
            self._lower_drop,
            self._corner_move,
            self._chin_damp_start,
            self._lip_line,
        )
        warped = _shade_mouth_gap(warped, o, self._lip_line, self._gap_shade)
        mask = _lip_blend_mask(h, w, self._chin_damp_start)
        blended = warped * mask[:, :, np.newaxis] + orig * (1.0 - mask[:, :, np.newaxis])
        out[ly1:ly2, lx1:lx2, :3] = np.clip(blended, 0, 255).astype(np.uint8)

        if os.environ.get("MOUTH_DEBUG", "0") == "1":
            lip_y = ly1 + int(h * self._lip_line)
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (0, 255, 0), 1)
            cv2.line(out, (lx1, lip_y), (lx2, lip_y), (0, 255, 255), 1)
            chin_y = ly1 + int(h * self._chin_damp_start)
            cv2.line(out, (lx1, chin_y), (lx2, chin_y), (255, 128, 0), 1)
            _draw_debug_lips(out, lx1, ly1, w, h)

        return out


def _chin_protect(yy: np.ndarray, h: float, chin_start_frac: float) -> np.ndarray:
    """Fade displacement to zero below chin_start_frac (keeps jaw static)."""
    chin_y = h * chin_start_frac
    tail = max(1.0, h - chin_y)
    return np.clip(1.0 - (yy - chin_y) / tail, 0.0, 1.0)


def _lip_band(yy: np.ndarray, center_y: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((yy - center_y) / max(1.0, sigma)) ** 2)


def _warp_mouth_open(
    roi: np.ndarray,
    openness: float,
    upper_frac: float,
    lower_frac: float,
    corner_move: float,
    chin_damp_start: float,
    lip_line_frac: float,
) -> np.ndarray:
    """Part lips by moving lip pixels apart — positive lifts upper, negative drops lower."""
    h, w = roi.shape[:2]
    upper_lift = max(0.4, h * upper_frac * openness)
    lower_drop = max(0.6, h * lower_frac * openness)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) * 0.5
    edge = np.clip(np.abs(xx - cx) / (cx + 0.5), 0.0, 1.0)
    falloff = corner_move + (1.0 - corner_move) * (1.0 - edge)
    chin = _chin_protect(yy, float(h), chin_damp_start)

    lip_y = lip_line_frac * h
    upper_y = lip_y - h * 0.10
    lower_y = lip_y + h * 0.12
    upper_band = _lip_band(yy, upper_y, h * 0.09)
    lower_band = _lip_band(yy, lower_y, h * 0.10)

    # remap: dst(y)=src(y+disp). +disp shifts content up, -disp shifts content down.
    disp = (
        upper_lift * falloff * upper_band * chin
        - lower_drop * falloff * lower_band * chin
    )

    map_y = yy + disp
    map_x = xx.copy()

    return cv2.remap(
        roi,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _shade_mouth_gap(
    roi: np.ndarray,
    openness: float,
    lip_line_frac: float,
    gap_shade: float,
) -> np.ndarray:
    """Subtle interior shading so the opening reads as depth, not a skin slit."""
    if openness < 0.12 or gap_shade <= 0.0:
        return roi

    h, w = roi.shape[:2]
    lip_y = int(h * lip_line_frac)
    gap_h = max(2, int(h * (0.04 + 0.05 * openness)))
    gap_w = max(4, int(w * (0.22 + 0.14 * openness)))
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, lip_y), (gap_w, gap_h), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    strength = gap_shade * min(1.0, openness)
    return roi * (1.0 - mask[:, :, np.newaxis] * strength)


def _lip_blend_mask(h: int, w: int, chin_damp_start: float) -> np.ndarray:
    """Blend only on lip tissue — avoid warping philtrum/chin skin into the mouth."""
    mask = np.zeros((h, w), dtype=np.float32)

    cv2.ellipse(
        mask,
        (w // 2, int(h * 0.34)),
        (max(4, int(w * 0.44)), max(3, int(h * 0.18))),
        0,
        0,
        360,
        1.0,
        -1,
    )
    cv2.ellipse(
        mask,
        (w // 2, int(h * 0.56)),
        (max(4, int(w * 0.45)), max(3, int(h * 0.19))),
        0,
        0,
        360,
        1.0,
        -1,
    )
    # Corner bridge only — not a full-height rectangle.
    y1, y2 = int(h * 0.36), int(h * 0.52)
    x1, x2 = int(w * 0.04), int(w * 0.96)
    mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], 0.65)

    chin = _chin_protect(np.arange(h, dtype=np.float32)[:, np.newaxis], float(h), chin_damp_start)
    mask *= chin

    blur = max(3, min(h, w) // 10) | 1
    return np.clip(cv2.GaussianBlur(mask, (blur, blur), 0), 0.0, 1.0)


def _draw_debug_lips(out: np.ndarray, ox: int, oy: int, w: int, h: int) -> None:
    cv2.ellipse(
        out,
        (ox + w // 2, oy + int(h * 0.34)),
        (max(3, int(w * 0.44)), max(2, int(h * 0.18))),
        0,
        0,
        360,
        (255, 0, 0),
        1,
    )
    cv2.ellipse(
        out,
        (ox + w // 2, oy + int(h * 0.56)),
        (max(3, int(w * 0.45)), max(2, int(h * 0.19))),
        0,
        0,
        360,
        (0, 0, 255),
        1,
    )
