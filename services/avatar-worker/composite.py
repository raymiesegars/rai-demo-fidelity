"""Blend Wav2Lip lip patches onto the idle loop."""

from __future__ import annotations

import os

import cv2
import numpy as np


def lip_rect(box: list[int]) -> tuple[int, int, int, int]:
    """Tight crop on the mouth — tuned to sit on the lip line, not the chin."""
    y1, y2, x1, x2 = box
    fh, fw = y2 - y1, x2 - x1
    # Nudge up via env (negative = higher on face)
    y_shift = float(os.environ.get("LIP_RECT_Y_SHIFT", "-0.04"))
    x_left = float(os.environ.get("LIP_RECT_X_LEFT", "0.30"))
    x_right = float(os.environ.get("LIP_RECT_X_RIGHT", "0.70"))
    top = float(os.environ.get("LIP_RECT_TOP", "0.64")) + y_shift
    bottom = float(os.environ.get("LIP_RECT_BOTTOM", "0.86")) + y_shift
    ly1 = y1 + int(fh * top)
    ly2 = y1 + int(fh * bottom)
    lx1 = x1 + int(fw * x_left)
    lx2 = x1 + int(fw * x_right)
    return ly1, ly2, lx1, lx2


def extract_lip_patch(frame: np.ndarray, box: list[int]) -> np.ndarray:
    ly1, ly2, lx1, lx2 = lip_rect(box)
    ly1 = max(0, ly1)
    lx1 = max(0, lx1)
    ly2 = min(frame.shape[0], ly2)
    lx2 = min(frame.shape[1], lx2)
    return frame[ly1:ly2, lx1:lx2].copy()


def _mouth_mask(h: int, w: int) -> np.ndarray:
    """Oval mask: full opacity in the center, soft edge only at the border."""
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    axes = (max(2, w // 2 - 3), max(2, h // 2 - 3))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
    fade = max(3, min(h, w) // 5)
    blur = cv2.GaussianBlur(mask, (fade | 1, fade | 1), 0)
    return np.clip(blur, 0.0, 1.0)


def _match_border_color(src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pull patch color toward the underlying skin at the mouth edge."""
    edge = mask < 0.45
    if not np.any(edge):
        return src
    dst_mean = dst[edge].mean(axis=0)
    src_mean = src[edge].mean(axis=0)
    corrected = src + (dst_mean - src_mean)
    return np.clip(corrected, 0, 255)


def composite_lip_patch(
    base: np.ndarray,
    patch: np.ndarray,
    box: list[int],
) -> np.ndarray:
    """Paste a Wav2Lip lip patch onto the current idle frame."""
    if patch.size == 0:
        return base

    ly1, ly2, lx1, lx2 = lip_rect(box)
    th, tw = ly2 - ly1, lx2 - lx1
    if th < 4 or tw < 4:
        return base

    out = base.copy()
    resized = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    if resized.shape[2] == 4:
        resized = resized[:, :, :3]

    mask = _mouth_mask(th, tw)[:, :, np.newaxis]
    roi = out[ly1:ly2, lx1:lx2, :3].astype(np.float32)
    src = resized.astype(np.float32)
    src = _match_border_color(src, roi, mask[:, :, 0])
    blended = src * mask + roi * (1.0 - mask)
    out[ly1:ly2, lx1:lx2, :3] = blended.astype(np.uint8)
    return out
