"""Blend Wav2Lip lip patches onto the idle loop."""

from __future__ import annotations

import cv2
import numpy as np


def lip_rect(box: list[int]) -> tuple[int, int, int, int]:
    """Tight crop around the mouth only — not the whole lower face."""
    y1, y2, x1, x2 = box
    fh, fw = y2 - y1, x2 - x1
    ly1 = y1 + int(fh * 0.74)
    ly2 = y1 + int(fh * 0.93)
    lx1 = x1 + int(fw * 0.34)
    lx2 = x1 + int(fw * 0.66)
    return ly1, ly2, lx1, lx2


def extract_lip_patch(frame: np.ndarray, box: list[int]) -> np.ndarray:
    ly1, ly2, lx1, lx2 = lip_rect(box)
    ly1 = max(0, ly1)
    lx1 = max(0, lx1)
    ly2 = min(frame.shape[0], ly2)
    lx2 = min(frame.shape[1], lx2)
    return frame[ly1:ly2, lx1:lx2].copy()


def _feather_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=np.float32)
    fade = max(2, min(h, w) // 6)
    for i in range(fade):
        a = (i + 1) / fade
        mask[i, :] *= a
        mask[-i - 1, :] *= a
        mask[:, i] *= a
        mask[:, -i - 1] *= a
    return mask


def composite_lip_patch(
    base: np.ndarray,
    patch: np.ndarray,
    box: list[int],
) -> np.ndarray:
    """Paste a Wav2Lip lip patch onto the current idle frame with soft edges."""
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

    mask = _feather_mask(th, tw)[:, :, np.newaxis]
    roi = out[ly1:ly2, lx1:lx2, :3].astype(np.float32)
    src = resized.astype(np.float32)
    blended = src * mask + roi * (1.0 - mask)
    out[ly1:ly2, lx1:lx2, :3] = blended.astype(np.uint8)
    return out
