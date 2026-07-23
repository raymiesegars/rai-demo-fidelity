"""Locked-framing composite helpers for face-only / portrait models.

Goal: keep the uploaded photo's full dimensions on screen. Models that only
animate a face (or tight head crop) get a FIXED source rectangle — the
animated patch is resized into that rect every frame. The rect never tracks
shoulders or recenters, so body motion in the crop does not "swim" against
the static background.
"""

from __future__ import annotations

from pathlib import Path

# framing 0..1: 0=close-up (model fills frame), >=0.25=composite static bg + face
DEFAULT_FRAMING = 1.0
FRAMING_COMPOSITE = 0.25
FRAMING_FULL = 0.98
FACE_RATIO_MIN = 1.4
FACE_RATIO_MAX = 8.0
ANIM_FACE_RATIO = 1.85


def framing_uses_composite(framing: float) -> bool:
    return framing >= FRAMING_COMPOSITE


def framing_to_ratio(framing: float) -> float | None:
    """Map UI framing to face crop ratio for generated (non-composite) mode."""
    if framing >= FRAMING_FULL:
        return FACE_RATIO_MAX
    t = max(0.0, min(framing, FRAMING_FULL)) / FRAMING_FULL
    return FACE_RATIO_MIN + t * (FACE_RATIO_MAX - FACE_RATIO_MIN)


def anim_ratio_for_framing(framing: float) -> float:
    if framing >= FRAMING_FULL:
        return 1.65
    if framing >= 0.66:
        return 2.0
    if framing >= 0.33:
        return 1.85
    return ANIM_FACE_RATIO


def bbox_to_norm(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> list[float]:
    return [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]


def compute_face_crop_box(
    boxes_abs: list[float], img_w: int, img_h: int, ratio: float
) -> tuple[int, int, int, int]:
    """Square face crop box kept in-bounds. Computed once at prepare time."""
    x1, y1, x2, y2 = boxes_abs
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    side = width * ratio
    dis_y_up = side * 0.55
    left = center_x - side / 2
    top = center_y - dis_y_up
    right = left + side
    bottom = top + side

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > img_w:
        left -= right - img_w
        right = img_w
    if bottom > img_h:
        top -= bottom - img_h
        bottom = img_h

    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(img_w), right)
    bottom = min(float(img_h), bottom)

    w, h = right - left, bottom - top
    side = min(w, h)
    if side < 1:
        raise ValueError("Face crop too small for this image")
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    left = max(0.0, min(cx - side / 2, img_w - side))
    top = max(0.0, min(cy - side / 2, img_h - side))
    right = left + side
    bottom = top + side
    return int(left), int(top), int(right), int(bottom)


def letterbox_pil(img, size: tuple[int, int]):
    from PIL import Image

    if not isinstance(img, Image.Image):
        img = Image.open(img).convert("RGB")
    else:
        img = img.convert("RGB")
    tw, th = size
    iw, ih = img.size
    scale = min(tw / iw, th / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def cover_pil(img, size: tuple[int, int]):
    from PIL import Image

    if not isinstance(img, Image.Image):
        img = Image.open(img).convert("RGB")
    tw, th = size
    iw, ih = img.size
    scale = max(tw / iw, th / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    left = max(0, (nw - tw) // 2)
    top = max(0, (nh - th) // 2)
    return resized.crop((left, top, left + tw, top + th))


def save_prep_image(pil_image, source_path: str) -> str:
    prep = Path(source_path).parent / f"_prep_{Path(source_path).stem}.png"
    pil_image.save(prep)
    return str(prep)


def feather_composite_bgr(
    background_bgr,
    face_bgr,
    overlay_norm: list[float],
    feather_px: int = 8,
):
    """Paste animated face into static full-res bg using locked normalized overlay.

    overlay_norm = [x1,y1,x2,y2] in 0..1 of the *source* image. Never recompute
    from landmarks each frame — that is what causes shoulder swim.
    """
    import cv2
    import numpy as np

    h, w = background_bgr.shape[:2]
    x1 = int(round(overlay_norm[0] * w))
    y1 = int(round(overlay_norm[1] * h))
    x2 = int(round(overlay_norm[2] * w))
    y2 = int(round(overlay_norm[3] * h))
    rw, rh = max(1, x2 - x1), max(1, y2 - y1)
    face = cv2.resize(face_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
    out = background_bgr.copy()
    if feather_px <= 0:
        out[y1:y2, x1:x2] = face
        return out

    mask = np.ones((rh, rw), dtype=np.float32)
    f = min(feather_px, rw // 4, rh // 4)
    if f > 0:
        for i in range(f):
            a = (i + 1) / (f + 1)
            mask[i, :] *= a
            mask[-(i + 1), :] *= a
            mask[:, i] *= a
            mask[:, -(i + 1)] *= a
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=f * 0.35)
    alpha = mask[..., None]
    roi = out[y1:y2, x1:x2].astype(np.float32)
    blended = face.astype(np.float32) * alpha + roi * (1.0 - alpha)
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out
