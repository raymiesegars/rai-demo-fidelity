"""Shim for LiveTalk's missing scripts.inference helpers.

Upstream inference_example.py imports match_size/resize_pad from here, but the
file is not shipped in the public repo. Keep this next to inference_example.py.
"""
from __future__ import annotations

import torch.nn.functional as F
from torchvision import transforms


def match_size(size_list, h, w):
    """Pick [H, W] from size_list closest to the source aspect ratio."""
    if not size_list:
        return [int(h), int(w)]
    src_ar = float(h) / max(float(w), 1e-6)
    best = size_list[0]
    best_diff = abs(float(best[0]) / max(float(best[1]), 1e-6) - src_ar)
    for s in size_list[1:]:
        d = abs(float(s[0]) / max(float(s[1]), 1e-6) - src_ar)
        if d < best_diff:
            best, best_diff = s, d
    return list(best)


def resize_pad(image, ori_size, tgt_size):
    h, w = ori_size
    scale_ratio = max(tgt_size[0] / h, tgt_size[1] / w)
    scale_h = int(h * scale_ratio)
    scale_w = int(w * scale_ratio)

    image = transforms.Resize(size=[scale_h, scale_w])(image)

    padding_h = tgt_size[0] - scale_h
    padding_w = tgt_size[1] - scale_w
    pad_top = padding_h // 2
    pad_bottom = padding_h - pad_top
    pad_left = padding_w // 2
    pad_right = padding_w - pad_left

    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
    return image
