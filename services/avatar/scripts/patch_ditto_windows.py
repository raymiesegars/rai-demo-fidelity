"""Apply Windows-safe Ditto patches (numpy putback / blend fallback)."""

from __future__ import annotations

from pathlib import Path

PUTBACK = r'''import cv2
import numpy as np
from ..utils.get_mask import get_mask


class PutBackNumpy:
    def __init__(
        self,
        mask_template_path=None,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            self.mask_ori_float = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR)
            self.mask_ori_float = mask.astype(np.float32) / 255.0

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        result = mask_warped * frame_warped + (1 - mask_warped) * frame_rgb
        result = np.clip(result, 0, 255)
        result = result.astype(np.uint8)
        return result


class PutBack:
    """Prefer Cython blend; fall back to PutBackNumpy on Windows build failures."""

    def __init__(self, mask_template_path=None):
        try:
            from ..utils.blend import blend_images_cy  # noqa: F401
            self._use_cy = True
        except Exception:
            self._use_cy = False
            self._numpy = PutBackNumpy(mask_template_path)
            return

        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            mask = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        self.mask_ori_float = np.ascontiguousarray(mask)[:, :, 0]
        self.result_buffer = None

    def __call__(self, frame_rgb, render_image, M_c2o):
        if not self._use_cy:
            return self._numpy(frame_rgb, render_image, M_c2o)

        from ..utils.blend import blend_images_cy

        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        self.result_buffer = np.empty((h, w, 3), dtype=np.uint8)
        blend_images_cy(mask_warped, frame_warped, frame_rgb, self.result_buffer)
        return self.result_buffer
'''

BLEND_INIT = r'''"""Blend helper — Cython when available, numpy fallback on Windows."""

from __future__ import annotations

import numpy as np

try:
    import pyximport

    pyximport.install(setup_args={"include_dirs": [np.get_include()]})
    from .blend import blend_images_cy  # type: ignore
except Exception:
    def blend_images_cy(mask_warped, frame_warped, frame_rgb, result):
        """Pure-numpy stand-in for the Cython blender."""
        mask = np.asarray(mask_warped, dtype=np.float32)
        if mask.ndim == 2:
            mask = mask[..., None]
        fw = np.asarray(frame_warped, dtype=np.float32)
        fr = np.asarray(frame_rgb, dtype=np.float32)
        out = mask * fw + (1.0 - mask) * fr
        np.clip(out, 0, 255, out=out)
        result[:] = out.astype(np.uint8)
'''


def patch_vendor(vendor: Path) -> list[str]:
    changed = []
    for name in ("stream_pipeline_offline.py", "stream_pipeline_online.py"):
        path = vendor / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        old = "from core.atomic_components.putback import PutBack\n"
        new = "from core.atomic_components.putback import PutBackNumpy as PutBack\n"
        if old in text:
            path.write_text(text.replace(old, new), encoding="utf-8")
            changed.append(name)

    putback = vendor / "core" / "atomic_components" / "putback.py"
    if putback.is_file():
        putback.write_text(PUTBACK, encoding="utf-8")
        changed.append("putback.py")

    blend = vendor / "core" / "utils" / "blend" / "__init__.py"
    if blend.is_file():
        blend.write_text(BLEND_INIT, encoding="utf-8")
        changed.append("blend/__init__.py")

    # Use imageio-ffmpeg binary when system ffmpeg is missing (common on Windows).
    infer = vendor / "inference.py"
    if infer.is_file():
        text = infer.read_text(encoding="utf-8")
        needle = 'cmd = f\'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}"'
        if "imageio_ffmpeg" not in text and needle in text:
            text = text.replace(
                needle,
                'try:\n'
                '        import imageio_ffmpeg\n'
                '        _ff = imageio_ffmpeg.get_ffmpeg_exe()\n'
                '    except Exception:\n'
                '        _ff = "ffmpeg"\n'
                '    cmd = f\'{_ff} -loglevel error -y -i "{SDK.tmp_output_path}"',
            )
            infer.write_text(text, encoding="utf-8")
            changed.append("inference.py")

    return changed


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "vendor/ditto-talkinghead")
    print("patched:", patch_vendor(root))
