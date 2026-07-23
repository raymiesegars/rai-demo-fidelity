"""Continuous streaming avatar engine on SoulX-FlashHead (Lite or Pro).

One generation loop per session: the model is fed a rolling audio buffer
(silence when idle, TTS speech when talking) and emits frame chunks at
25 fps with persistent motion state — so idle motion, speech, and the
transitions between them are all one continuous video stream.

Chunk math (Lite, LTX VAE stride 8):
  frame_num=33, motion_frames=9 -> 24 new frames per chunk = 0.96 s
  generation ~0.3-0.4 s per chunk on RTX 4090 => ~2.5-3x realtime

Chunk math (Pro, Wan VAE stride 4):
  frame_num=33, motion_frames=5 -> 28 new frames per chunk = 1.12 s
  ~11 FPS on RTX 4090 (paper) — better look, not chat-realtime.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger("avatar.engine")

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "SoulX-FlashHead"
sys.path.insert(0, str(VENDOR))

import compat  # noqa: E402

compat._install_xfuser_stub()  # must precede flash_head imports

from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline  # noqa: E402
import flash_head.src.pipeline.flash_head_pipeline as _fhp  # noqa: E402

# Inference constants (mirrors vendor flash_head/configs/infer_params.yaml)
FRAME_NUM = 33
TGT_FPS = 25
SAMPLE_RATE = 16000
SAMPLE_SHIFT = 5
CACHED_AUDIO_SECONDS = 8
TARGET_H = 512
TARGET_W = 512
SAMPLE_STEPS = 4
COLOR_CORRECTION = 1.0
MOTION_LATENT_NUM = 2
MOTION_SEED = int(os.environ.get("MOTION_SEED", "42"))
# Pro is ~0.2× realtime on a 4090 — live chunk streaming underruns mid-reply.
# Smooth buffer holds speaking chunks until the utterance is fully generated,
# then releases them so the client can play continuously at 25 fps.
JPEG_QUALITY = int(os.environ.get("FLASHHEAD_JPEG_QUALITY", "85"))
# Soft drive fed to FlashHead during silence (not played to speakers).
# 0 = old pure-silence idle. Raise if faces look dead; lower if lips twitch.
IDLE_AUDIO_AMP = float(os.environ.get("IDLE_AUDIO_AMP", "0.045"))
IDLE_BREATH_HZ = float(os.environ.get("IDLE_BREATH_HZ", "0.27"))
IDLE_NOD_HZ = float(os.environ.get("IDLE_NOD_HZ", "0.95"))
IDLE_BLINK_EVERY_S = float(os.environ.get("IDLE_BLINK_EVERY_S", "3.6"))

# framing 0..1: 0=close-up (generated), >=0.25=composite static bg + animated face
DEFAULT_FRAMING = 1.0
FRAMING_COMPOSITE = 0.25
FRAMING_FULL = 0.98
FACE_RATIO_MIN = 1.4
FACE_RATIO_MAX = 8.0
ANIM_FACE_RATIO = 1.85  # tight crop fed to FlashHead in composite mode


def framing_uses_composite(framing: float) -> bool:
    return framing >= FRAMING_COMPOSITE


def framing_to_ratio(framing: float) -> float | None:
    """Map UI framing to face crop ratio for generated (non-composite) mode."""
    if framing >= FRAMING_FULL:
        return FACE_RATIO_MAX
    t = max(0.0, min(framing, FRAMING_FULL)) / FRAMING_FULL
    return FACE_RATIO_MIN + t * (FACE_RATIO_MAX - FACE_RATIO_MIN)


def _anim_ratio_for_framing(framing: float) -> float:
    if framing >= FRAMING_FULL:
        return 1.65
    if framing >= 0.66:
        return 2.0
    if framing >= 0.33:
        return 1.85
    return ANIM_FACE_RATIO


def _bbox_to_norm(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> list[float]:
    return [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]


def _compute_face_crop_box(
    boxes_abs: list[float], img_w: int, img_h: int, ratio: float
) -> tuple[int, int, int, int]:
    """Square face crop box (matches FlashHead facecrop math, kept in-bounds)."""
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


def _read_face_boxes(image_path: str):
    from PIL import Image
    import numpy as np
    from flash_head.utils.cpu_face_handler import CPUFaceHandler

    image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image)
    img_h, img_w = image_rgb.shape[:2]
    boxes, _scores = CPUFaceHandler()(image_rgb)
    if len(boxes) == 0:
        raise ValueError("No face detected")
    boxes_abs = [
        boxes[0][0] * img_w,
        boxes[0][1] * img_h,
        boxes[0][2] * img_w,
        boxes[0][3] * img_h,
    ]
    return image, boxes_abs, img_w, img_h


def _detect_face_norm(image_path: str) -> list[float]:
    _image, boxes_abs, img_w, img_h = _read_face_boxes(image_path)
    return [
        boxes_abs[0] / img_w,
        boxes_abs[1] / img_h,
        boxes_abs[2] / img_w,
        boxes_abs[3] / img_h,
    ]


def _cover_pil(img, size: tuple[int, int] = (TARGET_W, TARGET_H)):
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


def _face_crop_for_composite(image_path: str, ratio: float) -> tuple[object, list[float]]:
    """Return model prep image + the exact source crop rect for client compositing."""
    from PIL import Image

    image, boxes_abs, img_w, img_h = _read_face_boxes(image_path)
    x1, y1, x2, y2 = _compute_face_crop_box(boxes_abs, img_w, img_h, ratio)
    crop_face = image.crop((x1, y1, x2, y2))
    # Square crop -> uniform scale to 512 matches the client's rect overlay mapping.
    prep = crop_face.resize((TARGET_W, TARGET_H), Image.BILINEAR)
    return prep, _bbox_to_norm(x1, y1, x2, y2, img_w, img_h)


def _letterbox_pil(img, size: tuple[int, int] = (TARGET_W, TARGET_H)):
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


def _letterbox_image(image_path: str, size: tuple[int, int] = (TARGET_W, TARGET_H)):
    from PIL import Image

    return _letterbox_pil(Image.open(image_path), size)


def _face_crop_preserve_aspect(image_path: str, ratio: float, size: tuple[int, int] = (TARGET_W, TARGET_H)):
    """Face-centered crop that keeps aspect ratio (no horizontal squish)."""
    from PIL import Image
    import numpy as np
    from flash_head.utils.cpu_face_handler import CPUFaceHandler
    from flash_head.utils.facecrop import get_scaled_bbox

    image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image)
    img_h, img_w = image_rgb.shape[:2]
    boxes, _scores = CPUFaceHandler()(image_rgb)
    if len(boxes) == 0:
        raise ValueError("No face detected")
    boxes_abs = [
        boxes[0][0] * img_w,
        boxes[0][1] * img_h,
        boxes[0][2] * img_w,
        boxes[0][3] * img_h,
    ]
    crop_face = get_scaled_bbox(boxes_abs, img_w, img_h, ratio, image)
    return _letterbox_pil(crop_face, size)


def _save_prep_image(pil_image, source_path: str) -> str:
    prep = Path(source_path).parent / f"_prep_{Path(source_path).stem}.png"
    pil_image.save(prep)
    return str(prep)


def _disable_compile_if_no_triton() -> None:
    try:
        import triton  # noqa: F401
    except ImportError:
        _fhp.COMPILE_MODEL = False
        _fhp.COMPILE_VAE = False
        logger.info("triton missing -> torch.compile disabled")


class AvatarEngine:
    """Owns the FlashHead pipeline and a single continuous session."""

    def __init__(
        self,
        ckpt_dir: str,
        wav2vec_dir: str,
        model_type: str = "lite",
    ) -> None:
        _disable_compile_if_no_triton()
        mt = (model_type or "lite").strip().lower()
        if mt not in ("lite", "pro"):
            raise ValueError(f"FlashHead model_type must be lite|pro, got {model_type!r}")
        t0 = time.time()
        self.model_type = mt
        self.pipeline = FlashHeadPipeline(
            checkpoint_dir=ckpt_dir,
            model_type=mt,
            wav2vec_dir=wav2vec_dir,
            device="cuda",
        )
        stride = self.pipeline.config.vae_stride[0]
        self.motion_frames_num = (MOTION_LATENT_NUM - 1) * stride + 1
        self.slice_frames = FRAME_NUM - self.motion_frames_num  # new frames/chunk
        self.slice_samples = self.slice_frames * SAMPLE_RATE // TGT_FPS
        self.chunk_seconds = self.slice_frames / TGT_FPS
        # Pro defaults: fewer denoise steps (pipeline supports 2) + utterance hold.
        default_steps = "2" if mt == "pro" else str(SAMPLE_STEPS)
        self.sample_steps = int(os.environ.get("FLASHHEAD_SAMPLE_STEPS", default_steps))
        if self.sample_steps < 2:
            self.sample_steps = 2
        smooth_default = "1" if mt == "pro" else "0"
        self.smooth_buffer = os.environ.get("FLASHHEAD_SMOOTH", smooth_default).strip() not in (
            "0", "false", "False", "no", "NO",
        )
        jpeg_default = "72" if mt == "pro" else str(JPEG_QUALITY)
        self.jpeg_quality = int(os.environ.get("FLASHHEAD_JPEG_QUALITY", jpeg_default))
        # Pro idle gen is ~5s/chunk → looks like 2fps freezes. Default: hold last
        # frame / static portrait until the next utterance.
        idle_default = "0" if mt == "pro" else "1"
        self.idle_gen = os.environ.get("FLASHHEAD_IDLE_GEN", idle_default).strip() not in (
            "0", "false", "False", "no", "NO",
        )

        logger.info(
            "FlashHead (%s) ready in %.1fs (chunk=%d frames / %.2fs, motion=%d, "
            "steps=%d, smooth=%s, idle_gen=%s, idle_amp=%.3f)",
            mt, time.time() - t0, self.slice_frames, self.chunk_seconds, self.motion_frames_num,
            self.sample_steps, self.smooth_buffer, self.idle_gen, IDLE_AUDIO_AMP,
        )

        # Rolling audio context fed to wav2vec each chunk.
        self._cache_len = SAMPLE_RATE * CACHED_AUDIO_SECONDS
        self._audio_cache: deque[float] = deque([0.0] * self._cache_len, maxlen=self._cache_len)
        self._audio_end_idx = CACHED_AUDIO_SECONDS * TGT_FPS
        self._audio_start_idx = self._audio_end_idx - FRAME_NUM

        # Speech input buffer (float32 mono 16 kHz).
        self._speech = np.zeros(0, dtype=np.float32)
        self._speech_lock = threading.Lock()

        # Continuous idle drive state (deterministic across faces for consistency).
        self._idle_amp = IDLE_AUDIO_AMP
        self._idle_rng = np.random.default_rng(MOTION_SEED ^ 0xA5A5)
        self._idle_brown = 0.0
        self._idle_sample = 0
        self._idle_next_blink = int(IDLE_BLINK_EVERY_S * SAMPLE_RATE)
        self._idle_blink_pos = -1  # sample index within current blink burst, -1 = none
        self._idle_blink_len = int(0.055 * SAMPLE_RATE)

        self._frame_sink = None       # callable(chunk dict) set by server
        self._gpu_lock = threading.Lock()  # prepare_avatar vs generate
        self._running = False
        self._gen_thread: threading.Thread | None = None
        self._active_clients = 0
        self._client_lock = threading.Lock()
        self._wake = threading.Event()
        self.prepared = False
        self.busy_ratio = 0.0
        self._source_image: str | None = None
        self._source_size: tuple[int, int] | None = None
        self._framing = DEFAULT_FRAMING
        self._composite = True
        self._face_bbox: list[float] | None = None
        self._overlay: list[float] | None = None
        self._prep_image: str | None = None

        # Smooth-buffer hold queue (Pro): speaking chunks released together.
        self._speak_hold: list[dict] = []
        self._playback_ready = threading.Event()
        self._playback_ready.set()
        self._last_render_error: str | None = None

        # Analytics (read by /stats)
        self.stats_lock = threading.Lock()
        self.gen_ms_history: deque[int] = deque(maxlen=300)
        self.chunks_total = 0
        self.chunks_speaking = 0
        self.started_at = time.time()

    def _reset_idle_drive(self) -> None:
        self._idle_rng = np.random.default_rng(MOTION_SEED ^ 0xA5A5)
        self._idle_brown = 0.0
        self._idle_sample = 0
        self._idle_next_blink = int(IDLE_BLINK_EVERY_S * SAMPLE_RATE)
        self._idle_blink_pos = -1

    def _idle_drive(self, n: int) -> np.ndarray:
        """Low-frequency presence signal → more consistent idle motion than pure silence.

        Continuity across chunks matters: brownian state carries forward so the face
        doesn't reset every 0.96s. Client speakers never hear this — model only.
        """
        if n <= 0 or self._idle_amp <= 0:
            return np.zeros(max(0, n), dtype=np.float32)

        t0 = self._idle_sample
        t = (t0 + np.arange(n, dtype=np.float64)) / SAMPLE_RATE

        # Integrated white noise (brown) — carries state across chunks.
        white = self._idle_rng.normal(0.0, 1.0, size=n)
        brown = np.empty(n, dtype=np.float64)
        b = self._idle_brown
        for i, w in enumerate(white):
            b = 0.992 * b + 0.08 * w
            brown[i] = b
        self._idle_brown = float(b)

        # Heavy smoothing kills speech-band energy so lips twitch less.
        kernel = 160  # 10 ms @ 16 kHz
        if n >= kernel:
            cum = np.cumsum(np.concatenate([[0.0], brown]))
            smooth = (cum[kernel:] - cum[:-kernel]) / kernel
            pad = np.full(kernel - 1, smooth[0] if len(smooth) else 0.0)
            brown = np.concatenate([pad, smooth])[:n]
        std = float(np.std(brown))
        if std > 1e-8:
            brown /= std * 3.0

        breath = 0.55 + 0.45 * np.sin(2 * np.pi * IDLE_BREATH_HZ * t)
        nod = 0.22 * np.sin(2 * np.pi * IDLE_NOD_HZ * t + 0.4)
        sig = self._idle_amp * breath * (0.65 * brown + nod)

        # Soft bursts on a steady cadence → nudge blinks / micro-expressions.
        # Burst state spans chunk boundaries so it doesn't clip mid-blink.
        blink_len = max(8, self._idle_blink_len)
        i = 0
        while i < n:
            abs_i = t0 + i
            if self._idle_blink_pos < 0 and abs_i >= self._idle_next_blink:
                self._idle_blink_pos = 0
                gap = IDLE_BLINK_EVERY_S * (0.82 + 0.36 * float(self._idle_rng.random()))
                self._idle_next_blink = abs_i + int(gap * SAMPLE_RATE)
            if self._idle_blink_pos >= 0:
                while i < n and self._idle_blink_pos < blink_len:
                    w = 0.5 - 0.5 * np.cos(2 * np.pi * self._idle_blink_pos / (blink_len - 1))
                    sig[i] += self._idle_amp * 0.85 * w
                    self._idle_blink_pos += 1
                    i += 1
                if self._idle_blink_pos >= blink_len:
                    self._idle_blink_pos = -1
                continue
            i += 1

        self._idle_sample = t0 + n
        return np.clip(sig, -0.25, 0.25).astype(np.float32)

    def _model_audio(self, chunk: np.ndarray, speech_s: float) -> np.ndarray:
        """Audio fed to FlashHead: real speech when talking, idle drive when silent."""
        if self._idle_amp <= 0:
            return chunk
        n = len(chunk)
        speech_n = int(round(speech_s * SAMPLE_RATE))
        speech_n = max(0, min(n, speech_n))
        if speech_n >= n:
            return chunk
        out = chunk.copy()
        out[speech_n:] = self._idle_drive(n - speech_n)
        return out

    # ---------------- avatar / image ----------------

    def prepare_avatar(self, image_path: str, framing: float = DEFAULT_FRAMING) -> dict:
        """Encode a portrait image as the active avatar (resets motion state)."""
        from PIL import Image

        posix = Path(image_path).as_posix()
        with Image.open(posix) as src:
            self._source_size = src.size  # (w, h)
        t0 = time.time()
        with self._gpu_lock:
            self._source_image = posix
            self._framing = framing
            self._prepare_locked(posix, framing)
        self.prepared = True
        logger.info("avatar prepared from %s in %.1fs (framing=%.2f)", posix, time.time() - t0, framing)
        return self.session_info()

    def reframe_avatar(self, framing: float) -> dict:
        """Re-crop the last uploaded source image with a new framing level."""
        if not self._source_image:
            raise ValueError("no source image uploaded yet")
        t0 = time.time()
        with self._gpu_lock:
            self._framing = framing
            self._prepare_locked(self._source_image, framing)
        logger.info("avatar reframed in %.1fs (framing=%.2f)", time.time() - t0, framing)
        return self.session_info()

    def session_info(self) -> dict:
        w, h = self._source_size or (TARGET_W, TARGET_H)
        return {
            "composite": self._composite,
            "face": self._face_bbox,
            "overlay": self._overlay,
            "source_w": w,
            "source_h": h,
            "framing": self._framing,
            "smooth_buffer": self.smooth_buffer,
            "idle_gen": self.idle_gen,
            "model_type": self.model_type,
        }

    def _prepare_locked(self, posix: str, framing: float) -> None:
        self._composite = framing_uses_composite(framing)
        self._face_bbox = _detect_face_norm(posix)
        anim_ratio = _anim_ratio_for_framing(framing)

        if self._composite:
            crop, self._overlay = _face_crop_for_composite(posix, anim_ratio)
        else:
            self._overlay = None
            crop = _face_crop_preserve_aspect(
                posix, framing_to_ratio(framing) or ANIM_FACE_RATIO, (TARGET_W, TARGET_H)
            )

        if self._prep_image and self._prep_image != str(Path(posix).parent / f"_prep_{Path(posix).stem}.png"):
            try:
                Path(self._prep_image).unlink(missing_ok=True)
            except OSError:
                pass
        cond_path = _save_prep_image(crop, posix)
        self._prep_image = cond_path

        self.pipeline.prepare_params(
            cond_image_path_or_dir=cond_path,
            target_size=(TARGET_H, TARGET_W),
            frame_num=FRAME_NUM,
            motion_frames_num=self.motion_frames_num,
            sampling_steps=self.sample_steps,
            seed=MOTION_SEED,
            shift=SAMPLE_SHIFT,
            color_correction_strength=COLOR_CORRECTION,
            use_face_crop=False,
        )
        with self._speech_lock:
            self._speech = np.zeros(0, dtype=np.float32)
        self._audio_cache.clear()
        self._audio_cache.extend([0.0] * self._cache_len)
        self._reset_idle_drive()

    # ---------------- audio input ----------------

    def push_speech(self, samples: np.ndarray) -> None:
        if self.smooth_buffer:
            self._playback_ready.clear()
            self._last_render_error = None
        with self._speech_lock:
            self._speech = np.concatenate([self._speech, samples.astype(np.float32)])
        self._wake.set()

    def speech_buffered_seconds(self) -> float:
        with self._speech_lock:
            return len(self._speech) / SAMPLE_RATE

    def wait_for_playback_or_error(self, timeout_s: float = 600.0) -> str | None:
        """Block until smooth-buffer flush (or no-op when smooth is off)."""
        if not self.smooth_buffer:
            return None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._last_render_error:
                return self._last_render_error
            if self._playback_ready.is_set():
                return None
            self._playback_ready.wait(timeout=0.25)
        return self._last_render_error or "Timed out waiting for FlashHead Pro smooth buffer"

    def _flush_speak_hold(self) -> None:
        if not self._speak_hold:
            self._playback_ready.set()
            return
        n = len(self._speak_hold)
        logger.info("FlashHead smooth flush: releasing %d speaking chunk(s)", n)
        sink = self._frame_sink
        if sink:
            for chunk in self._speak_hold:
                sink(chunk)
        self._speak_hold.clear()
        self._playback_ready.set()

    def _emit_chunk(self, chunk: dict, speaking: bool) -> None:
        """Send to clients, or hold speaking chunks until the utterance is done (Pro)."""
        if not self.smooth_buffer:
            if self._frame_sink:
                self._frame_sink(chunk)
            return
        if speaking:
            self._speak_hold.append(chunk)
            # Flush as soon as speech is drained — don't wait for a slow idle generate.
            if self.speech_buffered_seconds() < 0.05:
                self._flush_speak_hold()
            return
        if self._speak_hold:
            self._flush_speak_hold()
        # Skip streaming idle chunks when idle_gen is off (Pro default).
        if self.idle_gen and self._frame_sink:
            self._frame_sink(chunk)

    def _pop_slice(self) -> tuple[np.ndarray, float]:
        """Take one chunk of speech; pad with silence when empty.

        Returns (audio, speech_seconds_in_chunk).
        """
        with self._speech_lock:
            take = min(len(self._speech), self.slice_samples)
            chunk = self._speech[:take]
            self._speech = self._speech[take:]
        if take < self.slice_samples:
            chunk = np.concatenate([chunk, np.zeros(self.slice_samples - take, dtype=np.float32)])
        return chunk, take / SAMPLE_RATE

    def clear_speech(self) -> None:
        """Drop buffered speech (interruption support)."""
        with self._speech_lock:
            self._speech = np.zeros(0, dtype=np.float32)
        self._speak_hold.clear()
        self._playback_ready.set()

    # ---------------- generation loop ----------------

    def set_frame_sink(self, sink) -> None:
        self._frame_sink = sink

    def client_connected(self) -> None:
        with self._client_lock:
            self._active_clients += 1
        self._wake.set()

    def client_disconnected(self) -> None:
        with self._client_lock:
            self._active_clients = max(0, self._active_clients - 1)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._gen_thread = threading.Thread(target=self._loop, daemon=True, name="flashhead-gen")
        self._gen_thread.start()

    def _audio_embedding(self, audio_array: np.ndarray) -> torch.Tensor:
        emb = self.pipeline.preprocess_audio(audio_array, sr=SAMPLE_RATE, fps=TGT_FPS)
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        centers = torch.arange(self._audio_start_idx, self._audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
        centers = torch.clamp(centers, min=0, max=self._audio_end_idx - 1)
        return emb[centers][None, ...].contiguous()

    def _loop(self) -> None:
        seq = 0
        # Idle chunks are paced just ahead of realtime (small lead keeps the
        # client buffer shallow so speech isn't stuck behind queued silence).
        # When speech is buffered we generate EAGERLY — the client fast-forwards
        # through pending silence, so getting speech chunks out fast is what
        # determines time-to-first-word.
        LEAD_S = 0.25
        MAX_EAGER_LEAD_S = 3.0
        schedule_start: float | None = None
        while self._running:
            with self._client_lock:
                has_clients = self._active_clients > 0
            if not has_clients or not self.prepared or self._frame_sink is None:
                schedule_start = None
                self._wake.wait(timeout=0.25)
                self._wake.clear()
                continue

            now = time.monotonic()
            if schedule_start is None:
                schedule_start = now
                seq = 0
            deadline = schedule_start + seq * self.chunk_seconds - LEAD_S
            has_speech = self.speech_buffered_seconds() > 0

            # Pro: do not run ~5s GPU idle chunks (looks like multi-second freezes).
            # Hold last face / static portrait until the next utterance.
            if not has_speech and not self.idle_gen:
                if self.smooth_buffer and self._speak_hold:
                    self._flush_speak_hold()
                schedule_start = None
                self._wake.wait(timeout=0.25)
                self._wake.clear()
                continue

            if has_speech:
                # Eager mode: run ahead of the pacing clock (bounded).
                if now < deadline - MAX_EAGER_LEAD_S:
                    time.sleep(0.01)
                    continue
            elif now < deadline:
                self._wake.wait(timeout=min(deadline - now, 0.05))
                self._wake.clear()
                continue
            if now - deadline > 2 * self.chunk_seconds:
                # Fell behind (slow gen or hiccup) — resync the clock.
                schedule_start = now - seq * self.chunk_seconds

            chunk_audio, speech_s = self._pop_slice()
            speaking = speech_s > 0.02
            # Idle drive goes to the model only — speakers still get clean PCM
            # (zeros while idle) so there's no audible hiss.
            self._audio_cache.extend(self._model_audio(chunk_audio, speech_s).tolist())

            t0 = time.time()
            try:
                with self._gpu_lock:
                    emb = self._audio_embedding(np.asarray(self._audio_cache))
                    emb = emb.to(self.pipeline.device)
                    sample = self.pipeline.generate(emb)  # (C,F,H,W) float32 [-1,1]
            except Exception:
                logger.exception("generation failed; resetting chunk")
                if self.smooth_buffer and self._speak_hold:
                    self._last_render_error = "FlashHead generation failed mid-utterance"
                    self._speak_hold.clear()
                    self._playback_ready.set()
                continue

            frames = (((sample + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255)
            frames = frames[self.motion_frames_num:].to(torch.uint8).cpu().numpy()
            gen_s = time.time() - t0
            self.busy_ratio = gen_s / self.chunk_seconds

            jpegs = []
            for f in frames:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if ok:
                    jpegs.append(buf.tobytes())

            pcm16 = (np.clip(chunk_audio, -1, 1) * 32767).astype(np.int16).tobytes()
            with self.stats_lock:
                self.gen_ms_history.append(round(gen_s * 1000))
                self.chunks_total += 1
                if speaking:
                    self.chunks_speaking += 1
            self._emit_chunk({
                "seq": seq,
                "speaking": speaking,
                "speech_s": round(speech_s, 3),
                "audio_pcm16": pcm16,
                "jpegs": jpegs,
                "gen_ms": round(gen_s * 1000),
            }, speaking=speaking)
            seq += 1

    def stop(self) -> None:
        self._running = False
        self._wake.set()
