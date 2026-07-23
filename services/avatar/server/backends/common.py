"""Shared base for utterance / clip backends (Wav2Lip, MuseTalk, Ditto, …).

FlashHead is continuous generative. Most other models take an audio clip and
return video frames. This base:
  - emits a static full-image idle stream (locked framing)
  - on speech, runs model.render_utterance(audio) then streams frames + PCM
  - keeps the same chunk dict protocol as FlashHead for the FastAPI app
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from framing import (
    DEFAULT_FRAMING,
    anim_ratio_for_framing,
    bbox_to_norm,
    compute_face_crop_box,
    feather_composite_bgr,
    framing_uses_composite,
)

logger = logging.getLogger("avatar.common")

SAMPLE_RATE = 16000
TGT_FPS = 25
CHUNK_FRAMES = 12  # 0.48 s — small enough for responsive idle


def detect_face_box_mediapipe(image_bgr: np.ndarray) -> list[float]:
    """Return absolute [x1,y1,x2,y2] using mediapipe (already in requirements)."""
    import mediapipe as mp

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as fd:
        res = fd.process(rgb)
    if not res.detections:
        raise ValueError("No face detected")
    box = res.detections[0].location_data.relative_bounding_box
    x1 = max(0, int(box.xmin * w))
    y1 = max(0, int(box.ymin * h))
    x2 = min(w, int((box.xmin + box.width) * w))
    y2 = min(h, int((box.ymin + box.height) * h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError("Face box too small")
    return [float(x1), float(y1), float(x2), float(y2)]


class UtteranceStreamEngine:
    """Base class implementing AvatarBackend for clip/utterance models.

    metrics_mode=utterance: gen_ms_history stores full-utterance render times
    (not per stream chunk). busy_ratio = render_s / audio_s.
    """

    backend_id = "utterance"
    backend_name = "Utterance"
    metrics_mode = "utterance"

    def __init__(self, chunk_frames: int = CHUNK_FRAMES, fps: int = TGT_FPS) -> None:
        self.chunk_frames = chunk_frames
        self.fps = fps
        self.chunk_seconds = chunk_frames / fps
        self.slice_samples = int(SAMPLE_RATE * self.chunk_seconds)

        self.prepared = False
        self.busy_ratio = 0.0
        self.started_at = time.time()
        self.chunks_total = 0
        self.chunks_speaking = 0
        self.gen_ms_history: deque[int] = deque(maxlen=300)
        self.stats_lock = threading.Lock()

        self._frame_sink = None
        self._running = False
        self._gen_thread: threading.Thread | None = None
        self._active_clients = 0
        self._client_lock = threading.Lock()
        self._wake = threading.Event()
        self._speech = np.zeros(0, dtype=np.float32)
        self._speech_lock = threading.Lock()
        self._gpu_lock = threading.Lock()

        self._source_image: str | None = None
        self._source_bgr: np.ndarray | None = None
        self._source_size: tuple[int, int] | None = None
        self._framing = DEFAULT_FRAMING
        self._composite = False  # client-side: utterance backends emit full frames
        self._server_composite = True
        self._face_bbox: list[float] | None = None
        self._overlay: list[float] | None = None
        self._face_box_abs: list[int] | None = None  # x1,y1,x2,y2
        self._idle_jpeg: bytes | None = None

        # Queued speaking frames (BGR) + matching pcm float32 @ 16k
        self._play_frames: list[np.ndarray] = []
        self._play_audio = np.zeros(0, dtype=np.float32)
        self._play_lock = threading.Lock()
        self._render_busy = False
        self._last_render_error: str | None = None
        self._render_generation = 0  # bumped each queued utterance
        self._playback_ready = threading.Event()
        self._last_utterance_gen_ms = 0
        self._utterance_gen_ms_pending = 0  # stamp onto first speaking chunk only

    # ---- subclass API ----

    def render_utterance(self, audio_f32_16k: np.ndarray) -> list[np.ndarray]:
        """Return BGR frames at self.fps covering the audio. Subclasses implement."""
        raise NotImplementedError

    def on_prepare(self) -> None:
        """Optional subclass hook after face/source prepared."""
        return

    # ---- prepare / session ----

    def prepare_avatar(self, image_path: str, framing: float = DEFAULT_FRAMING) -> dict:
        posix = Path(image_path).as_posix()
        img = cv2.imread(posix)
        if img is None:
            raise ValueError(f"could not read image: {posix}")
        h, w = img.shape[:2]
        boxes = detect_face_box_mediapipe(img)

        self._source_image = posix
        self._source_bgr = img
        self._source_size = (w, h)
        self._framing = framing
        # Server pastes lips onto the full source; streamed JPEGs are already
        # full-canvas. Tell the client NOT to composite again (that caused
        # "full image stuck on the face" nesting).
        self._composite = False
        self._server_composite = framing_uses_composite(framing)
        self._face_bbox = bbox_to_norm(int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]), w, h)
        # Locked paste rect from a modest expansion of the detected face — not
        # the huge "full image" anim ratio (that made the box ~entire frame).
        paste_ratio = 1.35 if framing_uses_composite(framing) else anim_ratio_for_framing(framing)
        x1, y1, x2, y2 = compute_face_crop_box(boxes, w, h, paste_ratio)
        self._overlay = bbox_to_norm(x1, y1, x2, y2, w, h) if self._server_composite else None
        self._face_box_abs = [x1, y1, x2, y2]
        self._idle_jpeg = self._encode_jpeg(img)
        self.on_prepare()
        self.prepared = True
        logger.info(
            "%s prepared %s (%dx%d) client_composite=%s server_paste=%s box=%s",
            self.backend_id, posix, w, h, self._composite, self._server_composite, self._face_box_abs,
        )
        return self.session_info()

    def reframe_avatar(self, framing: float) -> dict:
        if not self._source_image:
            raise ValueError("no source image")
        return self.prepare_avatar(self._source_image, framing)

    def session_info(self) -> dict:
        w, h = self._source_size or (512, 512)
        return {
            "ready": bool(self.prepared),
            "composite": self._composite,
            "face": self._face_bbox,
            "overlay": self._overlay,
            "source_w": w,
            "source_h": h,
            "framing": self._framing,
            "backend_id": self.backend_id,
            "backend_name": self.backend_name,
        }

    def _encode_jpeg(self, bgr: np.ndarray) -> bytes:
        # 95 keeps mouth detail; 85 was visibly soft after MuseTalk's 256² decode.
        q = int(os.environ.get("AVATAR_JPEG_QUALITY", "95"))
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()

    def composite_model_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """Paste model face output into locked overlay on static source."""
        assert self._source_bgr is not None
        if not self._server_composite or not self._overlay:
            h, w = self._source_bgr.shape[:2]
            return cv2.resize(face_bgr, (w, h))
        return feather_composite_bgr(self._source_bgr, face_bgr, self._overlay)

    # ---- audio / clients ----

    def set_frame_sink(self, sink) -> None:
        self._frame_sink = sink

    def push_speech(self, samples: np.ndarray) -> None:
        with self._speech_lock:
            self._speech = np.concatenate([self._speech, samples.astype(np.float32)])
        self._playback_ready.clear()
        self._last_render_error = None
        self._wake.set()

    def wait_for_playback_or_error(self, timeout_s: float = 600.0) -> str | None:
        """Block until rendered speech is queued for play, or a render error occurs.

        Returns None on success, or an error string. Used so chat can hold the
        turn in a loading state until audio+video are ready together.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._last_render_error:
                return self._last_render_error
            if self._playback_ready.is_set():
                return None
            self._wake.set()
            remaining = max(0.01, deadline - time.time())
            self._playback_ready.wait(timeout=min(0.25, remaining))
        return self._last_render_error or "Timed out waiting for avatar video render"

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
        self._gen_thread = threading.Thread(
            target=self._loop, daemon=True, name=f"{self.backend_id}-gen"
        )
        self._gen_thread.start()

    def stop(self) -> None:
        self._running = False
        self._wake.set()

    # ---- internal loop ----

    def _pop_speech_utterance(self, min_s: float = 0.15, wait_silence_s: float = 0.18) -> np.ndarray | None:
        """Grab buffered speech once it stops growing for a short silence window."""
        with self._speech_lock:
            if len(self._speech) < int(min_s * SAMPLE_RATE):
                return None
            n0 = len(self._speech)
        time.sleep(wait_silence_s)
        with self._speech_lock:
            if len(self._speech) > n0 + int(0.05 * SAMPLE_RATE):
                return None  # still growing
            utt = self._speech.copy()
            self._speech = np.zeros(0, dtype=np.float32)
        return utt

    def _queue_utterance_playback(self, audio: np.ndarray, frames: list[np.ndarray]) -> None:
        # Resample frame count to match audio duration at fps
        n_expected = max(1, int(round(len(audio) / SAMPLE_RATE * self.fps)))
        if len(frames) == 0:
            frames = [self._source_bgr.copy()]  # type: ignore[union-attr]
        if len(frames) != n_expected:
            idx = np.linspace(0, len(frames) - 1, n_expected).astype(int)
            frames = [frames[i] for i in idx]
        with self._play_lock:
            self._play_frames.extend(frames)
            self._play_audio = np.concatenate([self._play_audio, audio.astype(np.float32)])

    def _maybe_render(self) -> None:
        if self._render_busy:
            return
        utt = self._pop_speech_utterance()
        if utt is None:
            return
        self._render_busy = True
        self._last_render_error = None
        self._render_generation += 1
        gen_id = self._render_generation

        def _job():
            t0 = time.time()
            try:
                with self._gpu_lock:
                    frames = self.render_utterance(utt)
                if gen_id != self._render_generation:
                    return
                gen_ms = round((time.time() - t0) * 1000)
                with self.stats_lock:
                    self.gen_ms_history.append(gen_ms)
                    self._last_utterance_gen_ms = gen_ms
                dur = max(0.01, len(utt) / SAMPLE_RATE)
                self.busy_ratio = (gen_ms / 1000) / dur
                self._queue_utterance_playback(utt, frames)
                self._utterance_gen_ms_pending = gen_ms
                self._playback_ready.set()
                logger.info(
                    "%s rendered %d frames for %.2fs audio in %dms (busy_ratio=%.2f)",
                    self.backend_id, len(frames), dur, gen_ms, self.busy_ratio,
                )
            except Exception as e:
                logger.exception("%s render_utterance failed — not playing AV (need a real render)", self.backend_id)
                self._last_render_error = str(e)
                self._playback_ready.clear()
            finally:
                self._render_busy = False
                self._wake.set()

        threading.Thread(target=_job, daemon=True, name=f"{self.backend_id}-render").start()

    def _emit_chunk(self, seq: int, jpegs: list[bytes], audio: np.ndarray, speaking: bool, gen_ms: int) -> None:
        pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
        speech_s = float(np.mean(np.abs(audio)) > 0.01) * (len(audio) / SAMPLE_RATE) if speaking else 0.0
        # better speech_s: non-near-zero samples
        if speaking:
            speech_s = len(audio) / SAMPLE_RATE
        with self.stats_lock:
            self.chunks_total += 1
            if speaking:
                self.chunks_speaking += 1
            # Do NOT append idle/chunk placeholders into gen_ms_history.
            # Utterance render times are recorded only in render_utterance.
        if self._frame_sink:
            self._frame_sink({
                "seq": seq,
                "speaking": speaking,
                "speech_s": round(speech_s, 3),
                "audio_pcm16": pcm16,
                "jpegs": jpegs,
                "gen_ms": gen_ms,
            })

    def _loop(self) -> None:
        seq = 0
        LEAD_S = 0.2
        schedule_start: float | None = None
        while self._running:
            if self.prepared:
                # Render speech even with zero stream clients so /chat and /say
                # can wait_for_playback_or_error without hanging.
                self._maybe_render()

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
            if now < deadline:
                self._wake.wait(timeout=min(deadline - now, 0.05))
                self._wake.clear()
                continue

            # Prefer playing queued speech frames
            need = self.chunk_frames
            need_samples = self.slice_samples
            with self._play_lock:
                if len(self._play_frames) >= need and len(self._play_audio) >= need_samples:
                    frames = self._play_frames[:need]
                    self._play_frames = self._play_frames[need:]
                    audio = self._play_audio[:need_samples]
                    self._play_audio = self._play_audio[need_samples:]
                    speaking = True
                else:
                    frames = []
                    audio = np.zeros(need_samples, dtype=np.float32)
                    speaking = False

            if speaking:
                jpegs = [self._encode_jpeg(f) for f in frames]
                # Attach full-utterance gen_ms on the first speaking chunk only
                # so the UI badge shows the real render cost, not 0.
                if self._utterance_gen_ms_pending:
                    gen_ms = self._utterance_gen_ms_pending
                    self._utterance_gen_ms_pending = 0
                else:
                    gen_ms = 0
            else:
                # idle: static full image — cheap, do not pollute gen_ms_history
                assert self._idle_jpeg is not None
                jpegs = [self._idle_jpeg] * need
                gen_ms = 0

            self._emit_chunk(seq, jpegs, audio, speaking, gen_ms)
            seq += 1
