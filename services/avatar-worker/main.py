"""Patient avatar: audio-reactive loop + optional Wav2Lip."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from livekit import api, rtc

from lip_sync import Wav2LipEngine, save_wav_int16
from composite import composite_lip_patch

load_dotenv()

logger = logging.getLogger("avatar-worker")
logging.basicConfig(level=logging.INFO)


def load_face_boxes(wav2lip_root: Path) -> list[list[int]] | None:
    path = wav2lip_root / "temp" / "alan_face_boxes.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        boxes = json.load(f)
    return boxes if boxes else None


class AudioReactiveLoop:
    """Ping-pong idle loop; lip patches are composited on top during speech."""

    def __init__(self, path: str, face_boxes: list[list[int]] | None = None) -> None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open patient loop: {path}")

        self.frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        cap.release()

        if len(self.frames) < 2:
            raise ValueError("Patient loop must have at least 2 frames")

        self._index = 0
        self._direction = 1
        self.height, self.width = self.frames[0].shape[:2]
        self._smooth_energy = 0.0
        self._face_boxes = face_boxes or []
        self._frame_step = max(1, int(os.environ.get("IDLE_FRAME_STEP", "2")))

        if self._face_boxes and len(self._face_boxes) >= len(self.frames):
            logger.info("Per-frame face boxes ready (%d frames)", len(self.frames))
        else:
            logger.warning("No face boxes — lip composite needs cached boxes on RunPod")

        logger.info("Loaded %d frames (%dx%d)", len(self.frames), self.width, self.height)

    def current_box(self) -> list[int] | None:
        if not self._face_boxes:
            return None
        return self._face_boxes[self._index % len(self._face_boxes)]

    def next_idle_frame(self) -> np.ndarray:
        frame = self.frames[self._index]
        for _ in range(self._frame_step - 1):
            if self._index >= len(self.frames) - 1:
                self._direction = -1
            elif self._index <= 0:
                self._direction = 1
            self._index += self._direction
        if self._index >= len(self.frames) - 1:
            self._direction = -1
        elif self._index <= 0:
            self._direction = 1
        self._index += self._direction
        return frame


def make_token(room: str, identity: str = "avatar-patient") -> str:
    token = api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
    token.with_identity(identity).with_name("Alan").with_grants(
        api.VideoGrants(
            room_join=True,
            room=room,
            can_publish=True,
            can_subscribe=True,
        )
    )
    return token.to_jwt()


def is_agent_participant(participant: rtc.RemoteParticipant) -> bool:
    identity = participant.identity.lower()
    if identity == "avatar-patient":
        return False
    if identity.startswith("user-"):
        return False
    return True


@dataclass
class TimedPatch:
    patch: np.ndarray
    play_at: float


class AvatarPublisher:
    def __init__(
        self,
        loop: AudioReactiveLoop,
        source: rtc.VideoSource,
        lipsync: Wav2LipEngine | None,
        fps: int,
    ) -> None:
        self.loop = loop
        self.source = source
        self.lipsync = lipsync
        self.fps = fps
        mouth_drive = os.environ.get("MOUTH_DRIVE", "composite").lower()
        self._use_composite = (
            mouth_drive == "composite" and lipsync is not None and lipsync.is_ready()
        )
        self._play_wav2lip = self._use_composite or os.environ.get(
            "WAV2LIP_PLAYBACK", "0"
        ).lower() in ("1", "true", "yes")
        self._lip_patch_queue: deque[TimedPatch] = deque()
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_start_time = 0.0
        self._chunks_sent = 0
        self._last_audio_at = 0.0
        self._last_speech_at = 0.0
        self._last_energy = 0.0
        self._silence_seconds = float(os.environ.get("UTTERANCE_SILENCE_SEC", "0.35"))
        self._min_utterance_sec = float(os.environ.get("MIN_UTTERANCE_SEC", "0.2"))
        self._first_chunk_sec = float(os.environ.get("LIP_SYNC_FIRST_CHUNK_SEC", "0.2"))
        self._chunk_sec = float(os.environ.get("LIP_SYNC_CHUNK_SEC", "0.4"))
        self._patch_stale_sec = float(os.environ.get("LIP_PATCH_STALE_SEC", "0.25"))
        self._buffer_threshold = float(os.environ.get("AUDIO_BUFFER_THRESHOLD", "40"))
        self._animation_threshold = float(os.environ.get("ANIMATION_ENERGY_THRESHOLD", "25"))
        self._active_agent: str | None = None
        self._subscribed_audio_sids: set[str] = set()
        self._lipsync_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="wav2lip"
        )
        self._wav_queue: asyncio.Queue[tuple[str, float, float] | None] = asyncio.Queue(
            maxsize=4
        )

        if lipsync and self._use_composite:
            logger.info(
                "Mouth drive: composite — lip patches scheduled to agent audio "
                "(first_chunk=%.2fs, chunk=%.2fs)",
                self._first_chunk_sec,
                self._chunk_sec,
            )
        elif lipsync and not self._play_wav2lip:
            logger.info("Mouth drive: idle loop only (set MOUTH_DRIVE=composite)")

    def set_active_agent(self, identity: str) -> None:
        if self._active_agent == identity:
            return
        logger.info("Active agent for lip sync: %s", identity)
        self._active_agent = identity
        self._audio_buffer.clear()
        self._buffer_start_time = 0.0
        self._chunks_sent = 0
        self._lip_patch_queue.clear()
        self._subscribed_audio_sids.clear()

    def append_audio(self, pcm: np.ndarray, agent_identity: str) -> None:
        if pcm.size == 0 or agent_identity != self._active_agent:
            return
        pcm = pcm.astype(np.int16)
        energy = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
        self._last_energy = energy
        now = time.monotonic()
        if energy >= self._animation_threshold:
            self._last_audio_at = now
        if energy < self._buffer_threshold:
            return
        if not self._audio_buffer:
            self._buffer_start_time = now
        self._audio_buffer.append(pcm)
        self._last_speech_at = now
        self._last_audio_at = now

    def _prepare_chunk(self) -> tuple[np.ndarray, float, float] | None:
        if not self._audio_buffer or not self.lipsync:
            return None
        samples = np.concatenate(self._audio_buffer)
        duration = len(samples) / 48000
        silence_elapsed = time.monotonic() - self._last_speech_at
        speech_active = silence_elapsed < self._silence_seconds
        chunk_start = self._buffer_start_time

        if speech_active:
            need = self._first_chunk_sec if self._chunks_sent == 0 else self._chunk_sec
            if self._buffer_start_time == 0:
                self._buffer_start_time = time.monotonic() - duration
                chunk_start = self._buffer_start_time
            if duration < need:
                return None
            n = int(need * 48000)
            chunk = samples[:n]
            remainder = samples[n:]
            self._audio_buffer = [remainder] if len(remainder) else []
            if remainder.size:
                self._buffer_start_time = chunk_start + len(chunk) / 48000
            else:
                self._buffer_start_time = 0.0
            self._chunks_sent += 1
            return chunk, chunk_start, len(chunk) / 48000

        if duration < self._min_utterance_sec:
            self._audio_buffer.clear()
            self._buffer_start_time = 0.0
            self._chunks_sent = 0
            return None
        if self._buffer_start_time == 0:
            self._buffer_start_time = time.monotonic() - duration
            chunk_start = self._buffer_start_time
        self._audio_buffer.clear()
        self._buffer_start_time = 0.0
        self._chunks_sent = 0
        return samples, chunk_start, duration

    def _enqueue_patches(
        self, patches: list[np.ndarray], chunk_start: float, chunk_dur: float
    ) -> None:
        n = len(patches)
        if n == 0:
            return
        now = time.monotonic()
        end = chunk_start + chunk_dur
        lag = now - chunk_start
        kept = 0

        if now >= end:
            if not self._is_speaking():
                logger.info(
                    "Dropped %d lip patches — speech ended (lag=%.0fms)",
                    n,
                    lag * 1000,
                )
                return
            t0, span = now, chunk_dur
            logger.info(
                "Late chunk lag=%.0fms — playing %d patches over %.2fs",
                lag * 1000,
                n,
                span,
            )
        else:
            t0, span = chunk_start, chunk_dur

        for i, patch in enumerate(patches):
            play_at = t0 + (i / n) * span if n > 1 else t0
            if play_at < now - self._patch_stale_sec:
                continue
            self._lip_patch_queue.append(TimedPatch(patch, play_at))
            kept += 1

        if kept == 0:
            logger.info(
                "Skipped %d stale lip patches (lag=%.0fms)",
                n,
                lag * 1000,
            )
            return

        self._lip_patch_queue = deque(
            sorted(self._lip_patch_queue, key=lambda item: item.play_at)
        )
        logger.info(
            "Queued %d/%d lip patches (lag=%.0fms, queue=%d)",
            kept,
            n,
            lag * 1000,
            len(self._lip_patch_queue),
        )

    async def maybe_flush_utterance(self) -> None:
        if not self._play_wav2lip:
            return
        prepared = self._prepare_chunk()
        if prepared is None:
            return
        chunk, chunk_start, chunk_dur = prepared
        if self._wav_queue.full():
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        save_wav_int16(wav_path, chunk, 48000)
        await self._wav_queue.put((wav_path, chunk_start, chunk_dur))

    async def _lipsync_worker(self) -> None:
        if not self._play_wav2lip:
            return
        loop = asyncio.get_running_loop()
        while True:
            item = await self._wav_queue.get()
            if item is None:
                break
            wav_path, chunk_start, chunk_dur = item
            try:
                if self.lipsync:
                    patches = await loop.run_in_executor(
                        self._lipsync_executor, self.lipsync.sync_utterance, wav_path
                    )
                    if patches:
                        self._enqueue_patches(patches, chunk_start, chunk_dur)
            except Exception:
                logger.exception("Lip sync failed")
            finally:
                Path(wav_path).unlink(missing_ok=True)
                self._wav_queue.task_done()

    def _is_speaking(self) -> bool:
        return time.monotonic() - self._last_audio_at < 0.45

    def next_display_frame(self) -> np.ndarray:
        base = self.loop.next_idle_frame()
        if not self._use_composite or not self._lip_patch_queue:
            return base

        now = time.monotonic()
        while (
            self._lip_patch_queue
            and self._lip_patch_queue[0].play_at < now - self._patch_stale_sec
        ):
            self._lip_patch_queue.popleft()

        if not self._lip_patch_queue or self._lip_patch_queue[0].play_at > now:
            return base

        timed = self._lip_patch_queue.popleft()
        box = self.loop.current_box()
        if box is not None and timed.patch.size > 0:
            return composite_lip_patch(base, timed.patch, box)
        return base

    async def _utterance_worker(self) -> None:
        while True:
            try:
                await self.maybe_flush_utterance()
            except Exception:
                logger.exception("Utterance worker error")
            await asyncio.sleep(0.05)

    async def publish_forever(self, room: rtc.Room) -> None:
        utterance_task = asyncio.create_task(self._utterance_worker())
        lipsync_task = asyncio.create_task(self._lipsync_worker())
        interval = 1.0 / self.fps
        tick = 0
        try:
            while room.isconnected():
                rgba = self.next_display_frame()
                frame = rtc.VideoFrame(
                    self.loop.width,
                    self.loop.height,
                    rtc.VideoBufferType.RGBA,
                    rgba.tobytes(),
                )
                self.source.capture_frame(frame)
                tick += 1
                if tick % (self.fps * 15) == 0:
                    speaking = self._is_speaking()
                    logger.info(
                        "Streaming OK — frame %d/%d, speaking=%s",
                        self.loop._index,
                        len(self.loop.frames),
                        speaking,
                    )
                await asyncio.sleep(interval)
        finally:
            utterance_task.cancel()
            if self._play_wav2lip:
                await self._wav_queue.put(None)
            lipsync_task.cancel()
            self._lipsync_executor.shutdown(wait=False, cancel_futures=True)


async def consume_agent_audio(
    track: rtc.Track,
    publisher: AvatarPublisher,
    agent_identity: str,
) -> None:
    stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1)
    logger.info("Listening to agent audio: %s", agent_identity)
    async for event in stream:
        if agent_identity != publisher._active_agent:
            return
        pcm = np.frombuffer(event.frame.data, dtype=np.int16)
        publisher.append_audio(pcm, agent_identity)


async def run_avatar(room_name: str, loop_path: str, fps: int, mode: str) -> None:
    url = os.environ["LIVEKIT_URL"]
    wav2lip_root = Path(os.environ.get("WAV2LIP_ROOT", "/workspace/Wav2Lip"))

    lipsync: Wav2LipEngine | None = None
    face_boxes = load_face_boxes(wav2lip_root)
    if mode == "wav2lip":
        lipsync = Wav2LipEngine(loop_video_path=loop_path)
        if lipsync.is_ready():
            logger.info("Wav2Lip engine ready")
            face_boxes = face_boxes or load_face_boxes(wav2lip_root)

    loop = AudioReactiveLoop(loop_path, face_boxes)

    room = rtc.Room()
    await room.connect(url, make_token(room_name))
    logger.info("Avatar joined room %s", room_name)

    source = rtc.VideoSource(loop.width, loop.height)
    track = rtc.LocalVideoTrack.create_video_track("patient-video", source)
    await room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=False,
            video_encoding=rtc.VideoEncoding(
                max_bitrate=4_000_000,
                max_framerate=float(fps),
            ),
        ),
    )
    logger.info("Publishing patient video track")

    publisher = AvatarPublisher(loop, source, lipsync, fps)

    def maybe_subscribe_agent_audio(
        track: rtc.Track,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if not is_agent_participant(participant):
            return
        if publisher._active_agent != participant.identity:
            return
        sid = track.sid
        if sid in publisher._subscribed_audio_sids:
            return
        publisher._subscribed_audio_sids.add(sid)
        logger.info("Agent audio track: %s", participant.identity)
        asyncio.create_task(consume_agent_audio(track, publisher, participant.identity))

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logger.info("Participant joined: %s", participant.identity)
        if not is_agent_participant(participant):
            return
        publisher.set_active_agent(participant.identity)
        for pub in participant.track_publications.values():
            if pub.track:
                maybe_subscribe_agent_audio(pub.track, participant)

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        maybe_subscribe_agent_audio(track, participant)

    await publisher.publish_forever(room)


async def main() -> None:
    room = os.environ.get("LIVEKIT_ROOM", "patient-demo")
    loop_path = os.environ.get("PATIENT_LOOP_PATH", "assets/alan-loop.mp4")
    fps = int(os.environ.get("TARGET_FPS", "25"))
    mode = os.environ.get("AVATAR_MODE", "mock")

    resolved = Path(loop_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    await run_avatar(room, str(resolved), fps, mode)


if __name__ == "__main__":
    asyncio.run(main())
