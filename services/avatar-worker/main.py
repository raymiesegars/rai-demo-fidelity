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
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from livekit import api, rtc

from lip_sync import Wav2LipEngine, save_wav_int16

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


def mouth_openness_score(frame: np.ndarray, box: list[int]) -> float:
    y1, y2, x1, x2 = box
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return 0.0
    mouth = face[int(face.shape[0] * 0.45) :, :]
    gray = cv2.cvtColor(mouth, cv2.COLOR_RGBA2GRAY)
    return float(np.std(gray) * 1.5 + np.mean(gray) * 0.05)


class AudioReactiveLoop:
    """Ping-pong idle loop; during speech, pick frames by audio energy vs mouth openness."""

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
        self._speak_frame_idx = 0

        self._openness_order: np.ndarray | None = None
        if face_boxes and len(face_boxes) >= len(self.frames):
            scores = [
                mouth_openness_score(f, face_boxes[i % len(face_boxes)])
                for i, f in enumerate(self.frames)
            ]
            self._openness_order = np.argsort(scores)
            logger.info(
                "Audio-reactive mouth drive ready (%d frames, openness mapped)",
                len(self.frames),
            )
        else:
            logger.warning("No face boxes — using fast idle loop only while speaking")

        logger.info("Loaded %d frames (%dx%d)", len(self.frames), self.width, self.height)

    def next_idle_frame(self) -> np.ndarray:
        frame = self.frames[self._index]
        if self._index >= len(self.frames) - 1:
            self._direction = -1
        elif self._index <= 0:
            self._direction = 1
        self._index += self._direction
        return frame

    def speaking_frame(self, energy: float) -> np.ndarray:
        if self._openness_order is None:
            steps = min(5, 2 + int(energy / 2000))
            frame = self.next_idle_frame()
            for _ in range(steps - 1):
                frame = self.next_idle_frame()
            return frame

        self._smooth_energy = 0.55 * self._smooth_energy + 0.45 * energy
        # Map loudness → how open the mouth should look (wider range for visibility)
        t = float(np.clip(self._smooth_energy / 3200.0, 0.0, 1.0))
        t = t**0.65  # exaggerate mouth movement
        rank = int(t * (len(self._openness_order) - 1))
        target = int(self._openness_order[rank])
        # Smooth frame transitions so it doesn't flicker
        blend = 0.35
        self._speak_frame_idx = int(
            (1 - blend) * self._speak_frame_idx + blend * target
        )
        return self.frames[self._speak_frame_idx]


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
        self._play_wav2lip = os.environ.get("WAV2LIP_PLAYBACK", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self._lipsync_queue: deque[np.ndarray] = deque()
        self._audio_buffer: list[np.ndarray] = []
        self._last_audio_at = 0.0
        self._last_speech_at = 0.0
        self._last_energy = 0.0
        self._silence_seconds = float(os.environ.get("UTTERANCE_SILENCE_SEC", "0.35"))
        self._min_utterance_sec = float(os.environ.get("MIN_UTTERANCE_SEC", "0.25"))
        self._chunk_sec = float(os.environ.get("LIP_SYNC_CHUNK_SEC", "2.0"))
        self._buffer_threshold = float(os.environ.get("AUDIO_BUFFER_THRESHOLD", "40"))
        self._animation_threshold = float(os.environ.get("ANIMATION_ENERGY_THRESHOLD", "25"))
        self._active_agent: str | None = None
        self._subscribed_audio_sids: set[str] = set()
        self._lipsync_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="wav2lip"
        )
        self._wav_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2)

        if lipsync and not self._play_wav2lip:
            logger.info(
                "Wav2Lip batch playback OFF — using real-time audio-reactive mouth "
                "(set WAV2LIP_PLAYBACK=1 to enable delayed batch replay)"
            )

    def set_active_agent(self, identity: str) -> None:
        if self._active_agent == identity:
            return
        logger.info("Active agent for lip sync: %s", identity)
        self._active_agent = identity
        self._audio_buffer.clear()
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
        self._audio_buffer.append(pcm)
        self._last_speech_at = now
        self._last_audio_at = now

    def _prepare_chunk(self) -> np.ndarray | None:
        if not self._audio_buffer or not self.lipsync:
            return None
        samples = np.concatenate(self._audio_buffer)
        duration = len(samples) / 48000
        silence_elapsed = time.monotonic() - self._last_speech_at
        speech_active = silence_elapsed < self._silence_seconds

        if speech_active:
            if duration < self._chunk_sec:
                return None
            n = int(self._chunk_sec * 48000)
            chunk = samples[:n]
            remainder = samples[n:]
            self._audio_buffer = [remainder] if len(remainder) else []
            return chunk

        if duration < self._min_utterance_sec:
            self._audio_buffer.clear()
            return None
        self._audio_buffer.clear()
        return samples

    async def maybe_flush_utterance(self) -> None:
        if not self._play_wav2lip:
            return
        chunk = self._prepare_chunk()
        if chunk is None:
            return
        if self._wav_queue.full():
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        save_wav_int16(wav_path, chunk, 48000)
        await self._wav_queue.put(wav_path)

    async def _lipsync_worker(self) -> None:
        if not self._play_wav2lip:
            return
        loop = asyncio.get_running_loop()
        while True:
            wav_path = await self._wav_queue.get()
            if wav_path is None:
                break
            try:
                if self.lipsync:
                    frames = await loop.run_in_executor(
                        self._lipsync_executor, self.lipsync.sync_utterance, wav_path
                    )
                    if frames and self._is_speaking():
                        self._lipsync_queue.extend(frames[: int(self.fps * 0.5)])
            except Exception:
                logger.exception("Lip sync failed")
            finally:
                Path(wav_path).unlink(missing_ok=True)
                self._wav_queue.task_done()

    def _is_speaking(self) -> bool:
        return time.monotonic() - self._last_audio_at < 0.45

    def next_display_frame(self) -> np.ndarray:
        if self._play_wav2lip and self._lipsync_queue and self._is_speaking():
            return self._lipsync_queue.popleft()
        if self._is_speaking():
            return self.loop.speaking_frame(self._last_energy)
        return self.loop.next_idle_frame()

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
