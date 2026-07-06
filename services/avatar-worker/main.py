"""Patient avatar: idle loop + Wav2Lip when agent speaks."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from livekit import api, rtc

from lip_sync import Wav2LipEngine, save_wav_int16

load_dotenv()

logger = logging.getLogger("avatar-worker")
logging.basicConfig(level=logging.INFO)


class PingPongLoop:
    def __init__(self, path: str) -> None:
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
        logger.info("Loaded %d frames (%dx%d)", len(self.frames), self.width, self.height)

    def next_frame(self) -> np.ndarray:
        frame = self.frames[self._index]
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


class AvatarPublisher:
    def __init__(
        self,
        loop: PingPongLoop,
        source: rtc.VideoSource,
        lipsync: Wav2LipEngine | None,
        fps: int,
    ) -> None:
        self.loop = loop
        self.source = source
        self.lipsync = lipsync
        self.fps = fps
        self._lipsync_queue: deque[np.ndarray] = deque()
        self._audio_buffer: list[np.ndarray] = []
        self._last_audio_at = 0.0
        self._last_speech_at = 0.0
        self._last_energy = 0.0
        self._processing = False
        self._silence_seconds = float(os.environ.get("UTTERANCE_SILENCE_SEC", "0.45"))
        self._min_utterance_sec = float(os.environ.get("MIN_UTTERANCE_SEC", "0.35"))
        self._max_utterance_sec = float(os.environ.get("MAX_UTTERANCE_SEC", "12"))
        self._speech_threshold = float(os.environ.get("SPEECH_ENERGY_THRESHOLD", "800"))
        self._subscribed_audio_sids: set[str] = set()

    def push_lipsync_frames(self, frames: list[np.ndarray]) -> None:
        self._lipsync_queue.extend(frames)

    def append_audio(self, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        pcm = pcm.astype(np.int16)
        energy = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
        self._last_energy = energy
        # Agent tracks include silent frames; only buffer actual speech.
        if energy < self._speech_threshold:
            return
        now = time.monotonic()
        self._audio_buffer.append(pcm)
        self._last_speech_at = now
        self._last_audio_at = now

    async def maybe_flush_utterance(self) -> None:
        if self._processing or not self._audio_buffer:
            return

        samples = np.concatenate(self._audio_buffer)
        duration = len(samples) / 48000
        silence_elapsed = time.monotonic() - self._last_speech_at
        force_chunk = duration >= self._max_utterance_sec
        if not force_chunk and silence_elapsed < self._silence_seconds:
            return

        if force_chunk:
            max_samples = int(self._max_utterance_sec * 48000)
            chunk = samples[:max_samples]
            remainder = samples[max_samples:]
            self._audio_buffer = [remainder] if len(remainder) else []
            samples = chunk
            duration = len(samples) / 48000
        else:
            self._audio_buffer.clear()
        if duration < self._min_utterance_sec:
            return
        if not self.lipsync or not self.lipsync.is_ready():
            logger.warning(
                "Agent spoke (%.1fs) but Wav2Lip not ready — run setup_wav2lip.sh",
                duration,
            )
            return

        self._processing = True
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            save_wav_int16(wav_path, samples, 48000)
            logger.info("Lip-syncing %.2fs of agent audio…", duration)
            frames = await asyncio.get_running_loop().run_in_executor(
                None, self.lipsync.sync_utterance, wav_path
            )
            Path(wav_path).unlink(missing_ok=True)
            if frames:
                # Resize to stream dimensions if needed
                resized = []
                for fr in frames:
                    if fr.shape[0] != self.loop.height or fr.shape[1] != self.loop.width:
                        fr = cv2.resize(
                            fr,
                            (self.loop.width, self.loop.height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    resized.append(fr)
                self.push_lipsync_frames(resized)
                logger.info("Queued %d lip-synced frames for playback", len(resized))
        except Exception:
            logger.exception("Lip sync failed")
        finally:
            self._processing = False

    def _is_speaking(self) -> bool:
        return time.monotonic() - self._last_audio_at < 0.12

    def next_display_frame(self) -> np.ndarray:
        if self._lipsync_queue:
            return self._lipsync_queue.popleft()
        # While agent TTS is playing, advance the idle loop faster so the face
        # moves until Wav2Lip frames are ready (or as fallback in mock mode).
        if self._is_speaking():
            extra_steps = min(4, 1 + int(self._last_energy / 4000))
            frame = self.loop.next_frame()
            for _ in range(extra_steps - 1):
                frame = self.loop.next_frame()
            return frame
        return self.loop.next_frame()

    async def publish_forever(self, room: rtc.Room) -> None:
        interval = 1.0 / self.fps
        while room.isconnected():
            await self.maybe_flush_utterance()
            rgba = self.next_display_frame()
            frame = rtc.VideoFrame(
                self.loop.width,
                self.loop.height,
                rtc.VideoBufferType.RGBA,
                rgba.tobytes(),
            )
            self.source.capture_frame(frame)
            await asyncio.sleep(interval)


async def consume_agent_audio(
    track: rtc.Track,
    publisher: AvatarPublisher,
) -> None:
    """Buffer agent TTS audio and trigger lip sync after each utterance."""
    stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1)
    logger.info("Subscribed to agent audio for lip sync")
    async for event in stream:
        frame = event.frame
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        publisher.append_audio(pcm)


async def run_avatar(room_name: str, loop_path: str, fps: int, mode: str) -> None:
    url = os.environ["LIVEKIT_URL"]
    loop = PingPongLoop(loop_path)

    lipsync: Wav2LipEngine | None = None
    if mode == "wav2lip":
        lipsync = Wav2LipEngine(loop_video_path=loop_path)
        if lipsync.is_ready():
            logger.info("Wav2Lip lip sync enabled (face=%s)", lipsync.loop_video_path)
        else:
            logger.warning(
                "AVATAR_MODE=wav2lip but Wav2Lip not installed. "
                "Run: bash setup_wav2lip.sh — idle loop only until then."
            )
    elif mode != "mock":
        logger.warning("Unknown AVATAR_MODE=%s, using mock loop", mode)

    room = rtc.Room()
    await room.connect(url, make_token(room_name))
    logger.info("Avatar joined room %s", room_name)

    source = rtc.VideoSource(loop.width, loop.height)
    track = rtc.LocalVideoTrack.create_video_track("patient-video", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
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
        sid = track.sid
        if sid in publisher._subscribed_audio_sids:
            return
        publisher._subscribed_audio_sids.add(sid)
        logger.info(
            "Agent audio track from %s (sid=%s) — lip sync %s",
            participant.identity,
            sid,
            "enabled" if lipsync and lipsync.is_ready() else "pending (run setup_wav2lip.sh)",
        )
        asyncio.create_task(consume_agent_audio(track, publisher))

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logger.info("Participant joined: %s", participant.identity)
        for pub in participant.track_publications.values():
            if pub.track and pub.subscribed:
                maybe_subscribe_agent_audio(pub.track, participant)

    @room.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if publication.track:
            maybe_subscribe_agent_audio(publication.track, participant)

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        maybe_subscribe_agent_audio(track, participant)

    # Catch participants already in room when avatar connects
    for participant in room.remote_participants.values():
        logger.info("Already in room: %s", participant.identity)
        for pub in participant.track_publications.values():
            if pub.track:
                maybe_subscribe_agent_audio(pub.track, participant)

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
