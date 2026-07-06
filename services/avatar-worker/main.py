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
from speech_reactive import SpeechReactiveMouth
from liveportrait_engine import LivePortraitEngine, preflight_liveportrait

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

    @property
    def current_index(self) -> int:
        return self._index

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
class TimedFrame:
    frame: np.ndarray
    play_at: float


class AvatarPublisher:
    def __init__(
        self,
        loop: AudioReactiveLoop,
        source: rtc.VideoSource,
        fps: int,
        lipsync: Wav2LipEngine | None = None,
        liveportrait: LivePortraitEngine | None = None,
    ) -> None:
        self.loop = loop
        self.source = source
        self.fps = fps
        mouth_drive = os.environ.get("MOUTH_DRIVE", "idle").lower()
        avatar_mode = os.environ.get("AVATAR_MODE", "mock").lower()

        self._lp = liveportrait if liveportrait and liveportrait.is_ready() else None
        self._wav = None
        self._reactive: SpeechReactiveMouth | None = None

        if avatar_mode == "reactive" or mouth_drive == "reactive":
            self._drive = "reactive"
            self._reactive = SpeechReactiveMouth()
        elif self._lp is not None:
            self._drive = "liveportrait"
        elif mouth_drive == "composite" and lipsync and lipsync.is_ready():
            self._wav = lipsync
            self._drive = "composite"
        else:
            self._drive = "idle"

        self._frame_queue: deque[TimedFrame] = deque()
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_start_time = 0.0
        self._chunks_sent = 0
        self._last_audio_at = 0.0
        self._last_speech_at = 0.0
        self._last_energy = 0.0
        self._silence_seconds = float(os.environ.get("UTTERANCE_SILENCE_SEC", "0.35"))
        self._min_utterance_sec = float(os.environ.get("MIN_UTTERANCE_SEC", "0.2"))
        self._first_chunk_sec = float(
            os.environ.get(
                "LIP_SYNC_FIRST_CHUNK_SEC",
                "0.5" if self._drive == "liveportrait" else "0.2",
            )
        )
        self._chunk_sec = float(
            os.environ.get(
                "LIP_SYNC_CHUNK_SEC",
                "0.8" if self._drive == "liveportrait" else "0.4",
            )
        )
        self._patch_stale_sec = float(os.environ.get("LIP_PATCH_STALE_SEC", "0.25"))
        self._max_lip_lag_sec = float(os.environ.get("MAX_LIP_LAG_SEC", "2.5"))
        self._buffer_threshold = float(os.environ.get("AUDIO_BUFFER_THRESHOLD", "40"))
        self._animation_threshold = float(os.environ.get("ANIMATION_ENERGY_THRESHOLD", "8"))
        self._active_agent: str | None = None
        self._agent_locked = False
        self._subscribed_audio_sids: set[str] = set()
        self._users_in_room = 0
        self._agent_seq = 0
        self._agent_order: dict[str, int] = {}
        self._drive_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lip-drive"
        )
        self._wav_queue: asyncio.Queue[tuple[str, float, float, int, bool] | None] = (
            asyncio.Queue(maxsize=1)
        )
        self._reset_next_motion = True
        self._logged_audio = False

        if self._drive == "reactive":
            logger.info(
                "Mouth drive: reactive — instant audio-synced mouth on loop "
                "(no LivePortrait / Wav2Lip)"
            )
        elif self._drive == "liveportrait":
            logger.info(
                "Mouth drive: liveportrait — JoyVASA + LivePortrait "
                "(first_chunk=%.2fs, chunk=%.2fs)",
                self._first_chunk_sec,
                self._chunk_sec,
            )
        elif self._drive == "composite":
            logger.info(
                "Mouth drive: composite (legacy wav2lip) "
                "(first_chunk=%.2fs, chunk=%.2fs)",
                self._first_chunk_sec,
                self._chunk_sec,
            )
        else:
            logger.info("Mouth drive: idle — ping-pong loop only")

    def note_agent(self, identity: str) -> None:
        if identity not in self._agent_order:
            self._agent_seq += 1
            self._agent_order[identity] = self._agent_seq

    def newest_agent_identity(self, room: rtc.Room) -> str | None:
        agents = [
            p.identity
            for p in room.remote_participants.values()
            if is_agent_participant(p)
        ]
        if not agents:
            return None
        return max(agents, key=lambda i: self._agent_order.get(i, 0))

    def unlock_agent(self) -> None:
        """Call when a user joins so we can bind to the live session agent."""
        if self._active_agent is None and not self._agent_locked:
            return
        logger.info("User joined — clearing agent lock for fresh session")
        self._active_agent = None
        self._agent_locked = False
        self._subscribed_audio_sids.clear()
        self._logged_audio = False
        if self._reactive is not None:
            self._reactive.reset()

    def user_joined(self) -> None:
        self._users_in_room += 1

    def bind_agent_audio(
        self,
        track: rtc.Track,
        identity: str,
        *,
        force: bool = False,
    ) -> None:
        if self._users_in_room == 0:
            return
        sid = track.sid
        if (
            not force
            and self._agent_locked
            and self._active_agent
            and self._active_agent != identity
        ):
            return
        self.set_active_agent(identity, force=force)
        if sid in self._subscribed_audio_sids:
            return
        self._subscribed_audio_sids.add(sid)
        logger.info("Agent audio track: %s", identity)
        asyncio.create_task(consume_agent_audio(track, self, identity))

    def set_active_agent(self, identity: str, *, force: bool = False) -> None:
        if self._active_agent == identity:
            return
        if not force and self._agent_locked and self._active_agent is not None:
            return
        if self._active_agent and self._active_agent != identity:
            logger.info(
                "Switching lip sync agent: %s -> %s",
                self._active_agent,
                identity,
            )
        else:
            logger.info("Active agent for lip sync: %s", identity)
        self._active_agent = identity
        self._agent_locked = True
        self._audio_buffer.clear()
        self._buffer_start_time = 0.0
        self._chunks_sent = 0
        self._frame_queue.clear()
        self._reset_next_motion = True
        self._subscribed_audio_sids.clear()
        self._logged_audio = False

    def append_audio(self, pcm: np.ndarray, agent_identity: str) -> None:
        if pcm.size == 0 or agent_identity != self._active_agent:
            return
        if not getattr(self, "_logged_audio", False):
            logger.info("Receiving agent audio from %s", agent_identity)
            self._logged_audio = True
        pcm = pcm.astype(np.int16)
        energy = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
        self._last_energy = energy
        now = time.monotonic()
        if self._drive == "reactive" and self._reactive is not None:
            self._reactive.update(energy)
            if energy >= self._animation_threshold:
                self._last_audio_at = now
            return
        if energy >= self._animation_threshold:
            self._last_audio_at = now
        if energy < self._buffer_threshold:
            return
        if not self._audio_buffer:
            self._buffer_start_time = now
            self._chunks_sent = 0
            self._reset_next_motion = True
        self._audio_buffer.append(pcm)
        self._last_speech_at = now
        self._last_audio_at = now

    def pulse_mouth(self, char_count: int) -> None:
        if self._reactive is None:
            return
        self._reactive.arm_reply()
        self._last_audio_at = time.monotonic()
        logger.info("Mouth armed for reply (%d chars) — syncing to live audio", char_count)

    def note_agent_speaking(self) -> None:
        if self._reactive is None:
            return
        self._reactive.note_active_speaker()
        self._last_audio_at = time.monotonic()

    def _prepare_chunk(self) -> tuple[np.ndarray, float, float] | None:
        if not self._audio_buffer or self._drive in ("idle", "reactive"):
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

    def _enqueue_frames(
        self, frames: list[np.ndarray], chunk_start: float, chunk_dur: float
    ) -> None:
        n = len(frames)
        if n == 0:
            return
        now = time.monotonic()
        end = chunk_start + chunk_dur
        lag = now - chunk_start
        kept = 0

        if now >= end:
            if lag > self._max_lip_lag_sec:
                logger.info(
                    "Skipping %d frames — too late for sync (lag=%.0fms, max=%.0fms)",
                    n,
                    lag * 1000,
                    self._max_lip_lag_sec * 1000,
                )
                return
            t0, span = now, chunk_dur
        else:
            t0, span = chunk_start, chunk_dur

        for i, frame in enumerate(frames):
            play_at = t0 + (i / n) * span if n > 1 else t0
            if play_at < now - self._patch_stale_sec:
                continue
            self._frame_queue.append(TimedFrame(frame, play_at))
            kept += 1

        if kept == 0:
            return

        self._frame_queue = deque(
            sorted(self._frame_queue, key=lambda item: item.play_at)
        )
        logger.info(
            "Queued %d/%d driven frames (lag=%.0fms, queue=%d)",
            kept,
            n,
            lag * 1000,
            len(self._frame_queue),
        )

    async def maybe_flush_utterance(self) -> None:
        if self._drive == "idle":
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
        start_idx = self.loop.current_index
        reset = self._reset_next_motion
        self._reset_next_motion = False
        await self._wav_queue.put((wav_path, chunk_start, chunk_dur, start_idx, reset))

    async def _drive_worker(self) -> None:
        if self._drive == "idle":
            return
        loop = asyncio.get_running_loop()
        while True:
            item = await self._wav_queue.get()
            if item is None:
                break
            wav_path, chunk_start, chunk_dur, start_idx, reset_motion = item
            try:
                if self._lp is not None:
                    frames = await loop.run_in_executor(
                        self._drive_executor,
                        self._lp.render_wav,
                        wav_path,
                        start_idx,
                        reset_motion,
                    )
                elif self._wav is not None:
                    patches = await loop.run_in_executor(
                        self._drive_executor, self._wav.sync_utterance, wav_path
                    )
                    frames = []
                    box = self.loop.current_box()
                    if box is not None:
                        for p in patches:
                            base = self.loop.frames[start_idx % len(self.loop.frames)]
                            frames.append(composite_lip_patch(base.copy(), p, box))
                else:
                    frames = []
                if frames:
                    self._enqueue_frames(frames, chunk_start, chunk_dur)
            except RuntimeError as exc:
                if "fix_onnx_cuda" in str(exc).lower():
                    logger.error("Fatal lip-sync setup error: %s", exc)
                    break
                logger.exception("Lip drive failed")
            except Exception:
                logger.exception("Lip drive failed")
            finally:
                Path(wav_path).unlink(missing_ok=True)
                self._wav_queue.task_done()

    def next_display_frame(self) -> np.ndarray:
        now = time.monotonic()
        while (
            self._frame_queue
            and self._frame_queue[0].play_at < now - self._patch_stale_sec
        ):
            self._frame_queue.popleft()

        if self._frame_queue and self._frame_queue[0].play_at <= now:
            return self._frame_queue.popleft().frame

        base = self.loop.next_idle_frame()
        if self._drive == "reactive" and self._reactive is not None:
            self._reactive.update(self._last_energy)
            return self._reactive.apply(base, self.loop.current_box())
        return base

    def _is_speaking(self) -> bool:
        if self._drive == "reactive" and self._reactive is not None:
            return (
                self._reactive.openness > 0.04
                or time.monotonic() - self._last_audio_at < 0.5
            )
        return time.monotonic() - self._last_audio_at < 0.8

    async def _utterance_worker(self) -> None:
        while True:
            try:
                await self.maybe_flush_utterance()
            except Exception:
                logger.exception("Utterance worker error")
            await asyncio.sleep(0.05)

    async def publish_forever(self, room: rtc.Room) -> None:
        utterance_task = asyncio.create_task(self._utterance_worker())
        drive_task = (
            asyncio.create_task(self._drive_worker())
            if self._drive not in ("idle", "reactive")
            else None
        )
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
                    detail = ""
                    if self._drive == "reactive" and self._reactive is not None:
                        detail = (
                            f", energy={self._last_energy:.0f}, "
                            f"open={self._reactive.openness:.2f}, "
                            f"agent={self._active_agent or 'none'}"
                        )
                    logger.info(
                        "Streaming OK — frame %d/%d, speaking=%s%s",
                        self.loop._index,
                        len(self.loop.frames),
                        speaking,
                        detail,
                    )
                await asyncio.sleep(interval)
        finally:
            utterance_task.cancel()
            if drive_task is not None:
                if self._drive not in ("idle", "reactive"):
                    await self._wav_queue.put(None)
                drive_task.cancel()
            self._drive_executor.shutdown(wait=False, cancel_futures=True)


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
        frame = event.frame
        pcm = np.frombuffer(bytes(frame.data), dtype=np.int16)
        if frame.num_channels > 1 and pcm.size:
            pcm = pcm.reshape(-1, frame.num_channels)[:, 0].copy()
        publisher.append_audio(pcm, agent_identity)


async def run_avatar(room_name: str, loop_path: str, fps: int, mode: str) -> None:
    url = os.environ["LIVEKIT_URL"]
    wav2lip_root = Path(os.environ.get("WAV2LIP_ROOT", "/workspace/Wav2Lip"))

    lipsync: Wav2LipEngine | None = None
    liveportrait: LivePortraitEngine | None = None
    face_boxes = load_face_boxes(wav2lip_root)
    mouth_drive = os.environ.get("MOUTH_DRIVE", "idle").lower()

    if mode == "liveportrait":
        liveportrait = LivePortraitEngine(loop_path)
        if liveportrait.is_ready():
            logger.info("LivePortrait engine found at %s", liveportrait.flp_root)
            preflight_liveportrait()
            logger.info("Preloading LivePortrait models (~30s first time)…")
            await asyncio.get_running_loop().run_in_executor(None, liveportrait.warmup)
            logger.info("LivePortrait preload complete")
        else:
            logger.warning(
                "AVATAR_MODE=liveportrait but models missing — falling back to reactive. "
                "Run: bash setup_liveportrait.sh"
            )
            liveportrait = None
            mode = "reactive"
    elif mode == "wav2lip" and mouth_drive == "composite":
        lipsync = Wav2LipEngine(loop_video_path=loop_path)
        if lipsync.is_ready():
            logger.info("Wav2Lip engine ready (legacy composite mode)")
            face_boxes = face_boxes or load_face_boxes(wav2lip_root)

    loop = AudioReactiveLoop(loop_path, face_boxes)

    if mode == "reactive":
        logger.info("Avatar mode: reactive — instant speech-synced mouth on loop")
    elif mode == "mock":
        logger.info("Avatar mode: mock — idle loop (voice + chat)")

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

    publisher = AvatarPublisher(loop, source, fps, lipsync, liveportrait)

    def rescan_agent_audio() -> None:
        if publisher._users_in_room == 0:
            return
        identity = publisher.newest_agent_identity(room)
        if not identity:
            logger.warning("No agent in room — mouth sync waits for agent")
            return
        participant = room.remote_participants.get(identity)
        if participant is None:
            return
        for pub in participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                publisher.bind_agent_audio(pub.track, identity, force=True)
                return
        logger.info("Newest agent %s has no audio track yet — waiting", identity)

    def maybe_subscribe_agent_audio(
        track: rtc.Track,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if not is_agent_participant(participant):
            return
        publisher.note_agent(participant.identity)
        if publisher._users_in_room == 0:
            return
        identity = publisher.newest_agent_identity(room)
        if identity != participant.identity:
            return
        publisher.bind_agent_audio(track, participant.identity, force=True)

    for participant in room.remote_participants.values():
        if participant.identity.startswith("user-"):
            publisher.user_joined()
        elif is_agent_participant(participant):
            publisher.note_agent(participant.identity)

    if publisher._users_in_room > 0:
        rescan_agent_audio()

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logger.info("Participant joined: %s", participant.identity)
        if participant.identity.startswith("user-"):
            publisher.user_joined()
            publisher.unlock_agent()
            rescan_agent_audio()
            return
        if not is_agent_participant(participant):
            return
        publisher.note_agent(participant.identity)
        if publisher._users_in_room > 0:
            rescan_agent_audio()
        else:
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

    @room.on("active_speakers_changed")
    def on_active_speakers(speakers: list[rtc.Participant]) -> None:
        if publisher._active_agent is None:
            return
        for speaker in speakers:
            if speaker.identity == publisher._active_agent:
                publisher.note_agent_speaking()
                return

    @room.on("data_received")
    def on_agent_data(data: rtc.DataPacket) -> None:
        if data.topic != "agent_reply":
            return
        try:
            payload = json.loads(data.data.decode("utf-8"))
            chars = int(payload.get("charCount", 40))
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
            chars = 40
        publisher.pulse_mouth(chars)

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
