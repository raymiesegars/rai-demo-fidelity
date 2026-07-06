"""Ping-pong patient loop video publisher for LiveKit."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv()

logger = logging.getLogger("avatar-worker")
logging.basicConfig(level=logging.INFO)


class PingPongLoop:
    """Load video frames and iterate forward then backward."""

    def __init__(self, path: str) -> None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open patient loop: {path}")

        self.frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.frames.append(rgb)

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
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    token = api.AccessToken(api_key, api_secret)
    token.with_identity(identity).with_name("Alan").with_grants(
        api.VideoGrants(
            room_join=True,
            room=room,
            can_publish=True,
            can_subscribe=True,
        )
    )
    return token.to_jwt()


async def publish_loop(room_name: str, loop_path: str, fps: int = 25) -> None:
    url = os.environ["LIVEKIT_URL"]
    loop = PingPongLoop(loop_path)

    room = rtc.Room()
    token = make_token(room_name)

    @room.on("disconnected")
    def _on_disconnect() -> None:
        logger.info("Disconnected from room")

    await room.connect(url, token)
    logger.info("Avatar joined room %s", room_name)

    source = rtc.VideoSource(loop.width, loop.height)
    track = rtc.LocalVideoTrack.create_video_track("patient-video", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await room.local_participant.publish_track(track, options)
    logger.info("Publishing patient video track")

    interval = 1.0 / fps
    while room.isconnected():
        rgba = loop.next_frame()
        frame = rtc.VideoFrame(
            loop.width,
            loop.height,
            rtc.VideoBufferType.RGBA,
            rgba.tobytes(),
        )
        source.capture_frame(frame)
        await asyncio.sleep(interval)


async def main() -> None:
    room = os.environ.get("LIVEKIT_ROOM", "patient-demo")
    loop_path = os.environ.get("PATIENT_LOOP_PATH", "assets/alan-loop.mp4")
    fps = int(os.environ.get("TARGET_FPS", "25"))
    mode = os.environ.get("AVATAR_MODE", "mock")

    resolved = Path(loop_path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    if mode == "gpu":
        logger.warning(
            "AVATAR_MODE=gpu requires FasterLivePortrait + JoyVASA. "
            "Falling back to loop publisher until GPU pipeline is installed. "
            "See services/avatar-worker/README.md"
        )

    await publish_loop(room, str(resolved), fps)


if __name__ == "__main__":
    asyncio.run(main())
