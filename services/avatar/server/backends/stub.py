"""Placeholder backend for models not yet wired.

Keeps the comparison UI / swap flow working: selecting an unimplemented
model starts the server in "bench-only" mode so you can still view and
edit scores, but live /stream will report not ready.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path


class StubBackend:
    def __init__(self, backend_id: str, backend_name: str) -> None:
        self.backend_id = backend_id
        self.backend_name = backend_name
        self.prepared = False
        self.busy_ratio = 0.0
        self.chunk_seconds = 0.96
        self.started_at = time.time()
        self.chunks_total = 0
        self.chunks_speaking = 0
        self.gen_ms_history: deque[int] = deque(maxlen=300)
        self.stats_lock = threading.Lock()
        self._source_image: str | None = None
        self._frame_sink = None
        self._error = (
            f"{backend_name} is registered but not implemented yet. "
            f"Set AVATAR_BACKEND=flashhead for the live demo, or implement "
            f"backends/{backend_id}.py. See docs/MODEL_BENCH.md."
        )

    def set_frame_sink(self, sink) -> None:
        self._frame_sink = sink

    def prepare_avatar(self, image_path: str, framing: float = 1.0) -> dict:
        raise RuntimeError(self._error)

    def reframe_avatar(self, framing: float) -> dict:
        raise RuntimeError(self._error)

    def session_info(self) -> dict:
        return {
            "ready": False,
            "backend_id": self.backend_id,
            "backend_name": self.backend_name,
            "error": self._error,
        }

    def start(self) -> None:
        pass

    def push_speech(self, audio_f32_16k) -> None:
        raise RuntimeError(self._error)

    def client_connected(self) -> None:
        pass

    def client_disconnected(self) -> None:
        pass


def create(root: Path, backend_id: str, backend_name: str):
    return StubBackend(backend_id, backend_name)
