"""Avatar backend protocol — one adapter per talking-head model.

Swap with AVATAR_BACKEND / run_server.ps1 -Backend <id>.
Only one backend loads at a time. Comparison data: bench/results/*.json.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class AvatarBackend(Protocol):
    """Minimal surface the FastAPI app needs from any model."""

    prepared: bool
    busy_ratio: float
    chunk_seconds: float
    started_at: float
    chunks_total: int
    chunks_speaking: int
    gen_ms_history: Any  # deque[int]
    stats_lock: Any

    def set_frame_sink(self, sink: Callable[[dict], None] | None) -> None: ...
    def prepare_avatar(self, image_path: str, framing: float = 1.0) -> dict: ...
    def reframe_avatar(self, framing: float) -> dict: ...
    def session_info(self) -> dict: ...
    def start(self) -> None: ...
    def push_speech(self, audio_f32_16k) -> None: ...
    def client_connected(self) -> None: ...
    def client_disconnected(self) -> None: ...

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_name(self) -> str: ...
