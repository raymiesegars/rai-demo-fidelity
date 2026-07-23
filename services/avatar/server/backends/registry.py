"""Load the active avatar backend from AVATAR_BACKEND."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("avatar.registry")

ROOT = Path(__file__).resolve().parents[2]

# id -> display name
BACKEND_NAMES: dict[str, str] = {
    "flashhead": "SoulX-FlashHead Lite",
    "flashhead-pro": "SoulX-FlashHead Pro",
    "wav2lip": "Wav2Lip",
    "musetalk": "MuseTalk",
    "sadtalker": "SadTalker",
    "sonic": "Sonic",
    "liveportrait": "LivePortrait",
    "livetalk": "LiveTalk",
    "echomimic": "EchoMimicV3-Flash",
    "ditto": "Ditto",
}

# Backends with a real create() module (may still fail if weights missing)
IMPLEMENTED = {
    "flashhead",
    "flashhead-pro",
    "wav2lip",
    "musetalk",
    "ditto",
    "sadtalker",
    "sonic",
    "liveportrait",
    "livetalk",
    "echomimic",
}


def active_backend_id() -> str:
    return (os.environ.get("AVATAR_BACKEND") or "flashhead").strip().lower()


def list_backends() -> list[dict]:
    active = active_backend_id()
    return [
        {
            "id": bid,
            "name": name,
            "active": bid == active,
            "implemented": bid in IMPLEMENTED,
        }
        for bid, name in BACKEND_NAMES.items()
    ]


def create_backend():
    bid = active_backend_id()
    if bid not in BACKEND_NAMES:
        known = ", ".join(BACKEND_NAMES)
        raise ValueError(f"Unknown AVATAR_BACKEND={bid!r}. Known: {known}")

    try:
        if bid == "flashhead":
            from backends.flashhead import create
            return create(ROOT)
        if bid == "flashhead-pro":
            from backends.flashhead import create_pro
            return create_pro(ROOT)
        if bid == "wav2lip":
            from backends.wav2lip import create
            return create(ROOT)
        if bid == "musetalk":
            from backends.musetalk import create
            return create(ROOT)
        if bid == "ditto":
            from backends.ditto import create
            return create(ROOT)
        if bid == "sadtalker":
            from backends.sadtalker import create
            return create(ROOT)
        if bid == "sonic":
            from backends.sonic import create
            return create(ROOT)
        if bid == "liveportrait":
            from backends.liveportrait import create
            return create(ROOT)
        if bid == "livetalk":
            from backends.livetalk import create
            return create(ROOT)
        if bid == "echomimic":
            from backends.echomimic import create
            return create(ROOT)
    except Exception as e:
        logger.exception("Failed to load backend %s — falling back to stub", bid)
        from backends.stub import create as stub_create
        eng = stub_create(ROOT, backend_id=bid, backend_name=BACKEND_NAMES[bid])
        eng._error = (  # type: ignore[attr-defined]
            f"{BACKEND_NAMES[bid]} failed to load: {e}\n"
            f"See docs/MODEL_BENCH.md and setup_{bid}.ps1"
        )
        return eng

    from backends.stub import create
    return create(ROOT, backend_id=bid, backend_name=BACKEND_NAMES[bid])
