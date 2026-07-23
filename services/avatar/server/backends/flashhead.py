"""SoulX-FlashHead backend adapters (Lite + Pro)."""

from __future__ import annotations

from pathlib import Path


def _create(root: Path, *, model_type: str, backend_id: str, backend_name: str):
    from engine import AvatarEngine

    ckpt = root / "models" / "SoulX-FlashHead-1_3B"
    if model_type == "pro":
        need = [
            ckpt / "Model_Pro",
            ckpt / "VAE_Wan" / "Wan2.1_VAE.pth",
        ]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "FlashHead Pro weights missing:\n  "
                + "\n  ".join(missing)
                + "\nRun: .\\setup_flashhead_pro.ps1"
            )

    eng = AvatarEngine(
        ckpt_dir=str(ckpt),
        wav2vec_dir=str(root / "models" / "wav2vec2-base-960h"),
        model_type=model_type,
    )
    eng.backend_id = backend_id  # type: ignore[attr-defined]
    eng.backend_name = backend_name  # type: ignore[attr-defined]
    return eng


def create(root: Path):
    return _create(
        root,
        model_type="lite",
        backend_id="flashhead",
        backend_name="SoulX-FlashHead Lite",
    )


def create_pro(root: Path):
    return _create(
        root,
        model_type="pro",
        backend_id="flashhead-pro",
        backend_name="SoulX-FlashHead Pro",
    )
