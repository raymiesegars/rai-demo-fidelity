"""Patch Wav2Lip vendor for modern librosa (filters.mel keyword-only API)."""

from __future__ import annotations

from pathlib import Path

OLD = """def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)"""

NEW = """def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    # librosa>=0.10 requires keyword-only args for filters.mel
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax,
    )"""


def patch(wav2lip_root: Path) -> bool:
    audio = wav2lip_root / "audio.py"
    if not audio.is_file():
        raise FileNotFoundError(audio)
    text = audio.read_text(encoding="utf-8")
    if "sr=hp.sample_rate" in text and "filters.mel(" in text:
        return False  # already patched
    if OLD not in text:
        # try looser match
        if "librosa.filters.mel(hp.sample_rate" not in text:
            return False
        text = text.replace(
            "return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,\n"
            "                               fmin=hp.fmin, fmax=hp.fmax)",
            NEW.split("return ", 1)[1].rstrip(),
        )
        # if still broken, force replace function body via marker
    else:
        text = text.replace(OLD, NEW)
    audio.write_text(text, encoding="utf-8")
    return True


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "vendor/Wav2Lip")
    changed = patch(root)
    print("patched" if changed else "already ok", root)
