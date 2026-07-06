"""Patch upstream Wav2Lip for Python 3.12 + modern torch/librosa."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"  patched {path.name}")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/Wav2Lip")
    if not root.is_dir():
        raise SystemExit(f"Wav2Lip not found at {root}")

    (root / "temp").mkdir(exist_ok=True)

    patch_file(
        root / "audio.py",
        [
            ("librosa.core.load", "librosa.load"),
            (
                "return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,\n"
                "\tfmin=hp.fmin, fmax=hp.fmax)",
                "return librosa.filters.mel("
                "sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, "
                "fmin=hp.fmin, fmax=hp.fmax)",
            ),
        ],
    )

    inference = root / "inference.py"
    text = inference.read_text(encoding="utf-8")
    if "weights_only=False" not in text:
        text = text.replace(
            "checkpoint = torch.load(checkpoint_path)",
            "checkpoint = torch.load(checkpoint_path, weights_only=False)",
        )
        text = text.replace(
            "checkpoint = torch.load(checkpoint_path,\n\t\t\t\t\t\tmap_location=lambda storage, loc: storage)",
            "checkpoint = torch.load("
            "checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)",
        )
        inference.write_text(text, encoding="utf-8")
        print("  patched inference.py (torch.load)")

    patch_file(
        inference,
        [("cv2.cv2.ROTATE_90_CLOCKWISE", "cv2.ROTATE_90_CLOCKWISE")],
    )

    text = inference.read_text(encoding="utf-8")
    if "--boxes_file" not in text:
        text = text.replace(
            "parser.add_argument('--nosmooth', default=False, action='store_true',\n"
            "\thelp='Prevent smoothing face detections over a short temporal window')",
            "parser.add_argument('--nosmooth', default=False, action='store_true',\n"
            "\thelp='Prevent smoothing face detections over a short temporal window')\n\n"
            "parser.add_argument('--boxes_file', type=str, default=None,\n"
            "\thelp='JSON file with per-frame face boxes [y1,y2,x1,x2]')",
        )
        text = text.replace(
            "\tif args.box[0] == -1:\n"
            "\t\tif not args.static:\n"
            "\t\t\tface_det_results = face_detect(frames) # BGR2RGB for CNN face detection\n"
            "\t\telse:\n"
            "\t\t\tface_det_results = face_detect([frames[0]])\n"
            "\telse:\n"
            "\t\tprint('Using the specified bounding box instead of face detection...')\n"
            "\t\ty1, y2, x1, x2 = args.box\n"
            "\t\tface_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]",
            "\tif args.boxes_file:\n"
            "\t\tprint('Using cached per-frame face boxes...')\n"
            "\t\twith open(args.boxes_file) as jf:\n"
            "\t\t\tcached_boxes = json.load(jf)\n"
            "\t\tface_det_results = []\n"
            "\t\tfor i, f in enumerate(frames):\n"
            "\t\t\ty1, y2, x1, x2 = cached_boxes[i % len(cached_boxes)]\n"
            "\t\t\tface_det_results.append([f[y1:y2, x1:x2], (y1, y2, x1, x2)])\n"
            "\telif args.box[0] == -1:\n"
            "\t\tif not args.static:\n"
            "\t\t\tface_det_results = face_detect(frames) # BGR2RGB for CNN face detection\n"
            "\t\telse:\n"
            "\t\t\tface_det_results = face_detect([frames[0]])\n"
            "\telse:\n"
            "\t\tprint('Using the specified bounding box instead of face detection...')\n"
            "\t\ty1, y2, x1, x2 = args.box\n"
            "\t\tface_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]",
        )
        inference.write_text(text, encoding="utf-8")
        print("  patched inference.py (boxes_file)")

    # librosa 0.10 removed positional mel args in some builds — ensure kwargs form
    audio = root / "audio.py"
    audio_text = audio.read_text(encoding="utf-8")
    if "sr=hp.sample_rate" not in audio_text and "librosa.filters.mel(" in audio_text:
        audio_text = re.sub(
            r"librosa\.filters\.mel\(([^)]+)\)",
            lambda m: m.group(0)
            if "sr=" in m.group(0)
            else m.group(0).replace(
                "librosa.filters.mel(hp.sample_rate, hp.n_fft,",
                "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,",
            ),
            audio_text,
            count=1,
        )
        audio.write_text(audio_text, encoding="utf-8")
        print("  patched audio.py (mel kwargs)")

    print("Wav2Lip compatibility patches applied.")


if __name__ == "__main__":
    main()
