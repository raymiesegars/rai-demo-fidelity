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


def patch_boxes_file(inference: Path) -> None:
    text = inference.read_text(encoding="utf-8")
    if "--boxes_file" in text and "args.boxes_file" in text:
        print("  inference.py already has boxes_file support")
        return

    # Add CLI argument (insert before parse_args — works with tabs or spaces)
    if "--boxes_file" not in text:
        text, n = re.subn(
            r"^(\s*)args = parser\.parse_args\(\)",
            r"\1parser.add_argument('--boxes_file', type=str, default=None,\n"
            r"\1    help='JSON per-frame face boxes [y1,y2,x1,x2]')\n"
            r"\1args = parser.parse_args()",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if n == 0:
            raise SystemExit("Could not patch inference.py: parse_args() not found")

    # Add datagen branch (match any indentation)
    if "args.boxes_file" not in text:
        datagen_patch = (
            "if args.boxes_file:\n"
            "\t\tprint('Using cached per-frame face boxes...')\n"
            "\t\twith open(args.boxes_file) as jf:\n"
            "\t\t\tcached_boxes = json.load(jf)\n"
            "\t\tface_det_results = []\n"
            "\t\tfor i, f in enumerate(frames):\n"
            "\t\t\ty1, y2, x1, x2 = cached_boxes[i % len(cached_boxes)]\n"
            "\t\t\tface_det_results.append([f[y1:y2, x1:x2], (y1, y2, x1, x2)])\n"
            "\tel"
        )
        # Normalize to file's indentation style
        m = re.search(r"^(\s*)if args\.box\[0\] == -1:", text, re.MULTILINE)
        if not m:
            raise SystemExit("Could not patch inference.py: datagen box check not found")
        indent = m.group(1)
        block = datagen_patch.replace("\t\t", indent + indent).replace("\t", indent).replace(
            "el", "elif"
        )
        # Fix the elif line
        block_lines = [
            f"{indent}if args.boxes_file:",
            f"{indent}{indent}print('Using cached per-frame face boxes...')",
            f"{indent}{indent}with open(args.boxes_file) as jf:",
            f"{indent}{indent}{indent}cached_boxes = json.load(jf)",
            f"{indent}{indent}face_det_results = []",
            f"{indent}{indent}for i, f in enumerate(frames):",
            f"{indent}{indent}{indent}y1, y2, x1, x2 = cached_boxes[i % len(cached_boxes)]",
            f"{indent}{indent}{indent}face_det_results.append([f[y1:y2, x1:x2], (y1, y2, x1, x2)])",
            f"{indent}elif args.box[0] == -1:",
        ]
        text, n = re.subn(
            r"^\s*if args\.box\[0\] == -1:",
            "\n".join(block_lines),
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if n == 0:
            raise SystemExit("Could not patch inference.py datagen")

    inference.write_text(text, encoding="utf-8")
    print("  patched inference.py (boxes_file)")


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

    patch_boxes_file(inference)

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

    # Verify
    final = inference.read_text(encoding="utf-8")
    if "--boxes_file" not in final or "args.boxes_file" not in final:
        raise SystemExit("boxes_file patch verification failed")

    print("Wav2Lip compatibility patches applied.")


if __name__ == "__main__":
    main()
