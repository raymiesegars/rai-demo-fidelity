"""Download LivePortrait / JoyVASA / HuBERT checkpoints into FLP checkpoints/."""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ckpt = Path(sys.argv[1]).resolve()
ckpt.mkdir(parents=True, exist_ok=True)

print("Downloading LivePortrait ONNX (~2GB)...")
snapshot_download(repo_id="warmshao/FasterLivePortrait", local_dir=str(ckpt))

print("Downloading JoyVASA...")
snapshot_download(repo_id="jdh-algo/JoyVASA", local_dir=str(ckpt / "JoyVASA"))

print("Downloading chinese-hubert-base...")
snapshot_download(
    repo_id="TencentGameMate/chinese-hubert-base",
    local_dir=str(ckpt / "chinese-hubert-base"),
)

motion_dir = ckpt / "JoyVASA" / "motion_generator"
expected = motion_dir / "motion_generator_hubert_chinese.pt"
if not expected.is_file():
    candidates = [
        "motion_generator_hubert_chinese.pt",
        "iter_0020000.pt",
        "motion_generator.pt",
    ]
    found = None
    for name in candidates:
        p = motion_dir / name
        if p.is_file():
            found = p
            break
    if found is None:
        pts = list(motion_dir.glob("*.pt")) if motion_dir.is_dir() else []
        found = pts[0] if pts else None
    if found is not None and found != expected:
        shutil.copy2(found, expected)
        print(f"Mapped JoyVASA weight: {found.name} -> {expected.name}")

need = [
    ckpt / "liveportrait_onnx" / "warping_spade.onnx",
    expected,
    ckpt / "JoyVASA" / "motion_template" / "motion_template.pkl",
    ckpt / "chinese-hubert-base" / "config.json",
]
missing = [str(p) for p in need if not p.is_file()]
for p in need:
    print(("OK  " if p.is_file() else "MISSING  ") + str(p))
if missing:
    raise SystemExit(f"Setup incomplete â€” missing {len(missing)} file(s)")
print("All required checkpoint files present.")
