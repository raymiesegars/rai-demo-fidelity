"""Download all MuseTalk model weights."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "vendor" / "MuseTalk" / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("==> MuseTalk weights...")
    snapshot_download("TMElyralab/MuseTalk", local_dir=str(MODELS))

    print("==> SD VAE...")
    for f in ("config.json", "diffusion_pytorch_model.bin"):
        hf_hub_download("stabilityai/sd-vae-ft-mse", f, local_dir=str(MODELS / "sd-vae"))

    print("==> Whisper...")
    for f in ("config.json", "pytorch_model.bin", "preprocessor_config.json"):
        hf_hub_download("openai/whisper-tiny", f, local_dir=str(MODELS / "whisper"))

    print("==> DWPose...")
    hf_hub_download("yzd-v/DWPose", "dw-ll_ucoco_384.pth", local_dir=str(MODELS / "dwpose"))

    print("==> SyncNet...")
    hf_hub_download(
        "ByteDance/LatentSync", "latentsync_syncnet.pt", local_dir=str(MODELS / "syncnet")
    )

    print("==> face-parse-bisent...")
    for f in ("79999_iter.pth", "resnet18-5c106cde.pth"):
        hf_hub_download(
            "ManyOtherFunctions/face-parse-bisent", f, local_dir=str(MODELS / "face-parse-bisent")
        )

    print("\nDone. Weights in:", MODELS)


if __name__ == "__main__":
    main()
