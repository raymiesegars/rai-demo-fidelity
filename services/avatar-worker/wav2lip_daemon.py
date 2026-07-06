"""Long-running Wav2Lip worker: loads model + anchor once, infers per wav job on stdin."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def _load_model(checkpoint_path: str, device: str):
    import torch
    from models import Wav2Lip

    model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    return model.to(device).eval()


def _mel_chunks(wav_path: str, fps: float, mel_step_size: int, audio_mod):
    import numpy as np

    wav = audio_mod.load_wav(wav_path, 16000)
    mel = audio_mod.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError("Mel contains nan in audio")
    chunks = []
    mult = 80.0 / fps
    i = 0
    while True:
        start = int(i * mult)
        if start + mel_step_size > mel.shape[1]:
            chunks.append(mel[:, mel.shape[1] - mel_step_size :])
            break
        chunks.append(mel[:, start : start + mel_step_size])
        i += 1
    return chunks


def _run_batch(model, device, faces, mels):
    import numpy as np
    import torch

    img = np.asarray(faces, dtype=np.float32)
    mel = np.asarray(mels, dtype=np.float32)
    masked = img.copy()
    masked[:, 48:, :] = 0
    img_in = np.concatenate((masked, img), axis=3) / 255.0
    mel_in = mel.reshape(len(mel), mel.shape[1], mel.shape[2], 1)
    img_t = torch.FloatTensor(np.transpose(img_in, (0, 3, 1, 2))).to(device)
    mel_t = torch.FloatTensor(np.transpose(mel_in, (0, 3, 1, 2))).to(device)
    with torch.no_grad():
        pred = model(mel_t, img_t)
    return pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0


def main() -> None:
    cfg = json.loads(sys.argv[1])
    root = Path(cfg["root"])
    os.chdir(root)
    sys.path.insert(0, str(root))

    import cv2
    import torch

    import audio  # noqa: Wav2Lip local module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fps = float(cfg["fps"])
    batch_size = int(cfg.get("batch_size", 128))
    y1, y2, x1, x2 = cfg["box"]
    mel_step_size = 16

    model = _load_model(cfg["checkpoint"], device)
    full_frame = cv2.imread(cfg["anchor"])
    if full_frame is None:
        sys.stderr.write(f"failed to read anchor {cfg['anchor']}\n")
        sys.exit(1)
    face_96 = cv2.resize(full_frame[y1:y2, x1:x2], (96, 96))
    coords = (y1, y2, x1, x2)
    frame_h, frame_w = full_frame.shape[:2]

    sys.stderr.write("ready\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line or line == "QUIT":
            break
        job = json.loads(line)
        wav_path = job["wav"]
        out_path = job["out"]
        t0 = time.monotonic()
        try:
            mels = _mel_chunks(wav_path, fps, mel_step_size, audio)
            avi_path = str(root / "temp" / "daemon_result.avi")
            writer = cv2.VideoWriter(
                avi_path,
                cv2.VideoWriter_fourcc(*"DIVX"),
                fps,
                (frame_w, frame_h),
            )

            faces, mel_batch, frames, coords_batch = [], [], [], []
            for m in mels:
                faces.append(face_96.copy())
                mel_batch.append(m)
                frames.append(full_frame.copy())
                coords_batch.append(coords)
                if len(faces) >= batch_size:
                    preds = _run_batch(model, device, faces, mel_batch)
                    for p, f, c in zip(preds, frames, coords_batch):
                        cy1, cy2, cx1, cx2 = c
                        mouth = cv2.resize(p.astype(np.uint8), (cx2 - cx1, cy2 - cy1))
                        f[cy1:cy2, cx1:cx2] = mouth
                        writer.write(f)
                    faces, mel_batch, frames, coords_batch = [], [], [], []

            if faces:
                preds = _run_batch(model, device, faces, mel_batch)
                for p, f, c in zip(preds, frames, coords_batch):
                    cy1, cy2, cx1, cx2 = c
                    mouth = cv2.resize(p.astype(np.uint8), (cx2 - cx1, cy2 - cy1))
                    f[cy1:cy2, cx1:cx2] = mouth
                    writer.write(f)
            writer.release()

            cmd = (
                f'ffmpeg -y -i "{avi_path}" -i "{wav_path}" '
                f'-strict -2 -q:v 1 "{out_path}"'
            )
            subprocess.call(cmd, shell=platform.system() != "Windows")

            print(
                json.dumps(
                    {
                        "ok": True,
                        "out": out_path,
                        "frames": len(mels),
                        "sec": round(time.monotonic() - t0, 3),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
