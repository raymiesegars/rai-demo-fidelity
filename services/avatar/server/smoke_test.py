"""Load FlashHead, prepare example avatar, generate 3 chunks, report timing."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from engine import AvatarEngine, SAMPLE_RATE

ROOT = Path(__file__).resolve().parents[1]

eng = AvatarEngine(
    ckpt_dir=str(ROOT / "models" / "SoulX-FlashHead-1_3B"),
    wav2vec_dir=str(ROOT / "models" / "wav2vec2-base-960h"),
)
eng.prepare_avatar(str(ROOT / "vendor" / "SoulX-FlashHead" / "examples" / "girl.png"), framing=0.33)

import librosa

audio, _ = librosa.load(
    str(ROOT / "vendor" / "SoulX-FlashHead" / "examples" / "podcast_sichuan_16k.wav"),
    sr=SAMPLE_RATE, mono=True,
)
eng.push_speech(audio[: SAMPLE_RATE * 3])

chunks = []
eng.set_frame_sink(lambda c: chunks.append(c))
eng.client_connected()
eng.start()

t0 = time.time()
while len(chunks) < 3 and time.time() - t0 < 300:
    time.sleep(0.2)

eng.stop()
for c in chunks:
    print(f"chunk seq={c['seq']} frames={len(c['jpegs'])} speaking={c['speaking']} gen_ms={c['gen_ms']}")

if chunks:
    out = ROOT / "uploads" / "smoke_frame.jpg"
    out.write_bytes(chunks[-1]["jpegs"][-1])
    print("wrote", out)
    print("PASS" if len(chunks) >= 3 else "PARTIAL")
else:
    print("FAIL: no chunks generated")
