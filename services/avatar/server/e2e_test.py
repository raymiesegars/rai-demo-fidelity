"""End-to-end verification (uses /say to bypass LLM quota issues):
1. idle motion — consecutive idle frames must differ visibly
2. speech latency — /say accepted -> first speaking chunk streamed
3. audio content — speaking chunk PCM must be non-silent
"""

import asyncio
import io
import json
import time

import httpx
import numpy as np
import websockets
from PIL import Image


def jdiff(a: bytes, b: bytes) -> float:
    ia = np.asarray(Image.open(io.BytesIO(a)).convert("L"), dtype=np.float32)
    ib = np.asarray(Image.open(io.BytesIO(b)).convert("L"), dtype=np.float32)
    return float(np.abs(ia - ib).mean())


async def main() -> None:
    idle_frames: list[bytes] = []
    speak_audio_rms = 0.0
    t_first_speak_frame = None

    async with websockets.connect("ws://127.0.0.1:8100/stream", max_size=None) as stream:
        # ---- phase 1: idle motion (~4 s) ----
        cur_speaking = False
        end = time.time() + 4
        while time.time() < end:
            data = await asyncio.wait_for(stream.recv(), timeout=15)
            if isinstance(data, str):
                cur_speaking = json.loads(data).get("speaking", False)
            elif data[0] == 1 and not cur_speaking and len(idle_frames) < 60:
                idle_frames.append(bytes(data[1:]))

        diffs = [jdiff(idle_frames[i], idle_frames[i + 1]) for i in range(0, len(idle_frames) - 1, 3)]
        span = jdiff(idle_frames[0], idle_frames[-1])

        # ---- phase 2: /say -> latency to first speaking frame ----
        async with httpx.AsyncClient() as http:
            t0 = time.time()
            r = await http.post("http://127.0.0.1:8100/say",
                                json={"text": "Hi there. I have mild lower back pain, about two weeks now."},
                                timeout=30)
            say = r.json()

            speaking_meta = False
            while t_first_speak_frame is None and time.time() - t0 < 20:
                data = await asyncio.wait_for(stream.recv(), timeout=20)
                if isinstance(data, str):
                    speaking_meta = json.loads(data).get("speaking", False)
                elif data[0] == 2 and speaking_meta:
                    pcm = np.frombuffer(bytes(data[1:]), dtype=np.int16).astype(np.float32)
                    speak_audio_rms = max(speak_audio_rms, float(np.sqrt((pcm ** 2).mean())))
                elif data[0] == 1 and speaking_meta:
                    t_first_speak_frame = time.time() - t0

    print(f"idle frames: {len(idle_frames)}, adjacent diff {np.mean(diffs):.2f}, 4s span {span:.2f}")
    print(f"/say tts_ms={say.get('tts_ms')} audio_s={say.get('audio_s')}")
    print(f"latency: say -> first speaking frame streamed = {t_first_speak_frame:.2f}s" if t_first_speak_frame else "NO SPEAKING FRAME")
    print(f"speaking audio RMS: {speak_audio_rms:.0f} (>200 = real voice)")
    ok = (
        len(idle_frames) > 20 and np.mean(diffs) > 0.5
        and t_first_speak_frame is not None
        and (t_first_speak_frame - say.get("tts_ms", 0) / 1000) < 1.6
        and speak_audio_rms > 200
    )
    print("PASS" if ok else "FAIL")


asyncio.run(main())
