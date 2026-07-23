"""MuseTalk 1.5 backend — warm worker (prepare once) → stream frames.

Uses MuseTalk's realtime Avatar path: face latents/masks prepared once, then
each utterance only runs Whisper + UNet + VAE (fp16 default for VRAM/speed).

Requires:
  .\\setup_musetalk.ps1
  models/musetalkV15/unet.pth + musetalk.json (+ sd-vae, whisper, dwpose, …)

Run: .\\run_server.ps1 -Backend musetalk
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from backends.common import SAMPLE_RATE, UtteranceStreamEngine

logger = logging.getLogger("avatar.musetalk")


class MuseTalkWorker:
    def __init__(self, python: Path, worker_py: Path, vendor: Path) -> None:
        self.python = python
        self.worker_py = worker_py
        self.vendor = vendor
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._prepared_source: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return
            env = os.environ.copy()
            env["MUSETALK_ROOT"] = str(self.vendor)
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("MUSETALK_VERSION", "v15")
            # FP16 keeps VRAM/speed usable; quality fixes are crop + .npy handoff.
            env.setdefault("MUSETALK_FP16", "1")
            env.setdefault("MUSETALK_BATCH_SIZE", "8")
            try:
                import imageio_ffmpeg

                ff_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
                env["PATH"] = ff_dir + os.pathsep + env.get("PATH", "")
            except Exception:
                pass
            env["PATH"] = str(self.python.parent) + os.pathsep + env.get("PATH", "")

            logger.info("Starting MuseTalk worker (v1.5, warm)…")
            self._proc = subprocess.Popen(
                [str(self.python), str(self.worker_py)],
                cwd=str(self.vendor),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
            threading.Thread(target=self._drain_stderr, daemon=True, name="musetalk-stderr").start()
            ready = self._read_locked(timeout_s=120.0)
            if not ready or not ready.get("ok"):
                err = (ready or {}).get("error", "no ready message")
                self.stop()
                raise RuntimeError(f"MuseTalk worker failed to start: {err}")
            logger.info(
                "MuseTalk worker ready in %ss (version=%s fp16=%s)",
                ready.get("load_s"), ready.get("version"), ready.get("fp16"),
            )

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
            self._prepared_source = None
        if not proc:
            return
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=5)
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass

    def _drain_stderr(self) -> None:
        proc = self._proc
        if not proc or not proc.stderr:
            return
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                logger.info("musetalk-worker: %s", line[:500])

    def _read_locked(self, timeout_s: float) -> dict | None:
        assert self._proc and self._proc.stdout
        box: dict = {}

        def _read() -> None:
            try:
                assert self._proc and self._proc.stdout
                box["line"] = self._proc.stdout.readline()
            except BaseException as e:  # noqa: BLE001
                box["err"] = e

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=timeout_s)
        if t.is_alive():
            raise TimeoutError(f"MuseTalk worker timed out after {timeout_s:.0f}s")
        if "err" in box:
            raise box["err"]
        line = box.get("line") or ""
        if not line:
            raise RuntimeError("MuseTalk worker exited (no response)")
        return json.loads(line)

    def _request(self, req: dict, timeout_s: float = 600.0) -> dict:
        if not self._proc or self._proc.poll() is not None:
            self.start()
        with self._lock:
            assert self._proc and self._proc.stdin
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            resp = self._read_locked(timeout_s=timeout_s)
        if not resp or not resp.get("ok"):
            err = (resp or {}).get("error", "empty response")
            trace = (resp or {}).get("trace", "")
            raise RuntimeError(f"MuseTalk {req.get('cmd')} failed: {err}\n{trace[-1500:]}")
        return resp

    def prepare(self, source_path: Path, bbox_shift: int = 0) -> None:
        key = f"{source_path}|{bbox_shift}"
        if self._prepared_source == key and self._proc and self._proc.poll() is None:
            return
        self._request({
            "cmd": "prepare",
            "source_path": str(source_path),
            "bbox_shift": bbox_shift,
        }, timeout_s=300.0)
        self._prepared_source = key

    def infer(self, audio_path: Path, output_path: Path) -> Path:
        resp = self._request({
            "cmd": "infer",
            "audio_path": str(audio_path),
            "output_path": str(output_path),
        }, timeout_s=600.0)
        logger.info("MuseTalk warm infer %dms → %s", resp.get("ms"), Path(resp["video_path"]).name)
        return Path(resp["video_path"])


class MuseTalkBackend(UtteranceStreamEngine):
    backend_id = "musetalk"
    backend_name = "MuseTalk"

    def __init__(self, root: Path) -> None:
        super().__init__(chunk_frames=12, fps=25)
        self.root = root
        self.vendor = Path(os.environ.get("MUSETALK_ROOT", root / "vendor" / "MuseTalk"))
        self.python = Path(os.environ.get(
            "MUSETALK_PYTHON",
            root / ".venv_musetalk" / "Scripts" / "python.exe",
        ))
        if not self.python.is_file():
            self.python = Path(sys.executable)

        worker_py = root / "scripts" / "musetalk_worker.py"
        v15 = self.vendor / "models" / "musetalkV15" / "unet.pth"
        if not self.vendor.is_dir() or not worker_py.is_file():
            raise RuntimeError(
                f"MuseTalk incomplete.\n  vendor: {self.vendor}\n  worker: {worker_py}\n"
                "Run: .\\setup_musetalk.ps1"
            )
        if not v15.is_file():
            raise RuntimeError(
                f"MuseTalk 1.5 weights missing: {v15}\nRun: .\\setup_musetalk.ps1"
            )

        self._work = root / "uploads" / "_musetalk_work"
        self._work.mkdir(parents=True, exist_ok=True)
        self._avatar_img: Path | None = None
        self._bbox_shift = int(os.environ.get("MUSETALK_BBOX_SHIFT", "0"))
        self._worker = MuseTalkWorker(self.python, worker_py, self.vendor)

    def on_prepare(self) -> None:
        assert self._source_image
        # MuseTalk blends lips onto the full frame itself — feed full source
        # (not a tight face crop) so identity/background stay correct.
        dest = self._work / "source.png"
        shutil.copy2(self._source_image, dest)
        self._avatar_img = dest
        self._worker.start()
        self._worker.prepare(dest, self._bbox_shift)

    def stop(self) -> None:
        try:
            self._worker.stop()
        finally:
            super().stop()

    def render_utterance(self, audio_f32_16k: np.ndarray) -> list[np.ndarray]:
        assert self._avatar_img is not None
        import soundfile as sf

        stamp = str(int(time.time() * 1000))
        wav = self._work / f"utt_{stamp}.wav"
        out_mp4 = self._work / f"out_{stamp}.mp4"
        sf.write(str(wav), audio_f32_16k, SAMPLE_RATE)

        video_path = self._worker.infer(wav, out_mp4)
        frames: list[np.ndarray] = []
        p = Path(video_path)
        if p.suffix.lower() == ".npy":
            stack = np.load(str(p))
            frames = [stack[i] for i in range(len(stack))]
        elif p.is_dir():
            for fp in sorted(p.glob("*.png")):
                fr = cv2.imread(str(fp))
                if fr is not None:
                    frames.append(fr)
        else:
            cap = cv2.VideoCapture(str(video_path))
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            cap.release()
        if not frames:
            raise RuntimeError("MuseTalk produced no frames")
        # Drop bulky intermediates so uploads/ doesn't fill mid-bench.
        try:
            p.unlink(missing_ok=True)
            wav.unlink(missing_ok=True)
        except Exception:
            pass

        assert self._source_bgr is not None
        sh, sw = self._source_bgr.shape[:2]
        fh, fw = frames[0].shape[:2]
        # MuseTalk outputs full blended frames; resize if needed. Avoid double-composite.
        if (fw, fh) != (sw, sh):
            return [
                cv2.resize(f, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
                for f in frames
            ]
        return frames


def create(root: Path):
    return MuseTalkBackend(root)
