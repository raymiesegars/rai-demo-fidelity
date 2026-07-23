"""SadTalker backend — warm worker (prepare once) → stream full-image frames.

Offline / clip-based portrait animation. Defaults:
  still + preprocess=full  → paste talking face back onto the full still
  size=256, enhancer off   → usable latency for local bench

Requires:
  .\\setup_sadtalker.ps1

Run: .\\run_server.ps1 -Backend sadtalker
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

logger = logging.getLogger("avatar.sadtalker")


class SadTalkerWorker:
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
            env["SADTALKER_ROOT"] = str(self.vendor)
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("SADTALKER_SIZE", "256")
            env.setdefault("SADTALKER_PREPROCESS", "full")
            env.setdefault("SADTALKER_STILL", "1")
            env.setdefault("SADTALKER_BATCH_SIZE", "2")
            try:
                import imageio_ffmpeg

                ff_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
                env["PATH"] = ff_dir + os.pathsep + env.get("PATH", "")
            except Exception:
                pass
            env["PATH"] = str(self.python.parent) + os.pathsep + env.get("PATH", "")

            logger.info("Starting SadTalker worker (warm)…")
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
            threading.Thread(target=self._drain_stderr, daemon=True, name="sadtalker-stderr").start()
            ready = self._read_locked(timeout_s=300.0)
            if not ready or not ready.get("ok"):
                err = (ready or {}).get("error", "no ready message")
                self.stop()
                raise RuntimeError(f"SadTalker worker failed to start: {err}")
            logger.info(
                "SadTalker worker ready in %ss (size=%s preprocess=%s still=%s)",
                ready.get("load_s"),
                ready.get("size"),
                ready.get("preprocess"),
                ready.get("still"),
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
                logger.info("sadtalker-worker: %s", line[:500])

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
            raise TimeoutError(f"SadTalker worker timed out after {timeout_s:.0f}s")
        if "err" in box:
            raise box["err"]
        line = box.get("line") or ""
        if not line:
            raise RuntimeError("SadTalker worker exited (no response)")
        return json.loads(line)

    def _request(self, req: dict, timeout_s: float = 900.0) -> dict:
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
            raise RuntimeError(f"SadTalker {req.get('cmd')} failed: {err}\n{trace[-1500:]}")
        return resp

    def prepare(self, source_path: Path) -> None:
        key = str(source_path)
        if self._prepared_source == key and self._proc and self._proc.poll() is None:
            return
        self._request({"cmd": "prepare", "source_path": str(source_path)}, timeout_s=300.0)
        self._prepared_source = key

    def infer(self, audio_path: Path, output_path: Path) -> Path:
        resp = self._request({
            "cmd": "infer",
            "audio_path": str(audio_path),
            "output_path": str(output_path),
        }, timeout_s=900.0)
        logger.info("SadTalker warm infer %dms → %s", resp.get("ms"), Path(resp["video_path"]).name)
        return Path(resp["video_path"])


class SadTalkerBackend(UtteranceStreamEngine):
    backend_id = "sadtalker"
    backend_name = "SadTalker"

    def __init__(self, root: Path) -> None:
        super().__init__(chunk_frames=12, fps=25)
        self.root = root
        self.vendor = Path(os.environ.get("SADTALKER_ROOT", root / "vendor" / "SadTalker"))
        self.python = Path(os.environ.get(
            "SADTALKER_PYTHON",
            root / ".venv_sadtalker" / "Scripts" / "python.exe",
        ))
        if not self.python.is_file():
            self.python = Path(sys.executable)

        worker_py = root / "scripts" / "sadtalker_worker.py"
        ckpt = self.vendor / "checkpoints" / "SadTalker_V0.0.2_256.safetensors"
        if not self.vendor.is_dir() or not worker_py.is_file():
            raise RuntimeError(
                f"SadTalker incomplete.\n  vendor: {self.vendor}\n  worker: {worker_py}\n"
                "Run: .\\setup_sadtalker.ps1"
            )
        if not ckpt.is_file():
            raise RuntimeError(
                f"SadTalker checkpoint missing: {ckpt}\nRun: .\\setup_sadtalker.ps1"
            )

        self._work = root / "uploads" / "_sadtalker_work"
        self._work.mkdir(parents=True, exist_ok=True)
        self._avatar_img: Path | None = None
        self._worker = SadTalkerWorker(self.python, worker_py, self.vendor)

    def on_prepare(self) -> None:
        assert self._source_image
        dest = self._work / "source.png"
        shutil.copy2(self._source_image, dest)
        self._avatar_img = dest
        self._worker.start()
        self._worker.prepare(dest)

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
        cap = cv2.VideoCapture(str(video_path))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        try:
            video_path.unlink(missing_ok=True)
            wav.unlink(missing_ok=True)
        except Exception:
            pass
        if not frames:
            raise RuntimeError("SadTalker produced no frames")

        assert self._source_bgr is not None
        sh, sw = self._source_bgr.shape[:2]
        fh, fw = frames[0].shape[:2]
        # full preprocess pastes onto original; crop mode returns face-only → composite
        if "full" in os.environ.get("SADTALKER_PREPROCESS", "full").lower():
            if (fw, fh) != (sw, sh):
                return [
                    cv2.resize(f, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
                    for f in frames
                ]
            return frames

        if self._server_composite and self._overlay and (fw < sw * 0.95 or fh < sh * 0.95):
            return [self.composite_model_face(f) for f in frames]
        if (fw, fh) != (sw, sh):
            return [
                cv2.resize(f, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
                for f in frames
            ]
        return frames


def create(root: Path):
    return SadTalkerBackend(root)
