"""Ditto talking-head backend (warm persistent worker → stream).

Keeps StreamSDK loaded in a long-lived subprocess so we don't pay ~8–15s of
weight reload on every utterance. Uses 10 diffusion steps (paper realtime path).

Requires:
  services/avatar/vendor/ditto-talkinghead/
  services/avatar/models/ditto/  (HF checkpoints)

Setup: .\\setup_ditto.ps1
Run:   .\\run_server.ps1 -Backend ditto
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from backends.common import SAMPLE_RATE, UtteranceStreamEngine

logger = logging.getLogger("avatar.ditto")


class DittoWorker:
    """JSONL client for scripts/ditto_worker.py."""

    def __init__(
        self,
        python: Path,
        worker_py: Path,
        vendor: Path,
        ckpt: Path,
        cfg: Path,
        steps: int = 10,
    ) -> None:
        self.python = python
        self.worker_py = worker_py
        self.vendor = vendor
        self.ckpt = ckpt
        self.cfg = cfg
        self.steps = steps
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return
            env = os.environ.copy()
            env["DITTO_DATA_ROOT"] = str(self.ckpt)
            env["DITTO_CFG"] = str(self.cfg)
            env["DITTO_SAMPLING_TIMESTEPS"] = str(self.steps)
            env["PYTHONUNBUFFERED"] = "1"
            try:
                import imageio_ffmpeg

                ff_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
                env["PATH"] = ff_dir + os.pathsep + env.get("PATH", "")
            except Exception:
                pass
            env["PATH"] = str(self.python.parent) + os.pathsep + env.get("PATH", "")

            logger.info("Starting Ditto worker (load once, reuse)…")
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
            # Drain stderr in background so the pipe never fills.
            threading.Thread(target=self._drain_stderr, daemon=True, name="ditto-stderr").start()
            ready = self._readline(timeout_s=180.0)
            if not ready or not ready.get("ok"):
                err = (ready or {}).get("error", "no ready message")
                self.stop()
                raise RuntimeError(f"Ditto worker failed to start: {err}")
            logger.info(
                "Ditto worker ready in %ss (steps=%s)",
                ready.get("load_s"), ready.get("sampling_timesteps"),
            )

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
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
                logger.debug("ditto-worker: %s", line[:500])

    def _readline(self, timeout_s: float = 600.0) -> dict | None:
        assert self._proc and self._proc.stdout
        deadline = time.time() + timeout_s
        # Blocking readline; worker should always answer. Timeout via thread.
        result: dict | None = None
        error: list[BaseException] = []

        def _read() -> None:
            nonlocal result
            try:
                assert self._proc and self._proc.stdout
                line = self._proc.stdout.readline()
                if not line:
                    return
                result = json.loads(line)
            except BaseException as e:  # noqa: BLE001
                error.append(e)

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=max(0.1, deadline - time.time()))
        if t.is_alive():
            raise TimeoutError(f"Ditto worker timed out after {timeout_s:.0f}s")
        if error:
            raise error[0]
        return result

    def infer(self, audio_path: Path, source_path: Path, output_path: Path) -> Path:
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                # Restart outside lock would deadlock — release by calling start carefully
                pass
        if not self._proc or self._proc.poll() is not None:
            self.start()

        with self._lock:
            assert self._proc and self._proc.stdin and self._proc.stdout
            req = {
                "cmd": "infer",
                "audio_path": str(audio_path),
                "source_path": str(source_path),
                "output_path": str(output_path),
                "sampling_timesteps": self.steps,
            }
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            # Read while holding lock — one infer at a time
            resp = self._read_locked(timeout_s=600.0)

        if not resp or not resp.get("ok"):
            err = (resp or {}).get("error", "empty response")
            trace = (resp or {}).get("trace", "")
            raise RuntimeError(f"Ditto infer failed: {err}\n{trace[-1500:]}")
        video = Path(resp["video_path"])
        logger.info(
            "Ditto warm infer %dms (steps=%s) → %s",
            resp.get("ms"), resp.get("sampling_timesteps"), video.name,
        )
        return video

    def _read_locked(self, timeout_s: float) -> dict | None:
        """Read one JSON line; must be called with self._lock held."""
        assert self._proc and self._proc.stdout
        # Can't easily timeout a blocking readline while holding lock without
        # another thread — use a short-poll pattern via concurrent read thread
        # that doesn't need the lock for the pipe itself.
        box: dict = {}

        def _read() -> None:
            try:
                assert self._proc and self._proc.stdout
                line = self._proc.stdout.readline()
                box["line"] = line
            except BaseException as e:  # noqa: BLE001
                box["err"] = e

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=timeout_s)
        if t.is_alive():
            raise TimeoutError(f"Ditto worker timed out after {timeout_s:.0f}s")
        if "err" in box:
            raise box["err"]
        line = box.get("line") or ""
        if not line:
            raise RuntimeError("Ditto worker exited (no response)")
        return json.loads(line)


class DittoBackend(UtteranceStreamEngine):
    backend_id = "ditto"
    backend_name = "Ditto"

    def __init__(self, root: Path) -> None:
        super().__init__(chunk_frames=12, fps=25)
        self.root = root
        self.vendor = Path(os.environ.get("DITTO_ROOT", root / "vendor" / "ditto-talkinghead"))
        self.ckpt = Path(os.environ.get("DITTO_DATA_ROOT", root / "models" / "ditto" / "ditto_pytorch"))
        self.cfg = Path(os.environ.get(
            "DITTO_CFG",
            root / "models" / "ditto" / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl",
        ))
        # Prefer TRT if present
        trt = root / "models" / "ditto" / "ditto_trt_Ampere_Plus"
        cfg_trt = root / "models" / "ditto" / "ditto_cfg" / "v0.4_hubert_cfg_trt.pkl"
        if trt.is_dir() and cfg_trt.is_file():
            self.ckpt = Path(os.environ.get("DITTO_DATA_ROOT", trt))
            self.cfg = Path(os.environ.get("DITTO_CFG", cfg_trt))

        self.python = Path(os.environ.get(
            "DITTO_PYTHON",
            root / ".venv_ditto" / "Scripts" / "python.exe",
        ))
        if not self.python.is_file():
            self.python = Path(sys.executable)

        infer = self.vendor / "inference.py"
        worker_py = root / "scripts" / "ditto_worker.py"
        if not infer.is_file() or not self.cfg.is_file() or not self.ckpt.is_dir():
            raise RuntimeError(
                "Ditto not installed.\n"
                f"  vendor: {self.vendor} (inference.py={'yes' if infer.is_file() else 'NO'})\n"
                f"  data:   {self.ckpt}\n"
                f"  cfg:    {self.cfg}\n"
                "Run: .\\setup_ditto.ps1"
            )
        if not worker_py.is_file():
            raise RuntimeError(f"Missing Ditto worker script: {worker_py}")

        self.steps = int(os.environ.get("DITTO_SAMPLING_TIMESTEPS", "10"))
        self._work = root / "uploads" / "_ditto_work"
        self._work.mkdir(parents=True, exist_ok=True)
        self._source_path: Path | None = None
        self._worker = DittoWorker(
            python=self.python,
            worker_py=worker_py,
            vendor=self.vendor,
            ckpt=self.ckpt,
            cfg=self.cfg,
            steps=self.steps,
        )

    def on_prepare(self) -> None:
        assert self._source_image
        dest = self._work / "source.png"
        import shutil
        shutil.copy2(self._source_image, dest)
        self._source_path = dest
        # Warm the worker during avatar prepare so first chat isn't +8s load.
        self._worker.start()

    def stop(self) -> None:
        try:
            self._worker.stop()
        finally:
            super().stop()

    def render_utterance(self, audio_f32_16k: np.ndarray) -> list[np.ndarray]:
        assert self._source_path is not None
        import soundfile as sf

        stamp = str(int(time.time() * 1000))
        wav = self._work / f"utt_{stamp}.wav"
        out_mp4 = self._work / f"out_{stamp}.mp4"
        sf.write(str(wav), audio_f32_16k, SAMPLE_RATE)

        video_path = self._worker.infer(wav, self._source_path, out_mp4)

        frames = []
        cap = cv2.VideoCapture(str(video_path))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            raise RuntimeError("Ditto MP4 had no frames")

        assert self._source_bgr is not None
        sh, sw = self._source_bgr.shape[:2]
        fh, fw = frames[0].shape[:2]
        if self._server_composite and self._overlay and (fw < sw * 0.95 or fh < sh * 0.95):
            return [self.composite_model_face(f) for f in frames]
        if (fw, fh) != (sw, sh):
            return [cv2.resize(f, (sw, sh)) for f in frames]
        return frames


def create(root: Path):
    return DittoBackend(root)
