"""LivePortrait backend — FasterLivePortrait ONNX + JoyVASA audio→motion.

Still image → prepare once → per-utterance motion + warp pasteback → stream frames.

Requires:
  .\\setup_liveportrait.ps1

Run: .\\run_server.ps1 -Backend liveportrait
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

logger = logging.getLogger("avatar.liveportrait")


class LivePortraitWorker:
    def __init__(self, python: Path, worker_py: Path, vendor: Path) -> None:
        self.python = python
        self.worker_py = worker_py
        self.vendor = vendor
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._prepared_source: str | None = None
        self._stopping = False

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._stopping:
            raise RuntimeError("LivePortrait worker is shutting down")
        if self._proc and self._proc.poll() is None:
            return
        env = os.environ.copy()
        env["FLP_ROOT"] = str(self.vendor)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("FLP_CFG", "configs/onnx_infer.yaml")
        env.setdefault("FLP_ANIMATION_REGION", "exp")
        env.setdefault("FLP_CFG_SCALE", "1.2")
        env.setdefault("LIVEPORTRAIT_FPS", "25")
        # Face-crop path (skip full-frame pasteback) — much faster for chat.
        env.setdefault("LIVEPORTRAIT_FACE_ONLY", "1")
        # Cap rendered frames so CPU-onnx warps don't hang chat forever.
        # ~10s/frame without CUDA EP; 12 frames ≈ 2 min worst case.
        env.setdefault("LIVEPORTRAIT_MAX_FRAMES", "12")
        env.setdefault("TRANSFORMERS_ATTN_IMPLEMENTATION", "eager")
        # Ensure pip nvidia-* DLL dirs are on PATH for onnxruntime CUDA EP.
        try:
            import glob
            import site

            extra = []
            for sp in site.getsitepackages():
                extra.extend(glob.glob(os.path.join(sp, "nvidia", "cudnn", "bin")))
                extra.extend(glob.glob(os.path.join(sp, "nvidia", "cublas", "bin")))
                extra.extend(glob.glob(os.path.join(sp, "nvidia", "cuda_runtime", "bin")))
            if extra:
                env["PATH"] = os.pathsep.join(extra) + os.pathsep + env.get("PATH", "")
        except Exception:
            pass
        try:
            import imageio_ffmpeg

            ff_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
            env["PATH"] = ff_dir + os.pathsep + env.get("PATH", "")
        except Exception:
            pass
        env["PATH"] = str(self.python.parent) + os.pathsep + env.get("PATH", "")

        logger.info("Starting LivePortrait worker (FasterLivePortrait + JoyVASA)…")
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
        proc = self._proc
        threading.Thread(
            target=self._drain_stderr, args=(proc,), daemon=True, name="lp-stderr"
        ).start()
        ready = self._read_from(proc, timeout_s=600.0)
        if not ready or not ready.get("ok"):
            err = (ready or {}).get("error", "no ready message")
            self._kill_proc(proc)
            self._proc = None
            raise RuntimeError(f"LivePortrait worker failed to start: {err}")
        logger.info(
            "LivePortrait worker ready in %ss (region=%s)",
            ready.get("load_s"),
            ready.get("animation_region"),
        )

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            proc = self._proc
            self._proc = None
            self._prepared_source = None
        self._kill_proc(proc)

    def _kill_proc(self, proc: subprocess.Popen | None) -> None:
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

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        if not proc.stderr:
            return
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                logger.info("lp-worker: %s", line[:500])

    def _read_from(self, proc: subprocess.Popen, timeout_s: float) -> dict | None:
        assert proc.stdout
        box: dict = {}

        def _read() -> None:
            try:
                box["line"] = proc.stdout.readline()
            except BaseException as e:  # noqa: BLE001
                box["err"] = e

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=timeout_s)
        if t.is_alive():
            raise TimeoutError(f"LivePortrait worker timed out after {timeout_s:.0f}s")
        if "err" in box:
            raise box["err"]
        line = box.get("line") or ""
        if not line:
            raise RuntimeError("LivePortrait worker exited (no response)")
        return json.loads(line)

    def _request(self, req: dict, timeout_s: float = 600.0) -> dict:
        with self._lock:
            if self._stopping:
                raise RuntimeError("LivePortrait worker is shutting down")
            if not self._proc or self._proc.poll() is not None:
                self._start_locked()
            proc = self._proc
            if proc is None or proc.stdin is None or proc.stdout is None:
                raise RuntimeError("LivePortrait worker has no stdin/stdout")
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            resp = self._read_from(proc, timeout_s=timeout_s)
        if not resp or not resp.get("ok"):
            err = (resp or {}).get("error", "empty response")
            trace = (resp or {}).get("trace", "")
            raise RuntimeError(f"LivePortrait {req.get('cmd')} failed: {err}\n{trace[-1500:]}")
        return resp

    def prepare(self, source_path: Path) -> None:
        key = str(source_path)
        if self._prepared_source == key and self.is_alive:
            return
        self._request({"cmd": "prepare", "source_path": str(source_path)}, timeout_s=300.0)
        self._prepared_source = key

    def infer(self, audio_path: Path, output_path: Path) -> Path:
        resp = self._request({
            "cmd": "infer",
            "audio_path": str(audio_path),
            "output_path": str(output_path),
        }, timeout_s=600.0)
        logger.info(
            "LivePortrait warm infer %dms (%s frames) → %s",
            resp.get("ms"),
            resp.get("n_frames"),
            Path(resp["video_path"]).name,
        )
        return Path(resp["video_path"])


class LivePortraitBackend(UtteranceStreamEngine):
    backend_id = "liveportrait"
    backend_name = "LivePortrait"

    def __init__(self, root: Path) -> None:
        super().__init__(chunk_frames=12, fps=25)
        self.root = root
        self.vendor = Path(os.environ.get("FLP_ROOT", root / "vendor" / "FasterLivePortrait"))
        self.python = Path(os.environ.get(
            "LIVEPORTRAIT_PYTHON",
            root / ".venv_liveportrait" / "Scripts" / "python.exe",
        ))
        if not self.python.is_file():
            self.python = Path(sys.executable)

        worker_py = root / "scripts" / "liveportrait_worker.py"
        onnx = self.vendor / "checkpoints" / "liveportrait_onnx" / "warping_spade.onnx"
        if not self.vendor.is_dir() or not worker_py.is_file():
            raise RuntimeError(
                f"LivePortrait incomplete.\n  vendor: {self.vendor}\n  worker: {worker_py}\n"
                "Run: .\\setup_liveportrait.ps1"
            )
        if not onnx.is_file():
            raise RuntimeError(
                f"LivePortrait ONNX missing: {onnx}\nRun: .\\setup_liveportrait.ps1"
            )

        self._work = root / "uploads" / "_liveportrait_work"
        self._work.mkdir(parents=True, exist_ok=True)
        self._avatar_img: Path | None = None
        self._worker = LivePortraitWorker(self.python, worker_py, self.vendor)
        self._warm_lock = threading.Lock()
        self._warm_started = False
        self._warm_error: str | None = None
        self._warm_done = threading.Event()

    def on_prepare(self) -> None:
        """Fast path: lock Jordan still for the UI. Warm FLP worker in background."""
        assert self._source_image
        dest = self._work / "source.png"
        shutil.copy2(self._source_image, dest)
        self._avatar_img = dest
        self._warm_error = None
        self._warm_done.clear()
        with self._warm_lock:
            if self._warm_started:
                return
            self._warm_started = True
        threading.Thread(target=self._warm_worker, args=(dest,), daemon=True, name="lp-warm").start()

    def _warm_worker(self, dest: Path) -> None:
        try:
            if self._worker._stopping:
                return
            logger.info("Warming LivePortrait worker in background…")
            self._worker.start()
            if self._worker._stopping:
                return
            self._worker.prepare(dest)
            logger.info("LivePortrait worker warm complete")
        except Exception as e:  # noqa: BLE001
            if self._worker._stopping:
                logger.info("LivePortrait warm aborted (server shutting down)")
                return
            self._warm_error = str(e)
            logger.exception("LivePortrait background warm failed")
        finally:
            self._warm_done.set()

    def _ensure_worker(self) -> None:
        if self._warm_error:
            raise RuntimeError(f"LivePortrait worker failed to warm: {self._warm_error}")
        if self._worker._prepared_source and self._worker.is_alive:
            return
        # Wait for background warm (first chat may block here once).
        if not self._warm_done.wait(timeout=600.0):
            raise TimeoutError("LivePortrait worker still warming after 10 minutes")
        if self._warm_error:
            raise RuntimeError(f"LivePortrait worker failed to warm: {self._warm_error}")
        if not (self._worker._prepared_source and self._worker.is_alive):
            # Warm thread finished without prepared source — start on demand.
            assert self._avatar_img is not None
            self._worker._stopping = False
            self._worker.start()
            self._worker.prepare(self._avatar_img)

    def stop(self) -> None:
        try:
            self._worker.stop()
        finally:
            self._warm_started = False
            self._warm_done.set()
            super().stop()

    def render_utterance(self, audio_f32_16k: np.ndarray) -> list[np.ndarray]:
        assert self._avatar_img is not None
        import soundfile as sf

        self._ensure_worker()

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
            raise RuntimeError("LivePortrait produced no frames")

        assert self._source_bgr is not None
        sh, sw = self._source_bgr.shape[:2]
        fh, fw = frames[0].shape[:2]
        face_only = os.environ.get("LIVEPORTRAIT_FACE_ONLY", "1").strip() not in (
            "0", "false", "False",
        )
        # Face crop → locked composite onto full still (Full image framing).
        if face_only and self._server_composite and self._overlay:
            return [self.composite_model_face(f) for f in frames]
        if (fw, fh) != (sw, sh):
            return [
                cv2.resize(f, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
                for f in frames
            ]
        return frames


def create(root: Path):
    return LivePortraitBackend(root)
