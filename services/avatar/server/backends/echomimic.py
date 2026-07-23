"""EchoMimicV3-Flash backend — warm worker → stream generative portrait video.

Ant Group EchoMimicV3 flash-pro (8-step, ~12GB+). Strong visual peer to
FlashHead; utterance/clip path (not continuous stream).

Requires: .\\setup_echomimic.ps1
Run:      .\\run_server.ps1 -Backend echomimic
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

logger = logging.getLogger("avatar.echomimic")


class EchoMimicWorker:
    def __init__(self, python: Path, worker_py: Path, vendor: Path, flash_dir: Path) -> None:
        self.python = python
        self.worker_py = worker_py
        self.vendor = vendor
        self.flash_dir = flash_dir
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._prepared_source: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None and self._ready.is_set():
                return
            # Stale / mid-start process: tear down and relaunch.
            if self._proc is not None:
                try:
                    self._proc.kill()
                except Exception:
                    pass
                self._proc = None
            self._ready.clear()

            env = os.environ.copy()
            env["ECHOMIMIC_ROOT"] = str(self.vendor)
            env["ECHOMIMIC_FLASH_DIR"] = str(self.flash_dir)
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("ECHOMIMIC_STEPS", "8")
            env.setdefault("ECHOMIMIC_SIZE", "512")
            try:
                import imageio_ffmpeg

                ff_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
                env["PATH"] = ff_dir + os.pathsep + env.get("PATH", "")
            except Exception:
                pass
            env["PATH"] = str(self.python.parent) + os.pathsep + env.get("PATH", "")

            logger.info("Starting EchoMimicV3-Flash worker (warm)…")
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
            threading.Thread(target=self._drain_stderr, daemon=True, name="echomimic-stderr").start()
            ready = self._read_json_line(timeout_s=900.0)
            if not ready or not ready.get("ok"):
                err = (ready or {}).get("error", "no ready message")
                self._kill_unlocked()
                raise RuntimeError(f"EchoMimic worker failed to start: {err}")
            self._ready.set()
            logger.info(
                "EchoMimic worker ready in %ss (steps=%s size=%s)",
                ready.get("load_s"), ready.get("steps"), ready.get("size"),
            )

    def stop(self) -> None:
        with self._lock:
            self._kill_unlocked()
            self._prepared_source = None
            self._ready.clear()

    def _kill_unlocked(self) -> None:
        proc = self._proc
        self._proc = None
        self._ready.clear()
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
                logger.info("echomimic-worker: %s", line[:500])

    def _read_json_line(self, timeout_s: float) -> dict | None:
        """Read stdout until a JSON object line appears (skip blanks / noise)."""
        assert self._proc and self._proc.stdout
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            box: dict = {}

            def _read() -> None:
                try:
                    assert self._proc and self._proc.stdout
                    box["line"] = self._proc.stdout.readline()
                except BaseException as e:  # noqa: BLE001
                    box["err"] = e

            t = threading.Thread(target=_read, daemon=True)
            t.start()
            t.join(timeout=remaining)
            if t.is_alive():
                raise TimeoutError(f"EchoMimic worker timed out after {timeout_s:.0f}s")
            if "err" in box:
                raise box["err"]
            line = (box.get("line") or "").strip()
            if not line:
                if self._proc.poll() is not None:
                    raise RuntimeError("EchoMimic worker exited (no response)")
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.warning("echomimic-worker stdout noise: %s", line[:200])
                continue
        raise TimeoutError(f"EchoMimic worker timed out after {timeout_s:.0f}s")

    def _request(self, req: dict, timeout_s: float = 1200.0) -> dict:
        self.start()
        with self._lock:
            if not self._ready.is_set() or not self._proc or self._proc.poll() is not None:
                raise RuntimeError("EchoMimic worker not ready")
            assert self._proc.stdin
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            resp = self._read_json_line(timeout_s=timeout_s)
        if not resp or not resp.get("ok"):
            err = (resp or {}).get("error", "empty response")
            trace = (resp or {}).get("trace", "")
            raise RuntimeError(f"EchoMimic {req.get('cmd')} failed: {err}\n{trace[-1500:]}")
        return resp

    def prepare(self, source_path: Path) -> None:
        key = str(source_path)
        if self._prepared_source == key and self._ready.is_set() and self._proc and self._proc.poll() is None:
            return
        self._request({"cmd": "prepare", "source_path": str(source_path)}, timeout_s=120.0)
        self._prepared_source = key

    def infer(self, audio_path: Path, output_path: Path) -> Path:
        resp = self._request({
            "cmd": "infer",
            "audio_path": str(audio_path),
            "output_path": str(output_path),
        }, timeout_s=1200.0)
        logger.info("EchoMimic infer %dms → %s", resp.get("ms"), Path(resp["video_path"]).name)
        return Path(resp["video_path"])


class EchoMimicBackend(UtteranceStreamEngine):
    backend_id = "echomimic"
    backend_name = "EchoMimicV3-Flash"

    def __init__(self, root: Path) -> None:
        super().__init__(chunk_frames=12, fps=25)
        self.root = root
        self.vendor = Path(os.environ.get("ECHOMIMIC_ROOT", root / "vendor" / "echomimic_v3"))
        self.flash_dir = Path(os.environ.get("ECHOMIMIC_FLASH_DIR", self.vendor / "flash"))
        self.python = Path(os.environ.get(
            "ECHOMIMIC_PYTHON",
            root / ".venv_echomimic" / "Scripts" / "python.exe",
        ))
        if not self.python.is_file():
            self.python = Path(sys.executable)

        worker_py = root / "scripts" / "echomimic_worker.py"
        xform = self.flash_dir / "transformer" / "diffusion_pytorch_model.safetensors"
        if not self.vendor.is_dir() or not worker_py.is_file():
            raise RuntimeError(
                f"EchoMimic incomplete.\n  vendor: {self.vendor}\n  worker: {worker_py}\n"
                "Run: .\\setup_echomimic.ps1"
            )
        if not xform.is_file():
            raise RuntimeError(
                f"EchoMimic flash transformer missing: {xform}\nRun: .\\setup_echomimic.ps1"
            )

        self._work = root / "uploads" / "_echomimic_work"
        self._work.mkdir(parents=True, exist_ok=True)
        self._avatar_img: Path | None = None
        self._worker = EchoMimicWorker(self.python, worker_py, self.vendor, self.flash_dir)
        self._warm_lock = threading.Lock()
        self._warm_started = False
        self._warm_error: str | None = None
        self._warm_done = threading.Event()

    def on_prepare(self) -> None:
        """Lock Jordan still for the UI; warm EchoMimic worker in the background."""
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
        threading.Thread(target=self._warm_worker, args=(dest,), daemon=True, name="em-warm").start()

    def _warm_worker(self, dest: Path) -> None:
        try:
            logger.info("Warming EchoMimic worker in background…")
            self._worker.start()
            self._worker.prepare(dest)
            logger.info("EchoMimic worker warm complete")
        except Exception as e:
            self._warm_error = str(e)
            logger.exception("EchoMimic background warm failed")
        finally:
            self._warm_done.set()

    def _ensure_worker(self) -> None:
        if self._warm_error:
            raise RuntimeError(f"EchoMimic worker failed to warm: {self._warm_error}")
        if not self._warm_done.wait(timeout=900.0):
            raise TimeoutError("EchoMimic worker warm timed out (15 min)")
        if self._warm_error:
            raise RuntimeError(f"EchoMimic worker failed to warm: {self._warm_error}")

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
            raise RuntimeError("EchoMimic produced no frames")

        assert self._source_bgr is not None
        sh, sw = self._source_bgr.shape[:2]
        fh, fw = frames[0].shape[:2]
        if (fw, fh) != (sw, sh):
            return [
                cv2.resize(f, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
                for f in frames
            ]
        return frames


def create(root: Path):
    return EchoMimicBackend(root)
