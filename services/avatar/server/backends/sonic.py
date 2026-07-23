"""Sonic backend — warm worker, face-crop quality path, full-image composite.

CVPR 2025 global-audio portrait animation. Best local quality path:
  crop (expand 0.5) → 512px / 25 steps / RIFE → feather paste onto full still.

Requires:
  .\\setup_sonic.ps1   (~15GB+ checkpoints; roomy GPU)

Run: .\\run_server.ps1 -Backend sonic

License: CC BY-NC-SA — non-commercial research/demo only.
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
from framing import bbox_to_norm, feather_composite_bgr

logger = logging.getLogger("avatar.sonic")


class SonicWorker:
    def __init__(self, python: Path, worker_py: Path, vendor: Path) -> None:
        self.python = python
        self.worker_py = worker_py
        self.vendor = vendor
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._prepared_source: str | None = None
        self.crop_bbox: list[int] | None = None

    def start(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return
            env = os.environ.copy()
            env["SONIC_ROOT"] = str(self.vendor)
            env["PYTHONUNBUFFERED"] = "1"
            # Quality defaults — override in .env for more steps / res.
            env.setdefault("SONIC_MIN_RES", "512")
            # 10 steps ≈ usable bench latency; 25 is official quality (very slow).
            env.setdefault("SONIC_STEPS", "10")
            env.setdefault("SONIC_DYNAMIC_SCALE", "1.0")
            env.setdefault("SONIC_EXPAND_RATIO", "0.5")
            env.setdefault("SONIC_RIFE", "1")
            env.setdefault("SONIC_SEED", "72589")
            try:
                import imageio_ffmpeg

                ff_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
                env["PATH"] = ff_dir + os.pathsep + env.get("PATH", "")
            except Exception:
                pass
            env["PATH"] = str(self.python.parent) + os.pathsep + env.get("PATH", "")

            logger.info("Starting Sonic worker (SVD + Sonic, warm)…")
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
            threading.Thread(target=self._drain_stderr, daemon=True, name="sonic-stderr").start()
            ready = self._read_locked(timeout_s=600.0)
            if not ready or not ready.get("ok"):
                err = (ready or {}).get("error", "no ready message")
                self.stop()
                raise RuntimeError(f"Sonic worker failed to start: {err}")
            logger.info(
                "Sonic worker ready in %ss (res=%s steps=%s rife=%s)",
                ready.get("load_s"),
                ready.get("min_res"),
                ready.get("steps"),
                ready.get("rife"),
            )

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
            self._prepared_source = None
            self.crop_bbox = None
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
                logger.info("sonic-worker: %s", line[:500])

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
            raise TimeoutError(f"Sonic worker timed out after {timeout_s:.0f}s")
        if "err" in box:
            raise box["err"]
        line = box.get("line") or ""
        if not line:
            raise RuntimeError("Sonic worker exited (no response)")
        return json.loads(line)

    def _request(self, req: dict, timeout_s: float = 1200.0) -> dict:
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
            raise RuntimeError(f"Sonic {req.get('cmd')} failed: {err}\n{trace[-1500:]}")
        return resp

    def prepare(self, source_path: Path) -> list[int]:
        key = str(source_path)
        if (
            self._prepared_source == key
            and self.crop_bbox is not None
            and self._proc
            and self._proc.poll() is None
        ):
            return self.crop_bbox
        resp = self._request({"cmd": "prepare", "source_path": str(source_path)}, timeout_s=300.0)
        self.crop_bbox = [int(v) for v in resp["crop_bbox"]]
        self._prepared_source = key
        return self.crop_bbox

    def infer(self, audio_path: Path, output_path: Path) -> Path:
        resp = self._request({
            "cmd": "infer",
            "audio_path": str(audio_path),
            "output_path": str(output_path),
        }, timeout_s=1200.0)
        logger.info("Sonic warm infer %dms → %s", resp.get("ms"), Path(resp["video_path"]).name)
        return Path(resp["video_path"])


class SonicBackend(UtteranceStreamEngine):
    backend_id = "sonic"
    backend_name = "Sonic"

    def __init__(self, root: Path) -> None:
        # RIFE doubles fps (12.5 → 25) when enabled — match stream cadence.
        super().__init__(chunk_frames=12, fps=25)
        self.root = root
        self.vendor = Path(os.environ.get("SONIC_ROOT", root / "vendor" / "Sonic"))
        self.python = Path(os.environ.get(
            "SONIC_PYTHON",
            root / ".venv_sonic" / "Scripts" / "python.exe",
        ))
        if not self.python.is_file():
            self.python = Path(sys.executable)

        worker_py = root / "scripts" / "sonic_worker.py"
        ckpt = self.vendor / "checkpoints" / "Sonic" / "unet.pth"
        if not self.vendor.is_dir() or not worker_py.is_file():
            raise RuntimeError(
                f"Sonic incomplete.\n  vendor: {self.vendor}\n  worker: {worker_py}\n"
                "Run: .\\setup_sonic.ps1"
            )
        if not ckpt.is_file():
            raise RuntimeError(
                f"Sonic checkpoint missing: {ckpt}\nRun: .\\setup_sonic.ps1"
            )

        self._work = root / "uploads" / "_sonic_work"
        self._work.mkdir(parents=True, exist_ok=True)
        self._avatar_img: Path | None = None
        self._crop_bbox: list[int] | None = None
        self._worker = SonicWorker(self.python, worker_py, self.vendor)

    def on_prepare(self) -> None:
        assert self._source_image and self._source_bgr is not None
        dest = self._work / "source.png"
        shutil.copy2(self._source_image, dest)
        self._avatar_img = dest
        self._worker.start()
        bbox = self._worker.prepare(dest)
        self._crop_bbox = bbox
        # Lock overlay to Sonic's crop so Full-image framing pastes correctly.
        sh, sw = self._source_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        self._overlay = bbox_to_norm(x1, y1, x2, y2, sw, sh)
        self._face_box_abs = [x1, y1, x2, y2]
        self._server_composite = True
        self._composite = False
        logger.info("Sonic crop locked %s on %dx%d", bbox, sw, sh)

    def stop(self) -> None:
        try:
            self._worker.stop()
        finally:
            super().stop()

    def render_utterance(self, audio_f32_16k: np.ndarray) -> list[np.ndarray]:
        assert self._avatar_img is not None and self._crop_bbox is not None
        assert self._source_bgr is not None
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
            raise RuntimeError("Sonic produced no frames")

        # Feather Sonic's portrait crop back onto the full still (locked box).
        x1, y1, x2, y2 = self._crop_bbox
        out: list[np.ndarray] = []
        for fr in frames:
            face = fr
            if face.shape[1] != (x2 - x1) or face.shape[0] != (y2 - y1):
                face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
            canvas = feather_composite_bgr(
                self._source_bgr,
                face,
                self._overlay,
            )
            out.append(canvas)
        return out


def create(root: Path):
    return SonicBackend(root)
