"""LLM (OpenAI) and TTS clients.

TTS returns float32 mono 16 kHz numpy arrays ready for the avatar engine.

Providers (TTS_PROVIDER=auto|openai|edge|elevenlabs|cartesia):
  auto — ElevenLabs → OpenAI → Edge (free) → Cartesia
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import tempfile
import wave
from collections.abc import Iterator

import httpx
import numpy as np

logger = logging.getLogger("avatar.clients")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"
ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"

_SENTENCE_END = re.compile(r"[.!?…](?=\s|$)|[\n]")
_SOFT_FLUSH = 72


class ChatClient:
    def __init__(self, api_key: str, model: str, system_prompt: str) -> None:
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self._http = httpx.Client(timeout=60.0)

    def stream_sentences(self, history: list[dict], user_text: str) -> Iterator[str]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for t in history[-12:]:
            if t.get("role") in ("user", "assistant") and t.get("content"):
                messages.append({"role": t["role"], "content": t["content"]})
        messages.append({"role": "user", "content": user_text})

        buf = ""
        with self._http.stream(
            "POST", OPENAI_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages,
                  "temperature": 0.7, "max_tokens": 220, "stream": True},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if not delta:
                    continue
                buf += delta
                while True:
                    m = _SENTENCE_END.search(buf)
                    if m:
                        chunk, buf = buf[:m.end()].strip(), buf[m.end():]
                        if chunk:
                            yield chunk
                        continue
                    if len(buf) >= _SOFT_FLUSH and " " in buf:
                        sp = buf.rfind(" ")
                        chunk, buf = buf[:sp].strip(), buf[sp + 1:]
                        if chunk:
                            yield chunk
                    break
        if buf.strip():
            yield buf.strip()


def _wav_bytes_to_f32_16k(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    if rate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
    return audio


def _mp3_bytes_to_f32_16k(data: bytes) -> np.ndarray:
    import librosa

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        audio, _ = librosa.load(path, sr=16000, mono=True)
        return audio.astype(np.float32)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class TTSClient:
    """Multi-provider TTS with free Edge fallback for local testing."""

    def __init__(self) -> None:
        self.openai_key = os.environ.get("OPENAI_API_KEY", "")
        self.openai_tts_model = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
        self.openai_tts_voice = os.environ.get("OPENAI_TTS_VOICE", "alloy")

        self.eleven_key = os.environ.get("ELEVENLABS_API_KEY", "")
        self.eleven_voice = os.environ.get("ELEVENLABS_VOICE_ID", "")
        self.eleven_model = os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")

        self.cartesia_key = os.environ.get("CARTESIA_API_KEY", "")
        self.cartesia_voice = os.environ.get("CARTESIA_VOICE_ID", "")
        self.cartesia_model = os.environ.get("CARTESIA_MODEL", "sonic-2")

        self.edge_voice = os.environ.get("EDGE_TTS_VOICE", "en-US-JennyNeural")

        self._http = httpx.Client(timeout=60.0)
        self._order = self._resolve_order()
        self.provider = self._order[0] if self._order else "none"
        logger.info("TTS providers (in order): %s", " → ".join(self._order) or "(none)")

    def _resolve_order(self) -> list[str]:
        forced = (os.environ.get("TTS_PROVIDER") or "auto").strip().lower()
        available: dict[str, bool] = {
            "elevenlabs": bool(self.eleven_key and self.eleven_voice),
            "openai": bool(self.openai_key),
            "edge": True,  # free, no key
            "cartesia": bool(self.cartesia_key and self.cartesia_voice),
        }
        if forced != "auto":
            if forced not in available:
                raise RuntimeError(f"Unknown TTS_PROVIDER={forced}")
            if not available[forced] and forced != "edge":
                raise RuntimeError(f"TTS_PROVIDER={forced} requested but credentials missing")
            return [forced]
        # Prefer paid quality when configured; Edge is the free testing default
        # ahead of Cartesia so a dead Cartesia balance doesn't block demos.
        order = []
        for name in ("elevenlabs", "openai", "edge", "cartesia"):
            if available[name]:
                order.append(name)
        return order

    def warmup(self) -> None:
        """Pre-establish the TLS connection so the first real call is fast."""
        try:
            self.synthesize("Hi.")
            logger.info("TTS connection warmed (%s)", self.provider)
        except Exception as e:  # noqa: BLE001
            logger.warning("TTS warmup failed: %s", e)

    def synthesize(self, text: str) -> np.ndarray:
        if not self._order:
            raise RuntimeError("No TTS provider configured")
        errors: list[str] = []
        for name in self._order:
            try:
                audio = self._call(name, text)
                if name != self.provider:
                    logger.info("TTS fell back to %s (was %s)", name, self.provider)
                    self.provider = name
                return audio
            except Exception as e:  # noqa: BLE001
                msg = f"{name}: {e}"
                logger.warning("TTS %s failed: %s", name, e)
                errors.append(msg)
        raise RuntimeError("All TTS providers failed:\n" + "\n".join(errors))

    def _call(self, name: str, text: str) -> np.ndarray:
        if name == "elevenlabs":
            return self._eleven(text)
        if name == "openai":
            return self._openai(text)
        if name == "edge":
            return self._edge(text)
        if name == "cartesia":
            return self._cartesia(text)
        raise RuntimeError(f"unknown TTS provider {name}")

    def _eleven(self, text: str) -> np.ndarray:
        resp = self._http.post(
            ELEVEN_URL.format(voice_id=self.eleven_voice),
            headers={"xi-api-key": self.eleven_key},
            params={"output_format": "pcm_16000", "optimize_streaming_latency": "3"},
            json={"text": text, "model_id": self.eleven_model},
        )
        resp.raise_for_status()
        pcm = np.frombuffer(resp.content, dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    def _openai(self, text: str) -> np.ndarray:
        resp = self._http.post(
            OPENAI_TTS_URL,
            headers={"Authorization": f"Bearer {self.openai_key}"},
            json={
                "model": self.openai_tts_model,
                "input": text,
                "voice": self.openai_tts_voice,
                "response_format": "wav",
            },
        )
        resp.raise_for_status()
        return _wav_bytes_to_f32_16k(resp.content)

    def _edge(self, text: str) -> np.ndarray:
        """Microsoft Edge neural TTS — free, no API key (good for local bench)."""
        try:
            import edge_tts
        except ImportError as e:
            raise RuntimeError(
                "edge-tts not installed. Run: pip install edge-tts"
            ) from e

        async def _run() -> bytes:
            communicate = edge_tts.Communicate(text, self.edge_voice)
            parts: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    parts.append(chunk["data"])
            if not parts:
                raise RuntimeError("edge-tts returned no audio")
            return b"".join(parts)

        # Avoid nested-loop issues if somehow already in an event loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                mp3 = pool.submit(asyncio.run, _run()).result(timeout=60)
        else:
            mp3 = asyncio.run(_run())
        return _mp3_bytes_to_f32_16k(mp3)

    def _cartesia(self, text: str) -> np.ndarray:
        resp = self._http.post(
            CARTESIA_URL,
            headers={"X-API-Key": self.cartesia_key,
                     "Cartesia-Version": "2024-06-10",
                     "Content-Type": "application/json"},
            json={"model_id": self.cartesia_model, "transcript": text,
                  "voice": {"mode": "id", "id": self.cartesia_voice},
                  "output_format": {"container": "wav", "encoding": "pcm_s16le",
                                    "sample_rate": 16000},
                  "language": "en"},
        )
        resp.raise_for_status()
        return _wav_bytes_to_f32_16k(resp.content)
