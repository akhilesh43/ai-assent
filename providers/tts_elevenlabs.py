"""
ElevenLabs — Ultra-realistic AI voice synthesis with zero-shot emotion and multilingual support.
"""
from __future__ import annotations

import asyncio
import io
import json
import urllib.request
import urllib.error
from typing import AsyncIterator, Optional

from . import audio
from .base import DEFAULT_CAPS, Field


class ElevenLabsTTS:
    def __init__(self, cfg: dict):
        self.api_key = str(cfg.get("api_key") or "").strip()
        self.voice_id = str(cfg.get("voice_id") or "RDWdsTU6N02BFftbIEAp").strip()
        self.model_id = str(cfg.get("model_id") or "eleven_multilingual_v2").strip()
        self.caps = dict(DEFAULT_CAPS)
        self.caps["audio_out"] = True

    async def speak(self, text: str, voice: str = "") -> AsyncIterator[bytes]:
        """Synthesise text via ElevenLabs API, yielding 24 kHz mono 16-bit PCM."""
        t = (text or "").strip()
        if not t:
            return

        key = self.api_key
        if not key:
            try:
                from memory.config_manager import get_elevenlabs_api_key
                key = get_elevenlabs_api_key()
            except Exception:
                pass
        if not key:
            raise RuntimeError("ElevenLabs API key is not configured.")

        vid = (voice or "").strip() or self.voice_id or "RDWdsTU6N02BFftbIEAp"
        # Request raw 24kHz mono PCM for direct real-time playback
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}/stream?output_format=pcm_24000"
        headers = {
            "Accept": "audio/pcm",
            "Content-Type": "application/json",
            "xi-api-key": key,
        }
        payload = json.dumps({
            "text": t,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
            }
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        def _fetch_stream():
            try:
                return urllib.request.urlopen(req, timeout=15.0)
            except urllib.error.HTTPError as err:
                if err.code == 402 and vid != "EXAVITQu4vr4xnSDxMaL":
                    # Library voice on free tier -> fallback to Bella (Soft, sweet girl voice)
                    fb_url = "https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL/stream?output_format=pcm_24000"
                    fb_req = urllib.request.Request(fb_url, data=payload, headers=headers, method="POST")
                    return urllib.request.urlopen(fb_req, timeout=15.0)
                detail = err.read().decode(errors="replace")
                raise RuntimeError(f"ElevenLabs HTTP {err.code}: {detail}") from err

        response = await asyncio.to_thread(_fetch_stream)

        # Stream chunks in ~50ms slices (2400 bytes at 24kHz 16-bit mono)
        chunk_size = 2400
        while True:
            chunk = await asyncio.to_thread(response.read, chunk_size)
            if not chunk:
                break
            yield chunk


def _make(cfg: dict) -> ElevenLabsTTS:
    return ElevenLabsTTS(cfg)


PROVIDER = {
    "id": "elevenlabs",
    "kind": "tts",
    "label": "ElevenLabs",
    "hint": "Ultra-realistic AI voice synthesis (29 languages)",
    "requires": (),
    "catalog": (
        {"id": "elevenlabs", "label": "ElevenLabs Voice AI",
         "note": "Ultra-realistic · 29 languages · hosted API", "local": False,
         "values": {}},
    ),
    "needs": (
        Field("api_key", "API Key (starts with sk_)", kind="secret",
              hint="Your ElevenLabs secret API key starting with sk_"),
        Field("voice_id", "Voice ID", default="RDWdsTU6N02BFftbIEAp",
              hint="Voice ID (e.g. RDWdsTU6N02BFftbIEAp)"),
        Field("model_id", "Model ID", default="eleven_multilingual_v2",
              hint="Model ID (e.g. eleven_multilingual_v2, eleven_turbo_v2_5)"),
    ),
    "caps": {"audio_out": True},
    "factory": _make,
}


def available() -> bool:
    return True
