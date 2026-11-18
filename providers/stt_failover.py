"""
Failover ears — local Whisper first (free, private), a cloud Whisper when the local one cannot run.

The commonest reason to need this: `faster-whisper` is not installed, or its model failed to
load. Without it, a person with only a cloud brain key would have a brain and a voice and no ears.
"""
from __future__ import annotations

import asyncio

from ._chain import Chain, build_links, parse_chain
from .base import DEFAULT_CAPS, Field


class FailoverSTT:
    def __init__(self, cfg: dict):
        links, self.skipped = build_links(parse_chain(cfg), cfg, "ears")
        self._chain = Chain(links, "ears", log=cfg.get("log"))
        self.caps = dict(DEFAULT_CAPS)

    @property
    def detected_language(self) -> str:
        return str(getattr(self._chain.active.obj, "detected_language", "") or "")

    @property
    def model(self) -> str:
        o = self._chain.active.obj
        return str(getattr(o, "model", "") or self._chain.active.pid)

    @property
    def device(self) -> str:
        return str(getattr(self._chain.active.obj, "device", "") or "")

    async def warmup(self, log=None) -> None:
        """Warm every link; one failing to load must not stop the others (nor the app)."""
        async def one(link):
            fn = getattr(link.obj, "warmup", None)
            if fn is None:
                return
            try:
                await fn(log)
            except Exception as e:
                self._chain.failed(link, e, remaining=1)
        await asyncio.gather(*(one(l) for l in self._chain.links), return_exceptions=True)

    async def transcribe(self, pcm: bytes, sample_rate: int = 16000, language: str = "") -> str:
        order = self._chain.order()
        errors = []
        for i, link in enumerate(order):
            try:
                text = await link.obj.transcribe(pcm, sample_rate, language)
                self._chain.ok(link)
                return text
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._chain.failed(link, e, remaining=len(order) - i - 1)
                errors.append(f"{link.pid}: {link.last_error}")
        raise RuntimeError("every speech recogniser failed — " + " | ".join(errors))


def _make(cfg: dict) -> FailoverSTT:
    return FailoverSTT(cfg)


PROVIDER = {
    "id": "failover_stt",
    "kind": "stt",
    "label": "Auto ears (local → Groq)",
    "hint": "Local Whisper first; Groq Whisper if the local one cannot run",
    "requires": (),
    "needs": (Field("chain", "Order (ids, comma separated)", default="whisper@local,whisper@groq"),),
    "caps": {},
    "factory": _make,
}


def available() -> bool:
    return True
