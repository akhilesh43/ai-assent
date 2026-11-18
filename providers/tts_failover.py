"""
Failover voice — Piper first (neural, on this computer), then the system's own voice.

Without this, a voice that fails (credit used up, key revoked, no internet) means the assistant
goes SILENT: the engine reports the error and carries on without sound. With it, the next
voice in the chain speaks the same sentence.

Failover happens only before the first sound. A voice that dies half-way through a sentence ends
there — a second voice repeating the start of it would be worse.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import AsyncIterator

from ._chain import Chain, build_links, parse_chain
from .base import DEFAULT_CAPS, Field


class FailoverTTS:
    def __init__(self, cfg: dict):
        links, self.skipped = build_links(parse_chain(cfg), cfg, "voice")
        self._chain = Chain(links, "voice", log=cfg.get("log"))
        self.caps = dict(DEFAULT_CAPS)
        self.caps["audio_out"] = True
        self.last_voice = ""

    async def warmup(self, log=None) -> None:
        async def one(link):
            fn = getattr(link.obj, "warmup", None)
            if fn is not None:
                try:
                    await fn(log)
                except Exception:
                    pass
        await asyncio.gather(*(one(l) for l in self._chain.links), return_exceptions=True)

    # `language` is in the signature on purpose: the engine only passes it to a voice whose
    # speak() declares it (Piper needs it, the system voice detects it). Each link gets it only if
    # IT declares it.
    async def speak(self, text: str, voice: str = "", language: str = "") -> AsyncIterator[bytes]:
        order = self._chain.order()
        errors = []
        for i, link in enumerate(order):
            kw = {}
            try:
                params = inspect.signature(link.obj.speak).parameters
            except Exception:
                params = {}
            if language and "language" in params:
                kw["language"] = language
            if voice and "voice" in params and i == 0:
                kw["voice"] = voice
            started = False
            try:
                async for pcm in link.obj.speak(text, **kw):
                    if pcm:
                        if not started:
                            started = True
                            self.last_voice = link.pid.split("@")[0]
                        yield pcm
                if started:
                    self._chain.ok(link)
                    return
                raise RuntimeError("the voice produced no sound")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if started:
                    raise
                self._chain.failed(link, e, remaining=len(order) - i - 1)
                errors.append(f"{link.pid}: {link.last_error}")
        raise RuntimeError("every voice failed — " + " | ".join(errors))


def _make(cfg: dict) -> FailoverTTS:
    return FailoverTTS(cfg)


PROVIDER = {
    "id": "failover_tts",
    "kind": "tts",
    "label": "Auto voice (Piper → system voice)",
    "hint": "The first voice that works speaks; a failing one is rested",
    "requires": (),
    "needs": (Field("chain", "Order (ids, comma separated)", default="piper,system_tts"),),
    "caps": {"audio_out": True},
    "factory": _make,
}


def available() -> bool:
    return True
