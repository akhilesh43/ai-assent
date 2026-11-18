"""
Failover language model — Gemini first, a local model (Ollama etc.) when Gemini is missing or fails (or any order).

One LLM to the rest of the app. Inside, an ordered chain of real ones (see providers/_chain.py
for the rules). Used by `enable_providers.py --preset auto`, which sets the chain from the keys
you actually have — one key gives a chain of one and everything still works.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from ._chain import Chain, build_links, parse_chain
from .base import DEFAULT_CAPS, Field


class FailoverLLM:
    def __init__(self, cfg: dict):
        ids = parse_chain(cfg)
        links, self.skipped = build_links(ids, cfg, "brain")
        self._chain = Chain(links, "brain", log=cfg.get("log"))
        # Conservative on purpose: a capability is advertised only if EVERY link has it,
        # because any of them may be the one that answers. Tools are declared once, before
        # anyone knows who will serve the turn.
        caps = dict(DEFAULT_CAPS)
        all_caps = [dict(getattr(l.obj, "caps", {}) or {}) for l in links]
        caps["tools"] = all(c.get("tools") for c in all_caps)
        caps["vision"] = all(c.get("vision") for c in all_caps)
        caps["tool_format"] = all_caps[0].get("tool_format")
        limits = [c.get("max_tools") for c in all_caps if c.get("max_tools")]
        caps["max_tools"] = min(limits) if limits else None
        lim2 = [c.get("max_tool_chars") for c in all_caps if c.get("max_tool_chars")]
        caps["max_tool_chars"] = min(lim2) if lim2 else None
        self.caps = caps

    # what the engine prints in its timing line: the model that is actually answering
    @property
    def model(self) -> str:
        return str(getattr(self._chain.active.obj, "model", "") or self._chain.active.pid)

    @property
    def base_url(self) -> str:
        return str(getattr(self._chain.active.obj, "base_url", "") or "")

    @property
    def chain_summary(self) -> str:
        return self._chain.summary()

    def ensure_vision_known(self, save=None) -> bool:
        """Settle 'can it see?' for every link, each saved under its OWN id (the `save`
        the caller hands in is bound to this failover's id, which would be the wrong place)."""
        from memory.config_manager import save_provider_config
        seen_all = True
        for link in self._chain.links:
            ensure = getattr(link.obj, "ensure_vision_known", None)
            if ensure is None:
                seen_all = seen_all and bool(getattr(link.obj, "caps", {}).get("vision"))
                continue
            try:
                seen = bool(ensure(lambda v, _id=link.pid: save_provider_config(_id, v)))
            except Exception:
                seen = bool(getattr(link.obj, "caps", {}).get("vision"))
            seen_all = seen_all and seen
        self.caps["vision"] = seen_all
        return seen_all

    async def chat(self, messages: list[dict], tools: Optional[list[dict]] = None,
                   *, stream: bool = True) -> AsyncIterator[dict]:
        order = self._chain.order()
        errors: list[str] = []
        for i, link in enumerate(order):
            started = False
            try:
                async for ev in link.obj.chat(messages, tools, stream=stream):
                    started = True
                    yield ev
                self._chain.ok(link)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if started:
                    raise                      # never splice two models into one answer
                self._chain.failed(link, e, remaining=len(order) - i - 1)
                errors.append(f"{link.pid}: {link.last_error}")
        raise RuntimeError("every brain failed — " + " | ".join(errors))

    async def aclose(self) -> None:
        for link in self._chain.links:
            for name in ("aclose", "close"):
                fn = getattr(link.obj, name, None)
                if fn is None:
                    continue
                try:
                    r = fn()
                    if asyncio.iscoroutine(r):
                        await r
                except Exception:
                    pass
                break


def _make(cfg: dict) -> FailoverLLM:
    return FailoverLLM(cfg)


PROVIDER = {
    "id": "failover_llm",
    "kind": "llm",
    "label": "Auto brain (Gemini → local model)",
    "hint": "Uses the first working brain of a list; falls over to the next on its own",
    "requires": ("httpx",),
    "install": "httpx",
    "needs": (
        Field("chain", "Order (ids, comma separated)", default="openai_compat@custom,openai_compat@ollama",
              hint="e.g. openai_compat@custom (Gemini), openai_compat@ollama"),
    ),
    "caps": {"tools": True, "tool_format": "openai"},
    "factory": _make,
}


def available() -> bool:
    from importlib.util import find_spec
    return find_spec("httpx") is not None
