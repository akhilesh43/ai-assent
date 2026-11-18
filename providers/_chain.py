"""
Shared machinery for the three failover providers (llm_failover, stt_failover, tts_failover).

THE PROMISE
    "If one key is missing or stops working, the assistant keeps working on the other."
    A chain is an ordered list of provider ids — say Gemini, then a local model. Each request goes
    to the first one that is healthy. If it fails BEFORE producing anything, the same request
    is retried on the next, and the failed one is rested for a while so the next request does
    not waste a round trip on it. When the rest is over it is tried first again, so Gemini
    takes over again by itself once its quota is back.

WHAT IT WILL NOT DO
    Splice two answers together. Once a provider has started producing output, a failure
    ends that turn with an error (the engine already reports it) instead of letting a second
    model continue a sentence the first one began. Repeating half of an answer is worse than
    a clear failure.

A provider whose required settings are empty (no key) is not in the chain at all — it is not
tried and failed each time, which is what "no Gemini key" must mean.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# seconds a failed provider is rested, by kind of failure
REST = {
    "install": 24 * 3600,   # a dependency that is not installed will not install itself
    "auth": 30 * 60,        # a rejected key stays rejected
    "quota": 10 * 60,       # out of credit / rate limited: Gemini's free pool refills slowly
    "network": 90,          # timeouts, 5xx, connection reset: probably brief
    "other": 0,             # a bad request: try the next one, but do not rest this one
}


def classify(exc: BaseException) -> str:
    """install | auth | quota | network | other — from the status code if the exception has one,
    otherwise from its wording (providers raise plain RuntimeError with the server's text)."""
    status = None
    resp = getattr(exc, "response", None)
    if resp is not None:
        status = getattr(resp, "status_code", None)
    status = status or getattr(exc, "status", None) or getattr(exc, "status_code", None)
    msg = (str(exc) or exc.__class__.__name__).lower()
    if not isinstance(status, int):
        # Cloud voices and friends raise RuntimeError("401 <their message>"): the status
        # is the first word, and it is the most reliable thing in the text.
        lead = re.match(r"\s*(\d{3})\b", msg)
        status = int(lead.group(1)) if lead else None
    if isinstance(exc, ImportError) or "not installed" in msg or "no module named" in msg \
            or "pip install" in msg:
        return "install"
    if status in (401, 403) or any(t in msg for t in ("unauthorized", "invalid api key", "invalid_api_key",
                                                         "incorrect api key", "api key not valid",
                                                         "permission denied", "forbidden", "no api key",
                                                         "key was refused")):
        return "auth"
    if status == 429 or any(t in msg for t in ("quota", "rate limit", "rate_limit", "resource_exhausted",
                                                  "insufficient_quota", "too many requests", "credit",
                                                  "exceeded your current")):
        return "quota"
    if (isinstance(status, int) and status >= 500) or any(
            t in msg for t in ("timeout", "timed out", "connect", "connection", "unreachable", "overloaded",
                               "temporarily", "502", "503", "504", "reset by peer", "name resolution")):
        return "network"
    return "other"


@dataclass
class Link:
    pid: str
    obj: Any
    rest_until: float = 0.0
    last_error: str = ""
    kind: str = ""
    served: int = 0


class Chain:
    """Ordered, self-healing list of providers."""

    def __init__(self, links: list[Link], what: str, log: Optional[Callable[[str], None]] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not links:
            raise RuntimeError(f"no {what} is configured — add a key (python enable_providers.py --status)")
        self.links = links
        self.what = what
        self._log = log or (lambda s: None)
        self._clock = clock
        self.active: Link = links[0]

    def order(self) -> list[Link]:
        """Healthy links in order; if every one is resting, all of them (a request must be
        tried somewhere — resting is a preference, not a ban)."""
        now = self._clock()
        ready = [l for l in self.links if l.rest_until <= now]
        return ready or list(self.links)

    def ok(self, link: Link) -> None:
        link.served += 1
        link.rest_until = 0.0
        link.last_error = ""
        if self.active is not link:
            self._log(f"SYS: {self.what} now served by {link.pid}.")
        self.active = link

    def failed(self, link: Link, exc: BaseException, remaining: int) -> str:
        kind = classify(exc)
        link.kind = kind
        link.last_error = " ".join(str(exc).split())[:140] or exc.__class__.__name__
        link.rest_until = self._clock() + REST[kind]
        nxt = f" — switching to the next {self.what} provider" if remaining else ""
        self._log(f"SYS: {self.what} '{link.pid}' failed ({kind}: {link.last_error}){nxt}.")
        return kind

    def summary(self) -> str:
        return " → ".join(l.pid + ("" if l.rest_until <= self._clock() else " (resting)") for l in self.links)


def build_links(chain_ids: list[str], base_cfg: dict, what: str) -> tuple[list[Link], list[str]]:
    """Build every usable provider of the chain. Returns (links, skipped explanations).

    Skipped, not failed: a provider with a required setting empty (no key) or one that cannot
    be built here. The caller decides whether an empty result is fatal."""
    from memory.config_manager import get_provider_config
    from providers import registry

    links: list[Link] = []
    skipped: list[str] = []
    for pid in chain_ids:
        cfg = dict(get_provider_config(pid))
        missing = registry.missing_fields(pid, cfg)
        if missing:
            skipped.append(f"{pid} (needs {', '.join(missing)})")
            continue
        if base_cfg.get("log"):
            cfg["log"] = base_cfg["log"]
        try:
            links.append(Link(pid, registry.build(pid, cfg)))
        except Exception as e:
            skipped.append(f"{pid} ({str(e)[:80]})")
    return links, skipped


def parse_chain(cfg: dict) -> list[str]:
    raw = cfg.get("chain") or ""
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [p.strip() for p in str(raw).split(",") if p.strip()]
