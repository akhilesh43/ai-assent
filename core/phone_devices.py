"""
core/phone_devices.py — which phones the owner has paired, remembered across restarts.

A paired phone holds a long random DEVICE TOKEN. Only its SHA-256 is stored here, so a copy of
config/phone_devices.json cannot be used to log in. The phone trades its device token for a
short-lived session token (see dashboard/server.py: /api/app-login), which is why the laptop can
restart or change network without the phone being scanned again.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Optional

_PATH = Path(__file__).resolve().parent.parent / "config" / "phone_devices.json"
_LOCK = threading.Lock()


def _load(path: Path) -> dict[str, Any]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _hash(tok: str) -> str:
    return hashlib.sha256(tok.encode()).hexdigest()


def issue(name: str = "Phone", path: Optional[Path] = None) -> tuple[str, str]:
    """Register a phone. Returns (device_id, device_token); the token is shown exactly once."""
    path = path or _PATH
    tok = secrets.token_urlsafe(40)
    dev_id = "ph-" + secrets.token_hex(4)
    with _LOCK:
        data = _load(path)
        data[_hash(tok)] = {"id": dev_id, "name": (name or "Phone")[:40], "created": time.time()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dev_id, tok


def verify(tok: str, path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    if not tok or len(tok) < 20:
        return None
    with _LOCK:
        return _load(path or _PATH).get(_hash(tok))


def list_devices(path: Optional[Path] = None) -> list[dict[str, Any]]:
    with _LOCK:
        return sorted(_load(path or _PATH).values(), key=lambda d: -d.get("created", 0))


def revoke_all(path: Optional[Path] = None) -> int:
    path = path or _PATH
    with _LOCK:
        n = len(_load(path))
        if path.exists():
            path.write_text("{}", encoding="utf-8")
        return n
