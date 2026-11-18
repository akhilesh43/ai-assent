"""
Background Focus / Pomodoro Timer plugin with settings schema and spoken alerts.
"""
from __future__ import annotations

import threading
import time

from memory.config_manager import get_plugin_config

PLUGIN = {
    "name": "focus_timer",
    "tier": "extra",
    "brief": "Start, check, or stop a background focus / Pomodoro timer.",
    "description": (
        "Start, check status of, or cancel a background focus or break timer "
        "(Pomodoro). Runs in the background and notifies/speaks when time is up."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "enum": ["start", "status", "stop"],
                "description": "Timer action.",
            },
            "minutes": {
                "type": "INTEGER",
                "description": "Duration in minutes (1 to 180). Uses configured default if omitted.",
            },
            "label": {
                "type": "STRING",
                "description": "Optional focus session label (e.g. 'coding', 'study', 'break').",
            },
        },
        "required": ["action"],
    },
}

PLUGIN_SETTINGS = {
    "namespace": "focus_timer",
    "title": "Focus Timer",
    "fields": [
        {"key": "focus_minutes", "label": "Default focus (min)", "type": "text", "default": "25"},
        {"key": "break_minutes", "label": "Default break (min)", "type": "text", "default": "5"},
    ],
}

_LOCK = threading.Lock()
_ACTIVE: dict | None = None
_CANCEL_EVENT: threading.Event | None = None


def _default_minutes(label: str) -> int:
    cfg = get_plugin_config("focus_timer")
    key = "break_minutes" if "break" in label.lower() else "focus_minutes"
    fallback = 5 if key == "break_minutes" else 25
    try:
        return max(1, min(180, int(cfg.get(key, fallback))))
    except (TypeError, ValueError):
        return fallback


def _timer_worker(duration_s: float, label: str, cancel_ev: threading.Event, player) -> None:
    global _ACTIVE, _CANCEL_EVENT
    finished = not cancel_ev.wait(timeout=duration_s)
    with _LOCK:
        if _CANCEL_EVENT is cancel_ev:
            _ACTIVE = None
            _CANCEL_EVENT = None
    if not finished:
        return
    msg = f"Your {label} timer is complete."
    if player is not None:
        try:
            player.write_log(f"SYS: ⏱ {msg}")
            req = getattr(player, "request_say", None)
            if callable(req):
                req(f"Tell the user briefly that their {label} timer just finished.")
        except Exception:
            pass


def run(parameters: dict, player=None, session_memory=None) -> str:
    global _ACTIVE, _CANCEL_EVENT
    action = str((parameters or {}).get("action") or "status").strip().lower()
    label = str((parameters or {}).get("label") or "focus").strip() or "focus"
    try:
        if action == "start":
            raw_min = (parameters or {}).get("minutes")
            mins = (
                max(1, min(180, int(raw_min)))
                if raw_min is not None
                else _default_minutes(label)
            )
            with _LOCK:
                if _CANCEL_EVENT is not None:
                    _CANCEL_EVENT.set()
                ev = threading.Event()
                _CANCEL_EVENT = ev
                _ACTIVE = {
                    "label": label,
                    "minutes": mins,
                    "ends_at": time.monotonic() + mins * 60.0,
                }
            t = threading.Thread(
                target=_timer_worker,
                args=(mins * 60.0, label, ev, player),
                daemon=True,
                name="plugin-focus-timer",
            )
            t.start()
            msg = f"Started a {mins}-minute {label} timer."
        elif action == "status":
            with _LOCK:
                cur = dict(_ACTIVE) if _ACTIVE else None
            if not cur:
                msg = "No focus timer is currently running."
            else:
                rem = max(0, int(cur["ends_at"] - time.monotonic()))
                m, s = divmod(rem, 60)
                msg = f"{cur['label'].capitalize()} timer: {m}m {s}s remaining."
        elif action == "stop":
            with _LOCK:
                was = _ACTIVE is not None
                if _CANCEL_EVENT is not None:
                    _CANCEL_EVENT.set()
                _ACTIVE = None
                _CANCEL_EVENT = None
            msg = "Focus timer stopped." if was else "No focus timer was running."
        else:
            msg = "Use start, status, or stop for focus_timer."
    except Exception as e:
        return f"Focus timer failed: {e}"

    if player is not None:
        try:
            player.write_log(f"JARVIS: {msg}")
        except Exception:
            pass
    return msg

