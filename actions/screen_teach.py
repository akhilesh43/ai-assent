"""
actions/screen_teach.py — "dekho, aise karna hai" → JARVIS learns a click sequence, then repeats it.

Voice examples
    "seekho: Chrome band karna"  … you click …  "bas"            (teach session)
    "Chrome band karo"                                           (replay)
    "live seekhna chalu karo"  … work normally …                 (live learning, opt-in)
    "jo abhi kiya wo yaad rakh lo: Chrome band karna"            (save what it just watched)
    "live seekhna band karo"                                     (stop and WIPE the buffer)

The mechanics live in core/screen_skills.py; this file is the thin, voice-facing layer:
session state, Gemini helpers (label a button / find it by label), Firebase mirroring.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np

from core import screen_skills as ss

_LOCK = threading.RLock()


class _State:
    backend: Optional[ss.Backend] = None
    store: Optional[ss.SkillStore] = None
    teach: Optional[ss.ClickRecorder] = None
    teach_name = ""
    live: Optional[ss.ClickRecorder] = None
    runner: Optional[ss.SkillRunner] = None
    running = False
    hint_timer: Optional[threading.Timer] = None
    hinted_for = 0.0
    player: Any = None


S = _State()


def _backend() -> ss.Backend:
    if S.backend is None:
        S.backend = ss.SystemBackend()
    return S.backend


def _store() -> ss.SkillStore:
    if S.store is None:
        S.store = ss.SkillStore()
    return S.store


# ── settings ────────────────────────────────────────────────────────────────
def _cfg() -> dict:
    try:
        from memory.config_manager import load_api_keys
        c = load_api_keys().get("screen_skills")
        return c if isinstance(c, dict) else {}
    except Exception:
        return {}


def _cloud_assist() -> bool:
    """Labels and the find-by-label fallback use Gemini (sends a small picture). Off with
    config/api_keys.json -> "screen_skills": {"cloud_assist": false}."""
    return bool(_cfg().get("cloud_assist", True))


# ── Gemini helpers (best effort; every failure returns "nothing") ───────────
def _gemini_label(png: bytes) -> str:
    if not _cloud_assist():
        return ""
    try:
        from google.genai import types
        from core import gemini as G
        out = G.text(
            [types.Part.from_bytes(data=png, mime_type="image/png"),
             "This is a small crop of a screen; the user clicked its CENTRE. Name what was "
             "clicked in 2 to 6 words (e.g. 'Close window button', 'File menu', 'Send button'). "
             "Reply with ONLY the name."],
            tier=G.SMART, timeout_ms=15_000)
        return (out or "").strip().strip('"\'.')[:60]
    except Exception:
        return ""


def _gemini_locate(region: np.ndarray, label: str, title: str) -> Optional[tuple]:
    if not _cloud_assist():
        return None
    try:
        from io import BytesIO
        from PIL import Image
        from google.genai import types
        from core import gemini as G
        img = Image.fromarray(region.astype(np.uint8))
        if img.width > 1280:
            img = img.resize((1280, max(1, int(img.height * 1280 / img.width))))
        buf = BytesIO(); img.save(buf, format="PNG")
        got = G.as_json(
            [types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
             f"This is a screenshot of the window '{title}'. Find the UI element called "
             f"'{label}'. Reply ONLY as JSON: {{\"found\": true, \"x\": <0..1 from left>, "
             f"\"y\": <0..1 from top>}} for the CENTRE of that element, or {{\"found\": false}} "
             f"if it is not clearly visible."],
            tier=G.SMART, timeout_ms=20_000, default=None)
        if isinstance(got, dict) and got.get("found") is True:
            return float(got["x"]), float(got["y"])
    except Exception:
        pass
    return None


# ── feedback + Firebase ─────────────────────────────────────────────────────
def _say(text: str) -> None:
    p = S.player
    if p is not None and hasattr(p, "push_feedback"):
        try:
            p.push_feedback(text)
        except Exception:
            pass


def _mirror(slug: str, skill: dict) -> None:
    """Text only (names, windows, positions) to Firebase + the trained-command table.
    The pictures never leave this PC."""
    def _go():
        try:
            from memory import firebase_sync
            clean = {k: v for k, v in skill.items() if k != "steps"}
            clean["steps"] = [{k: v for k, v in st.items() if k not in ("tight", "wide")}
                              for st in skill.get("steps", [])]
            clean["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            if firebase_sync.is_configured():
                firebase_sync.rtdb_put(f"jarvis_ui_skills/{slug}", clean)
            firebase_sync.train_command(
                skill["name"], f"screen_teach mode=run name={skill['name']}",
                aliases=list(skill.get("aliases", [])), notes="Learned screen skill")
        except Exception as e:
            print(f"[ScreenSkills] mirror skipped: {e}")
    threading.Thread(target=_go, daemon=True, name="SkillMirror").start()


def pull_remote_skills(timeout: float = 3.0) -> int:
    """Bring skills taught on another PC into this one. They arrive WITHOUT pictures, so
    they replay through the label/position fallbacks until re-taught here."""
    try:
        from memory import firebase_sync
        if not firebase_sync.is_configured():
            return 0
        data = firebase_sync.rtdb_get("jarvis_ui_skills", timeout=timeout)
        if not isinstance(data, dict):
            return 0
        store, n = _store(), 0
        have = store.all()
        for slug, sk in data.items():
            if slug in have or not isinstance(sk, dict) or not sk.get("name"):
                continue
            steps = sk.get("steps") or []
            if isinstance(steps, dict):
                steps = [steps[k] for k in sorted(steps, key=lambda z: int(z) if str(z).isdigit() else 0)]
            sk["steps"] = [s for s in steps if isinstance(s, dict)]
            sk["remote"] = True
            store.put(slug, sk)
            n += 1
        return n
    except Exception:
        return 0


# ── teach session ───────────────────────────────────────────────────────────
def _start_teach(name: str) -> str:
    if S.teach is not None and S.teach.running:
        return f"Already learning '{S.teach_name}'. Click what you want, then say 'bas' (or 'cancel')."
    if S.running:
        return "I'm in the middle of running a skill. Wait for it to finish first."
    if not name.strip():
        return "What should I call it? Say for example: seekho, Chrome band karna."
    try:
        rec = ss.ClickRecorder(_backend(), idle_timeout=ss.TEACH_IDLE_TIMEOUT,
                               on_timeout=lambda: _finish_teach(auto=True))
        rec.start()
    except Exception as e:
        return f"I can't watch the mouse here: {e}"
    S.teach, S.teach_name = rec, name.strip()
    _say(f"🎓 SEEKH RAHA HU · {name[:20]}")
    return (f"Okay, I'm watching. Do the clicks for '{name.strip()}' now — I only record clicks, "
            f"nothing else. Say 'bas' when you're done, or 'cancel'.")


def _finish_teach(auto: bool = False, cancel: bool = False) -> str:
    with _LOCK:
        rec, name = S.teach, S.teach_name
        if rec is None:
            return "I'm not learning anything right now."
        rec.stop()
        S.teach, S.teach_name = None, ""
        events = list(rec.events)
    if cancel or not events:
        _say("🎓 CANCELLED" if cancel else "🎓 NO CLICKS")
        return ("Cancelled, nothing saved." if cancel else
                "I didn't see any clicks (clicks on my own window don't count), so nothing was saved.")
    slug, skill = ss.save_skill(_store(), name, events, labeler=_gemini_label)
    _mirror(slug, skill)
    _say(f"✓ SKILL SAVED · {name[:18]}")
    labels = [s.get("label") for s in skill["steps"] if s.get("label")]
    extra = f" Steps: {', '.join(labels)}." if labels else ""
    prefix = "No clicks for a minute, so I stopped and saved it. " if auto else ""
    return (f"{prefix}Learned '{skill['name']}' — {len(skill['steps'])} click(s).{extra} "
            f"Say '{skill['name']}' any time and I'll do it.")


# ── live learning ───────────────────────────────────────────────────────────
def _live_hint(ev: ss.ClickEvent) -> None:
    """After a quiet moment, tell the user (silently, on screen) that a sequence was seen."""
    with _LOCK:
        if S.hint_timer is not None:
            S.hint_timer.cancel()

        def _check():
            live = S.live
            if live is None or not live.running:
                return
            b = live.bursts()
            if b and len(b[-1]) >= 3 and b[-1][-1].t != S.hinted_for:
                S.hinted_for = b[-1][-1].t
                _say(f"👁 {len(b[-1])} clicks dekhe — 'ye yaad rakh lo: <naam>' bolo")
        S.hint_timer = threading.Timer(ss.BURST_GAP + 1.0, _check)
        S.hint_timer.daemon = True
        S.hint_timer.start()


def _watch_on() -> str:
    with _LOCK:
        if S.live is not None and S.live.running:
            return "Live learning is already on."
        try:
            rec = ss.ClickRecorder(_backend(), keep=ss.LIVE_MAX_EVENTS, max_age=ss.LIVE_MAX_AGE,
                                   on_event=_live_hint)
            rec.start()
        except Exception as e:
            return f"I can't watch the mouse here: {e}"
        S.live = rec
    _say("👁 LIVE LEARNING ON")
    return ("Live learning is ON. I'm remembering your last clicks (in memory only — nothing is "
            "saved to disk, and clicks on my own window are ignored). After you do something "
            "worth keeping, say 'ye yaad rakh lo: <name>'. Say 'live seekhna band karo' to stop and wipe it.")


def _watch_off() -> str:
    with _LOCK:
        live = S.live
        S.live = None
        if S.hint_timer is not None:
            S.hint_timer.cancel()
    if live is None:
        return "Live learning wasn't on."
    n = len(live.events)
    live.stop(); live.clear()
    _say("👁 LIVE LEARNING OFF")
    return f"Live learning is off and I wiped what I had remembered ({n} click(s))."


def _save_recent(name: str, count: int = 0) -> str:
    live = S.live
    if live is None or not live.running:
        return "Live learning isn't on, so I haven't been watching. Say 'live seekhna chalu karo' first, or use 'seekho'."
    if not name.strip():
        return "What should I call it? Say for example: ye yaad rakh lo, Chrome band karna."
    if count and count > 0:
        events = list(live.events)[-int(count):]
    else:
        bursts = live.bursts()
        events = bursts[-1] if bursts else []
    if not events:
        return "I haven't seen any clicks yet."
    slug, skill = ss.save_skill(_store(), name, events, labeler=_gemini_label)
    _mirror(slug, skill)
    _say(f"✓ SKILL SAVED · {name[:18]}")
    return (f"Saved '{skill['name']}' from the {len(skill['steps'])} click(s) I just watched. "
            f"Say '{skill['name']}' any time and I'll repeat it.")


# ── run / list / forget / add_step ──────────────────────────────────────────
def _run(name: str, speed: float = 1.0) -> str:
    if S.teach is not None and S.teach.running:
        return "I'm learning something right now — finish with 'bas' first."
    with _LOCK:
        if S.running:
            return "I'm already running a skill."
        S.running = True
    try:
        slug, cands = _store().resolve(name)
        if slug is None:
            skills = _store().all()
            if not skills:
                return "I haven't learned any screen skills yet. Say 'seekho: <name>' and show me."
            if cands:
                opts = ", ".join(skills[c]["name"] for c in cands)
                return f"Which one did you mean: {opts}?"
            return f"I don't know a skill called '{name}'. I know: " + ", ".join(
                s["name"] for s in list(skills.values())[:12]) + "."
        runner = ss.SkillRunner(_backend(), _store(), locator=_gemini_locate,
                                speed=speed, progress=lambda m: _say(f"▶ {m}"))
        S.runner = runner
        _say(f"▶ {slug.replace('_', ' ')}")
        res = runner.run(slug)
        _say("✓ DONE" if res.ok else "⚠ STOPPED")
        return res.message
    except Exception as e:
        return f"Something went wrong while running it: {e}"
    finally:
        S.runner = None
        S.running = False


def _list(player=None) -> str:
    skills = _store().all()
    if not skills:
        return "No screen skills yet. Say 'seekho: <name>' and show me."
    lines = []
    for i, (slug, s) in enumerate(skills.items(), 1):
        tag = " (from another PC, no pictures)" if s.get("remote") else ""
        lines.append(f"{i}. {s.get('name', slug)} — {len(s.get('steps', []))} step(s), "
                     f"used {s.get('uses', 0)}x{tag}")
    if player is not None and hasattr(player, "show_content"):
        try:
            player.show_content("SCREEN SKILLS", "\n".join(lines))
        except Exception:
            pass
    return "Screen skills:\n" + "\n".join(lines[:15])


def _forget(name: str) -> str:
    slug, cands = _store().resolve(name)
    if slug is None:
        return f"I don't have a skill called '{name}'."
    ok = _store().forget(slug)
    try:                                           # also drop the cloud copy
        from memory import firebase_sync
        if firebase_sync.is_configured():
            threading.Thread(target=firebase_sync.rtdb_delete, args=(f"jarvis_ui_skills/{slug}",),
                             daemon=True).start()
    except Exception:
        pass
    return f"Forgot '{name}'." if ok else f"I couldn't find '{name}'."


def _add_step(name: str, kind: str, p: dict) -> str:
    slug, _ = _store().resolve(name)
    if slug is None:
        return f"I don't have a skill called '{name}'."
    kind = (kind or "").strip().lower()
    if kind == "hotkey":
        keys = [k.strip().lower() for k in str(p.get("keys", "")).replace("+", ",").split(",") if k.strip()]
        if not keys:
            return "Which keys? For example keys='ctrl,w'."
        step = {"type": "hotkey", "keys": keys, "delay": 0.4}
    elif kind == "type":
        text = str(p.get("text", ""))
        if not text:
            return "What should I type?"
        step = {"type": "type", "text": text, "delay": 0.4}
    elif kind == "wait":
        step = {"type": "wait", "seconds": float(min(30.0, max(0.2, float(p.get("seconds", 1.0))))), "delay": 0.0}
    else:
        return "step_type must be hotkey, type or wait. (Clicks are taught by showing me.)"
    sk = _store().get(slug)
    if len(sk["steps"]) >= ss.MAX_STEPS:
        return f"That skill already has {ss.MAX_STEPS} steps."
    sk["steps"].append(step)
    _store().put(slug, sk)
    _mirror(slug, sk)
    return f"Added a {kind} step to '{sk['name']}' ({len(sk['steps'])} steps now)."


def _status() -> str:
    teach = f"learning '{S.teach_name}'" if S.teach is not None and S.teach.running else "not learning"
    if S.live is not None and S.live.running:
        b = S.live.bursts()
        live = f"ON ({len(S.live.events)} click(s) remembered, {len(b)} sequence(s))"
    else:
        live = "off"
    return f"Teach: {teach}. Live learning: {live}. Skills known: {len(_store().all())}."


def abort_running() -> None:
    """Called when the user says stop / presses interrupt while a skill is running."""
    r = S.runner
    if r is not None:
        r.abort()


# ── tool entry point ────────────────────────────────────────────────────────
def screen_teach_action(parameters: dict = None, response=None, player=None,
                        session_memory=None) -> str:
    p = parameters or {}
    S.player = player
    mode = str(p.get("mode") or "list").strip().lower()
    name = str(p.get("name") or "").strip()
    try:
        if mode in ("teach", "start", "learn"):
            return _start_teach(name)
        if mode in ("stop", "finish", "done", "save"):
            return _finish_teach()
        if mode == "cancel":
            return _finish_teach(cancel=True)
        if mode in ("run", "do", "play"):
            return _run(name, float(p.get("speed") or 1.0))
        if mode == "list":
            return _list(player)
        if mode in ("forget", "delete"):
            return _forget(name)
        if mode == "add_step":
            return _add_step(name, str(p.get("step_type") or ""), p)
        if mode in ("watch_on", "live_on"):
            return _watch_on()
        if mode in ("watch_off", "live_off"):
            return _watch_off()
        if mode in ("save_recent", "remember_recent"):
            return _save_recent(name, int(p.get("count") or 0))
        if mode == "status":
            return _status()
        if mode == "abort":
            abort_running()
            return "Stopping."
        return f"Unknown mode '{mode}'."
    except Exception as e:
        return f"Screen skills failed: {e}"


TOOL = {
    "name": "screen_teach",
    "description": (
        "Teach and replay CLICK SEQUENCES on this PC's screen. "
        "mode='teach' + name: user says 'seekho / learn: <task>' — then THEY click the buttons "
        "and finally say 'bas / ho gaya / done' → call mode='stop'. 'cancel' to discard. "
        "mode='run' + name: user asks for something they taught before ('Chrome band karo', "
        "'jo sikhaya tha wo karo') — check [LEARNED SCREEN SKILLS] in your context and run it, "
        "do NOT ask them to explain again. "
        "mode='watch_on' ('live seekhna chalu karo', 'meri screen dekh kar seekho'): keep a short "
        "in-memory record of the user's recent clicks; afterwards mode='save_recent' + name "
        "('jo abhi kiya wo yaad rakh lo: <name>') turns the last thing they did into a skill; "
        "mode='watch_off' stops it and wipes the memory. "
        "Also: list, forget, status, abort, and add_step (step_type hotkey/type/wait). "
        "It only records mouse clicks the user makes — never keystrokes or screenshots in the "
        "background — and never clicks on its own window. If a replay stops, tell the user "
        "which step failed; do not retry blindly. "
        "NOT for closing the inbuilt browser (use inbuilt_browser action='close') and NOT for "
        "one-off clicks the user dictates (use computer_control)."
    ),
    "brief": "Teach JARVIS click sequences by showing them once, or by live learning, then replay them.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "mode": {"type": "STRING",
                     "enum": ["teach", "stop", "cancel", "run", "list", "forget", "add_step",
                              "watch_on", "watch_off", "save_recent", "status", "abort"],
                     "description": "What to do (see the tool description)."},
            "name": {"type": "STRING", "description": "Skill name, e.g. 'Chrome band karna'."},
            "count": {"type": "INTEGER", "description": "save_recent: how many recent clicks (default: the last sequence)."},
            "step_type": {"type": "STRING", "enum": ["hotkey", "type", "wait"], "description": "add_step: kind of step."},
            "keys": {"type": "STRING", "description": "add_step hotkey: e.g. 'ctrl,w'."},
            "text": {"type": "STRING", "description": "add_step type: text to type."},
            "seconds": {"type": "NUMBER", "description": "add_step wait: seconds."},
            "speed": {"type": "NUMBER", "description": "run: 1 = recorded pace, 2 = twice as fast."},
        },
        "required": ["mode"],
    },
    "handler": screen_teach_action,
}
