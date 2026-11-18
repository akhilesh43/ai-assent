"""
core/phone_bridge.py — the laptop's half of the phone link.

The phone app (Flutter UI + a small native service) talks to this module through the dashboard
server by plain HTTP: it POSTs what happens on the phone (a notification arrived, a call is
ringing) and long-polls for things the laptop wants done (answer the call, reply to that chat,
send an SMS). No Qt and no web framework in here, so all of it is unit-tested without either.

                    phone                                    laptop
        notification / call / battery  ── event ──▶  PhoneBridge.push_event()
                                                      └─ ReplyEngine decides: ignore / draft /
                                                         ask the owner / answer on its own
        execute command, report result ◀── poll ───  PhoneBridge.send_command()

WHAT IT WILL AND WILL NOT DO WITHOUT ASKING
    Replies to a message are drafted in the owner's own style (memory/reply_store.py). The default
    for every contact is ASK: the draft waits for one tap (phone app or HUD). Only a contact set to
    `auto` — or the global mode set to auto — is answered unattended, and then only when the
    message is a plain one-to-one chat. Group chats, ongoing notifications and anything the model
    marks as needing the owner's real decision (money, commitments, anything private) are never
    answered on their own. Answers are rate-limited per conversation so two auto-replying
    parties cannot loop forever.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Optional

from memory import reply_store

ONLINE_WINDOW_S = 60
PENDING_TTL_S = 3600
MIN_GAP_S = 20                 # between two automatic replies in one conversation
MAX_AUTO_PER_HOUR = 20
NEEDS_OWNER = "[NEEDS_OWNER]"

MESSAGING_PACKAGES = {
    "com.whatsapp", "com.whatsapp.w4b", "org.telegram.messenger", "org.telegram.plus",
    "com.google.android.apps.messaging", "com.samsung.android.messaging", "com.android.mms",
    "com.instagram.android", "org.thoughtcrime.securesms", "com.facebook.orca", "com.discord",
    "com.microsoft.teams", "com.Slack", "com.slack.android",
}

# What the phone is able to do (the native side implements exactly these).
ACTIONS = {
    "answer_call", "end_call", "place_call", "send_sms", "reply_notification", "dismiss_notification",
    "read_notifications", "open_app", "flashlight", "volume", "find_phone", "stop_find_phone",
    "battery", "toast", "lookup_contact", "open_url",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Drafting a reply in the owner's voice
# ─────────────────────────────────────────────────────────────────────────────
def build_reply_prompt(incoming: str, contact: str, app: str, examples: list[dict],
                       owner: str = "", persona: str = "") -> str:
    shots = "\n\n".join(
        f"[{'to ' + e['contact'] if e['contact'] else 'general'}] They: {e['incoming']}\nOwner: {e['reply']}"
        for e in examples) or "(no examples yet — be brief, natural and friendly)"
    return (
        f"You are drafting a chat reply on behalf of {owner or 'the phone owner'}, in THEIR voice.\n"
        "Study the examples: copy their language mix (Hindi / English / Hinglish), spelling habits, "
        "length, emoji use and level of formality. Never sound like an assistant. Do not mention AI.\n"
        + (f"Owner's note about their style: {persona}\n" if persona else "")
        + f"\nEXAMPLES OF HOW THE OWNER ACTUALLY REPLIES:\n{shots}\n\n"
        f"NOW REPLY TO THIS ({app or 'chat'}, from {contact or 'someone'}):\n{incoming}\n\n"
        "Rules: output ONLY the reply text, nothing else. Keep it short. If answering needs the owner's own "
        "decision or private knowledge — money, a time or commitment, an address, passwords, anything "
        f"sensitive, an emergency, bad news — output exactly {NEEDS_OWNER} and nothing else."
    )


def clean_draft(text: str) -> Optional[str]:
    t = (text or "").strip()
    t = re.sub(r"^(owner|reply|me)\s*:\s*", "", t, flags=re.I).strip().strip('"').strip()
    if not t or NEEDS_OWNER in t or len(t) > 600:
        return None
    return t


def _default_llm(prompt: str) -> str:
    from core import gemini
    return gemini.text(prompt, tier=gemini.FAST, timeout_ms=20_000)


# ─────────────────────────────────────────────────────────────────────────────
#  The bridge
# ─────────────────────────────────────────────────────────────────────────────
class PhoneBridge:
    def __init__(self, llm: Optional[Callable[[str], str]] = None, clock: Callable[[], float] = time.time,
                 owner: str = ""):
        self._llm = llm or _default_llm
        self._clock = clock
        self.owner = owner
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._devices: dict[str, dict[str, Any]] = {}
        self._cmds: dict[str, deque] = {}
        self._waiters: dict[str, dict[str, Any]] = {}
        self.notifications: deque = deque(maxlen=100)
        self.pending: dict[str, dict[str, Any]] = {}
        self._last_reply_at: dict[str, float] = {}
        self._auto_sent: deque = deque(maxlen=MAX_AUTO_PER_HOUR)
        self.call: dict[str, Any] = {"state": "idle"}
        self.alert_handler: Optional[Callable[[str], None]] = None
        self.log: deque = deque(maxlen=60)
        # Tests replace this with a direct call so they need no sleeping.
        self._spawn: Callable[..., Any] = lambda fn, *a: threading.Thread(
            target=fn, args=a, daemon=True, name="PhoneReply").start()

    # ── settings ─────────────────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        m = reply_store.get_setting("reply_mode", "ask")
        return m if m in reply_store.MODES else "ask"

    def set_mode(self, mode: str) -> str:
        if mode not in reply_store.MODES:
            raise ValueError(f"mode must be one of {', '.join(reply_store.MODES)}")
        reply_store.set_setting("reply_mode", mode)
        self._note(f"Reply mode: {mode}")
        return mode

    def _note(self, text: str) -> None:
        self.log.appendleft({"t": self._clock(), "text": text})

    def _alert(self, text: str) -> None:
        self._note(text)
        fn = self.alert_handler
        if fn:
            try:
                fn(text)
            except Exception:
                pass

    # ── devices ──────────────────────────────────────────────────────────────
    def register(self, device_id: str, name: str = "", info: Optional[dict] = None) -> None:
        with self._lock:
            d = self._devices.setdefault(device_id, {"id": device_id, "first_seen": self._clock()})
            was_online = self._online(d)
            d.update(name=name or d.get("name", "Phone"), last_seen=self._clock())
            if info:
                d.setdefault("info", {}).update(info)
            self._cmds.setdefault(device_id, deque())
        if not was_online:
            self._note(f"Phone connected: {d['name']}")

    def _online(self, d: dict) -> bool:
        return self._clock() - d.get("last_seen", 0) < ONLINE_WINDOW_S

    def online_devices(self) -> list[dict]:
        with self._lock:
            return [d for d in self._devices.values() if self._online(d)]

    def connected(self) -> bool:
        return bool(self.online_devices())

    def _pick(self, device: str = "") -> Optional[str]:
        with self._lock:
            if device and device in self._devices:
                return device
            online = sorted(self.online_devices(), key=lambda d: -d.get("last_seen", 0))
            return online[0]["id"] if online else None

    # ── laptop → phone ───────────────────────────────────────────────────────
    def queue_command(self, action: str, args: Optional[dict] = None, device: str = "") -> Optional[str]:
        """Put a command in the phone's queue and return its id, or None if no phone is connected."""
        if action not in ACTIONS:
            raise ValueError(f"unknown phone action {action!r}")
        dev = self._pick(device)
        if dev is None:
            return None
        rid = uuid.uuid4().hex[:10]
        with self._cond:
            self._cmds[dev].append({"id": rid, "action": action, "args": args or {}})
            self._waiters[rid] = {"event": threading.Event(), "result": None}
            self._cond.notify_all()
        return rid

    def send_command(self, action: str, args: Optional[dict] = None, *, device: str = "",
                     timeout: float = 15.0) -> dict[str, Any]:
        """Run something on the phone and wait for the answer. Never raises."""
        try:
            rid = self.queue_command(action, args, device)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if rid is None:
            return {"ok": False, "error": "no phone is connected"}
        w = self._waiters[rid]
        if not w["event"].wait(timeout):
            with self._lock:
                self._waiters.pop(rid, None)
            return {"ok": False, "error": "the phone did not answer in time"}
        with self._lock:
            self._waiters.pop(rid, None)
        return w["result"] or {"ok": False, "error": "empty result"}

    # ── phone → laptop ───────────────────────────────────────────────────────
    def poll(self, device_id: str, wait: float = 25.0, name: str = "") -> list[dict]:
        """Long-poll: block up to `wait` seconds for commands, return the ones queued."""
        self.register(device_id, name)
        end = self._clock() + max(0.0, wait)
        with self._cond:
            while not self._cmds[device_id]:
                left = end - self._clock()
                if left <= 0:
                    break
                self._cond.wait(timeout=min(left, 1.0))
            out = list(self._cmds[device_id])
            self._cmds[device_id].clear()
        with self._lock:
            self._devices[device_id]["last_seen"] = self._clock()
        return out

    def submit_result(self, device_id: str, req_id: str, ok: bool, data: Any = None, error: str = "") -> None:
        self.register(device_id)
        with self._lock:
            w = self._waiters.get(req_id)
        if w:
            w["result"] = {"ok": bool(ok), "data": data, "error": error}
            w["event"].set()

    def push_event(self, device_id: str, event: dict) -> None:
        """Entry point for everything the phone reports."""
        self.register(device_id)
        kind = str(event.get("type") or "")
        try:
            if kind == "notification":
                self._on_notification(device_id, event)
            elif kind == "call":
                self._on_call(event)
            elif kind == "battery":
                with self._lock:
                    self._devices[device_id].setdefault("info", {}).update(
                        battery=event.get("level"), charging=bool(event.get("charging")))
            elif kind == "removed":
                pass
        except Exception as e:                                   # one bad event must not stop the link
            self._note(f"event error: {type(e).__name__}: {e}")

    # ── calls ────────────────────────────────────────────────────────────────
    def _auto_decline_call(self, number: str, message: str, who: str) -> None:
        try:
            import time
            time.sleep(1.0)
            self.send_command("end_call")
            if number:
                self.send_command("send_sms", {"number": number, "message": message})
            self._alert(f"Call from {who} cut kar di gayi aur SMS bhej diya: '{message}'")
        except Exception as e:
            self._note(f"Auto decline call failed: {e}")

    def _on_call(self, ev: dict) -> None:
        state = str(ev.get("state") or "idle")
        who = str(ev.get("name") or ev.get("number") or "an unknown number")
        number = str(ev.get("number") or "")
        prev = self.call.get("state")
        self.call = {"state": state, "who": who, "number": number, "since": self._clock()}
        if state == "ringing" and prev != "ringing":
            auto_cut = str(reply_store.get_setting("auto_decline_calls", "false") or "").lower() in ("true", "1", "yes")
            if auto_cut:
                msg = str(reply_store.get_setting("auto_decline_message", "Main abhi thoda busy hu, kuch der me call karta hu.") or "").strip()
                self._note(f"Auto-declining call from {who} with message: {msg}")
                self._spawn(self._auto_decline_call, number, msg, who)
                return
            self._alert(f"Incoming call from {who}. Say 'pick up' to answer or 'cut it' to decline.")
        elif state == "idle" and prev == "ringing":
            self._note(f"Call from {who} ended or was missed")

    # ── notifications and replies ────────────────────────────────────────────
    def _on_notification(self, device_id: str, ev: dict) -> None:
        item = {
            "key": str(ev.get("key") or ""), "app": str(ev.get("app") or ev.get("package") or ""),
            "package": str(ev.get("package") or ""), "title": str(ev.get("title") or "")[:120],
            "text": str(ev.get("text") or "")[:600], "sender": str(ev.get("sender") or ""),
            "can_reply": bool(ev.get("can_reply")), "is_group": bool(ev.get("is_group")),
            "ongoing": bool(ev.get("ongoing")), "t": self._clock(), "device": device_id,
            "category": str(ev.get("category") or ""),
        }
        with self._lock:
            if any(n["key"] == item["key"] and n["text"] == item["text"] for n in list(self.notifications)[:5]):
                return                                          # the same notification re-posted
            self.notifications.appendleft(item)
        # Off the request thread: drafting calls a model and sending waits for the phone, and the
        # phone is waiting for THIS request to return before it polls for that very command.
        self._spawn(self._consider_reply, item)

    def _is_candidate(self, n: dict) -> bool:
        if not n["can_reply"] or n["ongoing"] or not n["text"].strip():
            return False
        return n["package"] in MESSAGING_PACKAGES or n.get("category") == "msg"

    def _consider_reply(self, n: dict) -> None:
        if not self._is_candidate(n):
            return
        contact = n["sender"] or n["title"]
        rule = reply_store.get_rule(contact)
        mode = rule["mode"] if rule else self.mode
        if mode in ("off", "ignore"):
            return
        if n["is_group"] and not (rule and rule["mode"] == "auto"):
            return                                              # groups only when explicitly allowed
        now = self._clock()
        if now - self._last_reply_at.get(n["key"], 0) < MIN_GAP_S:
            return
        draft = self.draft(n["text"], contact, n["app"])
        if mode == "auto" and draft and self._auto_allowed():
            self._send_reply(n, draft, source="auto")
            return
        pid = uuid.uuid4().hex[:8]
        with self._lock:
            self.pending[pid] = {"id": pid, "contact": contact, "app": n["app"], "package": n["package"],
                                 "incoming": n["text"], "draft": draft or "", "key": n["key"],
                                 "device": n["device"], "created": now,
                                 "needs_owner": draft is None}
        self._alert(f"New message from {contact}: “{n['text'][:80]}”. "
                    + (f"Draft ready: “{draft}”." if draft else "This one needs your own answer."))

    def _auto_allowed(self) -> bool:
        now = self._clock()
        while self._auto_sent and now - self._auto_sent[0] > 3600:
            self._auto_sent.popleft()
        return len(self._auto_sent) < MAX_AUTO_PER_HOUR

    def draft(self, incoming: str, contact: str = "", app: str = "") -> Optional[str]:
        """A reply in the owner's voice, or None when the model says it needs the owner (or is down)."""
        try:
            prompt = build_reply_prompt(incoming, contact, app, reply_store.examples_for(contact, 10),
                                        owner=self.owner, persona=reply_store.get_setting("persona", ""))
            return clean_draft(self._llm(prompt))
        except Exception as e:
            self._note(f"draft failed: {type(e).__name__}")
            return None

    def _send_reply(self, n: dict, text: str, *, source: str) -> dict:
        res = self.send_command("reply_notification", {"key": n["key"], "text": text}, device=n.get("device", ""))
        if res.get("ok"):
            self._last_reply_at[n["key"]] = self._clock()
            if source == "auto":
                self._auto_sent.append(self._clock())
            self._note(f"Replied to {n.get('contact') or n.get('sender') or n.get('title')}: {text[:60]}")
        else:
            self._note(f"Reply failed: {res.get('error')}")
        return res

    # ── the owner's decisions ────────────────────────────────────────────────
    def pending_list(self) -> list[dict]:
        now = self._clock()
        with self._lock:
            for pid in [p for p, v in self.pending.items() if now - v["created"] > PENDING_TTL_S]:
                del self.pending[pid]
            return sorted(self.pending.values(), key=lambda p: -p["created"])

    def approve(self, pid: str, text: Optional[str] = None) -> dict:
        """Send a pending reply — as drafted, or with the owner's own wording (learned from either way)."""
        with self._lock:
            p = self.pending.get(pid)
        if not p:
            return {"ok": False, "error": "that message is no longer waiting"}
        final = (text if text is not None else p["draft"]).strip()
        if not final:
            return {"ok": False, "error": "nothing to send — type your reply"}
        res = self._send_reply({"key": p["key"], "device": p["device"], "sender": p["contact"], "title": p["contact"]},
                               final, source="owner")
        if res.get("ok"):
            with self._lock:
                self.pending.pop(pid, None)
            source = "approved" if final == p["draft"] else "edited"
            reply_store.add_example(p["incoming"], final, contact=p["contact"], app=p["app"], source=source)
        return res

    def reject(self, pid: str) -> bool:
        with self._lock:
            return self.pending.pop(pid, None) is not None

    def teach(self, incoming: str, reply: str, contact: str = "") -> bool:
        ok = reply_store.add_example(incoming, reply, contact=contact, source="taught")
        if ok:
            self._note(f"Learned how you answer: “{incoming[:40]}” → “{reply[:40]}”")
        return ok

    # ── for the screens ──────────────────────────────────────────────────────
    def snapshot(self) -> dict[str, Any]:
        online = self.online_devices()
        dev = sorted(online, key=lambda d: -d.get("last_seen", 0))[0] if online else None
        info = (dev or {}).get("info", {})
        return {
            "connected": bool(dev),
            "device": (dev or {}).get("name", ""),
            "battery": info.get("battery"), "charging": info.get("charging"),
            "call": dict(self.call),
            "mode": self.mode,
            "pending": self.pending_list(),
            "notifications": list(self.notifications)[:25],
            "rules": reply_store.list_rules(),
            "learning": reply_store.stats(),
            "log": [{"t": x["t"], "text": x["text"]} for x in list(self.log)[:15]],
        }


_BRIDGE: Optional[PhoneBridge] = None
_BRIDGE_LOCK = threading.Lock()


def get_bridge() -> PhoneBridge:
    """The one bridge for this process (the server, the voice tool and the HUD share it)."""
    global _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            owner = ""
            try:
                from memory import memory_manager as mm
                owner = str(mm._entry_value(mm.load_memory().get("identity", {}).get("name", "")) or "")
            except Exception:
                pass
            _BRIDGE = PhoneBridge(owner=owner)
        return _BRIDGE
