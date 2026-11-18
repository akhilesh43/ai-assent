"""
core/phone_link.py — the phone link, on its own.

This REPLACES the "Remote Control" dashboard for the phone app. The dashboard needed fastapi,
uvicorn, cryptography, a certificate and port 8000; any one of those missing, or the port taken,
and the phone could not connect. This needs NOTHING beyond Python itself: one small HTTP server
on its own port (8765, or the next free one up to 8775), answering only what the phone app asks.

    phone app ──(same Wi-Fi, plain HTTP)──▶  PhoneLinkServer ──▶ PhoneBridge (calls, replies …)
                                                            └──▶ the assistant (typed commands)

PAIRING
    The laptop shows a QR holding   http://<laptop ip>:<port>/pair?key=<one-time key>&alt=<other ips>
    The app scans it, trades the key for a session token plus a long-lived device token (stored
    HASHED in config/phone_devices.json) — so the laptop can restart without scanning again.
    A key works once and expires after 10 minutes; five wrong keys in a minute lock that address out.

NO LOGIN-FREE DOORS except /api/ping, which only says "JARVIS is here" so the app can find the laptop.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

from core import phone_devices
from core.phone_bridge import PhoneBridge, get_bridge

BUILD = "Phone Link v2"
PORTS = tuple(range(8765, 8776))
TOKEN_TTL_S = 24 * 3600
KEY_TTL_S = 600
MAX_BODY = 64 * 1024
# Unambiguous characters only: no 0/O, 1/I/L — the key is sometimes read aloud or typed.
_KEY_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


# ─────────────────────────────────────────────────────────────────────────────
#  Finding this computer's addresses
# ─────────────────────────────────────────────────────────────────────────────
def _route_ip() -> str:
    for probe in ("8.8.8.8", "1.1.1.1", "192.168.1.1"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect((probe, 80))
            ip = s.getsockname()[0]
            s.close()
            if not ip.startswith("127."):
                return ip
        except Exception:
            pass
    return ""


def lan_ips() -> list[str]:
    """Every private IPv4 address of this computer, Wi-Fi-looking ones first. A VPN, WSL or VirtualBox
    adds extra ones; the phone app tries each, so a wrong first guess no longer matters."""
    found: list[str] = []

    def add(ip: str) -> None:
        p = ip.split(".")
        if len(p) != 4 or ip in found:
            return
        try:
            a, b = int(p[0]), int(p[1])
        except ValueError:
            return
        if a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168):
            found.append(ip)

    add(_route_ip())
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except Exception:
        pass
    found.sort(key=lambda ip: (not ip.startswith("192.168."), found.index(ip)))
    return found


# ─────────────────────────────────────────────────────────────────────────────
#  Windows Firewall
# ─────────────────────────────────────────────────────────────────────────────
FIREWALL_RULE = "JARVIS phone"


def windows_firewall_ok() -> bool:
    """True when the inbound rule exists (or this is not Windows). Querying needs no administrator."""
    import subprocess
    import sys
    if not sys.platform.startswith("win"):
        return True
    try:
        r = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule", f"name={FIREWALL_RULE}"],
                           capture_output=True, text=True, timeout=8,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode == 0 and FIREWALL_RULE.lower() in (r.stdout or "").lower()
    except Exception:
        return True                      # cannot tell: do not nag


def run_firewall_fix(bat_path: str) -> bool:
    """Run fix_phone_firewall.bat elevated (Windows shows its one "Do you want to allow…" prompt)."""
    import sys
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        return int(ctypes.windll.shell32.ShellExecuteW(None, "runas", bat_path, None, None, 1)) > 32  # type: ignore[attr-defined]
    except Exception:
        return False


_ACTIVE: Optional["PhoneLinkServer"] = None


def get_active() -> Optional["PhoneLinkServer"]:
    """The running server of this process (the PHONE LINK window asks it who has been in touch)."""
    return _ACTIVE


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # Windows lets a second socket bind a port that is already in use when this is True, which made
    # "port busy" invisible. Off, a busy port raises and we move to the next one.
    allow_reuse_address = False


# ─────────────────────────────────────────────────────────────────────────────
#  The server
# ─────────────────────────────────────────────────────────────────────────────
class PhoneLinkServer:
    def __init__(self, bridge: Optional[PhoneBridge] = None, clock: Callable[[], float] = time.time,
                 devices_path=None, ips: Optional[Callable[[], list[str]]] = None):
        global _ACTIVE
        _ACTIVE = self
        self.bridge = bridge or get_bridge()
        self._clock = clock
        self._devices_path = devices_path
        self._ips = ips or lan_ips
        self._keys: dict[str, float] = {}
        self._tokens: dict[str, float] = {}
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.RLock()
        self._log: deque = deque(maxlen=300)
        self._seq = 0
        self.status = "idle"
        self.port: Optional[int] = None
        self.bind_error = ""
        # The assistant side reads these (same names the old dashboard used, so main.py's relay is unchanged).
        self._command_queue: asyncio.Queue = asyncio.Queue()
        self._phone_audio_queue: asyncio.Queue = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connect_callback: Optional[Callable[[], None]] = None
        self._httpd: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None
        # "Did the phone reach us at all?" — the one question that separates a network/firewall problem
        # from an app problem. Any request counts, even a browser opening /api/ping.
        self.last_contact: Optional[dict] = None
        self._seen: set[str] = set()
        self.on_contact: Optional[Callable[[str], None]] = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def set_connect_callback(self, fn) -> None:
        self._connect_callback = fn

    def start(self) -> bool:
        """Bind the first free port and serve in a background thread. Never raises."""
        if self._httpd:
            return True
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):          # no console spam
                pass

            def _go(self, method: str) -> None:
                u = urlparse(self.path)
                body: dict = {}
                if method == "POST":
                    n = int(self.headers.get("Content-Length") or 0)
                    if n > MAX_BODY:
                        return self._send(413, {"ok": False, "error": "too large"})
                    try:
                        raw = self.rfile.read(n) if n else b""
                        body = json.loads(raw.decode("utf-8") or "{}")
                        if not isinstance(body, dict):
                            raise ValueError
                    except Exception:
                        return self._send(400, {"ok": False, "error": "bad json"})
                code, data = server.route(method, u.path, {k: v[0] for k, v in parse_qs(u.query).items()},
                                          body, self.headers.get("Authorization", ""), self.client_address[0])
                self._send(code, data)

            def _send(self, code: int, data: Any) -> None:
                raw = json.dumps(data, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except Exception:
                    pass

            def do_GET(self):
                self._go("GET")

            def do_POST(self):
                self._go("POST")

        last = ""
        for p in PORTS:
            try:
                self._httpd = _Server(("0.0.0.0", p), Handler)
                self.port = p
                break
            except OSError as e:
                last = f"{e.__class__.__name__}: {e}"
        if not self._httpd:
            self.bind_error = (f"ports {PORTS[0]}–{PORTS[-1]} are all in use ({last}). "
                               "Close any other copy of JARVIS and restart.")
            print(f"[PhoneLink] {self.bind_error}")
            return False
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="PhoneLink")
        self._thread.start()
        print(f"[PhoneLink] listening on port {self.port}  ({', '.join(self._ips()) or 'no LAN address'})")
        return True

    async def serve(self) -> None:
        """Coroutine form for main.py's asyncio.create_task: remember the loop, start, return."""
        self._loop = asyncio.get_running_loop()
        self.start()

    def stop(self) -> None:
        if self._httpd:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None

    # ── pairing keys ─────────────────────────────────────────────────────────
    def new_key(self, expiry_secs: int = KEY_TTL_S) -> str:
        key = "".join(secrets.choice(_KEY_CHARS) for _ in range(8))
        with self._lock:
            now = self._clock()
            self._keys = {k: t for k, t in self._keys.items() if t > now}
            self._keys[key] = now + expiry_secs
        return key

    def info(self, key: Optional[str] = None) -> dict[str, Any]:
        """Everything the QR window shows."""
        ips = self._ips()
        key = key or self.new_key()
        out: dict[str, Any] = {"ok": bool(self._httpd), "key": key, "port": self.port, "ips": ips,
                               "error": self.bind_error, "expires_in": KEY_TTL_S, "build": BUILD}
        if self._httpd and ips:
            alt = ",".join(ips[1:4])
            out["url"] = f"http://{ips[0]}:{self.port}/pair?key={key}" + (f"&alt={alt}" if alt else "")
            out["manual"] = f"{ips[0]}:{self.port}"
            out["test_url"] = f"http://{ips[0]}:{self.port}/api/ping"
        elif self._httpd:
            out["error"] = "No Wi-Fi / LAN address found — connect this laptop to your Wi-Fi."
        return out

    # ── assistant side ───────────────────────────────────────────────────────
    async def broadcast(self, msg: dict) -> None:
        """Same call the old dashboard took: log lines and status changes, kept for the phone to read."""
        kind = msg.get("type")
        with self._lock:
            if kind == "status":
                self.status = str(msg.get("state") or "")
            elif kind == "log":
                self._seq += 1
                self._log.append({"i": self._seq, "speaker": str(msg.get("speaker") or ""),
                                  "text": str(msg.get("text") or "")[:4000], "ts": msg.get("ts", "")})

    def _submit_command(self, text: str) -> None:
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._command_queue.put_nowait, text)
        else:
            self._command_queue.put_nowait(text)

    # ── auth ─────────────────────────────────────────────────────────────────
    def _authed(self, header: str) -> bool:
        tok = header.removeprefix("Bearer ").strip()
        with self._lock:
            exp = self._tokens.get(tok)
            if exp is None:
                return False
            if exp < self._clock():
                del self._tokens[tok]
                return False
            return True

    def _issue_token(self) -> str:
        tok = secrets.token_urlsafe(32)
        with self._lock:
            now = self._clock()
            self._tokens = {t: e for t, e in self._tokens.items() if e > now}
            self._tokens[tok] = now + TOKEN_TTL_S
        return tok

    def _locked_out(self, who: str) -> float:
        now = self._clock()
        with self._lock:
            recent = [t for t in self._fails.get(who, []) if now - t < 60]
            self._fails[who] = recent
            return (60 - (now - recent[0])) if len(recent) >= 5 else 0.0

    # ── routing (pure: easy to test without sockets) ─────────────────────────
    def contact(self) -> Optional[dict]:
        """{"who": ip, "ago": seconds, "path": ...} for the latest request from anywhere, or None."""
        c = self.last_contact
        if not c:
            return None
        return {"who": c["who"], "path": c["path"], "ago": max(0, int(self._clock() - c["t"]))}

    def _note_contact(self, who: str, path: str) -> None:
        self.last_contact = {"t": self._clock(), "who": who, "path": path}
        if who and who not in self._seen:
            self._seen.add(who)
            cb = self.on_contact
            if cb:
                try:
                    cb(who)
                except Exception:
                    pass

    def route(self, method: str, path: str, q: dict, body: dict, auth: str, who: str = "") -> tuple[int, dict]:
        self._note_contact(who, path)
        try:
            return self._route(method, path, q, body, auth, who)
        except Exception as e:                                   # never let one bad request kill a thread
            return 500, {"ok": False, "error": f"{type(e).__name__}: {e}"[:160]}

    def _route(self, method: str, path: str, q: dict, body: dict, auth: str, who: str) -> tuple[int, dict]:
        b = self.bridge
        if method == "GET" and path == "/api/ping":
            return 200, {"jarvis": True, "link": 1, "build": BUILD, "name": socket.gethostname(), "ips": self._ips()}

        # ── pairing ──────────────────────────────────────────────────────────
        if method == "POST" and path == "/api/app-pair":
            wait = self._locked_out(who)
            if wait > 0:
                return 429, {"ok": False, "error": f"Too many wrong keys. Try again in {int(wait) + 1} seconds."}
            key = str(body.get("key", "")).strip().upper()
            with self._lock:
                valid = self._keys.get(key, 0) > self._clock()
                if valid:
                    del self._keys[key]
                else:
                    self._fails.setdefault(who, []).append(self._clock())
            if not valid:
                return 401, {"ok": False, "error": "That key is wrong or has expired. Press “Phone link” on the laptop for a new one."}
            dev_id, dev_tok = phone_devices.issue(str(body.get("name") or "Phone"), self._devices_path)
            tok = self._issue_token()
            cb = self._connect_callback
            if cb:
                try:
                    cb()
                except Exception:
                    pass
            b.register(dev_id, str(body.get("name") or "Phone"))
            return 200, {"ok": True, "token": tok, "device_token": dev_tok, "device_id": dev_id,
                         "laptop": socket.gethostname()}

        if method == "POST" and path == "/api/app-login":
            rec = phone_devices.verify(str(body.get("device_token") or ""), self._devices_path)
            if not rec:
                return 401, {"ok": False}
            return 200, {"ok": True, "token": self._issue_token(), "device_id": rec["id"]}

        # ── everything else needs a session token ────────────────────────────
        if not self._authed(auth):
            return 401, {"error": "Unauthorized"}

        if method == "GET" and path == "/api/phone/poll":
            try:
                wait = min(max(float(q.get("wait", 25)), 0.0), 30.0)
            except ValueError:
                wait = 25.0
            return 200, {"commands": b.poll(q.get("device") or "phone", wait, q.get("name", ""))}
        if method == "GET" and path == "/api/phone/state":
            return 200, b.snapshot()
        if method == "POST" and path == "/api/phone/event":
            dev = str(body.get("device") or "phone")
            for ev in (body.get("events") or [])[:50]:
                if isinstance(ev, dict):
                    b.push_event(dev, ev)
            return 200, {"ok": True}
        if method == "POST" and path == "/api/phone/result":
            b.submit_result(str(body.get("device") or "phone"), str(body.get("id") or ""),
                            bool(body.get("ok")), body.get("data"), str(body.get("error") or ""))
            return 200, {"ok": True}
        if method == "POST" and path == "/api/phone/approve":
            pid = str(body.get("id") or "")
            if body.get("approve") is False:
                return 200, {"ok": b.reject(pid)}
            text = body.get("text")
            return 200, b.approve(pid, None if text is None else str(text))
        if method == "POST" and path == "/api/phone/teach":
            return 200, {"ok": b.teach(str(body.get("incoming") or ""), str(body.get("reply") or ""),
                                       str(body.get("contact") or ""))}
        if method == "POST" and path == "/api/phone/mode":
            from memory import reply_store
            try:
                if body.get("contact"):
                    return 200, {"ok": reply_store.set_rule(str(body["contact"]), str(body.get("mode") or ""))}
                return 200, {"ok": True, "mode": b.set_mode(str(body.get("mode") or ""))}
            except ValueError as e:
                return 400, {"ok": False, "error": str(e)}

        # ── talk to the assistant ────────────────────────────────────────────
        if method == "POST" and path == "/api/command":
            text = str(body.get("text") or "").strip()
            if not text:
                return 400, {"ok": False, "error": "empty command"}
            self._submit_command(text[:2000])
            return 200, {"ok": True}
        if method == "GET" and path == "/api/log":
            try:
                since = int(q.get("since", 0))
            except ValueError:
                since = 0
            with self._lock:
                items = [x for x in self._log if x["i"] > since]
                nxt = self._seq
                status = self.status
            if since > nxt:                       # the laptop restarted and numbering began again
                items, since = list(self._log), 0
            return 200, {"items": items, "next": nxt, "status": status}
        if method == "GET" and path == "/api/insights":
            from core import insights
            return 200, json.loads(insights.to_json(insights.snapshot()))
        return 404, {"error": "not found"}
