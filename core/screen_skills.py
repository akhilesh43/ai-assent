"""
core/screen_skills.py — show JARVIS a click sequence once, and it can repeat it.

TWO WAYS TO TEACH
    1. TEACH SESSION   "seekho: Chrome band karna" → you click → "bas".
    2. LIVE LEARNING   (opt-in) "live seekhna chalu karo" → JARVIS keeps a short rolling
                       memory of your last clicks while you work. Afterwards:
                       "jo abhi kiya wo yaad rakh lo: Chrome band karna".

WHAT IS RECORDED — and what is not
    Per click: the button, how long you paused first, which app/window it was in, where
    inside that window, and TWO tiny pictures of the pixels around the cursor (a tight and
    a wide one). That is all. There is no screen streaming, no keystroke logging, and no
    background screenshots. Live Learning keeps its rolling buffer IN MEMORY ONLY (60 clicks,
    2 hours): nothing reaches the disk until you say "yaad rakh lo", and "live seekhna band
    karo" wipes it. Clicks on JARVIS's own window are never recorded. The pictures stay on
    this PC; only text (labels, window names, positions) is mirrored to Firebase.

HOW A SKILL IS REPLAYED — it never clicks blind
    For every click, in this order:
      1. find the window it happened in (process name + title), and bring it forward;
      2. look for the saved picture inside that window (OpenCV, or a numpy fallback);
      3. ask Gemini to find the element by its label (only if a locator is supplied);
      4. use the recorded position inside the window — ONLY if the window is still within
         12 % of its recorded size.
    If none of those works, replay STOPS and says which step failed. It also refuses to
    click anything that belongs to JARVIS itself, or anything off-screen.

Everything that touches the operating system goes through a `Backend`, so the logic here
is exercised in tests with a fake screen, fake mouse and fake windows.
"""
from __future__ import annotations

import collections
import difflib
import json
import os
import platform
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SKILLS_FILE = BASE_DIR / "config" / "ui_skills.json"
CROPS_DIR = BASE_DIR / "config" / "ui_skills"

TIGHT = 48                      # px, picture of just the control
WIDE = 120                      # px, picture of the control and its surroundings
MAX_STEPS = 40
MAX_DELAY = 8.0                 # a step never waits longer than this on replay
BURST_GAP = 8.0                 # clicks closer than this belong to one "thing you did"
LIVE_MAX_EVENTS = 60
LIVE_MAX_AGE = 2 * 3600
TEACH_IDLE_TIMEOUT = 60.0
SIZE_TOLERANCE = 0.12           # window must be within 12 % of recorded size for "relative"
TIGHT_THRESHOLD = 0.90
WIDE_THRESHOLD = 0.85


# ─────────────────────────────────────────────────────────────────────────────
#  Types
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WinInfo:
    hwnd: int = 0
    pid: int = 0
    process: str = ""
    title: str = ""
    rect: tuple = (0, 0, 0, 0)          # left, top, right, bottom (screen pixels)

    @property
    def size(self) -> tuple[int, int]:
        return (self.rect[2] - self.rect[0], self.rect[3] - self.rect[1])


@dataclass
class ClickEvent:
    t: float
    x: int
    y: int
    button: str
    win: Optional[WinInfo]
    tight: Optional[np.ndarray]
    wide: Optional[np.ndarray]
    tight_off: tuple = (0, 0)           # where the click sits INSIDE each picture
    wide_off: tuple = (0, 0)
    screen: tuple = (0, 0)
    double: bool = False


@dataclass
class RunResult:
    ok: bool
    done: int
    total: int
    message: str
    how: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  Image helpers
# ─────────────────────────────────────────────────────────────────────────────
def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)


def crop_around(img: np.ndarray, cx: int, cy: int, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """`size`×`size` pixels around (cx, cy), shifted to stay inside the image.
    Returns (crop, (click_x_in_crop, click_y_in_crop))."""
    h, w = img.shape[:2]
    size = min(size, h, w)
    left = int(max(0, min(w - size, cx - size // 2)))
    top = int(max(0, min(h - size, cy - size // 2)))
    return img[top:top + size, left:left + size].copy(), (cx - left, cy - top)


def find_template(image: np.ndarray, template: np.ndarray, threshold: float
                  ) -> Optional[tuple[float, int, int]]:
    """Best normalised-cross-correlation match of `template` in `image`.
    Returns (score, left, top) if score >= threshold, else None. Flat templates (no
    detail to match on) are refused rather than matched to anything."""
    ih, iw = image.shape[:2]
    th, tw = template.shape[:2]
    if th > ih or tw > iw or th < 8 or tw < 8:
        return None
    t = _gray(template)
    if float(t.std()) < 6.0:
        return None
    g = _gray(image)
    score, left, top = None, 0, 0
    try:
        import cv2                                                   # type: ignore
        res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        idx = int(np.argmax(res))
        top, left = divmod(idx, res.shape[1])
        score = float(res[top, left])
    except Exception:
        score = None
    if score is None:                                                # numpy/scipy fallback
        score, left, top = _ncc_numpy(g, t)
    return (score, int(left), int(top)) if score >= threshold else None


def _ncc_numpy(g: np.ndarray, t: np.ndarray) -> tuple[float, int, int]:
    from scipy.signal import fftconvolve
    th, tw = t.shape
    n = th * tw
    tz = t - t.mean()
    tnorm = float(np.sqrt((tz * tz).sum())) + 1e-9
    num = fftconvolve(g, tz[::-1, ::-1], mode="valid")
    ii = np.pad(g, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ii2 = np.pad(g * g, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def box(a):
        return a[th:, tw:] - a[:-th, tw:] - a[th:, :-tw] + a[:-th, :-tw]
    s1, s2 = box(ii), box(ii2)
    var = np.maximum(s2 - s1 * s1 / n, 1e-6)
    ncc = num / (np.sqrt(var) * tnorm)
    ncc = np.nan_to_num(ncc, nan=0.0, posinf=0.0, neginf=0.0)
    idx = int(np.argmax(ncc))
    top, left = divmod(idx, ncc.shape[1])
    return float(ncc[top, left]), left, top


def _png_bytes(img: np.ndarray) -> bytes:
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _load_png(path: Path) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Backend (the only part that touches the real machine)
# ─────────────────────────────────────────────────────────────────────────────
class Backend:
    """Interface. `SystemBackend` is the real one; tests supply a fake."""

    def bounds(self) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) of everything grab() can see, in screen coordinates.
        left/top are negative when a monitor sits left of / above the primary one."""
        raise NotImplementedError

    def screen_size(self) -> tuple[int, int]:
        l, t, r, b = self.bounds()
        return r - l, b - t

    def grab(self, box: Optional[tuple] = None) -> np.ndarray:
        """RGB pixels of `box` (screen coordinates), or of all of bounds() when None."""
        raise NotImplementedError
    def windows(self) -> list[WinInfo]: raise NotImplementedError
    def window_at(self, x: int, y: int) -> Optional[WinInfo]: raise NotImplementedError
    def focus(self, win: WinInfo) -> None: raise NotImplementedError
    def click(self, x: int, y: int, button: str = "left", double: bool = False) -> None: raise NotImplementedError
    def hotkey(self, *keys: str) -> None: raise NotImplementedError
    def type_text(self, text: str) -> None: raise NotImplementedError
    def own_pids(self) -> set[int]: return {os.getpid()}
    def start_listener(self, on_press: Callable[[int, int, str], None]) -> Any: raise NotImplementedError
    def stop_listener(self, handle: Any) -> None: raise NotImplementedError


class SystemBackend(Backend):
    """Real screen, mouse and windows. Window features need Windows (ctypes); on other
    systems clicks are still recorded and replayed by picture + position, but without
    app/window names."""

    def __init__(self) -> None:
        self._win = platform.system() == "Windows"

    # -- screen ----------------------------------------------------------------
    def bounds(self) -> tuple[int, int, int, int]:
        try:
            import mss
            with mss.mss() as sct:
                m = sct.monitors[0]
                return int(m["left"]), int(m["top"]), int(m["left"] + m["width"]), int(m["top"] + m["height"])
        except Exception:
            import pyautogui
            s = pyautogui.size()
            return 0, 0, int(s.width), int(s.height)

    def grab(self, box: Optional[tuple] = None) -> np.ndarray:
        try:
            import mss
            with mss.mss() as sct:
                mon = sct.monitors[0]                                  # whole virtual screen
                if box:
                    l, t, r, b = [int(v) for v in box]
                    region = {"left": l, "top": t, "width": max(1, r - l), "height": max(1, b - t)}
                else:
                    region = mon
                shot = sct.grab(region)
                return np.asarray(shot)[..., :3][..., ::-1].copy()      # BGRA → RGB
        except Exception:
            from PIL import ImageGrab
            img = ImageGrab.grab(bbox=tuple(int(v) for v in box) if box else None, all_screens=True)
            return np.asarray(img.convert("RGB"))

    # -- windows ---------------------------------------------------------------
    def _proc_name(self, pid: int) -> str:
        try:
            import psutil
            return psutil.Process(pid).name()
        except Exception:
            return ""

    def _info(self, hwnd: int) -> Optional[WinInfo]:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        if not hwnd:
            return None
        n = u.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(hwnd, buf, n + 1)
        pid = wintypes.DWORD(0)
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        r = wintypes.RECT()
        u.GetWindowRect(hwnd, ctypes.byref(r))
        return WinInfo(int(hwnd), int(pid.value), self._proc_name(int(pid.value)),
                       buf.value.strip(), (r.left, r.top, r.right, r.bottom))

    def windows(self) -> list[WinInfo]:
        if not self._win:
            return []
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        out: list[WinInfo] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def _cb(hwnd, _lp):
            if u.IsWindowVisible(hwnd) and not u.IsIconic(hwnd) and u.GetWindowTextLengthW(hwnd) > 0:
                w = self._info(hwnd)
                if w and w.title and w.title not in ("Program Manager",):
                    out.append(w)
            return True
        u.EnumWindows(_cb, 0)
        return out

    def window_at(self, x: int, y: int) -> Optional[WinInfo]:
        if not self._win:
            return None
        try:
            import ctypes
            from ctypes import wintypes
            u = ctypes.windll.user32
            h = u.WindowFromPoint(wintypes.POINT(int(x), int(y)))
            root = u.GetAncestor(h, 2)                                   # GA_ROOT
            return self._info(root or h)
        except Exception:
            return None

    def focus(self, win: WinInfo) -> None:
        if not self._win or not win.hwnd:
            return
        import ctypes
        u = ctypes.windll.user32
        if u.IsIconic(win.hwnd):
            u.ShowWindow(win.hwnd, 9)                                    # SW_RESTORE
        u.SetForegroundWindow(win.hwnd)

    def own_pids(self) -> set[int]:
        """Cached for 30 s: appidentity walks the process table, and this is asked on every
        click — which has to be recorded before it lands."""
        now = time.monotonic()
        cached = getattr(self, "_own_cache", None)
        if cached and now - cached[0] < 30.0:
            return cached[1]
        try:
            from core import appidentity
            pids = set(appidentity.own_pids())
            pids.add(os.getpid())
        except Exception:
            pids = {os.getpid()}
        self._own_cache = (now, pids)
        return pids

    # -- input -----------------------------------------------------------------
    def click(self, x: int, y: int, button: str = "left", double: bool = False) -> None:
        import pyautogui
        pyautogui.click(int(x), int(y), clicks=2 if double else 1, button=button)

    def hotkey(self, *keys: str) -> None:
        import pyautogui
        pyautogui.hotkey(*keys)

    def type_text(self, text: str) -> None:
        import pyautogui
        if text.isascii():
            pyautogui.write(text, interval=0.02)
            return
        try:                                                              # non-ASCII → paste
            import pyperclip
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
        except Exception:
            pyautogui.write(text.encode("ascii", "ignore").decode(), interval=0.02)

    # -- mouse listener --------------------------------------------------------
    def start_listener(self, on_press: Callable[[int, int, str], None]) -> Any:
        try:
            from pynput import mouse                                     # type: ignore

            def _on(x, y, button, pressed):
                if pressed:
                    name = {"Button.left": "left", "Button.right": "right",
                            "Button.middle": "middle"}.get(str(button), "left")
                    on_press(int(x), int(y), name)
            lis = mouse.Listener(on_click=_on)
            lis.daemon = True
            lis.start()
            return ("pynput", lis)
        except Exception:
            pass
        if self._win:                                                     # no pynput: poll
            stop = threading.Event()

            def _poll():
                import ctypes
                from ctypes import wintypes
                u = ctypes.windll.user32
                down = {0x01: False, 0x02: False, 0x04: False}
                names = {0x01: "left", 0x02: "right", 0x04: "middle"}
                while not stop.is_set():
                    for vk, nm in names.items():
                        st = bool(u.GetAsyncKeyState(vk) & 0x8000)
                        if st and not down[vk]:
                            pt = wintypes.POINT()
                            u.GetCursorPos(ctypes.byref(pt))
                            on_press(int(pt.x), int(pt.y), nm)
                        down[vk] = st
                    time.sleep(0.012)
            th = threading.Thread(target=_poll, daemon=True, name="ClickPoll")
            th.start()
            return ("poll", stop)
        raise RuntimeError("no way to watch the mouse here — pip install pynput")

    def stop_listener(self, handle: Any) -> None:
        try:
            kind, h = handle
            if kind == "pynput":
                h.stop()
            else:
                h.set()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Store
# ─────────────────────────────────────────────────────────────────────────────
def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9\u0900-\u097F]+", "_", (text or "").strip().lower()).strip("_")
    return (s or "skill")[:48]


class SkillStore:
    def __init__(self, path: Path = SKILLS_FILE, crops_dir: Path = CROPS_DIR) -> None:
        self.path, self.crops_dir = Path(path), Path(crops_dir)
        self._lock = threading.RLock()

    def _read(self) -> dict:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(d, dict) and isinstance(d.get("skills"), dict):
                return d
        except Exception:
            pass
        return {"version": 1, "skills": {}}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def all(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._read()["skills"])

    def get(self, slug: str) -> Optional[dict]:
        return self.all().get(slug)

    def put(self, slug: str, skill: dict) -> None:
        with self._lock:
            d = self._read()
            d["skills"][slug] = skill
            self._write(d)

    def forget(self, slug: str) -> bool:
        with self._lock:
            d = self._read()
            if slug not in d["skills"]:
                return False
            del d["skills"][slug]
            self._write(d)
        import shutil
        shutil.rmtree(self.crops_dir / slug, ignore_errors=True)
        return True

    def resolve(self, spoken: str) -> tuple[Optional[str], list[str]]:
        """(slug, candidates). A slug only when the match is clear; otherwise None plus the
        plausible candidates so the caller can ask which one was meant."""
        skills = self.all()
        if not skills:
            return None, []
        q = slugify(spoken)
        names: dict[str, str] = {}
        for slug, sk in skills.items():
            names[slug] = slug
            names[slugify(sk.get("name", ""))] = slug
            for al in sk.get("aliases", []) or []:
                names[slugify(al)] = slug
        if q in names:
            return names[q], [names[q]]
        close = difflib.get_close_matches(q, list(names), n=3, cutoff=0.6)
        cands = list(dict.fromkeys(names[c] for c in close))
        if close and (len(cands) == 1 or difflib.SequenceMatcher(None, q, close[0]).ratio() >= 0.75):
            return names[close[0]], cands
        qw = {w for w in q.split("_") if len(w) > 2}
        ranked = sorted(((len(qw & set(k.split("_"))), k) for k in names), reverse=True)
        if ranked and ranked[0][0] >= 2 and (len(ranked) == 1 or ranked[1][0] < ranked[0][0]):
            return names[ranked[0][1]], [names[ranked[0][1]]]
        return None, cands


# ─────────────────────────────────────────────────────────────────────────────
#  Recording (teach session + live-learning buffer)
# ─────────────────────────────────────────────────────────────────────────────
class ClickRecorder:
    """Listens to the mouse and turns presses into ClickEvents.

    `keep=None`  → unbounded session list (teach), `keep=N` → rolling buffer (watch)."""

    def __init__(self, backend: Backend, keep: Optional[int] = None,
                 max_age: Optional[float] = None, idle_timeout: Optional[float] = None,
                 on_timeout: Optional[Callable[[], None]] = None,
                 on_event: Optional[Callable[[ClickEvent], None]] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.backend = backend
        self.events: collections.deque[ClickEvent] = collections.deque(maxlen=keep)
        self._max_age = max_age
        self._idle = idle_timeout
        self._on_timeout = on_timeout
        self._on_event = on_event
        self._clock = clock
        self._handle = None
        self._lock = threading.Lock()
        self._last_press = clock()
        self._watchdog: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.running = False
        self.ignored_own = 0

    def start(self) -> None:
        self.running = True
        self._stop.clear()
        self._last_press = self._clock()
        self._handle = self.backend.start_listener(self._press)
        if self._idle:
            self._watchdog = threading.Thread(target=self._watch, daemon=True, name="TeachIdle")
            self._watchdog.start()

    def stop(self) -> None:
        self.running = False
        self._stop.set()
        if self._handle is not None:
            self.backend.stop_listener(self._handle)
            self._handle = None

    def clear(self) -> None:
        with self._lock:
            self.events.clear()

    def _watch(self) -> None:
        while not self._stop.wait(1.0):
            if self._idle and self._clock() - self._last_press > self._idle:
                self.stop()
                if self._on_timeout:
                    try:
                        self._on_timeout()
                    except Exception:
                        pass
                return

    def _press(self, x: int, y: int, button: str) -> None:
        """Runs on the mouse-listener thread; must be quick (the click has not landed yet)."""
        try:
            now = self._clock()
            self._last_press = now
            win = self.backend.window_at(x, y)
            if win is not None and win.pid in self.backend.own_pids():
                self.ignored_own += 1                       # never record clicks on JARVIS itself
                return
            bl, bt, _br, _bb = self.backend.bounds()
            shot = self.backend.grab()
            h, w = shot.shape[:2]
            # The pictures are cut ONLY from inside the clicked window. A close button sits
            # on the window's edge, and whatever is behind the window (wallpaper, another
            # app) is different next time — it would make the saved picture unmatchable.
            src, ox, oy = shot, bl, bt
            if win is not None:
                wl, wt = max(win.rect[0], bl), max(win.rect[1], bt)
                wr, wb = min(win.rect[2], _br), min(win.rect[3], _bb)
                if wr - wl >= 24 and wb - wt >= 24:
                    src, ox, oy = shot[wt - bt:wb - bt, wl - bl:wr - bl], wl, wt
            tight, toff = crop_around(src, x - ox, y - oy, TIGHT)
            wide, woff = crop_around(src, x - ox, y - oy, WIDE)
            ev = ClickEvent(t=now, x=x, y=y, button=button, win=win, tight=tight, wide=wide,
                            tight_off=toff, wide_off=woff, screen=(w, h))
            with self._lock:
                prev = self.events[-1] if self.events else None
                if (prev is not None and prev.button == button and now - prev.t < 0.45
                        and abs(prev.x - x) < 6 and abs(prev.y - y) < 6):
                    prev.double = True                       # second click of a double-click
                    return
                self._expire(now)
                self.events.append(ev)
            if self._on_event:
                self._on_event(ev)
        except Exception:
            pass                                             # never break the mouse hook

    def _expire(self, now: float) -> None:
        if self._max_age is None:
            return
        while self.events and now - self.events[0].t > self._max_age:
            self.events.popleft()

    def bursts(self) -> list[list[ClickEvent]]:
        with self._lock:
            evs = list(self.events)
        out: list[list[ClickEvent]] = []
        for e in evs:
            if out and e.t - out[-1][-1].t <= BURST_GAP:
                out[-1].append(e)
            else:
                out.append([e])
        return out


def events_to_steps(events: list[ClickEvent]) -> list[dict]:
    """Click events → step dicts (without picture files yet)."""
    steps = []
    prev_t = None
    for e in events[:MAX_STEPS]:
        delay = 0.5 if prev_t is None else float(min(MAX_DELAY, max(0.15, e.t - prev_t)))
        prev_t = e.t
        step: dict = {"type": "click", "button": e.button, "double": bool(e.double),
                      "delay": round(delay, 2), "abs": [e.x, e.y], "screen": list(e.screen)}
        if e.win is not None:
            w = e.win
            step["window"] = {"process": w.process, "title": w.title[:120], "rect": list(w.rect)}
            ww, wh = max(1, w.rect[2] - w.rect[0]), max(1, w.rect[3] - w.rect[1])
            step["rel"] = [round((e.x - w.rect[0]) / ww, 4), round((e.y - w.rect[1]) / wh, 4)]
        step["tight_off"] = list(e.tight_off)
        step["wide_off"] = list(e.wide_off)
        steps.append(step)
    return steps


def save_skill(store: SkillStore, name: str, events: list[ClickEvent],
               extra_steps: Optional[list[dict]] = None,
               labeler: Optional[Callable[[bytes], str]] = None,
               aliases: Optional[list[str]] = None) -> tuple[str, dict]:
    """Write a skill: JSON + the pictures. Returns (slug, skill)."""
    slug = slugify(name)
    steps = events_to_steps(events)
    folder = store.crops_dir / slug
    import shutil
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    ci = 0
    for step in steps:
        if step["type"] != "click":
            continue
        ev = events[ci]
        ci += 1
        from PIL import Image
        for key, arr in (("tight", ev.tight), ("wide", ev.wide)):
            if arr is not None:
                fn = f"s{ci}_{key}.png"
                Image.fromarray(arr.astype(np.uint8)).save(folder / fn)
                step[key] = f"{slug}/{fn}"
        if labeler is not None and ev.wide is not None:
            try:
                step["label"] = (labeler(_png_bytes(ev.wide)) or "")[:60]
            except Exception:
                pass
    if extra_steps:
        steps.extend(extra_steps)
    skill = {"name": name.strip(), "aliases": list(aliases or []),
             "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "uses": 0, "steps": steps}
    store.put(slug, skill)
    return slug, skill


# ─────────────────────────────────────────────────────────────────────────────
#  Replay
# ─────────────────────────────────────────────────────────────────────────────
Locator = Callable[[np.ndarray, str, str], Optional[tuple]]      # (image, label, window title) → (fx, fy)


def _nap(seconds: float) -> None:
    """The runner's pause. A named hook (not time.sleep bound as a default) so tests can make
    replay instant without freezing every other thread's sleep."""
    time.sleep(seconds)


class SkillRunner:
    def __init__(self, backend: Backend, store: SkillStore, locator: Optional[Locator] = None,
                 sleep: Optional[Callable[[float], None]] = None, speed: float = 1.0,
                 progress: Optional[Callable[[str], None]] = None) -> None:
        self.backend, self.store, self.locator = backend, store, locator
        self._sleep = sleep or (lambda sec: _nap(sec))
        self._speed = max(0.25, float(speed))
        self._progress = progress or (lambda s: None)
        self._abort = threading.Event()

    def abort(self) -> None:
        self._abort.set()

    # -- finding a window ------------------------------------------------------
    def _find_window(self, rec: dict) -> Optional[WinInfo]:
        """The recorded window, by process name first and title second. Never one of ours."""
        want_p = (rec.get("process") or "").lower()
        want_t = (rec.get("title") or "").lower()
        own = self.backend.own_pids()
        best, best_s = None, 0
        for w in self.backend.windows():
            if w.pid in own:
                continue
            t = w.title.lower()
            title_lvl = 3 if (want_t and t == want_t) else \
                        1 if (want_t and (want_t[:14] in t or t[:14] in want_t)) else 0
            proc_ok = bool(want_p) and w.process.lower() == want_p
            if want_p:
                if not (proc_ok or title_lvl == 3):
                    continue                                  # a different app: never a candidate
            elif title_lvl == 0:
                continue
            score = (2 if proc_ok else 0) + title_lvl
            if score > best_s:                                # strict ">" keeps z-order: topmost wins ties
                best, best_s = w, score
        return best

    # -- finding the click point -----------------------------------------------
    def _locate(self, step: dict, win: Optional[WinInfo]) -> tuple[Optional[tuple[int, int]], str]:
        bl, bt, br, bb = self.backend.bounds()
        sw, sh = br - bl, bb - bt
        full = (bl, bt, br, bb)
        if win is not None:
            l, t, r, b = win.rect
            box = (max(bl, l), max(bt, t), min(br, r), min(bb, b))
            if box[2] - box[0] < 20 or box[3] - box[1] < 20:
                box = full
        else:
            box = full
        region = self.backend.grab(box)

        for key, thr, off_key in (("tight", TIGHT_THRESHOLD, "tight_off"),
                                  ("wide", WIDE_THRESHOLD, "wide_off")):
            rel = step.get(key)
            if not rel:
                continue
            tpl = _load_png(self.store.crops_dir / rel)
            if tpl is None:
                continue
            hit = find_template(region, tpl, thr)
            if hit:
                _score, left, top = hit
                ox, oy = step.get(off_key, [tpl.shape[1] // 2, tpl.shape[0] // 2])
                return (box[0] + left + int(ox), box[1] + top + int(oy)), f"picture({key})"

        label = (step.get("label") or "").strip()
        if self.locator is not None and label:
            try:
                got = self.locator(region, label, win.title if win else "")
            except Exception:
                got = None
            if got and len(got) >= 2 and 0.0 <= got[0] <= 1.0 and 0.0 <= got[1] <= 1.0:
                return (box[0] + int(got[0] * (box[2] - box[0])),
                        box[1] + int(got[1] * (box[3] - box[1]))), "gemini"

        rel_pos = step.get("rel")
        rec_win = step.get("window") or {}
        if win is not None and rel_pos and rec_win.get("rect"):
            rl, rt, rr, rb = rec_win["rect"]
            rw, rh = max(1, rr - rl), max(1, rb - rt)
            ww, wh = win.size
            if abs(ww - rw) / rw <= SIZE_TOLERANCE and abs(wh - rh) / rh <= SIZE_TOLERANCE:
                return (win.rect[0] + int(rel_pos[0] * ww), win.rect[1] + int(rel_pos[1] * wh)), "position"
        elif win is None and step.get("abs") and list(step.get("screen") or []) == [sw, sh]:
            return (int(step["abs"][0]), int(step["abs"][1])), "position"   # same screen, same spot
        return None, ""

    # -- run -------------------------------------------------------------------
    def run(self, slug: str) -> RunResult:
        skill = self.store.get(slug)
        if not skill:
            return RunResult(False, 0, 0, f"No skill named '{slug}'.")
        steps = skill.get("steps", [])
        total = len(steps)
        how: list = []
        own = self.backend.own_pids()
        bl, bt, br, bb = self.backend.bounds()
        for i, step in enumerate(steps, 1):
            if self._abort.is_set():
                return RunResult(False, i - 1, total, f"Stopped before step {i}.", how)
            self._sleep(min(MAX_DELAY, float(step.get("delay", 0.4))) / self._speed)
            kind = step.get("type", "click")
            self._progress(f"step {i}/{total}: {kind}")
            if kind == "wait":
                self._sleep(min(30.0, float(step.get("seconds", 1.0))))
                how.append("wait")
            elif kind == "hotkey":
                self.backend.hotkey(*[str(k) for k in step.get("keys", [])])
                how.append("hotkey")
            elif kind == "type":
                self.backend.type_text(str(step.get("text", "")))
                how.append("type")
            elif kind == "click":
                rec_win = step.get("window")
                win = self._find_window(rec_win) if rec_win else None
                if rec_win and win is None:
                    return RunResult(False, i - 1, total,
                                     f"Step {i}: I could not find the '{rec_win.get('process') or rec_win.get('title')}' "
                                     f"window. Open it and try again.", how)
                if win is not None:
                    self.backend.focus(win)
                    self._sleep(0.25)
                pt, via = self._locate(step, win)
                if pt is None:
                    lbl = step.get("label") or "the button"
                    return RunResult(False, i - 1, total,
                                     f"Step {i}: I could not find {lbl} on the screen, so I stopped "
                                     f"instead of clicking blindly.", how)
                x, y = pt
                if not (bl <= x < br and bt <= y < bb):
                    return RunResult(False, i - 1, total, f"Step {i}: that spot is off the screen.", how)
                under = self.backend.window_at(x, y)
                if under is not None and under.pid in own:
                    return RunResult(False, i - 1, total,
                                     f"Step {i}: that spot belongs to my own window — not clicking.", how)
                self.backend.click(x, y, step.get("button", "left"), bool(step.get("double")))
                how.append(via)
            else:
                how.append("skip")
        skill["uses"] = int(skill.get("uses", 0)) + 1
        skill["last_used"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.store.put(slug, skill)
        return RunResult(True, total, total, f"Done — {total} step(s).", how)


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt block (so Gemini knows which skills exist)
# ─────────────────────────────────────────────────────────────────────────────
def prompt_block(store: Optional[SkillStore] = None, limit: int = 25) -> str:
    try:
        skills = (store or SkillStore()).all()
    except Exception:
        return ""
    if not skills:
        return ""
    names = [s.get("name", slug) for slug, s in list(skills.items())[:limit]]
    return ("[LEARNED SCREEN SKILLS]\n"
            "The user has taught you these click-sequences: " + "; ".join(names) + ".\n"
            "When they ask for one of these (or nearly the same words), call screen_teach with "
            "mode='run' and name=<that skill> — do not ask them to explain it again.")
