"""
Playwright-backed inbuilt Chromium engine for JARVIS's HUD browser panel.

Runs Chromium on a dedicated background thread (never blocking the Qt GUI
thread), keeps a persistent browser profile in ``config/browser_profile`` so
logins/cookies survive restarts, and pushes rendered JPEG frames + URL updates
through thread-safe callbacks.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote_plus, urlparse

VIEW_W = 960
VIEW_H = 600
MAX_TABS = 10
_BURST_SECS = 2.5
_BURST_INTERVAL = 0.065
_IDLE_INTERVAL = 0.32


def _short_label(title: str, url: str) -> str:
    txt = (title or "").strip()
    if txt and txt.lower() != "about:blank":
        return txt[:18] + ("…" if len(txt) > 18 else "")
    host = urlparse(url or "").netloc.removeprefix("www.")
    if host:
        return host[:18] + ("…" if len(host) > 18 else "")
    if (url or "").startswith("data:"):
        return "Local Page"
    return "New Tab"


def resolve_target(target: str) -> str:
    """Turn a raw URL or search phrase into a navigable URL."""
    t = (target or "").strip()
    if not t:
        return "https://www.google.com"
    if "://" in t or t.lower().startswith(("data:", "about:", "file:")):
        return t
    if " " not in t and (t.startswith(("localhost", "127.0.0.1")) or "." in t):
        scheme = "http://" if t.startswith(("localhost", "127.0.0.1")) else "https://"
        return scheme + t
    return f"https://www.google.com/search?q={quote_plus(t)}"


class InbuiltBrowserEngine:
    """Thread-safe multi-tab controller around a Playwright Chromium session."""

    def __init__(
        self,
        profile_dir: Path,
        on_frame: Callable[[bytes, str], None],
        on_status: Optional[Callable[[str], None]] = None,
        on_tabs: Optional[Callable[[list[tuple[int, str, str]], int], None]] = None,
        on_closed: Optional[Callable[[], None]] = None,
    ) -> None:
        self._profile_dir = Path(profile_dir)
        self._on_frame = on_frame
        self._on_status = on_status or (lambda _s: None)
        self._on_tabs = on_tabs or (lambda _t, _i: None)
        # Fired (on the browser thread) when the browser closed ITSELF — closing the last tab.
        self._on_closed = on_closed or (lambda: None)
        self._cmds: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._visible = False
        self._current_url = "https://www.google.com"
        self._last_tabs_sig: Optional[tuple] = None
        self._lock = threading.Lock()

    @property
    def current_url(self) -> str:
        with self._lock:
            return self._current_url

    def ensure_started(self) -> None:
        with self._lock:
            if self._running and self._thread and self._thread.is_alive():
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="InbuiltBrowserThread"
            )
            self._thread.start()

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if visible:
            self.ensure_started()
            self._cmds.put(("refresh", None))

    def navigate(self, url_or_query: str) -> None:
        url = resolve_target(url_or_query)
        with self._lock:
            self._current_url = url
        self._visible = True
        self.ensure_started()
        self._cmds.put(("goto", url))

    def new_tab(self, url_or_query: str = "") -> None:
        url = resolve_target(url_or_query)
        with self._lock:
            self._current_url = url
        self._visible = True
        self.ensure_started()
        self._cmds.put(("new_tab", url))

    def switch_tab(self, index: int) -> None:
        self._visible = True
        self.ensure_started()
        self._cmds.put(("switch_tab", int(index)))

    def close_tab(self, index: int = -1) -> None:
        self.ensure_started()
        self._cmds.put(("close_tab", int(index)))

    def back(self) -> None:
        self.ensure_started()
        self._cmds.put(("back", None))

    def forward(self) -> None:
        self.ensure_started()
        self._cmds.put(("forward", None))

    def reload(self) -> None:
        self.ensure_started()
        self._cmds.put(("reload", None))

    def click(self, norm_x: float, norm_y: float, button: str = "left") -> None:
        x = max(0, min(VIEW_W - 1, int(norm_x * VIEW_W)))
        y = max(0, min(VIEW_H - 1, int(norm_y * VIEW_H)))
        self._cmds.put(("click", (x, y, button)))

    def wheel(self, dx: float, dy: float) -> None:
        self._cmds.put(("wheel", (dx, dy)))

    def key_press(self, key: str) -> None:
        self._cmds.put(("key", key))

    def type_text(self, text: str) -> None:
        if text:
            self._cmds.put(("type", text))

    def pause_media(self) -> None:
        if self._running:
            self._cmds.put(("pause_media", None))

    def resume_media(self) -> None:
        if self._running:
            self._cmds.put(("resume_media", None))

    def park(self) -> None:
        """Close the browser, as far as the user can tell: media stops, extra tabs are closed,
        the page goes blank and the panel hides.

        This is what "browser band karo" / the ✕ button must do. Hiding the panel alone
        (`set_visible(False)`) left Chromium running with the video still playing, which
        looked exactly like "it did not close". Chromium itself is kept alive on purpose:
        reopening is instant and the logins in the profile survive. Use `stop()` for a full
        shutdown."""
        self._visible = False
        if self._running:
            self._cmds.put(("park", None))

    def stop(self, timeout: float = 3.0) -> None:
        self._running = False
        try:
            self._cmds.put_nowait(("quit", None))
        except Exception:
            pass
        th = self._thread
        if th is not None and th.is_alive() and threading.current_thread() is not th:
            th.join(timeout=timeout)

    def _emit_tabs(self, pages: list, active_idx: int, force: bool = False) -> None:
        tabs: list[tuple[int, str, str]] = []
        for idx, p in enumerate(pages):
            try:
                u = p.url or ""
            except Exception:
                u = ""
            try:
                t = p.title() or ""
            except Exception:
                t = ""
            tabs.append((idx, _short_label(t, u), u))
        sig = (active_idx, tuple(tabs))
        if not force and sig == self._last_tabs_sig:
            return
        self._last_tabs_sig = sig
        try:
            self._on_tabs(tabs, active_idx)
        except Exception:
            pass

    def _capture(self, pages: list, active_idx: int, force_tabs: bool = False) -> None:
        if not pages:
            return
        active_idx = max(0, min(active_idx, len(pages) - 1))
        page = pages[active_idx]
        self._emit_tabs(pages, active_idx, force=force_tabs)
        if not self._visible:
            return
        try:
            url = page.url or self._current_url
            with self._lock:
                self._current_url = url
            jpg = page.screenshot(type="jpeg", quality=65, animations="allow")
            self._on_frame(jpg, url)
        except Exception:
            pass

    def _loop(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as e:
            self._on_status(f"Playwright unavailable: {e}")
            self._running = False
            return

        self._profile_dir.mkdir(parents=True, exist_ok=True)
        pw = None
        ctx = None
        try:
            pw = sync_playwright().start()
            launch_kw = dict(
                user_data_dir=str(self._profile_dir),
                headless=True,
                viewport={"width": VIEW_W, "height": VIEW_H},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                locale="en-US",
                ignore_default_args=["--enable-automation"],
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-infobars",
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-features=TranslateUI",
                ],
            )
            try:
                # First attempt launching real installed Google Chrome for highest authenticity & trust score
                launch_kw_chrome = dict(launch_kw)
                launch_kw_chrome["channel"] = "chrome"
                ctx = pw.chromium.launch_persistent_context(**launch_kw_chrome)
            except Exception:
                try:
                    ctx = pw.chromium.launch_persistent_context(**launch_kw)
                except Exception as first_err:
                    self._on_status(f"Browser launch failed: {first_err}")
                    self._running = False
                    return

            try:
                # Neutralize webdriver and automation flags that trigger "I am not a robot" bot detection
                ctx.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    if (!window.chrome) {
                        window.chrome = { runtime: {} };
                    }
                """)
            except Exception:
                pass

            pages: list = [ctx.pages[0] if ctx.pages else ctx.new_page()]
            active_idx = 0
            pending_popups: queue.Queue = queue.Queue()
            creating_tab = False

            def _on_new_page(new_p):
                if not creating_tab:
                    pending_popups.put(new_p)

            ctx.on("page", _on_new_page)
            if self._cmds.empty():
                try:
                    pages[0].goto(self.current_url, wait_until="commit", timeout=8_000)
                except Exception:
                    pass
                self._capture(pages, active_idx, force_tabs=True)

            def _park_pages() -> None:
                """Pause media, close every tab but the first, blank it. Navigating away
                destroys the document, which is what actually silences a playing video."""
                nonlocal active_idx
                for pg in pages:
                    try:
                        pg.evaluate(
                            "document.querySelectorAll('video,audio').forEach(m=>{try{m.pause();}catch(e){}})"
                        )
                    except Exception:
                        pass
                for victim in pages[1:]:
                    try:
                        victim.close()
                    except Exception:
                        pass
                del pages[1:]
                active_idx = 0
                try:
                    pages[0].goto("about:blank", wait_until="commit", timeout=5_000)
                except Exception:
                    try:                                   # a page that will not leave: replace it
                        fresh = ctx.new_page()
                        old = pages[0]
                        pages[0] = fresh
                        old.close()
                    except Exception:
                        pass

            def _exec_cmd(cmd_name: str, cmd_arg) -> bool:
                nonlocal active_idx, creating_tab
                if not pages:
                    return False
                active_idx = max(0, min(active_idx, len(pages) - 1))
                page = pages[active_idx]
                changed = False
                if cmd_name == "goto":
                    page.goto(str(cmd_arg), wait_until="commit", timeout=10_000)
                    changed = True
                elif cmd_name == "new_tab":
                    if len(pages) >= MAX_TABS:
                        self._on_status(f"Tab limit ({MAX_TABS}) reached; opening in current tab.")
                        page.goto(str(cmd_arg), wait_until="commit", timeout=10_000)
                    else:
                        creating_tab = True
                        try:
                            np = ctx.new_page()
                        finally:
                            creating_tab = False
                        try:
                            np.set_viewport_size({"width": VIEW_W, "height": VIEW_H})
                        except Exception:
                            pass
                        pages.append(np)
                        active_idx = len(pages) - 1
                        np.goto(str(cmd_arg), wait_until="commit", timeout=10_000)
                    changed = True
                elif cmd_name == "switch_tab":
                    idx = int(cmd_arg)
                    if 0 <= idx < len(pages):
                        active_idx = idx
                        try:
                            pages[active_idx].bring_to_front()
                        except Exception:
                            pass
                        changed = True
                elif cmd_name == "close_tab":
                    idx = active_idx if int(cmd_arg) < 0 else int(cmd_arg)
                    if len(pages) <= 1:
                        # Closing the LAST tab closes the browser, as in any browser. (It used
                        # to load Google again, so nothing ever seemed to close.)
                        _park_pages()
                        self._visible = False
                        changed = True
                        try:
                            self._on_closed()
                        except Exception:
                            pass
                    elif 0 <= idx < len(pages):
                        victim = pages.pop(idx)
                        try:
                            victim.close()
                        except Exception:
                            pass
                        if active_idx >= len(pages):
                            active_idx = len(pages) - 1
                        elif idx < active_idx:
                            active_idx -= 1
                        changed = True
                elif cmd_name == "park":
                    _park_pages()
                    changed = True
                elif cmd_name == "back":
                    page.go_back(wait_until="commit", timeout=8_000)
                    changed = True
                elif cmd_name == "forward":
                    page.go_forward(wait_until="commit", timeout=8_000)
                    changed = True
                elif cmd_name == "reload":
                    page.reload(wait_until="commit", timeout=8_000)
                    changed = True
                elif cmd_name == "click":
                    x, y, btn = cmd_arg
                    page.mouse.click(x, y, button=btn)
                elif cmd_name == "wheel":
                    dx, dy = cmd_arg
                    page.mouse.wheel(dx, dy)
                elif cmd_name == "key":
                    page.keyboard.press(str(cmd_arg))
                elif cmd_name == "type":
                    page.keyboard.type(str(cmd_arg))
                elif cmd_name == "pause_media":
                    for pg in pages:
                        try:
                            pg.evaluate(
                                "document.querySelectorAll('video,audio').forEach(m=>{try{if(!m.paused){m.dataset.jPaused='1';m.pause();}}catch(e){}})"
                            )
                        except Exception:
                            pass
                elif cmd_name == "resume_media":
                    for pg in pages:
                        try:
                            pg.evaluate(
                                "document.querySelectorAll('video,audio').forEach(m=>{try{if(m.dataset.jPaused==='1'){delete m.dataset.jPaused;m.play();}}catch(e){}})"
                            )
                        except Exception:
                            pass
                return changed

            last_shot = time.monotonic()
            burst_until = time.monotonic() + _BURST_SECS
            while self._running:
                # Adopt any popup / target=_blank tabs cleanly on the loop thread
                while not pending_popups.empty():
                    try:
                        np = pending_popups.get_nowait()
                    except queue.Empty:
                        break
                    if any(p is np for p in pages):
                        continue
                    if len(pages) < MAX_TABS:
                        try:
                            np.set_viewport_size({"width": VIEW_W, "height": VIEW_H})
                        except Exception:
                            pass
                        pages.append(np)
                        active_idx = len(pages) - 1
                        burst_until = time.monotonic() + _BURST_SECS
                        self._capture(pages, active_idx, force_tabs=True)
                    else:
                        try:
                            u = np.url
                            if u and u != "about:blank":
                                pages[active_idx].goto(u, wait_until="commit", timeout=8_000)
                        except Exception:
                            pass
                        try:
                            np.close()
                        except Exception:
                            pass

                now = time.monotonic()
                interval = _BURST_INTERVAL if now < burst_until else _IDLE_INTERVAL
                timeout = interval if self._visible else 0.75
                try:
                    cmd, arg = self._cmds.get(timeout=timeout)
                except queue.Empty:
                    if self._visible and (time.monotonic() - last_shot) >= interval:
                        self._capture(pages, active_idx)
                        last_shot = time.monotonic()
                    continue

                if cmd == "quit":
                    break

                burst_until = time.monotonic() + _BURST_SECS
                # Gather command batch and coalesce adjacent scroll/type events
                batch = [(cmd, arg)]
                while not self._cmds.empty():
                    try:
                        ncmd, narg = self._cmds.get_nowait()
                    except queue.Empty:
                        break
                    if ncmd == "quit":
                        self._running = False
                        batch.append((ncmd, narg))
                        break
                    if ncmd == "wheel" and batch and batch[-1][0] == "wheel":
                        px, py = batch[-1][1]
                        batch[-1] = ("wheel", (px + narg[0], py + narg[1]))
                    elif ncmd == "type" and batch and batch[-1][0] == "type":
                        batch[-1] = ("type", batch[-1][1] + str(narg))
                    else:
                        batch.append((ncmd, narg))

                tab_changed = False
                quit_now = False
                for bcmd, barg in batch:
                    if bcmd == "quit":
                        quit_now = True
                        break
                    try:
                        if _exec_cmd(bcmd, barg):
                            tab_changed = True
                    except Exception as e:
                        self._on_status(f"Browser error: {e}")

                if quit_now:
                    break

                self._capture(pages, active_idx, force_tabs=tab_changed)
                last_shot = time.monotonic()
        finally:
            self._running = False
            try:
                if ctx is not None:
                    ctx.close()
            except Exception:
                pass
            try:
                if pw is not None:
                    pw.stop()
            except Exception:
                pass
