"""
python -m unittest tests.test_browser_close -v

The REAL InbuiltBrowserEngine thread, driven against a fake Playwright (no Chromium needed),
plus a static check of the ui.py handlers. Regression for "browser band karo does not close":
the ✕ button only hid the panel (video kept playing) and closing the last tab reloaded Google.
"""
import ast, io, os, sys, tempfile, time, types, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.inbuilt_browser import InbuiltBrowserEngine      # noqa: E402


class Page:
    def __init__(self, ctx, url="about:blank"):
        self.ctx, self.url, self.closed = ctx, url, False
        self.playing = "play" in url
        self.fail_blank = False

    def title(self): return "T:" + self.url[:10]
    def goto(self, url, **kw):
        if url == "about:blank" and self.fail_blank:
            raise RuntimeError("beforeunload dialog")
        self.url = url
        self.playing = "play" in url                       # navigating destroys the old document
    def evaluate(self, js):
        if "pause" in js:
            self.playing = False
    def close(self): self.closed = True; self.playing = False
    def screenshot(self, **kw): return b"\xff\xd8fake"
    def set_viewport_size(self, *a): pass
    def bring_to_front(self): pass
    def __getattr__(self, n):
        if n in ("mouse", "keyboard"):
            return mock.MagicMock()
        raise AttributeError(n)


class Ctx:
    def __init__(self):
        self.pages = []
        self.closed = False
        self.fail_blank = False
    def new_page(self):
        p = Page(self); p.fail_blank = self.fail_blank
        self.pages.append(p); return p
    def on(self, *a): pass
    def close(self): self.closed = True


def fake_playwright(ctx):
    mod = types.ModuleType("playwright.sync_api")
    class PW:
        chromium = types.SimpleNamespace(launch_persistent_context=lambda **kw: ctx)
        def stop(self): pass
    mod.sync_playwright = lambda: types.SimpleNamespace(start=lambda: PW())
    pkg = types.ModuleType("playwright"); pkg.sync_api = mod
    return {"playwright": pkg, "playwright.sync_api": mod}


def wait(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


class T(unittest.TestCase):
    def setUp(self):
        self.ctx = Ctx()
        self.p = mock.patch.dict(sys.modules, fake_playwright(self.ctx)); self.p.start()
        self.td = tempfile.TemporaryDirectory()
        self.closed_calls = []
        self.eng = InbuiltBrowserEngine(Path(self.td.name) / "prof", on_frame=lambda j, u: None,
                                        on_closed=lambda: self.closed_calls.append(1))

    def tearDown(self):
        self.eng.stop(timeout=3.0); self.p.stop(); self.td.cleanup()

    def start(self, url="https://x.test/play1"):
        self.eng.navigate(url)
        self.assertTrue(wait(lambda: self.ctx.pages and self.ctx.pages[0].url == url))

    def test_park_stops_media_closes_tabs_blanks_page_and_hides(self):
        self.start()
        self.eng.new_tab("https://x.test/play2")
        self.assertTrue(wait(lambda: len(self.ctx.pages) == 2))
        self.assertTrue(self.ctx.pages[0].playing and self.ctx.pages[1].playing)
        self.eng.park()
        self.assertFalse(self.eng._visible)
        self.assertTrue(wait(lambda: self.ctx.pages[1].closed and self.ctx.pages[0].url == "about:blank"))
        self.assertFalse(any(p.playing for p in self.ctx.pages))
        self.assertTrue(self.eng._running)                              # Chromium stays alive
        self.assertFalse(self.ctx.closed)

    def test_closing_last_tab_closes_the_browser_instead_of_loading_google(self):
        self.start()
        self.eng.close_tab(-1)
        self.assertTrue(wait(lambda: self.closed_calls == [1]))
        p = self.ctx.pages[0]
        self.assertEqual(p.url, "about:blank"); self.assertNotIn("google", p.url)
        self.assertFalse(p.playing); self.assertFalse(self.eng._visible)

    def test_closing_one_of_two_tabs_keeps_the_browser(self):
        self.start()
        self.eng.new_tab("https://x.test/two")
        self.assertTrue(wait(lambda: len(self.ctx.pages) == 2))
        self.eng.close_tab(1)
        self.assertTrue(wait(lambda: self.ctx.pages[1].closed))
        time.sleep(0.2)
        self.assertEqual(self.closed_calls, []); self.assertEqual(self.ctx.pages[0].url, "https://x.test/play1")

    def test_browser_is_reusable_after_park(self):
        self.start()
        self.eng.park()
        self.assertTrue(wait(lambda: self.ctx.pages[0].url == "about:blank"))
        self.eng.navigate("https://y.test/")
        self.assertTrue(wait(lambda: self.ctx.pages[0].url == "https://y.test/"))
        self.assertTrue(self.eng._visible)

    def test_stuck_page_is_replaced(self):
        self.start()
        self.ctx.pages[0].fail_blank = True
        old = self.ctx.pages[0]
        self.eng.park()
        self.assertTrue(wait(lambda: old.closed))
        self.assertFalse(old.playing)
        self.assertEqual(len(self.ctx.pages), 2)                        # the replacement exists

    def test_park_before_engine_started_is_harmless(self):
        e = InbuiltBrowserEngine(Path(self.td.name) / "p2", on_frame=lambda j, u: None)
        e.park()
        self.assertFalse(e._visible)

    def test_set_visible_false_alone_does_not_stop_playback(self):
        """Documents WHY the old ✕ looked broken, and that hide-only still exists on purpose."""
        self.start()
        self.eng.set_visible(False)
        time.sleep(0.3)
        self.assertTrue(self.ctx.pages[0].playing)


class TestUiHandlers(unittest.TestCase):
    """ui.py needs Qt to import; check the handlers' bodies statically."""
    @classmethod
    def setUpClass(cls):
        src = (Path(__file__).resolve().parent.parent / "ui.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(src)
        cls.src = src

    def method(self, name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.unparse(node)
        self.fail(f"{name} not found")

    def test_close_parks(self):
        self.assertIn("_browser_engine.park()", self.method("_on_browser_close"))

    def test_hide_only_hides(self):
        body = self.method("_on_browser_hide")
        self.assertIn("set_visible(False)", body); self.assertNotIn("park(", body)

    def test_toggle_and_hud_switch_hide_not_close(self):
        self.assertIn("_on_browser_hide()", self.method("_toggle_inbuilt_browser"))
        self.assertNotIn("_on_browser_close()", self.method("_toggle_inbuilt_browser"))
        self.assertIn("_on_browser_hide()", self.method("_open_dashboard_panel"))

    def test_engine_reports_self_close_to_the_hud(self):
        self.assertIn("on_closed=lambda: self._browser_close_sig.emit()", self.src)

    def test_plugin_close_calls_player_close_browser(self):
        from plugins import inbuilt_browser as pl
        pl_player = mock.MagicMock()
        out = pl.run({"action": "close"}, player=pl_player)
        pl_player.close_browser.assert_called_once()
        self.assertIn("playback stopped", out)
        self.assertIn("browser band karo", pl.PLUGIN["description"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
