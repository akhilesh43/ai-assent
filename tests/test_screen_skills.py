"""
python -m unittest tests.test_screen_skills -v

A fake desktop (a numpy canvas, a window list, a mouse) stands in for the real machine, so
recording, live-learning and replay are exercised end to end without a display or a mouse.
It cannot prove the real Windows mouse hook or real DPI scaling — see the README.
"""
import os, sys, tempfile, time, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import screen_skills as ss      # noqa: E402

RNG = np.random.default_rng(3)


def glyph(seed, n=24):
    return np.random.default_rng(seed).integers(0, 255, (n, n, 3), dtype=np.uint8)


class FakeDesktop(ss.Backend):
    """Canvas + windows + mouse. Each window = {info, bg, glyphs:[(rx, ry, seed)]} drawn
    in z-order (last = topmost)."""

    def __init__(self, bounds=(0, 0, 800, 600)):
        self._bounds = bounds
        self.wins: list[dict] = []
        self.clicks, self.focused, self.keys, self.typed = [], [], [], []
        self._cb = None
        self._own = {999}
        self.on_click = None

    def add(self, process, title, rect, bg, glyphs=(), pid=None, topmost=False):
        w = {"info": ss.WinInfo(len(self.wins) + 1, pid or 100 + len(self.wins), process, title, rect),
             "bg": bg, "glyphs": list(glyphs), "topmost": topmost}
        self.wins.append(w)
        return w

    # -- Backend -------------------------------------------------------------
    def bounds(self):
        return self._bounds

    def _canvas(self):
        l, t, r, b = self._bounds
        img = np.full((b - t, r - l, 3), 30, dtype=np.uint8)
        for w in self.wins:
            wl, wt, wr, wb = w["info"].rect
            img[wt - t:wb - t, wl - l:wr - l] = w["bg"]
            for gx, gy, seed in w["glyphs"]:          # glyph position relative to window top-left
                g = glyph(seed)
                y0, x0 = wt - t + gy, wl - l + gx
                img[y0:y0 + 24, x0:x0 + 24] = g
        return img

    def grab(self, box=None):
        img = self._canvas()
        if box is None:
            return img
        l, t, r, b = self._bounds
        x0, y0, x1, y1 = box
        return img[y0 - t:y1 - t, x0 - l:x1 - l].copy()

    def windows(self):
        return [w["info"] for w in reversed(self.wins)]          # topmost first

    def window_at(self, x, y):
        for w in reversed(self.wins):
            l, t, r, b = w["info"].rect
            if l <= x < r and t <= y < b:
                return w["info"]
        return None

    def focus(self, win):
        self.focused.append(win.title)
        for i, w in enumerate(self.wins):
            if w["info"].hwnd == win.hwnd:
                self.wins.append(self.wins.pop(i))
                break
        self.wins.sort(key=lambda w: w["topmost"])                 # always-on-top windows stay on top

    def click(self, x, y, button="left", double=False):
        self.clicks.append((x, y, button, double))
        if self.on_click:
            self.on_click(x, y)

    def hotkey(self, *keys): self.keys.append(keys)
    def type_text(self, text): self.typed.append(text)
    def own_pids(self): return set(self._own)
    def start_listener(self, on_press): self._cb = on_press; return "h"
    def stop_listener(self, handle): self._cb = None

    def press(self, x, y, button="left"):
        self._cb(x, y, button)


def chrome_scene(offset=(0, 0), size=(600, 450)):
    d = FakeDesktop()
    ox, oy = offset
    d.add("notepad.exe", "Untitled - Notepad", (20, 20, 300, 250), (200, 200, 120), [(100, 100, 5)])
    d.add("chrome.exe", "Google Chrome",
          (100 + ox, 50 + oy, 100 + ox + size[0], 50 + oy + size[1]), (240, 240, 245),
          [(size[0] - 40, 4, 77), (40, 100, 78)])                      # glyph 77 = the ✕ button
    d.add("python.exe", "JARVIS HUD", (600, 380, 790, 590), (10, 20, 30),
          [(80, 80, 99)], pid=999, topmost=True)
    return d


def xy_of_close(d):
    for w in d.wins:
        if w["info"].process == "chrome.exe":
            l, t, _, _ = w["info"].rect
            gx, gy, _ = w["glyphs"][0]
            return l + gx + 12, t + gy + 12


class TmpStore:
    def __enter__(self):
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        return ss.SkillStore(root / "ui_skills.json", root / "ui_skills")

    def __exit__(self, *a):
        self.td.cleanup()


def teach_close_chrome(d, store, name="Chrome band karna", labeler=None):
    rec = ss.ClickRecorder(d)
    rec.start()
    x, y = xy_of_close(d)
    d.press(x, y)
    rec.stop()
    return ss.save_skill(store, name, list(rec.events), labeler=labeler), rec


class TestImage(unittest.TestCase):
    def test_find_template_exact_and_shifted(self):
        img = RNG.integers(0, 255, (200, 300, 3), dtype=np.uint8)
        tpl = img[80:128, 150:198].copy()
        hit = ss.find_template(img, tpl, 0.9)
        self.assertEqual(hit[1:], (150, 80))
        self.assertGreater(hit[0], 0.99)

    def test_numpy_fallback_agrees_with_cv2(self):
        img = RNG.integers(0, 255, (160, 220, 3), dtype=np.uint8)
        tpl = img[40:88, 70:118].copy()
        s, l, t = ss._ncc_numpy(ss._gray(img), ss._gray(tpl))
        self.assertEqual((l, t), (70, 40)); self.assertGreater(s, 0.99)

    def test_flat_template_refused(self):
        img = RNG.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        self.assertIsNone(ss.find_template(img, np.full((40, 40, 3), 128, np.uint8), 0.5))

    def test_absent_template_not_found(self):
        img = np.full((120, 160, 3), 50, np.uint8)
        self.assertIsNone(ss.find_template(img, glyph(1, 40), 0.9))

    def test_crop_clamps_at_edges_and_reports_offset(self):
        img = RNG.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        crop, (ox, oy) = ss.crop_around(img, 3, 97, 48)
        self.assertEqual(crop.shape[:2], (48, 48))
        self.assertTrue(np.array_equal(crop[oy, ox], img[97, 3]))


class TestRecorder(unittest.TestCase):
    def test_click_is_recorded_with_window_and_crops(self):
        d = chrome_scene()
        rec = ss.ClickRecorder(d); rec.start()
        x, y = xy_of_close(d); d.press(x, y)
        ev = rec.events[0]
        self.assertEqual(ev.win.process, "chrome.exe")
        self.assertEqual(ev.tight.shape[:2], (48, 48)); self.assertEqual(ev.wide.shape[:2], (120, 120))
        self.assertTrue(np.array_equal(ev.tight[ev.tight_off[1], ev.tight_off[0]], d.grab()[y, x]))

    def test_own_window_clicks_never_recorded(self):
        d = chrome_scene()
        rec = ss.ClickRecorder(d); rec.start()
        d.press(650, 450)                                  # inside the JARVIS HUD window
        self.assertEqual(len(rec.events), 0); self.assertEqual(rec.ignored_own, 1)

    def test_double_click_becomes_one_double_step(self):
        d = chrome_scene()
        rec = ss.ClickRecorder(d); rec.start()
        x, y = xy_of_close(d); d.press(x, y); d.press(x + 1, y)
        self.assertEqual(len(rec.events), 1); self.assertTrue(rec.events[0].double)

    def test_idle_timeout_stops_a_forgotten_session(self):
        d = chrome_scene(); fired = []
        rec = ss.ClickRecorder(d, idle_timeout=0.2, on_timeout=lambda: fired.append(1)); rec.start()
        time.sleep(1.4)
        self.assertEqual(fired, [1]); self.assertFalse(rec.running)

    def test_negative_origin_multi_monitor(self):
        d = FakeDesktop(bounds=(-300, -100, 800, 600))
        d.add("app.exe", "Left Monitor App", (-280, -80, 100, 300), (220, 220, 220), [(20, 20, 41)])
        rec = ss.ClickRecorder(d); rec.start()
        x, y = -280 + 20 + 12, -80 + 20 + 12
        d.press(x, y)
        ev = rec.events[0]
        self.assertTrue(np.array_equal(ev.tight[ev.tight_off[1], ev.tight_off[0]], d.grab()[y + 100, x + 300]))

    def test_live_buffer_is_bounded_and_expires(self):
        d = chrome_scene(); now = [1000.0]
        rec = ss.ClickRecorder(d, keep=5, max_age=100.0, clock=lambda: now[0]); rec.start()
        x, y = xy_of_close(d)
        for i in range(8):
            now[0] += 10; d.press(x + i * 10, y)
        self.assertEqual(len(rec.events), 5)
        now[0] += 500; d.press(x, y)
        self.assertEqual(len(rec.events), 1)                # old ones aged out

    def test_bursts_group_by_gap(self):
        d = chrome_scene(); now = [0.0]
        rec = ss.ClickRecorder(d, keep=60, clock=lambda: now[0]); rec.start()
        x, y = xy_of_close(d)
        for dt in (1, 2, 3, 30, 2, 2):
            now[0] += dt; d.press(x + int(now[0]) % 7, y)
        b = rec.bursts()
        self.assertEqual([len(g) for g in b], [3, 3])


class TestStoreAndSave(unittest.TestCase):
    def test_save_writes_json_and_pictures_and_label(self):
        with TmpStore() as store:
            d = chrome_scene()
            (slug, skill), _ = teach_close_chrome(d, store, labeler=lambda png: "Close button")
            self.assertEqual(slug, "chrome_band_karna")
            st = skill["steps"][0]
            self.assertEqual(st["label"], "Close button"); self.assertEqual(st["window"]["process"], "chrome.exe")
            self.assertTrue((store.crops_dir / st["tight"]).is_file()); self.assertTrue((store.crops_dir / st["wide"]).is_file())
            self.assertIn(slug, store.all())

    def test_resolve_exact_alias_and_fuzzy(self):
        with TmpStore() as store:
            d = chrome_scene()
            teach_close_chrome(d, store, "Chrome band karna")
            sk = store.get("chrome_band_karna"); sk["aliases"] = ["browser band"]; store.put("chrome_band_karna", sk)
            self.assertEqual(store.resolve("Chrome band karna")[0], "chrome_band_karna")
            self.assertEqual(store.resolve("browser band")[0], "chrome_band_karna")
            self.assertEqual(store.resolve("chrome band karo")[0], "chrome_band_karna")
            self.assertIsNone(store.resolve("spotify kholo")[0])

    def test_forget_removes_pictures(self):
        with TmpStore() as store:
            d = chrome_scene(); (slug, _), _ = teach_close_chrome(d, store)
            self.assertTrue(store.forget(slug)); self.assertFalse((store.crops_dir / slug).exists())
            self.assertFalse(store.forget(slug))

    def test_prompt_block_lists_skills(self):
        with TmpStore() as store:
            self.assertEqual(ss.prompt_block(store), "")
            d = chrome_scene(); teach_close_chrome(d, store, "Chrome band karna")
            self.assertIn("Chrome band karna", ss.prompt_block(store))
            self.assertIn("screen_teach", ss.prompt_block(store))


class TestReplay(unittest.TestCase):
    def runner(self, d, store, **kw):
        return ss.SkillRunner(d, store, sleep=lambda s: None, **kw)

    def test_replays_by_picture_even_when_window_moved(self):
        with TmpStore() as store:
            (slug, _), _ = teach_close_chrome(chrome_scene(), store)
            d2 = chrome_scene(offset=(55, 33))                       # same window, elsewhere
            res = self.runner(d2, store).run(slug)
            self.assertTrue(res.ok, res.message)
            self.assertEqual(res.how, ["picture(tight)"])
            self.assertEqual(d2.clicks[0][:2], xy_of_close(d2))
            self.assertIn("Google Chrome", d2.focused)               # window was brought forward

    def test_falls_back_to_position_when_picture_unusable(self):
        with TmpStore() as store:
            d = chrome_scene(); (slug, skill), _ = teach_close_chrome(d, store)
            for p in (store.crops_dir / slug).glob("*.png"):
                p.unlink()
            res = self.runner(chrome_scene(), store).run(slug)
            self.assertTrue(res.ok, res.message); self.assertEqual(res.how, ["position"])

    def test_gemini_locator_used_when_picture_fails(self):
        with TmpStore() as store:
            d = chrome_scene(); (slug, _), _ = teach_close_chrome(d, store, labeler=lambda b: "Close button")
            for p in (store.crops_dir / slug).glob("*.png"):
                p.unlink()
            d2 = chrome_scene(size=(500, 380))                       # resized: position fallback unsafe
            tx, ty = xy_of_close(d2)
            l, t, r, b = next(w for w in d2.wins if w["info"].process == "chrome.exe")["info"].rect
            seen = {}
            def locator(img, label, title):
                seen["label"], seen["title"] = label, title
                return ((tx - l) / (r - l), (ty - t) / (b - t))
            res = self.runner(d2, store, locator=locator).run(slug)
            self.assertTrue(res.ok, res.message); self.assertEqual(res.how, ["gemini"])
            self.assertEqual(seen["label"], "Close button")
            self.assertLessEqual(abs(d2.clicks[0][0] - tx), 1); self.assertLessEqual(abs(d2.clicks[0][1] - ty), 1)

    def test_stops_when_window_missing_no_click(self):
        with TmpStore() as store:
            (slug, _), _ = teach_close_chrome(chrome_scene(), store)
            d2 = FakeDesktop(); d2.add("notepad.exe", "Untitled", (20, 20, 300, 250), (200, 200, 120))
            res = self.runner(d2, store).run(slug)
            self.assertFalse(res.ok); self.assertEqual(d2.clicks, [])
            self.assertIn("chrome.exe", res.message)

    def test_stops_instead_of_clicking_blind(self):
        with TmpStore() as store:
            (slug, _), _ = teach_close_chrome(chrome_scene(), store)
            d2 = chrome_scene(size=(420, 300))                       # resized AND button changed
            next(w for w in d2.wins if w["info"].process == "chrome.exe")["glyphs"][0] = (380, 4, 12345)
            res = self.runner(d2, store).run(slug)
            self.assertFalse(res.ok); self.assertEqual(d2.clicks, [])
            self.assertIn("could not find", res.message)

    def test_never_clicks_jarvis_own_window(self):
        with TmpStore() as store:
            (slug, skill), _ = teach_close_chrome(chrome_scene(), store)
            d2 = chrome_scene()
            x, y = xy_of_close(d2)
            hud = next(w for w in d2.wins if w["info"].pid == 999)
            hud["info"].rect = (x - 30, y - 30, x + 60, y + 60)           # HUD now covers the spot
            for p in (store.crops_dir / slug).glob("*.png"):
                p.unlink()
            res = self.runner(d2, store).run(slug)
            self.assertFalse(res.ok); self.assertEqual(d2.clicks, [])

    def test_other_step_types_and_wait(self):
        with TmpStore() as store:
            store.put("combo", {"name": "combo", "steps": [
                {"type": "hotkey", "keys": ["ctrl", "w"], "delay": 0.1},
                {"type": "type", "text": "hello", "delay": 0.1},
                {"type": "wait", "seconds": 2, "delay": 0.0}]})
            waits = []
            d = FakeDesktop()
            r = ss.SkillRunner(d, store, sleep=waits.append).run("combo")
            self.assertTrue(r.ok); self.assertEqual(d.keys, [("ctrl", "w")]); self.assertEqual(d.typed, ["hello"])
            self.assertIn(2.0, waits)

    def test_abort_stops_between_steps(self):
        with TmpStore() as store:
            store.put("two", {"name": "two", "steps": [{"type": "hotkey", "keys": ["a"], "delay": 0},
                                                         {"type": "hotkey", "keys": ["b"], "delay": 0}]})
            d = FakeDesktop()
            run = ss.SkillRunner(d, store, sleep=lambda s: None)
            run._abort.set()
            r = run.run("two")
            self.assertFalse(r.ok); self.assertEqual(d.keys, [])

    def test_unknown_skill(self):
        with TmpStore() as store:
            self.assertFalse(ss.SkillRunner(FakeDesktop(), store).run("nope").ok)

    def test_use_count_recorded(self):
        with TmpStore() as store:
            (slug, _), _ = teach_close_chrome(chrome_scene(), store)
            self.runner(chrome_scene(), store).run(slug)
            self.assertEqual(store.get(slug)["uses"], 1)

    def test_replay_on_negative_origin_monitor(self):
        with TmpStore() as store:
            def scene(dx=0):
                d = FakeDesktop(bounds=(-300, -100, 800, 600))
                d.add("app.exe", "Left App", (-280 + dx, -80, 100 + dx, 300), (220, 220, 220), [(20, 20, 41)])
                return d
            d = scene(); rec = ss.ClickRecorder(d); rec.start()
            d.press(-280 + 32, -80 + 32)
            slug, _ = ss.save_skill(store, "left close", list(rec.events))
            d2 = scene(dx=40)
            res = self.runner(d2, store).run(slug)
            self.assertTrue(res.ok, res.message)
            self.assertEqual(d2.clicks[0][:2], (-280 + 40 + 32, -80 + 32))


class TestLiveLearning(unittest.TestCase):
    def test_save_last_burst_after_working_normally(self):
        with TmpStore() as store:
            d = chrome_scene(); now = [0.0]
            live = ss.ClickRecorder(d, keep=60, max_age=7200, clock=lambda: now[0]); live.start()
            x, y = xy_of_close(d)
            now[0] += 5; d.press(40 + 100, 150)                    # unrelated click, long ago
            now[0] += 60; d.press(x, y)                            # the thing worth remembering
            bursts = live.bursts()
            self.assertEqual(len(bursts), 2)
            slug, skill = ss.save_skill(store, "chrome band", bursts[-1])
            self.assertEqual(len(skill["steps"]), 1)
            res = ss.SkillRunner(chrome_scene(offset=(10, 10)), store, sleep=lambda s: None).run(slug)
            self.assertTrue(res.ok)

    def test_nothing_on_disk_until_saved(self):
        with TmpStore() as store:
            d = chrome_scene()
            live = ss.ClickRecorder(d, keep=60); live.start()
            x, y = xy_of_close(d); d.press(x, y); d.press(x, y + 80)
            self.assertFalse(store.path.exists()); self.assertFalse(store.crops_dir.exists())

    def test_clear_wipes_buffer(self):
        d = chrome_scene(); live = ss.ClickRecorder(d, keep=60); live.start()
        x, y = xy_of_close(d); d.press(x, y); live.clear()
        self.assertEqual(len(live.events), 0)


class TestSystemBackend(unittest.TestCase):
    def test_constructs_and_degrades_without_windows_apis(self):
        b = ss.SystemBackend()
        self.assertIn(os.getpid(), b.own_pids())
        if sys.platform != "win32":
            self.assertEqual(b.windows(), []); self.assertIsNone(b.window_at(1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
