"""python -m unittest tests.test_screen_teach_action -v  — the voice-facing layer, on the fake desktop."""
import os, sys, tempfile, time, types, unittest, shutil
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests import _stubs; _stubs.install()
from core import screen_skills as ss                     # noqa: E402
from actions import screen_teach as st                   # noqa: E402
from tests.test_screen_skills import chrome_scene, xy_of_close, FakeDesktop   # noqa: E402


class Player:
    def __init__(self): self.fb, self.content = [], []
    def push_feedback(self, t): self.fb.append(t)
    def show_content(self, title, text): self.content.append((title, text))


class Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        st.S.store = ss.SkillStore(root / "s.json", root / "crops")
        st.S.backend = chrome_scene()
        st.S.teach = st.S.live = st.S.runner = None
        st.S.running = False
        self.player = Player()
        self.mirrored = []
        self._p = [mock.patch.object(st, "_mirror", lambda slug, sk: self.mirrored.append(slug)),
                   mock.patch.object(st, "_gemini_label", lambda png: "Close button"),
                   mock.patch.object(st, "_gemini_locate", lambda *a: None),
                   mock.patch.object(ss, "_nap", lambda s: None)]
        for p in self._p: p.start()

    def tearDown(self):
        for p in self._p: p.stop()
        for rec in (st.S.teach, st.S.live):
            if rec: rec.stop()
        if st.S.hint_timer: st.S.hint_timer.cancel()
        self.td.cleanup()

    def act(self, **kw):
        return st.screen_teach_action(kw, player=self.player)

    def click_close(self, dx=0):
        x, y = xy_of_close(st.S.backend)
        st.S.backend.press(x + dx, y)


class TestTeach(Base):
    def test_teach_then_stop_then_run(self):
        self.assertIn("watching", self.act(mode="teach", name="Chrome band karna"))
        self.click_close()
        msg = self.act(mode="stop")
        self.assertIn("Learned 'Chrome band karna'", msg); self.assertIn("Close button", msg)
        self.assertEqual(self.mirrored, ["chrome_band_karna"])
        st.S.backend = chrome_scene(offset=(30, 20))
        out = self.act(mode="run", name="chrome band karo")
        self.assertIn("Done", out)
        x, y = xy_of_close(st.S.backend)
        self.assertEqual(st.S.backend.clicks[0][:2], (x, y))

    def test_name_required(self):
        self.assertIn("call it", self.act(mode="teach", name=""))

    def test_stop_without_clicks(self):
        self.act(mode="teach", name="x y z")
        self.assertIn("didn't see any clicks", self.act(mode="stop"))
        self.assertEqual(st.S.store.all(), {})

    def test_cancel(self):
        self.act(mode="teach", name="abc")
        self.click_close()
        self.assertIn("Cancelled", self.act(mode="cancel")); self.assertEqual(st.S.store.all(), {})

    def test_stop_when_not_learning(self):
        self.assertIn("not learning", self.act(mode="stop"))

    def test_double_start_refused(self):
        self.act(mode="teach", name="abc")
        self.assertIn("Already learning", self.act(mode="teach", name="def"))

    def test_clicks_on_jarvis_window_do_not_count(self):
        self.act(mode="teach", name="abc")
        st.S.backend.press(650, 450)                          # the HUD
        self.assertIn("didn't see any clicks", self.act(mode="stop"))

    def test_idle_timeout_auto_saves(self):
        with mock.patch.object(ss, "TEACH_IDLE_TIMEOUT", 0.2):
            self.act(mode="teach", name="Auto one")
            self.click_close()
            time.sleep(1.5)
        self.assertIn("auto_one", st.S.store.all()); self.assertIsNone(st.S.teach)


class TestRunAndManage(Base):
    def learn(self, name="Chrome band karna"):
        self.act(mode="teach", name=name); self.click_close(); self.act(mode="stop")

    def test_unknown_lists_what_it_knows(self):
        self.learn()
        out = self.act(mode="run", name="spotify kholo")
        self.assertIn("don't know", out); self.assertIn("Chrome band karna", out)

    def test_no_skills_yet(self):
        self.assertIn("haven't learned", self.act(mode="run", name="anything"))

    def test_cannot_run_while_teaching(self):
        self.learn(); self.act(mode="teach", name="other")
        self.assertIn("learning something", self.act(mode="run", name="chrome band karna"))

    def test_run_failure_message_is_reported(self):
        self.learn()
        st.S.backend = FakeDesktop(); st.S.backend.add("notepad.exe", "N", (0, 0, 100, 100), (1, 1, 1))
        out = self.act(mode="run", name="chrome band karna")
        self.assertIn("chrome.exe", out); self.assertEqual(st.S.backend.clicks, [])
        self.assertFalse(st.S.running)

    def test_list_forget(self):
        self.learn()
        self.assertIn("Chrome band karna", self.act(mode="list"))
        self.assertTrue(self.player.content)
        self.assertIn("Forgot", self.act(mode="forget", name="chrome band karna"))
        self.assertIn("No screen skills", self.act(mode="list"))

    def test_add_step(self):
        self.learn()
        self.assertIn("hotkey step", self.act(mode="add_step", name="chrome band karna", step_type="hotkey", keys="ctrl+w"))
        self.assertIn("type step", self.act(mode="add_step", name="chrome band karna", step_type="type", text="hi"))
        self.assertIn("wait step", self.act(mode="add_step", name="chrome band karna", step_type="wait", seconds=99))
        steps = st.S.store.get("chrome_band_karna")["steps"]
        self.assertEqual([s["type"] for s in steps], ["click", "hotkey", "type", "wait"])
        self.assertEqual(steps[1]["keys"], ["ctrl", "w"]); self.assertEqual(steps[3]["seconds"], 30.0)
        self.assertIn("must be", self.act(mode="add_step", name="chrome band karna", step_type="click"))

    def test_unknown_mode(self):
        self.assertIn("Unknown mode", self.act(mode="dance"))


class TestLive(Base):
    def test_watch_save_recent_run_off(self):
        self.assertIn("ON", self.act(mode="watch_on"))
        self.assertIn("already on", self.act(mode="watch_on"))
        self.click_close()
        self.assertIn("1 click", self.act(mode="status"))
        self.assertIn("Saved 'Chrome band karna'", self.act(mode="save_recent", name="Chrome band karna"))
        self.assertIn("Done", self.act(mode="run", name="chrome band karna"))
        self.assertIn("wiped", self.act(mode="watch_off"))
        self.assertIsNone(st.S.live)

    def test_save_recent_needs_live_mode(self):
        self.assertIn("isn't on", self.act(mode="save_recent", name="x y"))

    def test_save_recent_nothing_seen(self):
        self.act(mode="watch_on")
        self.assertIn("haven't seen any clicks", self.act(mode="save_recent", name="x y"))

    def test_watch_off_wipes_and_stops_recording(self):
        self.act(mode="watch_on"); self.click_close()
        live = st.S.live
        self.act(mode="watch_off")
        self.assertEqual(len(live.events), 0); self.assertFalse(live.running)
        self.assertIn("wasn't on", self.act(mode="watch_off"))

    def test_nothing_hits_disk_while_watching(self):
        self.act(mode="watch_on"); self.click_close(); self.click_close(10)
        self.assertFalse(st.S.store.path.exists()); self.assertFalse(st.S.store.crops_dir.exists())

    def test_hint_after_a_sequence(self):
        with mock.patch.object(ss, "BURST_GAP", 0.05):
            self.act(mode="watch_on")
            for i in range(3):
                self.click_close(i * 9); time.sleep(0.01)
            time.sleep(1.4)
        self.assertTrue(any("3 clicks" in f for f in self.player.fb), self.player.fb)


class TestIntegrationBits(Base):
    def test_cloud_assist_can_be_disabled(self):
        for p in self._p: p.stop()
        self._p = []
        with mock.patch.object(st, "_cfg", lambda: {"cloud_assist": False}):
            self.assertEqual(st._gemini_label(b"x"), "")
            self.assertIsNone(st._gemini_locate(None, "a", "b"))

    def test_tool_passes_action_loader_validation(self):
        from core.action_loader import discover_actions
        d = Path(self.td.name) / "acts"; d.mkdir()
        shutil.copy(Path(st.__file__), d / "screen_teach.py")
        reg = discover_actions(d, reserved_names=set(), logger=lambda m: None)
        self.assertTrue(reg.has("screen_teach"))
        decls = [x for x in reg.get_tool_declarations() if x["name"] == "screen_teach"]
        self.assertEqual(decls[0]["parameters"]["required"], ["mode"])

    def test_mirror_sends_text_only_and_trains_command(self):
        mock.patch.stopall()
        for p in self._p: p.stop()
        fs = types.ModuleType("memory.firebase_sync")
        calls = {}
        fs.is_configured = lambda: True
        fs.rtdb_put = lambda path, payload, timeout=6.0: calls.setdefault("put", (path, payload)) and True
        fs.train_command = lambda phrase, target, aliases=None, notes="": calls.setdefault("train", (phrase, target))
        import memory
        with mock.patch.dict(sys.modules, {"memory.firebase_sync": fs}), mock.patch.object(memory, "firebase_sync", fs, create=True):
            skill = {"name": "Chrome band karna", "aliases": [], "steps": [
                {"type": "click", "tight": "a/x.png", "wide": "a/y.png", "label": "Close"}]}
            real = st._mirror.__wrapped__ if hasattr(st._mirror, "__wrapped__") else None
            import importlib; importlib.reload(st)          # un-patch _mirror
            st.S.store = ss.SkillStore(Path(self.td.name) / "m.json", Path(self.td.name) / "mc")
            st._mirror("chrome_band_karna", skill)
            for _ in range(50):
                if "train" in calls: break
                time.sleep(0.05)
        path, payload = calls["put"]
        self.assertEqual(path, "jarvis_ui_skills/chrome_band_karna")
        self.assertNotIn("tight", payload["steps"][0]); self.assertNotIn("wide", payload["steps"][0])
        self.assertEqual(payload["steps"][0]["label"], "Close")
        self.assertEqual(calls["train"][0], "Chrome band karna")
        self.assertIn("screen_teach", calls["train"][1])

    def test_pull_remote_skills_adds_missing_only(self):
        fs = types.ModuleType("memory.firebase_sync")
        fs.is_configured = lambda: True
        fs.rtdb_get = lambda path, timeout=3.0: {
            "remote_one": {"name": "Remote One", "steps": {"0": {"type": "hotkey", "keys": ["a"]}}},
            "chrome_band_karna": {"name": "SHOULD NOT OVERWRITE", "steps": []}}
        self.act(mode="teach", name="Chrome band karna"); self.click_close(); self.act(mode="stop")
        with mock.patch.dict(sys.modules, {"memory.firebase_sync": fs}):
            import memory; memory.firebase_sync = fs
            n = st.pull_remote_skills()
        self.assertEqual(n, 1)
        sk = st.S.store.all()
        self.assertTrue(sk["remote_one"]["remote"]); self.assertEqual(sk["remote_one"]["steps"][0]["type"], "hotkey")
        self.assertEqual(sk["chrome_band_karna"]["name"], "Chrome band karna")

    def test_prompt_block_after_learning(self):
        self.act(mode="teach", name="Chrome band karna"); self.click_close(); self.act(mode="stop")
        self.assertIn("Chrome band karna", ss.prompt_block(st.S.store))


if __name__ == "__main__":
    unittest.main(verbosity=2)
