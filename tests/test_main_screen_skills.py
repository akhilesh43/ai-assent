"""python -m unittest tests.test_main_screen_skills -v — the real JarvisLive.interrupt() reaching a running skill."""
import asyncio, os, sys, tempfile, threading, time, unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests import _stubs; _stubs.install()
import main as app_main                                    # noqa: E402
from core import screen_skills as ss                       # noqa: E402
from actions import screen_teach as st                     # noqa: E402
from tests.test_screen_skills import chrome_scene, FakeDesktop   # noqa: E402


def live():
    jl = app_main.JarvisLive.__new__(app_main.JarvisLive)
    jl.ui = MagicMock(); jl._server_turn_active = False; jl._thinking_since = None
    jl._barge_in_streak = 0; jl._audio_gate = None; jl.audio_in_queue = None
    jl.set_speaking = MagicMock(); jl._visemes = MagicMock(); jl._play_cursor = 0.0
    jl._turn_done_event = None; jl._engine = None
    return jl


class T(unittest.TestCase):
    def test_interrupt_stops_a_running_skill_between_steps(self):
        td = tempfile.TemporaryDirectory()
        store = ss.SkillStore(Path(td.name) / "s.json", Path(td.name) / "c")
        store.put("slow", {"name": "slow", "steps": [{"type": "wait", "seconds": 0.0, "delay": 0.0}] * 6})
        d = FakeDesktop()
        reached = threading.Event()
        def nap(sec):
            reached.set(); time.sleep(0.05)
        runner = ss.SkillRunner(d, store, sleep=nap)
        st.S.runner = runner
        out = {}
        th = threading.Thread(target=lambda: out.setdefault("r", runner.run("slow")))
        th.start()
        self.assertTrue(reached.wait(2))
        live().interrupt()                                  # what Esc / barge-in calls
        th.join(3)
        r = out["r"]
        self.assertFalse(r.ok); self.assertIn("Stopped", r.message); self.assertLess(r.done, 6)
        st.S.runner = None; td.cleanup()

    def test_interrupt_is_harmless_without_the_action_loaded(self):
        saved = sys.modules.pop("actions.screen_teach")
        try:
            live().interrupt()
        finally:
            sys.modules["actions.screen_teach"] = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
