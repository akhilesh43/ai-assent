"""
python -m unittest tests.test_workspace_data -v

Tasks, human behaviour, brain stats and the insights snapshot — real code, real sqlite in a temp dir.
"""
import os, sys, tempfile, unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory import local_db, tasks_store as ts   # noqa: E402
from core import behavior                        # noqa: E402


class Db(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = (local_db.DB_PATH, local_db._INITIALIZED_PATH)
        local_db.DB_PATH = Path(self.tmp.name) / "t.db"
        local_db._INITIALIZED_PATH = None
        self.addCleanup(self._restore)

    def _restore(self):
        local_db.DB_PATH, local_db._INITIALIZED_PATH = self._old
        self.tmp.cleanup()


class TestParseDue(unittest.TestCase):
    T = date(2026, 11, 15)          # a Sunday

    def test_words(self):
        f = lambda s: ts.parse_due(s, self.T)
        self.assertEqual(f("today"), "2026-11-15")
        self.assertEqual(f("kal"), "2026-11-16")
        self.assertEqual(f("parso"), "2026-11-17")
        self.assertEqual(f("in 3 days"), "2026-11-18")
        self.assertEqual(f("in 2 weeks"), "2026-11-29")
        self.assertEqual(f("friday"), "2026-11-20")
        self.assertEqual(f("sunday"), "2026-11-22")          # never "today" for a weekday name
        self.assertEqual(f("2026-12-01"), "2026-12-01")

    def test_garbage_is_empty_not_an_error(self):
        for s in ("", "someday", "2026-13-45", "blah blah"):
            self.assertEqual(ts.parse_due(s, self.T), "")


class TestTasks(Db):
    def test_add_list_done_roundtrip(self):
        a = ts.add("Write report", due="tomorrow", priority=1, tags=["Work", "work", "q4"])
        self.assertEqual(a["status"], "todo")
        self.assertEqual(a["tags"], ["q4", "work"])
        self.assertEqual(a["priority_name"], "high")
        self.assertEqual(len(ts.list_tasks()), 1)
        d = ts.set_status(a["id"], "done")
        self.assertEqual(d["status"], "done"); self.assertTrue(d["done_at"])
        self.assertEqual(ts.list_tasks("open"), [])
        self.assertEqual(len(ts.list_tasks("done")), 1)
        self.assertEqual(ts.set_status(a["id"], "todo")["done_at"], "")

    def test_empty_title_rejected(self):
        with self.assertRaises(ValueError):
            ts.add("   ")

    def test_overdue_sorts_first_and_is_flagged(self):
        past = (date.today() - timedelta(days=3)).isoformat()
        ts.add("important", priority=1)
        late = ts.add("late one", due=past, priority=3)
        self.assertTrue(late["overdue"])
        self.assertEqual(ts.list_tasks()[0]["title"], "late one")

    def test_find_by_spoken_words_and_id(self):
        a = ts.add("Call the dentist"); ts.add("Buy milk")
        self.assertEqual(ts.find("dentist call")["id"], a["id"])
        self.assertEqual(ts.find(f"#{a['id']}")["id"], a["id"])
        self.assertIsNone(ts.find("zebra"))

    def test_update_ignores_bad_fields(self):
        a = ts.add("x1")
        b = ts.update(a["id"], status="weird", priority=9, nonsense="q", due="friday")
        self.assertEqual(b["status"], "todo"); self.assertEqual(b["priority"], 2)
        self.assertTrue(b["due"])

    def test_delete(self):
        a = ts.add("temp")
        self.assertTrue(ts.delete(a["id"])); self.assertFalse(ts.delete(a["id"]))

    def test_summary_counts_and_streak(self):
        a, b, c = ts.add("one"), ts.add("two", due="2000-01-01"), ts.add("three")
        ts.set_status(a["id"], "done"); ts.set_status(c["id"], "doing")
        s = ts.summary()
        self.assertEqual((s["total"], s["open"], s["done"], s["doing"], s["todo"]), (3, 2, 1, 1, 1))
        self.assertEqual(s["overdue"], 1)
        self.assertEqual(s["done_today"], 1)
        self.assertEqual(s["streak_days"], 1)
        self.assertEqual(s["completion_pct"], 33)
        self.assertEqual(sum(s["done_per_day"].values()), 1)

    def test_summary_empty_board(self):
        s = ts.summary()
        self.assertEqual((s["total"], s["completion_pct"], s["streak_days"]), (0, 0, 0))

    def test_clear_done(self):
        a = ts.add("a"); ts.add("b"); ts.set_status(a["id"], "done")
        self.assertEqual(ts.clear_done(), 1); self.assertEqual(len(ts.list_tasks("all")), 1)


class TestBehavior(Db):
    def test_bond_grows_and_names_a_stage(self):
        self.assertEqual(behavior.snapshot()["bond"]["name"], "Stranger")
        for _ in range(40):
            behavior.observe("open chrome please now", "neutral")
        b = behavior.snapshot()["bond"]
        self.assertEqual(b["name"], "Acquaintance")
        self.assertTrue(0 <= b["progress"] <= 1)

    def test_personal_sharing_grows_bond_faster(self):
        behavior.observe("hello there")
        a = behavior.snapshot()["bond"]["xp"]
        behavior.observe("my mom is sick and i feel really worried today")
        self.assertGreaterEqual(behavior.snapshot()["bond"]["xp"] - a, 4)

    def test_traits_drift_slowly_and_stay_in_range(self):
        for _ in range(100):
            behavior.observe("haha lol yaar", "happy")
        tr = behavior.snapshot()["traits"]
        self.assertEqual(tr["humour"], 1.0)
        self.assertLess(tr["formality"], 0.2)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in tr.values()))
        behavior.observe("hi", "happy")                        # one more turn moves at most one step
        self.assertLessEqual(abs(behavior.snapshot()["traits"]["humour"] - tr["humour"]), 0.021)

    def test_short_commands_make_it_concise(self):
        for _ in range(25):
            behavior.observe("play music", "neutral")
        self.assertGreater(behavior.snapshot()["traits"]["brevity"], 0.65)
        self.assertIn("very short", behavior.build_prompt_block())

    def test_mood_follows_recent_emotion(self):
        for _ in range(3):
            local_db.record_emotion("sad", 0.9, "udaas hoon")
        self.assertEqual(behavior.snapshot()["mood"], "gentle")
        self.assertIn("No jokes", behavior.build_prompt_block())

    def test_streak_counts_consecutive_days_only(self):
        behavior.observe("hi")
        self.assertEqual(behavior.snapshot()["streak_days"], 1)
        behavior.observe("hi again")                           # same day: no change
        self.assertEqual(behavior.snapshot()["streak_days"], 1)
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        behavior._put_many({"last_day": yesterday})
        behavior.observe("hi")
        self.assertEqual(behavior.snapshot()["streak_days"], 2)
        behavior._put_many({"last_day": (date.today() - timedelta(days=5)).isoformat()})
        behavior.observe("hi")
        s = behavior.snapshot()
        self.assertEqual((s["streak_days"], s["longest_streak"]), (1, 2))

    def test_absence_is_noticed(self):
        behavior.observe("hi")
        behavior._put_many({"last_seen": (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()})
        self.assertIn("4 days", behavior.build_prompt_block())

    def test_prompt_block_has_tasks_and_no_recitable_lines(self):
        block = behavior.build_prompt_block(task_lines=["File taxes (OVERDUE)"])
        self.assertIn("File taxes", block)
        self.assertIn("never use these as a script", block)
        self.assertTrue(block.startswith("[HUMAN BEHAVIOUR"))

    def test_observe_never_raises_on_garbage(self):
        for x in ("", None, "😂" * 500, "\x00"):
            behavior.observe(x)


class TestInsights(Db):
    def test_snapshot_sections_and_isolation(self):
        from core import insights
        ts.add("a task")
        local_db.record_turn("user", "hello"); local_db.record_turn("assistant", "hi")
        local_db.save_fact("pet", "cat named Momo", "preference")
        snap = insights.snapshot(include_system=False)
        for k in ("brain", "tasks", "mind", "activity", "generated"):
            self.assertIn(k, snap)
            self.assertNotIn("error", snap[k], snap[k])
        self.assertEqual(snap["tasks"]["open"], 1)
        self.assertEqual(snap["brain"]["facts"], 1)
        self.assertEqual(snap["brain"]["conversation_turns"], 2)
        self.assertEqual(snap["activity"]["total_week"], 2)
        self.assertEqual(len(snap["activity"]["days"]), 7)
        self.assertEqual(snap["brain"]["level"]["name"], "Newborn")

    def test_brain_levels(self):
        from core import insights
        self.assertEqual(insights.brain_level(0)["name"], "Newborn")
        self.assertEqual(insights.brain_level(50)["name"], "Sharp")
        top = insights.brain_level(99999)
        self.assertEqual((top["name"], top["progress"]), ("Sage", 1.0))

    def test_a_broken_section_does_not_blank_the_rest(self):
        from core import insights
        orig = ts.summary
        ts.summary = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            snap = insights.snapshot(include_system=False)
        finally:
            ts.summary = orig
        self.assertIn("error", snap["tasks"]); self.assertNotIn("error", snap["brain"])

    def test_brain_capacity_raised(self):
        from memory import memory_manager as mm
        self.assertGreaterEqual(mm.PROMPT_CORE_CHARS, 3000)
        self.assertGreaterEqual(mm.PROMPT_MAX_PER_CATEGORY, 12)
        for i in range(60):
            local_db.save_fact(f"k{i}", f"value {i}", "general")
        ctx = local_db.build_db_context()
        self.assertGreaterEqual(ctx.count("[general]"), 48)


class TestPlugin(Db):
    def run_p(self, **kw):
        import importlib; m = importlib.import_module("plugins.task_board")
        return m.run(kw)

    def test_voice_flow(self):
        self.assertIn("Added", self.run_p(action="add", title="Pay rent", due="kal", priority="high"))
        self.assertIn("Pay rent", self.run_p(action="list"))
        self.assertIn("Started", self.run_p(action="start", title="rent"))
        self.assertIn("Done", self.run_p(action="done", title="pay rent"))
        self.assertIn("Everything is done", self.run_p(action="list"))
        self.assertIn("couldn't find", self.run_p(action="done", title="unicorn"))

    def test_bad_date_is_explained(self):
        self.assertIn("couldn't read", self.run_p(action="add", title="x task", due="whenever"))

    def test_plugin_contract(self):
        import importlib, re
        m = importlib.import_module("plugins.task_board")
        self.assertTrue(re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,63}", m.PLUGIN["name"]))
        self.assertEqual(m.PLUGIN["parameters"]["type"], "OBJECT")



class TestTraitControl(Db):
    def test_set_and_reset(self):
        self.assertEqual(behavior.set_trait("humour", 1.4)["humour"], 1.0)
        self.assertEqual(behavior.snapshot()["traits"]["humour"], 1.0)
        with self.assertRaises(ValueError):
            behavior.set_trait("sarcasm", 0.5)
        self.assertEqual(behavior.reset_traits(), behavior.DEFAULT_TRAITS)
        self.assertEqual(behavior.snapshot()["traits"]["humour"], 0.6)


if __name__ == "__main__":
    unittest.main()
