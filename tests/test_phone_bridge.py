"""python -m unittest tests.test_phone_bridge -v"""
import os, sys, tempfile, threading, time, unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory import local_db, reply_store as rs      # noqa: E402
from core import phone_bridge as pb                 # noqa: E402


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = (local_db.DB_PATH, local_db._INITIALIZED_PATH)
        local_db.DB_PATH = Path(self.tmp.name) / "t.db"; local_db._INITIALIZED_PATH = None
        self.clock = Clock()
        self.prompts = []
        self.reply = "haan bhai, 5 min me"
        self.b = pb.PhoneBridge(llm=self._llm, clock=self.clock, owner="Asha")
        self.b._spawn = lambda fn, *a: fn(*a)
        self.addCleanup(self._restore)

    def _llm(self, prompt):
        self.prompts.append(prompt); return self.reply

    def _restore(self):
        local_db.DB_PATH, local_db._INITIALIZED_PATH = self._old; self.tmp.cleanup()

    def fake_phone(self, answer=None, device="p1"):
        """A background thread behaving like the phone's poll loop."""
        got = []
        stop = threading.Event()
        def loop():
            while not stop.is_set():
                for c in self.b.poll(device, wait=0.2, name="Pixel"):
                    got.append(c)
                    self.b.submit_result(device, c["id"], True, (answer or {}).get(c["action"], {}))
        t = threading.Thread(target=loop, daemon=True); t.start()
        self.addCleanup(lambda: (stop.set(), t.join(1)))
        return got

    def wa(self, text="kya kar raha hai?", sender="Rohan", key="k1", **kw):
        ev = {"type": "notification", "key": key, "package": "com.whatsapp", "app": "WhatsApp",
              "title": sender, "sender": sender, "text": text, "can_reply": True}
        ev.update(kw); return ev


class TestCommands(Base):
    def test_no_phone(self):
        self.assertEqual(self.b.send_command("answer_call")["error"], "no phone is connected")

    def test_unknown_action_rejected(self):
        self.assertFalse(self.b.send_command("format_disk")["ok"])

    def test_roundtrip(self):
        got = self.fake_phone({"battery": {"level": 80}})
        time.sleep(0.3)
        r = self.b.send_command("battery", timeout=3)
        self.assertTrue(r["ok"]); self.assertEqual(r["data"], {"level": 80})
        self.assertEqual(got[0]["action"], "battery")

    def test_timeout_when_phone_silent(self):
        self.b.register("p1", "Pixel")
        r = self.b.send_command("answer_call", timeout=0.2)
        self.assertFalse(r["ok"]); self.assertIn("did not answer", r["error"])

    def test_phone_goes_offline(self):
        self.b.register("p1", "Pixel"); self.assertTrue(self.b.connected())
        self.clock.t += pb.ONLINE_WINDOW_S + 1
        self.assertFalse(self.b.connected())
        self.assertFalse(self.b.snapshot()["connected"])


class TestCalls(Base):
    def test_ringing_alerts_once(self):
        alerts = []; self.b.alert_handler = alerts.append
        for _ in range(3):
            self.b.push_event("p1", {"type": "call", "state": "ringing", "name": "Mom", "number": "+91"})
        self.assertEqual(len(alerts), 1); self.assertIn("Mom", alerts[0])
        self.b.push_event("p1", {"type": "call", "state": "idle"})
        self.assertEqual(self.b.snapshot()["call"]["state"], "idle")


class TestReplies(Base):
    def test_default_is_ask_and_draft_waits(self):
        alerts = []; self.b.alert_handler = alerts.append
        got = self.fake_phone()
        self.b.push_event("p1", self.wa())
        p = self.b.pending_list()
        self.assertEqual(len(p), 1); self.assertEqual(p[0]["draft"], self.reply)
        self.assertEqual([c for c in got if c["action"] == "reply_notification"], [])   # nothing sent
        self.assertIn("Rohan", alerts[0])

    def test_approve_sends_and_learns(self):
        got = self.fake_phone(); time.sleep(0.3)
        self.b.push_event("p1", self.wa())
        pid = self.b.pending_list()[0]["id"]
        self.assertTrue(self.b.approve(pid)["ok"])
        sent = [c for c in got if c["action"] == "reply_notification"][0]
        self.assertEqual(sent["args"], {"key": "k1", "text": self.reply})
        ex = rs.examples_for("Rohan")
        self.assertEqual((ex[0]["source"], ex[0]["reply"]), ("approved", self.reply))
        self.assertEqual(self.b.pending_list(), [])

    def test_edit_is_learned_as_edited_and_reaches_next_prompt(self):
        self.fake_phone(); time.sleep(0.3)
        self.b.push_event("p1", self.wa())
        self.b.approve(self.b.pending_list()[0]["id"], text="abhi busy hu yaar, shaam ko baat karte hain")
        self.assertEqual(rs.examples_for("Rohan")[0]["source"], "edited")
        self.clock.t += 100
        self.b.push_event("p1", self.wa(text="free hai?", key="k2"))
        self.assertIn("abhi busy hu yaar", self.prompts[-1])

    def test_auto_mode_for_a_contact_sends_without_asking_and_does_not_learn(self):
        got = self.fake_phone(); time.sleep(0.3)
        rs.set_rule("Rohan", "auto")
        self.b.push_event("p1", self.wa())
        self.assertEqual(len([c for c in got if c["action"] == "reply_notification"]), 1)
        self.assertEqual(self.b.pending_list(), [])
        self.assertEqual(rs.stats()["examples"], 0)            # its own output is not evidence

    def test_other_contacts_still_ask_when_one_is_auto(self):
        self.fake_phone(); rs.set_rule("Rohan", "auto")
        self.b.push_event("p1", self.wa(sender="Boss", key="k9"))
        self.assertEqual(len(self.b.pending_list()), 1)

    def test_ignore_and_off(self):
        rs.set_rule("Spam", "ignore")
        self.b.push_event("p1", self.wa(sender="Spam", key="s"))
        self.b.set_mode("off")
        self.b.push_event("p1", self.wa(sender="Other", key="o"))
        self.assertEqual(self.b.pending_list(), [])
        self.assertEqual(self.prompts, [])                     # no model call either

    def test_groups_ongoing_and_non_replyable_are_skipped(self):
        for kw in ({"is_group": True}, {"ongoing": True}, {"can_reply": False}, {"text": "  "}):
            self.b.push_event("p1", self.wa(key=str(kw), **kw))
        self.assertEqual(self.b.pending_list(), [])
        self.b.push_event("p1", {"type": "notification", "key": "x", "package": "com.bank.app",
                                 "title": "Bank", "text": "OTP 1234", "can_reply": True})
        self.assertEqual(self.b.pending_list(), [])            # not a messaging app

    def test_sensitive_goes_to_owner_never_auto(self):
        self.reply = pb.NEEDS_OWNER
        rs.set_rule("Rohan", "auto"); got = self.fake_phone(); time.sleep(0.3)
        self.b.push_event("p1", self.wa(text="bhej de 5000 abhi"))
        p = self.b.pending_list()
        self.assertTrue(p[0]["needs_owner"]); self.assertEqual(p[0]["draft"], "")
        self.assertEqual([c for c in got if c["action"] == "reply_notification"], [])
        self.assertFalse(self.b.approve(p[0]["id"])["ok"])     # nothing to send until the owner types

    def test_llm_down_means_ask_not_silence(self):
        def boom(_): raise RuntimeError("quota")
        self.b._llm = boom
        self.b.push_event("p1", self.wa())
        self.assertTrue(self.b.pending_list()[0]["needs_owner"])

    def test_rate_limit_per_conversation(self):
        got = self.fake_phone(); time.sleep(0.3); rs.set_rule("Rohan", "auto")
        self.b.push_event("p1", self.wa(text="one"))
        self.b.push_event("p1", self.wa(text="two"))           # 0 s later, same conversation
        self.assertEqual(len([c for c in got if c["action"] == "reply_notification"]), 1)
        self.clock.t += pb.MIN_GAP_S + 1
        self.b.push_event("p1", self.wa(text="three"))
        self.assertEqual(len([c for c in got if c["action"] == "reply_notification"]), 2)

    def test_reposted_notification_is_not_processed_twice(self):
        self.b.push_event("p1", self.wa()); self.b.push_event("p1", self.wa())
        self.assertEqual(len(self.b.pending_list()), 1)

    def test_pending_expires(self):
        self.b.push_event("p1", self.wa())
        self.clock.t += pb.PENDING_TTL_S + 1
        self.assertEqual(self.b.pending_list(), [])

    def test_reject(self):
        self.b.push_event("p1", self.wa())
        pid = self.b.pending_list()[0]["id"]
        self.assertTrue(self.b.reject(pid)); self.assertFalse(self.b.reject(pid))

    def test_failed_send_keeps_the_draft_waiting(self):
        self.b.push_event("p1", self.wa())                     # phone registered but never polls
        pid = self.b.pending_list()[0]["id"]
        orig = self.b.send_command
        self.b.send_command = lambda *a, **k: {"ok": False, "error": "x"}
        self.assertFalse(self.b.approve(pid)["ok"])
        self.b.send_command = orig
        self.assertEqual(len(self.b.pending_list()), 1)


class TestStyleStore(Base):
    def test_taught_examples_and_priority(self):
        rs.add_example("kaisa hai", "badiya, tu bata", contact="Rohan", source="approved")
        rs.add_example("kaisa hai", "mast bhai 😄", contact="Rohan", source="edited")
        rs.add_example("hi", "hello ji", contact="Mom", source="taught")
        ex = rs.examples_for("Rohan")
        self.assertEqual(ex[0]["source"], "edited")             # strongest evidence first
        self.assertEqual({e["contact"] for e in ex}, {"rohan", "mom"})

    def test_duplicates_and_bad_input(self):
        self.assertTrue(rs.add_example("a", "b"))
        self.assertFalse(rs.add_example("a", "b"))
        self.assertFalse(rs.add_example("", "b")); self.assertFalse(rs.add_example("a", "b", source="auto"))

    def test_prompt_contains_style_rules_and_escape_hatch(self):
        p = pb.build_reply_prompt("hi", "Rohan", "WhatsApp", [], owner="Asha", persona="short, no emoji")
        for s in ("Asha", "short, no emoji", pb.NEEDS_OWNER, "ONLY the reply"):
            self.assertIn(s, p)

    def test_clean_draft(self):
        self.assertEqual(pb.clean_draft('Owner: "ok done"'), "ok done")
        self.assertIsNone(pb.clean_draft("[NEEDS_OWNER]")); self.assertIsNone(pb.clean_draft(""))
        self.assertIsNone(pb.clean_draft("x" * 700))

    def test_teach_and_modes(self):
        self.assertTrue(self.b.teach("kab aayega", "8 baje tak", contact="Mom"))
        with self.assertRaises(ValueError):
            self.b.set_mode("yolo")
        self.assertEqual(self.b.set_mode("auto"), "auto"); self.assertEqual(self.b.snapshot()["mode"], "auto")


if __name__ == "__main__":
    unittest.main()


class TestDevicesAndPlugin(Base):
    def test_device_tokens_are_hashed_and_revocable(self):
        from core import phone_devices as pd
        path = Path(self.tmp.name) / "devs.json"
        dev_id, tok = pd.issue("Pixel 8", path)
        self.assertNotIn(tok, path.read_text())                 # only the hash is stored
        self.assertEqual(pd.verify(tok, path)["id"], dev_id)
        self.assertIsNone(pd.verify(tok + "x", path)); self.assertIsNone(pd.verify("", path))
        self.assertEqual(pd.revoke_all(path), 1); self.assertIsNone(pd.verify(tok, path))

    def test_plugin_flow_through_the_shared_bridge(self):
        import importlib
        plug = importlib.import_module("plugins.phone_control")
        old = pb._BRIDGE; pb._BRIDGE = self.b
        self.addCleanup(lambda: setattr(pb, "_BRIDGE", old))
        self.assertIn("No phone is connected", plug.run({"action": "status"}))
        self.fake_phone({"answer_call": {"message": "Answered."}, "battery": {"level": 55, "charging": False}})
        time.sleep(0.3)
        self.assertEqual(plug.run({"action": "answer_call"}), "Answered.")
        self.assertIn("55%", plug.run({"action": "battery"}))
        self.b.push_event("p1", self.wa())
        self.assertIn("Rohan", plug.run({"action": "list_pending"}))
        self.assertIn("Sent to Rohan", plug.run({"action": "send_pending", "who": "rohan"}))
        self.assertIn("auto-reply is ask", plug.run({"action": "status"}))
        self.assertIn("now auto", plug.run({"action": "auto_reply", "mode": "auto"}))
        self.assertIn("Use auto", plug.run({"action": "auto_reply", "mode": "banana"}))
        self.assertIn("answer like that", plug.run({"action": "teach", "who": "Mom", "text": "ok beta", "incoming": "khana khaya?"}))
