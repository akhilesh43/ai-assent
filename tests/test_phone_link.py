"""python -m unittest tests.test_phone_link -v — real HTTP on localhost, real bridge, real sqlite."""
import asyncio, json, os, sys, tempfile, threading, time, unittest, urllib.request, urllib.error
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory import local_db                          # noqa: E402
from core import phone_link as pl, phone_bridge as pb  # noqa: E402


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = (local_db.DB_PATH, local_db._INITIALIZED_PATH)
        local_db.DB_PATH = Path(self.tmp.name) / "t.db"; local_db._INITIALIZED_PATH = None
        self.clock = Clock()
        self.bridge = pb.PhoneBridge(llm=lambda p: "ok bhai", clock=self.clock)
        self.bridge._spawn = lambda fn, *a: fn(*a)
        self.srv = pl.PhoneLinkServer(self.bridge, clock=self.clock,
                                      devices_path=Path(self.tmp.name) / "devs.json",
                                      ips=lambda: ["192.168.1.10", "10.0.0.4"])
        self.addCleanup(self._restore)

    def _restore(self):
        self.srv.stop()
        local_db.DB_PATH, local_db._INITIALIZED_PATH = self._old; self.tmp.cleanup()

    def pair(self, name="Pixel", who="1.2.3.4"):
        key = self.srv.new_key()
        code, d = self.srv.route("POST", "/api/app-pair", {}, {"key": key, "name": name}, "", who)
        self.assertEqual(code, 200, d)
        return d


class TestPairing(Base):
    def test_key_is_one_time_and_gives_tokens(self):
        key = self.srv.new_key()
        c, d = self.srv.route("POST", "/api/app-pair", {}, {"key": key.lower(), "name": "Pixel"}, "", "a")
        self.assertEqual(c, 200); self.assertTrue(d["token"] and d["device_token"])
        c2, _ = self.srv.route("POST", "/api/app-pair", {}, {"key": key}, "", "a")
        self.assertEqual(c2, 401)                              # spent

    def test_expired_key(self):
        key = self.srv.new_key(expiry_secs=10); self.clock.t += 11
        self.assertEqual(self.srv.route("POST", "/api/app-pair", {}, {"key": key}, "", "a")[0], 401)

    def test_lockout_after_five_wrong_keys_then_recovers(self):
        for _ in range(5):
            self.assertEqual(self.srv.route("POST", "/api/app-pair", {}, {"key": "WRONG123"}, "", "x")[0], 401)
        good = self.srv.new_key()
        self.assertEqual(self.srv.route("POST", "/api/app-pair", {}, {"key": good}, "", "x")[0], 429)
        self.assertEqual(self.srv.route("POST", "/api/app-pair", {}, {"key": good}, "", "other")[0], 200)  # other address fine
        self.clock.t += 61
        self.assertEqual(self.srv.route("POST", "/api/app-login", {}, {}, "", "x")[0], 401)

    def test_device_token_logs_in_again_after_restart(self):
        d = self.pair()
        fresh = pl.PhoneLinkServer(self.bridge, clock=self.clock, devices_path=self.srv._devices_path)
        c, r = fresh.route("POST", "/api/app-login", {}, {"device_token": d["device_token"]}, "", "a")
        self.assertEqual(c, 200); self.assertEqual(r["device_id"], d["device_id"])
        self.assertEqual(fresh.route("POST", "/api/app-login", {}, {"device_token": "x" * 40}, "", "a")[0], 401)

    def test_connect_callback_fires(self):
        hits = []; self.srv.set_connect_callback(lambda: hits.append(1)); self.pair()
        self.assertEqual(hits, [1])

    def test_keys_use_no_ambiguous_characters(self):
        for _ in range(200):
            k = self.srv.new_key(); self.assertEqual(len(k), 8); self.assertFalse(set(k) & set("01OIL"))


class TestAuthAndRoutes(Base):
    def test_ping_open_everything_else_closed(self):
        self.assertEqual(self.srv.route("GET", "/api/ping", {}, {}, "")[1]["jarvis"], True)
        for m, p in (("GET", "/api/phone/state"), ("POST", "/api/command"), ("GET", "/api/log"),
                     ("GET", "/api/insights"), ("GET", "/api/phone/poll")):
            self.assertEqual(self.srv.route(m, p, {}, {"text": "x"}, "")[0], 401, p)
            self.assertEqual(self.srv.route(m, p, {}, {"text": "x"}, "Bearer nope")[0], 401, p)

    def test_token_expires(self):
        d = self.pair(); auth = "Bearer " + d["token"]
        self.assertEqual(self.srv.route("GET", "/api/phone/state", {}, {}, auth)[0], 200)
        self.clock.t += pl.TOKEN_TTL_S + 1
        self.assertEqual(self.srv.route("GET", "/api/phone/state", {}, {}, auth)[0], 401)

    def test_unknown_path_and_exceptions_are_contained(self):
        d = self.pair(); auth = "Bearer " + d["token"]
        self.assertEqual(self.srv.route("GET", "/nope", {}, {}, auth)[0], 404)
        self.srv.bridge = None                                  # something broken inside
        self.assertEqual(self.srv.route("GET", "/api/phone/state", {}, {}, auth)[0], 500)

    def test_commands_reach_the_assistants_queue(self):
        async def go():
            self.srv._loop = asyncio.get_running_loop()
            d = self.pair(); auth = "Bearer " + d["token"]
            self.assertEqual(self.srv.route("POST", "/api/command", {}, {"text": "  open chrome "}, auth)[0], 200)
            self.assertEqual(self.srv.route("POST", "/api/command", {}, {"text": "  "}, auth)[0], 400)
            return await asyncio.wait_for(self.srv._command_queue.get(), 1)
        self.assertEqual(asyncio.run(go()), "open chrome")

    def test_log_feed_and_status(self):
        d = self.pair(); auth = "Bearer " + d["token"]
        async def feed():
            await self.srv.broadcast({"type": "log", "speaker": "user", "text": "hello"})
            await self.srv.broadcast({"type": "log", "speaker": "jarvis", "text": "hi!"})
            await self.srv.broadcast({"type": "status", "state": "thinking"})
        asyncio.run(feed())
        _, r = self.srv.route("GET", "/api/log", {"since": "0"}, {}, auth)
        self.assertEqual([x["text"] for x in r["items"]], ["hello", "hi!"]); self.assertEqual(r["status"], "thinking")
        _, r2 = self.srv.route("GET", "/api/log", {"since": str(r["next"])}, {}, auth)
        self.assertEqual(r2["items"], [])
        _, r3 = self.srv.route("GET", "/api/log", {"since": "999"}, {}, auth)   # laptop restarted: resend
        self.assertEqual(len(r3["items"]), 2)

    def test_phone_flow_through_the_routes(self):
        d = self.pair(); auth = "Bearer " + d["token"]; dev = d["device_id"]
        # the phone reports a WhatsApp message → draft waits for approval
        self.srv.route("POST", "/api/phone/event", {}, {"device": dev, "events": [
            {"type": "notification", "key": "k", "package": "com.whatsapp", "app": "WhatsApp",
             "title": "Rohan", "sender": "Rohan", "text": "kya kar raha hai?", "can_reply": True}]}, auth)
        _, st = self.srv.route("GET", "/api/phone/state", {}, {}, auth)
        self.assertTrue(st["connected"]); self.assertEqual(st["pending"][0]["draft"], "ok bhai")
        # approve: the phone's poll loop receives the reply command and reports success
        got = []
        def phone():
            for c in self.bridge.poll(dev, wait=3):
                got.append(c); self.srv.route("POST", "/api/phone/result", {}, {"device": dev, "id": c["id"], "ok": True}, auth)
        t = threading.Thread(target=phone); t.start(); time.sleep(0.2)
        _, r = self.srv.route("POST", "/api/phone/approve", {}, {"id": st["pending"][0]["id"]}, auth)
        t.join(5)
        self.assertTrue(r["ok"], r); self.assertEqual(got[0]["args"], {"key": "k", "text": "ok bhai"})
        self.assertEqual(self.srv.route("POST", "/api/phone/mode", {}, {"mode": "auto"}, auth)[1]["mode"], "auto")
        self.assertEqual(self.srv.route("POST", "/api/phone/mode", {}, {"mode": "wat"}, auth)[0], 400)
        self.assertTrue(self.srv.route("POST", "/api/phone/teach", {}, {"incoming": "hi", "reply": "hello ji"}, auth)[1]["ok"])


class TestRealSockets(Base):
    def req(self, path, body=None, auth="", port=None):
        r = urllib.request.Request(f"http://127.0.0.1:{port or self.srv.port}{path}",
                                   data=None if body is None else json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json", "Authorization": auth},
                                   method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(r, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_end_to_end_over_a_real_socket(self):
        self.assertTrue(self.srv.start()); self.assertIn(self.srv.port, pl.PORTS)
        self.assertEqual(self.req("/api/ping")[1]["jarvis"], True)
        key = self.srv.new_key()
        c, d = self.req("/api/app-pair", {"key": key, "name": "Pixel"})
        self.assertEqual(c, 200)
        auth = "Bearer " + d["token"]
        self.assertEqual(self.req("/api/phone/state", auth=auth)[0], 200)
        self.assertEqual(self.req("/api/phone/state")[0], 401)
        self.assertEqual(self.req("/api/app-pair", {"key": key})[0], 401)

    def test_long_poll_does_not_block_other_requests(self):
        self.srv.start()
        d = self.pair(); auth = "Bearer " + d["token"]
        res = {}
        t = threading.Thread(target=lambda: res.update(r=self.req("/api/phone/poll?device=p9&wait=2", auth=auth)))
        t.start(); time.sleep(0.3)
        t0 = time.time(); self.assertEqual(self.req("/api/ping")[0], 200)
        self.assertLess(time.time() - t0, 1.0)                  # answered while the poll was waiting
        rid = self.bridge.queue_command("battery", device="p9"); self.assertTrue(rid)
        t.join(5); self.assertEqual(res["r"][1]["commands"][0]["action"], "battery")

    def test_oversized_and_malformed_bodies(self):
        self.srv.start()
        r = urllib.request.Request(f"http://127.0.0.1:{self.srv.port}/api/app-pair", data=b"{not json", method="POST",
                                   headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(r, timeout=5)
        self.assertEqual(cm.exception.code, 400)

    def test_busy_port_moves_to_the_next(self):
        a = pl.PhoneLinkServer(self.bridge, ips=lambda: ["192.168.1.10"]); self.assertTrue(a.start())
        self.addCleanup(a.stop)
        self.assertTrue(self.srv.start()); self.assertNotEqual(a.port, self.srv.port)

    def test_info_for_the_qr(self):
        self.srv.start()
        i = self.srv.info("ABCD2345")
        self.assertEqual(i["url"], f"http://192.168.1.10:{self.srv.port}/pair?key=ABCD2345&alt=10.0.0.4")
        self.assertEqual(i["manual"], f"192.168.1.10:{self.srv.port}")
        noip = pl.PhoneLinkServer(self.bridge, ips=lambda: [])
        noip.start(); self.addCleanup(noip.stop)
        self.assertIn("No Wi-Fi", noip.info()["error"])

    def test_lan_ips_are_private(self):
        for ip in pl.lan_ips():
            a, b = map(int, ip.split(".")[:2])
            self.assertTrue(a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168))



class TestLiveContact(Base):
    def test_any_request_counts_as_contact_and_is_reported_once_per_phone(self):
        seen = []; self.srv.on_contact = seen.append
        self.assertIsNone(self.srv.contact())
        self.srv.route("GET", "/api/ping", {}, {}, "", "192.168.1.23")
        self.srv.route("GET", "/api/ping", {}, {}, "", "192.168.1.23")
        self.srv.route("GET", "/api/ping", {}, {}, "", "192.168.1.99")
        self.assertEqual(seen, ["192.168.1.23", "192.168.1.99"])
        self.clock.t += 30
        c = self.srv.contact(); self.assertEqual((c["who"], c["ago"], c["path"]), ("192.168.1.99", 30, "/api/ping"))

    def test_even_a_rejected_request_is_contact(self):
        self.srv.route("POST", "/api/app-pair", {}, {"key": "NOPE"}, "", "10.1.1.1")
        self.assertEqual(self.srv.contact()["who"], "10.1.1.1")

    def test_ping_and_info_carry_the_build_and_test_url(self):
        self.assertEqual(self.srv.route("GET", "/api/ping", {}, {}, "")[1]["build"], pl.BUILD)
        self.srv.start()
        i = self.srv.info("ABCD2345")
        self.assertEqual(i["test_url"], f"http://192.168.1.10:{self.srv.port}/api/ping"); self.assertEqual(i["build"], pl.BUILD)

    def test_firewall_helpers_are_harmless_off_windows(self):
        self.assertTrue(pl.windows_firewall_ok()); self.assertFalse(pl.run_firewall_fix("x.bat"))

    def test_active_server_is_registered(self):
        self.assertIs(pl.get_active(), self.srv)


if __name__ == "__main__":
    unittest.main()
