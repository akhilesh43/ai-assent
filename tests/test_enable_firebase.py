"""python -m unittest tests.test_enable_firebase -v  — fake HTTP + fake credentials, no network."""
import json, os, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import enable_firebase as ef      # noqa: E402

KEY = {"type": "service_account", "project_id": "demo-proj", "private_key": "-----BEGIN-----x",
       "client_email": "a@demo-proj.iam.gserviceaccount.com"}


class Resp:
    def __init__(self, status=200, body=None): self.status_code, self._b, self.text = status, body, ""
    def json(self): return self._b


class FakeHttp:
    """routes: {base_url: status} for the Realtime DB; firestore: status."""
    def __init__(self, rtdb, firestore=404):
        self.rtdb, self.firestore, self.store, self.log = rtdb, firestore, {}, []

    def _base(self, url):
        for b in self.rtdb:
            if url.startswith(b):
                return b
        return None

    def put(self, url, headers=None, json=None, timeout=0):
        self.log.append(("PUT", url)); b = self._base(url)
        if b and self.rtdb[b] == 200:
            self.store[url] = json; return Resp(200)
        return Resp(self.rtdb.get(b, 404))
    def get(self, url, headers=None, timeout=0):
        self.log.append(("GET", url))
        return Resp(200, self.store.get(url))
    def delete(self, url, headers=None, timeout=0):
        self.log.append(("DEL", url)); return Resp(200)
    def patch(self, url, headers=None, json=None, timeout=0):
        return Resp(self.firestore)


class T(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.kp = Path(self.td.name) / "config" / "firebase_service_account.json"
        self.out, self.saved = [], []
        self.base = "https://demo-proj-default-rtdb.firebaseio.com"
        self.asia = "https://demo-proj-default-rtdb.asia-southeast1.firebasedatabase.app"

    def tearDown(self): self.td.cleanup()

    def run_main(self, argv=(), http=None, creds=lambda p: "tok", key=KEY):
        if key is not None:
            self.kp.parent.mkdir(parents=True, exist_ok=True); self.kp.write_text(json.dumps(key))
        return ef.main(list(argv), http=http, creds_factory=creds, key_path=self.kp, out=self.out.append,
                       save_url=lambda u, d: self.saved.append((u, d)), verify_app_path=False)

    def text(self): return "\n".join(self.out)

    def test_default_url_works(self):
        rc = self.run_main(http=FakeHttp({self.base: 200}, firestore=200))
        self.assertEqual(rc, 0); self.assertIn("Realtime Database connected", self.text())
        self.assertIn("Firestore: write / delete OK", self.text())
        self.assertEqual(self.saved, [(self.base, self.base)])

    def test_finds_regional_database_and_saves_url(self):
        rc = self.run_main(http=FakeHttp({self.base: 404, self.asia: 200}))
        self.assertEqual(rc, 0); self.assertEqual(self.saved, [(self.asia, self.base)])
        self.assertIn("another region", self.text())
        self.assertIn("not created (optional)", self.text())

    def test_no_database_gives_the_console_instruction(self):
        rc = self.run_main(http=FakeHttp({}))
        self.assertEqual(rc, 1); self.assertIn("Create database", self.text()); self.assertEqual(self.saved, [])

    def test_permission_denied_is_explained(self):
        rc = self.run_main(http=FakeHttp({self.base: 403}))
        self.assertEqual(rc, 1); self.assertIn("permission denied", self.text())

    def test_bad_token_mentions_clock(self):
        def bad(p): raise RuntimeError("invalid_grant: Invalid JWT Signature")
        rc = self.run_main(http=FakeHttp({self.base: 200}), creds=bad)
        self.assertEqual(rc, 1); self.assertIn("clock", self.text())

    def test_missing_google_auth(self):
        def noauth(p): raise ImportError("google")
        self.assertEqual(self.run_main(http=FakeHttp({}), creds=noauth), 2)
        self.assertIn("pip install google-auth", self.text())

    def test_missing_key(self):
        self.assertEqual(self.run_main(http=FakeHttp({}), key=None), 2); self.assertIn("No key file", self.text())

    def test_not_a_service_account(self):
        self.assertEqual(self.run_main(http=FakeHttp({}), key={"type": "authorized_user"}), 2)
        self.assertIn("service-account", self.text())

    def test_key_flag_copies_file(self):
        src = Path(self.td.name) / "dl.json"; src.write_text(json.dumps(KEY))
        rc = ef.main(["--key", str(src)], http=FakeHttp({self.base: 200}), creds_factory=lambda p: "t",
                     key_path=self.kp, out=self.out.append, save_url=lambda u, d: None, verify_app_path=False)
        self.assertEqual(rc, 0); self.assertTrue(self.kp.is_file())

    def test_url_flag_is_tried_first(self):
        custom = "https://custom.example.firebasedatabase.app"
        http = FakeHttp({custom: 200})
        rc = self.run_main(["--url", custom], http=http)
        self.assertEqual(rc, 0); self.assertEqual(http.log[0][1].split("/jarvis")[0], custom)

    def test_secrets_are_never_printed(self):
        self.run_main(http=FakeHttp({self.base: 200}), creds=lambda p: "SECRET-TOKEN-123")
        self.assertNotIn("SECRET-TOKEN", self.text()); self.assertNotIn("BEGIN", self.text())

    def test_candidate_urls_dedupe_and_order(self):
        u = ef.candidate_urls("p", configured="https://c", forced="https://f")
        self.assertEqual(u[:2], ["https://f", "https://c"]); self.assertEqual(len(u), len(set(u)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
