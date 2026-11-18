"""python -m unittest tests.test_image_gen -v   (no network, no API key: backends are faked)"""
import base64, os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import image_gen as ig

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig = dict(ig._BACKENDS)
        self._cfg = ig._cfg
        ig._cfg = lambda: {}

    def tearDown(self):
        ig._BACKENDS.clear(); ig._BACKENDS.update(self._orig)
        ig._cfg = self._cfg
        ig._listeners.clear()

    def test_first_backend_wins_and_file_saved(self):
        ig._BACKENDS["gemini"] = lambda p, a, c, l: (PNG, "image/png", "gemini-x")
        ig._BACKENDS["pollinations"] = lambda *a: self.fail("should not be called")
        r = ig.generate_image("a red apple on a table", out_dir=self.tmp)
        self.assertTrue(r["ok"])
        self.assertEqual(r["backend"], "gemini-x")
        self.assertTrue(Path(r["path"]).read_bytes().startswith(b"\x89PNG"))
        self.assertTrue(r["name"].endswith(".png"))
        lst = ig.list_images(out_dir=self.tmp)
        self.assertEqual(lst[0]["prompt"], "a red apple on a table")

    def test_falls_back_when_first_fails(self):
        def boom(*a): raise RuntimeError("429 quota exceeded limit 0")
        ig._BACKENDS["gemini"] = boom
        ig._BACKENDS["pollinations"] = lambda p, a, c, l: (PNG, "image/jpeg", "pollinations/flux")
        r = ig.generate_image("a neon city", out_dir=self.tmp)
        self.assertTrue(r["ok"])
        self.assertEqual(r["backend"], "pollinations/flux")

    def test_all_fail_gives_readable_error(self):
        def boom(*a): raise RuntimeError("nope")
        ig._BACKENDS["gemini"] = boom; ig._BACKENDS["pollinations"] = boom
        r = ig.generate_image("a neon city", out_dir=self.tmp)
        self.assertFalse(r["ok"])
        self.assertIn("gemini", r["error"]); self.assertIn("pollinations", r["error"])

    def test_non_image_bytes_rejected(self):
        ig._BACKENDS["gemini"] = lambda p, a, c, l: (b"<html>error</html>" * 20, "image/png", "m")
        ig._BACKENDS["pollinations"] = lambda p, a, c, l: (PNG, "image/png", "p")
        r = ig.generate_image("something nice", out_dir=self.tmp)
        self.assertEqual(r["backend"], "p")

    def test_short_prompt_refused(self):
        self.assertFalse(ig.generate_image("a", out_dir=self.tmp)["ok"])

    def test_style_and_aspect_reach_backend(self):
        seen = {}
        def be(p, a, c, l): seen.update(p=p, a=a); return PNG, "image/png", "m"
        ig._BACKENDS["gemini"] = be
        ig.generate_image("a cat", style="anime", aspect="9:16", out_dir=self.tmp)
        self.assertIn("Style: anime", seen["p"]); self.assertEqual(seen["a"], "9:16")
        ig.generate_image("a cat", aspect="bogus", out_dir=self.tmp)
        self.assertEqual(seen["a"], "1:1")

    def test_listener_notified_without_bytes(self):
        got = []
        ig.add_listener(got.append)
        ig._BACKENDS["gemini"] = lambda p, a, c, l: (PNG, "image/png", "m")
        ig.generate_image("a lighthouse", out_dir=self.tmp)
        self.assertEqual(got[0]["prompt"], "a lighthouse"); self.assertNotIn("bytes", got[0])

    def test_filenames_never_collide(self):
        ig._BACKENDS["gemini"] = lambda p, a, c, l: (PNG, "image/png", "m")
        a = ig.generate_image("same prompt here", out_dir=self.tmp)["path"]
        b = ig.generate_image("same prompt here", out_dir=self.tmp)["path"]
        self.assertNotEqual(a, b)

    def test_gemini_backend_parses_inline_data(self):
        """Drive the real _gemini_backend with fake google modules."""
        import types as T_
        fake_types = T_.SimpleNamespace(
            ImageConfig=lambda **k: k,
            GenerateContentConfig=lambda **k: k)
        class Blob: data = PNG; mime_type = "image/png"
        class Part: inline_data = Blob()
        class Content: parts = [Part()]
        class Cand: content = Content()
        class Resp: candidates = [Cand()]
        calls = []
        class Models:
            def generate_content(self, model, contents, config):
                calls.append(model)
                if model == "bad-model": raise RuntimeError("404 model not found")
                return Resp()
        class Client: models = Models()
        google = T_.ModuleType("google"); genai = T_.ModuleType("google.genai")
        genai.types = fake_types; google.genai = genai
        gem = T_.ModuleType("core.gemini"); gem.client = lambda timeout_ms=0: Client()
        sys.modules.update({"google": google, "google.genai": genai,
                            "google.genai.types": fake_types})
        import core; core.gemini = gem; sys.modules["core.gemini"] = gem
        data, mime, model = ig._gemini_backend("x", "1:1", {"gemini_models": ["bad-model", "good"]}, lambda s: None)
        self.assertEqual((model, mime), ("good", "image/png")); self.assertEqual(calls, ["bad-model", "good"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
