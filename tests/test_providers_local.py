"""
python -m unittest tests.test_providers_local -v

The engine settings after OpenAI / ElevenLabs were removed: Gemini, a local model, or both.
Real config writer and real settings builder; a temp directory stands in for config/.
"""
import json, os, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import providers_setup as ps      # noqa: E402
from memory import config_manager as cm     # noqa: E402

GEM = "AIza" + "x" * 35


class Cfg(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file = Path(self.tmp.name) / "api_keys.json"
        self.patches = [mock.patch.object(cm, "CONFIG_FILE", self.file),
                        mock.patch.object(cm, "CONFIG_DIR", Path(self.tmp.name))]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])
        self.addCleanup(self.tmp.cleanup)

    def write(self, d):
        self.file.write_text(json.dumps(d), encoding="utf-8")

    def read(self):
        return json.loads(self.file.read_text(encoding="utf-8"))


class TestBuildAuto(unittest.TestCase):
    def test_needs_something(self):
        with self.assertRaises(ValueError):
            ps.build_auto()

    def test_gemini_only_stays_on_live(self):
        s = ps.build_auto(gemini_key=GEM, faster_whisper=False)
        self.assertEqual(s["mode"], "gemini")
        self.assertEqual(s["creds"][ps.FAILOVER_LLM]["chain"], ps.GEMINI_COMPAT_ID)
        self.assertEqual(s["sidecar"], "")

    def test_gemini_plus_local_has_a_spare_and_a_sidecar(self):
        s = ps.build_auto(gemini_key=GEM, local_model="qwen3:8b", faster_whisper=True)
        self.assertEqual(s["creds"][ps.FAILOVER_LLM]["chain"], f"{ps.GEMINI_COMPAT_ID},{ps.LOCAL_ID}")
        self.assertEqual(s["sidecar"], ps.LOCAL_ID)
        self.assertFalse(s["sidecar_first"])
        self.assertEqual(s["mode"], "gemini")

    def test_local_only_is_a_pipeline(self):
        s = ps.build_auto(local_model="qwen3:8b", faster_whisper=True)
        self.assertEqual(s["mode"], "pipeline")
        self.assertEqual(s["pipeline"]["llm"], ps.FAILOVER_LLM)
        self.assertTrue(s["sidecar_first"])
        self.assertEqual(s["extra"]["image_gen"]["backends"], ["pollinations"])

    def test_private_preference_runs_the_pipeline_even_with_gemini(self):
        s = ps.build_auto(gemini_key=GEM, prefer="private", faster_whisper=True)
        self.assertEqual(s["mode"], "pipeline")

    def test_legacy_voice_preference_still_means_live(self):
        self.assertEqual(ps.normalise_prefer("voice"), "speed")
        self.assertEqual(ps.build_auto(gemini_key=GEM, prefer="voice", faster_whisper=True)["mode"], "gemini")

    def test_nothing_names_a_removed_service(self):
        s = ps.build_auto(gemini_key=GEM, local_model="m", faster_whisper=False)
        blob = json.dumps(s).lower()
        for word in ("openai.com", "elevenlabs", "tts_openai", "whisper@openai"):
            self.assertNotIn(word, blob)

    def test_every_provider_the_chains_name_is_configured_or_builtin(self):
        s = ps.build_auto(gemini_key=GEM, local_model="m", faster_whisper=True)
        builtin = {ps.PIPER_ID, ps.SYSTEM_TTS}
        for fo in (ps.FAILOVER_LLM, ps.FAILOVER_STT, ps.FAILOVER_TTS):
            for pid in s["creds"][fo]["chain"].split(","):
                self.assertTrue(pid in s["creds"] or pid in builtin, pid)


class TestPresets(unittest.TestCase):
    def test_removed_presets_are_gone(self):
        self.assertNotIn("openai", ps.PRESETS)
        with self.assertRaises(ValueError):
            ps.build_settings("openai")

    def test_local_needs_a_model(self):
        with self.assertRaises(ValueError):
            ps.build_settings("local")
        s = ps.build_settings("local", local_model="qwen3:8b")
        self.assertEqual(s["pipeline"]["llm"], ps.LOCAL_ID)
        self.assertEqual(s["creds"][ps.WHISPER_ID]["backend"], "local")

    def test_backup_keeps_gemini_live(self):
        s = ps.build_settings("backup", local_model="qwen3:8b")
        self.assertEqual(s["sidecar"], ps.LOCAL_ID)
        self.assertNotIn("mode", s)


class TestRegistry(unittest.TestCase):
    def test_removed_voices_are_not_in_the_registry(self):
        from providers import registry
        ids = {d["id"] for d in registry.discover("tts")}
        self.assertNotIn("elevenlabs", ids)
        self.assertNotIn("tts_openai", ids)
        self.assertTrue({"piper", "system_tts", "failover_tts"} <= ids)

    def test_no_hosted_openai_entry_in_the_brain_catalog(self):
        from providers.llm_openai import CATALOG
        self.assertNotIn("openai", [e["id"] for e in CATALOG])
        self.assertIn("ollama", [e["id"] for e in CATALOG])

    def test_whisper_catalog_has_no_openai_row(self):
        from providers import registry
        decl = registry.get_declaration("whisper")
        self.assertNotIn("openai", [e["id"] for e in decl["catalog"]])


class TestCanStart(unittest.TestCase):
    def test_gemini_key_is_enough(self):
        self.assertTrue(ps.can_start({"os_system": "linux", "gemini_api_key": GEM}))

    def test_local_pipeline_needs_no_key(self):
        d = {"os_system": "linux", "providers": {"mode": "pipeline", "pipeline": {"llm": ps.LOCAL_ID},
                                                  "creds": {ps.LOCAL_ID: {"model": "m"}}}}
        self.assertTrue(ps.can_start(d))

    def test_nothing_configured(self):
        self.assertFalse(ps.can_start({"os_system": "linux"}))


class TestScrubRemoved(Cfg):
    OLD = {
        "os_system": "linux", "gemini_api_key": GEM,
        "image_gen": {"backends": ["openai", "pollinations"]},
        "providers": {
            "mode": "pipeline",
            "pipeline": {"stt": "failover_stt", "llm": "openai_compat@openai", "tts": "elevenlabs"},
            "sidecar": "openai_compat@openai", "sidecar_first": True,
            "creds": {
                "openai_compat@openai": {"api_key": "sk-old"},
                "elevenlabs": {"api_key": "xi-old"},
                "tts_openai": {"api_key": "sk-old"},
                "failover_tts": {"chain": "elevenlabs,tts_openai,piper"},
                "failover_llm": {"chain": "openai_compat@custom,openai_compat@openai"},
            },
        },
    }

    def test_old_config_is_cleaned_and_still_starts(self):
        self.write(self.OLD)
        self.assertTrue(ps.scrub_removed())
        d = self.read()
        blob = json.dumps(d)
        for word in ("sk-old", "xi-old", "elevenlabs", "tts_openai", "openai_compat@openai"):
            self.assertNotIn(word, blob)
        self.assertEqual(d["providers"]["creds"]["failover_tts"]["chain"], "piper")
        self.assertEqual(d["image_gen"]["backends"], ["pollinations"])
        self.assertEqual(d["providers"]["mode"], "gemini")       # brain was removed, Gemini key exists
        self.assertTrue(ps.can_start(d))

    def test_idempotent(self):
        self.write(self.OLD)
        ps.scrub_removed()
        self.assertFalse(ps.scrub_removed())

    def test_clean_config_untouched(self):
        self.write({"os_system": "linux", "gemini_api_key": GEM})
        self.assertFalse(ps.scrub_removed())


class TestSyncAuto(Cfg):
    def test_first_run_gemini_then_local_model_extends_the_chain(self):
        self.write({"os_system": "linux", "gemini_api_key": GEM})
        self.assertTrue(ps.apply_initial_keys())
        self.assertEqual(cm.get_provider_mode(), "gemini")
        self.assertTrue(ps.apply_initial_keys(local_model="qwen3:8b"))
        chain = cm.get_provider_config(ps.FAILOVER_LLM)["chain"]
        self.assertEqual(chain, f"{ps.GEMINI_COMPAT_ID},{ps.LOCAL_ID}")
        self.assertFalse(ps.sync_auto())                    # idempotent

    def test_hand_edit_wins(self):
        self.write({"os_system": "linux", "gemini_api_key": GEM})
        ps.apply_initial_keys(local_model="qwen3:8b")
        cm.save_providers(mode="pipeline", pipeline={"llm": ps.LOCAL_ID})
        self.assertFalse(ps.sync_auto())
        self.assertFalse(self.read()["engine_auto"])

    def test_local_only_first_run_starts(self):
        self.write({"os_system": "linux"})
        ps.apply_initial_keys(local_model="qwen3:8b")
        self.assertTrue(ps.can_start(self.read()))


class R:
    def __init__(self, code=200, body=None): self.status_code, self._b = code, body or {}
    def json(self): return self._b


class TestChecks(unittest.TestCase):
    def test_local_server_models(self):
        http = mock.Mock(); http.get.return_value = R(200, {"data": [{"id": "qwen3:8b"}, {"id": "x"}]})
        self.assertEqual(ps.check_local(http, preferred_model="qwen3:8b")[2], "qwen3:8b")
        self.assertEqual(ps.check_local(http)[2], "qwen3:8b")
        self.assertFalse(ps.check_local(http, preferred_model="nope")[0])

    def test_local_server_down(self):
        http = mock.Mock(); http.get.side_effect = ConnectionError("refused")
        ok, msg, model = ps.check_local(http)
        self.assertFalse(ok); self.assertIsNone(model); self.assertIn("Start Ollama", msg)

    def test_gemini_codes(self):
        http = mock.Mock()
        for code, ok in ((200, True), (403, False), (429, True), (500, False)):
            http.get.return_value = R(code)
            self.assertEqual(ps.check_gemini(http, GEM)[0], ok, code)


class TestCli(Cfg):
    def run_cli(self, *argv, http=None):
        import enable_providers as ep
        out = []
        rc = ep.main(list(argv), http=http or mock.Mock(), ask=lambda p: GEM, out=out.append)
        return rc, "\n".join(out)

    def test_nothing_given_prints_help(self):
        self.write({"os_system": "linux"})
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc, _ = self.run_cli()
        self.assertEqual(rc, 2)

    def test_local_only_with_server_up(self):
        self.write({"os_system": "linux"})
        http = mock.Mock(); http.get.return_value = R(200, {"data": [{"id": "qwen3:8b"}]})
        rc, text = self.run_cli("--local-model", "qwen3:8b", http=http)
        self.assertEqual(rc, 0, text)
        self.assertEqual(cm.get_provider_mode(), "pipeline")

    def test_dead_local_server_saves_nothing(self):
        self.write({"os_system": "linux"})
        http = mock.Mock(); http.get.side_effect = ConnectionError("x")
        rc, _ = self.run_cli("--local-model", "qwen3:8b", http=http)
        self.assertEqual(rc, 1)
        self.assertNotIn("providers", self.read())

    def test_removed_preset_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_cli("--preset", "openai")


if __name__ == "__main__":
    unittest.main()
