"""
python -m unittest tests.test_failover_auto -v

"Gemini, a local model, any combination — and if one fails the next takes over."
Fake providers + fake clock for the chains; the REAL registry, config writer and JarvisLive for the wiring.
"""
import asyncio, json, os, sys, tempfile, types, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import providers_setup as ps                 # noqa: E402
from memory import config_manager as cm                # noqa: E402
from providers import _chain, registry                 # noqa: E402
from providers._chain import Chain, Link, classify     # noqa: E402

GEM = "AIza" + "x" * 35


def run(coro):
    return asyncio.run(coro)


class fake_module:
    """Put ONE fake module in sys.modules and take only it out again. (mock.patch.dict(sys.modules)
    restores the whole dict, which also un-imports anything imported meanwhile — numpy then refuses
    to load a second time.)"""
    def __init__(self, name, mod): self.name, self.mod = name, mod
    def __enter__(self):
        self.old = sys.modules.get(self.name); sys.modules[self.name] = self.mod
    def __exit__(self, *a):
        if self.old is None: sys.modules.pop(self.name, None)
        else: sys.modules[self.name] = self.old


async def collect(agen):
    return [x async for x in agen]


# ───────────────────────────── fakes ─────────────────────────────
class FakeLLM:
    def __init__(self, name, events=None, fail=None, fail_after=None, caps=None):
        self.model, self.name, self.events = name, name, events or [{"text": name}]
        self.fail, self.fail_after, self.calls = fail, fail_after, 0
        self.caps = caps or {"tools": True, "vision": True, "tool_format": "openai", "max_tools": None}
    async def chat(self, messages, tools=None, *, stream=True):
        self.calls += 1
        if self.fail and self.fail_after is None:
            raise self.fail
        for i, ev in enumerate(self.events):
            if self.fail_after is not None and i == self.fail_after:
                raise self.fail
            yield ev


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def make_llm(*fakes, clock=None, log=None):
    from providers import llm_failover
    links = [Link(f.name, f) for f in fakes]
    with mock.patch.object(llm_failover, "build_links", lambda ids, cfg, what: (links, [])):
        obj = llm_failover.FailoverLLM({"chain": ",".join(f.name for f in fakes)})
    if clock is not None:
        obj._chain._clock = clock
    if log is not None:
        obj._chain._log = log
    return obj


class TestClassify(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(classify(RuntimeError("401 Unauthorized")), "auth")
        self.assertEqual(classify(RuntimeError("Incorrect API key provided")), "auth")
        self.assertEqual(classify(RuntimeError("429 You exceeded your current quota")), "quota")
        self.assertEqual(classify(RuntimeError("RESOURCE_EXHAUSTED")), "quota")
        self.assertEqual(classify(RuntimeError("insufficient_quota")), "quota")
        self.assertEqual(classify(asyncio.TimeoutError("timed out")), "network")
        self.assertEqual(classify(RuntimeError("503 overloaded")), "network")
        self.assertEqual(classify(ImportError("faster_whisper")), "install")
        self.assertEqual(classify(RuntimeError("pip install av")), "install")
        self.assertEqual(classify(RuntimeError("400 bad tool schema")), "other")

    def test_status_attribute(self):
        e = RuntimeError("x"); e.status = 429
        self.assertEqual(classify(e), "quota")
        e2 = RuntimeError("x"); e2.response = types.SimpleNamespace(status_code=401)
        self.assertEqual(classify(e2), "auth")


class TestChain(unittest.TestCase):
    def test_empty_chain_explains_itself(self):
        with self.assertRaises(RuntimeError) as c:
            Chain([], "brain")
        self.assertIn("enable_providers", str(c.exception))


class TestFailoverLLM(unittest.TestCase):
    def test_primary_serves_and_spare_untouched(self):
        a, b = FakeLLM("gemini"), FakeLLM("spare")
        f = make_llm(a, b)
        self.assertEqual(run(collect(f.chat([{}]))), [{"text": "gemini"}])
        self.assertEqual((a.calls, b.calls), (1, 0))

    def test_failure_before_output_switches_and_rests_the_failed_one(self):
        clock, logs = Clock(), []
        a, b = FakeLLM("gemini", fail=RuntimeError("429 quota exceeded")), FakeLLM("spare")
        f = make_llm(a, b, clock=clock, log=logs.append)
        self.assertEqual(run(collect(f.chat([{}]))), [{"text": "spare"}])
        self.assertTrue(any("quota" in l and "switching" in l for l in logs), logs)
        run(collect(f.chat([{}])))                         # second turn: gemini is resting, not retried
        self.assertEqual((a.calls, b.calls), (1, 2))
        self.assertEqual(f.model, "spare")                # the engine's timing line names who answers
        clock.t += 11 * 60                                 # rest is over -> Gemini is tried first again
        a.fail = None
        self.assertEqual(run(collect(f.chat([{}]))), [{"text": "gemini"}])
        self.assertEqual(a.calls, 2)

    def test_failure_after_output_is_not_spliced(self):
        a = FakeLLM("gemini", events=[{"text": "Hel"}, {"text": "lo"}], fail=RuntimeError("503"), fail_after=1)
        b = FakeLLM("spare")
        f = make_llm(a, b)
        out = []
        async def go():
            async for ev in f.chat([{}]):
                out.append(ev)
        with self.assertRaises(RuntimeError):
            run(go())
        self.assertEqual(out, [{"text": "Hel"}]); self.assertEqual(b.calls, 0)

    def test_all_fail_reports_each(self):
        f = make_llm(FakeLLM("gemini", fail=RuntimeError("401 invalid api key")),
                     FakeLLM("spare", fail=RuntimeError("429 quota")))
        with self.assertRaises(RuntimeError) as c:
            run(collect(f.chat([{}])))
        self.assertIn("gemini", str(c.exception)); self.assertIn("spare", str(c.exception))

    def test_everything_resting_still_tries(self):
        clock = Clock()
        a = FakeLLM("gemini", fail=RuntimeError("401 unauthorized"))
        f = make_llm(a, clock=clock)
        with self.assertRaises(RuntimeError): run(collect(f.chat([{}])))
        a.fail = None
        self.assertEqual(run(collect(f.chat([{}]))), [{"text": "gemini"}])      # not locked out for 30 min

    def test_tool_calls_pass_through(self):
        ev = {"tool_calls": ["call"]}
        f = make_llm(FakeLLM("gemini", fail=RuntimeError("503")), FakeLLM("spare", events=[ev]))
        self.assertEqual(run(collect(f.chat([{}], [{"x": 1}]))), [ev])

    def test_cancellation_is_not_swallowed(self):
        class Slow(FakeLLM):
            async def chat(self, *a, **k):
                await asyncio.sleep(10); yield {"text": "x"}
        f = make_llm(Slow("gemini"), FakeLLM("spare"))
        async def go():
            t = asyncio.create_task(collect(f.chat([{}])))
            await asyncio.sleep(0.05); t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
        run(go())

    def test_caps_are_the_safe_intersection(self):
        f = make_llm(FakeLLM("a", caps={"tools": True, "vision": True, "tool_format": "openai", "max_tools": 20}),
                     FakeLLM("b", caps={"tools": True, "vision": False, "tool_format": "openai", "max_tools": 12}))
        self.assertTrue(f.caps["tools"]); self.assertFalse(f.caps["vision"]); self.assertEqual(f.caps["max_tools"], 12)

    def test_single_key_chain_just_works(self):
        f = make_llm(FakeLLM("spare"))
        self.assertEqual(run(collect(f.chat([{}]))), [{"text": "spare"}])

    def test_vision_probe_saved_under_each_links_own_id(self):
        saved = []
        class V(FakeLLM):
            def ensure_vision_known(self, save=None):
                save({"vision": True}); return True
        f = make_llm(V("gem"), V("sp2"))
        with mock.patch("memory.config_manager.save_provider_config", lambda pid, v: saved.append(pid)):
            self.assertTrue(f.ensure_vision_known(lambda v: self.fail("must not use the caller's save")))
        self.assertEqual(saved, ["gem", "sp2"])


class FakeSTT:
    def __init__(self, name, fail=None, lang=""):
        self.name, self.fail, self.detected_language, self.calls = name, fail, lang, 0
    async def transcribe(self, pcm, sample_rate=16000, language=""):
        self.calls += 1
        if self.fail: raise self.fail
        return f"heard by {self.name}"
    async def warmup(self, log=None):
        if self.fail: raise self.fail


class FakeTTS:
    def __init__(self, name, fail=None, fail_after=None, with_language=False, sound=True):
        self.name, self.fail, self.fail_after, self.sound = name, fail, fail_after, sound
        self.seen = None
        if with_language:
            async def speak(text, voice="", language=""):
                self.seen = {"voice": voice, "language": language}
                async for x in self._gen(): yield x
            self.speak = speak
        else:
            async def speak(text, voice=""):
                self.seen = {"voice": voice}
                async for x in self._gen(): yield x
            self.speak = speak
    async def _gen(self):
        if self.fail and self.fail_after is None: raise self.fail
        if not self.sound: return
        yield b"\x01\x00" * 100
        if self.fail: raise self.fail
        yield b"\x02\x00" * 100


class TestFailoverSTTandTTS(unittest.TestCase):
    def make_stt(self, *fakes):
        from providers import stt_failover
        links = [Link(f.name, f) for f in fakes]
        with mock.patch.object(stt_failover, "build_links", lambda *a: (links, [])):
            return stt_failover.FailoverSTT({"chain": "x"})

    def make_tts(self, *fakes):
        from providers import tts_failover
        links = [Link(f.name, f) for f in fakes]
        with mock.patch.object(tts_failover, "build_links", lambda *a: (links, [])):
            return tts_failover.FailoverTTS({"chain": "x"})

    def test_local_whisper_missing_falls_to_cloud_and_is_rested_for_good(self):
        a, b = FakeSTT("local", fail=ImportError("faster_whisper")), FakeSTT("spare", lang="hi")
        f = self.make_stt(a, b)
        self.assertEqual(run(f.transcribe(b"x")), "heard by spare")
        run(f.transcribe(b"x"))
        self.assertEqual(a.calls, 1)                          # "not installed" is not retried every sentence
        self.assertEqual(f.detected_language, "hi")           # the engine reads this to choose a voice language

    def test_warmup_survives_a_broken_link(self):
        f = self.make_stt(FakeSTT("local", fail=RuntimeError("model failed to load")), FakeSTT("spare"))
        run(f.warmup(None))                                    # must not raise
        self.assertEqual(run(f.transcribe(b"x")), "heard by spare")

    def test_all_ears_fail(self):
        f = self.make_stt(FakeSTT("a", fail=RuntimeError("503")), FakeSTT("b", fail=RuntimeError("503")))
        with self.assertRaises(RuntimeError): run(f.transcribe(b"x"))

    def test_voice_falls_over_before_the_first_sound(self):
        el = FakeTTS("cloudvoice", fail=RuntimeError("401 key was refused"))
        op = FakeTTS("sysvoice")
        f = self.make_tts(el, op)
        pcm = run(collect(f.speak("hello")))
        self.assertTrue(pcm); self.assertEqual(f.last_voice, "sysvoice")

    def test_voice_dying_midway_is_not_repeated_by_another(self):
        el = FakeTTS("cloudvoice", fail=RuntimeError("503"), fail_after=1)
        op = FakeTTS("sysvoice")
        f = self.make_tts(el, op)
        got = []
        async def go():
            async for p in f.speak("hello"): got.append(p)
        with self.assertRaises(RuntimeError): run(go())
        self.assertEqual(len(got), 1); self.assertIsNone(op.seen)

    def test_silence_counts_as_failure(self):
        f = self.make_tts(FakeTTS("cloudvoice", sound=False), FakeTTS("sysvoice"))
        self.assertTrue(run(collect(f.speak("hello")))); self.assertEqual(f.last_voice, "sysvoice")

    def test_language_and_voice_only_reach_links_that_declare_them(self):
        a, b = FakeTTS("piper", with_language=True, fail=RuntimeError("503")), FakeTTS("cloudvoice")
        f = self.make_tts(a, b)
        run(collect(f.speak("hello", voice="V1", language="hi")))
        self.assertEqual(a.seen, {"voice": "V1", "language": "hi"})
        self.assertEqual(b.seen, {"voice": ""})                # no language kwarg, voice only to the FIRST link
        import inspect
        self.assertIn("language", inspect.signature(f.speak).parameters)   # the engine introspects this

    def test_all_voices_fail(self):
        f = self.make_tts(FakeTTS("a", fail=RuntimeError("503")), FakeTTS("b", fail=RuntimeError("401")))
        with self.assertRaises(RuntimeError): run(collect(f.speak("x")))
