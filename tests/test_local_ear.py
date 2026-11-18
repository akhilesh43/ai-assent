"""
Tests for core/local_ear.py — run:  python -m unittest tests.test_local_ear -v

No microphone, no GPU, no Whisper: speech is synthesised (harmonic tone with a
syllable envelope) and the transcriber is a fake that returns scripted text.
"""
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import local_ear as le          # noqa: E402

SR = 16000
RNG = np.random.default_rng(7)


def speech(seconds: float, amp: float = 0.2, f0: float = 140.0) -> np.ndarray:
    """Voice-like: harmonics up to ~3 kHz with 4 Hz syllable modulation."""
    t = np.arange(int(seconds * SR)) / SR
    sig = np.zeros_like(t)
    for h in range(1, 22):
        sig += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t + h)
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2
    sig = sig * env
    sig = sig / (np.sqrt(np.mean(sig ** 2)) + 1e-9) * amp
    return sig.astype(np.float32)


def noise(seconds: float, amp: float = 0.002) -> np.ndarray:
    return (RNG.standard_normal(int(seconds * SR)) * amp).astype(np.float32)


def pcm(x: np.ndarray) -> bytes:
    return le.f32_to_pcm16(x)


def blocks(x: np.ndarray, n: int = 1024):
    for i in range(0, len(x), n):
        yield pcm(x[i:i + n])


def run_gate(gate, x: np.ndarray, tail_s: float = 0.0, wait_s: float = 3.0):
    """Feed audio (then silence), collect every EarResult; keep polling with
    silence blocks until the worker thread has had time to answer."""
    out = []
    for b in blocks(x):
        out.append(gate.process_chunk(b))
    quiet = noise(0.064)
    deadline = time.time() + wait_s
    idle = 0
    while time.time() < deadline:
        r = gate.process_chunk(pcm(quiet))
        out.append(r)
        if r.emit_audio:
            idle = 0
        else:
            idle += 1
            if idle > 12 and not gate.active:
                break
        time.sleep(0.01)
    return out


def emitted(results):
    return b"".join(r.emit_audio for r in results if r.emit_audio)


class FakeTx:
    """Scripted transcriber."""
    def __init__(self, text="", words=None, avg_logprob=-0.2, no_speech=0.01, delay=0.0):
        self.text, self.words = text, words or []
        self.avg_logprob, self.no_speech, self.delay = avg_logprob, no_speech, delay
        self.calls = 0

    def usable(self):
        return True

    def transcribe(self, audio, words=False, wait_s=0):
        self.calls += 1
        time.sleep(self.delay)
        return le.Transcript(text=self.text, words=self.words if words else [],
                             avg_logprob=self.avg_logprob, no_speech_prob=self.no_speech)


class TestText(unittest.TestCase):
    def test_wake_variants(self):
        W, N = le.DEFAULTS["wake_words"], le.DEFAULTS["name_words"]
        self.assertEqual(le.find_wake("Hello.", W, N), 1)
        self.assertEqual(le.find_wake("Hello, Jarvis play the next song", W, N), 2)
        self.assertEqual(le.find_wake("hey hello what time is it", W, N), 2)
        self.assertEqual(le.find_wake("hallo", W, N), 1)
        self.assertEqual(le.find_wake("हेलो जार्विस", W, N), 2)
        self.assertEqual(le.find_wake("Jarvis volume down", W, N), 1)
        self.assertIsNone(le.find_wake("yellow submarine", W, N))
        # a lyric with hello in the MIDDLE must not wake it
        self.assertIsNone(le.find_wake("i said hello to the moon", W, N))

    def test_hallucination(self):
        T = le.Transcript
        self.assertEqual(le.looks_hallucinated(T(text=""), 1.0), "empty")
        self.assertEqual(le.looks_hallucinated(T(text="Hmm."), 1.0), "filler")
        self.assertEqual(le.looks_hallucinated(T(text="Thank you.", avg_logprob=-0.9), 1.0),
                         "known_hallucination")
        self.assertIsNone(le.looks_hallucinated(T(text="what is the weather", avg_logprob=-0.3), 2.0))
        self.assertEqual(le.looks_hallucinated(T(text="a b a b a b a b"), 2.0), "repetition_loop")
        self.assertEqual(le.looks_hallucinated(T(text="blah", avg_logprob=-2.0), 2.0),
                         "low_confidence")

    def test_devanagari_survives_normalisation(self):
        self.assertEqual(le.normalize_text("हेलो, जार्विस!"), "हेलो जार्विस")


class TestDsp(unittest.TestCase):
    def test_normalizer_raises_quiet_voice(self):
        x = speech(2.0, amp=0.004)
        y, info = le.normalize_utterance(x)
        rms = float(np.sqrt(np.mean(y ** 2)))
        self.assertGreater(info["gain_db"], 20)
        self.assertGreater(rms, 0.07)
        self.assertLessEqual(float(np.max(np.abs(y))), 1.0)

    def test_normalizer_tames_loud_voice(self):
        x = np.clip(speech(2.0, amp=0.6) * 2.5, -1, 1)
        y, info = le.normalize_utterance(x)
        self.assertLess(info["gain_db"], 0)
        self.assertLessEqual(float(np.max(np.abs(y))), 1.0)
        self.assertLess(float(np.sqrt(np.mean(y ** 2))), 0.35)

    def test_normalizer_does_not_boost_noise_between_words(self):
        # speech with long pauses: level must be measured on speech, not the pauses
        x = np.concatenate([speech(0.6, 0.1), noise(1.0, 0.002), speech(0.6, 0.1)])
        y, info = le.normalize_utterance(x)
        self.assertLess(info["gain_db"], 3.0)           # ~0 dB, not +20

    def test_frontend_reduces_noise_keeps_speech(self):
        fe = le.ClearVoiceFrontend(denoise=0.6)
        n = noise(4.0, 0.02)
        s = np.concatenate([np.zeros(int(1.5 * SR), np.float32),
                            speech(1.5, 0.15), np.zeros(int(1.0 * SR), np.float32)])
        noisy = (s + n[:s.size]).astype(np.float32)
        out = np.concatenate([fe.process_float32(noisy[i:i + 1024])
                              for i in range(0, noisy.size, 1024)])
        self.assertGreater(out.size, noisy.size - 1024)
        # noise-only part (first 1.0 s, after the 0.5 s learning) gets quieter
        a, b = int(0.5 * SR), int(1.4 * SR)
        n_in = np.sqrt(np.mean(noisy[a:b] ** 2))
        n_out = np.sqrt(np.mean(out[a + 256:b + 256] ** 2))
        self.assertLess(n_out, 0.6 * n_in)
        # the speech part keeps most of its energy
        sa, sb = int(1.7 * SR), int(2.9 * SR)
        s_in = np.sqrt(np.mean(noisy[sa:sb] ** 2))
        s_out = np.sqrt(np.mean(out[sa + 256:sb + 256] ** 2))
        self.assertGreater(s_out, 0.7 * s_in)

    def test_frontend_does_not_squash_soft_voice(self):
        """Regression: the old AudioFrontend cut anything under RMS 0.006 to 35%."""
        fe = le.ClearVoiceFrontend(denoise=0.5)
        x = np.concatenate([noise(1.0, 0.0008), speech(1.5, 0.004), noise(0.5, 0.0008)])
        out = np.concatenate([fe.process_float32(x[i:i + 1024]) for i in range(0, x.size, 1024)])
        sa, sb = int(1.2 * SR), int(2.4 * SR)
        s_in = np.sqrt(np.mean(x[sa:sb] ** 2))
        s_out = np.sqrt(np.mean(out[sa + 256:sb + 256] ** 2))
        self.assertGreater(s_out, 0.6 * s_in)

    def test_frontend_off_is_passthrough_length(self):
        fe = le.ClearVoiceFrontend(denoise=0.0)
        x = speech(0.5, 0.1)
        y = fe.process_float32(x)
        self.assertEqual(y.size, x.size)


class TestGateFast(unittest.TestCase):
    def mk(self, **kw):
        cfg = {"speed": "fast"}
        cfg.update(kw)
        return le.LocalEarGate(cfg, transcriber=None)

    def test_noise_only_sends_nothing(self):
        g = self.mk()
        res = run_gate(g, noise(3.0, 0.002))
        self.assertEqual(emitted(res), b"")

    def test_one_sentence_one_burst(self):
        g = self.mk()
        x = np.concatenate([noise(1.0), speech(1.5, 0.08), noise(1.5)])
        res = run_gate(g, x)
        starts = [r for r in res if r.started]
        stops = [r for r in res if r.stopped]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(stops), 1)
        audio = emitted(res)
        secs = len(audio) / 2 / SR
        self.assertGreater(secs, 1.4)
        self.assertLess(secs, 3.0)
        # started comes first, stopped comes last, nothing between is flagged
        flagged = [i for i, r in enumerate(res) if r.emit_audio]
        self.assertTrue(res[flagged[0]].started)
        self.assertTrue(res[flagged[-1]].stopped)

    def test_quiet_voice_is_loud_when_released(self):
        g = self.mk()
        x = np.concatenate([noise(1.0, 0.0006), speech(1.5, 0.006), noise(1.5, 0.0006)])
        res = run_gate(g, x)
        a = le.pcm16_to_f32(emitted(res))
        self.assertGreater(a.size, SR)
        self.assertGreater(float(np.sqrt(np.mean(a ** 2))), 0.06)

    def test_short_click_ignored(self):
        g = self.mk()
        x = np.concatenate([noise(1.0), speech(0.12, 0.2), noise(2.0)])
        res = run_gate(g, x)
        self.assertEqual(emitted(res), b"")

    def test_two_sentences_two_bursts(self):
        g = self.mk()
        x = np.concatenate([noise(1.0), speech(1.0, 0.08), noise(1.2),
                            speech(1.0, 0.08), noise(1.5)])
        res = run_gate(g, x)
        self.assertEqual(len([r for r in res if r.started]), 2)
        self.assertEqual(len([r for r in res if r.stopped]), 2)

    def test_slices_are_bounded(self):
        g = self.mk()
        x = np.concatenate([noise(1.0), speech(6.0, 0.08), noise(1.5)])
        res = run_gate(g, x)
        for r in res:
            self.assertLessEqual(len(r.emit_audio), le._SLICE_BYTES)

    def test_reset_discards_in_progress(self):
        g = self.mk()
        for b in blocks(np.concatenate([noise(1.0), speech(1.0, 0.08)])):
            g.process_chunk(b)
        self.assertTrue(g.active)
        g.reset()
        self.assertFalse(g.active)
        res = run_gate(g, noise(2.0))
        self.assertEqual(emitted(res), b"")

    def test_force_stop_flushes_ptt_release(self):
        g = self.mk()
        for b in blocks(np.concatenate([noise(1.0), speech(1.2, 0.08)])):
            g.process_chunk(b)
        r = g.force_stop()
        self.assertFalse(r.stopped)                       # the burst carries its own stop
        res = run_gate(g, noise(0.5))
        self.assertGreater(len(emitted(res)), SR)         # the sentence was released

    def test_stale_sentence_dropped(self):
        g = self.mk(stale_s=0.05)
        for b in blocks(np.concatenate([noise(1.0), speech(1.0, 0.08), noise(1.2)])):
            g.process_chunk(b)
        # simulate the mic callback being silent (assistant speaking) for a while
        time.sleep(0.2)
        res = [g.process_chunk(pcm(noise(0.064))) for _ in range(5)]
        self.assertEqual(emitted(res), b"")


class TestGateAccurate(unittest.TestCase):
    def mk(self, tx, **kw):
        cfg = {"speed": "accurate"}
        cfg.update(kw)
        return le.LocalEarGate(cfg, transcriber=tx)

    def sentence(self):
        return np.concatenate([noise(1.0), speech(1.5, 0.08), noise(1.5)])

    def test_real_text_is_released(self):
        tx = FakeTx("what is the time")
        heard = []
        g = le.LocalEarGate({"speed": "accurate"}, transcriber=tx, on_utterance=heard.append)
        res = run_gate(g, self.sentence())
        self.assertGreater(len(emitted(res)), SR)
        self.assertEqual(tx.calls, 1)
        self.assertEqual(heard, ["what is the time"])

    def test_hallucination_is_dropped(self):
        tx = FakeTx("Thank you.", avg_logprob=-0.9)
        g = self.mk(tx)
        res = run_gate(g, self.sentence())
        self.assertEqual(emitted(res), b"")
        self.assertEqual(g.stats["noise"], 1)

    def test_empty_transcript_is_dropped(self):
        g = self.mk(FakeTx(""))
        self.assertEqual(emitted(run_gate(g, self.sentence())), b"")

    def test_slow_transcriber_does_not_block_audio_thread(self):
        tx = FakeTx("hello there", delay=0.4)
        g = self.mk(tx)
        x = self.sentence()
        t0 = time.time()
        for b in blocks(x):
            g.process_chunk(b)
        self.assertLess(time.time() - t0, 0.35)           # process_chunk never waited for STT
        res = run_gate(g, noise(0.1), wait_s=3.0)
        self.assertGreater(len(emitted(res)), SR)

    def test_transcriber_missing_fails_open(self):
        g = self.mk(None)
        self.assertGreater(len(emitted(run_gate(g, self.sentence()))), SR)


class TestMediaMode(unittest.TestCase):
    """Song playing: silent unless the sentence starts with the wake word."""

    def mk(self, tx, woke):
        return le.LocalEarGate({"speed": "fast"}, transcriber=tx,
                               require_wake_fn=lambda: True, on_wake=woke.append)

    def sentence(self, loud=0.12):
        return np.concatenate([noise(1.0, 0.004), speech(1.6, loud), noise(1.5, 0.004)])

    def test_lyrics_ignored(self):
        woke = []
        g = self.mk(FakeTx("i said hello to the moon tonight"), woke)
        res = run_gate(g, self.sentence())
        self.assertEqual(emitted(res), b"")
        self.assertEqual(woke, [])
        self.assertEqual(g.stats["ignored"], 1)

    def test_hello_only_wakes_and_sends_nothing(self):
        woke = []
        g = self.mk(FakeTx("Hello."), woke)
        res = run_gate(g, self.sentence())
        self.assertEqual(len(woke), 1)
        self.assertEqual(emitted(res), b"")

    def test_hello_with_command_sends_only_the_command(self):
        woke = []
        words = [("Hello,", 0.0, 0.5), ("play", 0.6, 0.9), ("next", 0.95, 1.2), ("song", 1.25, 1.6)]
        tx = FakeTx("Hello, play next song", words=words)
        heard = []
        g = le.LocalEarGate({"speed": "fast"}, transcriber=tx, require_wake_fn=lambda: True,
                            on_wake=woke.append, on_utterance=heard.append)
        res = run_gate(g, self.sentence())
        self.assertEqual(len(woke), 1)
        a = emitted(res)
        self.assertGreater(len(a), SR // 2)
        # wake word (0.5 s) was cut off: shorter than the whole sentence
        full = (1.6 + 0.24 + 0.32) * SR * 2
        self.assertLess(len(a), full)
        self.assertTrue(heard and heard[0].startswith("hello "))

    def test_no_recogniser_stays_silent_during_media(self):
        woke = []
        g = self.mk(None, woke)
        self.assertEqual(emitted(run_gate(g, self.sentence())), b"")

    def test_wake_requirement_ends_when_media_stops(self):
        flag = {"media": True}
        woke = []
        tx = FakeTx("what is the weather")
        g = le.LocalEarGate({"speed": "fast"}, transcriber=tx,
                            require_wake_fn=lambda: flag["media"], on_wake=woke.append)
        self.assertEqual(emitted(run_gate(g, self.sentence())), b"")
        flag["media"] = False
        time.sleep(0.3)                                    # past the 250 ms cache
        self.assertGreater(len(emitted(run_gate(g, self.sentence()))), SR)

    def test_media_wake_can_be_disabled(self):
        g = le.LocalEarGate({"speed": "fast", "media_wake": False}, transcriber=None,
                            require_wake_fn=lambda: True)
        self.assertGreater(len(emitted(run_gate(g, self.sentence()))), SR)


if __name__ == "__main__":
    unittest.main(verbosity=2)
