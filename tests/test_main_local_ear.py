"""Integration test: the REAL JarvisLive methods that were patched for Local Ear, with a
fake Gemini session. Needs no mic, GPU, Qt or API key (tests/_stubs.py covers imports).

    python -m unittest tests.test_main_local_ear -v
"""
import asyncio, os, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests import _stubs
_stubs.install()

import numpy as np                                    # noqa: E402
import main as app_main                               # noqa: E402
from memory import config_manager as cm               # noqa: E402
from core import local_ear as le                      # noqa: E402
from tests.test_local_ear import speech, noise, pcm, FakeTx   # noqa: E402

SR = 16000


def make_live(mode="local_ear", speed="fast", media=False, tx=None):
    jl = app_main.JarvisLive.__new__(app_main.JarvisLive)
    jl.ui = MagicMock(); jl.ui.muted = False
    jl.ui.video_has_sound.return_value = media
    jl.ui.video_is_playing.return_value = media
    jl.ui.video_is_paused.return_value = False
    jl._awake = True; jl._dashboard = None; jl._loop = None; jl._engine = None
    jl._sleep_standby = False
    jl._explicit_vad_active = True
    jl._voice_isolator = None
    jl._tuned_live = True
    jl._last_user_speech = 0.0; jl._thinking_since = None; jl._rx_deadline = None
    jl._private_mode = False; jl._screen_watch_enabled = False
    jl._last_screen_sent = 0.0; jl._interrupted = False
    jl._last_user_pitch_hz = 0.0; jl._last_user_rms = 0.0
    jl._last_user_utterance = ""
    jl._ear_transcriber = tx
    cm.save_listening_mode(mode)
    cm.save_local_ear_config(speed=speed)
    return jl


class Session:
    def __init__(self): self.calls = []
    async def send_realtime_input(self, **kw): self.calls.append(kw)


async def drive(jl, audio, wait=1.5):
    jl.session = Session()
    jl.out_queue = asyncio.Queue()
    for i in range(0, len(audio), 1024):
        await jl.out_queue.put({"data": pcm(audio[i:i + 1024]), "mime_type": "audio/pcm"})
    task = asyncio.create_task(jl._send_realtime())
    while not jl.out_queue.empty():
        await asyncio.sleep(0.005)
    # keep feeding silence so the worker thread's verdict can be collected
    t_end = time.time() + wait
    while time.time() < t_end:
        await jl.out_queue.put({"data": pcm(noise(0.064)), "mime_type": "audio/pcm"})
        await asyncio.sleep(0.01)
    while not jl.out_queue.empty():
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.05)
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass
    return jl.session.calls


def kinds(calls):
    out = []
    for c in calls:
        if "activity_start" in c: out.append("start")
        elif "activity_end" in c: out.append("end")
        elif "audio_stream_end" in c: out.append("stream_end")
        elif "audio" in c: out.append("audio")
    return out


class T(unittest.TestCase):
    def setUp(self):
        self._orig = cm.CONFIG_FILE
        self._td = tempfile.TemporaryDirectory()
        cm.CONFIG_FILE = Path(self._td.name) / "api_keys.json"

    def tearDown(self):
        cm.CONFIG_FILE = self._orig
        self._td.cleanup()

    # -- builders ---------------------------------------------------------------
    def test_gate_and_frontend_follow_listening_mode(self):
        jl = make_live("local_ear", media=False)
        jl.ui.write_log = lambda *_: None
        self.assertIsInstance(jl._make_frontend(), le.ClearVoiceFrontend)
        gate = jl._build_audio_gate()
        self.assertIsInstance(gate, le.LocalEarGate)
        gate.close()
        cm.save_listening_mode("smart_vad")
        self.assertNotIsInstance(jl._make_frontend(), le.ClearVoiceFrontend)   # old frontend kept
        from core.voice_fingerprint import SmartSpeechGate
        self.assertIsInstance(jl._build_audio_gate(), SmartSpeechGate)

    def test_refresh_closes_old_gate_and_swaps_frontend(self):
        jl = make_live("local_ear")
        jl._barge_in_enabled = True
        jl._audio_gate = jl._build_audio_gate()
        old = jl._audio_gate
        cm.save_listening_mode("smart_vad")
        jl._refresh_audio_gate()
        self.assertTrue(old._closed)
        self.assertNotIsInstance(jl._audio_frontend, le.ClearVoiceFrontend)

    def test_local_ear_needs_explicit_vad_config(self):
        """Gemini's own VAD must be OFF when the PC decides the turn."""
        jl = make_live("local_ear")
        jl._audio_gate = jl._build_audio_gate()
        cfg = jl._tuning_config()
        self.assertTrue(cfg["realtime_input_config"].automatic_activity_detection.disabled)
        jl._audio_gate.close()

    def test_silence_rule(self):
        jl = make_live(media=False)
        self.assertFalse(jl._ear_must_stay_silent())
        jl.ui.video_has_sound.return_value = True
        self.assertTrue(jl._ear_must_stay_silent())
        jl.ui.video_has_sound.return_value = False
        jl._sleep_standby = True
        self.assertTrue(jl._ear_must_stay_silent())

    # -- the real _send_realtime loop -------------------------------------------
    def test_one_sentence_reaches_gemini_as_one_bracketed_turn(self):
        jl = make_live(speed="fast")
        jl._audio_gate = jl._build_audio_gate()
        x = np.concatenate([noise(1.0), speech(1.5, 0.08), noise(1.0)])
        calls = asyncio.run(drive(jl, x))
        k = kinds(calls)
        self.assertEqual(k.count("start"), 1, k)
        self.assertEqual(k.count("end"), 1, k)
        self.assertEqual(k[0], "start"); self.assertEqual(k[-1], "end")
        self.assertGreaterEqual(k.count("audio"), 1)
        for c in calls:
            if "audio" in c:
                self.assertIsNotNone(c["audio"])      # (types.Blob is a stub here)
        jl._audio_gate.close()

    def test_noise_never_reaches_gemini(self):
        jl = make_live(speed="fast")
        jl._audio_gate = jl._build_audio_gate()
        calls = asyncio.run(drive(jl, noise(3.0, 0.003)))
        self.assertEqual(kinds(calls), [])
        jl._audio_gate.close()

    def test_song_playing_lyrics_never_reach_gemini(self):
        jl = make_live(speed="fast", media=True,
                       tx=FakeTx("and i said hello to the moon tonight"))
        jl._audio_gate = jl._build_audio_gate()
        calls = asyncio.run(drive(jl, np.concatenate([noise(1.0, 0.004), speech(1.6, 0.12), noise(1.0, 0.004)])))
        self.assertEqual(kinds(calls), [])
        jl._audio_gate.close()

    def test_hello_during_song_pauses_song_and_sends_command(self):
        words = [("Hello", 0.0, 0.45), ("play", 0.55, 0.8), ("next", 0.85, 1.1), ("song", 1.15, 1.5)]
        jl = make_live(speed="fast", media=True, tx=FakeTx("Hello play next song", words=words))
        jl._audio_gate = jl._build_audio_gate()

        async def run():
            jl._loop = asyncio.get_running_loop()
            return await drive(jl, np.concatenate([noise(1.0, 0.004), speech(1.6, 0.12), noise(1.0, 0.004)]))
        calls = asyncio.run(run())
        jl.ui.pause_video.assert_called()            # song paused by the wake path
        k = kinds(calls)
        self.assertEqual(k.count("start"), 1, k)
        self.assertIn("hello", jl._last_user_utterance.lower())
        jl._audio_gate.close()

    def test_hello_alone_wakes_without_bothering_gemini(self):
        jl = make_live(speed="fast", media=True, tx=FakeTx("Hello."))
        jl._audio_gate = jl._build_audio_gate()

        async def run():
            jl._loop = asyncio.get_running_loop()
            return await drive(jl, np.concatenate([noise(1.0, 0.004), speech(1.2, 0.12), noise(1.0, 0.004)]))
        calls = asyncio.run(run())
        jl.ui.pause_video.assert_called()
        self.assertEqual(kinds(calls), [])
        jl._audio_gate.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
