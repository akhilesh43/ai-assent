"""
Unit and integration tests for voice-fingerprint registration, audio-stream
isolation, and immediate transition to thought mode on voice stop.

Run directly:
    python core/_voice_fingerprint_tests.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.voice_fingerprint import (  # noqa: E402
    SAMPLE_RATE,
    SmartSpeechGate,
    VoiceFingerprint,
    VoiceIsolator,
    enroll_from_pcm,
    enroll_from_samples,
    float32_to_pcm16,
)
from memory import config_manager  # noqa: E402
from providers.base import Msg  # noqa: E402
from providers.engine_pipeline import PipelineEngine  # noqa: E402


def _synth_voice(
    duration_s: float,
    f0_base: float,
    formants: list[tuple[float, float, float]],
    sr: int = SAMPLE_RATE,
    amp: float = 0.25,
    seed: int = 0,
) -> np.ndarray:
    """Synthesize glottal-harmonic speech shaped by vocal-tract formants."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / float(sr)
    f0 = f0_base * (
        1.0
        + 0.08 * np.sin(2.0 * np.pi * 1.4 * t)
        + 0.03 * np.sin(2.0 * np.pi * 3.1 * t)
    )
    phase = 2.0 * np.pi * np.cumsum(f0) / float(sr)
    sig = np.zeros(n, dtype=np.float32)
    for h in range(1, 22):
        sig += (1.0 / (h ** 1.15)) * np.sin(h * phase)
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, 1.0 / float(sr))
    env = np.zeros_like(freqs)
    for fc, bw, gain in formants:
        env += gain * np.exp(-0.5 * ((freqs - fc) / bw) ** 2)
    shaped = np.fft.irfft(spec * env, n=n).astype(np.float32)
    shaped /= float(np.max(np.abs(shaped)) + 1e-6)
    return (shaped * amp + rng.normal(0.0, 0.002, n)).astype(np.float32)


def _enrolled_speaker_a() -> VoiceFingerprint:
    samples = np.concatenate([
        _synth_voice(1.5, 125.0, [(500, 90, 1.0), (1450, 130, 0.7), (2450, 180, 0.4)], seed=1),
        _synth_voice(1.5, 132.0, [(350, 80, 1.0), (2100, 150, 0.8), (2900, 200, 0.3)], seed=2),
        _synth_voice(1.5, 120.0, [(680, 100, 1.0), (1150, 120, 0.8), (2550, 180, 0.35)], seed=3),
    ])
    return enroll_from_samples(samples, sr=SAMPLE_RATE, label="SpeakerA")


class VoiceFingerprintTests(unittest.TestCase):
    def test_enrollment_and_roundtrip(self) -> None:
        fp = _enrolled_speaker_a()
        self.assertGreaterEqual(fp.voiced_frames, 40)
        self.assertAlmostEqual(fp.f0_median, 126.0, delta=15.0)
        restored = VoiceFingerprint.from_dict(fp.to_dict())
        self.assertEqual(restored.label, "SpeakerA")
        self.assertEqual(len(restored.mean_mfcc), len(fp.mean_mfcc))
        self.assertEqual(len(restored.prototypes), len(fp.prototypes))

    def test_enrollment_rejects_silence_and_noise(self) -> None:
        silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        with self.assertRaises(ValueError):
            enroll_from_samples(silence, sr=SAMPLE_RATE)

        noise = np.random.default_rng(7).normal(0.0, 0.12, SAMPLE_RATE * 2).astype(np.float32)
        with self.assertRaises(ValueError):
            enroll_from_samples(noise, sr=SAMPLE_RATE)

        tone = (0.25 * np.sin(2.0 * np.pi * 440.0 * np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE)).astype(np.float32)
        with self.assertRaises(ValueError):
            enroll_from_samples(tone, sr=SAMPLE_RATE)

    def test_isolates_only_registered_speaker(self) -> None:
        fp = _enrolled_speaker_a()
        iso = VoiceIsolator(fp)

        spk_a = _synth_voice(1.0, 128.0, [(520, 95, 1.0), (1500, 135, 0.7), (2480, 180, 0.4)], seed=10)
        spk_b_diff_pitch = _synth_voice(1.0, 225.0, [(650, 110, 1.0), (1850, 150, 0.7), (2950, 200, 0.4)], seed=11)
        spk_c_same_pitch = _synth_voice(1.0, 128.0, [(720, 110, 1.0), (1950, 160, 0.8), (3200, 220, 0.5)], seed=12)
        noise = np.random.default_rng(13).normal(0.0, 0.14, SAMPLE_RATE).astype(np.float32)
        tone = (0.25 * np.sin(2.0 * np.pi * 660.0 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)).astype(np.float32)

        s_a, v_a, _ = iso.score_samples(spk_a)
        s_b, _, _ = iso.score_samples(spk_b_diff_pitch)
        s_c, _, _ = iso.score_samples(spk_c_same_pitch)
        s_n, v_n, _ = iso.score_samples(noise)
        s_t, v_t, _ = iso.score_samples(tone)

        self.assertTrue(v_a)
        self.assertGreaterEqual(s_a, iso.threshold + 0.15)
        self.assertLess(s_b, iso.threshold)
        self.assertLess(s_c, iso.threshold)
        self.assertFalse(v_n)
        self.assertLess(s_n, iso.threshold)
        self.assertFalse(v_t)
        self.assertLess(s_t, iso.threshold)

    def test_stream_disregards_other_signals_and_stops_on_silence(self) -> None:
        fp = _enrolled_speaker_a()
        iso = VoiceIsolator(fp, end_silence_ms=260.0)

        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 0.08, int(0.8 * SAMPLE_RATE)).astype(np.float32)
        other_voice = _synth_voice(0.8, 220.0, [(650, 110, 1.0), (1850, 150, 0.7), (2950, 200, 0.4)], seed=43)
        reg_voice = _synth_voice(1.0, 128.0, [(510, 95, 1.0), (1480, 135, 0.7), (2460, 180, 0.4)], seed=44)
        silence = np.zeros(int(0.6 * SAMPLE_RATE), dtype=np.float32)

        stream = np.concatenate([noise, other_voice, reg_voice, silence])
        reg_start = len(noise) + len(other_voice)
        reg_end = reg_start + len(reg_voice)

        started_events = []
        stopped_events = []
        leaked_before = 0
        emitted_during = 0

        for pos in range(0, len(stream) - 1024 + 1, 1024):
            res = iso.process_chunk(float32_to_pcm16(stream[pos:pos + 1024]), sr=SAMPLE_RATE)
            if res.started:
                started_events.append(pos)
            if res.stopped:
                stopped_events.append(pos)
            if pos + 1024 <= reg_start and res.emit_audio:
                leaked_before += 1
            if pos >= reg_start and pos < reg_end and res.emit_audio:
                emitted_during += len(res.emit_audio)

        self.assertEqual(leaked_before, 0, "Foreign speaker or noise leaked before registered voice!")
        self.assertEqual(len(started_events), 1)
        self.assertEqual(len(stopped_events), 1)
        self.assertGreater(emitted_during, 0)
        # Stopped shortly after registered voice ended
        self.assertGreaterEqual(stopped_events[0], reg_end - 1024)
        self.assertLessEqual(stopped_events[0], reg_end + int(0.45 * SAMPLE_RATE))

    def test_stream_transitions_when_foreign_speaker_takes_over(self) -> None:
        fp = _enrolled_speaker_a()
        iso = VoiceIsolator(fp, end_silence_ms=320.0)
        reg_voice = _synth_voice(0.9, 128.0, [(510, 95, 1.0), (1480, 135, 0.7), (2460, 180, 0.4)], seed=51)
        other_voice = _synth_voice(0.9, 225.0, [(650, 110, 1.0), (1850, 150, 0.7), (2950, 200, 0.4)], seed=52)
        stream = np.concatenate([reg_voice, other_voice])

        stopped = False
        other_emitted = 0
        for pos in range(0, len(stream) - 1024 + 1, 1024):
            res = iso.process_chunk(float32_to_pcm16(stream[pos:pos + 1024]), sr=SAMPLE_RATE)
            if pos >= len(reg_voice) + 1024 and res.emit_audio:
                other_emitted += 1
            if res.stopped:
                stopped = True
        self.assertTrue(stopped)
        self.assertEqual(other_emitted, 0)

    def test_config_manager_fingerprint_persistence(self) -> None:
        fp = _enrolled_speaker_a()
        orig_dir = config_manager.CONFIG_DIR
        orig_cfg = config_manager.CONFIG_FILE
        orig_fp = config_manager.VOICE_FINGERPRINT_FILE
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                config_manager.CONFIG_DIR = tmp_path
                config_manager.CONFIG_FILE = tmp_path / "api_keys.json"
                config_manager.VOICE_FINGERPRINT_FILE = tmp_path / "voice_fingerprint.json"

                self.assertFalse(config_manager.has_voice_fingerprint())
                self.assertFalse(config_manager.is_voice_isolation_active())

                config_manager.save_voice_fingerprint(fp.to_dict())
                config_manager.save_voice_isolation_enabled(True)
                self.assertTrue(config_manager.has_voice_fingerprint())
                self.assertTrue(config_manager.is_voice_isolation_active())

                loaded = VoiceFingerprint.from_dict(config_manager.get_voice_fingerprint())
                self.assertEqual(loaded.voiced_frames, fp.voiced_frames)

                config_manager.clear_voice_fingerprint()
                self.assertFalse(config_manager.has_voice_fingerprint())
                self.assertFalse(config_manager.is_voice_isolation_active())
        finally:
            config_manager.CONFIG_DIR = orig_dir
            config_manager.CONFIG_FILE = orig_cfg
            config_manager.VOICE_FINGERPRINT_FILE = orig_fp

    def test_pipeline_engine_immediate_flush_on_activity_end(self) -> None:
        class FakeSTT:
            detected_language = "en"

            async def transcribe(self, pcm: bytes, sr: int = 16000) -> str:
                return "hello jarvis" if pcm else ""

        class FakeLLM:
            caps = {"tools": False, "vision": False}

            async def chat(self, messages, tools=None):
                yield {"type": "text", "text": "Thought and answered."}
                yield {"type": "done"}

        async def _run():
            eng = PipelineEngine(FakeLLM(), stt=FakeSTT(), tts=None)
            eng.auto_turn = False
            await eng.send_realtime_input(activity_start={})
            pcm = float32_to_pcm16(_synth_voice(0.5, 128.0, [(500, 90, 1.0), (1500, 130, 0.7)], seed=61))
            await eng.send_realtime_input(audio={"data": pcm})
            # Should NOT wait for 700ms silence; activity_end flushes immediately
            await eng.send_realtime_input(activity_end={})
            events = []
            async for msg in eng.receive():
                events.append(msg)
                if msg.server_content and msg.server_content.turn_complete:
                    break
            await eng.close()
            return events

        events = asyncio.run(_run())
        heard = [
            e.server_content.input_transcription.text
            for e in events
            if e.server_content and e.server_content.input_transcription
        ]
        said = [
            e.server_content.output_transcription.text
            for e in events
            if e.server_content and e.server_content.output_transcription
        ]
        self.assertEqual(heard, ["hello jarvis"])
        self.assertTrue(any("Thought and answered." in s for s in said))

    def test_main_send_realtime_transitions_to_thought_mode(self) -> None:
        import main as app_main

        fp = _enrolled_speaker_a()
        dummy_ui = MagicMock()
        dummy_ui.muted = False

        jl = app_main.JarvisLive.__new__(app_main.JarvisLive)
        jl.ui = dummy_ui
        jl._awake = True
        jl._dashboard = None
        jl._loop = None
        jl._engine = None
        jl._explicit_vad_active = True
        jl._voice_isolator = VoiceIsolator(fp, end_silence_ms=200.0)
        jl._audio_gate = jl._voice_isolator
        jl._tuned_live = True
        jl._last_user_speech = 0.0
        jl._thinking_since = None
        jl._rx_deadline = None
        jl._private_mode = False
        jl._screen_watch_enabled = False

        # _tuning_config must disable AAD and default thinking_budget to 0 (fast) when thinking_enabled=False
        orig_cfg = config_manager.CONFIG_FILE
        try:
            with tempfile.TemporaryDirectory() as td:
                config_manager.CONFIG_FILE = Path(td) / "api_keys.json"
                config_manager.save_thinking_enabled(False)
                cfg = jl._tuning_config()
        finally:
            config_manager.CONFIG_FILE = orig_cfg
        self.assertIn("realtime_input_config", cfg)
        self.assertTrue(cfg["realtime_input_config"].automatic_activity_detection.disabled)
        self.assertIn("thinking_config", cfg)
        self.assertEqual(cfg["thinking_config"].thinking_budget, 0)

        sent_calls = []

        class FakeSession:
            async def send_realtime_input(self, **kw):
                sent_calls.append(kw)

        async def _run():
            jl.session = FakeSession()
            jl.out_queue = asyncio.Queue()

            noise = float32_to_pcm16(np.random.default_rng(77).normal(0.0, 0.08, 2048).astype(np.float32))
            reg = float32_to_pcm16(_synth_voice(0.7, 128.0, [(510, 95, 1.0), (1480, 135, 0.7), (2460, 180, 0.4)], seed=78))
            silence = float32_to_pcm16(np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32))

            for buf in (noise, reg, silence):
                for p in range(0, len(buf) - 2048 + 1, 2048):
                    await jl.out_queue.put({"data": buf[p:p + 2048], "mime_type": "audio/pcm"})

            task = asyncio.create_task(jl._send_realtime())
            while not jl.out_queue.empty():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())

        has_activity_start = any("activity_start" in c for c in sent_calls)
        has_audio = any("audio" in c for c in sent_calls)
        has_activity_end = any("activity_end" in c for c in sent_calls)
        self.assertTrue(has_activity_start)
        self.assertTrue(has_audio)
        self.assertTrue(has_activity_end)
        dummy_ui.set_state.assert_any_call("THINKING")

    def test_smart_speech_gate_filters_noise_and_stops_immediately(self) -> None:
        gate = SmartSpeechGate(end_silence_ms=250.0)
        rng = np.random.default_rng(88)
        noise = rng.normal(0.0, 0.09, int(0.6 * SAMPLE_RATE)).astype(np.float32)
        tone = (0.25 * np.sin(2.0 * np.pi * 500.0 * np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE)).astype(np.float32)
        speech = _synth_voice(0.8, 175.0, [(550, 100, 1.0), (1650, 140, 0.7)], seed=89)
        silence = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)

        stream = np.concatenate([noise, tone, speech, silence])
        speech_start = len(noise) + len(tone)
        leaked = 0
        started = stopped = False
        for pos in range(0, len(stream) - 1024 + 1, 1024):
            res = gate.process_chunk(float32_to_pcm16(stream[pos:pos + 1024]), sr=SAMPLE_RATE)
            if pos + 1024 <= speech_start and res.emit_audio:
                leaked += 1
            if res.started:
                started = True
            if res.stopped:
                stopped = True
        self.assertEqual(leaked, 0)
        self.assertTrue(started)
        self.assertTrue(stopped)

    def test_listening_modes_and_max_turn_safety_cap(self) -> None:
        # max_turn_ms forces stop so a stuck signal never hangs in LISTENING
        gate = SmartSpeechGate(end_silence_ms=300.0, max_turn_ms=3000.0)
        long_speech = _synth_voice(3.5, 160.0, [(520, 95, 1.0), (1500, 135, 0.7)], seed=91)
        stopped_at = None
        for pos in range(0, len(long_speech) - 1024 + 1, 1024):
            res = gate.process_chunk(float32_to_pcm16(long_speech[pos:pos + 1024]), sr=SAMPLE_RATE)
            if res.stopped:
                stopped_at = pos / float(SAMPLE_RATE)
                break
        self.assertIsNotNone(stopped_at)
        self.assertLessEqual(stopped_at, 3.3)

        orig_dir = config_manager.CONFIG_DIR
        orig_cfg = config_manager.CONFIG_FILE
        orig_fp = config_manager.VOICE_FINGERPRINT_FILE
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                config_manager.CONFIG_DIR = tmp_path
                config_manager.CONFIG_FILE = tmp_path / "api_keys.json"
                config_manager.VOICE_FINGERPRINT_FILE = tmp_path / "voice_fingerprint.json"

                self.assertEqual(config_manager.get_listening_mode(), "smart_vad")
                # Without fingerprint, voice_lock falls back to smart_vad
                self.assertEqual(config_manager.save_listening_mode("voice_lock"), "smart_vad")
                self.assertEqual(config_manager.save_listening_mode("continuous"), "continuous")
                self.assertEqual(config_manager.get_listening_mode(), "continuous")

                fp = _enrolled_speaker_a()
                config_manager.save_voice_fingerprint(fp.to_dict())
                self.assertEqual(config_manager.save_listening_mode("voice_lock"), "voice_lock")
                self.assertEqual(config_manager.get_listening_mode(), "voice_lock")
        finally:
            config_manager.CONFIG_DIR = orig_dir
            config_manager.CONFIG_FILE = orig_cfg
            config_manager.VOICE_FINGERPRINT_FILE = orig_fp

    def test_plugin_hot_reload_and_voice_lock_guard(self) -> None:
        from core.plugin_loader import discover_plugins

        orig_dir = config_manager.CONFIG_DIR
        orig_cfg = config_manager.CONFIG_FILE
        orig_fp = config_manager.VOICE_FINGERPRINT_FILE
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                plug_dir = tmp_path / "plugins"
                cfg_dir = tmp_path / "cfg"
                plug_dir.mkdir()
                cfg_dir.mkdir()

                config_manager.CONFIG_DIR = cfg_dir
                config_manager.CONFIG_FILE = cfg_dir / "api_keys.json"
                config_manager.VOICE_FINGERPRINT_FILE = cfg_dir / "voice_fingerprint.json"

                (plug_dir / "guarded_test.py").write_text(
                    "PLUGIN = {\n"
                    "    'name': 'guarded_test',\n"
                    "    'description': 'Test guarded plugin',\n"
                    "    'require_voice_lock': True,\n"
                    "    'parameters': {'type': 'OBJECT', 'properties': {}},\n"
                    "}\n"
                    "def run(parameters, player=None, session_memory=None):\n"
                    "    return 'ran-ok'\n",
                    encoding="utf-8",
                )

                reg = discover_plugins(plug_dir, set(), logger=lambda _m: None)
                self.assertTrue(reg.has("guarded_test"))

                # No fingerprint yet -> allowed
                self.assertEqual(reg.run("guarded_test", {}), "ran-ok")

                # Fingerprint saved, Voice Lock OFF -> blocked
                fp = _enrolled_speaker_a()
                config_manager.save_voice_fingerprint(fp.to_dict())
                config_manager.save_listening_mode("smart_vad")
                blocked = reg.run("guarded_test", {})
                self.assertIn("requires Voice Lock", blocked)

                # Voice Lock ON -> allowed
                config_manager.save_listening_mode("voice_lock")
                self.assertEqual(reg.run("guarded_test", {}), "ran-ok")

                # Hot-reload picks up a newly added plugin and drops a removed one
                (plug_dir / "second_test.py").write_text(
                    "PLUGIN = {'name': 'second_test', 'description': 'Second', 'parameters': {'type': 'OBJECT', 'properties': {}}}\n"
                    "def run(parameters):\n"
                    "    return 'two'\n",
                    encoding="utf-8",
                )
                (plug_dir / "guarded_test.py").unlink()
                active, rejected = reg.reload()
                self.assertEqual((active, rejected), (1, 0))
                self.assertFalse(reg.has("guarded_test"))
                self.assertTrue(reg.has("second_test"))
                self.assertEqual(reg.run("second_test", {}), "two")
        finally:
            config_manager.CONFIG_DIR = orig_dir
            config_manager.CONFIG_FILE = orig_cfg
            config_manager.VOICE_FINGERPRINT_FILE = orig_fp

    def test_voice_barge_in_and_reply_language_lock(self) -> None:
        import main as app_main

        orig_dir = config_manager.CONFIG_DIR
        orig_cfg = config_manager.CONFIG_FILE
        orig_fp = config_manager.VOICE_FINGERPRINT_FILE
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                config_manager.CONFIG_DIR = tmp_path
                config_manager.CONFIG_FILE = tmp_path / "api_keys.json"
                config_manager.VOICE_FINGERPRINT_FILE = tmp_path / "voice_fingerprint.json"

                self.assertEqual(config_manager.save_reply_language("hinglish"), "hinglish")
                self.assertEqual(config_manager.get_reply_language(), "hinglish")
                config_manager.save_voice_barge_in_enabled(True)
                self.assertTrue(config_manager.get_voice_barge_in_enabled())
        finally:
            config_manager.CONFIG_DIR = orig_dir
            config_manager.CONFIG_FILE = orig_cfg
            config_manager.VOICE_FINGERPRINT_FILE = orig_fp

        dummy_ui = MagicMock()
        dummy_ui.muted = False
        jl = app_main.JarvisLive.__new__(app_main.JarvisLive)
        jl.ui = dummy_ui
        jl._speaking_lock = __import__("threading").Lock()
        jl._is_speaking = True
        jl._server_turn_active = False
        jl._interrupted = False
        jl._thinking_since = None
        jl._barge_in_streak = 4
        jl._audio_gate = None
        jl._voice_isolator = None
        jl.audio_in_queue = asyncio.Queue()
        jl._visemes = MagicMock()
        jl._turn_done_event = None
        jl._engine = None
        jl._out_latency = 0.2

        # When server already finished sending (turn_complete arrived), interrupting
        # playback must NOT leave _interrupted=True to drop the next reply.
        jl._on_voice_barge_in()
        self.assertFalse(jl._is_speaking)
        self.assertFalse(jl._interrupted)

        # When server is still mid-turn, interrupt marks _interrupted=True
        jl._is_speaking = True
        jl._server_turn_active = True
        jl._on_voice_barge_in()
        self.assertFalse(jl._is_speaking)
        self.assertTrue(jl._interrupted)

        # Changing reply language clears session resumption (keep_context=False)
        jl._active_reply_lang = "auto"
        jl._refresh_audio_gate = MagicMock()
        jl.request_reconnect = MagicMock()
        with tempfile.TemporaryDirectory() as tmp2:
            orig_dir = config_manager.CONFIG_DIR
            orig_cfg = config_manager.CONFIG_FILE
            orig_fp = config_manager.VOICE_FINGERPRINT_FILE
            try:
                tp = Path(tmp2)
                config_manager.CONFIG_DIR = tp
                config_manager.CONFIG_FILE = tp / "api_keys.json"
                config_manager.VOICE_FINGERPRINT_FILE = tp / "voice_fingerprint.json"
                config_manager.save_reply_language("hinglish")
                jl._on_audio_device_change()
                jl.request_reconnect.assert_called_with(
                    keep_context=False, reason="reply language (hinglish)"
                )
            finally:
                config_manager.CONFIG_DIR = orig_dir
                config_manager.CONFIG_FILE = orig_cfg
                config_manager.VOICE_FINGERPRINT_FILE = orig_fp

    def test_voice_lock_preserves_unvoiced_consonants_and_brief_pauses(self) -> None:
        fp = _enrolled_speaker_a()
        iso = VoiceIsolator(fp)  # default threshold=0.52, end_silence_ms=480.0, denoise=False
        self.assertGreaterEqual(iso.end_silence_ms, 480.0)
        iso_slow = VoiceIsolator(fp, end_silence_ms=650.0)
        self.assertGreaterEqual(iso_slow.end_silence_ms, 650.0)

        rng = np.random.default_rng(77)
        p1 = _synth_voice(0.55, 128.0, [(510, 95, 1.0), (1480, 135, 0.7), (2460, 180, 0.4)], seed=71)
        # Simulated unvoiced consonant ('s'/'t'/'k' burst ~160ms) + brief 280ms natural pause
        unvoiced = rng.normal(0.0, 0.04, int(0.16 * SAMPLE_RATE)).astype(np.float32)
        pause = np.zeros(int(0.28 * SAMPLE_RATE), dtype=np.float32)
        p2 = _synth_voice(0.55, 128.0, [(510, 95, 1.0), (1480, 135, 0.7), (2460, 180, 0.4)], seed=72)
        end_sil = np.zeros(int(0.85 * SAMPLE_RATE), dtype=np.float32)
        stream = np.concatenate([p1, unvoiced, pause, p2, end_sil])

        starts, stops = 0, 0
        for pos in range(0, len(stream) - 1024 + 1, 1024):
            res = iso.process_chunk(float32_to_pcm16(stream[pos:pos + 1024]), sr=SAMPLE_RATE)
            if res.started:
                starts += 1
            if res.stopped:
                stops += 1
        self.assertEqual(starts, 1, "Sentence should stay in 1 continuous turn")
        self.assertEqual(stops, 1, "Should stop once after 650ms end silence")

    def test_inbuilt_browser_target_plugin_and_engine(self) -> None:
        import time
        from core.inbuilt_browser import InbuiltBrowserEngine, resolve_target
        from plugins import inbuilt_browser as browser_plugin
        from plugins import hud_dashboard as dash_plugin

        self.assertEqual(resolve_target(""), "https://www.google.com")
        self.assertEqual(resolve_target("youtube.com"), "https://youtube.com")
        self.assertEqual(resolve_target("localhost:8080"), "http://localhost:8080")
        self.assertEqual(resolve_target("https://openai.com"), "https://openai.com")
        self.assertIn("google.com/search?q=", resolve_target("weather in delhi"))

        player = MagicMock()
        r_open = browser_plugin.run({"action": "open", "target": "github.com"}, player=player)
        self.assertIn("Opened", r_open)
        player.open_browser.assert_called_with("github.com")

        r_search = browser_plugin.run({"action": "search", "target": "python basics"}, player=player)
        self.assertIn("Opened", r_search)
        player.open_browser.assert_called_with("https://www.google.com/search?q=python+basics")

        r_nt = browser_plugin.run({"action": "new_tab", "target": "openai.com"}, player=player)
        self.assertIn("new inbuilt browser tab", r_nt)
        player.new_browser_tab.assert_called_with("openai.com")

        r_sw = browser_plugin.run({"action": "switch_tab", "tab_index": 2}, player=player)
        self.assertIn("tab 2", r_sw)
        player.switch_browser_tab.assert_called_with(1)

        r_ct = browser_plugin.run({"action": "close_tab", "tab_index": 2}, player=player)
        self.assertIn("Closed", r_ct)
        player.close_browser_tab.assert_called_with(1)

        r_close = browser_plugin.run({"action": "close"}, player=player)
        self.assertIn("Closed", r_close)
        player.close_browser.assert_called_once()

        r_dash = dash_plugin.run({"feature": "memory"}, player=player)
        self.assertIn("system dashboard", r_dash)
        player.open_dashboard_panel.assert_called_with("memory", "")

        r_dash_b = dash_plugin.run({"feature": "browser", "target": "docs.python.org"}, player=player)
        self.assertIn("system dashboard browser", r_dash_b)
        player.open_dashboard_panel.assert_called_with("browser", "docs.python.org")

        r_dash_bad = dash_plugin.run({"feature": "unknown_panel"}, player=player)
        self.assertIn("Unknown system dashboard feature", r_dash_bad)

        frames: list[tuple[bytes, str]] = []
        statuses: list[str] = []
        tab_events: list[tuple[list, int]] = []
        with tempfile.TemporaryDirectory() as tmp:
            eng = InbuiltBrowserEngine(
                profile_dir=Path(tmp) / "prof",
                on_frame=lambda jpg, url: frames.append((jpg, url)),
                on_status=lambda msg: statuses.append(msg),
                on_tabs=lambda tabs, idx: tab_events.append((tabs, idx)),
            )
            try:
                eng.navigate("data:text/html,<title>TabOne</title><h1>OK</h1><input id='q'>")
                deadline = time.monotonic() + 12.0
                while time.monotonic() < deadline and not frames:
                    time.sleep(0.1)
                self.assertTrue(frames, f"Expected browser JPEG frame; statuses={statuses}")
                jpg, url = frames[-1]
                self.assertTrue(jpg.startswith(b"\xff\xd8"))
                self.assertIn("TabOne", url)

                # Multi-tab lifecycle: open 2nd tab, switch back to 1st, close 2nd
                eng.new_tab("data:text/html,<title>TabTwo</title><h2>Two</h2>")
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline and not any(len(t) == 2 and i == 1 for t, i in tab_events):
                    time.sleep(0.1)
                self.assertTrue(any(len(t) == 2 and i == 1 for t, i in tab_events), f"tabs={tab_events}")

                eng.switch_tab(0)
                deadline = time.monotonic() + 6.0
                while time.monotonic() < deadline and not (tab_events and tab_events[-1][1] == 0):
                    time.sleep(0.05)
                self.assertEqual(tab_events[-1][1], 0)

                eng.close_tab(1)
                deadline = time.monotonic() + 6.0
                while time.monotonic() < deadline and not (tab_events and len(tab_events[-1][0]) == 1):
                    time.sleep(0.05)
                self.assertEqual(len(tab_events[-1][0]), 1)

                eng.click(0.5, 0.2)
                eng.type_text("hello")
                eng.key_press("Tab")
                eng.key_press("Enter")
                eng.set_visible(False)
            finally:
                eng.stop(timeout=5.0)

    def test_uploaded_image_analysis_and_gemini_key_config(self) -> None:
        from PIL import Image
        import main as app_main

        dummy_ui = MagicMock()
        dummy_ui.muted = False
        jl = app_main.JarvisLive.__new__(app_main.JarvisLive)
        jl.ui = dummy_ui
        jl._engine_caps = None
        jl._wake_enabled = False
        jl._awake = True
        jl._loop = asyncio.new_event_loop()
        sent_coros = []

        class FakeSession:
            async def send_client_content(self, **kw):
                sent_coros.append(kw)

        jl.session = FakeSession()
        jl._enter_thought_mode = MagicMock()
        jl._action_registry = MagicMock()

        try:
            with tempfile.TemporaryDirectory() as tmp:
                img_path = Path(tmp) / "sample_err.png"
                Image.new("RGB", (64, 64), (10, 120, 220)).save(img_path)

                orig_rc = asyncio.run_coroutine_threadsafe
                captured = []
                def _fake_rc(coro, loop):
                    captured.append(coro)
                    return MagicMock()
                asyncio.run_coroutine_threadsafe = _fake_rc
                try:
                    jl._on_image_uploaded(str(img_path))
                finally:
                    asyncio.run_coroutine_threadsafe = orig_rc

                self.assertEqual(len(captured), 1)
                jl._loop.run_until_complete(captured[0])
                self.assertEqual(len(sent_coros), 1)
                parts = sent_coros[0]["turns"]["parts"]
                self.assertEqual(parts[0]["inline_data"]["mime_type"], "image/jpeg")
                self.assertTrue(len(parts[0]["inline_data"]["data"]) > 20)
                self.assertIn("IMAGE_UPLOADED", parts[1]["text"])
                jl._enter_thought_mode.assert_called_once()
        finally:
            jl._loop.close()

    def test_enrollment_handles_quiet_mic_50hz_hum_and_8khz_bluetooth(self) -> None:
        from core.voice_fingerprint import resample_float32

        # 1. Quiet laptop/USB mic (~ -48 dBFS)
        quiet = np.concatenate([
            _synth_voice(1.5, 130.0, [(500, 90, 1.0), (1450, 130, 0.7)], amp=0.004, seed=91),
            _synth_voice(1.5, 135.0, [(350, 80, 1.0), (2100, 150, 0.8)], amp=0.004, seed=92),
        ])
        fp_q = enroll_from_samples(quiet, sr=SAMPLE_RATE)
        self.assertGreaterEqual(fp_q.voiced_frames, 40)

        # 2. Speech with strong 50 Hz AC/fan hum
        t = np.arange(quiet.size, dtype=np.float32) / float(SAMPLE_RATE)
        hummed = _synth_voice(3.0, 140.0, [(520, 95, 1.0), (1500, 135, 0.7)], amp=0.05, seed=93)
        hummed = hummed + (0.04 * np.sin(2.0 * np.pi * 50.0 * t[:hummed.size])).astype(np.float32)
        fp_h = enroll_from_samples(hummed, sr=SAMPLE_RATE)
        self.assertGreaterEqual(fp_h.voiced_frames, 40)

        # 3. 8 kHz Bluetooth Hands-Free headset sample rate
        bt_8k = resample_float32(hummed, SAMPLE_RATE, 8000)
        fp_bt = enroll_from_samples(bt_8k, sr=8000)
        self.assertGreaterEqual(fp_bt.voiced_frames, 40)

    def test_screen_watch_companion_chatter_and_sweet_voices(self) -> None:
        import io
        from PIL import Image
        import main as app_main
        from actions.proactive import ProactiveEngine

        self.assertIn("Leda", config_manager.AVAILABLE_VOICES)
        self.assertIn("Zephyr", config_manager.AVAILABLE_VOICES)
        self.assertIn("Aoede", config_manager.AVAILABLE_VOICES)
        self.assertEqual(config_manager.DEFAULT_VOICE, "Leda")

        pe = ProactiveEngine()
        import time
        self.assertTrue(pe.should_trigger(time.monotonic() - 35.0, min_silence=28.0, cooldown=38.0))
        prompt = pe.build_prompt(
            memory={},
            companion=True,
            screen_watch_active=True,
            screen_changed=True,
            reply_language="hinglish",
        )
        self.assertIn("screen", prompt.lower())
        self.assertIn("Hinglish", prompt)
        self.assertIn("NEVER speak Spanish", prompt)

        b1, b2 = io.BytesIO(), io.BytesIO()
        Image.new("RGB", (64, 36), (10, 10, 10)).save(b1, format="JPEG")
        Image.new("RGB", (64, 36), (220, 220, 220)).save(b2, format="JPEG")
        d1, t1 = app_main.JarvisLive._screen_diff_score(b1.getvalue(), None)
        d_same, _ = app_main.JarvisLive._screen_diff_score(b1.getvalue(), t1)
        d_diff, _ = app_main.JarvisLive._screen_diff_score(b2.getvalue(), t1)
        self.assertGreater(d1, 200.0)
        self.assertLess(d_same, 2.0)
        self.assertGreater(d_diff, 150.0)

    def test_local_db_emotions_fast_thinking_and_raaz(self) -> None:
        import main as app_main
        from memory import local_db
        from core.avatar import HoloAvatar
        from core.voice_fingerprint import SmartSpeechGate

        # 1. Emotion detection across English + Hinglish cues
        emo_sad, _ = local_db.detect_emotion("I feel very sad and tired today, mood off hai")
        self.assertEqual(emo_sad, "sad")
        emo_happy, _ = local_db.detect_emotion("Yay awesome! Bahut khush hoon!")
        self.assertIn(emo_happy, ("happy", "excited"))
        emo_angry, _ = local_db.detect_emotion("Gussa aa raha hai, bakwas hai")
        self.assertEqual(emo_angry, "angry")
        emo_curious, _ = local_db.detect_emotion("Wait why is that happening? Kaise?")
        self.assertEqual(emo_curious, "curious")

        # 2. Local SQLite persistence in isolated temp DB
        orig_db = local_db.DB_PATH
        try:
            with tempfile.TemporaryDirectory() as tmp:
                local_db.DB_PATH = Path(tmp) / "test_jarvis.db"
                local_db.save_fact("favorite_drink", "masala chai", category="preferences")
                learned = local_db.learn_from_text("Mera naam Aman hai and I live in Pune. Always open Notepad before writing.")
                self.assertTrue(any(k == "user_name" for k, _ in learned))
                self.assertTrue(any(k.startswith("always_rule") for k, _ in learned))
                local_db.set_active_topic("Python Backend Refactor", locked=True)
                self.assertEqual(local_db.get_active_topic(), ("Python Backend Refactor", True))
                local_db.record_emotion("sad", 0.78, "tired day")
                local_db.record_turn("user", "Aaj mood sad hai", "sad")
                ctx = local_db.build_db_context()
                self.assertIn("LOCAL PERSISTENT KNOWLEDGE", ctx)
                self.assertIn("Python Backend Refactor", ctx)
                self.assertIn("STAY FOCUSED ON THIS TOPIC", ctx)
                self.assertIn("masala chai", ctx)
                self.assertIn("Recent user emotion trend: sad", ctx)
                stats = local_db.get_stats()
                self.assertGreaterEqual(stats["facts"], 2)
                self.assertEqual(stats["turns"], 1)
        finally:
            local_db.DB_PATH = orig_db

        # 3. Fast thinking budget = 0 even when _audio_gate is active
        jl = object.__new__(app_main.JarvisLive)
        jl._audio_gate = SmartSpeechGate(end_silence_ms=480.0)
        orig_cfg = config_manager.CONFIG_FILE
        try:
            with tempfile.TemporaryDirectory() as td:
                config_manager.CONFIG_FILE = Path(td) / "api_keys.json"
                config_manager.save_thinking_enabled(False)
                cfg = jl._tuning_config()
        finally:
            config_manager.CONFIG_FILE = orig_cfg
        self.assertIn("thinking_config", cfg)
        self.assertEqual(cfg["thinking_config"].thinking_budget, 0)

        # 4. Private / Raaz mode command parser
        self.assertEqual(app_main.JarvisLive._detect_private_command("private"), "on")
        self.assertEqual(app_main.JarvisLive._detect_private_command("raaz mode on karo"), "on")
        self.assertEqual(app_main.JarvisLive._detect_private_command("private off"), "off")
        self.assertEqual(app_main.JarvisLive._detect_private_command("normal mode"), "off")
        self.assertEqual(app_main.JarvisLive._detect_private_command("what is weather"), "")

        # 5. HoloAvatar facial expression weights & reaction animation
        av = HoloAvatar()
        for emo in ("sad", "happy", "angry", "private", "alert"):
            av.set_emotion(emo)
            for _ in range(18):
                av.step(0.05, 0.25)
            self.assertGreater(av._emo_weights.get(emo, 0.0), 0.7)
        av.trigger_reaction(1.0, 0.08)
        self.assertGreater(av._react_pulse, 0.8)

        # 6. Focus-safe computer_control typing + UI Mood presets
        from actions import computer_control as cc
        orig_focus = cc._focus_window
        orig_clip = cc._require_pyperclip
        orig_req_py = cc._require_pyautogui
        orig_click = cc._click
        calls = []
        try:
            cc._focus_window = lambda t: (calls.append(("focus", t)) or f"Focused {t}")
            cc._click = lambda *a, **kw: calls.append(("click", a[:2]))
            cc._require_pyperclip = lambda: True
            cc._require_pyautogui = lambda: None
            cc.pyperclip = MagicMock()
            cc.pyautogui = MagicMock()
            res_t = cc.computer_control({"action": "type", "title": "Notepad", "text": "नमस्ते दुनिया", "x": 120, "y": 240})
            self.assertIn("Focused Notepad", res_t)
            self.assertEqual(calls[0], ("focus", "Notepad"))
            self.assertEqual(calls[1], ("click", (120, 240)))
            cc.pyperclip.copy.assert_called_with("नमस्ते दुनिया")
        finally:
            cc._focus_window = orig_focus
            cc._click = orig_click
            cc._require_pyperclip = orig_clip
            cc._require_pyautogui = orig_req_py

        self.assertIn("aurora", config_manager.UI_MOOD_PRESETS)
        self.assertIn("rose", config_manager.UI_MOOD_PRESETS)

    def test_sleep_hello_song_pause_audio_dsp_and_dashboard_router(self) -> None:
        import asyncio
        import threading
        import main as app_main
        from core.voice_fingerprint import AudioFrontend
        from actions import video_player

        # 1. AudioFrontend SciPy DSP strips 50 Hz hum & amplifies quiet voice
        fe = AudioFrontend(sample_rate=16000)
        t = np.arange(2048, dtype=np.float32) / 16000.0
        hum_only = (0.08 * np.sin(2.0 * np.pi * 50.0 * t)).astype(np.float32)
        for _ in range(3):
            out_hum = fe.process_float32(hum_only)
        self.assertLess(float(np.sqrt(np.mean(out_hum ** 2))), 0.03)
        quiet_voice = _synth_voice(0.15, 145.0, [(520, 90, 1.0), (1550, 130, 0.7)], amp=0.045, seed=33)
        fe.reset()
        in_rms = float(np.sqrt(np.mean(quiet_voice ** 2)))
        for _ in range(4):
            out_voice = fe.process_float32(quiet_voice)
        out_rms = float(np.sqrt(np.mean(out_voice ** 2)))
        self.assertGreater(out_rms, in_rms * 1.2)

        # 2. video_player: sound ON by default + pause/resume + playlist/next controls
        p_mock = MagicMock()
        p_mock.video_is_playing.return_value = True
        with tempfile.TemporaryDirectory() as tmpd:
            vpath = Path(tmpd) / "song.mp4"
            vpath2 = Path(tmpd) / "song2.mp4"
            vpath.write_bytes(b"fake")
            vpath2.write_bytes(b"fake2")
            r_play = video_player.video_player({"action": "play", "source": str(vpath)}, player=p_mock)
            self.assertNotIn("muted", r_play.lower())
            p_mock.show_video.assert_called()
            self.assertFalse(p_mock.show_video.call_args.kwargs["muted"])
            r_pl = video_player.video_player({"action": "playlist", "source": f"{vpath}, {vpath2}"}, player=p_mock)
            self.assertIn("2 tracks", r_pl)
            r_nxt = video_player.video_player({"action": "next"}, player=p_mock)
            self.assertIn("2/2", r_nxt)
        r_pause = video_player.video_player({"action": "pause"}, player=p_mock)
        self.assertIn("Paused", r_pause)
        p_mock.pause_video.assert_called_once()
        r_res = video_player.video_player({"action": "resume"}, player=p_mock)
        self.assertIn("Resumed", r_res)
        p_mock.resume_video.assert_called_once()

        # 3. "sleeping" -> standby sleep; "hello" -> pause playing song + wake + activate AI
        jl = object.__new__(app_main.JarvisLive)
        jl.ui = MagicMock()
        jl.ui.muted = False
        jl.ui.video_is_playing.return_value = True
        jl.ui.video_is_paused.return_value = False
        jl.ui.video_has_sound.return_value = True
        jl._awake = True
        jl._sleep_standby = False
        jl._last_user_utterance = ""
        jl._speaking_lock = threading.Lock()
        jl._is_speaking = False
        jl._out_latency = 0.0
        jl._tail_until = 0.0
        jl._visemes = MagicMock()
        jl._barge_in_streak = 0
        jl._audio_gate = None

        # Random song lyrics while music plays must NOT pause song or trigger spurious tool calls
        jl._sleep_standby = False
        jl.ui.pause_video.reset_mock()
        res_lyr = jl._apply_sleep_or_hello_from_text("la la la random lyrics playing", from_voice=True)
        self.assertEqual(res_lyr, "")
        jl.ui.pause_video.assert_not_called()

        self.assertEqual(jl._apply_sleep_or_hello_from_text("sleep mode", from_voice=True), "sleep")
        self.assertTrue(jl._sleep_standby)
        self.assertFalse(jl.ui.muted)
        jl.ui.set_state.assert_called_with("SLEEPING")

        # Hindi Devanagari "हेलो" or "hello" both wake from sleep & pause song
        self.assertEqual(jl._apply_sleep_or_hello_from_text("हेलो", from_voice=True), "wake")
        self.assertFalse(jl._sleep_standby)
        jl.ui.pause_video.assert_called()
        jl.ui.pause_browser_media.assert_called()
        jl.ui.set_state.assert_called_with("LISTENING")

        # 4. Direct PC tool execution without requiring "system" keyword
        jl.ui.video_has_sound.return_value = False
        jl._last_user_utterance = "notepad kholo"
        jl._action_registry = MagicMock()
        jl._action_registry.has.return_value = True
        jl._action_registry.scheduling.return_value = None
        jl._action_registry.run.return_value = "Opened Notepad"
        jl._plugin_registry = MagicMock()
        jl._plugin_registry.has.return_value = False
        jl._plugin_registry.scheduling.return_value = None
        fc_app = MagicMock()
        fc_app.id = "1"
        fc_app.name = "open_app"
        fc_app.args = {"app_name": "notepad"}
        resp_app = asyncio.run(jl._execute_tool(fc_app))
        self.assertIn("Opened Notepad", resp_app.response["result"])
        self.assertNotIn("DASHBOARD_ONLY_MODE", resp_app.response["result"])
        jl._action_registry.run.assert_called_once()

        fc_browser = MagicMock()
        fc_browser.id = "2"
        fc_browser.name = "browser_control"
        fc_browser.args = {"action": "open", "url": "https://www.youtube.com"}
        jl._last_user_utterance = "open youtube"
        jl._action_registry.run.return_value = "Opened YouTube"
        resp_b = asyncio.run(jl._execute_tool(fc_browser))
        self.assertIn("Opened YouTube", resp_b.response["result"])
        self.assertEqual(jl._action_registry.run.call_count, 2)

    def test_firebase_realtime_training_and_fuzzy_voice_recovery(self) -> None:
        from memory import firebase_sync, local_db
        from actions import train_command

        orig_db = local_db.DB_PATH
        orig_enqueue = firebase_sync._enqueue
        pushed = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                local_db.DB_PATH = Path(tmp) / "fb_test.db"
                firebase_sync._enqueue = lambda fn, *a, **kw: pushed.append(fn)

                self.assertTrue(firebase_sync.is_configured())
                st = firebase_sync.get_status()
                self.assertEqual(st.get("project_id"), "ailive-1a4b1")

                # 1. Explicit training + unclear/misheard voice fuzzy matching
                res_ext = firebase_sync.extract_explicit_training(
                    "yaad rakhna jab main work mode bolu to open dashboard browser"
                )
                self.assertEqual(res_ext, ("work mode", "open dashboard browser"))

                r_act = train_command.train_command_action({
                    "mode": "train",
                    "phrase": "work mode",
                    "action": "open inbuilt browser with github.com",
                    "aliases": "wark mod, vrk mode",
                })
                self.assertIn("Trained 'work mode'", r_act)

                matched = firebase_sync.resolve_unclear_command("wurk mod")
                self.assertIsNotNone(matched)
                self.assertEqual(matched[0], "work mode")
                self.assertIn("github.com", matched[1])
                self.assertGreaterEqual(matched[2], 0.72)

                # 2. Important memory + exchange training (no raw AI replies saved to RTDB)
                r_imp = train_command.train_command_action({
                    "mode": "important",
                    "key": "laptop_pin",
                    "value": "keep secret",
                    "category": "notes",
                })
                self.assertIn("Saved important memory", r_imp)

                orig_rtdb_put = firebase_sync.rtdb_put
                rtdb_calls = []
                firebase_sync.rtdb_put = lambda path, payload, **kw: rtdb_calls.append((path, payload)) or True
                try:
                    # Casual greeting/chitchat without tools must NOT push anything to RTDB
                    n_before = len(pushed)
                    firebase_sync.train_from_exchange("hello", "Hi sir, how can I help?")
                    self.assertEqual(len(pushed), n_before)

                    # Tool turn learns pattern WITHOUT saving jarvis/training_turns or assistant text
                    firebase_sync.train_from_exchange(
                        "search python docs", "Opened docs.", tools_used=["inbuilt_browser"]
                    )
                    for fn in pushed[n_before:]:
                        fn()
                finally:
                    firebase_sync.rtdb_put = orig_rtdb_put

                self.assertFalse(any("training_turns" in p for p, _ in rtdb_calls))
                self.assertTrue(any(p.startswith("jarvis/learned_patterns/") for p, _ in rtdb_calls))
                for _p, pl in rtdb_calls:
                    self.assertNotIn("assistant", pl)

                ctx = local_db.build_db_context()
                self.assertIn("Trained commands, workflows", ctx)
                self.assertIn("PERMANENTLY LEARNED", ctx)
                self.assertIn("work mode", ctx)
                self.assertIn("inbuilt_browser", ctx)
                self.assertGreaterEqual(len(pushed), 2)
        finally:
            local_db.DB_PATH = orig_db
            firebase_sync._enqueue = orig_enqueue


if __name__ == "__main__":
    unittest.main(verbosity=2)

