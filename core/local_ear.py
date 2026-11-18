"""
core/local_ear.py — the PC listens first, Gemini hears only what was worth hearing.

WHY THIS FILE EXISTS
    With every other listening mode the microphone is streamed to Google and
    Google's server decides "the user stopped talking". Nothing on this machine
    can say "that was a song", "that was a cough", "that was too quiet to
    understand" or "that was hallucinated from fan noise". The assistant then
    answers things nobody said.

    This module moves the ear onto the PC:

        mic ──► ClearVoiceFrontend ──► LocalEarGate ──► Gemini
                 (hum/hiss removed)     │  1. finds the end of the sentence itself
                                        │  2. normalises the loudness of the WHOLE
                                        │     sentence (quiet voice up, shout down)
                                        │  3. optionally transcribes it locally
                                        │     (faster-whisper) and drops noise,
                                        │     hallucinations and song lyrics
                                        └► 4. sends ONE clean sentence, bracketed by
                                              explicit activity start/end, so the
                                              turn boundary is OURS, not Google's

THE TWO SPEEDS  (config/api_keys.json -> "local_ear" -> "speed")
    "fast"      End of speech found from the audio alone, no local transcription.
                Adds only the end-of-sentence wait (~0.55 s). Transcription is
                still used when it is needed — while a song plays or while asleep.
    "accurate"  Every sentence is transcribed locally before Gemini sees it.
                Adds the transcription time (about 0.3 s on a GPU, 1-2 s on a
                CPU) but nothing that is not real speech ever reaches Gemini.

SONG / SLEEP BEHAVIOUR
    While media is playing (or the assistant was put to sleep) the gate is
    SILENT: audio is never sent to Gemini. It transcribes locally and acts only
    when the sentence starts with the wake word ("Hello", "Hello Jarvis", …).
    Then it fires `on_wake` (which pauses the song and wakes the assistant) and
    sends only what was said AFTER the wake word.

HONEST LIMITS
    * Local recognition of "Hello" over loud music is best-effort. Headphones, or
      a song volume lower than your voice, make it far more reliable.
    * Without `faster-whisper` the gate still works (better DSP, own turn
      detection) but cannot verify wake words, so while media plays it stays
      silent and the existing push-to-talk / wake-word detector must be used.

Pure numpy; scipy and faster-whisper are optional.
"""
from __future__ import annotations

import collections
import difflib
import queue
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, Callable, Optional

import numpy as np

from core.voice_fingerprint import IsolationResult, resample_float32

SR = 16_000
FRAME = 320               # 20 ms @ 16 kHz
FRAME_MS = 20.0
_N_FFT = 512
_STFT_HOP = 256

# ─────────────────────────────────────────────────────────────────────────────
#  Defaults — every one can be overridden from config/api_keys.json["local_ear"]
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS: dict[str, Any] = {
    "speed": "accurate",          # "fast" | "accurate"
    "model": "small",             # faster-whisper model: tiny/base/small/medium
    "device": "auto",             # auto | cuda | cpu
    "language": "",               # "" = detect each time; "hi", "en" to pin
    "initial_prompt": "",         # optional vocabulary hint for Whisper
    "end_silence_ms": 0,          # 0 = pick by speed (800 accurate / 550 fast)
    "min_voiced_ms": 220,         # shorter than this is a click or a cough
    "margin": 2.2,                # speech must be this many x above the noise floor
    "media_margin": 2.6,          # …and this many x above the music level
    "abs_min_rms": 0.002,         # never trigger below this (hiss on a silent mic)
    "denoise": 0.5,               # 0 = off … 1 = strongest
    "target_rms": 0.10,           # loudness every sentence is brought to (~-20 dBFS)
    "max_gain_db": 30.0,          # how much a very quiet voice may be raised
    "min_gain_db": -12.0,         # how much a shout may be lowered
    "media_wake": True,           # stay silent while media plays; wake on "Hello"
    "wake_words": ["hello", "hallo", "halo", "helo", "hullo", "hellow",
                   "हेलो", "हैलो", "हेल्लो", "हलो"],
    "name_words": ["jarvis", "jarvish", "jervis", "jarwis", "javis", "जार्विस"],
    "name_wakes": True,           # "Jarvis" alone also wakes
    "stale_s": 6.0,               # a finished sentence older than this is dropped
}


def merged_config(user: Optional[dict]) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if isinstance(user, dict):
        for k, v in user.items():
            if k in DEFAULTS and v is not None:
                cfg[k] = v
    cfg["speed"] = "fast" if str(cfg["speed"]).lower().strip() == "fast" else "accurate"
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Small helpers
# ─────────────────────────────────────────────────────────────────────────────
def pcm16_to_f32(pcm: bytes) -> np.ndarray:
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def f32_to_pcm16(x: np.ndarray) -> bytes:
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2").tobytes()


_FILLER_ONLY = {"um", "uh", "umm", "uhh", "hmm", "hm", "mm", "mhm", "ah", "oh", "eh",
                "huh", "ha", "haha", "music", "applause", "laughter", "silence"}
_HALLUCINATIONS = {
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "thank you very much", "you", "bye", "bye bye", "goodbye",
    "please subscribe", "like and subscribe", "subtitles by the amaraorg community",
    "amaraorg", "mbc", "the end", "धन्यवाद",
}
_WAKE_FILLERS = {"hey", "hi", "ok", "okay", "oye", "arre", "are", "o", "ae", "ay"}


def normalize_text(text: str) -> str:
    """Lower-case, drop punctuation/symbols but KEEP letters and combining marks
    (so Devanagari survives — `\\w` would strip its vowel signs)."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    out = []
    for ch in t:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S", "C"):
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


# ─────────────────────────────────────────────────────────────────────────────
#  1. ClearVoiceFrontend — streaming clean-up in the mic callback
# ─────────────────────────────────────────────────────────────────────────────
class ClearVoiceFrontend:
    """Streaming DC/rumble removal + stationary-noise suppression.

    Deliberately does NOT change loudness. The old frontend did, per 64 ms block,
    and treated a soft voice (RMS < 0.006) as noise and attenuated it to 35% —
    which is exactly how a quiet speaker became unintelligible. Loudness is now
    fixed once per whole sentence in `normalize_utterance`, where the level of
    the speech can actually be measured.

    Same interface as `AudioFrontend`: process_pcm / process_float32 / reset.
    Output may be a few samples shorter or longer than the input on a given call
    (the STFT works in 256-sample hops); over time it is the same length, with a
    fixed 16 ms delay.
    """

    def __init__(self, sample_rate: int = SR, highpass_hz: float = 90.0,
                 denoise: float = 0.5) -> None:
        self._sr = int(sample_rate)
        self._strength = float(min(1.0, max(0.0, denoise)))
        self._lock = threading.Lock()
        try:
            from scipy import signal as _sig  # type: ignore
            self._sig = _sig
        except Exception:
            self._sig = None
        self._hp_hz = highpass_hz
        self._win = np.sin(np.pi * (np.arange(_N_FFT) + 0.5) / _N_FFT).astype(np.float32)
        self._alpha = 0.8 + 1.2 * self._strength
        self._gmin = 10.0 ** ((-6.0 - 12.0 * self._strength) / 20.0)
        self._freqs = np.fft.rfftfreq(_N_FFT, 1.0 / SR)
        self.reset()

    def reset(self) -> None:
        with self._lock:
            if self._sig is not None and 20.0 < self._hp_hz < self._sr * 0.4:
                self._sos = self._sig.butter(4, self._hp_hz, btype="highpass",
                                             fs=self._sr, output="sos")
                self._zi = self._sig.sosfilt_zi(self._sos) * 0.0
            else:
                self._sos = None
                self._zi = None
            self._px = 0.0
            self._py = 0.0
            self._inbuf = np.zeros(0, dtype=np.float32)
            self._olap = np.zeros(_STFT_HOP, dtype=np.float32)
            self._noise: Optional[np.ndarray] = None
            self._warm = 0
            self._gprev = np.ones(_N_FFT // 2 + 1, dtype=np.float32)

    # -- public ---------------------------------------------------------------
    def process_pcm(self, pcm: bytes) -> bytes:
        if not pcm:
            return pcm
        y = self.process_float32(pcm16_to_f32(pcm))
        return f32_to_pcm16(y)

    def process_float32(self, samples: np.ndarray) -> np.ndarray:
        if samples is None or samples.size == 0:
            return np.zeros(0, dtype=np.float32)
        x = np.asarray(samples, dtype=np.float32)
        with self._lock:
            x = self._highpass(x)
            if self._strength <= 0.0:
                return x
            return self._denoise(x)

    # -- internals ------------------------------------------------------------
    def _highpass(self, x: np.ndarray) -> np.ndarray:
        if self._sos is not None:
            y, self._zi = self._sig.sosfilt(self._sos, x, zi=self._zi)
            return y.astype(np.float32, copy=False)
        # 1-pole fallback (no scipy): y[n] = 0.97 * (y[n-1] + x[n] - x[n-1])
        out = np.empty_like(x)
        px, py = self._px, self._py
        for i in range(x.size):
            cur = float(x[i])
            py = 0.97 * (py + cur - px)
            px = cur
            out[i] = py
        self._px, self._py = px, py
        return out

    def _denoise(self, x: np.ndarray) -> np.ndarray:
        self._inbuf = np.concatenate([self._inbuf, x]) if self._inbuf.size else x
        outs: list[np.ndarray] = []
        w = self._win
        while self._inbuf.size >= _N_FFT:
            frame = self._inbuf[:_N_FFT] * w
            spec = np.fft.rfft(frame)
            psd = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)

            # --- stationary noise tracker (minimum follower, slow rise) -------
            if self._noise is None:
                self._noise = psd.copy()
            elif self._warm < 12:
                self._noise = 0.8 * self._noise + 0.2 * psd      # learn the room first
            else:
                down = psd < self._noise
                self._noise = np.where(down, 0.85 * self._noise + 0.15 * psd,
                                       self._noise * 1.0015).astype(np.float32)
            self._warm += 1

            # --- Wiener-style gain, fast attack / slow release ---------------
            # a minimum-follower sits BELOW the mean noise power; scale it back up
            snr = psd / (self._noise * 2.0 + 1e-9)
            g = np.clip(1.0 - self._alpha / np.maximum(snr, 1e-6), self._gmin, 1.0)
            g = np.convolve(g, [0.25, 0.5, 0.25], mode="same")      # less musical noise
            g = np.where(g > self._gprev, g, 0.7 * self._gprev + 0.3 * g).astype(np.float32)
            g[self._freqs < 70.0] = self._gmin
            self._gprev = g

            frame_out = np.fft.irfft(spec * g, n=_N_FFT).astype(np.float32) * w
            outs.append(frame_out[:_STFT_HOP] + self._olap)
            self._olap = frame_out[_STFT_HOP:]
            self._inbuf = self._inbuf[_STFT_HOP:]
        if not outs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outs)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Whole-sentence loudness normaliser  ("dheere" up, "taiz/loud" down)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_utterance(x: np.ndarray, target_rms: float = 0.10,
                        max_gain_db: float = 30.0, min_gain_db: float = -12.0
                        ) -> tuple[np.ndarray, dict]:
    """Bring one finished sentence to a steady speech level.

    The level is measured on the *speaking* frames only (frames within a factor
    of the loud ones), so pauses and the noise between words do not drag the
    estimate down and over-amplify the hiss. A soft-knee limiter then keeps a
    shout from clipping.
    """
    info = {"gain_db": 0.0, "level": 0.0, "peak": 0.0}
    if x.size < FRAME:
        return x.astype(np.float32), info
    x = np.asarray(x, dtype=np.float32)
    n = x.size // FRAME
    fr = x[: n * FRAME].reshape(n, FRAME)
    rms = np.sqrt(np.mean(fr * fr, axis=1) + 1e-12)
    p95 = float(np.percentile(rms, 95))
    active = rms >= max(0.3 * p95, 1e-4)
    level = float(np.sqrt(np.mean(rms[active] ** 2))) if active.any() else float(np.sqrt(np.mean(x * x)))
    info["level"] = level
    if level < 1e-5:
        return x, info
    gain = target_rms / level
    gain = float(min(10.0 ** (max_gain_db / 20.0), max(10.0 ** (min_gain_db / 20.0), gain)))
    y = x * gain
    # soft-knee limiter above 0.85
    knee = 0.85
    mag = np.abs(y)
    over = mag > knee
    if over.any():
        y = np.where(over, np.sign(y) * (knee + (1.0 - knee) * np.tanh((mag - knee) / (1.0 - knee))), y)
    # 5 ms fades so the burst never starts or ends on a click
    f = min(80, y.size // 4)
    if f > 1:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        y[:f] *= ramp
        y[-f:] *= ramp[::-1]
    info["gain_db"] = 20.0 * np.log10(gain)
    info["peak"] = float(np.max(np.abs(y)))
    return y.astype(np.float32), info


# ─────────────────────────────────────────────────────────────────────────────
#  3. Local transcription (faster-whisper), optional
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Transcript:
    text: str = ""
    words: list = field(default_factory=list)   # [(word, start_s, end_s), ...]
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    language: str = ""


class LocalTranscriber:
    """faster-whisper wrapped for a worker thread.

    The model loads on a background thread when the app starts, so the first
    sentence does not pay for it. GPU is tried first (float16, then
    int8_float16) and falls back to CPU int8 on ANY failure — a missing CUDA
    library is the commonest one and must never mean "cannot hear".
    """

    def __init__(self, model: str = "small", device: str = "auto", language: str = "",
                 initial_prompt: str = "", log: Optional[Callable[[str], None]] = None,
                 beam_size: int = 1) -> None:
        self.model_name = model
        self.device_pref = (device or "auto").lower()
        self.language = (language or "").strip().lower() or None
        self.initial_prompt = (initial_prompt or "").strip() or None
        self.beam_size = max(1, int(beam_size))
        self._log = log or (lambda s: None)
        self._model = None
        self._ready = threading.Event()
        self.error = ""
        self.device_used = ""
        threading.Thread(target=self._load, name="local-ear-load", daemon=True).start()

    def usable(self) -> bool:
        return not self.error

    def _candidates(self) -> list[tuple[str, str]]:
        want_cuda = self.device_pref in ("auto", "cuda")
        if want_cuda and self.device_pref == "auto":
            try:
                import ctranslate2  # type: ignore
                want_cuda = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                want_cuda = False
        out: list[tuple[str, str]] = []
        if want_cuda:
            out += [("cuda", "float16"), ("cuda", "int8_float16")]
        out.append(("cpu", "int8"))
        return out

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:
            self.error = f"faster-whisper not installed ({e})"
            self._ready.set()
            return
        last = ""
        for dev, comp in self._candidates():
            try:
                self._log(f"EAR: loading Whisper '{self.model_name}' on {dev} ({comp})…")
                m = WhisperModel(self.model_name, device=dev, compute_type=comp)
                # warm-up: the first real call is otherwise slow
                list(m.transcribe(np.zeros(SR, dtype=np.float32), beam_size=1)[0])
                self._model, self.device_used = m, f"{dev}/{comp}"
                self._log(f"EAR: Whisper ready on {self.device_used}.")
                self._ready.set()
                return
            except Exception as e:
                last = f"{dev}/{comp}: {e}"
                self._log(f"EAR: Whisper {dev}/{comp} failed — trying next ({str(e)[:120]})")
        self.error = last or "no device could load the model"
        self._ready.set()

    def transcribe(self, audio: np.ndarray, words: bool = False,
                   wait_s: float = 90.0) -> Optional[Transcript]:
        if not self._ready.wait(wait_s) or self._model is None:
            return None
        try:
            segs, info = self._model.transcribe(
                audio.astype(np.float32),
                language=self.language,
                beam_size=self.beam_size,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,                 # the gate already cut the speech
                word_timestamps=bool(words),
                initial_prompt=self.initial_prompt,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
            )
            parts, ws, lp, ns, n = [], [], 0.0, 0.0, 0
            for s in segs:
                parts.append(s.text)
                lp += float(getattr(s, "avg_logprob", 0.0))
                ns = max(ns, float(getattr(s, "no_speech_prob", 0.0)))
                n += 1
                for w in (getattr(s, "words", None) or []):
                    ws.append((str(w.word), float(w.start), float(w.end)))
            return Transcript(text=" ".join(p.strip() for p in parts).strip(), words=ws,
                              avg_logprob=(lp / n if n else 0.0), no_speech_prob=ns,
                              language=str(getattr(info, "language", "") or ""))
        except Exception as e:
            self._log(f"EAR: transcription error: {str(e)[:140]}")
            return None


def build_transcriber(cfg: dict, log: Optional[Callable[[str], None]] = None
                      ) -> Optional[LocalTranscriber]:
    """A transcriber, or None when faster-whisper is not installed."""
    if find_spec("faster_whisper") is None:
        if log:
            log("EAR: faster-whisper not installed — local verification off "
                "(pip install faster-whisper).")
        return None
    return LocalTranscriber(model=str(cfg.get("model", "small")),
                            device=str(cfg.get("device", "auto")),
                            language=str(cfg.get("language", "")),
                            initial_prompt=str(cfg.get("initial_prompt", "")),
                            log=log,
                            beam_size=2 if cfg.get("speed") == "accurate" else 1)


# ─────────────────────────────────────────────────────────────────────────────
#  4. Wake-phrase + hallucination logic (pure functions, easy to test)
# ─────────────────────────────────────────────────────────────────────────────
def _is_wake_token(tok: str, wake_set: set[str]) -> bool:
    if tok in wake_set:
        return True
    if tok.isascii() and len(tok) >= 4:
        return difflib.SequenceMatcher(None, tok, "hello").ratio() >= 0.78
    return False


def find_wake(text: str, wake_words, name_words, name_wakes: bool = True
              ) -> Optional[int]:
    """If the sentence BEGINS with a wake phrase, return how many leading tokens
    it uses (so the caller can cut them off); otherwise None.

    Accepts: "hello", "hey hello", "hello jarvis", "hey jarvis", "jarvis ...".
    A wake word buried later in the sentence does NOT count — a song lyric that
    happens to contain "hello" in its middle must not wake the assistant.
    """
    toks = normalize_text(text).split()
    if not toks:
        return None
    wake_set = {normalize_text(w) for w in wake_words}
    name_set = {normalize_text(w) for w in name_words}
    i = 0
    if toks[0] in _WAKE_FILLERS and len(toks) > 1:
        i = 1
    if i < len(toks) and _is_wake_token(toks[i], wake_set):
        i += 1
        if i < len(toks) and toks[i] in name_set:
            i += 1
        return i
    j = 1 if (toks[0] in _WAKE_FILLERS and len(toks) > 1) else 0
    if name_wakes and j < len(toks) and toks[j] in name_set:
        return j + 1
    return None


def looks_hallucinated(tr: Transcript, duration_s: float) -> Optional[str]:
    """Reason string if this transcript is probably not real speech."""
    norm = normalize_text(tr.text)
    if len(norm) < 2:
        return "empty"
    toks = norm.split()
    if all(t in _FILLER_ONLY for t in toks):
        return "filler"
    if tr.no_speech_prob > 0.7 and tr.avg_logprob < -0.8:
        return "no_speech"
    if tr.avg_logprob < -1.5:
        return "low_confidence"
    if norm in _HALLUCINATIONS and (tr.avg_logprob < -0.6 or tr.no_speech_prob > 0.3
                                    or duration_s < 0.8):
        return "known_hallucination"
    if "amara org" in norm or "subtitles by" in norm:
        return "known_hallucination"
    if len(toks) >= 6 and len(set(toks)) <= 2:
        return "repetition_loop"
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  5. The gate
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EarResult(IsolationResult):
    """IsolationResult plus the extras main.py can use. `live_level` is the
    current speech level for the HUD waveform (0 when it is only noise)."""
    rms: float = 0.0
    f0_hz: float = 0.0
    live_level: float = 0.0
    text: str = ""


@dataclass
class _Utterance:
    audio: np.ndarray
    voiced_ms: float
    ts: float
    need_wake: bool


@dataclass
class _Ready:
    pcm: bytes
    text: str
    ts: float
    level: float = 0.0          # RMS BEFORE normalisation (main.py uses it to spot whispering)


_SLICE_BYTES = 48_000           # 1.5 s of 16-bit mono per process_chunk call


class LocalEarGate:
    """Drop-in replacement for SmartSpeechGate (same process_chunk / reset /
    force_stop / active / end_silence_ms) that buffers a whole sentence, judges
    it locally, and releases it to Gemini in one clean burst."""

    live_meter = True

    def __init__(self, config: Optional[dict] = None,
                 transcriber: Optional[LocalTranscriber] = None,
                 require_wake_fn: Optional[Callable[[], bool]] = None,
                 on_wake: Optional[Callable[[str], None]] = None,
                 on_utterance: Optional[Callable[[str], None]] = None,
                 log: Optional[Callable[[str], None]] = None) -> None:
        self.cfg = merged_config(config)
        self.transcriber = transcriber
        self._require_wake = require_wake_fn or (lambda: False)
        self._on_wake = on_wake or (lambda t: None)
        self._on_utt = on_utterance or (lambda t: None)
        self._log = log or (lambda s: None)

        spd = self.cfg["speed"]
        es = float(self.cfg["end_silence_ms"] or 0)
        self._end_silence_ms = es if es > 0 else (550.0 if spd == "fast" else 800.0)
        self._min_voiced_ms = float(self.cfg["min_voiced_ms"])
        self._max_turn_ms = 25_000.0
        self._lock = threading.RLock()

        # band mask for "is this voice-like?" (100-4000 Hz). NOT 250+: a low male voice
        # keeps a large share of its energy at 100-200 Hz (its fundamental).
        fr = np.fft.rfftfreq(_N_FFT, 1.0 / SR)
        self._band = (fr >= 100.0) & (fr <= 4000.0)
        self._hann = np.hanning(FRAME).astype(np.float32)

        self._carry = np.zeros(0, dtype=np.float32)
        self._floor = 0.002
        self._ambient_hist: collections.deque[float] = collections.deque(maxlen=150)
        self._pre_roll: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, int(320 / FRAME_MS)))
        self._frames: list[np.ndarray] = []
        self._active = False
        self._speech_run = 0
        self._silence_ms = 0.0
        self._voiced_ms = 0.0
        self._level = 0.0
        self._media_ts = 0.0
        self._media_val = False

        self._ready: collections.deque[_Ready] = collections.deque()
        self._slices: collections.deque[tuple[bytes, bool, bool, str, float]] = collections.deque()
        self._pending_jobs = 0
        self._reject_streak = 0
        self._no_verify_until = 0.0

        self._q: "queue.Queue[Optional[_Utterance]]" = queue.Queue(maxsize=4)
        self._worker: Optional[threading.Thread] = None
        self._closed = False
        self.stats = {"sent": 0, "ignored": 0, "noise": 0, "woke": 0, "last_text": "",
                      "last_reason": ""}

    # -- compatibility surface -------------------------------------------------
    @property
    def active(self) -> bool:
        with self._lock:
            return self._active or bool(self._slices) or bool(self._ready) \
                or self._pending_jobs > 0

    @property
    def end_silence_ms(self) -> float:
        return self._end_silence_ms

    @end_silence_ms.setter
    def end_silence_ms(self, v: float) -> None:
        self._end_silence_ms = float(max(200.0, min(2500.0, v)))

    def close(self) -> None:
        self._closed = True
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    def reset(self) -> None:
        with self._lock:
            self._clear_turn()
            self._ready.clear()
            self._slices.clear()
            self._carry = np.zeros(0, dtype=np.float32)

    def force_stop(self) -> IsolationResult:
        """Push-to-talk released: judge what was captured so far, right now."""
        with self._lock:
            if self._active and self._frames:
                self._finalize(forced=True)
            else:
                self._clear_turn()
        return IsolationResult(emit_audio=b"", started=False, stopped=False,
                               active=self.active, reason="flush")

    # -- main entry ------------------------------------------------------------
    def process_chunk(self, pcm: bytes, sr: int = SR) -> EarResult:
        if not pcm:
            return EarResult(emit_audio=b"", active=self.active, reason="empty")
        x = pcm16_to_f32(pcm)
        if sr != SR:
            x = resample_float32(x, sr, SR)
        with self._lock:
            self._feed(x)
            return self._next_result()

    # -- framing + VAD ---------------------------------------------------------
    def _feed(self, x: np.ndarray) -> None:
        buf = np.concatenate([self._carry, x]) if self._carry.size else x
        n = buf.size // FRAME
        for k in range(n):
            self._frame(buf[k * FRAME:(k + 1) * FRAME])
        self._carry = buf[n * FRAME:].copy()

    def _band_ratio(self, fr: np.ndarray) -> float:
        spec = np.fft.rfft(fr * self._hann, n=_N_FFT)
        p = spec.real ** 2 + spec.imag ** 2
        tot = float(p.sum()) + 1e-12
        return float(p[self._band].sum() / tot)

    def _media_now(self, fresh: bool = False) -> bool:
        """Is the assistant supposed to stay silent right now? Cached for 250 ms:
        the answer comes from UI state and is asked 50 times a second otherwise."""
        if not self.cfg["media_wake"]:
            return False
        now = time.monotonic()
        if fresh or now - self._media_ts > 0.25:
            try:
                self._media_val = bool(self._require_wake())
            except Exception:
                self._media_val = False
            self._media_ts = now
        return self._media_val

    def _frame(self, fr: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(fr * fr) + 1e-12))
        media = self._media_now()
        self._ambient_hist.append(rms)

        floor = self._floor
        if media and len(self._ambient_hist) >= 25:
            floor = max(floor, float(np.percentile(self._ambient_hist, 40)))
        margin = float(self.cfg["media_margin"] if media else self.cfg["margin"])
        abs_min = float(self.cfg["abs_min_rms"])

        thr_on = max(floor * margin, abs_min)
        thr_hold = max(floor * margin * 0.7, abs_min * 0.8)
        thr = thr_hold if self._active else thr_on
        voiced = rms >= thr and self._band_ratio(fr) >= 0.50
        self._level = rms if voiced else 0.0

        if not self._active:
            # learn the room only from frames that are not speech
            if not voiced:
                if rms < self._floor:
                    self._floor = 0.9 * self._floor + 0.1 * max(rms, 1e-4)
                else:
                    self._floor += (rms - self._floor) * 0.004
                self._floor = float(min(0.05, max(0.0004, self._floor)))
            self._pre_roll.append(fr.copy())
            self._speech_run = self._speech_run + 1 if voiced else 0
            if self._speech_run >= 4:                 # 80 ms of voice-like energy
                self._active = True
                self._frames = list(self._pre_roll)
                self._voiced_ms = self._speech_run * FRAME_MS
                self._silence_ms = 0.0
            return

        self._frames.append(fr.copy())
        if voiced:
            self._voiced_ms += FRAME_MS
            self._silence_ms = 0.0
        else:
            self._silence_ms += FRAME_MS
        total_ms = len(self._frames) * FRAME_MS
        if self._silence_ms >= self._end_silence_ms or total_ms >= self._max_turn_ms:
            self._finalize(forced=False)

    def _clear_turn(self) -> None:
        self._active = False
        self._frames = []
        self._speech_run = 0
        self._silence_ms = 0.0
        self._voiced_ms = 0.0
        self._pre_roll.clear()

    # -- finishing a sentence --------------------------------------------------
    def _finalize(self, forced: bool) -> None:
        frames, voiced_ms, silence_ms = self._frames, self._voiced_ms, self._silence_ms
        self._clear_turn()
        if not frames:
            return
        # keep ~240 ms of the trailing quiet, drop the rest of the wait
        drop = max(0, int((silence_ms - 240.0) / FRAME_MS))
        if drop:
            frames = frames[:-drop] or frames
        if voiced_ms < self._min_voiced_ms:
            self.stats["noise"] += 1
            self.stats["last_reason"] = "too_short"
            return
        audio = np.concatenate(frames).astype(np.float32)
        need_wake = self._media_now(fresh=True)
        utt = _Utterance(audio=audio, voiced_ms=voiced_ms, ts=time.monotonic(),
                         need_wake=need_wake)

        verify = need_wake or self.cfg["speed"] == "accurate"
        long_clip = (audio.size / SR) >= 5.0
        if need_wake and long_clip and time.monotonic() < self._no_verify_until:
            # long, loud, wake-less audio just now = the song itself; do not burn
            # the recogniser on it again straight away
            self.stats["ignored"] += 1
            return
        if not verify or self.transcriber is None and not need_wake:
            self._release(utt, text="", tr=None)
            return
        self._pending_jobs += 1
        self._ensure_worker()
        try:
            self._q.put_nowait(utt)
        except queue.Full:
            self._pending_jobs -= 1
            self._release(utt, text="", tr=None)       # never lose speech to a full queue

    # -- worker ----------------------------------------------------------------
    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="local-ear", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while not self._closed:
            utt = self._q.get()
            if utt is None:
                return
            try:
                self._judge(utt)
            except Exception as e:                     # never let the ear die
                self._log(f"EAR: error judging sentence: {str(e)[:120]}")
                with self._lock:
                    if not utt.need_wake:
                        self._release(utt, text="", tr=None)
            finally:
                with self._lock:
                    self._pending_jobs = max(0, self._pending_jobs - 1)

    def _judge(self, utt: _Utterance) -> None:
        tr = None
        tx = self.transcriber
        if tx is not None and tx.usable():
            tr = tx.transcribe(utt.audio, words=utt.need_wake)
        if tr is None:                                 # no recogniser / it failed
            if utt.need_wake:
                with self._lock:
                    self.stats["ignored"] += 1
                    self.stats["last_reason"] = "no_stt_media"
                return
            with self._lock:
                self._release(utt, text="", tr=None)   # fail open
            return

        dur = utt.audio.size / SR
        why = looks_hallucinated(tr, dur)
        if why:
            with self._lock:
                self.stats["noise"] += 1
                self.stats["last_reason"] = why
                if utt.need_wake and dur >= 5.0:
                    self._no_verify_until = time.monotonic() + 3.0
            return

        text = tr.text.strip()
        n_wake = find_wake(text, self.cfg["wake_words"], self.cfg["name_words"],
                           bool(self.cfg["name_wakes"]))
        if utt.need_wake:
            if n_wake is None:
                with self._lock:
                    self.stats["ignored"] += 1
                    self.stats["last_reason"] = "ignored_media"
                    self.stats["last_text"] = text
                    if dur >= 5.0:
                        self._no_verify_until = time.monotonic() + 3.0
                return
            toks = normalize_text(text).split()
            rest_toks = toks[n_wake:]
            self.stats["woke"] += 1
            self._log(f"EAR: wake word heard: “{text}”")
            try:
                self._on_wake(text)
            except Exception:
                pass
            if not rest_toks:
                with self._lock:
                    self.stats["last_reason"] = "wake_only"
                return
            audio = self._cut_after_wake(utt.audio, tr, n_wake)
            utt = _Utterance(audio=audio, voiced_ms=utt.voiced_ms, ts=utt.ts, need_wake=False)
            text = " ".join(rest_toks)
            # hand Gemini the sentence WITH its wake word in the transcript hint, so
            # the existing "hello" checks in main.py keep working
            text = "hello " + text
        with self._lock:
            self._release(utt, text=text, tr=tr)

    @staticmethod
    def _cut_after_wake(audio: np.ndarray, tr: Transcript, n_wake_tokens: int) -> np.ndarray:
        """Cut the audio right after the wake word using word timestamps."""
        try:
            ws = tr.words
            if ws and len(ws) >= n_wake_tokens:
                end_s = ws[n_wake_tokens - 1][2]
                i = int(max(0.0, end_s - 0.05) * SR)
                if 0 < i < audio.size - SR // 4:
                    return audio[i:]
        except Exception:
            pass
        return audio

    # -- release to Gemini -----------------------------------------------------
    def _release(self, utt: _Utterance, text: str, tr: Optional[Transcript]) -> None:
        y, info = normalize_utterance(
            utt.audio, target_rms=float(self.cfg["target_rms"]),
            max_gain_db=float(self.cfg["max_gain_db"]),
            min_gain_db=float(self.cfg["min_gain_db"]))
        pcm = f32_to_pcm16(y)
        self._ready.append(_Ready(pcm=pcm, text=text, ts=utt.ts, level=float(info.get("level", 0.0))))
        self.stats["sent"] += 1
        self.stats["last_text"] = text
        self.stats["last_reason"] = f"sent (gain {info['gain_db']:+.0f} dB)"
        self._reject_streak = 0
        if text:
            self._log(f"EAR: ✓ “{text[:80]}”  (gain {info['gain_db']:+.0f} dB)")
            try:
                self._on_utt(text)
            except Exception:
                pass

    def _next_result(self) -> EarResult:
        if not self._slices:
            now = time.monotonic()
            while self._ready:
                item = self._ready.popleft()
                if now - item.ts > float(self.cfg["stale_s"]):
                    self.stats["ignored"] += 1
                    self.stats["last_reason"] = "stale"
                    continue
                n = max(1, -(-len(item.pcm) // _SLICE_BYTES))
                for i in range(n):
                    self._slices.append((item.pcm[i * _SLICE_BYTES:(i + 1) * _SLICE_BYTES],
                                         i == 0, i == n - 1, item.text, item.level))
                break
        if self._slices:
            data, first, last, text, level = self._slices.popleft()
            return EarResult(emit_audio=data, started=first, stopped=last, active=True,
                             reason="speech", rms=level, live_level=self._level,
                             text=text)
        return EarResult(emit_audio=b"", active=self._active, reason="idle",
                         rms=self._level, live_level=self._level)
