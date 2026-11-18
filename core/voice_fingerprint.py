"""
Voice-fingerprint registration, speaker isolation, and immediate turn-end
detection for JARVIS.

Design goals
------------
* Zero extra dependencies: runs entirely on ``numpy`` (already in
  ``requirements.txt``) and the Python standard library.
* Real-time safe: each 64 ms audio chunk (1024 samples @ 16 kHz) takes well
  under 1 ms to analyse, isolate, and classify on CPU.
* Strict isolation: only audio frames matching the registered speaker's
  vocal-tract timbre (MFCC prototypes + diagonal distribution), harmonic
  speech structure, and pitch register are forwarded. Background noise,
  non-speech signals (tones, clicks, hums), and other speakers' voices are
  disregarded.
* Immediate thought-mode trigger: the moment the registered voice stops
  speaking (either into silence or superseded by another signal), the isolator
  emits ``stopped=True`` so the caller can close the turn and transition the
  model and HUD into thought mode immediately without waiting for a long
  server-side silence timeout.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

SAMPLE_RATE = 16_000
FRAME_LEN = 400       # 25 ms @ 16 kHz
FRAME_HOP = 160       # 10 ms @ 16 kHz
N_FFT = 512
N_MELS = 32
N_MFCC = 20           # C1..C20 (C0 energy excluded for loudness invariance)
N_PROTOTYPES = 12

DEFAULT_THRESHOLD = 0.52
DEFAULT_END_SILENCE_MS = 480.0
DEFAULT_MAX_TURN_MS = 25_000.0
MIN_ENROLL_VOICED_FRAMES = 40

_F0_MIN_HZ = 65.0
_F0_MAX_HZ = 420.0
_PRE_EMPHASIS = 0.97


# ---------------------------------------------------------------------------
# Precomputed DSP tables (built once at import; pure numpy)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_bank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT,
                    n_mels: int = N_MELS,
                    f_min: float = 80.0, f_max: float = 7600.0) -> np.ndarray:
    f_max = min(f_max, sr * 0.5 - 50.0)
    m_min = _hz_to_mel(f_min)
    m_max = _hz_to_mel(f_max)
    mels = np.linspace(m_min, m_max, n_mels + 2)
    hz = np.array([_mel_to_hz(float(m)) for m in mels], dtype=np.float32)
    bins = np.floor((n_fft + 1) * hz / sr).astype(np.int32)

    n_freqs = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(max(0, left), min(n_freqs, center)):
            fb[m - 1, k] = (k - left) / float(center - left)
        for k in range(max(0, center), min(n_freqs, right)):
            fb[m - 1, k] = (right - k) / float(right - center)
    return fb


def _build_dct_matrix(n_mfcc: int = N_MFCC, n_mels: int = N_MELS) -> np.ndarray:
    """Orthonormal DCT-II rows 1..n_mfcc (skipping row 0 = overall energy)."""
    n = np.arange(n_mels, dtype=np.float32)[None, :]
    k = np.arange(1, n_mfcc + 1, dtype=np.float32)[:, None]
    scale = math.sqrt(2.0 / n_mels)
    return (scale * np.cos(math.pi * k * (2.0 * n + 1.0) / (2.0 * n_mels))).astype(np.float32)


_MEL_BANK = _build_mel_bank()
_DCT_MAT = _build_dct_matrix()
_HAMMING = np.hamming(FRAME_LEN).astype(np.float32)
_FREQS = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE).astype(np.float32)
_SPEECH_BAND_MASK = (_FREQS >= 100.0) & (_FREQS <= 4000.0)
_FLATNESS_MASK = (_FREQS >= 120.0) & (_FREQS <= 4000.0)
_HIGHPASS_MASK = _FREQS >= 80.0
_LAG_MIN = max(1, int(SAMPLE_RATE / _F0_MAX_HZ))
_LAG_MAX = min(FRAME_LEN - 20, int(SAMPLE_RATE / _F0_MIN_HZ))


# ---------------------------------------------------------------------------
# PCM / resampling helpers
# ---------------------------------------------------------------------------

def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    if len(pcm) % 2:
        pcm = pcm[:-1]
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    if samples.size == 0:
        return b""
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def resample_float32(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    n_out = int(round(samples.size * (dst_rate / float(src_rate))))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    idx = np.linspace(0.0, float(samples.size), num=n_out, endpoint=False)
    return np.interp(idx, np.arange(samples.size), samples).astype(np.float32)


# ---------------------------------------------------------------------------
# Frame-level acoustic feature extraction
# ---------------------------------------------------------------------------

@dataclass
class _FrameBatch:
    rms: np.ndarray            # (T,)
    speech_ratio: np.ndarray   # (T,)
    flatness: np.ndarray       # (T,)
    active_bands: np.ndarray   # (T,)
    periodicity: np.ndarray    # (T,)
    f0: np.ndarray             # (T,)
    mfcc: np.ndarray           # (T, N_MFCC)
    mfcc_norm: np.ndarray      # (T, N_MFCC) L2-normalized
    pow_spec: np.ndarray       # (T, N_FFT // 2 + 1)


def _extract_frames(samples: np.ndarray, sr: int = SAMPLE_RATE) -> _FrameBatch:
    x = resample_float32(np.asarray(samples, dtype=np.float32), sr, SAMPLE_RATE)
    if x.size < FRAME_LEN:
        empty = np.zeros(0, dtype=np.float32)
        return _FrameBatch(
            rms=empty, speech_ratio=empty, flatness=empty,
            active_bands=empty, periodicity=empty, f0=empty,
            mfcc=np.zeros((0, N_MFCC), dtype=np.float32),
            mfcc_norm=np.zeros((0, N_MFCC), dtype=np.float32),
            pow_spec=np.zeros((0, N_FFT // 2 + 1), dtype=np.float32),
        )

    n_frames = 1 + (x.size - FRAME_LEN) // FRAME_HOP
    strides = (x.strides[0] * FRAME_HOP, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, FRAME_LEN), strides=strides
    ).copy()

    # Zero-mean raw frames for RMS and autocorrelation pitch
    frames_zm = frames - frames.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(frames_zm * frames_zm, axis=1) + 1e-12).astype(np.float32)

    # Pre-emphasis for spectral / vocal-tract envelope
    emph = np.empty_like(frames_zm)
    emph[:, 0] = frames_zm[:, 0]
    emph[:, 1:] = frames_zm[:, 1:] - _PRE_EMPHASIS * frames_zm[:, :-1]

    spec = np.fft.rfft(emph * _HAMMING[None, :], n=N_FFT, axis=1)
    pow_spec = ((spec.real ** 2 + spec.imag ** 2) / float(N_FFT)).astype(np.float32)

    total_e = pow_spec.sum(axis=1) + 1e-12
    speech_e = pow_spec[:, _SPEECH_BAND_MASK].sum(axis=1)
    speech_ratio = (speech_e / total_e).astype(np.float32)

    flat_band = pow_spec[:, _FLATNESS_MASK] + 1e-12
    gmean = np.exp(np.mean(np.log(flat_band), axis=1))
    amean = np.mean(flat_band, axis=1)
    flatness = np.clip(gmean / (amean + 1e-12), 0.0, 1.0).astype(np.float32)

    mel_e = pow_spec @ _MEL_BANK.T
    mel_peak = mel_e.max(axis=1, keepdims=True) + 1e-12
    active_bands = (mel_e >= (mel_peak * 1e-3)).sum(axis=1).astype(np.float32)

    mfcc = (np.log(mel_e + 1e-9) @ _DCT_MAT.T).astype(np.float32)
    norms = np.linalg.norm(mfcc, axis=1, keepdims=True) + 1e-9
    mfcc_norm = (mfcc / norms).astype(np.float32)

    # Vectorized exact normalized autocorrelation for pitch (F0)
    ac_fft_n = 1024
    f_spec = np.fft.rfft(frames_zm, n=ac_fft_n, axis=1)
    ac_raw = np.fft.irfft(f_spec.real ** 2 + f_spec.imag ** 2, n=ac_fft_n, axis=1)

    sq = frames_zm * frames_zm
    cs = np.concatenate([np.zeros((n_frames, 1), dtype=np.float32),
                         np.cumsum(sq, axis=1)], axis=1)
    total_sq = cs[:, -1:]

    lags = np.arange(_LAG_MIN, _LAG_MAX + 1)
    e_left = cs[:, FRAME_LEN - lags]
    e_right = total_sq - cs[:, lags]
    denom = np.sqrt(np.maximum(e_left * e_right, 1e-12))
    r_lags = ac_raw[:, lags] / denom

    # Octave-halving guard: a true period T0 and its sub-harmonic 2*T0 have
    # nearly identical autocorrelation. Pick the earliest peak >= 0.90 of the
    # global maximum so 130 Hz speech doesn't get misread as 65 Hz.
    max_r = np.max(r_lags, axis=1, keepdims=True)
    strong = r_lags >= (0.90 * max_r)
    first_strong = np.argmax(strong, axis=1)
    # Refine locally within +/- 3 lags around first_strong to hit the local peak
    best_idx = np.empty(n_frames, dtype=np.int32)
    for fi in range(n_frames):
        c0 = int(first_strong[fi])
        lo = max(0, c0 - 2)
        hi = min(r_lags.shape[1], c0 + 5)
        best_idx[fi] = lo + int(np.argmax(r_lags[fi, lo:hi]))

    periodicity = np.clip(r_lags[np.arange(n_frames), best_idx], 0.0, 1.0).astype(np.float32)
    best_lag = lags[best_idx].astype(np.float32)
    f0 = (float(SAMPLE_RATE) / best_lag).astype(np.float32)

    return _FrameBatch(
        rms=rms,
        speech_ratio=speech_ratio,
        flatness=flatness,
        active_bands=active_bands,
        periodicity=periodicity,
        f0=f0,
        mfcc=mfcc,
        mfcc_norm=mfcc_norm,
        pow_spec=pow_spec,
    )


# ---------------------------------------------------------------------------
# K-means clustering for speaker acoustic prototypes (pure numpy)
# ---------------------------------------------------------------------------

def _kmeans_prototypes(vecs: np.ndarray, k: int = N_PROTOTYPES,
                       iters: int = 18) -> np.ndarray:
    """Compute K unit-norm MFCC prototypes via deterministic farthest-first
    initialization + spherical k-means."""
    n = vecs.shape[0]
    if n <= k:
        out = vecs.copy()
        if n < k:
            pad = np.repeat(vecs[:1], k - n, axis=0)
            out = np.vstack([out, pad])
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return (out / norms).astype(np.float32)

    # Deterministic farthest-first seed selection (no randomness -> reproducible)
    centers = np.empty((k, vecs.shape[1]), dtype=np.float32)
    # Start from the vector closest to the global mean
    mean_v = vecs.mean(axis=0)
    mean_v /= (np.linalg.norm(mean_v) + 1e-9)
    sims0 = vecs @ mean_v
    centers[0] = vecs[int(np.argmax(sims0))]
    min_sim = vecs @ centers[0]

    for idx in range(1, k):
        pick = int(np.argmin(min_sim))
        centers[idx] = vecs[pick]
        sim_new = vecs @ centers[idx]
        min_sim = np.maximum(min_sim, sim_new)

    for _ in range(iters):
        assign = np.argmax(vecs @ centers.T, axis=1)
        new_centers = np.empty_like(centers)
        for c in range(k):
            mask = assign == c
            if np.any(mask):
                m = vecs[mask].mean(axis=0)
                new_centers[c] = m / (np.linalg.norm(m) + 1e-9)
            else:
                new_centers[c] = centers[c]
        if np.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers
    return centers.astype(np.float32)


# ---------------------------------------------------------------------------
# VoiceFingerprint model & enrollment
# ---------------------------------------------------------------------------

@dataclass
class VoiceFingerprint:
    """Compact, JSON-serializable acoustic voiceprint of one speaker."""
    mean_mfcc: list[float]
    std_mfcc: list[float]
    prototypes: list[list[float]]
    f0_median: float
    f0_tol_semitones: float
    voiced_frames: int
    duration_s: float
    label: str = "Primary User"
    sample_rate: int = SAMPLE_RATE
    version: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "label": str(self.label),
            "sample_rate": int(self.sample_rate),
            "created_at": str(self.created_at),
            "voiced_frames": int(self.voiced_frames),
            "duration_s": round(float(self.duration_s), 2),
            "f0_median": round(float(self.f0_median), 2),
            "f0_tol_semitones": round(float(self.f0_tol_semitones), 2),
            "mean_mfcc": [round(float(v), 5) for v in self.mean_mfcc],
            "std_mfcc": [round(float(v), 5) for v in self.std_mfcc],
            "prototypes": [
                [round(float(v), 5) for v in row] for row in self.prototypes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceFingerprint":
        if not isinstance(data, dict):
            raise ValueError("VoiceFingerprint payload must be a dict")
        mean = [float(v) for v in data["mean_mfcc"]]
        std = [float(v) for v in data["std_mfcc"]]
        protos = [[float(v) for v in row] for row in data["prototypes"]]
        if len(mean) != N_MFCC or len(std) != N_MFCC or not protos:
            raise ValueError("Invalid VoiceFingerprint dimensions")
        return cls(
            mean_mfcc=mean,
            std_mfcc=std,
            prototypes=protos,
            f0_median=float(data["f0_median"]),
            f0_tol_semitones=float(data.get("f0_tol_semitones", 4.5)),
            voiced_frames=int(data.get("voiced_frames", 0)),
            duration_s=float(data.get("duration_s", 0.0)),
            label=str(data.get("label", "Primary User")),
            sample_rate=int(data.get("sample_rate", SAMPLE_RATE)),
            version=int(data.get("version", 1)),
            created_at=str(data.get("created_at", "")),
        )

    def describe(self) -> str:
        return (
            f"{self.label} ({self.voiced_frames} voiced frames, "
            f"~{self.f0_median:.0f} Hz)"
        )


def enroll_from_samples(samples: np.ndarray, sr: int = SAMPLE_RATE,
                        label: str = "Primary User") -> VoiceFingerprint:
    """Build a VoiceFingerprint from float32 audio samples containing the
    speaker's voice. Raises ValueError if insufficient voiced speech is found."""
    x = resample_float32(np.asarray(samples, dtype=np.float32), sr, SAMPLE_RATE)
    if x.size < SAMPLE_RATE // 2:
        raise ValueError("Audio too short to register a voice fingerprint.")

    # Remove DC offset and sub-65 Hz USB/fan hum so quiet laptop/headset mics
    # do not fail speech-ratio checks from low-frequency electrical rumble.
    x_zm = x - float(np.mean(x))
    peak_raw = float(np.max(np.abs(x_zm))) if x_zm.size else 0.0
    if peak_raw < 0.0008:
        raise ValueError(
            "Microphone captured silence (0% audio level). "
            "Check your mic mute switch or select the active mic in AUDIO DEVICES."
        )
    spec_x = np.fft.rfft(x_zm)
    freqs_x = np.fft.rfftfreq(x_zm.size, 1.0 / float(SAMPLE_RATE))
    spec_x[freqs_x < 65.0] = 0.0
    x_clean = np.fft.irfft(spec_x, n=x_zm.size).astype(np.float32)

    fb = _extract_frames(x_clean, SAMPLE_RATE)
    if fb.rms.size == 0:
        raise ValueError("No audio frames found for enrollment.")

    p10 = float(np.percentile(fb.rms, 10))
    p95 = float(np.percentile(fb.rms, 95))
    rms_thr = max(min(p10 * 2.2, p95 * 0.25), 0.0012)
    f0_ok = (fb.f0 >= _F0_MIN_HZ) & (fb.f0 <= _F0_MAX_HZ)
    voiced_mask = (
        (fb.rms >= rms_thr)
        & (fb.speech_ratio >= 0.36)
        & (fb.flatness <= 0.54)
        & (fb.active_bands >= 4.0)
        & (fb.periodicity >= 0.28)
        & f0_ok
    )
    if int(voiced_mask.sum()) < MIN_ENROLL_VOICED_FRAMES:
        # Fallback pass for quiet USB mics or band-limited (8 kHz Bluetooth)
        # headsets while still strictly rejecting noise and pure tones.
        relaxed_rms = max(min(p10 * 1.6, p95 * 0.16), 0.0006)
        relaxed_mask = (
            (fb.rms >= relaxed_rms)
            & (fb.speech_ratio >= 0.24)
            & (fb.flatness <= 0.62)
            & (fb.active_bands >= 3.0)
            & (fb.periodicity >= 0.20)
            & f0_ok
        )
        if int(relaxed_mask.sum()) > int(voiced_mask.sum()):
            voiced_mask = relaxed_mask

    n_voiced = int(voiced_mask.sum())
    if n_voiced < MIN_ENROLL_VOICED_FRAMES:
        med_per = float(np.median(fb.periodicity)) if fb.periodicity.size else 0.0
        raise ValueError(
            f"Not enough clear speech detected ({n_voiced} voiced frames; "
            f"need at least {MIN_ENROLL_VOICED_FRAMES}, peak={p95:.4f}, "
            f"periodicity={med_per:.2f}). Speak clearly or check AUDIO DEVICES."
        )

    v_mfcc = fb.mfcc[voiced_mask]
    v_norm = fb.mfcc_norm[voiced_mask]
    v_f0 = fb.f0[voiced_mask]

    # Trim extreme pitch outliers (e.g. 5th..95th percentile) for robust pitch stats
    f0_med = float(np.median(v_f0))
    semi = 12.0 * np.log2(np.maximum(v_f0, 1.0) / max(f0_med, 1.0))
    iqr_semi = float(np.percentile(semi, 75) - np.percentile(semi, 25))
    f0_tol = float(max(3.5, min(6.0, 2.0 * max(iqr_semi, 1.5))))

    mean_mfcc = v_mfcc.mean(axis=0)
    std_mfcc = np.maximum(v_mfcc.std(axis=0), 0.55)
    prototypes = _kmeans_prototypes(v_norm, k=N_PROTOTYPES)

    return VoiceFingerprint(
        mean_mfcc=mean_mfcc.tolist(),
        std_mfcc=std_mfcc.tolist(),
        prototypes=prototypes.tolist(),
        f0_median=f0_med,
        f0_tol_semitones=f0_tol,
        voiced_frames=n_voiced,
        duration_s=float(x.size) / float(SAMPLE_RATE),
        label=(label or "Primary User").strip(),
        sample_rate=SAMPLE_RATE,
    )


def enroll_from_pcm(pcm: bytes, sr: int = SAMPLE_RATE,
                    label: str = "Primary User") -> VoiceFingerprint:
    """Build a VoiceFingerprint from signed 16-bit mono PCM bytes."""
    return enroll_from_samples(pcm16_to_float32(pcm), sr=sr, label=label)


def enroll_from_microphone(
    duration_s: float = 4.5,
    device: Any = None,
    label: str = "Primary User",
    on_progress: Optional[Callable[[float, float], None]] = None,
) -> VoiceFingerprint:
    """Capture `duration_s` seconds directly from sounddevice and enroll."""
    import sounddevice as sd

    sr = SAMPLE_RATE
    dev_idx = device
    try:
        info = sd.query_devices(dev_idx, kind="input") if dev_idx is not None else sd.query_devices(kind="input")
        native_sr = int(round(float(info.get("default_samplerate") or SAMPLE_RATE)))
    except Exception:
        native_sr = SAMPLE_RATE

    rates_to_try = [SAMPLE_RATE] if native_sr == SAMPLE_RATE else [SAMPLE_RATE, native_sr]
    chunks: list[np.ndarray] = []
    used_sr = SAMPLE_RATE
    last_err: Exception | None = None

    for try_sr in rates_to_try:
        chunks = []
        collected = 0
        total_samples = int(duration_s * try_sr)
        done = threading.Event()
        blk = max(512, int(round(1024 * try_sr / float(SAMPLE_RATE))))

        def _cb(indata, frames, _time_info, _status):
            nonlocal collected
            mono = indata[:, 0].copy() if getattr(indata, "ndim", 1) > 1 else indata.copy()
            chunks.append(mono)
            collected += len(mono)
            if on_progress is not None:
                try:
                    lvl = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2))) / 32768.0
                    on_progress(min(1.0, collected / float(total_samples)), lvl)
                except Exception:
                    pass
            if collected >= total_samples:
                done.set()

        try:
            with sd.InputStream(
                samplerate=try_sr,
                channels=1,
                dtype="int16",
                blocksize=blk,
                device=dev_idx,
                callback=_cb,
            ):
                done.wait(timeout=duration_s + 2.0)
            used_sr = try_sr
            if chunks:
                break
        except Exception as e:
            last_err = e

    if not chunks:
        if last_err is not None:
            raise last_err
        raise RuntimeError("No audio captured from microphone.")
    total_samples = int(duration_s * used_sr)
    arr = np.concatenate(chunks)[:total_samples]
    return enroll_from_pcm(arr.astype("<i2").tobytes(), sr=used_sr, label=label)


# ---------------------------------------------------------------------------
# Real-time Speaker Verification & Stream Isolation
# ---------------------------------------------------------------------------

@dataclass
class IsolationResult:
    """Result of evaluating one incoming audio chunk."""
    emit_audio: bytes
    started: bool = False
    stopped: bool = False
    active: bool = False
    score: float = 0.0
    disregarded: bool = False
    reason: str = "silence"


class _BaseSpeechGate:
    """Shared streaming VAD / noise-suppression / max-turn safety base."""

    def __init__(
        self,
        end_silence_ms: float = DEFAULT_END_SILENCE_MS,
        max_turn_ms: float = DEFAULT_MAX_TURN_MS,
        denoise: bool = False,
    ) -> None:
        self._lock = threading.Lock()
        self._end_silence_ms = float(max(120.0, min(2000.0, end_silence_ms)))
        self._max_turn_ms = float(max(3000.0, max_turn_ms))
        self._denoise = bool(denoise)

        self._active = False
        self._match_streak = 0
        self._silence_ms = 0.0
        self._foreign_ms = 0.0
        self._unvoiced_ms = 0.0
        self._active_ms = 0.0
        self._noise_floor = 0.0015
        self._noise_psd = np.zeros(N_FFT // 2 + 1, dtype=np.float32)
        self._noise_psd_frames = 0
        self._tail_samples = np.zeros(0, dtype=np.float32)
        self._pre_roll: collections.deque[bytes] = collections.deque(maxlen=6)
        self._ring_unvoiced: collections.deque[bytes] = collections.deque(maxlen=5)
        self._recent_scores: collections.deque[float] = collections.deque(maxlen=4)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def end_silence_ms(self) -> float:
        return self._end_silence_ms

    @end_silence_ms.setter
    def end_silence_ms(self, val: float) -> None:
        with self._lock:
            self._end_silence_ms = float(max(120.0, min(2000.0, val)))

    def _clear_turn_locked(self) -> None:
        self._active = False
        self._match_streak = 0
        self._silence_ms = 0.0
        self._foreign_ms = 0.0
        self._unvoiced_ms = 0.0
        self._active_ms = 0.0
        self._pre_roll.clear()
        self._ring_unvoiced.clear()
        self._recent_scores.clear()

    def reset(self) -> None:
        """Clear utterance state (keep learned room noise profile)."""
        with self._lock:
            self._clear_turn_locked()
            self._tail_samples = np.zeros(0, dtype=np.float32)

    def force_stop(self) -> IsolationResult:
        """If currently active, transition immediately to idle and emit stopped=True."""
        with self._lock:
            was_active = self._active
            self._clear_turn_locked()
            return IsolationResult(
                emit_audio=b"",
                started=False,
                stopped=was_active,
                active=False,
                score=0.0,
                disregarded=False,
                reason="forced_stop" if was_active else "idle",
            )

    def _concat_with_tail_locked(self, samples: np.ndarray) -> np.ndarray:
        if self._tail_samples.size > 0:
            combined = np.concatenate([self._tail_samples, samples])
        else:
            combined = samples
        keep_tail = min(FRAME_LEN - FRAME_HOP, samples.size)
        self._tail_samples = samples[-keep_tail:].copy()
        return combined

    def _isolate_audio(self, samples: np.ndarray) -> np.ndarray:
        """Return clean natural speech (continuous DC-offset removal by default so
        Gemini Live hears un-distorted consonants/formants; optional Wiener denoise
        when explicitly enabled)."""
        if samples.size == 0:
            return samples
        if not self._denoise or samples.size < 256:
            return (samples - float(np.mean(samples))).astype(np.float32)
        if self._noise_psd_frames < 4:
            spec = np.fft.rfft(samples)
            freqs = np.fft.rfftfreq(samples.size, 1.0 / SAMPLE_RATE)
            spec[freqs < 80.0] = 0.0
            return np.fft.irfft(spec, n=samples.size).astype(np.float32)

        n = samples.size
        win = np.hanning(N_FFT).astype(np.float32)
        hop = N_FFT // 2
        pad = N_FFT - (n % hop if n % hop else hop)
        padded = np.pad(samples, (hop, pad + hop))
        out = np.zeros_like(padded)
        wsum = np.zeros_like(padded)

        noise_ref = self._noise_psd
        for pos in range(0, padded.size - N_FFT + 1, hop):
            seg = padded[pos:pos + N_FFT] * win
            spec = np.fft.rfft(seg)
            psd = (spec.real ** 2 + spec.imag ** 2) / float(N_FFT)
            gain = np.maximum(0.20, 1.0 - 1.05 * (noise_ref / (psd + 1e-10)))
            gain[~_HIGHPASS_MASK] = 0.05
            cleaned = np.fft.irfft(spec * gain, n=N_FFT).astype(np.float32)
            out[pos:pos + N_FFT] += cleaned * win
            wsum[pos:pos + N_FFT] += win * win

        valid = wsum > 1e-6
        out[valid] /= wsum[valid]
        return out[hop:hop + n].astype(np.float32)

    def _update_noise_profile(self, fb: _FrameBatch, block_rms: float) -> None:
        if block_rms < self._noise_floor:
            self._noise_floor = 0.80 * self._noise_floor + 0.20 * max(block_rms, 0.0005)
        else:
            self._noise_floor = 0.992 * self._noise_floor + 0.008 * max(block_rms, 0.0005)
        self._noise_floor = max(0.0005, min(0.08, self._noise_floor))

        if fb.pow_spec.shape[0] > 0:
            mean_psd = fb.pow_spec.mean(axis=0)
            if self._noise_psd_frames == 0:
                self._noise_psd = mean_psd.copy()
            else:
                self._noise_psd = 0.92 * self._noise_psd + 0.08 * mean_psd
            self._noise_psd_frames = min(self._noise_psd_frames + 1, 1000)


class SmartSpeechGate(_BaseSpeechGate):
    """
    Speaker-agnostic harmonic speech gate with immediate turn-end detection.

    Used in ``smart_vad`` listening mode (no registered voiceprint needed):
    disregards ambient noise, fans, hums, and tones, streams clean speech, and
    immediately emits ``stopped=True`` once the speaker stops so the model
    transitions straight to THINKING without getting stuck in LISTENING.
    """

    def process_chunk(self, pcm: bytes, sr: int = SAMPLE_RATE) -> IsolationResult:
        if not pcm:
            return IsolationResult(emit_audio=b"", active=self.active, reason="empty")
        raw_samples = pcm16_to_float32(pcm)
        if raw_samples.size == 0:
            return IsolationResult(emit_audio=b"", active=self.active, reason="empty")

        samples = resample_float32(raw_samples, sr, SAMPLE_RATE)
        chunk_ms = (samples.size / float(SAMPLE_RATE)) * 1000.0
        block_rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))

        with self._lock:
            combined = self._concat_with_tail_locked(samples)
            fb = _extract_frames(combined, SAMPLE_RATE)
            if fb.rms.size == 0:
                return IsolationResult(emit_audio=b"", active=self._active, reason="empty")

            e_thr = max(self._noise_floor * 2.0, 0.0011)
            energy_mask = (fb.rms >= e_thr) & (fb.speech_ratio >= 0.30)
            voiced_mask = (
                energy_mask
                & (fb.speech_ratio >= 0.34)
                & (fb.flatness <= 0.52)
                & (fb.active_bands >= 4.0)
                & (fb.periodicity >= 0.25)
            )
            has_voiced = bool(np.any(voiced_mask))
            has_energy = bool(np.any(energy_mask))
            score = float(np.max(fb.periodicity[voiced_mask])) if has_voiced else 0.0

            if not has_energy or (not self._active and not has_voiced):
                if not has_energy or float(fb.flatness.mean()) > 0.45:
                    self._update_noise_profile(fb, block_rms)
                self._match_streak = 0
                if not self._active:
                    self._pre_roll.clear()
                    # Retain brief unvoiced consonant onset right before voiced vowel
                    if has_energy and float(fb.flatness.mean()) <= 0.58:
                        self._ring_unvoiced.append(float32_to_pcm16(self._isolate_audio(samples)))
                    else:
                        self._ring_unvoiced.clear()
                    return IsolationResult(
                        emit_audio=b"",
                        active=False,
                        score=score,
                        disregarded=has_energy,
                        reason="non_speech" if has_energy else "silence",
                    )
                if has_energy and self._unvoiced_ms < 320.0:
                    self._unvoiced_ms += chunk_ms
                    return IsolationResult(
                        emit_audio=float32_to_pcm16(self._isolate_audio(samples)),
                        started=False,
                        stopped=False,
                        active=True,
                        score=score,
                        disregarded=False,
                        reason="unvoiced_cont",
                    )
                self._silence_ms += chunk_ms
                self._active_ms += chunk_ms
                if self._silence_ms >= self._end_silence_ms or self._active_ms >= self._max_turn_ms:
                    reason = "stopped_max_turn" if self._active_ms >= self._max_turn_ms else "stopped_silence"
                    self._clear_turn_locked()
                    return IsolationResult(
                        emit_audio=b"",
                        started=False,
                        stopped=True,
                        active=False,
                        score=0.0,
                        disregarded=False,
                        reason=reason,
                    )
                return IsolationResult(
                    emit_audio=b"",
                    started=False,
                    stopped=False,
                    active=True,
                    score=0.0,
                    disregarded=False,
                    reason="pause_grace",
                )

            self._unvoiced_ms = 0.0
            isolated = float32_to_pcm16(self._isolate_audio(samples))
            if not self._active:
                self._match_streak += 1
                if self._match_streak >= 2:
                    self._active = True
                    self._silence_ms = 0.0
                    self._foreign_ms = 0.0
                    self._unvoiced_ms = 0.0
                    self._active_ms = chunk_ms
                    onset_bytes = (
                        b"".join(self._ring_unvoiced)
                        + b"".join(self._pre_roll)
                        + isolated
                    )
                    self._ring_unvoiced.clear()
                    self._pre_roll.clear()
                    return IsolationResult(
                        emit_audio=onset_bytes,
                        started=True,
                        stopped=False,
                        active=True,
                        score=score,
                        disregarded=False,
                        reason="onset",
                    )
                self._pre_roll.append(isolated)
                return IsolationResult(
                    emit_audio=b"",
                    started=False,
                    stopped=False,
                    active=False,
                    score=score,
                    disregarded=False,
                    reason="pending_onset",
                )

            self._active_ms += chunk_ms
            if self._active_ms >= self._max_turn_ms:
                self._clear_turn_locked()
                return IsolationResult(
                    emit_audio=b"",
                    started=False,
                    stopped=True,
                    active=False,
                    score=score,
                    disregarded=False,
                    reason="stopped_max_turn",
                )
            self._silence_ms = 0.0
            return IsolationResult(
                emit_audio=isolated,
                started=False,
                stopped=False,
                active=True,
                score=score,
                disregarded=False,
                reason="matched",
            )


class VoiceIsolator(_BaseSpeechGate):
    """
    Stateful real-time voice-fingerprint gate and speech isolator.

    Usage:
        isolator = VoiceIsolator(fingerprint)
        res = isolator.process_chunk(pcm_bytes, sr=16000)
        if res.started:
            ... # send activity_start
        if res.emit_audio:
            ... # forward isolated speech audio
        if res.stopped:
            ... # send activity_end & transition immediately to THINKING
    """

    def __init__(
        self,
        fingerprint: VoiceFingerprint,
        threshold: float = DEFAULT_THRESHOLD,
        end_silence_ms: float = DEFAULT_END_SILENCE_MS,
        max_turn_ms: float = DEFAULT_MAX_TURN_MS,
        denoise: bool = False,
    ) -> None:
        super().__init__(
            end_silence_ms=end_silence_ms,
            max_turn_ms=max_turn_ms,
            denoise=denoise,
        )
        self.set_fingerprint(fingerprint)
        self._threshold = float(max(0.30, min(0.90, threshold)))

    def set_fingerprint(self, fp: VoiceFingerprint) -> None:
        with self._lock:
            self._fp = fp
            self._mean = np.asarray(fp.mean_mfcc, dtype=np.float32)
            self._std = np.maximum(np.asarray(fp.std_mfcc, dtype=np.float32), 0.5)
            self._protos = np.asarray(fp.prototypes, dtype=np.float32)
            self._f0_med = max(_F0_MIN_HZ, float(fp.f0_median))
            self._f0_tol = max(3.0, float(fp.f0_tol_semitones))

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, val: float) -> None:
        with self._lock:
            self._threshold = float(max(0.30, min(0.90, val)))

    def score_samples(self, samples: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[float, bool, bool]:
        """Return ``(score, has_voiced, has_speech_energy)`` for an audio segment.

        Public helper so callers and unit tests can inspect raw match scores
        without mutating streaming state.
        """
        fb = _extract_frames(samples, sr)
        with self._lock:
            score, has_voiced, has_energy, _ = self._score_frame_batch(fb, self._noise_floor)
        return score, has_voiced, has_energy

    def matches_pcm(self, pcm: bytes, sr: int = SAMPLE_RATE, margin: float = 0.08) -> bool:
        """Stateless check whether a PCM16 chunk matches the registered speaker."""
        samples = pcm16_to_float32(pcm)
        if samples.size == 0:
            return False
        score, has_voiced, _ = self.score_samples(samples, sr=sr)
        return bool(has_voiced and score >= max(0.42, self._threshold - margin))

    def _score_frame_batch(
        self, fb: _FrameBatch, noise_floor: float
    ) -> tuple[float, bool, bool, np.ndarray]:
        if fb.rms.size == 0:
            return 0.0, False, False, np.zeros(0, dtype=bool)

        e_thr = max(noise_floor * 2.2, 0.0015)
        energy_mask = (fb.rms >= e_thr) & (fb.speech_ratio >= 0.34)
        voiced_mask = (
            energy_mask
            & (fb.speech_ratio >= 0.38)
            & (fb.flatness <= 0.52)
            & (fb.active_bands >= 4.0)
            & (fb.periodicity >= 0.28)
            & (fb.f0 >= _F0_MIN_HZ)
            & (fb.f0 <= _F0_MAX_HZ)
        )
        has_voiced = bool(np.any(voiced_mask))
        has_energy = bool(np.any(energy_mask))

        if not has_energy:
            return 0.0, False, False, voiced_mask

        # Prefer voiced frames when available; fall back to speech-energy frames
        # (for unvoiced consonants mid-utterance).
        use = voiced_mask if has_voiced else energy_mask
        mfcc = fb.mfcc[use]
        mfcc_norm = fb.mfcc_norm[use]

        # 1. Prototype cosine similarity (closest vowel/phoneme cluster)
        proto_sims = np.max(mfcc_norm @ self._protos.T, axis=1)
        # Cosine similarity of same-speaker frames typically sits ~0.60..0.95
        proto_score = float(np.clip((np.mean(proto_sims) - 0.35) / 0.55, 0.0, 1.0))

        # 2. Diagonal standardized distance from global speaker envelope
        z = (mfcc - self._mean[None, :]) / self._std[None, :]
        d = np.sqrt(np.mean(z * z, axis=1))
        gauss_score = float(np.exp(-0.34 * (np.maximum(0.0, np.mean(d) - 0.72) ** 2)))

        # 3. Pitch compatibility (only when voiced frames exist)
        if has_voiced:
            f0_v = fb.f0[voiced_mask]
            semi_diff = np.abs(12.0 * np.log2(np.maximum(f0_v, 1.0) / self._f0_med))
            pitch_scores = np.exp(-0.5 * ((semi_diff / self._f0_tol) ** 2))
            pitch_score = float(np.median(pitch_scores))
            combined = 0.50 * proto_score + 0.30 * gauss_score + 0.20 * pitch_score
            # Hard guard against huge timbre or pitch mismatch
            if pitch_score < 0.22:
                combined *= 0.65
            if proto_score < 0.35:
                combined *= 0.65
        else:
            # Unvoiced consonant: score on timbre only, discounted so noise can't
            # open the gate on its own.
            combined = (0.58 * proto_score + 0.42 * gauss_score) * 0.85

        return float(np.clip(combined, 0.0, 1.0)), has_voiced, has_energy, voiced_mask

    def process_chunk(self, pcm: bytes, sr: int = SAMPLE_RATE) -> IsolationResult:
        """Process one incoming 16-bit mono PCM chunk and return an IsolationResult."""
        if not pcm:
            return IsolationResult(emit_audio=b"", active=self.active, reason="empty")

        raw_samples = pcm16_to_float32(pcm)
        if raw_samples.size == 0:
            return IsolationResult(emit_audio=b"", active=self.active, reason="empty")

        samples = resample_float32(raw_samples, sr, SAMPLE_RATE)
        chunk_ms = (samples.size / float(SAMPLE_RATE)) * 1000.0
        block_rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))

        with self._lock:
            combined = self._concat_with_tail_locked(samples)
            fb = _extract_frames(combined, SAMPLE_RATE)
            score, has_voiced, has_energy, _ = self._score_frame_batch(fb, self._noise_floor)

            enter_thr = max(0.36, self._threshold - 0.06) if block_rms >= max(self._noise_floor * 2.5, 0.004) else self._threshold
            stay_thr = max(0.30, self._threshold - 0.12)
            # Clear non-matching voiced audio (different speaker)
            foreign_thr = max(0.25, self._threshold - 0.14)

            # ── Case 1: Silence / stationary low-energy background ─────────
            if not has_energy:
                self._update_noise_profile(fb, block_rms)
                self._match_streak = 0
                if not self._active:
                    self._pre_roll.clear()
                    return IsolationResult(
                        emit_audio=b"",
                        active=False,
                        score=0.0,
                        disregarded=False,
                        reason="silence",
                    )
                self._silence_ms += chunk_ms
                self._active_ms += chunk_ms
                if self._silence_ms >= self._end_silence_ms or self._active_ms >= self._max_turn_ms:
                    reason = "stopped_max_turn" if self._active_ms >= self._max_turn_ms else "stopped_silence"
                    self._clear_turn_locked()
                    return IsolationResult(
                        emit_audio=b"",
                        started=False,
                        stopped=True,
                        active=False,
                        score=0.0,
                        disregarded=False,
                        reason=reason,
                    )
                # Brief inter-word pause while registered voice is still mid-turn:
                # keep quiet without forwarding background noise, wait for next chunk
                return IsolationResult(
                    emit_audio=b"",
                    started=False,
                    stopped=False,
                    active=True,
                    score=0.0,
                    disregarded=False,
                    reason="pause_grace",
                )

            # ── Case 2: IDLE (waiting for registered voice) ────────────────
            if not self._active:
                if has_voiced and score >= enter_thr:
                    self._match_streak += 1
                    self._recent_scores.append(score)
                    isolated = float32_to_pcm16(self._isolate_audio(samples))
                    # Confirm after 2 consecutive matched blocks or high confidence score
                    if self._match_streak >= 2 or score >= (enter_thr + 0.12):
                        self._active = True
                        self._silence_ms = 0.0
                        self._foreign_ms = 0.0
                        self._unvoiced_ms = 0.0
                        self._active_ms = chunk_ms
                        onset_bytes = (
                            b"".join(self._ring_unvoiced)
                            + b"".join(self._pre_roll)
                            + isolated
                        )
                        self._ring_unvoiced.clear()
                        self._pre_roll.clear()
                        return IsolationResult(
                            emit_audio=onset_bytes,
                            started=True,
                            stopped=False,
                            active=True,
                            score=score,
                            disregarded=False,
                            reason="onset",
                        )
                    # Hold matching onset block in pre-roll until confirmed
                    self._pre_roll.append(isolated)
                    return IsolationResult(
                        emit_audio=b"",
                        started=False,
                        stopped=False,
                        active=False,
                        score=score,
                        disregarded=False,
                        reason="pending_onset",
                    )

                # Non-matching voiced speech or non-speech noise: disregard
                self._match_streak = 0
                self._pre_roll.clear()
                self._recent_scores.clear()
                if not has_voiced and float(fb.flatness.mean()) <= 0.56 and score >= (foreign_thr * 0.85):
                    # Possible unvoiced initial consonant from user right before vowel
                    self._ring_unvoiced.append(float32_to_pcm16(self._isolate_audio(samples)))
                else:
                    self._ring_unvoiced.clear()
                if not has_voiced and float(fb.flatness.mean()) > 0.45:
                    # Broadband non-voiced noise: fold slowly into noise profile
                    self._update_noise_profile(fb, block_rms)

                reason = "foreign_voice" if has_voiced else "non_speech"
                return IsolationResult(
                    emit_audio=b"",
                    started=False,
                    stopped=False,
                    active=False,
                    score=score,
                    disregarded=True,
                    reason=reason,
                )

            # ── Case 3: ACTIVE (registered voice is speaking) ──────────────
            self._active_ms += chunk_ms
            if self._active_ms >= self._max_turn_ms:
                self._clear_turn_locked()
                return IsolationResult(
                    emit_audio=b"",
                    started=False,
                    stopped=True,
                    active=False,
                    score=score,
                    disregarded=False,
                    reason="stopped_max_turn",
                )

            self._recent_scores.append(score)
            avg_score = sum(self._recent_scores) / len(self._recent_scores)

            # Registered voice continues cleanly, including unvoiced Hindi/English
            # consonants ('k', 't', 's', 'sh', 'ch', 'p', 'h') inside words.
            voiced_match = has_voiced and (score >= stay_thr or avg_score >= stay_thr)
            unvoiced_cont = (not has_voiced) and has_energy and (self._unvoiced_ms < 320.0)
            if voiced_match or unvoiced_cont:
                self._silence_ms = 0.0
                self._foreign_ms = 0.0
                if voiced_match:
                    self._unvoiced_ms = 0.0
                else:
                    self._unvoiced_ms += chunk_ms
                isolated = float32_to_pcm16(self._isolate_audio(samples))
                return IsolationResult(
                    emit_audio=isolated,
                    started=False,
                    stopped=False,
                    active=True,
                    score=score,
                    disregarded=False,
                    reason="matched" if voiced_match else "unvoiced_cont",
                )

            # Non-matching signal while registered speaker was talking:
            # Disregard this audio chunk and advance end-of-turn timers.
            self._silence_ms += chunk_ms
            if has_voiced and score < foreign_thr:
                self._foreign_ms += chunk_ms
            else:
                self._foreign_ms = 0.0

            # If another speaker replaced the registered voice, or registered voice
            # has been absent past end_silence_ms, transition immediately!
            foreign_takeover_ms = min(self._end_silence_ms, 260.0)
            if (
                self._silence_ms >= self._end_silence_ms
                or self._foreign_ms >= foreign_takeover_ms
            ):
                stopped_by = (
                    "stopped_foreign"
                    if self._foreign_ms >= foreign_takeover_ms
                    else "stopped_silence"
                )
                self._clear_turn_locked()
                return IsolationResult(
                    emit_audio=b"",
                    started=False,
                    stopped=True,
                    active=False,
                    score=score,
                    disregarded=True,
                    reason=stopped_by,
                )

            return IsolationResult(
                emit_audio=b"",
                started=False,
                stopped=False,
                active=True,
                score=score,
                disregarded=True,
                reason="foreign_voice" if has_voiced else "non_speech",
            )


class AudioFrontend:
    """Stateful real-time 16 kHz mono PCM cleaner & normalizer (SciPy DSP + AGC)."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        highpass_hz: float = 85.0,
        target_rms: float = 0.085,
        max_gain: float = 2.4,
    ) -> None:
        try:
            from scipy import signal as _sp_signal  # type: ignore
            self._sp_signal = _sp_signal
        except Exception:
            self._sp_signal = None

        self._sr = max(8000, int(sample_rate))
        self._target_rms = float(target_rms)
        self._max_gain = float(max(1.0, max_gain))
        self._lock = threading.Lock()
        self._gain = 1.0
        self._noise_rms = 0.0025

        if self._sp_signal is not None and 20.0 < highpass_hz < (self._sr * 0.4):
            self._sos = self._sp_signal.butter(
                4, highpass_hz, btype="highpass", fs=self._sr, output="sos"
            )
            self._zi = self._sp_signal.sosfilt_zi(self._sos) * 0.0
        else:
            self._sos = None
            self._zi = None
        self._prev_x = 0.0
        self._prev_y = 0.0

    def reset(self) -> None:
        with self._lock:
            self._gain = 1.0
            self._noise_rms = 0.0025
            self._prev_x = 0.0
            self._prev_y = 0.0
            if self._sos is not None and self._sp_signal is not None:
                self._zi = self._sp_signal.sosfilt_zi(self._sos) * 0.0

    def process_pcm(self, pcm: bytes) -> bytes:
        if not pcm:
            return pcm
        arr = np.frombuffer(pcm, dtype=np.int16)
        if arr.size == 0:
            return pcm
        y = self.process_float32(arr.astype(np.float32) / 32768.0)
        return np.clip(np.round(y * 32767.0), -32768.0, 32767.0).astype("<i2").tobytes()

    def process_float32(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return samples
        x = np.asarray(samples, dtype=np.float32)
        with self._lock:
            x = x - float(np.mean(x))
            if self._sos is not None and self._sp_signal is not None:
                y, self._zi = self._sp_signal.sosfilt(self._sos, x, zi=self._zi)
                y = y.astype(np.float32, copy=False)
            else:
                out = np.empty_like(x)
                px, py = self._prev_x, self._prev_y
                for i in range(x.size):
                    cur = float(x[i])
                    py = 0.965 * (py + cur - px)
                    px = cur
                    out[i] = py
                self._prev_x, self._prev_y = px, py
                y = out

            rms = float(np.sqrt(np.mean(y * y) + 1e-12))
            if rms < self._noise_rms * 1.4:
                self._noise_rms = 0.94 * self._noise_rms + 0.06 * rms
            elif rms < 0.02:
                self._noise_rms = 0.985 * self._noise_rms + 0.015 * rms

            if rms >= max(self._noise_rms * 3.0, 0.010):
                desired = min(self._max_gain, max(1.0, self._target_rms / max(rms, 1e-4)))
                self._gain += (desired - self._gain) * (0.20 if desired < self._gain else 0.10)
            else:
                self._gain += (1.0 - self._gain) * 0.15
                if rms < max(self._noise_rms * 2.0, 0.006):
                    y = y * 0.35

            scaled = y * self._gain
            if float(np.max(np.abs(scaled))) > 0.88:
                scaled = np.tanh(scaled * 1.08) * 0.92
            return scaled.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# CLI utility for quick enrollment / verification testing
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    from memory.config_manager import (
        clear_voice_fingerprint,
        get_voice_fingerprint,
        get_voice_isolation_enabled,
        save_voice_fingerprint,
        save_voice_isolation_enabled,
    )

    p = argparse.ArgumentParser(description="Manage JARVIS registered voice fingerprint.")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll", help="Record from microphone and save voice fingerprint")
    e.add_argument("--seconds", type=float, default=4.5)
    e.add_argument("--label", type=str, default="Primary User")
    sub.add_parser("status", help="Show registered voice fingerprint status")
    sub.add_parser("clear", help="Delete saved voice fingerprint")
    args = p.parse_args()

    if args.cmd == "enroll":
        print(f"Recording for {args.seconds:.1f}s — speak normally...")
        fp = enroll_from_microphone(duration_s=args.seconds, label=args.label)
        save_voice_fingerprint(fp.to_dict())
        save_voice_isolation_enabled(True)
        print(f"Registered and enabled: {fp.describe()}")
    elif args.cmd == "status":
        raw = get_voice_fingerprint()
        en = get_voice_isolation_enabled()
        if not raw:
            print(f"No voice fingerprint registered (isolation enabled={en}).")
        else:
            fp = VoiceFingerprint.from_dict(raw)
            print(f"Registered: {fp.describe()} | isolation enabled={en}")
    elif args.cmd == "clear":
        clear_voice_fingerprint()
        save_voice_isolation_enabled(False)
        print("Voice fingerprint cleared.")


if __name__ == "__main__":
    _cli()
