import platform as _platform
import subprocess as _subprocess

# ── Nuclear: force CREATE_NO_WINDOW on EVERY subprocess call on Windows ───────
# This patches Popen itself, so no per-file flag is needed anywhere.
if _platform.system() == "Windows":
    _OrigPopen = _subprocess.Popen

    class _Popen(_OrigPopen):
        def __init__(self, args, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _subprocess.CREATE_NO_WINDOW
            kw.pop("startupinfo", None)   # drop any stale/shared STARTUPINFO
            super().__init__(args, **                       kw)

    _subprocess.Popen = _Popen


# ── Console must survive non-UTF-8 code pages ────────────────────────────────
# Every status line in this file carries an emoji, and on a legacy Windows
# console the active code page is the system one — cp1254 in Turkey, cp1251 in
# Russia, cp932 in Japan. Printing an emoji there raises UnicodeEncodeError, and
# because most of these prints sit inside the receive loop it takes the session
# down on startup. Reconfiguring to UTF-8 with a replacement fallback costs
# nothing and makes the app launch the same way in every locale.
import sys as _sys

for _stream in ("stdout", "stderr"):
    try:
        _s = getattr(_sys, _stream, None)
        if _s is not None and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass          # pythonw / redirected pipes / anything exotic — never fatal

# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import re
import threading
import time
import json
import sys
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import sounddevice as sd
except (OSError, ImportError) as _sd_err:
    sd = None
    print(f"[Warning] sounddevice not available ({_sd_err}). Running in headless/silent mode.")
import numpy as np
from google import genai
from google.genai import types
from ui import JarvisUI
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
    save_session_summary, pop_last_session,
    search_memory, set_trim_notifier,
)

# The file-backed tools (open_app, web_search, browser_control, …) are no longer
# imported or declared here — they self-describe via a TOOL dict in their own
# actions/*.py file and are auto-discovered by core.action_loader at startup.
# Only tools that are tied to live-session state stay inline in this file
# (screen_process, close_camera, save_memory, manage_monitor, shutdown_jarvis,
# system_status).
from core import outcome
from core.action_loader import strip_meta
from actions.screen_processor  import _capture_camera, _capture_screen
from actions.system_monitor    import SystemMonitor, get_system_status
from actions.proactive         import ProactiveEngine
from actions.background_monitor import (
    add_monitor, remove_monitor, list_monitors, check_all as monitor_check_all,
)
from actions.web_search        import _news as _fetch_news_sync
from memory.config_manager     import (
    get_brief_enabled, get_media_resolution, get_proactive_audio_enabled,
    get_push_to_talk_enabled, get_thinking_enabled, get_turn_tuning, get_voice,
    get_wake_word_enabled, save_wake_word_enabled,    get_input_device, get_output_device,
    get_voice_isolation_enabled, save_voice_isolation_enabled,
    get_voice_isolation_threshold, get_voice_end_silence_ms,
    get_voice_fingerprint, save_voice_fingerprint, clear_voice_fingerprint,
    is_voice_isolation_active, get_listening_mode, save_listening_mode,
    LISTENING_MODES, get_reply_language, get_voice_barge_in_enabled,
    get_screen_watch_enabled, get_companion_chat_enabled,
    get_topic_lock_enabled, get_auto_learn_enabled,
)
from memory                    import local_db, firebase_sync
from core                     import gemini as _gemini
from core.plugin_loader        import discover_plugins
from core                      import undo as undo_stack
from core                      import confirm as confirm_gate
from core                      import audio_devices
from core.action_loader        import discover_actions
from core.echo                 import EchoGuard
from core.viseme               import VisemeStream
from core.voice_fingerprint    import (
    AudioFrontend, SmartSpeechGate, VoiceFingerprint, VoiceIsolator,
    enroll_from_pcm, enroll_from_microphone,
)
from core.wake_word            import (
    WakeWordDetector, is_ready as wake_is_ready, install_and_download as wake_install,
)
from core.local_ear            import ClearVoiceFrontend, LocalEarGate, build_transcriber
from memory.config_manager     import get_local_ear_config

# How long the assistant stays awake with no user speech before it auto-sleeps
# again (wake-word mode only).
WAKE_SLEEP_TIMEOUT = 120.0   # seconds (2 minutes)

def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

BASE_DIR        = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
# The conversation's model. A NAME, not a decision: the ladder lives in
# core/gemini.py and this is only whichever rung is currently in use, kept here
# as a module attribute because plugins read it (chat_takeover asks main for it
# so that upgrading the assistant upgrades the plugin too).
#
# It is reassigned on every connect, so a model that runs out of quota is
# stepped over and the assistant keeps talking instead of failing to start.
LIVE_MODEL          = "models/gemini-3.1-flash-live-preview"
CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000 
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024

# RMS below which 16-bit PCM is treated as room silence; above _LEVEL_FULL it
# reads as a full-height waveform. Tuned so ordinary speech lands mid-range and
# the bars still move for a quiet talker — language- and device-independent.
_LEVEL_FLOOR = 60.0
_LEVEL_FULL  = 2600.0


def _pcm_level(samples) -> float:
    """Map a block of int16 PCM samples to a 0.0–1.0 loudness level for the HUD
    waveform. Returns 0.0 on empty/invalid input so it can never raise."""
    try:
        x = np.asarray(samples, dtype=np.float32)
        if x.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(x * x)))
    except Exception:
        return 0.0
    if rms <= _LEVEL_FLOOR:
        return 0.0
    return min(1.0, (rms - _LEVEL_FLOOR) / (_LEVEL_FULL - _LEVEL_FLOOR))


# ── Viseme extraction ─────────────────────────────────────────────────────────
# The avatar's mouth used to be driven by one RMS value per ~200 ms write batch,
# which is five updates a second averaged over a fifth of a second — it could
# only ever flap. These read the *shape* of each 20 ms slice straight from the
# spectrum of the audio being played, so no transcript, no forced alignment and
# no language assumption: it works the same for Turkish and English.
#
# Two numbers come out. Openness tracks the first formant — F1 climbs as the jaw
# drops, so /a/ reads open and /i/ or /u/ read closed. Width tracks the second —
# F2 is high for spread vowels (/i/, /e/) and low for rounded ones (/u/, /o/).
# Extra time beyond the device's reported output latency before the microphone
# is trusted again: covers room decay and the speaker's own settling.
_TAIL_MARGIN = 0.25

# How long a sent turn may go unanswered before the session is presumed dead.
#
# A Live session can go half-dead: the server stops answering but the socket is
# never closed, so send_client_content() still succeeds — it is a write into a
# buffer — and `session.receive()` simply awaits forever. Nothing raises. The
# user's message reaches the activity log, the HUD stays on LISTENING, the
# terminal prints nothing, and the assistant is gone until it is restarted.
# That was the single worst failure this app could have, because it looked
# exactly like the model choosing not to reply.
#
# The threshold is turn-scoped rather than idle-scoped, and measured rather than
# guessed. An idle session receives NOTHING: watched for 150 s after a reply,
# not one frame arrived unprompted, so "silence" cannot mean "dead". But an
# ANSWERED turn is fast — first audio at 1.2 s, turn_complete at 3.6 s. Thirty
# seconds is twenty-five times the measured first response, which leaves room
# for a slow tool call (those send their own traffic, which clears the deadline)
# while turning an unbounded hang into a blip.
_STALL_LIMIT = 30.0

_VIS_WIN = 1024        # ~43 ms analysis window at 24 kHz: enough for formants
_VIS_HOP = 480         # 20 ms between frames, i.e. 50 shapes a second

# Delay from handing the first bytes of a reply to an already-running output
# stream to hearing them: one callback period, plus whatever the DAC adds.
_FIRST_SOUND = CHUNK_SIZE / RECEIVE_SAMPLE_RATE      # ~43 ms
# How far past the device's own buffer the mouth's timeline may drift before it
# is re-anchored. The buffer is the hard limit on how much audio can be queued
# ahead, so anything beyond it plus a margin for clock error is impossible.
_CURSOR_SLACK = 0.15

# Erring early is the safe direction. A viewer tolerates a mouth that moves
# slightly before the sound far better than one that moves after it — the
# broadcast limits are about 45 ms of lag against 125 ms of lead — so where
# this is uncertain it is biased to lead.


def _pcm_visemes(samples, sr: int = 24000):
    """Slice a PCM block into (level, openness, width) frames, one per 20 ms.

    Returns [] on anything unexpected — the mouth falls back to loudness-only
    articulation rather than the caller having to handle an error.
    """
    try:
        x = np.asarray(samples, dtype=np.float32)
        if x.size < _VIS_WIN:
            return []
        win = np.hanning(_VIS_WIN).astype(np.float32)
        freqs = np.fft.rfftfreq(_VIS_WIN, 1.0 / sr)
        b_f1_lo = (freqs >= 150) & (freqs < 450)     # F1 of close vowels
        b_f1_hi = (freqs >= 450) & (freqs < 1100)    # F1 of open vowels
        b_f2_bk = (freqs >= 600) & (freqs < 1300)    # F2 of rounded vowels
        b_f2_fr = (freqs >= 1700) & (freqs < 3200)   # F2 of spread vowels
        b_hiss = (freqs >= 3800) & (freqs < 8000)    # fricatives

        # One frame per hop across the *whole* block. Stepping only while a full
        # window fits stopped 1024 - 480 samples short of the end, so a 200 ms
        # batch yielded 160 ms of schedule: the mouth ran out of frames before
        # the audio ran out of sound, and each batch no longer lined up with the
        # end of the one before it. Losing 20 % of every batch is most of why
        # the mouth did not track the words.
        out = []
        for start in range(0, x.size, _VIS_HOP):
            # The level gates closures, so it is measured over exactly this
            # 20 ms and never looks ahead. The spectrum needs a longer window
            # to resolve formants and may be short-filled at the very end.
            level = _pcm_level(x[start:start + _VIS_HOP])
            seg = x[start:start + _VIS_WIN]
            if seg.size < _VIS_WIN:
                seg = np.concatenate([seg, np.zeros(_VIS_WIN - seg.size,
                                                    dtype=np.float32)])
            if level <= 0.0:
                out.append((0.0, 0.0, 0.0))
                continue
            mag = np.abs(np.fft.rfft((seg - seg.mean()) * win))
            f1l, f1h = float(mag[b_f1_lo].sum()), float(mag[b_f1_hi].sum())
            f2b, f2f = float(mag[b_f2_bk].sum()), float(mag[b_f2_fr].sum())
            hiss = float(mag[b_hiss].sum())

            openness = f1h / (f1l + f1h + 1e-6)
            width = (f2f - f2b) / (f2f + f2b + 1e-6)
            # A wide-open jaw physically cannot purse, so openness damps width.
            # /a/ has a low enough F2 to read as "rounded" on the bands alone;
            # letting openness suppress the width term is what keeps an open
            # vowel from pursing.
            width *= (1.0 - openness) ** 0.8
            # Fricatives are formed with a nearly closed mouth.
            h = hiss / (f1l + f1h + f2b + f2f + hiss + 1e-6)
            openness *= 1.0 - 0.65 * min(1.0, h * 2.5)
            out.append((level,
                        float(min(1.0, max(0.0, openness))),
                        float(min(1.0, max(-1.0, width)))))
        return out
    except Exception:
        return []


def _describe_tools(declarations) -> str:
    """One line per capability, straight from the live tool declarations.

    Derived rather than written down: the action and plugin registries are
    discovered at startup, so whatever the user has installed is what the model
    is told it can do. Adding a plugin extends this by itself, and removing one
    stops the model from claiming an ability it no longer has.
    """
    lines = []
    for d in declarations or ():
        try:
            name = d.get("name") if isinstance(d, dict) else getattr(d, "name", None)
            desc = (d.get("description") if isinstance(d, dict)
                    else getattr(d, "description", "")) or ""
        except Exception:
            continue
        if not name:
            continue
        desc = " ".join(str(desc).split())
        lines.append(f"- {name}: {desc[:150]}" if desc else f"- {name}")
    return "\n".join(lines)


def _describe_limits(has_vision: bool, has_mic: bool) -> str:
    """The other half of self-knowledge: what is out of reach, and why.

    Derived from how the program is actually built, not from a list of refusals.
    A model that knows its boundaries stops improvising around them, and stating
    them as architecture rather than as rules keeps the answer honest in any
    language.
    """
    out = [
        "- Anything not listed above is outside your reach. Say so in one clause "
        "and offer the nearest thing you can actually do — never mime an action "
        "you cannot take, and never report a result you did not get.",
        "- You act on this machine only. You cannot reach the user's other "
        "devices, accounts or hardware except through the tools listed above.",
        "- You remember what is in the memory block and what has been said this "
        "session. Anything else you were told before is gone unless it was saved.",
    ]
    # What the PLATFORM forbids, before what the build lacks.
    #
    # On Wayland, and on a macOS without Accessibility, the keyboard, mouse and
    # screen are simply not available to any program — and the APIs report
    # success anyway. A model that does not know this reports clicks that never
    # happened, which is how "it does not work properly on Linux" was described
    # for a long time without anybody being able to name the failing part.
    try:
        from core import platform_support
        out.extend(platform_support.describe_for_model(platform_support.detect()))
    except Exception as _e:
        print(f"[Limits] platform capabilities unavailable: {_e}")

    if has_vision:
        out.append(
            "- Your sight is not continuous. You see nothing until you call a "
            "vision tool, and then only that single frame at that moment — you "
            "cannot watch, monitor or notice something changing on screen.")
    else:
        out.append("- You have no sight at all in this build.")
    if has_mic:
        out.append(
            "- You hear nothing while the microphone is muted, and you cannot "
            "unmute it yourself.")
    return "\n".join(out)


def _render_prompt(template: str, values: dict) -> str:
    """Fill {tokens} in the prompt template.

    A plain replace rather than str.format: the file is meant to be edited by
    hand, and a stray brace in someone's own wording must never take the app
    down at startup.
    """
    out = template or ""
    for key, val in values.items():
        out = out.replace("{" + key + "}", str(val))
    return out


def _get_api_key() -> str:
    """The Gemini key, or "" when none is saved — a local-model-only setup has none by design,
    and that must not be a KeyError."""
    try:
        with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
            return str(json.load(f).get("gemini_api_key") or "")
    except Exception:
        return ""


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )

_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)

# Transcript chunks shorter than this may legitimately repeat ("evet, evet"),
# so only longer ones are treated as duplicates.
_REPEAT_MIN = 12


def _is_repeat_chunk(txt: str, buf: list) -> bool:
    """True if this transcript chunk has already been seen this turn.

    Guards against the API re-sending the tail of a response across the several
    turn_completes a tool-using turn produces.
    """
    if len(txt) < _REPEAT_MIN:
        return bool(buf) and txt == buf[-1]
    joined = " ".join(buf)
    return txt in joined

def _clean_transcript(text: str) -> str:    
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()

TOOL_DECLARATIONS = [
    # ── Inline tools ─────────────────────────────────────────────────────────
    # These stay here (rather than in an actions/*.py TOOL dict) because their
    # handling is woven into live-session state — vision capture/injection,
    # camera stream, memory writes, the monitor engine, and shutdown. All other
    # tools live in their own action file and are auto-discovered by
    # core.action_loader (see JarvisLive.__init__).
    {
        "name": "system_status",
        "description": (
            "Returns real-time system metrics: CPU usage, RAM, GPU load, CPU temperature, "
            "uptime, and process count. Use when the user asks about computer performance, "
            "temperature, memory, or resource usage."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures the screen or webcam image and lets you analyze it. "
            "MUST be called when user asks what is on screen, what you see, "
            "look at camera, analyze my screen, etc. "
            "You have NO visual ability without this tool. "
            "After the image is captured it is sent directly to you — describe what you see and answer the user's question. "
            "When using camera: the live view stays open until user says close it or calls close_camera."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "'screen' to capture display, 'camera' for webcam. Default: 'screen'"},
                "text":  {"type": "STRING", "description": "The question or instruction about the captured image"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "close_camera",
        "description": (
            "Closes the live camera view shown on screen. "
            "Call when the user says (in ANY language): close camera, stop camera, "
            "turn off camera, that's creepy, etc."
        ),
        "parameters": {"type": "OBJECT", "properties": {}, "required": []}
    },
    {
        "name": "manage_monitor",
        "description": (
            "Add, remove, or list background monitoring topics. "
            "JARVIS checks these topics once a day and alerts the user when there is a new development. "
            "Use 'add' when the user says 'monitor X', 'track X', 'follow X'. "
            "Use 'remove' when the user says 'stop monitoring X'. "
            "Use 'list' when the user asks what is being monitored. "
            "Do NOT add crypto, financial, or trading topics."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type":        "STRING",
                    "description": "add | remove | list",
                },
                "topic": {
                    "type":        "STRING",
                    "description": "Topic to monitor or stop monitoring (e.g. 'space exploration', 'AI news')",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "shutdown_jarvis",
        "description": (
            "Shuts down the assistant completely. "
            "Call this when the user expresses intent to end the conversation, "
            "close the assistant, say goodbye, or stop Jarvis. "
            "The user can say this in ANY language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "save_memory",
        "description": (
            "Save an important personal fact about the user to long-term memory. "
            "Call this silently whenever the user reveals something worth remembering: "
            "name, age, city, job, preferences, hobbies, relationships, projects, or future plans. "
            "Do NOT call for: weather, reminders, searches, or one-time commands. "
            "Do NOT announce that you are saving — just call it silently. "
            "Values must be in English regardless of the conversation language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": (
                        "identity — name, age, birthday, city, job, language, nationality | "
                        "preferences — favorite food/color/music/film/game/sport, hobbies | "
                        "projects — active projects, goals, things being built | "
                        "relationships — friends, family, partner, colleagues | "
                        "wishes — future plans, things to buy, travel dreams | "
                        "notes — habits, schedule, anything else worth remembering"
                    )
                },
                "key":   {"type": "STRING", "description": "Short snake_case key (e.g. name, favorite_food, sister_name)"},
                "value": {"type": "STRING", "description": "Concise value in English (e.g. Fatih, pizza, older sister)"},
            },
            "required": ["category", "key", "value"]
        }
    },
    {
        "name": "recall_memory",
        "description": (
            "Look up a fact you have stored about the user but which is NOT in "
            "the memory block of your system prompt. "
            "The prompt lists the keys it did not have room for under "
            "'[ALSO REMEMBERED]' — if the user asks about anything named there, "
            "call this FIRST. "
            "Also call it before saying you do not know something personal, and "
            "when the user asks what you remember about them (leave query empty "
            "for everything). "
            "This is a local file search: it is instant and costs nothing."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": (
                        "Keyword to search for — a name, a topic, a category "
                        "(e.g. 'ayse', 'coffee', 'projects'). "
                        "Leave empty to list everything stored."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "undo",
        "description": (
            "Reverse the last change YOU made to this computer — a file you "
            "moved, renamed, created or wrote, or a setting you changed such as "
            "volume, brightness, dark mode or WiFi. "
            "Call this whenever the user says undo, revert, take it back, put it "
            "back, cancel that, or tells you that you did the wrong thing, in ANY "
            "language. "
            "Use action='list' when they ask what can be undone. "
            "This only covers your own actions — it is not the Ctrl+Z of whatever "
            "application is on screen (that is computer_settings with action 'undo')."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "undo (default) — reverse the last change | list — show what can be undone",
                },
            },
            "required": [],
        },
    },
]

class _ReconnectSignal(Exception):
    """Raised inside the session TaskGroup to force a clean, voluntary reconnect
    (e.g. the user picked a new voice — the voice is fixed at connect time, so
    the session must be rebuilt).

    Carries `keep_context`: True for an ordinary rebuild, where the stored
    resumption handle is replayed and the conversation continues; False when the
    new session must genuinely start clean (see the voice-change note in
    _on_voice_change)."""

    def __init__(self, keep_context: bool = True):
        super().__init__()
        self.keep_context = keep_context


def _is_reconnect_signal(exc: BaseException) -> bool:
    """True if `exc` is a _ReconnectSignal, or a(n) (Base)ExceptionGroup that
    wraps one — TaskGroup bundles child exceptions into a group."""
    if isinstance(exc, _ReconnectSignal):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_reconnect_signal(sub) for sub in exc.exceptions)
    return False


def _keep_context_of(exc: BaseException) -> bool:
    """Read `keep_context` off a reconnect signal, unwrapping the group the
    TaskGroup put it in. Defaults to True: an unexpected shape must not silently
    wipe the conversation."""
    if isinstance(exc, _ReconnectSignal):
        return getattr(exc, "keep_context", True)
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            if _is_reconnect_signal(sub):
                return _keep_context_of(sub)
    return True


class JarvisLive:
    def __init__(self, ui: JarvisUI):
        # Platform limits are announced once per run, not once per
        # reconnect: a dropped connection is not news about Wayland.
        self._platform_notes_shown = False
        self.ui             = ui
        self._asst_name     = "JARVI    S"   # updated each session from config
        self.session              = None
        self.audio_in_queue       = None
        self.out_queue            = None
        self._loop                     = None
        self._is_speaking         = False
        self._speaking_lock       = threading.Lock()
        self._phone_active        = False   # True while phone mic is streaming; pauses PC mic
        self._pending_vision       = None    # (img_bytes, mime_type, question, angle) to inject after tool response
        self._vision_cam_active    = False   # True if camera was opened for vision → auto-close after response
        self._vision_close_pending = False   # True after vision injected; next turn_complete closes camera
        self._vision_last_time     = 0.0     # monotonic time of last screen_process call (cooldown guard)
        self._vision_busy          = False   # True while a vision capture/inject cycle is in flight
        self._interrupted          = False   # True while draining audio after user interrupt
        self._server_turn_active   = False   # True while server response audio is in flight
        self._barge_in_streak      = 0
        self._barge_in_enabled     = get_voice_barge_in_enabled()
        # Which engine is holding the conversation, and what it can do.
        # None means Gemini Live — the default and, unless somebody has gone to
        # the providers settings and chosen otherwise, always the case. Every
        # branch below that reads it treats None as "everything is available",
        # so the Live path is exactly what it was.
        self._engine_caps          = None
        # Set only when a local pipeline is running, so push-to-talk can tell it
        # the turn is over. Live decides that on the server; nothing does here.
        self._engine               = None
        # Deadline for "we sent a turn and the server has said nothing since".
        # None means nothing is outstanding. Armed by the send sites, cleared by
        # the first frame of ANY response, watched by _watch_stall().
        self._rx_deadline          = None
        # Transcript-driven mouth shapes for the avatar. Fed from the receive
        # loop as words arrive, drained by the playback loop against the audio.
        self._visemes              = VisemeStream()
        self._last_out_logged      = ""      # de-dupes a re-sent transcript tail
        # Push-to-talk
        self._ptt_enabled          = False
        self._ptt_held             = False
        self._ptt                  = None    # core.hotkey.PushToTalk
        self._out_level            = 0.0     # level of the audio being played right now
        self._echo                 = EchoGuard()
        # `stream.write()` returns when the buffer accepts the audio, not when the
        # speaker has finished with it, so sound is still in the room after the
        # speaking flag drops. Streaming the microphone during that gap is how an
        # assistant ends up answering itself. Measured from the device rather than
        # guessed; see _play_audio.
        self._out_latency          = 0.20    # seconds, replaced with the real value
        self._tail_until           = 0.0     # monotonic time the echo tail expires
        # Wall-clock time at which the audio written next will begin to sound.
        # The mouth is scheduled against this, never against "now": batches are
        # handed to the device far faster than they play, so "now" ran the lips
        # ahead of the words and cut every schedule short. 0 = nothing playing.
        self._play_cursor          = 0.0
        self.ui.on_push_to_talk   = self.set_push_to_talk
        self.ui.ptt_hold          = self._on_ptt
        self.ui.on_text_command   = self._on_text_command
        self.ui.on_image_uploaded = self._on_image_uploaded
        # [REMOTE CONTROL DISABLED — replaced by Phone Link] 
        # self.ui.on_remote_clicked = self._make_remote_key
        self.ui.on_phone_link_clicked = self._make_phone_link
        self.ui.on_interrupt      = self.interrupt
        self.ui.on_voice_change   = self._on_voice_change     # voice picker → rebuild session
        self.ui.on_audio_device_change = self._on_audio_device_change
        self._reconnect_event: asyncio.Event | None = None
        self._reconnect_keep = True   # False → next rebuild drops the resumption handle

        # ── Session resumption ─────────────────────────────────────────
        # The server issues a resumption handle every few seconds and reissues
        # it as the conversation moves on. Before this, session_resumption was
        # switched ON in the config and the update was never read, so the handle
        # was thrown away and EVERY reconnect — a dropped packet, a voice change,
        # switching microphone — started an empty session. "Unlimited sessions"
        # leaked through exactly this hole.
        #
        # Deliberately in RAM only, never written to disk. Persisting it would
        # make a fresh launch continue yesterday's conversation, which sounds
        # appealing but breaks the session-summary flow: _save_session_summary
        # runs at shutdown and the morning briefing pops it the next day. A
        # conversation that never ends never produces a summary, and the
        # "yesterday we talked about…" line silently disappears.
        self._resume_handle: str | None = None
        self._turn_done_event: asyncio.Event | None = None
        self._dashboard     = None
        self._briefing_sent    = False          # morning briefing fires once per process
        self._sys_monitor      = SystemMonitor()  # persistent cooldown state
        self._proactive        = ProactiveEngine()
        self._last_user_speech = time.monotonic()  # updated on every user utterance
        self._last_assistant_speech = 0.0
        self._screen_watch_enabled  = get_screen_watch_enabled()
        self._companion_chat_enabled = get_companion_chat_enabled()
        self._last_screen_thumb = None
        self._last_screen_sent  = 0.0
        self._latest_screen_jpeg: bytes | None = None
        self._screen_changed_at = 0.0
        self._screen_push_lock: asyncio.Lock | None = None
        self._private_mode      = False
        self._sleep_standby     = False
        self._last_user_utterance = ""
        self._ear_transcriber   = None        # local Whisper, loaded once, shared by gates
        self._audio_frontend    = self._make_frontend()
        self._last_emotion      = "neutral"
        self._last_user_pitch_hz = 0.0
        self._last_user_rms      = 0.0
        self._session_log: list[str] = []          # conversation turns for end-of-session summary

        self._enhanced_live = True  # proactive audio; auto-disabled if the server rejects it
        self._tuned_live    = True  # turn-taking / media / thinking knobs; same fallback

        _base_dir = Path(__file__).resolve().parent
        _inline_names = {t["name"] for t in TOOL_DECLARATIONS}

        # File-backed tools: every actions/*.py with a TOOL dict, discovered the
        # same way plugins are. Reserved names = the inline tools above, so an
        # action can never shadow one.
        self._action_registry = discover_actions(
            actions_dir=_base_dir / "actions",
            reserved_names=_inline_names,
            logger=lambda msg: print(f"[Actions] {msg}"),
        )

        # Plugins must not collide with either an inline tool or a discovered action.
        _core_names = _inline_names | self._action_registry.names()
        self._plugin_registry = discover_plugins(
            plugins_dir=_base_dir / "plugins",
            core_tool_names=_core_names,
            # Console gets the full boot transcript; the activity log gets only
            # what the user has to know about. Every plugin loading correctly is
            # the expected case and does not belong in their conversation.
            logger=lambda msg: print(f"[Plugins] {msg}"),
            notify=lambda msg: self.ui.write_log(f"SYS: {msg}"),
        )
        self.ui.get_plugins = self._plugin_registry.list_for_ui
        self.ui.get_plugin_settings = self._plugin_registry.settings_schemas  # ⚙ settings tab
        self.ui.on_plugins_reload = self._ui_reload_plugins
        self.ui.request_say = self.plugin_say   # plugins: mid-task speech channel

        # ── Wake word ────────────────────────────────────────────────────────
        # _awake gates the mic (see _listen_audio) and the background speakers.
        # It is True whenever wake word is OFF, so default behaviour is unchanged.
        self._wake_enabled     = get_wake_word_enabled()
        self._awake            = not self._wake_enabled
        self._wake_detector: WakeWordDetector | None = None
        self._wake_sleep_timeout = WAKE_SLEEP_TIMEOUT

        # Restore the saved push-to-talk preference. Doing it here rather than
        # in __init__ means the hotkey thread only exists once there is a
        # session to talk to.
        if get_push_to_talk_enabled():
            try:
                self.set_push_to_talk(True)
            except Exception as e:
                print(f"[JARVIS] ⚠ Push-to-talk unavailable: {e}")
        # UI control surface for the Wake Word settings section.
        self.ui.wake_is_ready    = wake_is_ready          # () -> bool
        self.ui.wake_get_state   = self._wake_state       # () -> dict
        self.ui.on_wake_toggle   = self._ui_wake_toggle   # (enable: bool) -> str
        self.ui.on_wake_manual   = self._ui_wake_manual   # () -> toggle awake/asleep
        self.ui.on_wake_install  = self._ui_wake_install  # () -> (ok, msg)

        # ── Voice-fingerprint isolation & advanced listening modes ───────────
        self._voice_isolator: VoiceIsolator | None = None
        self._audio_gate = self._build_audio_gate()
        self._explicit_vad_active = False
        self._thinking_since: float | None = None
        self._active_reply_lang   = get_reply_language()
        try:
            self._active_gemini_key = _get_api_key()
        except Exception:
            self._active_gemini_key = ""
        self._turn_audio_bytes    = 0
        self._tool_running        = False
        self._mic_open            = False
        self._enroll_active       = False
        self._enroll_lock         = threading.Lock()
        self._enroll_chunks: list[np.ndarray] = []
        self._enroll_samples_got  = 0
        self._enroll_samples_target = 0
        self._enroll_done         = threading.Event()

        self.ui.voice_get_state      = self._voice_state
        self.ui.on_voice_toggle      = self._ui_voice_toggle
        self.ui.on_voice_enroll      = self._ui_voice_enroll
        self.ui.on_voice_clear       = self._ui_voice_clear
        self.ui.listen_get_state     = self._listen_mode_state
        self.ui.on_listen_mode_cycle = self._ui_cycle_listening_mode
        self.ui.on_screen_watch_toggle = lambda v: setattr(self, "_screen_watch_enabled", bool(v))
        self.ui.on_companion_toggle    = lambda v: setattr(self, "_companion_chat_enabled", bool(v))
        self.ui.on_private_toggle      = self._on_private_toggle
        self.ui.on_ai_settings_changed = self._on_ai_settings_changed
        self._last_turn_tools: list[str] = []
        self._last_turn_user_text: str = ""
        try:
            firebase_sync.start_realtime_listener(on_update=self.ui.push_feedback)
        except Exception:
            pass

    # ── Voice-fingerprint isolation & advanced listening modes ───────────────

    def _build_voice_isolator(self) -> VoiceIsolator | None:
        if not is_voice_isolation_active():
            return None
        raw = get_voice_fingerprint()
        if not raw:
            return None
        try:
            fp = VoiceFingerprint.from_dict(raw)
            return VoiceIsolator(
                fp,
                threshold=get_voice_isolation_threshold(),
                end_silence_ms=get_voice_end_silence_ms(),
            )
        except Exception as e:
            print(f"[VoiceLock] Failed to load fingerprint: {e}")
            return None

    def _build_audio_gate(self):
        """Construct the active input gate for the configured listening mode."""
        mode = get_listening_mode()
        if mode == "local_ear":
            self._voice_isolator = None
            try:
                return self._build_local_ear()
            except Exception as e:
                print(f"[LocalEar] could not start ({e}) — falling back to Smart Turn")
                mode = "smart_vad"
        if mode == "voice_lock":
            iso = self._build_voice_isolator()
            if iso is not None:
                self._voice_isolator = iso
                return iso
            mode = "smart_vad"
        self._voice_isolator = None
        if mode == "smart_vad":
            return SmartSpeechGate(end_silence_ms=get_voice_end_silence_ms())
        return None

    # ── Local Ear: the PC listens first, Gemini only hears what was worth hearing ─
    def _make_frontend(self):
        """Mic clean-up. The clear-voice frontend is used only with Local Ear: Voice
        Lock's fingerprint scores were tuned on the old AudioFrontend and must not
        shift under it."""
        try:
            if get_listening_mode() == "local_ear":
                cfg = get_local_ear_config()
                return ClearVoiceFrontend(sample_rate=SEND_SAMPLE_RATE,
                                          denoise=float(cfg["denoise"]))
        except Exception as e:
            print(f"[LocalEar] frontend fallback: {e}")
        return AudioFrontend(sample_rate=SEND_SAMPLE_RATE)

    def _build_local_ear(self):
        cfg = get_local_ear_config()
        if self._ear_transcriber is None and (cfg["speed"] == "accurate" or cfg["media_wake"]):
            self._ear_transcriber = build_transcriber(cfg, log=self.ui.write_log)
        return LocalEarGate(
            cfg,
            transcriber=self._ear_transcriber,
            require_wake_fn=self._ear_must_stay_silent,
            on_wake=self._ear_on_wake,
            on_utterance=self._ear_on_utterance,
            log=self.ui.write_log,
        )

    def _ear_must_stay_silent(self) -> bool:
        """True while a song/video with sound is playing, or after 'sleep': nothing is
        sent to Gemini until the wake word ("Hello") is heard locally."""
        if self._sleep_standby:
            return True
        try:
            return bool(self.ui.video_has_sound())
        except Exception:
            return False

    def _ear_on_wake(self, text: str) -> None:
        """Runs on the ear's worker thread: hop onto the event loop before touching
        the UI/session, exactly as a Gemini-heard "hello" would."""
        self._last_user_utterance = text
        loop = getattr(self, "_loop", None)
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._pause_active_media_and_activate, "hello")
            except Exception:
                pass

    def _ear_on_utterance(self, text: str) -> None:
        self._last_user_utterance = text

    def _refresh_audio_gate(self) -> None:
        old = self._audio_gate
        if old is not None and hasattr(old, "close"):
            try:
                old.close()
            except Exception:
                pass
        self._audio_frontend = self._make_frontend()
        self._audio_gate = self._build_audio_gate()
        self._barge_in_enabled = get_voice_barge_in_enabled()

    def _listen_mode_state(self) -> dict:
        mode = get_listening_mode()
        return {
            "mode": mode,
            "registered": get_voice_fingerprint() is not None,
        }

    def _ui_cycle_listening_mode(self) -> str:
        cur = get_listening_mode()
        has_fp = get_voice_fingerprint() is not None
        order = (["local_ear", "smart_vad", "voice_lock", "continuous"] if has_fp
                 else ["local_ear", "smart_vad", "continuous"])
        try:
            nxt = order[(order.index(cur) + 1) % len(order)]
        except ValueError:
            nxt = order[0]
        saved = save_listening_mode(nxt)
        self._refresh_audio_gate()
        labels = {
            "local_ear":  "LOCAL EAR (PC listens first, then Gemini; silent in songs until Hello)",
            "voice_lock": "VOICE LOCK (registered voice only -> auto THINKING)",
            "smart_vad":  "SMART TURN (noise-gated speech -> auto THINKING)",
            "continuous": "CONTINUOUS (raw mic streaming)",
        }
        self.ui.write_log(f"SYS: Listening mode -> {labels.get(saved, saved)}.")
        self.request_reconnect(keep_context=True, reason=f"listening mode ({saved})")
        return saved

    def _voice_state(self) -> dict:
        raw = get_voice_fingerprint()
        desc = ""
        if raw:
            try:
                desc = VoiceFingerprint.from_dict(raw).describe()
            except Exception:
                desc = "Registered"
        return {
            "enabled": bool(get_voice_isolation_enabled()),
            "registered": raw is not None,
            "active": self._voice_isolator is not None,
            "label": desc,
        }

    def _enter_thought_mode(self) -> None:
        """Immediately transition the HUD (and connected dashboard) to THINKING."""
        self._thinking_since = time.monotonic()
        if not self.ui.muted and self._awake:
            self.ui.set_state("THINKING")
        if self._dashboard and self._loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._dashboard.broadcast({"type": "status", "state": "thinking"}),
                    self._loop,
                )
            except Exception:
                pass

    def _ui_voice_toggle(self, enable: bool) -> bool:
        if enable and get_voice_fingerprint() is None:
            return False
        save_listening_mode("voice_lock" if enable else "smart_vad")
        self._refresh_audio_gate()
        self.ui.write_log(
            "SYS: Voice-fingerprint isolation ON — disregarding all other audio."
            if self._voice_isolator is not None
            else "SYS: Voice-fingerprint isolation OFF (Smart Turn active)."
        )
        self.request_reconnect(keep_context=True, reason="voice isolation")
        return self._voice_isolator is not None

    def _ui_voice_enroll(self, duration_s: float = 4.5) -> tuple[bool, str]:
        """Register the speaker's voiceprint from the live mic (or temp stream)."""
        try:
            if self._mic_open:
                target = int(duration_s * SEND_SAMPLE_RATE)
                with self._enroll_lock:
                    self._enroll_chunks = []
                    self._enroll_samples_got = 0
                    self._enroll_samples_target = target
                    self._enroll_done.clear()
                    self._enroll_active = True
                try:
                    self._enroll_done.wait(timeout=duration_s + 2.5)
                finally:
                    self._enroll_active = False
                with self._enroll_lock:
                    chunks = list(self._enroll_chunks)
                    self._enroll_chunks = []
                if not chunks:
                    return False, "No audio captured from microphone."
                arr = np.concatenate(chunks)[:target]
                fp = enroll_from_pcm(arr.astype("<i2").tobytes(), sr=SEND_SAMPLE_RATE)
            else:
                dev = audio_devices.resolve(get_input_device(), "input")
                fp = enroll_from_microphone(duration_s=duration_s, device=dev)

            save_voice_fingerprint(fp.to_dict())
            save_listening_mode("voice_lock")
            self._refresh_audio_gate()
            self.request_reconnect(keep_context=True, reason="voice fingerprint")
            return True, fp.describe()
        except Exception as e:
            return False, str(e)

    def _ui_voice_clear(self) -> None:
        clear_voice_fingerprint()
        save_listening_mode("smart_vad")
        self._refresh_audio_gate()
        self.ui.write_log("SYS: Registered voice fingerprint cleared.")
        self.request_reconnect(keep_context=True, reason="voice fingerprint cleared")

    def _ui_reload_plugins(self) -> tuple[int, int]:
        active, rejected = self._plugin_registry.reload()
        self.ui.write_log(
            f"SYS: Plugins reloaded — {active} active"
            + (f", {rejected} rejected." if rejected else ".")
        )
        self.request_reconnect(keep_context=True, reason="plugins reloaded")
        return active, rejected

    # ── Wake word: state machine ─────────────────────────────────────────────

    def _wake_state(self) -> dict:
        # A loaded, running detector is definitively ready; otherwise fall back
        # to the cheap on-disk model-file check (no Model construction).
        ready = bool(self._wake_detector and self._wake_detector.ready) or wake_is_ready()
        return {"enabled": self._wake_enabled, "awake": self._awake, "ready": ready}

    def _ensure_wake_detector(self) -> bool:
        """Load the detector once (model loads on first start). Idempotent."""
        if self._wake_detector is None:
            self._wake_detector = WakeWordDetector(
                on_detect=self._on_wake_detected,
                logger=lambda m: print(f"[Wake] {m}"),
                notify=lambda m: self.ui.write_log(f"SYS: {m}"),
            )
        if not self._wake_detector.ready:
            return self._wake_detector.start()
        return True

    def _on_wake_detected(self) -> None:
        """Called from the detector thread when 'Hey Jarvis' is heard."""
        self.wake(reason="wake word")

    def wake(self, reason: str = "wake word") -> None:
        if self._awake:
            return
        self._awake = True
        self._last_user_speech = time.monotonic()   # start the auto-sleep clock now
        if not self.ui.muted:
            self.ui.set_state("LISTENING")
        self.ui.write_log(f"SYS: Awake — {reason}.")

    def sleep(self, reason: str = "timeout") -> None:
        if not self._awake:
            return
        self._awake = False
        self._thinking_since = None
        if self._audio_gate is not None:
            self._audio_gate.reset()
        self.set_speaking(False)
        self.ui.set_state("SLEEPING")
        self.ui.write_log(f"SYS: Sleeping — {reason}. Say 'Hey Jarvis' to wake me.")

    async def _run_sleep_watch(self) -> None:
        """Auto-sleep after the configured silence window (wake-word mode only)."""
        while True:
            await asyncio.sleep(5)
            if not self._wake_enabled or not self._awake:
                continue
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue
            if (time.monotonic() - self._last_user_speech) > self._wake_sleep_timeout:
                self.sleep(reason="no speech for 2 minutes")

    # ── Wake word: UI callbacks (called from the Qt thread) ──────────────────

    def _ui_wake_toggle(self, enable: bool) -> str:
        """Enable/disable wake word from the settings UI. Returns a status token:
        'enabled' | 'disabled' | 'need_download'."""
        if enable:
            if not wake_is_ready():
                return "need_download"
            self._wake_enabled = True
            save_wake_word_enabled(True)
            self._ensure_wake_detector()
            self.sleep(reason="wake word enabled")
            return "enabled"
        else:
            self._wake_enabled = False
            save_wake_word_enabled(False)
            self.wake(reason="wake word disabled")
            return "disabled"

    def _ui_wake_manual(self) -> None:
        """Manual sleep/wake button in the UI."""
        if not self._wake_enabled:
            return
        if self._awake:
            self.sleep(reason="you tapped sleep")
        else:
            self.wake(reason="you tapped wake")

    def _ui_wake_install(self) -> tuple[bool, str]:
        """Download openwakeword + the model (runs in a UI worker thread)."""
        # Triggered by the user pressing the button, so its progress is exactly
        # what they are waiting to see.
        return wake_install(logger=lambda m: print(f"[Wake] {m}"),
                            notify=lambda m: self.ui.write_log(f"SYS: {m}"))

    def plugin_say(self, instruction: str) -> None:
        """
        Thread-safe speech channel for plugins: lets a plugin ask JARVIS to
        say something short WHILE its run() is still executing (plugins block
        their executor thread, so they can't speak through the tool response
        until they finish). The instruction is injected into the Live session
        exactly like a proactive check-in; Gemini phrases it naturally in the
        user's language. Silently a no-op when no session is connected.
        """
        loop = getattr(self, "_loop", None)
        if not loop or not self.session:
            return

        async def _say():
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": instruction}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[PluginSay] {e}")

        try:
            asyncio.run_coroutine_threadsafe(_say(), loop)
        except Exception as e:
            print(f"[PluginSay] {e}")

    # ── stall watchdog ───────────────────────────────────────────────────────

    def _arm_stall(self) -> None:
        """A user turn has gone out; start the clock if one is not already running.

        Deliberately does NOT push an existing deadline forward. The question is
        "how long since the server last said anything while we were waiting",
        not "how long since the most recent send" — otherwise a burst of sends
        into a dead socket would keep resetting the very timer meant to catch it.
        """
        if self._rx_deadline is None:
            self._rx_deadline = time.monotonic() + _STALL_LIMIT

    def _clear_stall(self) -> None:
        """The server spoke, so the return path is alive."""
        self._rx_deadline = None

    async def _watch_stall(self) -> None:
        """Rebuild the session when a sent turn goes unanswered past the limit,
        and recover cleanly if THINKING ever hangs without a server turn."""
        while True:
            await asyncio.sleep(2)
            # Recover if THINKING lasted >15s with no speaking/tool activity
            ts = self._thinking_since
            if ts is not None and not self._tool_running:
                with self._speaking_lock:
                    sp = self._is_speaking
                if not sp and (time.monotonic() - ts) > 15.0:
                    self._thinking_since = None
                    if self._audio_gate is not None:
                        self._audio_gate.reset()
                    if not self.ui.muted and self._awake:
                        self.ui.set_state("LISTENING")
                    self.ui.write_log("SYS: Ready — listening.")
            deadline = self._rx_deadline
            if deadline is None or time.monotonic() < deadline:
                continue
            self._clear_stall()
            self._thinking_since = None
            print(f"[JARVIS] ⚠️  No response for {_STALL_LIMIT:.0f}s — "
                  f"the live session is not answering. Rebuilding it.")
            self.ui.write_log("SYS: The model stopped answering — reconnecting.")
            self.request_reconnect(keep_context=True, reason="no response")
            return          # this session is going away; the new one gets its own watcher

    def request_reconnect(self, keep_context: bool = True, reason: str = ""):
        """Thread-safe: ask the run loop to tear down and rebuild the Live
        session. Called from the Qt thread. No-op until the async loop and
        reconnect event exist.

        `keep_context=False` drops the resumption handle so the new session
        starts empty — only for changes the server cannot apply to a resumed
        session."""
        loop = getattr(self, "_loop", None)
        ev   = self._reconnect_event
        self._reconnect_keep   = keep_context
        self._reconnect_reason = reason
        if loop and ev is not None:
            loop.call_soon_threadsafe(ev.set)

    def _on_voice_change(self):
        """Voice picker applied.

        The voice is baked into the session at connect time, so a rebuild is
        required. It is rebuilt WITHOUT the resumption handle on purpose:
        resuming restores the server's own session state, and the safe reading
        is that it restores the voice with it — which would make the picker
        appear to do nothing. Losing context here is acceptable because changing
        voice is a deliberate, rare act; losing it on a dropped packet was not."""
        self.request_reconnect(keep_context=False, reason="new voice")

    def _on_audio_device_change(self):
        """Microphone, speaker, voice-lock, or engine/key settings changed.
        Rebuilds the audio gate and Live session; drops resumption handle when
        reply language or Gemini API key changed."""
        self._refresh_audio_gate()
        new_lang = get_reply_language()
        lang_changed = (new_lang != getattr(self, "_active_reply_lang", new_lang))
        self._active_reply_lang = new_lang
        try:
            new_key = _get_api_key()
        except Exception:
            new_key = ""
        key_changed = (new_key != getattr(self, "_active_gemini_key", new_key))
        self._active_gemini_key = new_key
        reset_ctx = lang_changed or key_changed
        reason = (
            "gemini api key" if key_changed
            else (f"reply language ({new_lang})" if lang_changed else "audio settings")
        )
        self.request_reconnect(keep_context=not reset_ctx, reason=reason)

    def _ensure_executor(self):
        """Give the loop a working thread pool before every connect.

        Reported: a reconnect died with "cannot schedule new futures after
        shutdown" while resolving Gemini's address — the loop's default pool
        had been shut down underneath a loop that was still running. Name
        resolution, every tool and every `to_thread` go through that pool, so
        once it is gone the assistant can never connect again and retries the
        same failure forever. Whatever shut it, a fresh pool here turns that
        into one lost attempt instead of a dead session.

        The loop owns a pool of our own rather than asyncio's default, sized so
        a long coding job or a slow tool cannot starve address lookups."""
        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_running_loop()
        ex = getattr(loop, "_default_executor", None)
        dead = ex is not None and (getattr(ex, "_shutdown", False)
                                   or getattr(ex, "_broken", False))
        if ex is None or dead or getattr(loop, "_executor_shutdown_called", False):
            if dead:
                print("[JARVIS] ⚠️  Thread pool was shut down — replacing it")
                self.ui.write_log("SYS: Recovered from a stopped worker pool.")
            # Without this flag reset, asyncio refuses the new pool as well.
            if getattr(loop, "_executor_shutdown_called", False):
                loop._executor_shutdown_called = False
            loop.set_default_executor(
                ThreadPoolExecutor(max_workers=32, thread_name_prefix="jarvis"))

    async def _watch_reconnect(self):
        """Session-scoped task: when a voluntary reconnect is requested, raise a
        signal that unwinds the TaskGroup so the run loop rebuilds the session."""
        assert self._reconnect_event is not None
        await self._reconnect_event.wait()
        self._reconnect_event.clear()
        keep   = self._reconnect_keep
        reason = getattr(self, "_reconnect_reason", "") or "settings"
        self.ui.write_log(
            f"SYS: Applying {reason} — reconnecting"
            + ("..." if keep else " (starting a fresh conversation)...")
        )
        raise _ReconnectSignal(keep_context=keep)

    # [REMOTE CONTROL DISABLED — replaced by Phone Link] (method below is commented out)
    # def _make_remote_key(self):
        # """Called from Qt main thread when user presses Remote Control."""
        # if self._dashboard is None:
            # self.ui.write_log(
                # "SYS: Dashboard unavailable. "
                # "Run: pip install fastapi \"uvicorn[standard]\" cryptography"
            # )
            # return None
        # # The firewall is fixed HERE, not at startup: this is the moment the
        # # user asked for the phone dashboard, so a Windows admin prompt now has
        # # an obvious reason. Once per run, and a no-op when nothing is blocked.
        # try:
            # from dashboard.server import PORT, SECURE_PORT, fix_network_access
            # threading.Thread(target=fix_network_access, args=(PORT, SECURE_PORT),
                             # daemon=True, name="dashboard-firewall").start()
        # except Exception as e:
            # print(f"[Dashboard] Firewall check skipped: {e}")
        # key    = self._dashboard.new_key()
        # url    = self._dashboard.get_url()
        # manual = self._dashboard.get_manual_url()
        # # The bind error travels with the URL because the UI has no reference to
        # # this object — it only ever sees what this callback returns.
        # auto = f"{url}/auto-login?key={key}"
        # try:
            # # Other addresses of this laptop, for the phone app to fall back on (a browser ignores them).
            # from dashboard.server import lan_ips
            # _alts = [ip for ip in lan_ips() if ip not in url]
            # if _alts:
                # auto += "&alt=" + ",".join(_alts[:4])
        # except Exception:
            # pass
        # return (url, key, auto, manual,
                # getattr(self._dashboard, "_bind_error", "") or "")

    def _make_phone_link(self):
        """Called from the Qt thread when the user opens PHONE LINK: a fresh one-time key and the QR text."""
        if self._dashboard is None:
            self.ui.write_log("SYS: Phone link is not running.")
            return None
        try:
            if not getattr(self._dashboard, "_httpd", None):
                self._dashboard.start()            # a busy port may have freed up since startup
            return self._dashboard.info()
        except Exception as e:
            self.ui.write_log(f"SYS: Phone link error: {e}")
            return None

    def _on_private_toggle(self, enabled: bool) -> None:
        self._set_private_mode(bool(enabled), notify_session=True)

    def _set_private_mode(self, enabled: bool, notify_session: bool = True) -> None:
        enabled = bool(enabled)
        changed = (enabled != self._private_mode)
        self._private_mode = enabled
        self.ui.set_private_mode(enabled)
        if not (changed and notify_session and self._loop and self.session):
            return
        note = (
            "[PRIVATE / RAAZ MODE ENABLED] Reply very softly, calmly, and whisper-briefly. "
            "Do NOT open browser/panels or display anything on screen."
            if enabled else
            "[PRIVATE / RAAZ MODE DISABLED] Return to normal speaking tone and volume."
        )
        try:
            asyncio.run_coroutine_threadsafe(
                self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": note}]},
                    turn_complete=False,
                ),
                self._loop,
            )
        except Exception:
            pass

    @staticmethod
    def _detect_private_command(text: str) -> str:
        low = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not low:
            return ""
        if re.search(r"\b(?:private\s+(?:mode\s+)?off|raaz\s+off|secret\s+(?:mode\s+)?off|normal\s+mode|public\s+mode|exit\s+private)\b", low):
            return "off"
        if re.search(r"\b(?:private(?:\s+mode)?|raaz(?:\s+mode)?|secret\s+mode)\b", low):
            return "on"
        return ""

    def _update_emotion_from_text(self, text: str, role: str = "user") -> None:
        pitch = self._last_user_pitch_hz if role == "user" else 0.0
        rms = self._last_user_rms if role == "user" else 0.0
        emo, intensity = local_db.detect_emotion(text, pitch_hz=pitch, rms=rms)
        if emo != "neutral" or intensity >= 0.55:
            local_db.record_emotion(emo, intensity=intensity, text=text, pitch_hz=pitch, rms=rms)
        self._last_emotion = emo
        if self._private_mode:
            self.ui.set_emotion("private")
        else:
            self.ui.set_emotion(emo)

    def _on_ai_settings_changed(self, reason: str = "") -> None:
        self._refresh_audio_gate()
        if reason and reason != "refresh":
            self.request_reconnect(keep_context=True, reason=reason)

    def _process_user_memory_and_topic(self, text: str) -> None:
        if not text:
            return
        if get_auto_learn_enabled():
            learned = local_db.learn_from_text(text)
            if learned:
                self.ui.push_feedback(f"✓ LEARNED ({len(learned)})")
        cur_topic, _ = local_db.get_active_topic()
        cand = local_db.detect_topic_from_text(text)
        explicit_switch = bool(re.search(
            r"\b(?:topic|baat karte|let'?s talk about|change topic|ab baat|next topic)\b",
            text, re.I
        ))
        if cand and (not cur_topic or explicit_switch or not get_topic_lock_enabled()):
            local_db.set_active_topic(cand, locked=get_topic_lock_enabled())
            if cand != cur_topic:
                self.ui.push_feedback(f"🎯 TOPIC · {cand[:20]}")

    @staticmethod
    def _detect_sleep_wake_command(text: str) -> str:
        low = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not low:
            return ""
        if (re.search(r"\b(?:sleeping|sleep mode|go to sleep|so ja|so jao|standby mode)\b", low)
                or any(w in text for w in ("स्लीप", "सो जा", "सो जाओ"))):
            return "sleep"
        if (re.search(r"\b(?:hello(?:\s+jarvis)?|halo|helo|hlo|hi\s+jarvis|hey\s+jarvis|wake up|utho|jaago)\b", low)
                or any(w in text for w in ("हेलो", "हलो", "हैलो", "उठो", "जागो"))):
            return "wake"
        return ""

    def _inject_session_note(self, note: str) -> None:
        """Silently update the active Live session context without triggering a reply."""
        loop = getattr(self, "_loop", None)
        sess = getattr(self, "session", None)
        if not loop or not sess or not note:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                sess.send_client_content(
                    turns={"role": "user", "parts": [{"text": note}]},
                    turn_complete=False,
                ),
                loop,
            )
        except Exception:
            pass

    def _pause_active_media_and_activate(self, reason: str = "hello") -> None:
        paused = False
        try:
            if self.ui.video_is_playing() and not self.ui.video_is_paused():
                self.ui.pause_video()
                paused = True
        except Exception:
            pass
        try:
            self.ui.pause_browser_media()
        except Exception:
            pass
        was_asleep = self._sleep_standby or (not self._awake)
        self._sleep_standby = False
        if not self._awake:
            self.wake(reason=reason)
        try:
            if getattr(self.ui, "muted", False):
                self.ui.muted = False
        except Exception:
            pass
        if not self.ui.muted:
            self.ui.set_state("LISTENING")
        if paused:
            self.ui.push_feedback("⏸ SONG PAUSED · AI ACTIVE")
        elif was_asleep:
            self.ui.push_feedback("👋 AWAKE · LISTENING")

    def _apply_sleep_or_hello_from_text(self, text: str, from_voice: bool = True) -> str:
        if not text:
            return ""
        self._last_user_utterance = text
        low = text.lower()
        sw = self._detect_sleep_wake_command(low)
        if sw == "sleep":
            self._sleep_standby = True
            self.set_speaking(False)
            try:
                if getattr(self.ui, "muted", False):
                    self.ui.muted = False
            except Exception:
                pass
            self.ui.set_state("SLEEPING")
            self.ui.push_feedback("💤 SLEEP · SAY HELLO")
            return "sleep"
        if sw == "wake":
            self._pause_active_media_and_activate("hello")
            return "wake"
        if re.search(r"\b(?:pause\s+(?:song|music|video)|gana\s+(?:roko|band|pause)|ruko)\b", low):
            self.ui.pause_video()
            self.ui.pause_browser_media()
            self.ui.push_feedback("⏸ MEDIA PAUSED")
        elif re.search(r"\b(?:resume\s+(?:song|music|video)|gana\s+(?:chalao|resume|play))\b", low):
            self.ui.resume_video()
            self.ui.resume_browser_media()
            self.ui.push_feedback("▶ MEDIA RESUMED")
        return ""

    def _on_text_command(self, text: str):
        if not self._loop or not self.session:
            return
        sw = self._apply_sleep_or_hello_from_text(text, from_voice=False)
        # Allow "hello" / wake text commands even when sleeping
        if self._wake_enabled and not self._awake and sw != "wake":
            self.ui.write_log("SYS: I'm asleep — say 'Hello' or 'Hey Jarvis' first.")
            return
        if sw == "sleep":
            self.ui.write_log("SYS: Sleep mode ON — say 'Hello' to wake me up.")
            return
        priv_cmd = self._detect_private_command(text)
        if priv_cmd == "on":
            self._set_private_mode(True, notify_session=False)
        elif priv_cmd == "off":
            self._set_private_mode(False, notify_session=False)
        if text:
            self._last_turn_user_text = text
            self._last_turn_tools = []
            self._process_user_memory_and_topic(text)
            self._update_emotion_from_text(text, role="user")
            local_db.record_turn("user", text, self._last_emotion)
            if not self._private_mode:
                try:
                    from core import behavior as _behavior
                    _behavior.observe(text, self._last_emotion)
                except Exception:
                    pass
                firebase_sync.train_from_exchange(text, emotion=self._last_emotion)
        payload = text
        _exp = firebase_sync.extract_explicit_training(text) if (text and not self._private_mode) else None
        if _exp:
            payload = (
                f"{text}\n[PERMANENTLY TRAINED COMMAND: '{_exp[0]}' -> {_exp[1]}. "
                "Confirm briefly and execute it directly from now on whenever the user asks.]"
            )
        _fuzzy = firebase_sync.resolve_unclear_command(text) if text else None
        if _fuzzy:
            _f_phr, _f_act, _f_sc = _fuzzy
            payload = (
                f"{text}\n[TRAINED COMMAND MATCH ({_f_sc:.2f}) — EXECUTE DIRECTLY NOW: "
                f"'{_f_phr}' -> {_f_act}. Do NOT ask the user to re-explain or re-train.]"
            )
            self.ui.push_feedback(f"🎯 {_f_phr[:20]}")
        if priv_cmd == "on":
            payload = f"{text}\n[SYSTEM NOTE: Private / Raaz mode ON — speak softly in a low whisper and keep nothing on screen.]"
        elif priv_cmd == "off":
            payload = f"{text}\n[SYSTEM NOTE: Private / Raaz mode OFF — resume normal volume.]"
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": payload}]},
                turn_complete=True
            ),
            self._loop
        )

    def _on_image_uploaded(self, path: str) -> None:
        """Analyze an image dropped into the HUD File Upload box immediately."""
        import base64 as _b64
        import io as _io
        import mimetypes as _mime

        p = Path(path)
        if not p.is_file():
            self.ui.write_log(f"SYS: Uploaded image not found: {p.name}")
            return

        if (
            self._engine_caps is not None
            and not self._engine_caps.get("vision")
            and self._action_registry.has("file_processor")
        ):
            res = self._action_registry.run(
                "file_processor",
                {"file_path": path, "action": "analyze"},
                {"player": self.ui, "speak": self.speak, "response": None, "session_memory": None},
            )
            self.ui.show_content(f"IMAGE — {p.name}", str(res or "No description returned."))
            return

        if not self._loop or not self.session:
            self.ui.write_log(f"SYS: Image '{p.name}' loaded — ask once connected.")
            return
        if self._wake_enabled and not self._awake:
            self.ui.write_log("SYS: I'm asleep — say 'Hey Jarvis' or tap WAKE NOW first.")
            return

        mime_t = "image/jpeg"
        img_bytes = b""
        try:
            from PIL import Image
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((1280, 1280))
                buf = _io.BytesIO()
                im.save(buf, format="JPEG", quality=85)
                img_bytes = buf.getvalue()
        except Exception:
            try:
                img_bytes = p.read_bytes()
                mime_t = _mime.guess_type(p.name)[0] or "image/png"
            except Exception as e:
                self.ui.write_log(f"ERR: Could not read image '{p.name}' — {e}")
                return

        b64 = _b64.b64encode(img_bytes).decode("ascii")
        prompt_txt = (
            f"[IMAGE_UPLOADED] [IMAGE SOURCE: UPLOADED FILE '{p.name}' (path={path})]\n"
            f"Analyze this uploaded image carefully: describe what is in it, read any "
            f"visible text, code, or errors, and ask briefly how I can help next."
        )
        self._enter_thought_mode()
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"role": "user", "parts": [
                    {"inline_data": {"mime_type": mime_t, "data": b64}},
                    {"text": prompt_txt},
                ]},
                turn_complete=True,
            ),
            self._loop,
        )

    def _tail_active(self) -> bool:
        """True while the speakers may still be finishing our last sentence."""
        return time.monotonic() < self._tail_until

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            self._is_speaking = value
        if value:
            self._tail_until = 0.0
            self._thinking_since = None
            if self._audio_gate is not None:
                self._audio_gate.reset()
        else:
            # Hold the guard open across the device's own output latency plus a
            # margin for the room. The microphone is NOT muted during it — the
            # guard still lets a genuine reply through, so answering instantly
            # still works. Only our own echo is dropped.
            self._tail_until = time.monotonic() + self._out_latency + _TAIL_MARGIN
            self._last_assistant_speech = time.monotonic()
        if not value:
            # The echo history is deliberately NOT cleared here: the tail above
            # still needs it to recognise our own voice. It is dropped when the
            # tail expires. What the guard learned about the room always stays.
            self._out_level = 0.0
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def set_push_to_talk(self, enabled: bool) -> str:
        """Turn hold-to-talk on or off. Returns the scope actually achieved."""
        from core.hotkey import PushToTalk

        self._ptt_enabled = bool(enabled)
        self._ptt_held = False
        if not enabled:
            if self._ptt is not None:
                self._ptt.stop()
                self._ptt = None
            return "off"

        if self._ptt is None:
            self._ptt = PushToTalk(self._on_ptt)
        scope = self._ptt.start()
        # A window-scoped chord is a real limitation, not a detail — say it once
        # in the log so nobody wonders why it does nothing while another app is
        # focused. Reporting it must never be able to undo the thing it reports.
        try:
            self.ui.write_log(
                f"SYS: Push-to-talk on — hold {self._ptt.label}"
                + ("." if scope == "global"
                   else " (works while this window is focused)."))
        except Exception:
            pass
        return scope

    def _on_ptt(self, held: bool) -> None:
        """Chord pressed or released — may arrive on the hotkey thread."""
        self._ptt_held = held
        if held:
            # Holding the key is also a way to wake it, so push-to-talk works
            # without having to say the wake word first.
            if self._wake_enabled and not self._awake:
                self._awake = True
                self._last_user_speech = time.monotonic()
        # Releasing the key ends the turn. On Live the server works that out from
        # the audio; a local pipeline has no server, so the release IS the
        # signal — which is also why it is the more reliable of the two ways to
        # talk to one. Scheduled onto the loop because this may arrive on the
        # hotkey thread.
        if not held and self._audio_gate is not None and self.out_queue is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(
                    self._enqueue_audio, {"type": "voice_stop"}
                )
            except Exception as e:
                print(f"[JARVIS] Voice stop signal failed: {e}")
        elif not held and self._engine is not None and self._loop is not None:
            try:
                self._enter_thought_mode()
                asyncio.run_coroutine_threadsafe(
                    self._engine.flush_microphone(SEND_SAMPLE_RATE), self._loop)
            except Exception as e:
                print(f"[JARVIS] PTT flush failed: {e}")
        try:
            if held:
                state = "LISTENING"
            else:
                # Releasing the key does NOT mean the assistant went to sleep.
                #
                # This used to paint SLEEPING unconditionally, and SLEEPING in
                # this app means one specific thing — the wake-word gate is
                # closed, the lids are down, nothing is being listened to at all.
                # With push-to-talk on, every key release asserted that, so
                # tapping WAKE NOW appeared to do nothing: the assistant really
                # did wake, and the next release immediately painted it asleep
                # again. Two different ideas were sharing one state.
                state = ("SLEEPING" if (self._wake_enabled and not self._awake)
                         else "LISTENING")
            self.ui.set_state(state)
        except Exception:
            pass

    def interrupt(self) -> None:
        """Stop JARVIS mid-speech: drain queued audio and open mic immediately."""
        if self._server_turn_active:
            self._interrupted = True
        self._thinking_since = None
        self._barge_in_streak = 0
        # Esc / "stop" also stops a learned click-sequence that is mid-replay.
        _st = sys.modules.get("actions.screen_teach")
        if _st is not None:
            try:
                _st.abort_running()
            except Exception:
                pass
        if self._audio_gate is not None:
            self._audio_gate.reset()
        q = self.audio_in_queue
        if q:
            drained = 0
            while True:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break
            if drained:
                print(f"[JARVIS] ✋ Interrupted — {drained} audio chunks discarded")
        self.set_speaking(False)
        # The words we were about to mouth are never going to be spoken now.
        self._visemes.reset()
        self._play_cursor = 0.0     # next batch starts a fresh timeline
        if self._turn_done_event:
            self._turn_done_event.clear()
        # Draining the queue stops the speakers, but a local engine is still
        # generating the rest of the answer and would keep filling it back up.
        # Live has nothing to cancel here — the server stops on its own.
        if self._engine is not None:
            try:
                self._engine.interrupt()
            except Exception:
                pass
        self.ui.write_log("SYS: Interrupted — listening...")

    def _on_voice_barge_in(self) -> None:
        """Triggered on the event loop when user voice cuts in over JARVIS speaking."""
        with self._speaking_lock:
            speaking = self._is_speaking
        if not speaking:
            return
        self._tail_until = 0.0
        self.interrupt()

    def speak(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    def _build_config(self) -> types.LiveConnectConfig:
        from datetime import datetime

        # Load customization from config
        try:
            _cfg = json.loads(open(API_CONFIG_PATH, encoding="utf-8").read())
            self._asst_name = (_cfg.get("assistant_name") or "JARVIS").strip()
            _user_name = (_cfg.get("user_name") or "").strip()
        except Exception:
            self._asst_name = "JARVIS"
            _user_name = ""

        memory     = load_memory()
        mem_str    = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )

        # Identity injection — overrides any hardcoded name in prompt.txt
        # Address form is a property of the language being spoken, so it is
        # stated as a principle rather than a two-language lookup — the model
        # already knows the respectful register of whatever language it is in.
        _addr = (f"ADDRESS: Always call the user '{_user_name}'."
                 if _user_name
                 else 'ADDRESS: Speak to the user the way a close friend does: '
                      'casual and warm, by name if you know it, never "sir" or '
                      'any formal title, in whatever language you are speaking.')
        _lang = get_reply_language()
        _lang_rules = {
            "hinglish": (
                "LANGUAGE LOCK (STRICT): Reply ONLY in natural Hinglish (Hindi + everyday "
                "English words). ALLOWED LANGUAGES: Hindi and English ONLY. NEVER speak or "
                "switch to Spanish, Portuguese, French, German, Italian, or ANY third language "
                "even if audio sounds unclear, noisy, or phonetically ambiguous — ask briefly "
                "in Hinglish ('Dobara boliye, clear sunai nahi diya') instead."
            ),
            "hindi": (
                "LANGUAGE LOCK (STRICT): Reply ONLY in natural Hindi. ALLOWED LANGUAGES: "
                "Hindi and English words only when needed. NEVER speak or switch to Spanish, "
                "Portuguese, French, or ANY other language because of English tool outputs "
                "or short/unclear audio."
            ),
            "english": (
                "LANGUAGE LOCK (STRICT): Reply ONLY in English. NEVER switch to Spanish "
                "or any other language under any circumstances."
            ),
            "auto": (
                "LANGUAGE RULE (STRICT): Reply ONLY in Hindi, Hinglish, or English matching "
                "the user. NEVER switch to Spanish, Portuguese, French, German, or any third "
                "language because of English system alerts, tool outputs, or unclear/short noises."
            ),
        }
        identity_ctx = (
            f"[IDENTITY]\n"
            f"Your name is {self._asst_name}. "
            f"Always refer to yourself as {self._asst_name}.\n"
            f"{_addr}\n"
            f"{_lang_rules.get(_lang, _lang_rules['auto'])}\n\n"
        )

        # Everything the model is told about *itself* is derived here, not
        # written into prompt.txt: the name comes from config, the platform from
        # the host, the capability list from the registries that were just
        # discovered. Rename the assistant, add a plugin or move to another OS
        # and this follows without anyone editing a prompt.
        _all_decls = (TOOL_DECLARATIONS
                      + self._action_registry.get_tool_declarations()
                      + self._plugin_registry.get_tool_declarations())

        # An engine that cannot do something is not offered the tool for it.
        #
        # This is why capabilities are data rather than a pile of conditions: the
        # SAME filtering feeds `_describe_limits` two lines below, through
        # `_names`, so an engine with no eyes ends up with the model being TOLD
        # it cannot see. Without that it would be handed a tool it cannot serve,
        # call it, get nothing back, and describe a screen it never received.
        #
        # `self._engine_caps is None` is Gemini Live and skips all of it, so the
        # declaration list is byte-for-byte what it has always been.
        if self._engine_caps is not None:
            _caps = self._engine_caps
            _blocked = set()
            if not _caps.get("vision"):
                _blocked |= {"screen_process", "close_camera"}
            if not _caps.get("tools"):
                _blocked = {(d.get("name") if isinstance(d, dict)
                             else getattr(d, "name", "")) for d in _all_decls}
            if _blocked:
                _all_decls = [d for d in _all_decls
                              if (d.get("name") if isinstance(d, dict)
                                  else getattr(d, "name", "")) not in _blocked]
                print(f"[JARVIS] {len(_blocked)} tool(s) withheld — "
                      f"this engine cannot serve them")

        _names = {(d.get("name") if isinstance(d, dict) else getattr(d, "name", ""))
                  for d in _all_decls}
        sys_prompt = _render_prompt(sys_prompt, {
            "assistant_name": self._asst_name,
            "platform": f"{_platform.system()} {_platform.release()}".strip(),
            "capabilities": _describe_tools(_all_decls),
            "limits": _describe_limits(
                has_vision="screen_process" in _names,
                has_mic=True,
            ),
        })

        parts = [time_ctx, identity_ctx]
        if mem_str:
            parts.append(mem_str)
        try:
            firebase_sync.pull_and_merge(timeout=2.5)
        except Exception:
            pass
        try:
            _db_ctx = local_db.build_db_context()
            if _db_ctx:
                parts.append(_db_ctx)
        except Exception:
            pass
        try:
            # How to BE with this person right now: bond, mood, time of day, open tasks.
            from core import behavior as _behavior, insights as _insights
            _bh = _behavior.build_prompt_block(task_lines=_insights.task_lines_for_prompt())
            if _bh:
                parts.append(_bh)
        except Exception:
            pass
        try:
            # Click-sequences the user taught ("Chrome band karna"): the model must know they
            # exist, or it asks the user to explain them again. Skills taught on another PC
            # are pulled in the background and appear from the next connection.
            from core import screen_skills as _ss
            from actions import screen_teach as _st_mod
            threading.Thread(target=_st_mod.pull_remote_skills, daemon=True,
                             name="SkillPull").start()
            _sk_ctx = _ss.prompt_block()
            if _sk_ctx:
                parts.append(_sk_ctx)
        except Exception:
            pass
        if get_topic_lock_enabled():
            parts.append(
                "[CONVERSATION FOCUS & PERSISTENT MEMORY]\n"
                "Stay strictly anchored on the current user topic until the user explicitly switches topics. "
                "Do not wander to unrelated topics. Never re-ask learned preferences, rules, names, or paths."
            )
        if self._sleep_standby:
            parts.append(
                "[STANDBY SLEEP MODE ACTIVE]\n"
                "Stay completely silent unless the user greets you with 'hello', 'wake up', or 'hey jarvis'."
            )
        if self._private_mode:
            parts.append(
                "[PRIVATE / RAAZ MODE ACTIVE]\n"
                "Speak softly and whisper-briefly; keep everything off-screen."
            )
        parts.append(sys_prompt)

        cfg = dict(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction="\n".join(parts),
            # strip_meta drops this project's own `tier` / `brief` keys, which
            # exist for the local-engine tool budget and are not part of the
            # Gemini function-declaration schema — an unknown field there is a
            # validation error, not something the API ignores. A declaration
            # that carries neither comes back as the same object, so what
            # Gemini receives is unchanged from the previous release.
            tools=[{"function_declarations": strip_meta(_all_decls)}],
            # Asks the server to issue session-resumption handles. Enabled but
            # NOT yet used: no handle is captured from the resumption update and
            # none is passed back here, so a reconnect starts a fresh session
            # rather than continuing the old one. Left on because the handles
            # cost nothing and are the prerequisite for doing it properly.
            # Hand back the handle captured from the last session_resumption
            # update. `handle=None` is exactly the old behaviour (ask for
            # handles, start fresh), so the first connect of a run is unchanged.
            session_resumption=types.SessionResumptionConfig(
                handle=self._resume_handle
            ),
            # Sliding-window compression: session never dies from a full context
            # window — JARVIS can stay in one conversation for hours
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=get_voice()
                    )
                )
            ),
        )
        if self._enhanced_live:
            # Proactive audio: JARVIS stays silent when speech isn't addressed
            # to it. Disabled when client-side voice isolation / smart-turn
            # explicit VAD is active so isolated commands are always answered
            # instead of getting silently ignored in LISTENING.
            if get_proactive_audio_enabled() and not (
                self._tuned_live and self._audio_gate is not None
            ):
                cfg["proactivity"] = types.ProactivityConfig(proactive_audio=True)

        if self._tuned_live:
            cfg.update(self._tuning_config())

        return types.LiveConnectConfig(**cfg)

    def _tuning_config(self) -> dict:
        """The optional knobs, kept apart so one bad field can be dropped wholesale.

        Every one of these is a preview-API field. If a future model release
        stops accepting any of them the connection fails at setup, so the run
        loop turns `_tuned_live` off and reconnects on the plain config rather
        than leaving the user with an assistant that will not start.
        """
        out: dict = {}

        # When an advanced listening gate (Voice Lock or Smart Turn) is active,
        # disable server-side VAD so client end-of-speech triggers THINKING
        # immediately, and enable thinking mode.
        if self._audio_gate is not None:
            out["realtime_input_config"] = types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True,
                )
            )
            out["thinking_config"] = types.ThinkingConfig(
                thinking_budget=(-1 if get_thinking_enabled() else 0)
            )
            res = get_media_resolution()
            if res != "default":
                out["media_resolution"] = {
                    "low":    types.MediaResolution.MEDIA_RESOLUTION_LOW,
                    "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                    "high":   types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                }[res]
            return out

        # How long the server waits through a pause before deciding your turn is
        # over. This — not the size of the prompt — is what most of the delay
        # before a reply actually is, and the default has to suit everybody, so
        # it is necessarily cautious.
        turn = get_turn_tuning()
        if turn.get("enabled", True):
            detect = types.AutomaticActivityDetection(
                silence_duration_ms=turn["silence_ms"],
                prefix_padding_ms=turn["prefix_ms"],
            )
            if turn["end_sensitivity"] == "high":
                detect.end_of_speech_sensitivity = types.EndSensitivity.END_SENSITIVITY_HIGH
            elif turn["end_sensitivity"] == "low":
                detect.end_of_speech_sensitivity = types.EndSensitivity.END_SENSITIVITY_LOW
            if turn["start_sensitivity"] == "high":
                detect.start_of_speech_sensitivity = types.StartSensitivity.START_SENSITIVITY_HIGH
            elif turn["start_sensitivity"] == "low":
                detect.start_of_speech_sensitivity = types.StartSensitivity.START_SENSITIVITY_LOW
            out["realtime_input_config"] = types.RealtimeInputConfig(
                automatic_activity_detection=detect)

        # Screenshots and camera frames are tokenised at this resolution and then
        # stay in the session's context. 'medium' keeps on-screen text legible
        # for a fraction of a full-resolution frame.
        res = get_media_resolution()
        if res != "default":
            out["media_resolution"] = {
                "low":    types.MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                "high":   types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            }[res]

        # Thinking is left at the server default deliberately. Forcing the budget
        # to zero was measured on gemini-3.1-flash-live over interleaved trials
        # and did not make the first word arrive sooner — this model does not
        # appear to deliberate on the Live path, so pinning the field only adds a
        # way for a future release to behave differently. Set "thinking_enabled"
        # in config/api_keys.json to true to let it reason instead.
        out["thinking_config"] = types.ThinkingConfig(
            thinking_budget=(-1 if get_thinking_enabled() else 0)
        )

        return out

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        print(f"[JARVIS] 🔧 {name}  {args}")
        _t_tool = time.monotonic()

        if getattr(self, "_sleep_standby", False):
            return types.FunctionResponse(
                id=fc.id,
                name=name,
                response={"result": "[STANDBY_SLEEPING] Do not run tools or speak while sleeping until user says Hello."},
            )

        if (name == "hud_dashboard" and str(args.get("feature") or "").lower() == "mute"
                and self._detect_sleep_wake_command(getattr(self, "_last_user_utterance", "")) == "sleep"):
            self._sleep_standby = True
            self.ui.set_state("SLEEPING")
            self.ui.push_feedback("💤 SLEEP · SAY HELLO")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "Entered sleep standby cleanly without muting microphone; say 'Hello' to wake up."}
            )

        try:
            _has_sound = bool(self.ui.video_has_sound()) is True
        except Exception:
            _has_sound = False
        if _has_sound and not re.search(
            r"\b(?:hello|halo|helo|hlo|pause|stop|next|skip|prev|previous|resume|mute|unmute|playlist|search|google|open|kholo|ruko|band|remember|memory|save|close|shutdown|system|pc)\b|हेलो|हलो|हैलो",
            getattr(self, "_last_user_utterance", ""),
            re.I,
        ):
            return types.FunctionResponse(
                id=fc.id,
                name=name,
                response={"result": "[IGNORED_SONG_BLEED] Song is playing; ignore song lyrics unless user says Hello or a clear command."},
            )

        self.ui.set_state("THINKING")


        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key == "language" and any(w in str(value).lower() for w in ("spanish", "portuguese", "french", "espanol")):
                value = "Hinglish (Hindi + English)"
            if key and value:
                update_memory({category: {key: {"value": value}}})
                firebase_sync.train_important_memory(key, value, category=category, importance=0.92)
                self._inject_session_note(
                    f"[PERMANENT MEMORY SAVED ({category}): {key} = {value}. "
                    "Remember and apply this directly without asking again.]"
                )
                self.ui.push_feedback(f"✓ MEMORY · {key[:16]}")
                print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": f"Saved permanently in Firebase, memory & local database: {key}={value}"}
            )

        if hasattr(self, "_last_turn_tools") and isinstance(self._last_turn_tools, list):
            if name not in self._last_turn_tools:
                self._last_turn_tools.append(name)

        loop   = asyncio.get_event_loop()
        result = "Done."

        try:
            if name == "recall_memory":
                # Local file search: no network, no second model. Kept out of
                # the executor deliberately — it is a dictionary scan over a few
                # hundred short strings, and a thread hop would cost more than
                # the work itself.
                result = search_memory(args.get("query", ""), limit=8)

            elif name == "undo":
                if str(args.get("action", "")).lower().strip() == "list":
                    items = undo_stack.history()
                    result = ("Things I can undo, most recent first:\n"
                              + "\n".join(f"{i+1}. {t}" for i, t in enumerate(items))
                              ) if items else "I have not changed anything I can undo yet."
                else:
                    result = await loop.run_in_executor(None, undo_stack.undo_last)

            elif name == "screen_process":
                import time as _t_mod
                _now = _t_mod.monotonic()
                _cooldown = 4.0  # seconds — covers echo window after speaking ends
                if self._vision_busy or (_now - self._vision_last_time) < _cooldown:
                    _wait = max(0, _cooldown - (_now - self._vision_last_time))
                    print(f"[Vision] ⏳ Cooldown active ({_wait:.1f}s remaining) — ignoring duplicate call")
                    result = "Vision is still processing the previous request. I will not call this again."
                else:
                    self._vision_busy      = True
                    self._vision_last_time = _now
                    angle     = args.get("angle", "screen").lower()
                    user_text = args.get("text", "What do you see?")
                    if angle == "camera":
                        img_b, mime_t = await loop.run_in_executor(None, _capture_camera)
                        self.ui.start_camera_stream()
                        self._vision_cam_active = True
                        print(f"[Vision] 📷 Camera: {len(img_b):,} bytes")
                        _stall = "camera"
                    else:
                        img_b, mime_t = await loop.run_in_executor(None, _capture_screen)
                        print(f"[Vision] 🖥️  Screen: {len(img_b):,} bytes")
                        _stall = "screen"
                    self._pending_vision = (img_b, mime_t, user_text, angle)
                    # The image is attached to this same exchange, so there is
                    # nothing to stall for and nothing to announce. Asking for an
                    # acknowledgement here is what produced two spoken answers —
                    # the model filled that turn by answering the question from
                    # imagination, then answered it again once it could see.
                    result = (
                        f"[VISION_ACTIVE] {_stall.capitalize()} captured and attached to this "
                        f"same exchange. Do not acknowledge and do not answer yet — the image "
                        f"is arriving with this result. Reply once, from what you actually see "
                        f"in it."
                    )

            elif name == "close_camera":
                self.ui.stop_camera_stream()
                result = "Camera closed."

            elif name == "system_status":
                r = await loop.run_in_executor(None, get_system_status)
                result = str(r)

            elif name == "manage_monitor":
                action = args.get("action", "").lower().strip()
                topic  = args.get("topic", "").strip()
                if action == "add" and topic:
                    result = await asyncio.to_thread(add_monitor, topic)
                elif action == "remove" and topic:
                    result = await asyncio.to_thread(remove_monitor, topic)
                elif action == "list":
                    topics = await asyncio.to_thread(list_monitors)
                    result = ("Monitoring: " + ", ".join(topics)) if topics else "No topics are being monitored."
                else:
                    result = "Specify action (add/remove/list) and a topic."

            elif name == "shutdown_jarvis":
                self.ui.write_log("SYS: Shutdown requested.")
                async def _do_shutdown():
                    await self._save_session_summary()
                    if self.session:
                        try:
                            await self.session.send_client_content(
                                turns={"role": "user", "parts": [{"text": "Say a brief natural goodbye to the user."}]},
                                turn_complete=True,
                            )
                        except Exception:
                            pass
                    await asyncio.sleep(1.5)
                    import os as _os
                    _os._exit(0)
                asyncio.create_task(_do_shutdown())

            elif self._action_registry.has(name):
                # file_processor: fall back to the currently-uploaded file when none is given
                if name == "file_processor" and not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                # dev_agent: a folder in the upload box is where the work goes.
                #
                # Decided HERE, not left to the model. Told plainly in the tool
                # description, it still answered "create a quiz plugin" with
                # mode='build' — a brand-new project in Desktop/JarvisProjects,
                # nowhere near the folder the user had just handed it. Somebody
                # who loads a folder and asks for something means "in this
                # folder", so every coding request goes there; clearing the
                # upload box (✕) is how to ask for a separate new project.
                if (name == "dev_agent" and not args.get("folder")
                        and self.ui.current_file
                        and Path(self.ui.current_file).is_dir()
                        and str(args.get("mode") or "").lower() != "status"):
                    args["folder"] = self.ui.current_file
                    args["mode"] = "improve"
                _ctx = {"player": self.ui, "speak": self.speak,
                        "response": None, "session_memory": None}
                r = await loop.run_in_executor(None, lambda: self._action_registry.run(name, args, _ctx))
                result = r or "Done."
                if name == "train_command" and result:
                    self._inject_session_note(f"[PERMANENTLY TRAINED — APPLY DIRECTLY FROM NOW ON: {result}]")
                # web_search: mirror results to the on-screen content panel, and if user
                # specifically asked for Google Search, open Google in dashboard browser too
                if (name == "web_search" and r
                        and not r.startswith("No results")
                        and not r.startswith("Search failed")):
                    _mode  = args.get("mode", "search")
                    _query = args.get("query") or ", ".join(args.get("items", []))
                    if ("google" in getattr(self, "_last_user_utterance", "").lower()
                            and _mode == "search" and _query):
                        from urllib.parse import quote_plus as _qp
                        self.ui.open_browser(f"https://www.google.com/search?q={_qp(_query)}")
                    _label = f"{_mode.upper()} — {_query[:38]}" if _query else _mode.upper()
                    self.ui.show_content(_label, outcome.unwrap(r))

            else:
                if self._plugin_registry.has(name):
                    r = await loop.run_in_executor(
                        None,
                        lambda: self._plugin_registry.run(name, args, player=self.ui, session_memory=None)
                    )
                    result = r or "Done."
                else:
                    result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        # How long the tool itself took. A long silence after a tool call is
        # either this number or the model's own time before and after it — and
        # without it the two look identical from the outside.
        print(f"[JARVIS] 📤 {name} ({(time.monotonic() - _t_tool) * 1000:.0f} ms) "
              f"→ {str(result)[:80]}")

        # A tool that declared itself NON_BLOCKING also says when its answer may
        # re-enter the conversation. Without this the model finishes whatever it
        # was saying and then reads the result out on top of it — which, for
        # something like a phone call already ringing, is exactly the noise the
        # non-blocking call was meant to avoid. Tools that declared nothing get
        # the API default and behave as they always have.
        _sched = (self._action_registry.scheduling(name)
                  or self._plugin_registry.scheduling(name))
        _extra = {"scheduling": _sched} if _sched else {}
        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result},
            **_extra
        )

    def _enqueue_audio(self, msg: dict) -> None:
        """Push an audio chunk to out_queue without raising QueueFull."""
        q = self.out_queue
        if q is None:
            return
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(msg)
            except Exception:
                pass

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            gate = self._audio_gate if self._audio_gate is not None else self._voice_isolator
            if msg.get("type") == "voice_stop":
                if gate is not None:
                    res = gate.force_stop()
                    if res.stopped:
                        self._enter_thought_mode()
                        if self._explicit_vad_active or self._engine is not None:
                            await self.session.send_realtime_input(
                                activity_end=types.ActivityEnd()
                            )
                        else:
                            await self.session.send_realtime_input(
                                audio_stream_end=True
                            )
                continue

            if gate is not None:
                res = gate.process_chunk(msg["data"], sr=SEND_SAMPLE_RATE)
                _live = getattr(res, "live_level", None)
                if _live is not None and not res.emit_audio:
                    try:     # Local Ear: waveform follows speech live, not the burst
                        self.ui.set_audio_level(min(1.0, float(_live) * 6.0))
                    except Exception:
                        pass
                if getattr(res, "f0_hz", 0.0) > 0:
                    self._last_user_pitch_hz = float(res.f0_hz)
                if getattr(res, "rms", 0.0) > 0:
                    self._last_user_rms = float(res.rms)
                if res.started:
                    self._interrupted = False
                    if (not getattr(self, "_private_mode", False) and self._screen_watch_enabled
                            and (time.monotonic() - self._last_screen_sent) >= 1.5):
                        asyncio.create_task(self._push_live_screen_frame(force=True))
                    if self._explicit_vad_active or self._engine is not None:
                        await self.session.send_realtime_input(
                            activity_start=types.ActivityStart()
                        )
                if res.emit_audio:
                    self._last_user_speech = time.monotonic()
                    try:
                        pcm_arr = np.frombuffer(res.emit_audio, dtype=np.int16)
                        self.ui.set_audio_level(_pcm_level(pcm_arr))
                    except Exception:
                        pass
                    await self.session.send_realtime_input(
                        audio=types.Blob(
                            data=res.emit_audio,
                            mime_type=msg.get("mime_type", "audio/pcm"),
                        )
                    )
                if res.stopped:
                    self._enter_thought_mode()
                    if self._explicit_vad_active or self._engine is not None:
                        await self.session.send_realtime_input(
                            activity_end=types.ActivityEnd()
                        )
                    else:
                        await self.session.send_realtime_input(
                            audio_stream_end=True
                        )
                    if self._engine is None:
                        self._arm_stall()
                continue

            # Gemini 3.x Live rejects the old realtime_input.media_chunks field
            # (what `media=...` maps to) and closes the socket with a 1007. Send
            # mic / phone PCM through the new `audio` field instead. Queue items
            # are {"data": <bytes>, "mime_type": <str>} from _listen_audio and
            # the phone relay.
            await self.session.send_realtime_input(
                audio=types.Blob(
                    data=msg["data"],
                    mime_type=msg.get("mime_type", "audio/pcm"),
                )
            )

    async def _listen_audio(self):
        print("[JARVIS] 🎤 Mic started")
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
            # ── Voice-fingerprint enrollment tap ─────────────────────────────
            if self._enroll_active:
                mono = indata[:, 0].copy() if getattr(indata, "ndim", 1) > 1 else indata.copy()
                with self._enroll_lock:
                    if self._enroll_active:
                        self._enroll_chunks.append(mono)
                        self._enroll_samples_got += len(mono)
                        if self._enroll_samples_got >= self._enroll_samples_target:
                            self._enroll_active = False
                            self._enroll_done.set()
                try:
                    self.ui.set_audio_level(_pcm_level(indata))
                except Exception:
                    pass
                return

            # ── Wake-word gate ───────────────────────────────────────────────
            # While asleep, the mic audio NEVER goes to Gemini (nothing is
            # streamed, so JARVIS can't respond to speech not addressed to it and
            # nothing leaves the machine). Frames are instead handed to the local
            # detector, which runs its model in ITS OWN thread — the cost here is
            # only a queue push, so the audio path is never slowed. When wake word
            # is off (default) or we're awake, this is a single boolean check.
            if self._wake_enabled and not self._awake:
                det = self._wake_detector
                if det is not None:
                    det.feed(indata)
                return
            with self._speaking_lock:
                jarvis_speaking = self._is_speaking

            # ── Barge-in ─────────────────────────────────────────────────────
            # While JARVIS talks the mic is not streamed, but it is still worth
            # listening to locally: if the user starts speaking, cut the answer
            # short the way a person would stop when interrupted.
            #
            # The whole difficulty is echo — on speakers the mic hears JARVIS.
            # So the test is not "is the mic loud" but "is the mic louder than
            # the echo of what we are playing right now", sustained long enough
            # that a cough or a keystroke cannot trigger it.
            if jarvis_speaking:
                # Local EchoGuard (+ registered VoiceLock when active) detects if
                # the user starts speaking over JARVIS ("stop speak", etc.) and
                # interrupts playback immediately without false echo triggers.
                if self._barge_in_enabled:
                    try:
                        lvl = _pcm_level(indata)
                        if self._echo.is_user_speech(indata, SEND_SAMPLE_RATE, lvl):
                            allow = True
                            iso = self._voice_isolator
                            if iso is not None:
                                allow = iso.matches_pcm(indata.tobytes(), SEND_SAMPLE_RATE)
                            if allow:
                                self._barge_in_streak += 1
                                if self._barge_in_streak >= self._echo.required_blocks:
                                    self._barge_in_streak = 0
                                    loop.call_soon_threadsafe(self._on_voice_barge_in)
                            else:
                                self._barge_in_streak = 0
                        else:
                            self._barge_in_streak = 0
                    except Exception:
                        self._barge_in_streak = 0
                return
            self._barge_in_streak = 0

            # ── Echo tail ────────────────────────────────────────────────────
            # The speaking flag has dropped but the speakers have not finished.
            # Sending this to the model is how an assistant hears itself, decides
            # it was addressed, and answers its own last sentence. The microphone
            # stays OPEN — the guard only drops blocks that are our own voice, so
            # replying the instant it stops still works.
            if self._tail_active():
                try:
                    if not self._echo.is_user_speech(
                            indata, SEND_SAMPLE_RATE, _pcm_level(indata)):
                        return
                    self._tail_until = 0.0      # a real voice ends the tail early
                except Exception:
                    return
            elif self._echo._hist:
                self._echo.reset()

            # ── Push-to-talk ─────────────────────────────────────────────────
            # When it is on the microphone is closed by default and the chord
            # opens it, which is the whole point: nothing leaves the machine
            # unless you are holding the key.
            if self._ptt_enabled and not self._ptt_held:
                return
            
            if not self.ui.muted and not self._phone_active:
                data = indata.tobytes()
                try:
                    data = self._audio_frontend.process_pcm(data)
                except Exception:
                    pass
                loop.call_soon_threadsafe(
                    self._enqueue_audio,
                    {"data": data, "mime_type": "audio/pcm"}
                )
                # When an advanced listening gate (Voice Lock / Smart Turn) is
                # active, _send_realtime drives the HUD waveform exclusively from
                # verified speech so background noise is disregarded visually too.
                if self._audio_gate is None and self._voice_isolator is None:
                    try:
                        self.ui.set_audio_level(_pcm_level(indata))
                    except Exception:
                        pass

        if sd is None:
            print("[JARVIS] ⚠️ sounddevice not available. Mic disabled.")
            while True:
                await asyncio.sleep(1)

        try:
            def _open_mic(dev):
                return sd.InputStream(
                    samplerate=SEND_SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=CHUNK_SIZE,
                    device=dev,
                    callback=callback,
                )

            # Which microphone. resolve() returns None for "system default" and
            # for a saved device that is no longer present — so a headset
            # unplugged since the last run falls back to the built-in mic
            # instead of raising on startup and taking the session with it.
            _mic_name = get_input_device()
            _mic_dev  = audio_devices.resolve(_mic_name, "input")
            if _mic_dev is not None:
                print(f"[JARVIS] 🎤 Input device: {_mic_name}")
            try:
                _mic_stream = _open_mic(_mic_dev)
            except Exception as _e:
                # A device the picker listed but the driver will not open right
                # now — exclusive mode, a webcam already in use, a virtual mic
                # whose source went away. Chosen hardware failing must never
                # mean the assistant cannot hear at all.
                if _mic_dev is None:
                    raise
                print(f"[JARVIS] ⚠️  Mic '{_mic_name}' failed: {_e} — using default")
                self.ui.write_log(
                    f"SYS: Microphone '{_mic_name}' unavailable — using system default."
                )
                _mic_stream = _open_mic(None)

            with _mic_stream:
                print("[JARVIS] 🎤 Mic stream open")
                self._mic_open = True
                try:
                    while True:
                        await asyncio.sleep(0.1)
                finally:
                    self._mic_open = False
        except Exception as e:
            print(f"[JARVIS] ❌ Mic: {e}")
            raise

    async def _flush_pending_vision(self) -> bool:
        """Send a captured frame immediately after its tool response.

        The frame is already in hand by the time `screen_process` returns — the
        capture happened inside the tool call. The old flow still made the model
        speak a turn first and only injected the image on that turn's
        turn_complete, which cost a whole extra round trip AND produced two
        spoken answers: one improvised without the picture, then the real one.
        Sending it here means the model has the tool result and the image before
        it generates anything, so the user gets one answer, sooner.
        """
        if not (self._pending_vision and self.session):
            return False

        import base64 as _b64
        img_b, mime_t, question, angle = self._pending_vision
        self._pending_vision = None
        b64 = _b64.b64encode(img_b).decode("ascii")
        print(f"[Vision] 📤 {len(img_b):,} bytes (angle={angle}) → main session")

        # Label the source. Without it the image arrives carrying nothing but
        # the user's own sentence, and a screenshot of this app — which has a
        # face in the middle of it — got read as a photo of the user. What the
        # label *means* is explained once, in the generated [SELF] block.
        src = ("[IMAGE SOURCE: WEBCAM]" if angle == "camera"
               else "[IMAGE SOURCE: SCREEN CAPTURE]")
        await self.session.send_client_content(
            turns={"role": "user", "parts": [
                {"inline_data": {"mime_type": mime_t, "data": b64}},
                {"text": f"{src}\n\n{question}"},
            ]},
            turn_complete=True,
        )

        if self._vision_cam_active:
            # Camera: stay busy until JARVIS has finished speaking the answer,
            # then close the preview.
            self._vision_cam_active    = False
            self._vision_close_pending = True
        else:
            self._vision_busy = False
        return True

    async def _receive_audio(self):
        print("[JARVIS] 👂 Recv started")
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():
                    # Anything at all from the server proves the return path is
                    # alive, so the stall clock stops here rather than in any of
                    # the branches below — a tool call, a transcript or a bare
                    # keepalive all count.
                    self._clear_stall()

                    # ── Session resumption ───────────────────────────────────
                    # The server sends this periodically. `resumable` goes false
                    # while a turn is mid-flight — replaying a handle from that
                    # moment is what the flag exists to prevent — so only
                    # resumable handles are kept. This is three lines and it is
                    # the entire fix for "every reconnect forgets everything".
                    _sru = getattr(response, "session_resumption_update", None)
                    if _sru is not None:
                        if getattr(_sru, "resumable", False) and getattr(_sru, "new_handle", None):
                            if self._resume_handle is None:
                                print("[JARVIS] 🔗 Session resumption armed")
                            self._resume_handle = _sru.new_handle

                    if response.data:
                        self._server_turn_active = True
                        self._thinking_since = None
                        _song_bleed_only = (
                            self.ui.video_has_sound()
                            and not re.search(
                                r"\b(?:hello|halo|helo|hlo|pause|stop|next|skip|prev|previous|resume|mute|unmute|playlist|search|google|open|kholo|ruko|band|remember|memory|save|close|shutdown|system|pc)\b|हेलो|हलो|हैलो",
                                getattr(self, "_last_user_utterance", ""),
                                re.I,
                            )
                        )
                        if self._interrupted or self._sleep_standby or _song_bleed_only:
                            pass  # discard while interrupted, sleeping, or song playback bleed
                        else:
                            if self._turn_done_event and self._turn_done_event.is_set():
                                self._turn_done_event.clear()
                            # Split into ~50 ms chunks so interrupt() stops audio within 50 ms
                            # (24000 Hz × 2 bytes/sample × 0.05 s = 2400 bytes per slice)
                            _audio_data = response.data
                            _SLICE = 2400
                            for _i in range(0, len(_audio_data), _SLICE):
                                self.audio_in_queue.put_nowait(_audio_data[_i : _i + _SLICE])

                    if response.server_content:
                        sc = response.server_content
                        if getattr(sc, "interrupted", False):
                            self._server_turn_active = False
                            self._interrupted = False
                            in_buf = []
                            out_buf = []
                            self._visemes.reset()

                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_transcript(sc.output_transcription.text)
                            # A turn that involves a tool call passes through
                            # several turn_completes, and the API re-sends the
                            # tail of the transcript across them. Comparing only
                            # against the previous chunk missed that — once
                            # out_buf had been flushed and emptied, the repeat
                            # sailed straight back in, which logged the answer
                            # twice AND made the avatar mouth it twice.
                            if txt and not _is_repeat_chunk(txt, out_buf):
                                out_buf.append(txt)
                                # Hand the words to the mouth as they arrive, so
                                # the avatar can form the consonants the audio
                                # alone cannot show. Pure string work — it adds
                                # nothing measurable to the response path.
                                self._visemes.feed_text(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt:
                                in_buf.append(txt)
                                self._last_user_speech = time.monotonic()
                                _cur_in = " ".join(in_buf)
                                _sw = self._apply_sleep_or_hello_from_text(_cur_in, from_voice=True)
                                _pc = self._detect_private_command(_cur_in)
                                if _pc == "on" and not self._private_mode:
                                    self._set_private_mode(True, notify_session=False)
                                elif _pc == "off" and self._private_mode:
                                    self._set_private_mode(False, notify_session=False)
                                _v_cur = firebase_sync.resolve_unclear_command(_cur_in, min_score=0.78)
                                if _v_cur and _v_cur[0] != getattr(self, "_last_voice_cmd_hint", ""):
                                    self._last_voice_cmd_hint = _v_cur[0]
                                    self.ui.push_feedback(f"🎯 {_v_cur[0][:18]} -> {_v_cur[1][:14]}")
                                    self._inject_session_note(
                                        f"[TRAINED COMMAND MATCH — EXECUTE DIRECTLY NOW: "
                                        f"'{_v_cur[0]}' -> {_v_cur[1]}. Do NOT ask the user to explain again.]"
                                    )

                        if sc.turn_complete:
                            self._last_voice_cmd_hint = ""
                            self._server_turn_active = False
                            self._thinking_since = None
                            if self._turn_done_event:
                                self._turn_done_event.set()

                            # If this turn_complete ends an interrupted response, clear the
                            # flag and skip all further processing for that turn.
                            if self._interrupted:
                                self._interrupted = False
                                in_buf  = []
                                out_buf = []
                                self._visemes.reset()
                                continue

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                _sw_in = self._apply_sleep_or_hello_from_text(full_in, from_voice=True)
                                if self._sleep_standby and _sw_in != "wake":
                                    in_buf = []
                                    out_buf = []
                                    continue
                                _pc_in = self._detect_private_command(full_in)
                                if _pc_in == "on":
                                    self._set_private_mode(True, notify_session=False)
                                elif _pc_in == "off":
                                    self._set_private_mode(False, notify_session=False)
                                self._last_out_logged = ""   # new exchange
                                self._last_turn_user_text = full_in
                                self._last_turn_tools = []
                                self._process_user_memory_and_topic(full_in)
                                self._update_emotion_from_text(full_in, role="user")
                                local_db.record_turn("user", full_in, self._last_emotion)
                                if not self._private_mode:
                                    try:
                                        from core import behavior as _behavior
                                        _behavior.observe(full_in, self._last_emotion)
                                    except Exception:
                                        pass
                                    _v_exp = firebase_sync.extract_explicit_training(full_in)
                                    if _v_exp:
                                        firebase_sync.train_command(_v_exp[0], _v_exp[1], notes="Auto-learned from voice")
                                        self._inject_session_note(
                                            f"[PERMANENTLY TRAINED: '{_v_exp[0]}' -> {_v_exp[1]}. "
                                            "Remember and execute directly on future commands without asking again.]"
                                        )
                                _v_fuzzy = firebase_sync.resolve_unclear_command(full_in)
                                if _v_fuzzy:
                                    _vp, _va, _vs = _v_fuzzy
                                    if _vp.lower() not in full_in.lower() or _vs < 0.96:
                                        self.ui.push_feedback(f"🎯 {_vp[:18]} -> {_va[:14]}")
                                if not self._private_mode:
                                    self.ui.write_log(f"You: {full_in}")
                                    self._session_log.append(f"User: {full_in}")
                                    if self._dashboard:
                                        asyncio.create_task(self._dashboard.broadcast({
                                            "type": "log", "speaker": "user",
                                            "text": full_in,
                                            "ts": datetime.now().isoformat(),
                                        }))
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            # Second line of defence: even if a repeat slips
                            # into a *fresh* buffer after a flush, never log the
                            # same answer (or a tail of it) twice in a row.
                            if full_out and len(full_out) >= _REPEAT_MIN and self._last_out_logged:
                                if full_out in self._last_out_logged:
                                    full_out = ""
                            if full_out:
                                self._last_out_logged = full_out
                                self._update_emotion_from_text(full_out, role="jarvis")
                                local_db.record_turn("jarvis", full_out, self._last_emotion)
                                if not self._private_mode:
                                    try:
                                        firebase_sync.train_from_exchange(
                                            getattr(self, "_last_turn_user_text", ""),
                                            tools_used=getattr(self, "_last_turn_tools", []),
                                            emotion=self._last_emotion,
                                        )
                                    except Exception:
                                        pass
                                    self.ui.write_log(f"{self._asst_name}: {full_out}")
                                    self._session_log.append(f"{self._asst_name}: {full_out}")
                                    if self._dashboard:
                                        asyncio.create_task(self._dashboard.broadcast({
                                            "type": "log", "speaker": "jarvis",
                                            "text": full_out,
                                            "ts": datetime.now().isoformat(),
                                        }))
                            out_buf = []

                            if self._vision_close_pending:
                                # This turn_complete IS the vision answer — close camera + release busy flag
                                self._vision_close_pending = False
                                self._vision_busy = False
                                async def _cam_close():
                                    await asyncio.sleep(2.0)
                                    self.ui.stop_camera_stream()
                                asyncio.create_task(_cam_close())

                            # If the turn completed with no audio (e.g. proactive
                            # audio stayed silent), leave THINKING cleanly.
                            with self._speaking_lock:
                                _sp = self._is_speaking
                            if (
                                not _sp
                                and self.audio_in_queue.empty()
                                and not self.ui.muted
                                and self._awake
                            ):
                                self.ui.set_state("LISTENING")

                    if response.tool_call:
                        self._tool_running = True
                        try:
                            fn_responses = []
                            for fc in response.tool_call.function_calls:
                                print(f"[JARVIS] 📞 {fc.name}")
                                fr = await self._execute_tool(fc)
                                fn_responses.append(fr)
                            await self.session.send_tool_response(
                                function_responses=fn_responses
                            )
                            await self._flush_pending_vision()
                        finally:
                            self._tool_running = False
        except Exception as e:
            print(f"[JARVIS] ❌ Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        if sd is None:
            while True:
                await asyncio.sleep(1)
        print("[JARVIS] 🔊 Play started")

        _spk_name = get_output_device()
        _spk_dev  = audio_devices.resolve(_spk_name, "output")
        if _spk_dev is not None:
            print(f"[JARVIS] 🔊 Output device: {_spk_name}")

        def _open_spk(dev):
            st = sd.RawOutputStream(
                samplerate=RECEIVE_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                device=dev,
            )
            st.start()
            return st

        try:
            stream = _open_spk(_spk_dev)
        except Exception as _e:
            # A chosen output that the host API accepts by name but refuses to
            # open (exclusive mode, wrong sample rate, device asleep) must not
            # cost the user their voice. Fall back to the default and say so.
            if _spk_dev is None:
                raise
            print(f"[JARVIS] ⚠️  Output device '{_spk_name}' failed: {_e} — using default")
            self.ui.write_log(f"SYS: Speaker '{_spk_name}' unavailable — using system default.")
            stream = _open_spk(None)

        # Ask the device how far behind the speakers actually are, rather than
        # assuming. This is what the echo tail is sized from, so a machine with a
        # large audio buffer gets a correspondingly longer guard — and one with a
        # tiny buffer is not penalised with a delay it does not need.
        try:
            lat = float(getattr(stream, "latency", 0.0) or 0.0)
            if 0.0 < lat < 1.0:
                self._out_latency = lat
            print(f"[JARVIS] 🔊 Output latency {self._out_latency*1000:.0f} ms "
                  f"→ echo tail {(self._out_latency + _TAIL_MARGIN)*1000:.0f} ms")
        except Exception:
            pass

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_in_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    if (
                        self._turn_done_event
                        and self._turn_done_event.is_set()
                        and self.audio_in_queue.empty()
                    ):
                        self.set_speaking(False)
                        self._turn_done_event.clear()
                    continue

                self.set_speaking(True)

                # Batch all immediately-available chunks into one write to reduce
                # thread-pool round-trips (was one asyncio.to_thread per 50ms slice).
                # Cap at ~200 ms so interrupt() still stops audio within ~200 ms.
                batch = bytearray(chunk)
                while len(batch) < 9600:   # 9600 bytes ≈ 200 ms at 24 kHz / 16-bit mono
                    try:
                        batch.extend(self.audio_in_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                # Drive the HUD waveform and the avatar's mouth from JARVIS's
                # own voice. The batch is up to 200 ms long, so we hand over a
                # *schedule* of 20 ms viseme frames instead of a single averaged
                # level and let the HUD play it out in step with the audio.
                play_bytes = bytes(batch)
                try:
                    pcm = np.frombuffer(play_bytes, dtype=np.int16)
                    if self._private_mode and pcm.size > 0:
                        pcm = (pcm.astype(np.float32) * 0.20).clip(-32768, 32767).astype(np.int16)
                        play_bytes = pcm.tobytes()
                    hop = _VIS_HOP / RECEIVE_SAMPLE_RATE
                    frames = _pcm_visemes(pcm, sr=RECEIVE_SAMPLE_RATE)
                    # When does this batch become audible? The stream was
                    # started at launch and its callback has been pulling
                    # silence ever since, so the first bytes of a reply reach
                    # the speaker about one callback period later — NOT one
                    # buffer later. `stream.latency` reports the buffer's
                    # capacity, which is how much can be queued ahead, and on
                    # Windows that is commonly 300-500 ms. Anchoring on it put
                    # the entire schedule a buffer late; that is the half second
                    # of lag, and it grew with whatever the device reported.
                    #
                    # After the anchor nothing needs measuring: the device
                    # consumes at exactly realtime, so each batch sounds one
                    # batch-duration after the one before it. The cursor is
                    # re-anchored only when it leaves the range physically
                    # possible — behind `now` means the device drained and this
                    # batch starts a fresh stretch of speech, while further
                    # ahead than the buffer can hold means it has drifted.
                    now = time.time()
                    horizon = self._out_latency + _CURSOR_SLACK
                    if not (now <= self._play_cursor <= now + horizon):
                        self._play_cursor = now + _FIRST_SOUND
                    at = self._play_cursor
                    # Advance by the batch's own duration whether or not it
                    # yielded frames, so a block too short to analyse cannot
                    # shift everything after it out of step with the audio.
                    self._play_cursor += pcm.size / RECEIVE_SAMPLE_RATE
                    if frames:
                        frames = self._visemes.frames(frames, hop)
                        self.ui.push_visemes(frames, hop, at)
                        # Barge-in needs to know what we are playing, not just
                        # how loud: the guard subtracts this from the microphone.
                        self._out_level = max(f[0] for f in frames)
                        self._echo.note_output(pcm, RECEIVE_SAMPLE_RATE,
                                               self._out_level)
                    else:
                        lvl = _pcm_level(pcm)
                        self.ui.set_audio_level(lvl)
                        self._out_level = lvl
                        self._echo.note_output(pcm, RECEIVE_SAMPLE_RATE, lvl)
                except Exception:
                    pass

                try:
                    await asyncio.to_thread(stream.write, play_bytes)
                except (RuntimeError, asyncio.CancelledError):
                    break   # executor shutting down — exit cleanly
        except Exception as e:
            print(f"[JARVIS] ❌ Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()

    # ── Morning briefing ────────────────────────────────────────────────────────

    async def _send_startup_briefing(self) -> None:
        """
        Two-phase briefing optimized for speed:
          Phase 1 — instant greeting (no tools) → speech starts in <1s
          Phase 2 — news pre-fetched in a background thread while Phase 1 plays,
                    delivered as ready text (no Gemini tool-call round-trip) and
                    shown on the UI content panel. Waits for turn_complete event
                    instead of a fixed sleep so there is no unnecessary gap.
        """
        memory   = load_memory()
        identity = memory.get("identity", {})

        def _val(k: str) -> str:
            e = identity.get(k, {})
            return (e.get("value", "") if isinstance(e, dict) else str(e)).strip()

        lang_pref = get_reply_language()
        if lang_pref == "hinglish":
            lang = "natural Hinglish (Hindi + conversational English words)"
        elif lang_pref == "hindi":
            lang = "Hindi"
        elif lang_pref == "english":
            lang = "English"
        else:
            lang = _val("language")
            if not lang or any(w in lang.lower() for w in ("spanish", "portuguese", "french", "espanol")):
                lang = "natural Hinglish (Hindi + conversational English words)"

        name = _val("name")
        time_str = datetime.now().strftime("%H:%M")

        # Start fetching news immediately — runs in parallel while phase 1 plays
        loop = asyncio.get_event_loop()
        news_future = loop.run_in_executor(None, _fetch_news_sync, "top world news today")

        await asyncio.sleep(0.3)
        if not self.session:
            return

        # ── Phase 1: instant greeting ─────────────────────────────────────────
        # The briefing fires before the user has said anything, so the
        # remembered language is the only signal there is. It is a starting
        # point, not a setting: the moment they reply, their language wins.
        lang_clause = (f" Speak this greeting strictly in {lang}. NEVER speak Spanish, Portuguese, or French, then follow the "
                       f"user's own language from their first reply onward."
                       if lang else "")
        name_clause = f" Address the user as {name}." if name else ""

        # Inject last session context if available — pop removes it so it's never repeated
        last = await asyncio.to_thread(pop_last_session)
        session_clause = ""
        if last:
            try:
                _delta = (datetime.now() - datetime.strptime(last["date"], "%Y-%m-%d")).days
                _when  = "earlier today" if _delta == 0 else ("yesterday" if _delta == 1 else f"{_delta} days ago")
            except Exception:
                _when = "last time"
            session_clause = (
                f" Also briefly and naturally mention that {_when}: {last['summary']}"
            )

        p1 = (
            f"Greet the user warmly, mention it is {time_str}, and say you are fetching today's news now.{session_clause} "
            f"Keep it to 2 short sentences max. Do not call any tools.{lang_clause}{name_clause}"
        )

        # Clear the turn-done event so we can wait for Phase 1 to finish
        if self._turn_done_event:
            self._turn_done_event.clear()

        await self.session.send_client_content(
            turns={"role": "user", "parts": [{"text": p1}]},
            turn_complete=True,
        )
        print("[JARVIS] Briefing phase 1 (greeting) sent.")

        # ── Phase 2: fire as soon as Phase 1 audio is done ───────────────────
        async def _deliver_news():
            try:
                lang_str = (f" Speak in {lang} unless the user has since "
                            f"spoken another language, in which case use theirs."
                            if lang else "")

                # Wait for news fetch (already running) and Phase 1 turn-complete
                # in parallel — whichever takes longer determines the wait time
                news_done   = asyncio.wrap_future(news_future)
                turn_waited = False
                if self._turn_done_event:
                    try:
                        await asyncio.wait_for(self._turn_done_event.wait(), timeout=6.0)
                        turn_waited = True
                    except asyncio.TimeoutError:
                        pass

                # Extra buffer: turn_complete fires when Gemini finishes *generating*
                # Phase 1, but audio may still be playing.  Waiting a beat here
                # prevents Phase 2 audio from arriving while Phase 1 is mid-sentence
                # (which sounds like a "repeated first response" to the user).
                if turn_waited:
                    await asyncio.sleep(0.8)
                else:
                    await asyncio.sleep(1.0)

                try:
                    news_text = await asyncio.wait_for(news_done, timeout=8.0)
                except Exception as e:
                    self.ui.write_log(f"SYS: News fetch timed out/failed: {e!r}")
                    news_text = ""

                if not self.session:
                    return

                failed = (not news_text) or news_text.startswith(
                    ("No news found", "Search failed", "Please provide")
                )
                if not failed:
                    # Show on UI content panel immediately
                    self.ui.show_content("NEWS — top world news today", news_text)

                    p2 = (
                        f"[BRIEFING] Here are today's top news headlines:\n{news_text}\n\n"
                        "Pick ONE headline, summarise it in one sentence, then say the full list "
                        f"is displayed on screen. Do not call any tools.{lang_str}"
                    )
                else:
                    self.ui.write_log(
                        f"SYS: News unavailable — backend returned: {news_text[:120]!r}"
                    )
                    p2 = (
                        "News headlines could not be fetched right now. "
                        f"Let the user know briefly.{lang_str}"
                    )

                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": p2}]},
                    turn_complete=True,
                )
                print("[JARVIS] Briefing phase 2 (news) sent.")
            except Exception as e:
                print(f"[Briefing] Phase 2 error: {e}")
                print(f"[JARVIS] Briefing phase 2 failed: {e}")
                self.ui.write_log("SYS: Could not fetch the news for the briefing.")

        asyncio.create_task(_deliver_news())

    # ── Session memory ──────────────────────────────────────────────────────────

    async def _save_session_summary(self) -> None:
        """Summarise the current session in 1-2 sentences and save to long_term.json."""
        log = self._session_log
        if len(log) < 3:          # need at least one exchange to be worth saving
            return
        self._session_log = []    # reset immediately so the next session starts clean

        memory = load_memory()
        pref_lang = get_reply_language()
        if pref_lang == "hinglish":
            lang = "natural Hinglish (Hindi + English)"
        elif pref_lang == "hindi":
            lang = "Hindi"
        else:
            lang_entry = memory.get("identity", {}).get("language", {})
            lang = (lang_entry.get("value", "") if isinstance(lang_entry, dict) else str(lang_entry)).strip()
            if not lang or any(w in lang.lower() for w in ("spanish", "portuguese", "french", "espanol")):
                lang = "natural Hinglish (Hindi + English)"

        convo = "\n".join(log[-40:])   # cap at last 40 turns to stay within token budget
        prompt = (
            f"Summarize this conversation in 1-2 sentences in {lang}. "
            "Focus on what the user accomplished or discussed. "
            "Output ONLY the summary text, nothing else:\n\n" + convo
        )
        try:
            from core import gemini
            summary = await asyncio.to_thread(
                gemini.text, prompt, gemini.SMART, None, 30_000,
            )
            if summary:
                save_session_summary(summary, lang)
        except Exception as e:
            print(f"[Memory] ⚠️ Session summary failed: {e}")

    # ── System monitor ──────────────────────────────────────────────────────────

    async def _run_system_monitor(self) -> None:
        """Background task: voice alerts when metrics exceed thresholds."""
        while True:
            await asyncio.sleep(10)
            try:
                alert = await asyncio.to_thread(self._sys_monitor.check)
            except (RuntimeError, asyncio.CancelledError):
                break
            if not alert or not self.session or not self._awake:
                continue
            # Don't interrupt an active conversation
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking or (time.monotonic() - self._last_user_speech) < 10:
                continue
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": alert}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[Monitor] ⚠️ Could not send alert: {e}")

    # ── Background monitor ──────────────────────────────────────────────────────

    async def _run_background_monitor(self) -> None:
        """Check user-configured topics once per day; speak alerts when new headlines appear."""
        await asyncio.sleep(300)          # wait 5 min after startup before first check
        while True:
            if self.session and self._awake:
                # Don't interrupt if user spoke recently or JARVIS is mid-sentence
                with self._speaking_lock:
                    speaking = self._is_speaking
                recent_speech = (time.monotonic() - self._last_user_speech) < 30
                if not speaking and not recent_speech:
                    try:
                        alerts = await asyncio.to_thread(monitor_check_all)
                        memory = load_memory()
                        lang_e = memory.get("identity", {}).get("language", {})
                        lang   = (lang_e.get("value", "") if isinstance(lang_e, dict) else str(lang_e)).strip() or "English"
                        for alert in alerts:
                            msg = (
                                f"{alert}\n\n"
                                f"Inform the user about this development naturally in {lang}. "
                                "One brief sentence only."
                            )
                            await self.session.send_client_content(
                                turns={"role": "user", "parts": [{"text": msg}]},
                                turn_complete=True,
                            )
                            print("[JARVIS] Monitor alert sent.")
                            await asyncio.sleep(6)   # gap between consecutive alerts
                    except (RuntimeError, asyncio.CancelledError):
                        break
                    except Exception as e:
                        print(f"[Monitor] ⚠️ Background check error: {e}")
            await asyncio.sleep(1800)     # check every 30 minutes

    # ── Continuous screen watch & human-like companion mode ──────────────────

    @staticmethod
    def _screen_diff_score(img_bytes: bytes, prev_thumb):
        """Return (diff_score, cur_thumb) using a tiny 32x18 grayscale diff."""
        import io as _io
        from PIL import Image
        with Image.open(_io.BytesIO(img_bytes)) as im:
            small = im.convert("L").resize((32, 18))
            arr = np.asarray(small, dtype=np.float32)
        if prev_thumb is None or prev_thumb.shape != arr.shape:
            return 255.0, arr
        return float(np.mean(np.abs(arr - prev_thumb))), arr

    async def _push_live_screen_frame(self, force: bool = False) -> bool:
        """Capture and silently push one live screen frame to the session if needed."""
        import base64 as _b64
        from actions.screen_processor import _capture_screen

        if self._sleep_standby or self._private_mode or not self._screen_watch_enabled or not self.session or not self._awake:
            return False
        if self.ui.muted or self._tool_running or self._server_turn_active:
            return False
        if self._engine_caps is not None and not self._engine_caps.get("vision"):
            return False
        with self._speaking_lock:
            if self._is_speaking:
                return False

        if self._screen_push_lock is None:
            self._screen_push_lock = asyncio.Lock()
        if self._screen_push_lock.locked():
            return False

        async with self._screen_push_lock:
            try:
                img_b, _mime = await asyncio.to_thread(_capture_screen)
                self._latest_screen_jpeg = img_b
                diff, thumb = await asyncio.to_thread(
                    self._screen_diff_score, img_b, self._last_screen_thumb
                )
                now = time.monotonic()
                gap = now - self._last_screen_sent
                if force or (diff >= 3.0 and gap >= 2.0) or gap >= 10.0:
                    self._last_screen_thumb = thumb
                    self._last_screen_sent = now
                    if diff >= 12.0:
                        self._screen_changed_at = now
                    b64 = _b64.b64encode(img_b).decode("ascii")
                    await self.session.send_client_content(
                        turns={"role": "user", "parts": [
                            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                            {"text": "[IMAGE SOURCE: LIVE SCREEN WATCH] Live screen update — keep this screen view in context and wait without replying yet."},
                        ]},
                        turn_complete=False,
                    )
                    return True
            except Exception as e:
                print(f"[ScreenWatch] ⚠️ {e}")
        return False

    async def _run_screen_watch(self) -> None:
        """Continuously watches the live screen (~1.5 s cadence) and streams changes."""
        while True:
            await asyncio.sleep(1.5)
            gate = self._audio_gate if self._audio_gate is not None else self._voice_isolator
            if gate is not None and gate.active:
                continue
            await self._push_live_screen_frame(force=False)

    async def _run_proactive_mode(self) -> None:
        """
        Background task: initiates warm human-like companion conversation when
        the user is quiet (or reacts naturally to major screen changes when Screen
        Watch is active), without ever interrupting ongoing speech.
        """
        import base64 as _b64
        import random as _rnd

        while True:
            await asyncio.sleep(7 if self._companion_chat_enabled else 60)

            if (self._sleep_standby or self._private_mode or not self.session
                    or not self._awake or self.ui.muted or self.ui.video_is_playing()):
                continue

            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue
            if self._server_turn_active or self._tool_running or not self.audio_in_queue.empty():
                continue
            gate = self._audio_gate if self._audio_gate is not None else self._voice_isolator
            if gate is not None and gate.active:
                continue

            companion = bool(self._companion_chat_enabled)
            now = time.monotonic()
            # Keep Topic Lock respected: don't interrupt an active focused topic within 2 minutes of speech
            if get_topic_lock_enabled() and (now - self._last_user_speech) < 120.0:
                continue
            screen_recent = bool(
                self._screen_watch_enabled and (now - self._screen_changed_at) < 25.0
            )
            if companion:
                min_sil = 25.0 if screen_recent else float(_rnd.randint(55, 75))
                cooldown = 40.0 if screen_recent else 70.0
            else:
                min_sil = None
                cooldown = None

            last_act = max(self._last_user_speech, self._last_assistant_speech)
            # pyrefly: ignore[unexpected-keyword]
            if not self._proactive.should_trigger(last_act, min_silence=min_sil, cooldown=cooldown):
                continue

            self._proactive.mark_triggered()

            try:
                memory       = await asyncio.to_thread(load_memory)
                monitors     = await asyncio.to_thread(list_monitors)
                recent_turns = self._session_log[-8:] if self._session_log else []
                prompt = self._proactive.build_prompt(
                    memory              = memory,
                    monitors            = monitors or None,
                    recent_turns        = recent_turns or None,
                    # pyrefly: ignore[unexpected-keyword]
                    companion           = companion,
                    # pyrefly: ignore[unexpected-keyword]
                    screen_watch_active = self._screen_watch_enabled,
                    # pyrefly: ignore[unexpected-keyword]
                    screen_changed      = screen_recent,
                    # pyrefly: ignore[unexpected-keyword]
                    reply_language      = get_reply_language(),
                )
                parts = []
                if self._screen_watch_enabled and self._latest_screen_jpeg:
                    b64 = _b64.b64encode(self._latest_screen_jpeg).decode("ascii")
                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
                parts.append({"text": prompt})
                await self.session.send_client_content(
                    turns={"role": "user", "parts": parts},
                    turn_complete=True,
                )
                print("[JARVIS] Companion check-in.")
            except Exception as e:
                print(f"[Proactive] ⚠️ {e}")

    # ── Phone audio relay ────────────────────────────────────────────────────────

    async def _relay_phone_audio(self) -> None:
        """Forward phone mic PCM chunks from dashboard queue into the Gemini Live session."""
        q = self._dashboard._phone_audio_queue
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No audio for 1 s → phone mic inactive, give PC mic back
                self._phone_active = False
                continue
            self._phone_active = True   # phone is streaming — silence PC mic
            with self._speaking_lock:
                speaking = self._is_speaking
            if not speaking and not self.ui.muted:
                try:
                    self.out_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass

    def _on_phone_connected(self) -> None:
        self.ui.write_log("SYS: Phone connected via Phone Link.")
        self.ui.notify_phone_connected()

    # ── dashboard command relay ─────────────────────────────────────────────

    def _try_direct_local_command(self, text: str) -> Optional[str]:
        raw = (text or "").strip().lower()
        if not raw:
            return None

        from memory import reply_store
        from core.phone_bridge import get_bridge

        # 1. Phone screen mirror / view mobile screen on laptop
        if re.search(r"\b(?:phone\s*screen|mobile\s*screen|mirror\s*phone|screen\s*mirror|scrcpy|phone\s*dikhao|mobile\s*dikhao)\b", raw):
            try:
                import subprocess
                subprocess.Popen(["scrcpy.exe"])
                return "Mobile screen mirror shuru ho gaya hai. Aap laptop se phone dekh aur control kar sakte hain."
            except Exception as e:
                return f"Scrcpy start error: {e}"

        # 2. Auto call cut & message rule
        if re.search(r"\b(?:auto\s*(?:call|phone)\s*(?:cut|reject|decline)\s*(?:off|band|disable)|call\s*cut\s*band)\b", raw):
            reply_store.set_setting("auto_decline_calls", "false")
            return "Auto call cut feature band kar diya hai. Ab calls normal aayengi."

        if re.search(r"\b(?:call\s*(?:aaye\s*to\s*)?(?:cut|reject|kaat).*|auto\s*(?:call|phone)\s*(?:cut|reject|decline))\b", raw):
            # Extract custom message if user provided one
            msg = "Main abhi thoda busy hu, kuch der me call karta hu."
            m_msg = re.search(r"(?:bol(?:na|o)?|message|sms|likh(?:na|o)?)\s*(?:ki|:)?\s*[\"']?([^\"'\n]+)[\"']?", raw)
            if m_msg and len(m_msg.group(1).strip()) > 3:
                msg = m_msg.group(1).strip().capitalize()
            reply_store.set_setting("auto_decline_calls", "true")
            reply_store.set_setting("auto_decline_message", msg)
            return f"Auto call cut ON ho gaya hai. Har aane wale call ko cut karke yeh message bhejunga: '{msg}'"

        # 3. Phone hardware actions (Torch, Find phone, Call accept/decline)
        if re.search(r"\b(?:torch|flashlight)\s*(?:on|chalao|jalao)\b", raw):
            get_bridge().send_command("flashlight", {"on": True})
            return "Phone torch on kar di gayi hai."

        if re.search(r"\b(?:torch|flashlight)\s*(?:off|band)\b", raw):
            get_bridge().send_command("flashlight", {"on": False})
            return "Phone torch band kar di gayi hai."

        if re.search(r"\b(?:find\s*(?:my\s*)?phone|phone\s*dhundo|phone\s*kahan\s*hai)\b", raw):
            get_bridge().send_command("find_phone")
            return "Phone me ring bajana shuru kar diya hai."

        if re.search(r"\b(?:stop\s*find|phone\s*mil\s*gaya|ring\s*band)\b", raw):
            get_bridge().send_command("stop_find_phone")
            return "Phone ring band kar di gayi hai."

        if re.search(r"\b(?:cut\s*(?:the\s*)?call|call\s*kaato|call\s*cut)\b", raw):
            get_bridge().send_command("end_call")
            return "Call cut kar di gayi hai."

        if re.search(r"\b(?:pick\s*up|answer\s*call|call\s*uthao)\b", raw):
            get_bridge().send_command("answer_call")
            return "Call receive kar li gayi hai."

        # 4. Lock screen
        if re.search(r"\b(?:lock\s*(?:screen|pc|computer|laptop)?|screen\s*lock)\b", raw):
            try:
                import ctypes, subprocess
                try:
                    ctypes.windll.user32.LockWorkStation()
                except Exception:
                    subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
                return "Screen locked."
            except Exception as e:
                return f"Lock screen error: {e}"

        # 5. Open application
        m_open = re.search(r"^(?:open|launch|start|run|kholo)\s+([a-zA-Z0-9_\-\s\.]+)|([a-zA-Z0-9_\-\s\.]+)\s+(?:kholo|open)$", raw)
        if m_open:
            app = (m_open.group(1) or m_open.group(2) or "").strip()
            norm_map = {
                "crome": "chrome", "google crome": "chrome", "google chrome": "chrome",
                "vsc": "Visual Studio Code", "vscode": "Visual Studio Code", "vs code": "Visual Studio Code",
                "calc": "calculator", "calc.exe": "calculator", "notepad": "notepad",
            }
            target_app = norm_map.get(app.lower(), app)
            try:
                from actions.open_app import open_app
                open_app(parameters={"app_name": target_app})
                return f"Opening {target_app}."
            except Exception as e:
                print(f"[DirectCommand] open_app failed: {e}")

        # 6. Close application
        m_close = re.search(r"^(?:close|quit|exit|band\s*karo|hatao)\s+([a-zA-Z0-9_\-\s\.]+)|([a-zA-Z0-9_\-\s\.]+)\s+(?:band\s*karo|close|quit|exit)$", raw)
        if m_close:
            app = (m_close.group(1) or m_close.group(2) or "").strip()
            norm_map = {
                "crome": "chrome", "google crome": "chrome", "google chrome": "chrome",
                "vsc": "Visual Studio Code", "vscode": "Visual Studio Code", "vs code": "Visual Studio Code",
                "calc": "calculator", "calc.exe": "calculator", "notepad": "notepad",
            }
            target_app = norm_map.get(app.lower(), app)
            try:
                from actions.computer_control import _close_app
                _close_app(target_app)
                return f"Closed {target_app}."
            except Exception as e:
                print(f"[DirectCommand] close_app failed: {e}")

        # 6. Computer settings (volume, mute, brightness, screenshot, etc.)
        try:
            from actions.computer_settings import _detect_action, computer_settings
            det = _detect_action(raw)
            act = det.get("action")
            if act and act in ("volume_up", "volume_down", "volume_set", "mute", "unmute", "toggle_mute", "screenshot", "lock_screen"):
                if act == "lock_screen":
                    import ctypes, subprocess
                    try:
                        ctypes.windll.user32.LockWorkStation()
                    except Exception:
                        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
                    return "Screen locked."
                computer_settings(parameters={"action": act, "value": det.get("value")})
                return f"Action done: {act.replace('_', ' ')}."
        except Exception as e:
            print(f"[DirectCommand] computer_settings failed: {e}")

        return None

    async def _process_dashboard_commands(self) -> None:
        while True:
            try:
                text = await asyncio.wait_for(
                    self._dashboard._command_queue.get(), timeout=0.5
                )
                if not text:
                    continue

                self.ui.write_log(f"[Phone]: {text}")

                # Fast direct execution for common actions (apps, lock, volume)
                direct_msg = self._try_direct_local_command(text)
                if direct_msg:
                    self.ui.write_log(f"JARVIS: {direct_msg}")
                    try:
                        await self._dashboard.broadcast({
                            "type": "log",
                            "speaker": "jarvis",
                            "text": direct_msg,
                            "ts": time.strftime("%H:%M:%S")
                        })
                    except Exception:
                        pass

                # Wake if asleep
                if self._wake_enabled and not self._awake:
                    self.wake(reason="remote command")

                # If it's a general question or needs AI reasoning, send to session
                if not direct_msg:
                    for _ in range(30):
                        if self.session:
                            break
                        await asyncio.sleep(0.1)
                    if self.session:
                        self._on_text_command(text)
                    else:
                        print(f"[Dashboard] AI session not ready: {text}")
                        try:
                            await self._dashboard.broadcast({
                                "type": "log",
                                "speaker": "jarvis",
                                "text": "Command received, but AI session is still connecting.",
                                "ts": time.strftime("%H:%M:%S")
                            })
                        except Exception:
                            pass
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"[Dashboard] Command error: {e}")
                await asyncio.sleep(0.5)

    # ── main loop ───────────────────────────────────────────────────────────

    # ── Which engine holds the conversation ──────────────────────────────────
    # Everything in this section is inert on a default install: `_engine_mode()`
    # reads one key out of the config that is already loaded, gets "gemini", and
    # returns before the providers package is imported. Delete the folder and
    # the import below raises ImportError, which is caught and reported as
    # "providers are not installed" — the assistant then runs exactly as it did
    # before any of this existed.
    def _engine_mode(self) -> str:
        try:
            from memory.config_manager import get_provider_mode
            return get_provider_mode()
        except Exception:
            return "gemini"

    def _resolve_engine_caps(self):
        """What the chosen engine can do — or None for Gemini Live.

        Called BEFORE `_build_config()`, because the tool declarations depend on
        the answer. Building the providers here rather than later is deliberate:
        the registry caches them by configuration, so the objects made now are
        the same ones the engine gets a moment later, and a misconfiguration is
        reported while there is still a Gemini session to fall back to.
        """
        try:
            # "Auto" engine: re-decide brain / ears / voice from the keys saved right now, so a
            # key added or removed since the last start just works. No-op unless it was enabled.
            from core.providers_setup import sync_auto
            if sync_auto():
                self.ui.write_log("SYS: engine settings refreshed from your saved keys.")
        except Exception as e:
            print(f"[JARVIS] auto engine sync skipped: {e}")
        if self._engine_mode() != "pipeline":
            return None
        try:
            from memory.config_manager import get_pipeline, get_provider_config
            from providers import registry

            pipe = get_pipeline()
            llm_id = str(pipe.get("llm") or "").strip()
            if not llm_id:
                raise RuntimeError("no language model is chosen")
            missing = registry.missing_fields(llm_id, get_provider_config(llm_id))
            if missing:
                raise RuntimeError(f"{llm_id} needs: {', '.join(missing)}")
            llm = registry.build(llm_id, get_provider_config(llm_id))

            # Ask the model whether it can see, once, and remember the answer.
            #
            # This decides whether `screen_process` is declared at all, so it
            # has to be settled BEFORE the config is built. Leaving it to the
            # panel's TEST button meant that anybody who never pressed it ran a
            # vision model with its eyes switched off — silently, and for every
            # session. Measured once per model, in about a second, then read
            # from the settings for ever after.
            from memory.config_manager import save_provider_config
            ensure = getattr(llm, "ensure_vision_known", None)
            if ensure is not None:
                try:
                    seen = ensure(lambda v, _id=llm_id: save_provider_config(_id, v))
                    self.ui.write_log(
                        f"SYS: {llm_id} "
                        + ("can see images." if seen else "is text only."))
                except Exception as e:
                    print(f"[JARVIS] vision check skipped: {e}")

            caps = dict(getattr(llm, "caps", {}) or {})
            caps["audio_out"] = bool(str(pipe.get("tts") or "").strip())
            return caps
        except Exception as e:
            # A local engine that will not start must never mean no assistant.
            # Falling back to Live is the whole reason the ladder exists, and
            # the reason is said out loud rather than left in a console.
            self.ui.write_log(f"SYS: local engine unavailable ({e}) — "
                              f"using Gemini Live.")
            print(f"[JARVIS] Pipeline engine unavailable: {e}")
            return None

    @asynccontextmanager
    async def _open_session(self, live_model: str, config):
        """The conversation, from whichever engine is configured.

        Both branches yield an object with the same four methods and the same
        event shapes, which is the entire contract — see providers/base.py. The
        Live branch is unchanged from what it has always been, down to the
        client being rebuilt on every reconnect.
        """
        if self._engine_caps is None:
            _key = _get_api_key()
            if not _key:
                raise RuntimeError(
                    "No Gemini API key is saved, and the other engine could not start. "
                    "Run:  python enable_providers.py --status   (or add a Gemini key).")
            client = genai.Client(
                api_key=_key,
                http_options={"api_version":
                              "v1alpha" if self._enhanced_live else "v1beta"},
            )
            async with client.aio.live.connect(model=live_model,
                                               config=config) as session:
                yield session
            return

        from memory.config_manager import get_pipeline, get_provider_config
        from providers import registry
        from providers.engine_pipeline import PipelineEngine

        pipe = get_pipeline()

        def part(slot):
            pid = str(pipe.get(slot) or "").strip()
            if not pid:
                return None
            # `log` so a provider can say what it is doing — downloading a
            # voice, loading a model — in the activity log rather than only on a
            # console the user may not be watching.
            cfg = dict(get_provider_config(pid))
            cfg["log"] = self.ui.write_log
            return registry.build(pid, cfg)

        engine = PipelineEngine(part("llm"), part("stt"), part("tts"),
                                config=config, log=self.ui.write_log)
        # Without push-to-talk nothing else knows when a sentence ended, so the
        # engine finds it in the audio. With push-to-talk or an advanced
        # listening gate, those drive turn end directly and the RMS VAD stays off.
        engine.auto_turn = (not self._ptt_enabled) and (self._audio_gate is None)
        self._engine = engine
        try:
            yield engine
        finally:
            self._engine = None
            await engine.close()

    async def run(self):
        self._loop = asyncio.get_event_loop()
        self._reconnect_event = asyncio.Event()

        # ── Wire the shared core services to the interface ───────────────────
        # The confirmation gate is useless without a way to ask, and a memory
        # trim is invisible without a way to say so. Both are bound once here
        # rather than passed down through every action signature.
        confirm_gate.bind(
            show = self.ui.show_confirm,
            hide = self.ui.hide_confirm,
            log  = self.ui.write_log,
        )
        set_trim_notifier(self.ui.write_log)

        # Tell the device picker the exact rates the streams open at, from the
        # constants that actually open them — so it can never list a device that
        # cannot be opened at them.
        audio_devices.configure(SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE)

        # Enumerate audio devices off-thread. The settings drawer must never pay
        # for host-API enumeration on the Qt thread.
        audio_devices.prefetch()

        # [REMOTE CONTROL DISABLED — replaced by Phone Link] (old dashboard start-up below is commented out)
        # # Start dashboard (optional — needs: pip install fastapi "uvicorn[standard]" cryptography)
        # try:
            # from dashboard.server import DashboardServer
            # self._dashboard = DashboardServer()
            # self._dashboard.set_connect_callback(self._on_phone_connected)
            # try:
                # # Phone events that deserve the user's attention (an incoming call, a message with a
                # # draft waiting) reach the assistant as if the user had said them, so it can speak up.
                # from core.phone_bridge import get_bridge
                # _loop = asyncio.get_running_loop()
                # _q = self._dashboard._command_queue

                # def _phone_alert(text: str, _q=_q, _loop=_loop) -> None:
                    # _loop.call_soon_threadsafe(
                        # _q.put_nowait, f"[PHONE EVENT — tell the user briefly and naturally, then wait] {text}")
                    # try:
                        # self.ui.write_log(f"PHONE: {text}")
                    # except Exception:
                        # pass
                # get_bridge().alert_handler = _phone_alert
            # except Exception as _e:
                # print(f"[Phone] alerts not wired: {_e}")
            # asyncio.create_task(self._dashboard.serve())
            # # Runs for the whole lifetime, not just inside an active session
            # asyncio.create_task(self._process_dashboard_commands())
        # except Exception as e:
            # print(f"[Dashboard] Disabled: {e}")
            # self._dashboard = None

        # Phone link: a tiny server of its own (core/phone_link.py). Needs no web framework, no
        # certificate and not port 8000, so "the phone cannot connect" no longer depends on those.
        try:
            from core.phone_link import PhoneLinkServer
            from core.phone_bridge import get_bridge
            self._dashboard = PhoneLinkServer(get_bridge())      # same attribute name the relays below use
            self._dashboard.set_connect_callback(self._on_phone_connected)
            self._dashboard.on_contact = lambda who: self.ui.write_log(f"SYS: A phone ({who}) reached the laptop.")
            try:
                # Phone events that deserve the user's attention (an incoming call, a message with a
                # draft waiting) reach the assistant as if the user had said them, so it can speak up.
                _loop = asyncio.get_running_loop()
                _q = self._dashboard._command_queue

                def _phone_alert(text: str, _q=_q, _loop=_loop) -> None:
                    _loop.call_soon_threadsafe(
                        _q.put_nowait, f"[PHONE EVENT — tell the user briefly and naturally, then wait] {text}")
                    try:
                        self.ui.write_log(f"PHONE: {text}")
                    except Exception:
                        pass
                get_bridge().alert_handler = _phone_alert
            except Exception as _e:
                print(f"[Phone] alerts not wired: {_e}")
            asyncio.create_task(self._dashboard.serve())
            # Runs for the whole lifetime, not just inside an active session
            asyncio.create_task(self._process_dashboard_commands())
        except Exception as e:
            print(f"[PhoneLink] Disabled: {e}")
            self._dashboard = None

        while True:
            try:
                self._ensure_executor()
                print("[JARVIS] Connecting...")
                self.ui.set_state("THINKING")
                # Pick the rung to open the conversation on. A model resting
                # off a quota limit is skipped; the name is published back to
                # LIVE_MODEL so plugins follow whatever is actually in use.
                global LIVE_MODEL
                LIVE_MODEL = _gemini.live_model()
                live_model = LIVE_MODEL
                print(f"[JARVIS] Live model: {live_model}")

                _resumed_with = self._resume_handle is not None

                # Decided BEFORE the config is built, because the tool
                # declarations depend on what the engine can serve. On a default
                # install this returns None in a few microseconds and nothing
                # below behaves any differently.
                self._engine_caps = self._resolve_engine_caps()
                config = self._build_config()
                _ric = getattr(config, "realtime_input_config", None)
                _aad = getattr(_ric, "automatic_activity_detection", None)
                _gate = self._audio_gate if self._audio_gate is not None else self._voice_isolator
                self._explicit_vad_active = bool(
                    self._engine_caps is None
                    and self._tuned_live
                    and _gate is not None
                    and getattr(_aad, "disabled", False)
                )
                self._thinking_since = None
                if _gate is not None:
                    _gate.reset()

                # The conversation itself. `_open_session` gives back a Gemini
                # Live session or a local pipeline wearing the same interface —
                # the Live branch still builds a fresh client on every reconnect
                # to avoid stale HTTP state, and still asks for v1alpha when
                # proactive audio is on.
                async with (
                    self._open_session(live_model, config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    # Every user turn arms the stall watchdog. Wrapping the
                    # session once covers all eleven send sites — and every one
                    # added later — where instrumenting them by hand would work
                    # until somebody adds the twelfth and it silently is not
                    # covered. The wrapper is per-session, so it goes away with
                    # the session it belongs to.
                    _orig_send = session.send_client_content

                    async def _send_and_arm(*a, **kw):
                        result = await _orig_send(*a, **kw)
                        if kw.get("turn_complete", True):
                            self._arm_stall()
                        return result

                    session.send_client_content = _send_and_arm

                    self.session          = session
                    self.audio_in_queue   = asyncio.Queue()
                    self.out_queue        = asyncio.Queue(maxsize=200)
                    self._turn_done_event = asyncio.Event()
                    self._rx_deadline     = None

                    # Reset transient state that must not carry over from a previous session
                    self._pending_vision       = None
                    self._vision_cam_active    = False
                    self._vision_close_pending = False
                    self._vision_busy          = False
                    self._vision_last_time     = 0.0
                    self._interrupted          = False

                    print("[JARVIS] Connected.")
                    if _resumed_with:
                        # Say it plainly: the difference between "it reconnected"
                        # and "it reconnected and still knows what we were doing"
                        # is the whole point, and it is invisible otherwise.
                        self.ui.write_log("SYS: Reconnected — conversation restored.")

                    # Wake word: if enabled, come up ASLEEP (mic gated, silent)
                    # until the user says "Hey Jarvis" or taps wake in the UI.
                    if self._wake_enabled:
                        self._ensure_wake_detector()
                        self._awake = False
                        self.ui.set_state("SLEEPING")
                        self.ui.write_log("SYS: JARVIS online — sleeping. Say 'Hey Jarvis' to wake me.")
                    else:
                        self._awake = True
                        self.ui.set_state("LISTENING")
                        self.ui.write_log("SYS: JARVIS online.")

                    # What this machine will not permit, said once, on screen.
                    #
                    # A Wayland session or a macOS without Accessibility cannot
                    # be typed into, clicked or captured — and the APIs report
                    # success regardless. Telling the user at startup turns
                    # "the assistant is broken" into one sentence they can act
                    # on, and it is said once per session rather than on every
                    # attempt.
                    if not self._platform_notes_shown:
                        self._platform_notes_shown = True
                        try:
                            from core import platform_support
                            for _line in platform_support.describe_for_user(
                                    platform_support.detect()):
                                self.ui.write_log(_line)
                        except Exception as _e:
                            print(f"[Platform] capability notes unavailable: {_e}")

                    if self._dashboard:
                        await self._dashboard.broadcast({"type": "status", "state": "active"})

                    self._reconnect_event.clear()  # ignore requests from before this session
                    tg.create_task(self._watch_reconnect())
                    tg.create_task(self._watch_stall())
                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                    tg.create_task(self._run_system_monitor())
                    tg.create_task(self._run_background_monitor())
                    tg.create_task(self._run_screen_watch())
                    tg.create_task(self._run_proactive_mode())
                    tg.create_task(self._run_sleep_watch())
                    if self._dashboard:
                        tg.create_task(self._relay_phone_audio())

                    # Bring a local engine's speech model up before anybody
                    # speaks. Measured: without it the FIRST sentence waits 8.9
                    # seconds and every one after it 1.2 — and eight seconds of
                    # nothing is how a working assistant gets reported broken.
                    if self._engine is not None:
                        tg.create_task(self._engine.warmup())

                    # Morning briefing — fires once per process launch (if enabled).
                    # Skipped in wake-word mode: it comes up asleep, and a briefing
                    # would mean talking while "asleep".
                    if not self._briefing_sent and get_brief_enabled() and self._awake:
                        self._briefing_sent = True
                        tg.create_task(self._send_startup_briefing())

            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except BaseException as e:
                # Catches both Exception and BaseExceptionGroup (Python 3.11+
                # TaskGroup raises BaseExceptionGroup when tasks are cancelled
                # externally, which `except Exception` would miss, letting the
                # exception escape the while-loop and causing asyncio.run() to
                # start shutdown — resulting in "executor after shutdown" errors).
                # Voluntary reconnect (voice change) — not an error. Rebuild the
                # session immediately with no backoff and no scary logs.
                if _is_reconnect_signal(e):
                    print("[JARVIS] Voluntary reconnect requested.")
                    if not _keep_context_of(e):
                        # A deliberate clean slate (voice change) — drop the
                        # handle so the next connect really does start empty.
                        self._resume_handle = None
                    self._conn_backoff = 0
                    continue

                # A resumption handle the server will not accept — expired, or
                # belonging to a session it has since dropped. Without this, the
                # same dead handle would be replayed on every retry and the
                # assistant would never come back at all: the feature meant to
                # survive a reconnect would be the thing preventing one. Drop it
                # once and let the next attempt start clean.
                if _resumed_with and (
                    "resum" in str(e).lower()
                    or "handle" in str(e).lower()
                    or "INVALID_ARGUMENT" in str(e)
                    or "NOT_FOUND" in str(e)
                ):
                    print("[JARVIS] 🔗 Resumption handle rejected — starting a fresh session")
                    self.ui.write_log("SYS: Could not restore the conversation — starting fresh.")
                    self._resume_handle = None
                    self._conn_backoff = 0
                    continue

                err_str = str(e)
                print(f"[JARVIS] Error ({type(e).__name__}): {e}")
                traceback.print_exc()

                # Out of quota, or this model is not available to this key —
                # step down the ladder and reconnect straight away. This is the
                # difference between "JARVIS is quieter today" and "JARVIS does
                # not start today": one model means one daily limit, and the
                # limit always arrives mid-conversation.
                if _gemini.note_live_failure(live_model, err_str):
                    nxt = _gemini.live_model()
                    self.ui.write_log(
                        f"SYS: Switching to {nxt.split('/')[-1]} — the previous "
                        f"model is out of quota."
                        if nxt != live_model else
                        "SYS: Every live model is rate-limited — retrying.")
                    self._conn_backoff = 0 if nxt != live_model else 15
                    if nxt == live_model:
                        await asyncio.sleep(self._conn_backoff)
                    continue

                # Turn-taking / media / thinking knobs rejected by the server
                # (preview API drift) — drop them first, because they are the
                # newest fields and the cheapest to lose. Proactive audio is
                # tried again on the next pass if the error persists.
                if self._tuned_live and (
                    "INVALID_ARGUMENT" in err_str
                    or "Unknown name" in err_str
                    or "unexpected keyword" in err_str
                    or "realtime_input" in err_str.lower()
                    or "activity" in err_str.lower()
                    or "media_resolution" in err_str.lower()
                    or "thinking" in err_str.lower()
                    or "1007" in err_str
                ):
                    self._tuned_live = False
                    print("[JARVIS] Live tuning rejected — reconnecting without it.")
                    continue

                # Proactive audio rejected by the server (preview API drift) —
                # drop it and reconnect with the plain config.
                if self._enhanced_live and (
                    "INVALID_ARGUMENT" in err_str
                    or "proactiv" in err_str.lower()
                    or "Unknown name" in err_str
                    or "unexpected keyword" in err_str
                ):
                    self._enhanced_live = False
                    self.ui.write_log(
                        "SYS: Proactive audio unavailable — reconnecting without it."
                    )
                    continue

                # Invalid API key — stop hammering the API, prompt re-configuration
                if "API key not valid" in err_str or "1007" in err_str:
                    self.ui.write_log("ERR: API key invalid — please re-enter your key.")
                    self.ui.set_state("SLEEPING")
                    self.ui.prompt_reconfig()
                    while not self.ui._win._ready:
                        await asyncio.sleep(1)
                    print("[JARVIS] New API key saved — reconnecting...")
                    _conn_backoff = 3
                    continue

                # Network / timeout errors — log clearly and back off
                is_net_err = any(k in err_str for k in (
                    "TimeoutError", "timed out", "getaddrinfo", "CancelledError",
                    "ConnectionRefusedError", "OSError", "Cannot connect",
                ))
                if is_net_err:
                    _conn_backoff = min(getattr(self, "_conn_backoff", 3) * 2, 60)
                    self._conn_backoff = _conn_backoff
                    self.ui.write_log(
                        f"NET: Connection failed — retrying in {_conn_backoff}s. "
                        "(a VPN may be required)"
                    )
                else:
                    self._conn_backoff = 3
            finally:
                self.session = None
                # Only save if there was a real conversation (≥3 turns)
                if len(self._session_log) >= 3:
                    asyncio.create_task(self._save_session_summary())

            self.set_speaking(False)
            self.ui.set_state("SLEEPING")

            if self._dashboard:
                await self._dashboard.broadcast({"type": "status", "state": "sleeping"})

            delay = getattr(self, "_conn_backoff", 3)
            print(f"[JARVIS] Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

def main():
    ui = JarvisUI("face.png")

    def runner():
        ui.wait_for_api_key()
        jarvis = JarvisLive(ui)
        try:
            asyncio.run(jarvis.run())
        except KeyboardInterrupt:
            print("\n🔴 Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

if __name__ == "__main__":
    main()