"""
Voice-controlled manager for JARVIS advanced listening modes and voice
fingerprint settings.
"""
from __future__ import annotations

from memory.config_manager import (
    LISTENING_MODES,
    clear_voice_fingerprint,
    get_listening_mode,
    get_voice_end_silence_ms,
    get_voice_fingerprint,
    get_voice_isolation_threshold,
    save_listening_mode,
    save_voice_end_silence_ms,
    save_voice_isolation_threshold,
)

PLUGIN = {
    "name": "voice_profiles",
    "tier": "extra",
    "brief": "Inspect or tune Voice Lock, Smart Turn listening mode, and sensitivity.",
    "description": (
        "Check or adjust JARVIS's listening mode (local_ear, smart_vad, voice_lock, continuous), "
        "voice-fingerprint match sensitivity, end-of-speech pause delay, or clear the "
        "saved voice fingerprint."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "enum": ["status", "set_mode", "set_sensitivity", "set_pause_ms", "clear"],
                "description": "Operation to perform.",
            },
            "mode": {
                "type": "STRING",
                "enum": ["local_ear", "smart_vad", "voice_lock", "continuous"],
                "description": "Target listening mode when action='set_mode'.",
            },
            "threshold": {
                "type": "NUMBER",
                "description": "Voice match threshold in [0.35, 0.85] for action='set_sensitivity'.",
            },
            "pause_ms": {
                "type": "INTEGER",
                "description": "End-of-speech silence ms in [120, 2000] for action='set_pause_ms'.",
            },
        },
        "required": ["action"],
    },
}


def run(parameters: dict, player=None, session_memory=None) -> str:
    action = str((parameters or {}).get("action") or "status").strip().lower()
    try:
        if action == "status":
            mode = get_listening_mode()
            fp = get_voice_fingerprint()
            thr = get_voice_isolation_threshold()
            ms = get_voice_end_silence_ms()
            fp_desc = (
                f"registered ({fp.get('voiced_frames', 0)} frames, ~{fp.get('f0_median', 0):.0f} Hz)"
                if isinstance(fp, dict)
                else "not registered"
            )
            msg = (
                f"Listening mode is {mode}; voiceprint is {fp_desc}; "
                f"threshold {thr:.2f}; end pause {ms} ms."
            )
        elif action == "set_mode":
            want = str((parameters or {}).get("mode") or "smart_vad").strip().lower()
            if want == "voice_lock" and get_voice_fingerprint() is None:
                return "No voice fingerprint is registered yet. Open Controls and click Register Voice first."
            saved = save_listening_mode(want)
            msg = f"Listening mode saved as {saved}. Reconnect or restart to apply to the live stream."
        elif action == "set_sensitivity":
            val = float((parameters or {}).get("threshold", 0.58))
            save_voice_isolation_threshold(val)
            msg = f"Voice match threshold set to {get_voice_isolation_threshold():.2f}."
        elif action == "set_pause_ms":
            ms = int((parameters or {}).get("pause_ms", 320))
            save_voice_end_silence_ms(ms)
            msg = f"End-of-speech pause delay set to {get_voice_end_silence_ms()} milliseconds."
        elif action == "clear":
            clear_voice_fingerprint()
            save_listening_mode("smart_vad")
            msg = "Cleared the registered voice fingerprint and switched to Smart Turn."
        else:
            msg = f"Supported listening modes: {', '.join(LISTENING_MODES)}."
    except Exception as e:
        return f"Voice profiles failed: {e}"

    if player is not None:
        try:
            player.write_log(f"JARVIS: {msg}")
        except Exception:
            pass
    return msg

