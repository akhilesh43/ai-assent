"""
One-time setup for "friend mode": run this once, then restart JARVIS.

    python apply_friend_mode.py

It does two things, through the project's own config functions (never by
editing config/api_keys.json by hand):

  1. Turn tuning ON so she WAITS for you to finish speaking instead of
     answering at the first pause.
  2. Picks a female voice (Kore, one of the 5 Gemini Live voices).

To undo: set the voice back in the settings drawer and run
    python apply_friend_mode.py --off
"""
import sys

from memory.config_manager import save_turn_tuning, save_voice

if "--off" in sys.argv:
    save_turn_tuning({"enabled": False})
    print("Friend mode off: turn tuning disabled (voice unchanged).")
else:
    save_turn_tuning({
        "enabled": True,
        "silence_ms": 1100,          # wait 1.1 s of quiet before your turn ends
        "prefix_ms": 300,            # keep the start of your sentence
        "end_sensitivity": "low",    # slow to decide you have stopped
        "start_sensitivity": "default",
    })
    save_voice("Kore")               # female voice; "Aoede" is another option
    print("Friend mode on: she waits for you to finish, and the voice is Kore.")
    print("Restart JARVIS for it to take effect.")
