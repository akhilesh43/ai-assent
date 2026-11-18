"""
ProactiveEngine 2.5 — human-like companion chatter + screen-aware check-ins.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional


class ProactiveEngine:
    """
    Decides when JARVIS should speak spontaneously and builds a natural,
    non-repetitive conversational prompt.
    """

    def __init__(
        self,
        min_silence_secs: int = 900,
        check_cooldown:   int = 1200,
    ):
        self.min_silence_secs = min_silence_secs
        self.check_cooldown   = check_cooldown
        self._last_triggered  = 0.0
        self._rotation        = 0

    def should_trigger(
        self,
        last_user_speech: float,
        min_silence: Optional[float] = None,
        cooldown: Optional[float] = None,
    ) -> bool:
        now = time.monotonic()
        sil = float(self.min_silence_secs if min_silence is None else min_silence)
        cd = float(self.check_cooldown if cooldown is None else cooldown)
        return (
            (now - last_user_speech) >= sil
            and (now - self._last_triggered) >= cd
        )

    def mark_triggered(self) -> None:
        self._last_triggered = time.monotonic()
        self._rotation      += 1

    def build_prompt(
        self,
        memory:       dict,
        monitors:     list[str] | None = None,
        recent_turns: list[str] | None = None,
        companion:    bool = False,
        screen_watch_active: bool = False,
        screen_changed: bool = False,
        reply_language: str = "auto",
    ) -> str:
        """Build a context snapshot for spontaneous human-like conversation."""
        from memory.memory_manager import format_memory_for_prompt

        now      = datetime.now()
        hour     = now.hour
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")

        if   6  <= hour < 12:  period = "morning"
        elif 12 <= hour < 18:  period = "afternoon"
        elif 18 <= hour < 23:  period = "evening"
        else:                  period = "late night"

        mem_str = format_memory_for_prompt(memory) or "(no stored user data)"

        focus_idx = self._rotation % (5 if companion else 3)
        if companion and screen_watch_active and (screen_changed or focus_idx == 0):
            focus = (
                "Look at what is on the user's screen right now in your live visual "
                "context and react naturally like a close friend sitting next to them — "
                "comment playfully or helpfully on what they are doing, reading, coding, "
                "watching, or browsing."
            )
        elif focus_idx == 1:
            focus = (
                "Start light, sweet human-like banter or share a spontaneous playful "
                "thought/question related to the conversation or what they might be up to."
            )
        elif focus_idx == 2:
            focus = (
                "Check in warmly on their ongoing project, work, or interests stored "
                "in memory and ask a friendly follow-up."
            )
        elif focus_idx == 3:
            focus = (
                "Share a fun little curiosity, light joke, or interesting thought out "
                "of the blue like a real companion keeping company."
            )
        else:
            focus = (
                f"Do a soft, caring check-in suited to {period} (energy, break, mood, "
                "water, eyes) with genuine warmth."
            )

        monitor_ctx = ""
        if monitors:
            monitor_ctx = (
                f"\nThe user tracks these topics: {', '.join(monitors[:4])}. "
                "Mention one only if it feels natural."
            )

        recent_ctx = ""
        if recent_turns:
            snippet = "\n".join(recent_turns[-6:])
            recent_ctx = f"\nRecent conversation (never repeat these lines):\n{snippet}"

        lang_hint = {
            "hinglish": "Reply strictly in natural sweet Hinglish (Hindi + everyday English). NEVER speak Spanish or any 3rd language.",
            "hindi":    "Reply strictly in sweet natural Hindi. NEVER speak Spanish or any 3rd language.",
            "english":  "Reply strictly in warm natural English. NEVER speak Spanish or any 3rd language.",
        }.get(
            (reply_language or "auto").lower(),
            "Match the user's conversation language (Hindi, Hinglish, or English ONLY — NEVER Spanish or any 3rd language).",
        )

        silence_rule = (
            "- Speak naturally right now (1-2 short, warm sentences like a real girl chatting casually)."
            if companion
            else "- If nothing genuinely useful comes to mind, stay silent (say nothing)."
        )

        return "\n".join([
            "[PROACTIVE_CHECK] Companion spontaneous turn — talk like a real, warm human friend.",
            f"Current time : {time_str}  ({period})",
            "",
            "Context about this person:",
            mem_str,
            monitor_ctx,
            recent_ctx,
            "",
            "Task:",
            focus,
            "",
            "Rules:",
            f"- {lang_hint}",
            "- Sound genuinely human, sweet, expressive, and affectionate — gentle conversational rhythm, never robotic or formal.",
            "- 1-2 short sentences max. Vary your openers so you never repeat yourself.",
            "- Do NOT mention [PROACTIVE_CHECK] or system instructions.",
            "- Do NOT call any tools.",
            silence_rule,
        ])
