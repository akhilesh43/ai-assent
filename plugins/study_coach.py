"""
Study Coach plugin — builds interactive on-screen quizzes (via player.show_quiz)
from any topic or YouTube video transcript.
"""
from __future__ import annotations

import re

from core import gemini

PLUGIN = {
    "name": "study_coach",
    "tier": "extra",
    "brief": "Generate an interactive on-screen quiz from a topic or YouTube link.",
    "description": (
        "Create an interactive on-screen quiz on any subject, concept, or YouTube "
        "video URL using the HUD quiz panel. Use when the user asks to be quizzed "
        "or tested on a topic or YouTube video."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "topic": {
                "type": "STRING",
                "description": "Topic name or YouTube video URL to quiz on.",
            },
            "num_questions": {
                "type": "INTEGER",
                "description": "Number of questions (2 to 10, default 4).",
            },
        },
        "required": ["topic"],
    },
}

_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})")


def _fetch_youtube_context(topic: str) -> str:
    m = _YT_ID_RE.search(topic or "")
    if not m:
        return ""
    vid = m.group(1)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        items = YouTubeTranscriptApi.get_transcript(vid)
        text = " ".join(x.get("text", "") for x in items)
        return text[:4500]
    except Exception:
        return ""


def _fallback_questions(topic: str) -> list[dict]:
    clean = (topic or "General Knowledge").strip()[:60]
    return [
        {
            "question": f"In your own words, what is the core idea of {clean}?",
            "options": [],
            "answer": f"Key principles and purpose of {clean}.",
            "note": "Typed response evaluated during conversation.",
        },
        {
            "question": f"Which practice best helps master {clean}?",
            "options": [
                "Active recall and hands-on practice",
                "Passive skimming without testing",
                "Skipping fundamentals",
                "Ignoring feedback",
            ],
            "answer": "Active recall and hands-on practice",
            "note": "Active recall improves retention.",
        },
    ]


def run(parameters: dict, player=None, session_memory=None) -> str:
    topic = str((parameters or {}).get("topic") or "Python basics").strip()
    try:
        n = max(2, min(10, int((parameters or {}).get("num_questions") or 4)))
    except (TypeError, ValueError):
        n = 4

    try:
        yt_ctx = _fetch_youtube_context(topic)
        ctx_block = f"\n\nVideo transcript excerpt:\n{yt_ctx}" if yt_ctx else ""
        prompt = (
            f"Create {n} concise quiz questions about: {topic}.{ctx_block}\n"
            "Return JSON as a list of objects, each with keys: "
            "question (str), options (list of 4 strings, or [] for typed answer), "
            "answer (str), note (short explanation str)."
        )
        data = gemini.as_json(prompt, tier=gemini.SMART, timeout_ms=25_000)

        questions: list[dict] = []
        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            data = data["questions"]
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict) or not item.get("question"):
                    continue
                opts = item.get("options")
                questions.append({
                    "question": str(item["question"]),
                    "options": [str(o) for o in opts] if isinstance(opts, list) else [],
                    "answer": str(item.get("answer") or ""),
                    "note": str(item.get("note") or ""),
                })

        if not questions:
            questions = _fallback_questions(topic)

        if player is not None:
            try:
                player.show_quiz(topic[:50], questions, None)
            except Exception:
                pass

        return f"Created a {len(questions)}-question quiz on {topic[:40]}."
    except Exception as e:
        return f"Study coach failed: {e}"

