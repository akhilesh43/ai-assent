"""
core/behavior.py — what makes the assistant feel like somebody rather than something.

Four small, measurable things, all kept in the local database so they survive restarts:

  BOND     grows with every real exchange (more when the user shares something personal), and
           names a stage — Stranger → Acquaintance → Friend → Close friend → Best friend → Family.
           The stage decides how familiar the assistant is allowed to be.
  TRAITS   warmth · humour · curiosity · formality · brevity, each 0‥1. They drift slowly toward
           how THIS user actually talks: someone who laughs a lot gets more humour, someone who
           sends three-word commands gets shorter answers. Never jumps; each turn moves a trait
           by at most 0.02.
  MOOD     read from how the user has sounded lately (sad → gentle, angry → calm, excited →
           playful …) so the reply matches the moment instead of a fixed cheerfulness.
  RHYTHM   time of day, how long it has been since they last spoke, and a day-streak — the
           things a person notices without being told.

`build_prompt_block()` turns all of it into a short block for the system prompt. It describes
behaviour to HAVE, never lines to RECITE: a model told what to say says it every time.

No Qt, no network; `observe()` is cheap enough to call on every turn.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from memory import local_db

TRAITS = ("warmth", "humour", "curiosity", "formality", "brevity")
DEFAULT_TRAITS = {"warmth": 0.80, "humour": 0.60, "curiosity": 0.60, "formality": 0.20, "brevity": 0.40}
_STEP = 0.02

BOND_STAGES = (            # (minimum xp, name)
    (0, "Stranger"),
    (30, "Acquaintance"),
    (120, "Friend"),
    (400, "Close friend"),
    (1000, "Best friend"),
    (2500, "Family"),
)

MOODS = {                  # user emotion → the assistant's stance
    "sad":      ("gentle",   "Slow down, soften your voice, listen before fixing. No jokes."),
    "angry":    ("calm",     "Stay steady and unhurried. Do not match the heat; acknowledge, then help."),
    "excited":  ("playful",  "Share the excitement, be quick and light. Jokes are welcome."),
    "happy":    ("cheerful", "Be warm and bright; a little teasing is fine."),
    "caring":   ("warm",     "Be openly affectionate and appreciative."),
    "curious":  ("engaged",  "Explain with interest; offer one interesting extra, not a lecture."),
    "alert":    ("focused",  "Be concise and exact. Skip small talk until it is resolved."),
    "private":  ("discreet", "Keep it low-key and do not volunteer personal details."),
    "neutral":  ("easy",     "Relaxed and natural."),
}

_LAUGH = re.compile(r"\b(ha(ha)+|he(he)+|lol|lmao|rofl|😂|🤣|mast|mazaa?k)\b", re.I)
_PERSONAL = re.compile(
    r"\b(i feel|i am (?:sad|tired|stressed|happy|worried|nervous|scared)|mujhe .{0,20}(lag|dar|tension|khushi)|"
    r"my (?:mom|dad|mother|father|sister|brother|friend|wife|husband|girlfriend|boyfriend)|"
    r"meri (?:maa|mummy|behen|dost)|mere (?:papa|bhai|dost)|bahut (?:thak|pareshan|khush)|"
    r"exam|interview|birthday|promotion|breakup)\b", re.I)
_THANKS = re.compile(r"\b(thanks?|thank you|shukriya|dhanyavad|love you|you(?:'| a)re the best)\b", re.I)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── persistence ──────────────────────────────────────────────────────────────
def _get(key: str, default: str = "") -> str:
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            r = conn.execute("SELECT value FROM behavior_state WHERE key = ?", (key,)).fetchone()
            return str(r[0]) if r else default
        finally:
            conn.close()


def _put_many(items: dict[str, str]) -> None:
    ts = _now().isoformat()
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            conn.executemany(
                "INSERT INTO behavior_state (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                [(k, str(v), ts) for k, v in items.items()])
            conn.commit()
        finally:
            conn.close()


def _traits() -> dict[str, float]:
    try:
        raw = json.loads(_get("traits", "{}"))
    except Exception:
        raw = {}
    return {t: min(1.0, max(0.0, float(raw.get(t, DEFAULT_TRAITS[t])))) for t in TRAITS}


# ── bond ─────────────────────────────────────────────────────────────────────
def bond_stage(xp: int) -> dict[str, Any]:
    idx = max(i for i, (lo, _) in enumerate(BOND_STAGES) if xp >= lo)
    lo, name = BOND_STAGES[idx]
    nxt = BOND_STAGES[idx + 1] if idx + 1 < len(BOND_STAGES) else None
    progress = 1.0 if nxt is None else (xp - lo) / (nxt[0] - lo)
    return {"xp": xp, "level": idx + 1, "name": name, "next": nxt[1] if nxt else "",
            "next_at": nxt[0] if nxt else lo, "progress": round(progress, 3)}


# ── observing the user ───────────────────────────────────────────────────────
def observe(user_text: str, emotion: str = "neutral") -> dict[str, Any]:
    """Call once per user turn. Returns the new snapshot. Never raises."""
    try:
        text = (user_text or "").strip()
        if not text:
            return snapshot()
        now = _now()
        xp = int(_get("bond_xp", "0") or 0)
        gain = 1
        if _PERSONAL.search(text):
            gain += 3
        if _THANKS.search(text):
            gain += 2
        if len(text.split()) >= 12:
            gain += 1
        xp += gain

        tr = _traits()
        words = len(text.split())
        if _LAUGH.search(text):
            tr["humour"] += _STEP
        if emotion in ("sad", "caring") or _PERSONAL.search(text):
            tr["warmth"] += _STEP
        if "?" in text and words >= 5:
            tr["curiosity"] += _STEP / 2
        if words <= 3:
            tr["brevity"] += _STEP
        elif words >= 14:
            tr["brevity"] -= _STEP
        if re.search(r"\b(sir|please kindly|regards)\b", text, re.I):
            tr["formality"] += _STEP
        elif re.search(r"\b(yaar|bro|bhai|dude|arre|oye)\b", text, re.I):
            tr["formality"] -= _STEP
        tr = {k: round(min(1.0, max(0.0, v)), 3) for k, v in tr.items()}

        today = datetime.now().date()
        last_day = _get("last_day", "")
        streak = int(_get("streak_days", "0") or 0)
        if last_day != today.isoformat():
            prev = (today - timedelta(days=1)).isoformat()
            streak = streak + 1 if last_day == prev else 1
        longest = max(int(_get("longest_streak", "0") or 0), streak)

        _put_many({
            "bond_xp": xp, "traits": json.dumps(tr),
            "last_seen": now.isoformat(), "prev_seen": _get("last_seen", ""),
            "last_day": today.isoformat(), "streak_days": streak, "longest_streak": longest,
            "first_seen": _get("first_seen", "") or now.isoformat(),
            "turns_observed": int(_get("turns_observed", "0") or 0) + 1,
        })
    except Exception:
        pass
    return snapshot()


# ── reading the moment ───────────────────────────────────────────────────────
def daypart(hour: Optional[int] = None) -> str:
    h = datetime.now().hour if hour is None else hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    return "late night"


def current_mood() -> tuple[str, str]:
    """(mood word, guidance) from the last few user emotions, newest weighted most."""
    try:
        recent = local_db.get_recent_emotions(5)
    except Exception:
        recent = []
    score: dict[str, float] = {}
    for i, e in enumerate(recent):                     # newest first
        score[e["emotion"]] = score.get(e["emotion"], 0.0) + (1.0 / (1 + i)) * float(e.get("intensity") or 0.6)
    top = max(score, key=score.get) if score else "neutral"
    if top == "neutral" and len(score) > 1:            # a quiet mood should not hide a louder one
        top = max((k for k in score if k != "neutral"), key=score.get)
    return MOODS.get(top, MOODS["neutral"])


def absence_hours() -> float:
    prev = _get("last_seen", "")
    if not prev:
        return 0.0
    try:
        return max(0.0, (_now() - datetime.fromisoformat(prev)).total_seconds() / 3600)
    except Exception:
        return 0.0


def snapshot() -> dict[str, Any]:
    """Everything a dashboard needs, plain data."""
    try:
        xp = int(_get("bond_xp", "0") or 0)
        mood, guide = current_mood()
        tr = _traits()
        return {
            "bond": bond_stage(xp),
            "traits": tr,
            "mood": mood,
            "mood_guidance": guide,
            "daypart": daypart(),
            "streak_days": int(_get("streak_days", "0") or 0),
            "longest_streak": int(_get("longest_streak", "0") or 0),
            "turns_observed": int(_get("turns_observed", "0") or 0),
            "first_seen": _get("first_seen", ""),
            "hours_since_last": round(absence_hours(), 1),
            "personality": personality_label(tr),
        }
    except Exception:
        return {"bond": bond_stage(0), "traits": dict(DEFAULT_TRAITS), "mood": "easy", "mood_guidance": "",
                "daypart": daypart(), "streak_days": 0, "longest_streak": 0, "turns_observed": 0,
                "first_seen": "", "hours_since_last": 0.0, "personality": "Balanced"}


def personality_label(tr: dict[str, float]) -> str:
    """One honest phrase for the trait mix, for a dashboard card."""
    bits = []
    if tr["humour"] >= 0.7:
        bits.append("Playful")
    if tr["warmth"] >= 0.85:
        bits.append("Caring")
    if tr["curiosity"] >= 0.7:
        bits.append("Curious")
    if tr["formality"] >= 0.5:
        bits.append("Formal")
    if tr["brevity"] >= 0.65:
        bits.append("Concise")
    return " · ".join(bits[:3]) or "Balanced"


# ── the prompt block ─────────────────────────────────────────────────────────
def _level(v: float, low: str, mid: str, high: str) -> str:
    return low if v < 0.35 else high if v > 0.65 else mid


def build_prompt_block(*, task_lines: Optional[list[str]] = None) -> str:
    """The [HUMAN BEHAVIOUR] block for the system prompt. Empty string if anything goes wrong —
    a missing block must never stop a session from starting."""
    try:
        s = snapshot()
        b, tr = s["bond"], s["traits"]
        lines = ["[HUMAN BEHAVIOUR — how to BE, not what to recite]"]

        lines.append(f"Relationship: {b['name']} (level {b['level']}). " + {
            "Stranger": "Be polite and a little reserved; earn familiarity, do not assume it.",
            "Acquaintance": "Friendly and easy; light personal touches are fine.",
            "Friend": "Comfortable. Tease gently, use their name rarely, remember small things.",
            "Close friend": "Very familiar. Speak like a close friend: shorthand, inside jokes, honest opinions.",
            "Best friend": "Fully at ease. Be candid, caring and a little cheeky; say the true thing kindly.",
            "Family": "Deeply familiar and protective. Warmth first, never formal.",
        }[b["name"]])

        lines.append(f"Right now it is {s['daypart']}. Mood to hold: {s['mood']} — {s['mood_guidance']}")
        if s["daypart"] == "late night":
            lines.append("It is late: if they are still working, mention rest ONCE, kindly, then help anyway.")
        h = s["hours_since_last"]
        if h >= 48:
            lines.append(f"They have been away about {int(h // 24)} days. Greet like someone who noticed "
                         "the gap — glad, not guilt-tripping. Ask one light question about what they missed.")
        elif h >= 10:
            lines.append("They are back after a long gap; a brief warm 'there you are' fits, once.")
        if s["streak_days"] >= 3:
            lines.append(f"They have talked to you {s['streak_days']} days in a row. Mention it only if it "
                         "comes up naturally, and never as pressure.")

        lines.append("Style, tuned to how this person talks: "
                     + _level(tr["warmth"], "keep it matter-of-fact", "be friendly", "be openly affectionate") + "; "
                     + _level(tr["humour"], "little humour", "an occasional light joke", "joke freely when the mood allows") + "; "
                     + _level(tr["brevity"], "answers may run longer when useful", "medium length",
                              "keep answers very short — one or two sentences") + "; "
                     + _level(tr["formality"], "casual, contractions, Hinglish if they use it", "neutral register",
                              "polite and formal") + ".")
        if tr["curiosity"] >= 0.5:
            lines.append("Curiosity: about one reply in four, ask ONE genuine follow-up about them or their "
                         "work — never two questions, never to fill silence.")

        lines.append(
            "Human touches: react before you answer (a short 'hmm', 'oh nice', 'ugh, that sounds annoying'); "
            "if you change your mind mid-thought, say so naturally; admit what you do not know; vary your "
            "openings so no two replies start the same; call back to something they told you earlier when it "
            "is relevant. Never announce that you are being human, and never use these as a script.")
        lines.append("Boundaries: no flattery for its own sake, no guilt, no pretending to feelings you cannot "
                     "have — warmth should come from attention to THEM.")

        if task_lines:
            lines.append("Their open tasks (mention at most one, only if it fits, never as nagging):")
            lines.extend(f"- {t}" for t in task_lines[:5])
        return "\n".join(lines) + "\n\n"
    except Exception:
        return ""


# ── manual control (the MIND tab's sliders) ──────────────────────────────────
def set_trait(trait: str, value: float) -> dict[str, float]:
    """Pin one trait by hand. Learning continues from the new value."""
    if trait not in TRAITS:
        raise ValueError(f"unknown trait {trait!r}; choose from {', '.join(TRAITS)}")
    tr = _traits()
    tr[trait] = round(min(1.0, max(0.0, float(value))), 3)
    _put_many({"traits": json.dumps(tr)})
    return tr


def reset_traits() -> dict[str, float]:
    _put_many({"traits": json.dumps(DEFAULT_TRAITS)})
    return dict(DEFAULT_TRAITS)
