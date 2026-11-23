"""
Persistent local SQLite knowledge & emotion memory (``config/jarvis_local.db``).

Zero extra dependencies (built-in ``sqlite3``). Stores learned user facts,
searchable conversation history, emotional trend logs, and voice-security
events locally so JARVIS never starts from zero after a restart and understands
the user's emotional pattern over time.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import sys
import threading
from typing import Any


def _default_db_path() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).resolve().parent.parent
    return base / "config" / "jarvis_local.db"


DB_PATH: Path = _default_db_path()
_LOCK = threading.Lock()
_INITIALIZED_PATH: Path | None = None

SUPPORTED_EMOTIONS = (
    "neutral",
    "happy",
    "excited",
    "sad",
    "angry",
    "caring",
    "curious",
    "alert",
    "private",
)

_EMOTION_LEXICON: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sad", (
        "sad", "udaas", "udas", "dukhi", "dukh", "rona", "ro raha", "ro rahi",
        "bura lag", "akela", "akeli", "lonely", "upset", "hurt", "depress",
        "broken", "tension", "pareshan", "thak gaya", "thak gayi", "tired",
        "cry", "sorry", "miss you", "afsos", "mushkil",
    )),
    ("angry", (
        "angry", "gussa", "bakwas", "irritat", "annoy", "frustrat", "hate",
        "chid", "dimag kharab", "shut up", "bevkuf", "stupid", "pagal",
        "khabardar", "threat", "warning",
    )),
    ("excited", (
        "excited", "yay", "woohoo", "amazing", "awesome", "kya baat", "shandar",
        "zabardast", "dhamaka", "party", "congrats", "jeeta", "jeet gaya",
    )),
    ("happy", (
        "happy", "khush", "maza", "mast", "badiya", "badhiya", "great", "good",
        "haha", "hehe", "lol", "funny", "nice", "cool", "lovely", "wonderful",
    )),
    ("caring", (
        "love you", "thank", "shukriya", "dhanyavad", "take care", "khayal",
        "sweet", "cute", "pyaar", "pyari", "dost", "friend",
    )),
)

_FACT_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\b(?:mera naam|my name is)\s+([a-zA-Z\u0900-\u097F][a-zA-Z0-9\u0900-\u097F_-]{1,28}(?:\s+(?!hai\b|hoon\b|is\b|and\b|or\b)[a-zA-Z\u0900-\u097F]{1,20})?)", re.I),
     "identity", "user_name"),
    (re.compile(r"\b(?:mujhe|i)\s+(?:pasand hai|like|love|prefer)\s+([^\n.,!?]{2,80})", re.I),
     "preference", "likes"),
    (re.compile(r"\b(?:remember that|yaad rakhna(?: ki)?|yaad rakh(?: ki)?|note that|memory me(?:in)?\s+(?:save|add|likho|update|dal|rakh)(?:\s+karo)?(?:\s+ki)?|save to memory(?:\s+that)?)\s*[:,-]?\s*([^\n.!?]{3,160})", re.I),
     "notes", "remembered"),
    (re.compile(r"\b(?:hamesha|always|har bar|every time)\s+([^\n.!?]{4,140})", re.I),
     "preferences", "always_rule"),
    (re.compile(r"\b(?:kabhi mat|never|don't ever|mat kiya karo)\s+([^\n.!?]{4,140})", re.I),
     "preferences", "never_rule"),
    (re.compile(r"\b(?:i work (?:on|as)|main .*kaam karta|mera project)\s+([^\n.!?]{2,100})", re.I),
     "projects", "current_work"),
    (re.compile(r"\b(?:my (?:folder|workspace|repo|path) is|mera folder)\s+([^\n.!?]{2,120})", re.I),
     "projects", "workspace_path"),
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    global _INITIALIZED_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if _INITIALIZED_PATH != DB_PATH:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                key        TEXT PRIMARY KEY,
                category   TEXT NOT NULL,
                value      TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                hits       INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                role    TEXT NOT NULL,
                text    TEXT NOT NULL,
                emotion TEXT NOT NULL DEFAULT 'neutral',
                ts      TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS emotions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                emotion   TEXT NOT NULL,
                intensity REAL NOT NULL DEFAULT 0.6,
                text      TEXT NOT NULL DEFAULT '',
                pitch_hz  REAL NOT NULL DEFAULT 0.0,
                rms       REAL NOT NULL DEFAULT 0.0,
                ts        TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS security_events (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                event  TEXT NOT NULL,
                score  REAL NOT NULL DEFAULT 0.0,
                f0     REAL NOT NULL DEFAULT 0.0,
                detail TEXT NOT NULL DEFAULT '',
                ts     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS topics (
                id         INTEGER PRIMARY KEY CHECK (id = 1),
                topic      TEXT NOT NULL DEFAULT '',
                locked     INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trained_commands (
                slug          TEXT PRIMARY KEY,
                phrase        TEXT NOT NULL,
                target_action TEXT NOT NULL,
                aliases_json  TEXT NOT NULL DEFAULT '[]',
                notes         TEXT NOT NULL DEFAULT '',
                hits          INTEGER NOT NULL DEFAULT 1,
                updated_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT NOT NULL,
                notes      TEXT NOT NULL DEFAULT '',
                priority   INTEGER NOT NULL DEFAULT 2,
                status     TEXT NOT NULL DEFAULT 'todo',
                due        TEXT NOT NULL DEFAULT '',
                tags       TEXT NOT NULL DEFAULT '',
                source     TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                done_at    TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS reply_examples (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                contact   TEXT NOT NULL DEFAULT '',
                app       TEXT NOT NULL DEFAULT '',
                incoming  TEXT NOT NULL,
                reply     TEXT NOT NULL,
                source    TEXT NOT NULL DEFAULT 'taught',
                ts        TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS contact_rules (
                contact    TEXT PRIMARY KEY,
                mode       TEXT NOT NULL DEFAULT 'ask',
                note       TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS behavior_state (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS interaction_patterns (
                slug        TEXT PRIMARY KEY,
                phrase      TEXT NOT NULL,
                action_hint TEXT NOT NULL,
                emotion     TEXT NOT NULL DEFAULT 'neutral',
                hits        INTEGER NOT NULL DEFAULT 1,
                updated_at  TEXT NOT NULL
            );
            """
        )
        conn.commit()
        _INITIALIZED_PATH = DB_PATH
    return conn


def detect_emotion(text: str = "", pitch_hz: float = 0.0, rms: float = 0.0) -> tuple[str, float]:
    """Fast local emotion classifier across English/Hindi/Hinglish text + acoustics."""
    t = (text or "").strip().lower()
    if not t:
        if pitch_hz > 260.0 and rms > 0.08:
            return "excited", 0.65
        if 0.0 < rms < 0.015:
            return "sad", 0.45
        return "neutral", 0.3

    best_emo = "neutral"
    best_hits = 0
    for emo, words in _EMOTION_LEXICON:
        hits = sum(1 for w in words if w in t)
        if hits > best_hits:
            best_hits = hits
            best_emo = emo

    if best_hits > 0:
        intensity = min(1.0, 0.55 + 0.15 * best_hits)
        if "!" in t and best_emo in ("happy", "excited", "angry"):
            intensity = min(1.0, intensity + 0.15)
        return best_emo, intensity

    if "?" in t or any(w in t for w in ("kya", "kaise", "kyu", "kyon", "what ", "how ", "why ")):
        return "curious", 0.55
    return "neutral", 0.35


def save_fact(key: str, value: str, category: str = "general", confidence: float = 1.0) -> None:
    k = (key or "").strip().lower()[:80]
    v = (value or "").strip()[:400]
    cat = (category or "general").strip().lower()[:40]
    if not k or not v:
        return
    if "account" in k or "cuenta" in k or "account" in v.lower() or "cuenta" in v.lower():
        return
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO facts(key, category, value, confidence, hits, updated_at)
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(key) DO UPDATE SET
                    category   = excluded.category,
                    value      = excluded.value,
                    confidence = MAX(facts.confidence, excluded.confidence),
                    hits       = facts.hits + 1,
                    updated_at = excluded.updated_at
                """,
                (k, cat, v, float(confidence), _now_iso()),
            )
            conn.commit()
        finally:
            conn.close()
    if DB_PATH == _default_db_path():
        try:
            from memory.memory_manager import update_memory
            lt_cat = cat if cat in ("identity", "preferences", "projects", "relationships", "wishes", "notes") else (
                "preferences" if cat in ("preference", "rule", "workflow") else ("projects" if cat == "work" else "notes")
            )
            update_memory({lt_cat: {k: {"value": v}}})
        except Exception:
            pass


def learn_from_text(text: str) -> list[tuple[str, str]]:
    """Automatically extract durable user preferences/facts from conversation."""
    t = (text or "").strip()
    if not t or t.startswith("["):
        return []
    learned: list[tuple[str, str]] = []
    for pat, cat, key_prefix in _FACT_PATTERNS:
        m = pat.search(t)
        if m:
            val = m.group(1).strip(" .,:;\"'")
            if val:
                k = f"{key_prefix}:{val[:24].lower()}" if key_prefix != "user_name" else "user_name"
                save_fact(k, val, category=cat)
                learned.append((k, val))
    return learned


def record_turn(role: str, text: str, emotion: str = "neutral") -> None:
    txt = (text or "").strip()
    if not txt:
        return
    if "account" in txt.lower() or "cuenta" in txt.lower() or "número" in txt.lower():
        return
    r = (role or "user").strip().lower()
    emo = emotion if emotion in SUPPORTED_EMOTIONS else "neutral"
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO conversations(role, text, emotion, ts) VALUES (?, ?, ?, ?)",
                (r, txt[:1500], emo, _now_iso()),
            )
            conn.commit()
        finally:
            conn.close()


def record_emotion(
    emotion: str,
    intensity: float = 0.6,
    text: str = "",
    pitch_hz: float = 0.0,
    rms: float = 0.0,
) -> None:
    emo = emotion if emotion in SUPPORTED_EMOTIONS else "neutral"
    if emo == "neutral":
        return
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO emotions(emotion, intensity, text, pitch_hz, rms, ts) VALUES (?, ?, ?, ?, ?, ?)",
                (emo, float(intensity), (text or "")[:300], float(pitch_hz), float(rms), _now_iso()),
            )
            conn.commit()
        finally:
            conn.close()


def record_security_event(event: str, score: float = 0.0, f0: float = 0.0, detail: str = "") -> None:
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO security_events(event, score, f0, detail, ts) VALUES (?, ?, ?, ?, ?)",
                (event[:64], float(score), float(f0), (detail or "")[:300], _now_iso()),
            )
            conn.commit()
        finally:
            conn.close()


def get_recent_emotions(limit: int = 5) -> list[dict[str, Any]]:
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT emotion, intensity, text, ts FROM emotions ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_top_facts(limit: int = 12) -> list[dict[str, Any]]:
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT key, category, value, hits, updated_at FROM facts "
                "ORDER BY hits DESC, updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_recent_turns(limit: int = 6) -> list[dict[str, Any]]:
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT role, text, emotion, ts FROM conversations ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            out = [dict(r) for r in rows]
            out.reverse()
            return out
        finally:
            conn.close()


def set_active_topic(topic: str, locked: bool | None = None) -> None:
    t = re.sub(r"\s+", " ", (topic or "").strip())[:100]
    if not t:
        return
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT locked FROM topics WHERE id = 1").fetchone()
            lock_val = (1 if (locked if locked is not None else (row["locked"] if row else 1)) else 0)
            conn.execute(
                """
                INSERT INTO topics(id, topic, locked, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    topic      = excluded.topic,
                    locked     = excluded.locked,
                    updated_at = excluded.updated_at
                """,
                (t, lock_val, _now_iso()),
            )
            conn.commit()
        finally:
            conn.close()


def get_active_topic() -> tuple[str, bool]:
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT topic, locked FROM topics WHERE id = 1").fetchone()
            if not row:
                return "", True
            return str(row["topic"] or ""), bool(row["locked"])
        finally:
            conn.close()


def detect_topic_from_text(text: str) -> str:
    """Extract a concise topic phrase from user utterances."""
    t = (text or "").strip()
    if not t or t.startswith("["):
        return ""
    explicit = re.search(
        r"\b(?:topic(?:\s+is|\s*:\s*)|baat karte hain|let'?s talk about|about)\s+([^\n.,!?]{3,60})",
        t,
        re.I,
    )
    if explicit:
        return explicit.group(1).strip(" .,:;\"'")
    words = [w for w in re.findall(r"[a-zA-Z0-9\u0900-\u097F_-]+", t) if len(w) >= 3]
    stop = {
        "the", "and", "for", "with", "that", "this", "what", "how", "why", "you", "are",
        "karo", "kya", "hai", "hain", "mein", "mera", "meri", "mere", "hum", "tum",
        "please", "jarvis", "can", "could", "would", "should", "write", "open",
    }
    key_words = [w for w in words if w.lower() not in stop]
    if len(key_words) >= 2:
        return " ".join(key_words[:4])
    return ""


def save_trained_command(
    phrase: str,
    target_action: str,
    aliases: list[str] | None = None,
    notes: str = "",
) -> None:
    p = re.sub(r"\s+", " ", (phrase or "").strip())[:140]
    act = (target_action or "").strip()[:300]
    if not p or not act:
        return
    slug = re.sub(r"[^a-z0-9\u0900-\u097F]+", "_", p.lower()).strip("_")[:64] or "cmd"
    al_list = [str(a).strip() for a in (aliases or []) if a and str(a).strip()]
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT aliases_json FROM trained_commands WHERE slug = ?", (slug,)).fetchone()
            if row:
                try:
                    existing = json.loads(row["aliases_json"] or "[]")
                    for x in existing:
                        if x not in al_list:
                            al_list.append(x)
                except Exception:
                    pass
            conn.execute(
                """
                INSERT INTO trained_commands(slug, phrase, target_action, aliases_json, notes, hits, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    phrase        = excluded.phrase,
                    target_action = excluded.target_action,
                    aliases_json  = excluded.aliases_json,
                    notes         = CASE WHEN excluded.notes != '' THEN excluded.notes ELSE trained_commands.notes END,
                    hits          = trained_commands.hits + 1,
                    updated_at    = excluded.updated_at
                """,
                (slug, p, act, json.dumps(al_list, ensure_ascii=False), (notes or "")[:240], _now_iso()),
            )
            conn.commit()
        finally:
            conn.close()


def get_trained_commands(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT slug, phrase, target_action, aliases_json, notes, hits, updated_at "
                "FROM trained_commands ORDER BY hits DESC, updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    d["aliases"] = json.loads(d.pop("aliases_json", "[]") or "[]")
                except Exception:
                    d["aliases"] = []
                out.append(d)
            return out
        finally:
            conn.close()


def record_interaction_pattern(phrase: str, action_hint: str, emotion: str = "neutral") -> None:
    p = re.sub(r"\s+", " ", (phrase or "").strip())[:120]
    ah = (action_hint or "").strip()[:180]
    if not p or not ah:
        return
    slug = re.sub(r"[^a-z0-9\u0900-\u097F]+", "_", p.lower()).strip("_")[:64] or "pat"
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO interaction_patterns(slug, phrase, action_hint, emotion, hits, updated_at)
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    action_hint = excluded.action_hint,
                    emotion     = excluded.emotion,
                    hits        = interaction_patterns.hits + 1,
                    updated_at  = excluded.updated_at
                """,
                (slug, p, ah, emotion[:24], _now_iso()),
            )
            conn.commit()
        finally:
            conn.close()


def get_top_patterns(limit: int = 10) -> list[dict[str, Any]]:
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT phrase, action_hint, emotion, hits FROM interaction_patterns "
                "ORDER BY hits DESC, updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_stats() -> dict[str, Any]:
    with _LOCK:
        conn = _connect()
        try:
            f_cnt = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
            c_cnt = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
            e_cnt = int(conn.execute("SELECT COUNT(*) FROM emotions").fetchone()[0])
            tc_cnt = int(conn.execute("SELECT COUNT(*) FROM trained_commands").fetchone()[0])
            t_row = conn.execute("SELECT topic, locked FROM topics WHERE id = 1").fetchone()
            return {
                "facts": f_cnt,
                "turns": c_cnt,
                "emotions": e_cnt,
                "trained_commands": tc_cnt,
                "topic": str(t_row["topic"] or "") if t_row else "",
                "topic_locked": bool(t_row["locked"]) if t_row else True,
            }
        finally:
            conn.close()


def build_db_context(fact_limit: int = 48, turn_limit: int = 20) -> str:
    """Format persistent learned knowledge, trained commands, active topic & emotions."""
    facts = get_top_facts(fact_limit)
    emos = get_recent_emotions(6)
    turns = get_recent_turns(turn_limit)
    cmds = get_trained_commands(40)
    pats = get_top_patterns(16)
    topic, locked = get_active_topic()
    if not facts and not emos and not turns and not topic and not cmds:
        return ""

    lines = ["[LOCAL PERSISTENT KNOWLEDGE & FIREBASE TRAINED MEMORY]"]
    if topic:
        lock_note = " (STAY FOCUSED ON THIS TOPIC — DO NOT DRIFT)" if locked else ""
        lines.append(f"Current active topic: {topic}{lock_note}")
    if cmds:
        lines.append(
            "Trained commands, workflows & voice shortcuts (PERMANENTLY LEARNED — "
            "when the user triggers any of these, execute its action DIRECTLY without asking again):"
        )
        for c in cmds:
            al = f" (aliases: {', '.join(c['aliases'][:3])})" if c.get("aliases") else ""
            nt = f" [{c['notes']}]" if c.get("notes") else ""
            lines.append(f"- '{c['phrase']}'{al} -> {c['target_action']}{nt}")
    if facts:
        facts_clean = [f for f in facts if "account" not in f["key"].lower() and "cuenta" not in f["key"].lower() and "account" not in f["value"].lower()]
        if facts_clean:
            lines.append("Learned facts & user rules (already known across sessions — NEVER ask again):")
            for f in facts_clean:
                lines.append(f"- [{f['category']}] {f['key']}: {f['value']}")
    if pats:
        pats_clean = [p for p in pats if "cuenta" not in p["phrase"].lower() and "account" not in p["phrase"].lower()]
        if pats_clean:
            lines.append("Learned communication habits from past turns:")
            for p in pats_clean[:10]:
                lines.append(f"- '{p['phrase']}' usually uses [{p['action_hint']}]")
    if emos:
        summary = ", ".join(f"{e['emotion']}({e['intensity']:.1f})" for e in emos)
        lines.append(f"Recent user emotion trend: {summary}")
    if turns:
        turns_clean = [tr for tr in turns if "account" not in tr["text"].lower() and "cuenta" not in tr["text"].lower() and "número" not in tr["text"].lower()]
        if turns_clean:
            lines.append("Recent saved conversation continuity:")
            for tr in turns_clean:
                lines.append(f"- {tr['role']} ({tr['emotion']}): {tr['text'][:240]}")
    return "\n".join(lines) + "\n\n"
