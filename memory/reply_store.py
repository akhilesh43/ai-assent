"""
memory/reply_store.py — how the owner answers messages, learned from the owner.

Three kinds of evidence, in order of how much they are trusted:

    taught    the owner said "reply like this to this kind of message"
    edited    a draft was changed before sending — the strongest signal there is, because it
              shows exactly what the draft got wrong
    approved  a draft was sent as written

Replies the assistant sent ON ITS OWN ("auto") are deliberately never stored as examples:
learning from its own output would lock in whatever it guessed first.

Per-contact rules say how each person is handled: `auto` (answer without asking), `ask`
(draft, wait for approval), `ignore` (never touch).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

from . import local_db

SOURCES = ("taught", "edited", "approved")
MODES = ("auto", "ask", "ignore", "off")
_WEIGHT = {"edited": 3, "taught": 2, "approved": 1}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm_contact(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())[:80]


def add_example(incoming: str, reply: str, *, contact: str = "", app: str = "",
                source: str = "taught") -> bool:
    incoming, reply = (incoming or "").strip()[:500], (reply or "").strip()[:500]
    if source not in SOURCES or not incoming or not reply:
        return False
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            dup = conn.execute("SELECT id FROM reply_examples WHERE contact = ? AND incoming = ? AND reply = ?",
                               (norm_contact(contact), incoming, reply)).fetchone()
            if dup:
                return False
            conn.execute("INSERT INTO reply_examples (contact, app, incoming, reply, source, ts) "
                         "VALUES (?, ?, ?, ?, ?, ?)",
                         (norm_contact(contact), app[:60], incoming, reply, source, _now()))
            # a bounded memory: keep the newest 2000
            conn.execute("DELETE FROM reply_examples WHERE id NOT IN "
                         "(SELECT id FROM reply_examples ORDER BY id DESC LIMIT 2000)")
            conn.commit()
            return True
        finally:
            conn.close()


def examples_for(contact: str, limit: int = 10) -> list[dict[str, Any]]:
    """Up to `limit` examples: this contact's first, then the owner's general style. Edited
    examples outrank taught ones, which outrank approved; within a tier, newest first."""
    c = norm_contact(contact)
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM reply_examples ORDER BY id DESC LIMIT 400").fetchall()]
        finally:
            conn.close()
    rows.sort(key=lambda r: (r["contact"] != c, -_WEIGHT.get(r["source"], 0), -r["id"]))
    own = [r for r in rows if r["contact"] == c][: max(1, limit // 2 + 1)]
    rest = [r for r in rows if r["contact"] != c][: limit]
    return (own + rest)[:limit]


def stats() -> dict[str, Any]:
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            by = {r[0]: r[1] for r in conn.execute("SELECT source, COUNT(*) FROM reply_examples GROUP BY source")}
            contacts = conn.execute("SELECT COUNT(DISTINCT contact) FROM reply_examples WHERE contact != ''").fetchone()[0]
        finally:
            conn.close()
    return {"examples": sum(by.values()), "by_source": by, "contacts": contacts}


# ── rules ────────────────────────────────────────────────────────────────────
def set_rule(contact: str, mode: str, note: str = "") -> bool:
    c = norm_contact(contact)
    if not c or mode not in MODES:
        return False
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            conn.execute("INSERT INTO contact_rules (contact, mode, note, updated_at) VALUES (?, ?, ?, ?) "
                         "ON CONFLICT(contact) DO UPDATE SET mode = excluded.mode, note = excluded.note, "
                         "updated_at = excluded.updated_at", (c, mode, note[:200], _now()))
            conn.commit()
            return True
        finally:
            conn.close()


def get_rule(contact: str) -> Optional[dict[str, Any]]:
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            r = conn.execute("SELECT * FROM contact_rules WHERE contact = ?", (norm_contact(contact),)).fetchone()
            return dict(r) if r else None
        finally:
            conn.close()


def list_rules() -> list[dict[str, Any]]:
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM contact_rules ORDER BY contact").fetchall()]
        finally:
            conn.close()


def delete_rule(contact: str) -> bool:
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            n = conn.execute("DELETE FROM contact_rules WHERE contact = ?", (norm_contact(contact),)).rowcount
            conn.commit()
            return n > 0
        finally:
            conn.close()


# ── small settings kept beside the rest of the behaviour state ───────────────
def get_setting(key: str, default: str = "") -> str:
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            r = conn.execute("SELECT value FROM behavior_state WHERE key = ?", (f"phone.{key}",)).fetchone()
            return str(r[0]) if r else default
        finally:
            conn.close()


def set_setting(key: str, value: str) -> None:
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            conn.execute("INSERT INTO behavior_state (key, value, updated_at) VALUES (?, ?, ?) "
                         "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                         (f"phone.{key}", str(value), _now()))
            conn.commit()
        finally:
            conn.close()
