"""
memory/tasks_store.py — the task board's data, kept in the same local database as everything
JARVIS remembers (config/jarvis_local.db, table `tasks`).

Plain Python, no Qt: the HUD tab, the phone dashboard and the voice plugin all call these same
functions, so a task added by voice shows up on both screens and the other way round.

    add(title, due="tomorrow", priority=1)        -> task dict
    list_tasks(status="open")                     -> [task dict]
    set_status(task_id, "done")                   -> task dict | None
    summary()                                     -> counts, overdue, streak …
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from . import local_db

STATUSES = ("todo", "doing", "done")
PRIORITY_NAMES = {1: "high", 2: "normal", 3: "low"}

_WEEKDAYS = {
    "monday": 0, "mon": 0, "somvar": 0, "tuesday": 1, "tue": 1, "mangalvar": 1,
    "wednesday": 2, "wed": 2, "budhvar": 2, "thursday": 3, "thu": 3, "guruvar": 3, "veervar": 3,
    "friday": 4, "fri": 4, "shukravar": 4, "saturday": 5, "sat": 5, "shanivar": 5,
    "sunday": 6, "sun": 6, "ravivar": 6, "itwar": 6,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_due(text: str, today: Optional[date] = None) -> str:
    """\"tomorrow\" / \"kal\" / \"friday\" / \"in 3 days\" / \"2026-12-01\" → \"YYYY-MM-DD\", or \"\" when
    there is no date in it. Hindi and English, because that is how this assistant is spoken to."""
    s = (text or "").strip().lower()
    if not s:
        return ""
    today = today or date.today()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            return date.fromisoformat(s).isoformat()
        except ValueError:
            return ""
    if s in ("today", "aaj", "aj", "tonight", "aaj raat"):
        return today.isoformat()
    if s in ("tomorrow", "kal", "tmrw", "tomorow"):
        return (today + timedelta(days=1)).isoformat()
    if s in ("day after tomorrow", "parso", "parson"):
        return (today + timedelta(days=2)).isoformat()
    if s in ("next week", "agle hafte", "agle week"):
        return (today + timedelta(days=7)).isoformat()
    m = re.fullmatch(r"in\s+(\d{1,3})\s+(day|days|din|week|weeks|hafte)", s)
    if m:
        n = int(m.group(1)) * (7 if m.group(2).startswith(("week", "hafte")) else 1)
        return (today + timedelta(days=n)).isoformat()
    m = re.fullmatch(r"(?:next\s+|is\s+|agle\s+)?([a-z]+)", s)
    if m and m.group(1) in _WEEKDAYS:
        ahead = (_WEEKDAYS[m.group(1)] - today.weekday()) % 7 or 7
        return (today + timedelta(days=ahead)).isoformat()
    return ""


def _row(r) -> dict[str, Any]:
    d = dict(r)
    d["priority_name"] = PRIORITY_NAMES.get(int(d.get("priority") or 2), "normal")
    d["tags"] = [t for t in str(d.get("tags") or "").split(",") if t]
    due = str(d.get("due") or "")
    d["overdue"] = bool(due) and d.get("status") != "done" and due < date.today().isoformat()
    return d


def add(title: str, *, notes: str = "", priority: int = 2, due: str = "",
        tags: Optional[list[str]] = None, source: str = "user") -> dict[str, Any]:
    title = re.sub(r"\s+", " ", (title or "")).strip()[:240]
    if not title:
        raise ValueError("a task needs a title")
    due_iso = parse_due(due) if due and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", due.strip()) else (due or "").strip()
    pr = priority if priority in PRIORITY_NAMES else 2
    tag_s = ",".join(sorted({re.sub(r"[^\w-]", "", t.lower()) for t in (tags or []) if t.strip()}))
    now = _now()
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            cur = conn.execute(
                "INSERT INTO tasks (title, notes, priority, status, due, tags, source, created_at, updated_at) "
                "VALUES (?, ?, ?, 'todo', ?, ?, ?, ?, ?)",
                (title, notes[:1000], pr, due_iso, tag_s, source[:20], now, now))
            conn.commit()
            return _row(conn.execute("SELECT * FROM tasks WHERE id = ?", (cur.lastrowid,)).fetchone())
        finally:
            conn.close()


def get(task_id: int) -> Optional[dict[str, Any]]:
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            r = conn.execute("SELECT * FROM tasks WHERE id = ?", (int(task_id),)).fetchone()
            return _row(r) if r else None
        finally:
            conn.close()


def list_tasks(status: str = "open", limit: int = 200) -> list[dict[str, Any]]:
    """status: "open" (todo+doing, default) | "todo" | "doing" | "done" | "all".
    Order: overdue first, then priority, then soonest due date, then oldest."""
    where, args = "", []
    if status == "open":
        where = "WHERE status != 'done'"
    elif status in STATUSES:
        where, args = "WHERE status = ?", [status]
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM tasks {where} ORDER BY "
                "CASE WHEN status = 'done' THEN 1 ELSE 0 END, "
                "CASE WHEN due = '' THEN 1 ELSE 0 END, due, priority, id LIMIT ?",
                (*args, max(1, int(limit)))).fetchall()
        finally:
            conn.close()
    out = [_row(r) for r in rows]
    # Overdue first, then by priority — a task a week late outranks one that is merely important.
    out.sort(key=lambda t: (t["status"] == "done", not t["overdue"], t["priority"], t["due"] or "9999", t["id"]))
    return out


def update(task_id: int, **fields) -> Optional[dict[str, Any]]:
    allowed = {"title", "notes", "priority", "status", "due", "tags"}
    sets, args = [], []
    for k, v in fields.items():
        if k not in allowed or v is None:
            continue
        if k == "status":
            if v not in STATUSES:
                continue
        elif k == "priority":
            v = int(v) if int(v) in PRIORITY_NAMES else 2
        elif k == "due":
            v = parse_due(str(v)) if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v).strip() or "x") else str(v).strip()
        elif k == "tags":
            v = ",".join(v) if isinstance(v, (list, tuple)) else str(v)
        elif k == "title":
            v = re.sub(r"\s+", " ", str(v)).strip()[:240]
            if not v:
                continue
        sets.append(f"{k} = ?"); args.append(v)
    if not sets:
        return get(task_id)
    if fields.get("status") == "done":
        sets.append("done_at = ?"); args.append(_now())
    elif fields.get("status") in ("todo", "doing"):
        sets.append("done_at = ''")
    sets.append("updated_at = ?"); args.append(_now())
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", (*args, int(task_id)))
            conn.commit()
        finally:
            conn.close()
    return get(task_id)


def set_status(task_id: int, status: str) -> Optional[dict[str, Any]]:
    return update(task_id, status=status)


def delete(task_id: int) -> bool:
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            n = conn.execute("DELETE FROM tasks WHERE id = ?", (int(task_id),)).rowcount
            conn.commit()
            return n > 0
        finally:
            conn.close()


def find(query: str, status: str = "open") -> Optional[dict[str, Any]]:
    """Best open task for words spoken aloud (\"mark the report one done\"): a number is an id,
    otherwise the task sharing the most words with the query. None when nothing matches."""
    q = (query or "").strip().lower()
    if not q:
        return None
    m = re.fullmatch(r"#?(\d+)", q)
    if m:
        t = get(int(m.group(1)))
        return t if t else None
    words = {w for w in re.findall(r"\w+", q) if len(w) > 2}
    best, best_score = None, 0
    for t in list_tasks(status):
        hay = set(re.findall(r"\w+", (t["title"] + " " + " ".join(t["tags"])).lower()))
        score = len(words & hay) * 2 + (1 if q in t["title"].lower() else 0) * 3
        if score > best_score:
            best, best_score = t, score
    return best


def clear_done(older_than_days: int = 0) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max(0, older_than_days))).isoformat()
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            n = conn.execute("DELETE FROM tasks WHERE status = 'done' AND done_at <= ?", (cutoff,)).rowcount
            conn.commit()
            return n
        finally:
            conn.close()


def summary() -> dict[str, Any]:
    """Everything a dashboard card needs about the board, in one read."""
    today = date.today().isoformat()
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT status, priority, due, done_at, created_at FROM tasks").fetchall()]
        finally:
            conn.close()
    open_ = [r for r in rows if r["status"] != "done"]
    done = [r for r in rows if r["status"] == "done"]
    local_today = datetime.now().date()
    done_days = set()
    done_today = 0
    for r in done:
        try:
            d = datetime.fromisoformat(r["done_at"]).astimezone().date()
        except Exception:
            continue
        done_days.add(d)
        done_today += (d == local_today)
    streak, d = 0, local_today if local_today in done_days else local_today - timedelta(days=1)
    while d in done_days:
        streak += 1; d -= timedelta(days=1)
    week_ago = local_today - timedelta(days=6)
    per_day = {(week_ago + timedelta(days=i)).isoformat(): 0 for i in range(7)}
    for r in done:
        try:
            dd = datetime.fromisoformat(r["done_at"]).astimezone().date().isoformat()
        except Exception:
            continue
        if dd in per_day:
            per_day[dd] += 1
    total = len(rows)
    return {
        "total": total,
        "open": len(open_),
        "todo": sum(r["status"] == "todo" for r in rows),
        "doing": sum(r["status"] == "doing" for r in rows),
        "done": len(done),
        "overdue": sum(1 for r in open_ if r["due"] and r["due"] < today),
        "due_today": sum(1 for r in open_ if r["due"] == today),
        "high_priority": sum(1 for r in open_ if r["priority"] == 1),
        "done_today": done_today,
        "streak_days": streak,
        "completion_pct": round(100 * len(done) / total) if total else 0,
        "done_per_day": per_day,
    }
