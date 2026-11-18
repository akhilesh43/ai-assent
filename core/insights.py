"""
core/insights.py — ONE place that gathers what the dashboards show.

The HUD's DASHBOARD / BRAIN / MIND tabs and the phone dashboard's API all call `snapshot()`, so
the two screens can never disagree about a number. Plain data in, plain data out: no Qt, no web
framework. Every section is gathered on its own and a failure in one yields `{"error": ...}` for
that section only — a broken task table must not blank the CPU graph.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

BRAIN_LEVELS = (          # (minimum score, name)
    (0, "Newborn"), (10, "Learning"), (50, "Sharp"), (150, "Expert"), (400, "Genius"), (1000, "Sage"),
)


def _section(fn) -> dict[str, Any]:
    try:
        return fn()
    except Exception as e:                      # one broken source must not take the rest down
        return {"error": f"{type(e).__name__}: {e}"[:160]}


# ── system ───────────────────────────────────────────────────────────────────
def system() -> dict[str, Any]:
    import psutil
    vm = psutil.virtual_memory()
    out = {
        "cpu": round(psutil.cpu_percent(interval=None), 1),
        "mem": round(vm.percent, 1),
        "mem_used_gb": round(vm.used / 2**30, 1),
        "mem_total_gb": round(vm.total / 2**30, 1),
        "processes": len(psutil.pids()),
        "uptime_s": int(datetime.now().timestamp() - psutil.boot_time()),
    }
    try:
        b = psutil.sensors_battery()
        if b is not None:
            out["battery"] = round(b.percent); out["plugged"] = bool(b.power_plugged)
    except Exception:
        pass
    try:
        out["disk"] = round(psutil.disk_usage("/").percent, 1)
    except Exception:
        pass
    return out


# ── brain ────────────────────────────────────────────────────────────────────
def brain_level(score: int) -> dict[str, Any]:
    idx = max(i for i, (lo, _) in enumerate(BRAIN_LEVELS) if score >= lo)
    lo, name = BRAIN_LEVELS[idx]
    nxt = BRAIN_LEVELS[idx + 1] if idx + 1 < len(BRAIN_LEVELS) else None
    prog = 1.0 if nxt is None else (score - lo) / (nxt[0] - lo)
    return {"level": idx + 1, "name": name, "score": score,
            "next": nxt[1] if nxt else "", "next_at": nxt[0] if nxt else lo, "progress": round(prog, 3)}


def brain() -> dict[str, Any]:
    from memory import local_db, memory_manager as mm
    mem = mm.load_memory()
    cats = {}
    chars = 0
    for cat, items in mem.items():
        if not isinstance(items, dict):
            continue
        n = 0
        for entry in items.values():
            v = mm._entry_value(entry)
            if v:
                n += 1
                chars += len(v)
        if n:
            cats[cat] = n
    entries = sum(cats.values())
    db = local_db.get_stats()
    score = entries + db["facts"] + 3 * db["trained_commands"] + db["turns"] // 25
    try:
        from pathlib import Path
        plugins = sum(1 for f in (Path(__file__).resolve().parent.parent / "plugins").glob("*.py")
                      if not f.name.startswith("_"))
    except Exception:
        plugins = 0
    return {
        "level": brain_level(score),
        "memories": entries,
        "by_category": cats,
        "facts": db["facts"],
        "conversation_turns": db["turns"],
        "emotions_logged": db["emotions"],
        "trained_commands": db["trained_commands"],
        "topic": db["topic"],
        "storage_chars": chars,
        "storage_capacity": mm.MEMORY_MAX_CHARS,
        "storage_pct": round(100 * chars / mm.MEMORY_MAX_CHARS, 2),
        "prompt_budget": {"core_chars": mm.PROMPT_CORE_CHARS, "index_chars": mm.PROMPT_INDEX_CHARS,
                          "per_category": mm.PROMPT_MAX_PER_CATEGORY},
        "top_facts": [{"key": f["key"], "value": f["value"][:80], "category": f["category"], "hits": f["hits"]}
                      for f in local_db.get_top_facts(8)],
        "top_patterns": [{"phrase": p["phrase"][:50], "hint": p["action_hint"], "hits": p["hits"]}
                         for p in local_db.get_top_patterns(5)],
        "plugins": plugins,
    }


# ── activity ─────────────────────────────────────────────────────────────────
def activity() -> dict[str, Any]:
    from memory import local_db
    since = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    with local_db._LOCK:
        conn = local_db._connect()
        try:
            turns = conn.execute("SELECT role, ts FROM conversations WHERE ts >= ?", (since,)).fetchall()
            emos = conn.execute("SELECT emotion FROM emotions ORDER BY id DESC LIMIT 300").fetchall()
            hours = conn.execute("SELECT ts FROM conversations WHERE role = 'user' ORDER BY id DESC LIMIT 1500").fetchall()
        finally:
            conn.close()
    today = datetime.now().date()
    days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    per = {d.isoformat(): {"user": 0, "assistant": 0} for d in days}
    for role, ts in turns:
        try:
            d = datetime.fromisoformat(ts).astimezone().date().isoformat()
        except Exception:
            continue
        if d in per:
            per[d]["user" if role == "user" else "assistant"] += 1
    by_hour = Counter()
    for (ts,) in hours:
        try:
            by_hour[datetime.fromisoformat(ts).astimezone().hour] += 1
        except Exception:
            pass
    peak = by_hour.most_common(1)[0][0] if by_hour else None
    return {
        "days": [{"day": d.isoformat(), "label": d.strftime("%a"), **per[d.isoformat()]} for d in days],
        "emotions": dict(Counter(e[0] for e in emos).most_common()),
        "peak_hour": peak,
        "total_week": sum(v["user"] + v["assistant"] for v in per.values()),
    }


# ── the one call ─────────────────────────────────────────────────────────────
def snapshot(include_system: bool = True) -> dict[str, Any]:
    from memory import tasks_store
    from core import behavior
    def _phone() -> dict[str, Any]:
        from core.phone_bridge import get_bridge
        s = get_bridge().snapshot()
        return {k: s[k] for k in ("connected", "device", "battery", "charging", "call", "mode")} | {
            "pending": len(s["pending"]), "learned": s["learning"]["examples"]}

    out = {
        "phone": _section(_phone),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "brain": _section(brain),
        "tasks": _section(tasks_store.summary),
        "mind": _section(behavior.snapshot),
        "activity": _section(activity),
    }
    if include_system:
        out["system"] = _section(system)
    return out


def task_lines_for_prompt(limit: int = 5) -> list[str]:
    """Short lines describing the most pressing open tasks, for the system prompt."""
    try:
        from memory import tasks_store
        lines = []
        for t in tasks_store.list_tasks("open", limit=limit):
            tag = " (OVERDUE)" if t["overdue"] else (f" (due {t['due']})" if t["due"] else "")
            lines.append(f"{t['title']}{tag}")
        return lines
    except Exception:
        return []


def to_json(snap: dict[str, Any]) -> str:
    return json.dumps(snap, default=str)
