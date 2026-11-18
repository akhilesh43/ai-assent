"""
Task board — add, list, finish, reschedule and delete tasks by voice.

The same tasks are shown on the HUD's TASKS tab and on the phone dashboard, because all three go
through memory/tasks_store.py.
"""
from __future__ import annotations

PLUGIN = {
    "name": "task_board",
    "description": (
        "The user's to-do list. Use for 'add a task / reminder-to-do / mujhe ... karna hai, task me daalo', "
        "'what's on my list / mere tasks kya hain', 'mark X done / X ho gaya', 'start X', "
        "'move X to friday', 'delete task X', 'clear finished tasks'. "
        "NOT for timed alarms or 'remind me in 10 minutes' — use the reminder tool for those."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING",
                       "description": "add | list | done | start | reopen | reschedule | delete | clear_done | summary"},
            "title": {"type": "STRING", "description": "Task wording (add), or words that identify an existing task"},
            "due": {"type": "STRING", "description": "today, tomorrow/kal, parso, a weekday, 'in 3 days' or YYYY-MM-DD"},
            "priority": {"type": "STRING", "description": "high | normal | low"},
            "tags": {"type": "STRING", "description": "comma separated, optional"},
            "notes": {"type": "STRING", "description": "extra detail, optional"},
        },
        "required": ["action"],
    },
}

_PRIORITY = {"high": 1, "urgent": 1, "important": 1, "normal": 2, "medium": 2, "low": 3}


def _say(t: dict) -> str:
    bits = [t["title"]]
    if t["overdue"]:
        bits.append("overdue")
    elif t["due"]:
        bits.append(f"due {t['due']}")
    if t["priority"] == 1:
        bits.append("high priority")
    return ", ".join(bits)


def run(parameters: dict, player=None, session_memory=None) -> str:
    try:
        from memory import tasks_store as ts
        p = parameters or {}
        action = str(p.get("action") or "").strip().lower()
        title = str(p.get("title") or "").strip()

        if action == "add":
            if not title:
                return "What should I put on the list?"
            t = ts.add(title, due=str(p.get("due") or ""),
                       priority=_PRIORITY.get(str(p.get("priority") or "").lower(), 2),
                       tags=[x for x in str(p.get("tags") or "").split(",") if x.strip()],
                       notes=str(p.get("notes") or ""), source="voice")
            msg = f"Added: {t['title']}" + (f", due {t['due']}" if t["due"] else "") + "."
            if p.get("due") and not t["due"]:
                msg += f" I couldn't read '{p['due']}' as a date, so there's no deadline on it."
        elif action in ("list", "summary", ""):
            s = ts.summary()
            open_ = ts.list_tasks("open", limit=5)
            if not open_:
                msg = "Your list is empty." if not s["total"] else "Everything is done. Nothing open."
            else:
                msg = (f"You have {s['open']} open"
                       + (f", {s['overdue']} overdue" if s["overdue"] else "")
                       + (f", {s['due_today']} due today" if s["due_today"] else "") + ". Top: "
                       + "; ".join(_say(t) for t in open_[:3]) + ".")
        elif action in ("done", "start", "reopen", "reschedule", "delete"):
            t = ts.find(title)
            if not t:
                return f"I couldn't find a task matching '{title}'." if title else "Which task?"
            if action == "done":
                ts.set_status(t["id"], "done"); msg = f"Done: {t['title']}. Nice."
            elif action == "start":
                ts.set_status(t["id"], "doing"); msg = f"Started: {t['title']}."
            elif action == "reopen":
                ts.set_status(t["id"], "todo"); msg = f"Reopened: {t['title']}."
            elif action == "delete":
                ts.delete(t["id"]); msg = f"Deleted: {t['title']}."
            else:
                new = ts.update(t["id"], due=str(p.get("due") or ""))
                if not new or not new["due"]:
                    return "I need a date I can read — try 'tomorrow', 'friday' or 'in 3 days'."
                msg = f"Moved {t['title']} to {new['due']}."
        elif action == "clear_done":
            msg = f"Cleared {ts.clear_done()} finished task(s)."
        else:
            return "I can add, list, finish, start, reschedule or delete tasks."
        if player:
            try:
                player.write_log(f"TASKS: {msg}")
            except Exception:
                pass
        return msg
    except Exception as e:
        return f"Sorry, the task board hit a problem: {e}"
