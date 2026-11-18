"""
Routine / Macro Runner plugin — save and trigger multi-step desktop/web routines
by voice (protected by Voice Lock when a voiceprint is registered).
"""
from __future__ import annotations

import json
import subprocess
import threading
import webbrowser

from memory.config_manager import get_plugin_config, save_plugin_config

PLUGIN = {
    "name": "routine_runner",
    "tier": "extra",
    "require_voice_lock": True,
    "brief": "Save, list, or run multi-step desktop and URL routines.",
    "description": (
        "Create, list, delete, or execute multi-step personal routines (macros) such "
        "as 'work mode' or 'study mode' that open URLs and launch applications together."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "enum": ["run", "save", "list", "delete"],
                "description": "Routine operation.",
            },
            "name": {
                "type": "STRING",
                "description": "Routine name, e.g. 'work' or 'study'.",
            },
            "steps": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": (
                    "List of URLs (https://...) or executable/app commands to launch "
                    "when action='save'."
                ),
            },
        },
        "required": ["action"],
    },
}


def _load_routines() -> dict[str, list[str]]:
    raw = get_plugin_config("routine_runner").get("routines_json", "{}")
    if isinstance(raw, dict):
        return {str(k).lower(): list(v) for k, v in raw.items() if isinstance(v, list)}
    try:
        parsed = json.loads(str(raw or "{}"))
        if isinstance(parsed, dict):
            return {str(k).lower(): [str(x) for x in v] for k, v in parsed.items() if isinstance(v, list)}
    except Exception:
        pass
    return {}


def _save_routines(routines: dict[str, list[str]]) -> None:
    save_plugin_config("routine_runner", {"routines_json": json.dumps(routines)})


def _execute_steps(name: str, steps: list[str], player) -> None:
    ran = 0
    for step in steps:
        s = str(step or "").strip()
        if not s:
            continue
        try:
            if s.startswith(("http://", "https://")):
                webbrowser.open(s)
            else:
                subprocess.Popen(s, shell=True)
            ran += 1
        except Exception as e:
            if player is not None:
                try:
                    player.write_log(f"ERR: Routine '{name}' step '{s}' failed — {e}")
                except Exception:
                    pass
    if player is not None:
        try:
            player.write_log(f"SYS: Routine '{name}' launched {ran} step(s).")
        except Exception:
            pass


def run(parameters: dict, player=None, session_memory=None) -> str:
    action = str((parameters or {}).get("action") or "list").strip().lower()
    name = str((parameters or {}).get("name") or "").strip().lower()
    try:
        routines = _load_routines()
        if action == "list":
            if not routines:
                return "No routines saved yet."
            summary = ", ".join(f"{k} ({len(v)} steps)" for k, v in sorted(routines.items()))
            if player is not None:
                try:
                    player.show_content("SAVED ROUTINES", summary)
                except Exception:
                    pass
            return f"Saved routines: {summary}."
        if action == "save":
            steps = [str(x).strip() for x in ((parameters or {}).get("steps") or []) if str(x).strip()]
            if not name or not steps:
                return "Specify both a routine name and at least one URL or app command."
            routines[name] = steps
            _save_routines(routines)
            return f"Saved routine '{name}' with {len(steps)} step(s)."
        if action == "delete":
            if name not in routines:
                return f"No routine named '{name}' found."
            del routines[name]
            _save_routines(routines)
            return f"Deleted routine '{name}'."
        if action == "run":
            if name not in routines:
                return f"Routine '{name}' is not saved yet."
            steps = routines[name]
            threading.Thread(
                target=_execute_steps,
                args=(name, steps, player),
                daemon=True,
                name=f"routine-{name}",
            ).start()
            return f"Running '{name}' routine ({len(steps)} steps)."
        return "Use run, save, list, or delete for routine_runner."
    except Exception as e:
        return f"Routine runner failed: {e}"

