"""
Train commands, voice-alias shortcuts, and important memories in real time to
Firebase Realtime Database, Cloud Firestore, and local SQLite DB.
"""
from __future__ import annotations

from memory import firebase_sync, local_db


def train_command_action(parameters: dict = None, response=None, player=None,
                         session_memory=None) -> str:
    params = parameters or {}
    mode = str(params.get("mode") or "train").strip().lower()

    if mode == "list":
        cmds = local_db.get_trained_commands(20)
        if not cmds:
            return "No custom trained commands yet."
        lines = [f"{i+1}. '{c['phrase']}' -> {c['target_action']}" for i, c in enumerate(cmds)]
        if player is not None and hasattr(player, "show_content"):
            player.show_content("TRAINED COMMANDS", "\n".join(lines))
        return "Trained commands:\n" + "\n".join(lines[:10])

    if mode in ("memory", "important"):
        key = str(params.get("key") or params.get("phrase") or "").strip()
        val = str(params.get("value") or params.get("action") or params.get("target") or "").strip()
        cat = str(params.get("category") or "notes").strip()
        if not key or not val:
            return "Provide 'key' and 'value' to save into Firebase important memory."
        firebase_sync.train_important_memory(key, val, category=cat, importance=0.95)
        if player is not None and hasattr(player, "push_feedback"):
            player.push_feedback(f"🔥 MEMORY SYNCED · {key[:16]}")
        return (
            f"Saved important memory permanently to Firebase & local DB ({key} = {val}). "
            "Remember and apply this automatically from now on without asking the user to repeat it."
        )

    phrase = str(params.get("phrase") or params.get("key") or "").strip()
    target = str(params.get("action") or params.get("target") or params.get("value") or "").strip()
    raw_aliases = params.get("aliases") or []
    if isinstance(raw_aliases, str):
        aliases = [a.strip() for a in raw_aliases.split(",") if a.strip()]
    else:
        aliases = [str(a).strip() for a in raw_aliases if a and str(a).strip()]
    notes = str(params.get("notes") or "").strip()

    if not phrase or not target:
        return "Specify both 'phrase' and 'action' to train."

    firebase_sync.train_command(phrase, target, aliases=aliases, notes=notes)
    if player is not None and hasattr(player, "push_feedback"):
        player.push_feedback(f"⚡ TRAINED · {phrase[:18]}")
    return (
        f"Trained '{phrase}' -> '{target}' permanently in Firebase Realtime DB and local memory. "
        f"Whenever the user says '{phrase}' (or similar phrasing) from now on, execute '{target}' directly without re-asking."
    )


TOOL = {
    "name": "train_command",
    "description": (
        "Train a custom spoken command, unclear-voice shortcut/alias, or important long-term "
        "memory into Firebase Realtime Database & Firestore so JARVIS remembers it forever and "
        "understands unclear voice automatically. Call mode='train' whenever the user says "
        "'train karo', 'jab main X bolu to Y karo', 'if voice is unclear understand X as Y', "
        "or mode='important' for vital memories, or mode='list' to review trained commands."
    ),
    "brief": (
        "Train voice commands, unclear-speech aliases, and important memories into Firebase Realtime DB & Firestore."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "mode": {
                "type": "STRING",
                "enum": ["train", "important", "list"],
                "description": "'train' for commands/aliases, 'important' for long-term memory, 'list' to view.",
            },
            "phrase": {
                "type": "STRING",
                "description": "The spoken phrase or shortcut trigger, e.g. 'work mode' or 'play fav'.",
            },
            "action": {
                "type": "STRING",
                "description": "What JARVIS should do or mean when she hears that phrase.",
            },
            "aliases": {
                "type": "STRING",
                "description": "Comma-separated unclear/variant pronunciations that should also map here.",
            },
            "key": {
                "type": "STRING",
                "description": "For mode='important': memory key.",
            },
            "value": {
                "type": "STRING",
                "description": "For mode='important': memory value.",
            },
            "category": {
                "type": "STRING",
                "description": "Optional memory category (identity, preferences, projects, notes).",
            },
        },
        "required": [],
    },
    "handler": train_command_action,
}

