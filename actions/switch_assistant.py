"""
actions/switch_assistant.py — AI tool to switch active AI persona (JARVIS, FRIDAY, EDITH, ULTRON, or custom).
"""
from typing import Any, Dict


def switch_assistant_handler(parameters: Dict[str, Any], player=None, **kwargs) -> str:
    target = str(parameters.get("name") or parameters.get("assistant") or parameters.get("target") or "").strip()
    if not target or target.lower() in ("next", "cycle", "badlo", "switch"):
        from core.assistants_manager import cycle_assistant
        ok, msg, asst = cycle_assistant()
    else:
        from core.assistants_manager import switch_assistant
        ok, msg, asst = switch_assistant(target)

    if ok and asst:
        new_name = asst["name"]
        new_color = asst.get("ui_color", "#00d4ff")
        # Apply theme live
        try:
            from ui import apply_ui_accent, current_palette, retheme_all_widgets
            old_pal = current_palette()
            if apply_ui_accent(new_color):
                retheme_all_widgets(old_pal, current_palette())
        except Exception:
            pass

        if player and hasattr(player, "write_log"):
            player.write_log(f"SYS: Switched active AI assistant to {new_name}")

        return msg

    return f"Could not switch assistant: {msg}"


TOOL = {
    "name": "switch_assistant",
    "description": "Switch the active AI assistant persona, voice, and theme. Supports 'JARVIS', 'FRIDAY', 'EDITH', 'ULTRON', or 'next' to cycle.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "name": {
                "type": "STRING",
                "description": "Name or ID of the assistant (e.g. 'FRIDAY', 'JARVIS', 'EDITH', 'ULTRON', 'next')."
            }
        },
        "required": ["name"]
    },
    "handler": switch_assistant_handler,
}

