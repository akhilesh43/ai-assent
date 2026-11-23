"""
actions/customize_ui.py — AI tool to customize UI theme color, accent, and open customizer overlay.
"""
from typing import Any, Dict

_NAMED_COLORS = {
    "cyan": "#00d4ff",
    "blue": "#00a2ff",
    "pink": "#ff3366",
    "rose": "#ff2a85",
    "green": "#00ffaa",
    "emerald": "#00e676",
    "orange": "#ff7700",
    "amber": "#ffaa00",
    "gold": "#ffd700",
    "yellow": "#ffd700",
    "purple": "#b300ff",
    "violet": "#8a2be2",
    "red": "#ff3333",
}


def customize_ui_handler(parameters: Dict[str, Any], player=None, **kwargs) -> str:
    action = str(parameters.get("action") or "set_color").strip().lower()
    raw_color = str(parameters.get("color") or "").strip().lower()

    if action in ("open_dialog", "open_customizer", "open_settings", "show_dialog"):
        if player and hasattr(player, "openCustomize"):
            try:
                player.openCustomize()
                return "Customise Assistant dialog open kar diya hai sir."
            except Exception:
                pass
        return "UI customization dialog open kar diya hai."

    # Color resolution
    target_hex = ""
    if raw_color in _NAMED_COLORS:
        target_hex = _NAMED_COLORS[raw_color]
    elif raw_color.startswith("#") and len(raw_color) == 7:
        target_hex = raw_color
    elif not target_hex:
        # Check if color name is inside the string
        for name, hx in _NAMED_COLORS.items():
            if name in raw_color:
                target_hex = hx
                break

    if not target_hex:
        target_hex = "#00d4ff"

    try:
        from ui import apply_ui_accent, current_palette, retheme_all_widgets
        old_pal = current_palette()
        if apply_ui_accent(target_hex):
            retheme_all_widgets(old_pal, current_palette())
        # Persist to config
        from memory.config_manager import load_api_keys, save_ui_color
        try:
            save_ui_color(target_hex)
        except Exception:
            pass

        msg = f"UI theme color {raw_color or target_hex} ({target_hex}) me update kar diya hai sir."
        if player and hasattr(player, "write_log"):
            player.write_log(f"SYS: {msg}")
        return msg
    except Exception as e:
        return f"UI color change error: {e}"


TOOL = {
    "name": "customize_ui",
    "description": "Customize UI theme accent color, or open the visual customizer dialog. Supports colors like 'cyan', 'pink', 'green', 'orange', 'purple', 'gold', 'red', 'blue', or any hex code.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "'set_color' to change UI accent color, or 'open_dialog' to open the visual customization overlay."
            },
            "color": {
                "type": "STRING",
                "description": "Color name (e.g. 'pink', 'cyan', 'green', 'orange', 'gold', 'purple') or hex code (e.g. '#ff3366')."
            }
        }
    },
    "handler": customize_ui_handler,
}

