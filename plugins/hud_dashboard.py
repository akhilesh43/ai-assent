"""
JARVIS HUD Dashboard plugin — opens and controls JARVIS's own internal
("system") dashboard panels and features without touching external PC apps.
"""
from __future__ import annotations

_VALID_FEATURES = (
    "browser",
    "hud",
    "camera",
    "camera_off",
    "audio_devices",
    "voice_lock",
    "memory",
    "plugins",
    "plugin_settings",
    "providers",
    "api_key",
    "customize",
    "remote",
    "setup",
    "controls",
    "screen_watch",
    "screen_watch_off",
    "companion_chat",
    "companion_chat_off",
    "private",
    "private_off",
    "mute",
    "unmute",
)

PLUGIN = {
    "name": "hud_dashboard",
    "tier": "core",
    "brief": (
        "Open or control features inside JARVIS's own SYSTEM HUD dashboard "
        "(browser, voice lock, audio devices, memory, plugins, camera, settings)."
    ),
    "description": (
        "Controls JARVIS's OWN internal HUD dashboard ('system'). Use this tool "
        "whenever the user says 'system' or asks to open/toggle an internal JARVIS "
        "dashboard feature: inbuilt browser, hud/avatar home, camera feed, "
        "audio_devices, voice_lock, memory, plugins, plugin_settings, providers, "
        "customize, remote control, setup drawer, controls drawer, or mic mute. "
        "NEVER use this tool when the user says 'device' (their external Windows PC) "
        "— use computer_settings / open_app / browser_control for 'device' requests."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "feature": {
                "type": "STRING",
                "enum": list(_VALID_FEATURES),
                "description": "Internal HUD dashboard panel or action to run.",
            },
            "target": {
                "type": "STRING",
                "description": "Optional URL or query when feature='browser'.",
            },
        },
        "required": ["feature"],
    },
}


def run(parameters: dict, player=None, session_memory=None) -> str:
    params = parameters or {}
    feature = str(params.get("feature") or "").strip().lower().replace("-", "_").replace(" ", "_")
    target = str(params.get("target") or "").strip()

    aliases = {
        "home": "hud",
        "avatar": "hud",
        "reactor": "hud",
        "inbuilt_browser": "browser",
        "chrome": "browser",
        "audio": "audio_devices",
        "voice": "voice_lock",
        "listen": "voice_lock",
        "plugin": "plugins",
        "plugin_manager": "plugins",
        "settings": "plugin_settings",
        "engine": "providers",
        "gemini_key": "api_key",
        "gemini_api_key": "api_key",
        "customise": "customize",
        "remote_control": "remote",
    }
    feature = aliases.get(feature, feature)

    if feature not in _VALID_FEATURES:
        return (
            f"Unknown system dashboard feature '{feature}'. "
            f"Available features: {', '.join(_VALID_FEATURES)}."
        )
    if player is None or not hasattr(player, "open_dashboard_panel"):
        return "JARVIS HUD dashboard UI is not attached."
    try:
        player.open_dashboard_panel(feature, target)
        if feature == "browser" and target:
            return f"Opened '{target}' inside JARVIS's system dashboard browser."
        return f"Opened '{feature}' on JARVIS's system dashboard."
    except Exception as e:
        return f"Dashboard action failed: {e}"

