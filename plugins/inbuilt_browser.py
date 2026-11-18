"""
Inbuilt Chrome / Chromium browser plugin — open URLs, manage tabs, or run
Google searches inside JARVIS's HUD dashboard ("system" browser).
"""
from __future__ import annotations

PLUGIN = {
    "name": "inbuilt_browser",
    "tier": "core",
    "brief": (
        "DEFAULT browser: open websites, Google searches, YouTube, or tabs inside JARVIS's HUD dashboard."
    ),
    "description": (
        "Open websites, run Google searches, or manage tabs (new_tab, switch_tab, "
        "close_tab) INSIDE JARVIS's own HUD dashboard inbuilt Chromium browser. "
        "THIS IS THE DEFAULT BROWSER for every search/Google/website/open command "
        "whenever the user did NOT explicitly ask for external 'system'/'pc' Chrome. "
        "Whenever the user says 'google search ...', 'search ...', 'open youtube', "
        "use action='search' or 'open' here. "
        "When the user says 'browser band karo', 'close the browser', 'browser close' or "
        "'inbuilt browser hata do', use action='close' — it really stops playback and tabs."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "enum": ["open", "search", "new_tab", "switch_tab", "close_tab", "close"],
                "description": (
                    "open (active tab), search (Google in active tab), new_tab (open URL/"
                    "search in a new tab), switch_tab (switch by 1-based tab_index), "
                    "close_tab (close tab_index or current tab), or close (CLOSE the browser: "
                    "stops any video/music, closes all tabs, returns to the HUD)."
                ),
            },
            "target": {
                "type": "STRING",
                "description": "URL (e.g. 'youtube.com') or Google search query.",
            },
            "tab_index": {
                "type": "INTEGER",
                "description": "1-based tab number for switch_tab or close_tab.",
            },
        },
        "required": [],
    },
}


def run(parameters: dict, player=None, session_memory=None) -> str:
    params = parameters or {}
    action = str(params.get("action") or "open").strip().lower()
    target = str(params.get("target") or "").strip()
    raw_idx = params.get("tab_index")
    try:
        if action == "close":
            if player is not None and hasattr(player, "close_browser"):
                player.close_browser()
                return "Closed the inbuilt browser — playback stopped, tabs closed, back on the HUD."
            return "Inbuilt browser UI is not attached."

        if action == "switch_tab":
            idx = max(1, int(raw_idx)) if raw_idx is not None else 1
            if player is not None and hasattr(player, "switch_browser_tab"):
                player.switch_browser_tab(idx - 1)
                return f"Switched to inbuilt browser tab {idx}."
            return "Inbuilt browser tab switching is not available."

        if action == "close_tab":
            idx = (max(1, int(raw_idx)) - 1) if raw_idx is not None else -1
            if player is not None and hasattr(player, "close_browser_tab"):
                player.close_browser_tab(idx)
                return "Closed inbuilt browser tab."
            return "Inbuilt browser tab closing is not available."

        if action == "search" and target and "://" not in target:
            from urllib.parse import quote_plus
            target = f"https://www.google.com/search?q={quote_plus(target)}"

        if action == "new_tab":
            if player is not None and hasattr(player, "new_browser_tab"):
                dest = target or "https://www.google.com"
                player.new_browser_tab(dest)
                return f"Opened {target or 'Google'} in a new inbuilt browser tab."
            return "Inbuilt browser new-tab UI is not attached."

        if player is not None and hasattr(player, "open_browser"):
            player.open_browser(target or "https://www.google.com")
            label = target or "Google"
            return f"Opened {label} in the inbuilt browser."
        return "Inbuilt browser UI is not attached."
    except Exception as e:
        return f"Inbuilt browser failed: {e}"
