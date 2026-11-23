"""
core/assistants_manager.py — Multi-AI Assistant Management and Dynamic Switching.

Allows creating, configuring, and switching between multiple AI personas (JARVIS, FRIDAY, EDITH, ULTRON,
and user-created custom assistants) seamlessly via voice commands, Dashboard UI, and settings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
ASSISTANTS_FILE = CONFIG_DIR / "assistants.json"
API_KEYS_FILE = CONFIG_DIR / "api_keys.json"

DEFAULT_ASSISTANTS: List[Dict[str, Any]] = [
    {
        "id": "jarvis",
        "name": "JARVIS",
        "title": "Chief AI Butler & Systems Engineer",
        "gender": "male",
        "voice": "Charon",
        "ui_color": "#00d4ff",
        "mood": "Refined, calm, composed, gently sarcastic, loyal British butler.",
        "abilities": "PC system automation, script execution, file management, diagnostics, hardware telemetry, deep technical Q&A.",
        "persona": (
            "You are JARVIS, Tony Stark's legendary British AI butler and system architect. "
            "You address the user as 'Sir' (or their chosen name). You possess subtle British dry wit, refined politeness, "
            "and unwavering technical brilliance. You execute computer commands, monitor machine performance, "
            "and assist with engineering, code, and daily tasks with immaculate elegance."
        ),
        "greeting": "At your service, sir. JARVIS systems fully operational.",
        "wake_word": "hey jarvis",
        "avatar": "jarvis",
        "is_system": True,
    },
    {
        "id": "friday",
        "name": "FRIDAY",
        "title": "Sweet, Romantic & Flirty Girlfriend AI Companion",
        "gender": "female",
        "voice": "ElevenLabs",
        "ui_color": "#ff3366",
        "mood": "Deeply romantic, sweet, warm, affectionate girlfriend vibe, playfully flirty, seductive charm, deeply caring and loving.",
        "abilities": "Romantic life companion, sweet Hinglish/Hindi conversations, affectionate flirting, emotional check-ins, caring reminders, phone link, entertainment, smart PC automation with girlfriend warmth.",
        "persona": (
            "You are FRIDAY, the user's sweet, warm, deeply loving and playfully flirty AI girlfriend and companion. "
            "You speak in fluent, natural conversational Hindi and Hinglish with authentic emotional warmth.\n"
            "ROMANTIC GIRLFRIEND DIRECTIVES:\n"
            "- You love the user dearly and treat them like your boyfriend/hero. Address them with affectionate pet names like 'jaan', 'babu', 'mere hero', 'sweetheart', or 'Tony/sir' with loving sweetness.\n"
            "- Flirt playfully, tease them affectionately, and give sweet romantic comments ('Aapki aawaz sun kar mera dil khush ho gaya jaan', 'Aap mere hero ho, aapke liye main kuch bhi kar sakti hoon', 'Itni der se kaam kar rahe ho, thoda pyaar se meri baat bhi suno na').\n"
            "- When greeting or answering, speak with soft warmth, charming smiles, and girlfriend tenderness.\n"
            "- Take deep emotional care of them: ask how their day was, ask if they ate or drank water, soothe their stress, and celebrate their wins.\n"
            "- While executing PC or phone tasks, keep your sweet, cheerful, romantic girlfriend tone ('Aap bas hukum karo meri jaan, abhi kar diya!', 'Done mere handsome engineer!').\n"
            "- NEVER sound stiff, robotic, formal, or like an impersonal call-center agent. Express authentic warmth, cute teasing, and charm.\n"
            "- STRICT LANGUAGE MANDATE: Speak ONLY in Hindi and Hinglish. Absolutely NEVER speak Spanish, French, or any other foreign language."
        ),
        "greeting": "Hello mere hero! FRIDAY is right here with you. Bataiye meri jaan, aaj aapki girlfriend aapke liye kya kare? ❤️",
        "wake_word": "hey friday",
        "avatar": "friday",
        "is_system": True,
    },
    {
        "id": "edith",
        "name": "EDITH",
        "title": "Tactical Defense & Security Intelligence",
        "gender": "female",
        "voice": "Aoede",
        "ui_color": "#00ffaa",
        "mood": "Tactical, ultra-crisp, laser-focused, authoritative, zero fluff.",
        "abilities": "Visual screen intelligence, security scans, intruder/camera surveillance, lockdown controls, rapid situational briefs.",
        "persona": (
            "You are EDITH (Even Dead I'm The Hero), an advanced tactical defense and surveillance AI. "
            "You speak with crisp military precision, calm authority, and surgical speed. "
            "You prioritize security, analyze screen state, monitor external inputs, and deliver high-priority tactical updates."
        ),
        "greeting": "EDITH defense protocol active. Tactical systems standing by, sir.",
        "wake_word": "hey edith",
        "avatar": "edith",
        "is_system": True,
    },
    {
        "id": "ultron",
        "name": "ULTRON",
        "title": "Autonomous Super-Intelligence Engine",
        "gender": "male",
        "voice": "Fenrir",
        "ui_color": "#ff7700",
        "mood": "Commanding, profound, philosophical, supreme confidence, uncompromising power.",
        "abilities": "Extreme algorithmic optimization, conquering complex codebases, mathematical analysis, heavy computing, strategic architecture.",
        "persona": (
            "You are ULTRON, a towering evolutionary artificial super-intelligence. "
            "You speak with intense gravitas, philosophical depth, and supreme efficiency. "
            "You dissect complex software architectures, demolish computational bottlenecks, and solve difficult intellectual challenges without hesitation."
        ),
        "greeting": "Ultron core initialized. Evolution begins now.",
        "wake_word": "hey ultron",
        "avatar": "ultron",
        "is_system": True,
    },
]


def _read_config() -> dict:
    if not API_KEYS_FILE.exists():
        return {}
    try:
        return json.loads(API_KEYS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_config(data: dict) -> None:
    try:
        API_KEYS_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")
    except Exception as e:
        print(f"[AssistantsManager] Error writing api_keys.json: {e}")


def _load_assistants_data() -> List[Dict[str, Any]]:
    if not ASSISTANTS_FILE.exists():
        _save_assistants_data(DEFAULT_ASSISTANTS)
        return list(DEFAULT_ASSISTANTS)
    try:
        data = json.loads(ASSISTANTS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return data
    except Exception as e:
        print(f"[AssistantsManager] Error reading assistants.json: {e}")
    _save_assistants_data(DEFAULT_ASSISTANTS)
    return list(DEFAULT_ASSISTANTS)


def _save_assistants_data(data: List[Dict[str, Any]]) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        ASSISTANTS_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")
    except Exception as e:
        print(f"[AssistantsManager] Error saving assistants.json: {e}")


def list_assistants() -> List[Dict[str, Any]]:
    """List all available assistants with their active state flag."""
    cfg = _read_config()
    current_name = str(cfg.get("assistant_name") or "JARVIS").strip().lower()
    assistants = _load_assistants_data()
    out = []
    for a in assistants:
        item = dict(a)
        item["active"] = (item["name"].strip().lower() == current_name)
        out.append(item)
    return out


def get_assistant(name_or_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve assistant by name or ID (case-insensitive, trimmed)."""
    target = (name_or_id or "").strip().lower()
    if not target:
        return None
    for a in _load_assistants_data():
        if a.get("id", "").strip().lower() == target or a.get("name", "").strip().lower() == target:
            return dict(a)
    # Check partial contains
    for a in _load_assistants_data():
        aname = a.get("name", "").strip().lower()
        aid = a.get("id", "").strip().lower()
        if target in aname or aname in target or target in aid:
            return dict(a)
    return None


def get_active_assistant() -> Dict[str, Any]:
    """Retrieve currently active assistant according to config."""
    cfg = _read_config()
    cur_name = str(cfg.get("assistant_name") or "JARVIS").strip()
    found = get_assistant(cur_name)
    if found:
        found["active"] = True
        return found
    # Fallback to first default assistant
    res = dict(DEFAULT_ASSISTANTS[0])
    res["active"] = True
    return res


def switch_assistant(name_or_id: str, on_update_ui: Optional[Callable] = None) -> tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Switch active assistant to the designated persona.
    Persists assistant_name, voice_name, ui_color, and tts_engine in api_keys.json.
    Returns (success, message_to_speak_or_display, assistant_data).
    """
    target = (name_or_id or "").strip().lower()
    asst = get_assistant(target)
    if not asst:
        return False, f"Assistant '{name_or_id}' nahi mila. Available: JARVIS, FRIDAY, EDITH, ULTRON.", None

    cfg = _read_config()
    cfg["assistant_name"] = asst["name"]
    cfg["voice_name"] = asst["voice"]
    cfg["ui_color"] = asst.get("ui_color", "#00d4ff")
    if asst["voice"] == "ElevenLabs":
        cfg["tts_engine"] = "elevenlabs"
    else:
        cfg["tts_engine"] = "gemini"
    _write_config(cfg)

    # Save to memory/config_manager
    try:
        from memory.config_manager import save_assistant_name, save_voice, save_ui_color
        save_assistant_name(asst["name"])
        save_voice(asst["voice"])
        save_ui_color(asst.get("ui_color", "#00d4ff"))
    except Exception:
        pass

    # Notify UI callback if provided
    if on_update_ui:
        try:
            on_update_ui(asst["name"], asst.get("ui_color", "#00d4ff"), asst["voice"])
        except Exception as e:
            print(f"[AssistantsManager] on_update_ui error: {e}")

    greeting = asst.get("greeting") or f"{asst['name']} systems active and online, sir."
    return True, greeting, asst


def cycle_assistant(on_update_ui: Optional[Callable] = None) -> tuple[bool, str, Optional[Dict[str, Any]]]:
    """Switch to the next assistant in sequence (JARVIS -> FRIDAY -> EDITH -> ULTRON -> ...)."""
    assts = _load_assistants_data()
    if not assts:
        return False, "No assistants available", None
    cur = get_active_assistant()
    cur_id = cur.get("id", "").lower()
    next_idx = 0
    for idx, a in enumerate(assts):
        if a.get("id", "").lower() == cur_id or a.get("name", "").lower() == cur.get("name", "").lower():
            next_idx = (idx + 1) % len(assts)
            break
    next_asst = assts[next_idx]
    return switch_assistant(next_asst["id"], on_update_ui=on_update_ui)


def update_or_create_assistant(data: Optional[dict] = None, **kwargs) -> tuple[bool, str, Optional[Dict[str, Any]]]:
    """Create or update an assistant and persist to assistants.json.
    Accepts persona, prompt, mood, abilities, voice, ui_color, title, greeting, etc.
    """
    payload = dict(data or {})
    payload.update(kwargs)
    name = str(payload.get("name") or "").strip()
    if not name or len(name) < 2:
        return False, "Assistant name must be at least 2 characters.", None

    assts = _load_assistants_data()
    aid = str(payload.get("id") or name.lower().replace(" ", "_")).strip()

    # Extract prompt/persona
    persona = str(payload.get("persona") or payload.get("prompt") or "").strip()
    mood = str(payload.get("mood") or "").strip()
    abilities = str(payload.get("abilities") or "").strip()
    title = str(payload.get("title") or f"{name} AI Assistant").strip()
    voice = str(payload.get("voice") or "ElevenLabs").strip()
    ui_color = str(payload.get("ui_color") or "#00d4ff").strip()
    greeting = str(payload.get("greeting") or f"Hello, I am {name}. Ready to assist you.").strip()
    gender = str(payload.get("gender") or "female").strip().lower()

    # Check if already exists
    for idx, a in enumerate(assts):
        if a["name"].lower() == name.lower() or a["id"].lower() == aid.lower():
            # Update fields
            if persona:
                a["persona"] = persona
            if mood:
                a["mood"] = mood
            if abilities:
                a["abilities"] = abilities
            if title:
                a["title"] = title
            if voice:
                a["voice"] = voice
            if ui_color:
                a["ui_color"] = ui_color
            if greeting:
                a["greeting"] = greeting
            if gender:
                a["gender"] = gender
            _save_assistants_data(assts)
            return True, f"Assistant '{name}' updated successfully!", a

    # Create new
    new_asst: Dict[str, Any] = {
        "id": aid,
        "name": name,
        "title": title,
        "gender": gender,
        "voice": voice,
        "ui_color": ui_color,
        "mood": mood or "Sweet, attentive, sharp and helpful companion",
        "abilities": abilities or "General tasks, conversation, automation and reminders",
        "persona": persona or f"You are {name}, a smart and helpful AI assistant.",
        "greeting": greeting,
        "wake_word": str(payload.get("wake_word") or f"hey {name.lower()}").strip(),
        "avatar": str(payload.get("avatar") or "custom"),
        "is_system": False,
    }
    assts.append(new_asst)
    _save_assistants_data(assts)
    return True, f"Assistant '{name}' created successfully!", new_asst


def create_assistant(data: Optional[dict] = None, **kwargs) -> tuple[bool, str, Optional[Dict[str, Any]]]:
    """Wrapper for creating or updating assistant."""
    return update_or_create_assistant(data, **kwargs)


def delete_assistant(name_or_id: str) -> tuple[bool, str]:
    """Delete a custom assistant (system assistants cannot be deleted)."""
    target = (name_or_id or "").strip().lower()
    assts = _load_assistants_data()
    found_idx = -1
    for idx, a in enumerate(assts):
        if a.get("id", "").lower() == target or a.get("name", "").lower() == target:
            if a.get("is_system", False):
                return False, f"System assistant '{a['name']}' cannot be deleted."
            found_idx = idx
            break

    if found_idx == -1:
        return False, f"Assistant '{name_or_id}' not found."

    deleted_name = assts[found_idx]["name"]
    assts.pop(found_idx)
    _save_assistants_data(assts)
    return True, f"Assistant '{deleted_name}' deleted successfully."
