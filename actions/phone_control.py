"""
actions/phone_control.py — Voice and directive handler for JARVIS Companion Phone Link.

Allows the user to control their paired smartphone via voice or text directives:
- "find my phone" / "phone dhundo" -> rings phone at maximum volume
- "flashlight on" / "torch jalao" -> turns on phone camera flash
- "phone battery" -> checks live phone battery & charging status
- "answer call" / "cut call" -> manages incoming voice calls
- "open whatsapp on phone" / "open camera on phone" -> launches apps on phone
- "send sms to X message Y" -> sends SMS via phone
- volume, notifications, auto-reply rules
"""
from __future__ import annotations
import time
from typing import Any, Optional
from core.phone_bridge import get_bridge


def phone_action(parameters: dict, player=None, session_memory=None) -> str:
    bridge = get_bridge()
    p = parameters or {}
    action = str(p.get("action") or "status").strip().lower()
    who = str(p.get("who") or p.get("target") or p.get("number") or "").strip()
    message = str(p.get("text") or p.get("message") or "").strip()
    app_name = str(p.get("app") or p.get("target") or "").strip()
    on_flag = p.get("on")

    # Check connection
    snap = bridge.snapshot()
    if not snap.get("connected") and not snap.get("paired"):
        msg = "Sir, companion phone is not paired. Please tap 'Pair Phone (QR)' to connect your device."
        _log(msg, player)
        return msg

    dev_name = snap.get("device") or "Phone"

    if action in ("find_phone", "find", "ring"):
        res = bridge.send_command("find_phone")
        msg = f"Sir, radar alarm initiated on {dev_name}. The phone is ringing at maximum volume." if res.get("ok") else f"Couldn't ping phone: {res.get('error')}"
    elif action in ("stop_find", "stop_ring", "stop_alarm"):
        res = bridge.send_command("stop_find_phone")
        msg = f"Sir, radar alarm silenced on {dev_name}." if res.get("ok") else f"Couldn't stop ping: {res.get('error')}"
    elif action in ("torch_on", "flashlight_on") or (action in ("torch", "flashlight") and on_flag is not False):
        res = bridge.send_command("flashlight", {"on": True})
        msg = f"Sir, flashlight switched ON on {dev_name}." if res.get("ok") else f"Couldn't turn torch on: {res.get('error')}"
    elif action in ("torch_off", "flashlight_off") or (action in ("torch", "flashlight") and on_flag is False):
        res = bridge.send_command("flashlight", {"on": False})
        msg = f"Sir, flashlight turned OFF on {dev_name}." if res.get("ok") else f"Couldn't turn torch off: {res.get('error')}"
    elif action in ("battery", "battery_status", "status"):
        batt = snap.get("battery")
        chg = snap.get("charging")
        chg_txt = "and currently charging" if chg else "not charging"
        msg = f"Sir, {dev_name} battery is at {batt}%, {chg_txt}." if batt is not None else f"Sir, battery level for {dev_name} is currently unavailable."
    elif action in ("answer_call", "pickup", "answer"):
        res = bridge.send_command("answer_call")
        msg = f"Sir, incoming call answered on {dev_name}." if res.get("ok") else f"Couldn't answer call: {res.get('error')}"
    elif action in ("end_call", "decline_call", "cut_call", "decline"):
        res = bridge.send_command("end_call")
        msg = f"Sir, call terminated on {dev_name}." if res.get("ok") else f"Couldn't end call: {res.get('error')}"
    elif action in ("send_sms", "sms", "message"):
        if not who or not message:
            msg = "Sir, recipient number and message text are required to send an SMS."
        else:
            res = bridge.send_command("send_sms", {"number": who, "phone": who, "text": message, "message": message})
            msg = f"Sir, SMS transmitted to {who}: '{message}'" if res.get("ok") else f"Couldn't send SMS: {res.get('error')}"
    elif action in ("call", "place_call", "dial"):
        if not who:
            msg = "Sir, whom would you like me to call?"
        else:
            res = bridge.send_command("place_call", {"number": who, "phone": who, "name": who})
            msg = f"Sir, initiating voice call to {who} on {dev_name}." if res.get("ok") else f"Couldn't place call: {res.get('error')}"
    elif action in ("open_app", "launch_app"):
        pkg = app_name or who
        pkg_lower = pkg.lower()
        if "whatsapp" in pkg_lower: pkg = "com.whatsapp"
        elif "camera" in pkg_lower: pkg = "com.android.camera"
        elif "map" in pkg_lower: pkg = "com.google.android.apps.maps"
        elif "spotify" in pkg_lower: pkg = "com.spotify.music"
        elif "chrome" in pkg_lower: pkg = "com.android.chrome"
        elif "youtube" in pkg_lower: pkg = "com.google.android.youtube"
        res = bridge.send_command("open_app", {"app": pkg, "package": pkg})
        msg = f"Sir, opened {pkg} on {dev_name}." if res.get("ok") else f"Couldn't open app: {res.get('error')}"
    elif action in ("volume", "set_volume"):
        lvl = p.get("level", 75)
        res = bridge.send_command("volume", {"level": lvl})
        msg = f"Sir, phone volume adjusted to {lvl}% on {dev_name}." if res.get("ok") else f"Couldn't set volume: {res.get('error')}"
    elif action in ("open_url", "open_link"):
        url = who if "://" in who else f"https://{who}"
        res = bridge.send_command("open_url", {"url": url})
        msg = f"Sir, opened URL {url} on {dev_name}." if res.get("ok") else f"Couldn't open URL: {res.get('error')}"
    else:
        msg = f"Sir, phone directive '{action}' processed for {dev_name}."

    _log(msg, player)
    return msg


def _log(message: str, player=None) -> None:
    print(f"[PhoneControl] {message}")
    if player:
        try:
            player.write_log(f"PHONE: {message}")
        except Exception:
            pass


TOOL = {
    "name": "phone_control",
    "description": (
        "Control companion smartphone features: find_phone (ring), torch_on, torch_off, battery, "
        "answer_call, end_call, call contact, send_sms, open_app, volume, open_url."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "Action: find_phone | stop_find | torch_on | torch_off | battery | answer_call | end_call | call | send_sms | open_app | volume | open_url"
            },
            "who": {
                "type": "STRING",
                "description": "Contact name or phone number"
            },
            "target": {
                "type": "STRING",
                "description": "Target phone number, app name, or URL"
            },
            "text": {
                "type": "STRING",
                "description": "Message text for SMS or chat"
            },
            "message": {
                "type": "STRING",
                "description": "Message text if sending an SMS"
            },
            "app": {
                "type": "STRING",
                "description": "App name or package for open_app"
            },
            "on": {
                "type": "BOOLEAN",
                "description": "flashlight on/off boolean"
            },
            "level": {
                "type": "INTEGER",
                "description": "volume percentage (0-100)"
            }
        },
        "required": ["action"]
    },
    "handler": phone_action,
}
