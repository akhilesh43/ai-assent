"""
Phone control — drive the paired Android phone from the laptop, by voice.

Answer / decline a call, call someone, send an SMS, reply to a WhatsApp (or any chat) notification
in the owner's own style, read what notifications are waiting, flashlight, volume, find my phone.
Needs the JARVIS phone app paired (Remote Control → scan the QR in the app).
"""
from __future__ import annotations

PLUGIN = {
    "name": "phone_control",
    "description": (
        "Control the user's paired Android PHONE. Use for: 'pick up / call utha lo / answer', 'cut the call / "
        "decline / call kaat do', 'call Mom / Mummy ko call karo', 'SMS bhejo', 'WhatsApp message ka reply karo / "
        "reply to Rohan', 'phone pe kya notifications aaye', 'phone ki battery', 'torch on', 'phone dhundo / find my "
        "phone', 'phone volume', 'open <app> on the phone', and 'auto-reply on/off', 'is tarah reply karna sikho'. "
        "Replies are drafted in the user's own learned style. NOT for the laptop's own apps, and NOT for "
        "WhatsApp on the laptop (use send_message for that)."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "description": (
                "answer_call | end_call | call | sms | reply | send_pending | list_pending | notifications | "
                "battery | flashlight | volume | find_phone | stop_find | open_app | auto_reply | teach | status")},
            "who": {"type": "STRING", "description": "Contact name (call / sms / reply / teach / auto_reply for one person)"},
            "number": {"type": "STRING", "description": "Phone number, if the user said digits"},
            "text": {"type": "STRING", "description": "Message to send, or the reply wording the user dictated"},
            "app": {"type": "STRING", "description": "App name or package for open_app"},
            "mode": {"type": "STRING", "description": "auto_reply: auto | ask | ignore | off"},
            "on": {"type": "BOOLEAN", "description": "flashlight: true=on false=off"},
            "level": {"type": "INTEGER", "description": "volume 0-100"},
            "incoming": {"type": "STRING", "description": "teach: the message the reply answers"},
        },
        "required": ["action"],
    },
}


def _fail(res: dict) -> str:
    return f"Couldn't do that: {res.get('error') or 'the phone said no'}."


def run(parameters: dict, player=None, session_memory=None) -> str:
    try:
        from core.phone_bridge import get_bridge
        b = get_bridge()
        p = parameters or {}
        a = str(p.get("action") or "status").strip().lower()
        who, number, text = (str(p.get(k) or "").strip() for k in ("who", "number", "text"))

        def say(msg: str) -> str:
            if player:
                try:
                    player.write_log(f"PHONE: {msg}")
                except Exception:
                    pass
            return msg

        if a == "status":
            s = b.snapshot()
            if not s["connected"]:
                return say("No phone is connected. Press Remote Control and scan the QR code in the JARVIS phone app.")
            bits = [f"{s['device']} is connected"]
            if s["battery"] is not None:
                bits.append(f"battery {s['battery']}%" + (", charging" if s["charging"] else ""))
            if s["call"].get("state") == "ringing":
                bits.append(f"{s['call']['who']} is calling right now")
            if s["pending"]:
                bits.append(f"{len(s['pending'])} repl{'y' if len(s['pending']) == 1 else 'ies'} waiting for your OK")
            bits.append(f"auto-reply is {s['mode']}")
            return say(", ".join(bits) + ".")

        if a == "teach":
            if not text:
                return "What should the reply say?"
            ok = b.teach(str(p.get("incoming") or "general"), text, who)
            return say("Got it, I'll answer like that." if ok else "I already know that one.")

        if a == "auto_reply":
            from memory import reply_store
            mode = str(p.get("mode") or "").lower()
            if who:
                if not reply_store.set_rule(who, mode):
                    return "Use auto, ask, ignore or off."
                return say(f"{who}: {mode}.")
            try:
                b.set_mode(mode)
            except ValueError:
                return "Use auto, ask, ignore or off."
            return say(f"Auto-reply is now {mode}."
                       + (" I'll still ask before answering anything private." if mode == "auto" else ""))

        if a == "list_pending":
            pend = b.pending_list()
            if not pend:
                return say("Nothing is waiting for your approval.")
            first = pend[0]
            return say(f"{len(pend)} waiting. {first['contact']} wrote “{first['incoming'][:80]}”. "
                       + (f"My draft: “{first['draft']}”." if first["draft"] else "That one needs your own answer."))

        if a == "send_pending":
            pend = b.pending_list()
            target = next((x for x in pend if not who or who.lower() in x["contact"].lower()), None)
            if not target:
                return say("There's no waiting reply" + (f" for {who}." if who else "."))
            res = b.approve(target["id"], text or None)
            return say(f"Sent to {target['contact']}." if res.get("ok") else _fail(res))

        if a == "reply":
            # Dictated wording wins; otherwise draft one in the owner's style and send it.
            pend = [x for x in b.pending_list() if not who or who.lower() in x["contact"].lower()]
            if pend:
                res = b.approve(pend[0]["id"], text or (pend[0]["draft"] or None))
                if res.get("ok"):
                    return say(f"Replied to {pend[0]['contact']}.")
                return say(_fail(res))
            notes = [n for n in b.snapshot()["notifications"]
                     if n["can_reply"] and (not who or who.lower() in (n["sender"] or n["title"]).lower())]
            if not notes:
                return say("I don't see a chat to reply to" + (f" from {who}." if who else "."))
            n = notes[0]
            final = text or b.draft(n["text"], n["sender"] or n["title"], n["app"])
            if not final:
                return say("That one needs your own words. What should I say?")
            res = b.send_command("reply_notification", {"key": n["key"], "text": final})
            return say(f"Replied to {n['sender'] or n['title']}: {final}" if res.get("ok") else _fail(res))

        if a == "notifications":
            notes = b.snapshot()["notifications"][:5]
            if not notes:
                return say("No notifications from the phone yet.")
            return say("; ".join(f"{n['app'] or 'app'}: {n['sender'] or n['title']} — {n['text'][:60]}" for n in notes) + ".")

        # ── everything below goes to the phone itself ───────────────────────
        if a == "answer_call":
            res = b.send_command("answer_call")
        elif a == "end_call":
            res = b.send_command("end_call")
        elif a == "call":
            res = b.send_command("place_call", {"number": number, "name": who})
        elif a == "sms":
            if not text:
                return "What should the SMS say?"
            res = b.send_command("send_sms", {"number": number, "name": who, "text": text})
        elif a == "battery":
            res = b.send_command("battery")
            if res.get("ok"):
                d = res.get("data") or {}
                return say(f"Phone battery is {d.get('level')}%" + (", charging." if d.get("charging") else "."))
        elif a == "flashlight":
            res = b.send_command("flashlight", {"on": bool(p.get("on", True))})
        elif a == "volume":
            res = b.send_command("volume", {"level": int(p.get("level") or 50)})
        elif a == "find_phone":
            res = b.send_command("find_phone")
        elif a == "stop_find":
            res = b.send_command("stop_find_phone")
        elif a == "open_app":
            res = b.send_command("open_app", {"app": str(p.get("app") or who)})
        else:
            return "I can answer or end calls, call, SMS, reply to chats, read notifications, and more — what do you need?"
        if not res.get("ok"):
            return say(_fail(res))
        d = res.get("data") or {}
        return say(str(d.get("message") or "Done."))
    except Exception as e:
        return f"Sorry, the phone link hit a problem: {e}"
