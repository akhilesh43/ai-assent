"""
JARVIS Image Studio — JARVIS draws its own pictures.

Voice: "ek sher ki photo banao", "generate an image of a neon city at night",
"draw me a logo for my cafe". The picture is saved in generated_images/, shown on
the HUD, and appears in the phone/web dashboard's IMAGE STUDIO panel.
"""
from __future__ import annotations

from core.image_gen import ASPECTS, generate_image

PLUGIN = {
    "name": "image_studio",
    "brief": "Generate a new picture from a text description and show it on the HUD.",
    "description": (
        "Creates a NEW image with AI from a description. Use when the user asks to "
        "generate, create, draw, design or make a picture / image / photo / art / logo / "
        "wallpaper / poster ('image banao', 'photo generate karo', 'tasveer banao', "
        "'draw a ...'). The picture is shown on the HUD and saved to the gallery. "
        "NEVER use this to FIND an existing image on the web (use web_search) or to look "
        "at the screen or camera (use the screen/camera tools). Pass the user's wording "
        "as `prompt`, adding useful visual detail only if they gave very little."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "prompt": {"type": "STRING",
                       "description": "What the picture should show (any language)."},
            "style": {"type": "STRING",
                      "description": "Optional style: realistic photo, anime, 3D render, "
                                     "watercolor, neon cyberpunk, logo, sketch…"},
            "aspect": {"type": "STRING", "enum": list(ASPECTS),
                       "description": "Shape: 1:1 square (default), 16:9 wide, 9:16 tall "
                                      "(phone wallpaper), 4:3, 3:4."},
        },
        "required": ["prompt"],
    },
}


def run(parameters: dict, player=None, session_memory=None) -> str:
    p = parameters or {}

    def _log(msg: str) -> None:
        if player is not None:
            try:
                player.write_log(f"SYS: {msg}")
            except Exception:
                pass

    try:
        res = generate_image(str(p.get("prompt", "")), str(p.get("style", "") or ""),
                             str(p.get("aspect", "1:1") or "1:1"), log=_log)
    except Exception as e:                                  # generate_image never raises, but
        return f"Sir, the image studio failed: {e}"         # a spoken error beats silence

    if not res.get("ok"):
        msg = f"Sir, I could not draw that: {res.get('error', 'unknown error')}"
        if player is not None:
            try:
                player.write_log(f"JARVIS: {msg[:300]}")
            except Exception:
                pass
        return msg

    if player is not None:
        try:
            show = getattr(player, "show_generated_image", None)
            if callable(show):
                show(res["bytes"], res["prompt"])
            else:
                player.show_camera_frame(res["bytes"])
            player.write_log(f"JARVIS: 🖼 Image ready — {res['name']} (saved in generated_images)")
        except Exception:
            pass
    return (f"Done, sir. I drew it and put it on the screen. "
            f"It is saved as {res['name']} in the generated images folder.")
