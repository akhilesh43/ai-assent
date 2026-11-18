"""
core/image_gen.py — JARVIS draws its own pictures.

One function, `generate_image()`, used by both the voice tool (plugins/image_studio.py)
and the phone/web dashboard (dashboard/server.py). Files land in
`generated_images/` next to main.py, with a small index.json holding the prompts.

BACKENDS, tried in the order of config["image_gen"]["backends"]
    "gemini"        Gemini image models through the SAME key and client factory as the
                    rest of the app (core.gemini.client). Several model names are tried
                    in turn because Google renames preview models; a model that does
                    not exist, or has no quota on this key, simply hands over to the
                    next one. NOTE: image generation is not available on every free
                    Gemini key — when it is not, you get a quota error and the next
                    backend runs.
    "pollinations"  Free, no key. Anonymous use is rate-limited (about one image per
                    15 s) and may carry a watermark; an optional "pollinations_key"
                    in the config removes both. Needs internet, nothing else.

Config (config/api_keys.json -> "image_gen"), every field optional:
    {"backends": ["gemini", "pollinations"],
     "gemini_models": ["gemini-3.1-flash-image", "gemini-2.5-flash-image",
                       "gemini-3-pro-image-preview"],
     "pollinations_key": "", "translate": true}

Never raises: returns {"ok": False, "error": "..."} instead.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "generated_images"

DEFAULT_BACKENDS = ("gemini", "pollinations")
DEFAULT_GEMINI_MODELS = ("gemini-3.1-flash-image", "gemini-2.5-flash-image",
                         "gemini-3-pro-image-preview")
ASPECTS = {                      # aspect -> (width, height) for the pixel-sized backend
    "1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344),
    "4:3": (1152, 864), "3:4": (864, 1152),
}
_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp"}
_lock = threading.Lock()          # one generation at a time: the free tier allows no more
_listeners: list[Callable[[dict], None]] = []


def add_listener(fn: Callable[[dict], None]) -> None:
    """Called (on the generating thread) with {"name","prompt","backend"} after every
    successful picture — how the phone dashboard learns about one made by voice."""
    if fn not in _listeners:
        _listeners.append(fn)


def _notify(info: dict) -> None:
    for fn in list(_listeners):
        try:
            fn(info)
        except Exception:
            pass


def _cfg() -> dict:
    try:
        from memory.config_manager import load_api_keys
        c = load_api_keys().get("image_gen")
        return c if isinstance(c, dict) else {}
    except Exception:
        return {}


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (s[:40] or "image")


def _save(data: bytes, mime: str, prompt: str, backend: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = _EXT.get((mime or "").lower().split(";")[0].strip(), ".png")
    name = f"img_{datetime.now():%Y%m%d_%H%M%S}_{_slug(prompt)}{ext}"
    path = out_dir / name
    n = 1
    while path.exists():
        path = out_dir / f"{path.stem}_{n}{ext}"
        n += 1
    path.write_bytes(data)
    try:
        idx = out_dir / "index.json"
        rec = json.loads(idx.read_text(encoding="utf-8")) if idx.exists() else []
        rec.append({"file": path.name, "prompt": prompt, "backend": backend,
                    "ts": datetime.now().isoformat(timespec="seconds")})
        idx.write_text(json.dumps(rec[-500:], ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass                                  # the picture is saved; the caption is a bonus
    return path


def _looks_like_image(data: bytes) -> bool:
    return (data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff"
            or (data[:4] == b"RIFF" and data[8:12] == b"WEBP"))


# ── backend 1: Gemini ────────────────────────────────────────────────────────
def _gemini_backend(prompt: str, aspect: str, cfg: dict, log) -> tuple[bytes, str, str]:
    """Returns (bytes, mime, model). Raises RuntimeError(reason) when nothing worked."""
    from core import gemini as G
    from google.genai import types

    models = [m for m in (cfg.get("gemini_models") or DEFAULT_GEMINI_MODELS) if m]
    client = G.client(timeout_ms=90_000)
    last = "no model answered"
    for model in models:
        for with_aspect in (True, False):
            try:
                kw = {"response_modalities": ["IMAGE", "TEXT"]}
                if with_aspect:
                    kw["image_config"] = types.ImageConfig(aspect_ratio=aspect)
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(**kw))
                for cand in (getattr(resp, "candidates", None) or []):
                    for part in (getattr(getattr(cand, "content", None), "parts", None) or []):
                        blob = getattr(part, "inline_data", None)
                        data = getattr(blob, "data", None) if blob else None
                        if data:
                            if isinstance(data, str):
                                import base64
                                data = base64.b64decode(data)
                            return bytes(data), getattr(blob, "mime_type", "image/png"), model
                last = f"{model}: answered with text only (prompt may have been refused)"
                break                                   # same model again would not differ
            except (TypeError, ValueError) as e:        # old SDK without image_config
                last = f"{model}: {str(e)[:120]}"
                if with_aspect:
                    continue
                break
            except Exception as e:
                last = f"{model}: {str(e)[:160]}"
                break                                   # quota / not found → next model
        log(f"IMG: {last}")
    raise RuntimeError(last)


# ── backend 2: Pollinations ──────────────────────────────────────────────────
_last_pollinations = 0.0


def _to_english(prompt: str, log) -> str:
    """Pollinations models follow English far better; Hindi/Hinglish go through a
    short Gemini rewrite. Any failure keeps the original wording."""
    try:
        from core import gemini as G
        out = G.text(
            "Rewrite this as one vivid English image-generation prompt (max 50 words). "
            "Reply with ONLY the prompt.\n\n" + prompt, tier=G.FAST, timeout_ms=10_000)
        out = (out or "").strip().strip('"')
        return out if 3 <= len(out) <= 600 else prompt
    except Exception as e:
        log(f"IMG: translate skipped ({str(e)[:80]})")
        return prompt


def _pollinations_backend(prompt: str, aspect: str, cfg: dict, log) -> tuple[bytes, str, str]:
    import requests
    from urllib.parse import quote
    global _last_pollinations

    text = _to_english(prompt, log) if cfg.get("translate", True) else prompt
    w, h = ASPECTS.get(aspect, ASPECTS["1:1"])
    key = str(cfg.get("pollinations_key") or "").strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    params = {"width": w, "height": h, "model": "flux", "nologo": "true", "safe": "true",
              "seed": random.randint(1, 10**6)}
    url = f"https://image.pollinations.ai/prompt/{quote(text)}"
    last = "no answer"
    for attempt in range(2):
        if not key:                                   # anonymous: ~1 request / 15 s
            wait = 15.5 - (time.monotonic() - _last_pollinations)
            if wait > 0:
                log(f"IMG: waiting {wait:.0f}s for the free tier…")
                time.sleep(min(wait, 16.0))
        try:
            _last_pollinations = time.monotonic()
            r = requests.get(url, params=params, headers=headers, timeout=120)
            ctype = r.headers.get("content-type", "")
            if r.status_code == 200 and ctype.startswith("image/") and _looks_like_image(r.content):
                return r.content, ctype, "pollinations/flux"
            last = f"HTTP {r.status_code} {ctype[:30]} {r.text[:80] if 'text' in ctype else ''}"
            if r.status_code in (401, 403):
                break                                 # needs a key now; retrying will not help
        except Exception as e:
            last = str(e)[:140]
    raise RuntimeError(f"pollinations: {last}")


_BACKENDS: dict[str, Callable] = {"gemini": _gemini_backend,
                                  "pollinations": _pollinations_backend}


# ── public API ───────────────────────────────────────────────────────────────
def list_images(limit: int = 60, out_dir: Optional[Path] = None) -> list[dict]:
    """Newest first: [{"name", "prompt", "backend", "ts", "size"}]."""
    out_dir = out_dir or OUT_DIR
    meta: dict[str, dict] = {}
    try:
        for r in json.loads((out_dir / "index.json").read_text(encoding="utf-8")):
            meta[r.get("file", "")] = r
    except Exception:
        pass
    items = []
    try:
        files = [p for p in out_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            m = meta.get(p.name, {})
            items.append({"name": p.name, "prompt": m.get("prompt", ""),
                          "backend": m.get("backend", ""), "ts": m.get("ts", ""),
                          "size": p.stat().st_size})
    except Exception:
        pass
    return items


def generate_image(prompt: str, style: str = "", aspect: str = "1:1",
                   log: Optional[Callable[[str], None]] = None,
                   out_dir: Optional[Path] = None) -> dict:
    """Make one picture. Returns
       {"ok": True,  "path": str, "name": str, "backend": str, "prompt": str, "bytes": bytes}
       {"ok": False, "error": str}"""
    log = log or (lambda s: None)
    prompt = (prompt or "").strip()
    if len(prompt) < 3:
        return {"ok": False, "error": "Please describe the picture you want."}
    aspect = aspect if aspect in ASPECTS else "1:1"
    full = prompt if not style.strip() else f"{prompt}. Style: {style.strip()}"
    cfg = _cfg()
    order = [b for b in (cfg.get("backends") or DEFAULT_BACKENDS) if b in _BACKENDS]
    if not order:
        order = list(DEFAULT_BACKENDS)
    if not _lock.acquire(timeout=1.0):
        return {"ok": False, "error": "Another image is still being drawn — try again in a moment."}
    try:
        errors = []
        for name in order:
            try:
                log(f"IMG: trying {name}…")
                data, mime, model = _BACKENDS[name](full, aspect, cfg, log)
                if not data or not _looks_like_image(data):
                    raise RuntimeError("backend returned something that is not an image")
                path = _save(data, mime, prompt, model, out_dir or OUT_DIR)
                _notify({"name": path.name, "prompt": prompt, "backend": model})
                return {"ok": True, "path": str(path), "name": path.name, "backend": model,
                        "prompt": prompt, "bytes": data}
            except Exception as e:
                errors.append(f"{name}: {str(e)[:140]}")
                log(f"IMG: {name} failed — {str(e)[:140]}")
        return {"ok": False, "error": "No image backend worked. " + " | ".join(errors)}
    finally:
        _lock.release()
