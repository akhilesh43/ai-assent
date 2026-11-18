"""
Run JARVIS on Gemini, on a model of your own (Ollama, LM Studio …), or on both.

    python enable_providers.py --status
    python enable_providers.py --gemini-key - --local-model qwen3:8b     # AUTO: Gemini + local spare
    python enable_providers.py --preset local --local-model qwen3:8b     # everything on this PC
    python enable_providers.py --preset backup --local-model qwen3:8b    # Gemini stays; local is the spare
    python enable_providers.py --preset gemini-voice                     # Gemini brain + Piper voice (experimental)
    python enable_providers.py --preset gemini                           # back to plain Gemini Live

Presets (details in core/providers_setup.py):
  auto          (the default when you pass a key or model) Uses what you have, any combination:
                  brain  Gemini → local model     ears  Whisper on this PC
                  voice  Piper → this computer's voice
                A failing brain never stops the assistant — the next one takes over — and the choice
                is re-made from your saved settings at every start. A Gemini key stays on Gemini Live
                (fastest). --prefer private runs the whole pipeline on this machine instead.
  local         Local model brain · Whisper ears · Piper voice. No account, no internet.
  gemini-voice  Gemini brain via its chat-completions endpoint · Whisper · Piper voice.
  backup        Gemini Live unchanged; a local model answers background jobs when Gemini is out of quota.
  gemini        plain Gemini Live again.

Keys: pass them, set GEMINI_API_KEY, or use '-' to be asked (hidden — keeps them out of your shell
history). The Gemini key and the local server are TESTED before anything is saved (skip with
--no-check). Restart JARVIS afterwards.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from typing import Callable, Optional

from core import providers_setup as ps


def _resolve_key(arg: Optional[str], env: str, stored: str, label: str,
                 ask: Callable[[str], str]) -> str:
    if arg == "-":
        return ask(f"{label} (hidden): ").strip()
    if arg:
        return arg.strip()
    if os.environ.get(env, "").strip():
        return os.environ[env].strip()
    return stored.strip()


def main(argv: Optional[list[str]] = None, *, http=None, ask: Callable[[str], str] = getpass.getpass,
         out: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--preset", choices=ps.PRESETS)
    ap.add_argument("--gemini-key")
    ap.add_argument("--local-model", default="", help="model name on your local server, e.g. qwen3:8b")
    ap.add_argument("--local-url", default="", help=f"default {ps.LOCAL_BASE} (Ollama)")
    ap.add_argument("--prefer", choices=ps.PREFER, default="speed",
                    help="auto: speed = Gemini Live when there is a Gemini key; private = all-local pipeline")
    ap.add_argument("--whisper-model", default="small", choices=("tiny", "base", "small", "medium"))
    ap.add_argument("--no-check", action="store_true", help="save without testing")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)

    from memory import config_manager as cm

    if a.status:
        s = ps.current_status()
        out(f"Engine mode   : {s['mode']}  (gemini = Gemini Live · pipeline = ears → brain → voice)"
            + ("   [AUTO: re-decided from your settings at every start]" if s.get("engine_auto") else ""))
        for k, v in (s.get("chains") or {}).items():
            out(f"  {k:<13}: {v.replace(',', ' → ')}")
        out(f"Pipeline      : {s['pipeline'] or '—'}")
        out(f"Side jobs via : {s['sidecar'] or 'Gemini only'}{'  (tried FIRST)' if s['sidecar_first'] else ''}")
        out("Keys saved    : " + (", ".join(f"{k}={v}" for k, v in s["keys"].items() if v) or "none"))
        out(f"Can start     : {'yes' if s['can_start'] else 'NO — no Gemini key and no pipeline brain configured'}")
        return 0

    if http is None:
        try:
            import requests as http                                    # type: ignore
        except ImportError:
            out("✗ pip install requests"); return 2

    cfg_now = cm.load_api_keys()
    gem = _resolve_key(a.gemini_key, "GEMINI_API_KEY", str(cfg_now.get("gemini_api_key") or ""),
                       "Gemini key", ask)
    creds = cm.get_providers().get("creds") or {}
    saved_local = creds.get(ps.LOCAL_ID) if isinstance(creds.get(ps.LOCAL_ID), dict) else {}
    local_model = (a.local_model or str(saved_local.get("model") or "")).strip()
    local_url = (a.local_url or str(saved_local.get("base_url") or "")).strip()

    if not a.preset:
        if a.gemini_key or a.local_model:
            a.preset = "auto"                                  # given something, no preset named
        else:
            ap.print_help(); return 2
    if a.preset == "auto" and not (gem or local_model):
        out("✗ Give at least one of: --gemini-key (or '-' to type it hidden), --local-model NAME"); return 2
    if a.preset in ("local", "backup") and not local_model:
        out("✗ This preset needs a local model: --local-model qwen3:8b"); return 2
    if a.preset == "gemini-voice" and not gem:
        out("✗ 'gemini-voice' uses your Gemini key, and none is saved yet."); return 2

    # ── test before saving anything ──────────────────────────────────────────
    uses_gemini = a.preset in ("auto", "gemini-voice") and bool(gem)
    uses_local = a.preset in ("auto", "local", "backup") and bool(local_model)
    if uses_gemini:
        out(f"Gemini key {ps.mask(gem)}")
    if not a.no_check:
        if uses_gemini:
            ok, msg = ps.check_gemini(http, gem)
            out(("✓ Gemini: " if ok else "✗ Gemini: ") + msg)
            if not ok:
                out("  Nothing was saved. Fix the key (or add --no-check to save anyway)."); return 1
        if uses_local:
            ok, msg, picked = ps.check_local(http, local_url or ps.LOCAL_BASE, local_model)
            out(("✓ Local model: " if ok else "✗ Local model: ") + msg)
            if not ok:
                out("  Nothing was saved. Start the server / fix the name (or add --no-check)."); return 1

    try:
        if a.preset == "auto":
            if gem:
                cm._patch_config(gemini_api_key=gem)           # the one place the Gemini key lives
            settings = ps.build_auto(gemini_key=gem, local_model=local_model, local_url=local_url,
                                     whisper_model=a.whisper_model, prefer=a.prefer)
        else:
            settings = ps.build_settings(a.preset, gemini_key=gem, local_model=local_model,
                                         local_url=local_url, whisper_model=a.whisper_model)
        ps.apply_settings(settings)
        if a.preset == "auto":                      # remember what auto wrote, so a later hand-edit is respected
            ps.record_auto_applied()
    except ValueError as e:
        out(f"✗ {e}"); return 2

    out("\nSaved. What is set now:")
    for n in settings.get("notes", []):
        out(f"  • {n}")
    s = ps.current_status()
    out(f"  • mode={s['mode']}  pipeline={s['pipeline'] or '—'}  side jobs via={s['sidecar'] or 'Gemini'}")
    for k, v in (s.get("chains") or {}).items():
        out(f"  • {k}: {v.replace(',', ' → ')}")
    if s["mode"] == "pipeline":
        out("  • Replies take ~1.5–4 s in this mode (ears → brain → voice). Gemini Live is faster; "
            "'--preset gemini' (or --prefer speed) switches back any time.")
    out("\nRestart JARVIS to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
