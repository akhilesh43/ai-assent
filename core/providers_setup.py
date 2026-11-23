"""
core/providers_setup.py — run JARVIS on Gemini, on a model of your own, or on both.

The machinery lives in providers/: a chat-completions brain (Ollama, LM Studio, llama.cpp, vLLM,
Groq …), Whisper speech-to-text, Piper / system text-to-speech and a "sidecar" for one-shot
calls. This module is the ONE place that turns "what do I have?" into the settings for all of
them, so nobody has to hand-edit config/api_keys.json.

PRESETS
    auto          RECOMMENDED. Uses what you have — a Gemini key, a local model, or both —
                  and keeps working if one stops. Brain: Gemini first, your local model as the
                  automatic spare. Ears: Whisper on this computer. Voice: Piper, then the
                  computer's own. With a Gemini key it stays on Gemini Live (its own voice,
                  fastest); `--prefer private` runs the whole pipeline on this machine instead.
    local         Brain = your local model (Ollama / LM Studio …), ears = Whisper, voice = Piper.
                  No account and no internet needed.
    gemini-voice  Brain = Gemini through Google's chat-completions endpoint, ears = Whisper,
                  voice = Piper.   (EXPERIMENTAL: not exercised against the live endpoint.)
    backup        Keep Gemini Live exactly as it is. A local model only answers the BACKGROUND
                  jobs (summaries, code review …) when Gemini is out of quota.
    gemini        Plain Gemini Live again.

KEYS
    Stored in config/api_keys.json (already git-ignored) through memory.config_manager. Never
    printed in full, never logged.
"""
from __future__ import annotations

import platform
from typing import Any, Optional

GEMINI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_COMPAT_ID = "openai_compat@custom"
GEMINI_COMPAT_MODEL = "gemini-2.5-flash"

LOCAL_BASE = "http://localhost:11434/v1"          # Ollama's default address
LOCAL_ID = "openai_compat@ollama"
LOCAL_DEFAULT_MODEL = "qwen3:8b"

WHISPER_ID = "whisper"
PIPER_ID = "piper"

FAILOVER_LLM, FAILOVER_STT, FAILOVER_TTS = "failover_llm", "failover_stt", "failover_tts"
WHISPER_LOCAL = "whisper@local"
SYSTEM_TTS = "system_tts"

PRESETS = ("auto", "local", "gemini-voice", "backup", "gemini")
PREFER = ("speed", "private")


def mask(secret: str) -> str:
    s = (secret or "").strip()
    if len(s) <= 8:
        return "•" * len(s)
    return f"{s[:4]}…{s[-4:]}"


def normalise_prefer(prefer: str) -> str:
    """\"voice\" was the old default (it meant: prefer a paid human voice). Nothing replaces
    that, so a saved \"voice\" keeps meaning what it effectively did — stay on Gemini Live."""
    p = (prefer or "").strip().lower()
    return p if p in PREFER else "speed"


# ─────────────────────────────────────────────────────────────────────────────
#  Checks (network; `http` is injectable so tests use a fake)
# ─────────────────────────────────────────────────────────────────────────────
def check_gemini(http, key: str) -> tuple[bool, str]:
    """Does this Gemini key work? The key goes in a header, never the URL, so an error message
    can never carry it."""
    try:
        r = http.get("https://generativelanguage.googleapis.com/v1beta/models",
                     headers={"x-goog-api-key": key}, timeout=15)
    except Exception as e:
        return False, f"could not reach Google: {e.__class__.__name__}"
    if r.status_code == 200:
        return True, "key OK"
    if r.status_code in (400, 401, 403):
        return False, "Google says this key is not valid (copy it again from aistudio.google.com → API keys)."
    if r.status_code == 429:
        return True, "key OK (currently rate-limited — your local model will cover that)"
    return False, f"Google answered HTTP {r.status_code}"


def check_local(http, base_url: str = LOCAL_BASE, preferred_model: str = "") -> tuple[bool, str, Optional[str]]:
    """Is a model server answering at `base_url`, and which model should be used?
    Returns (ok, message, model)."""
    base = (base_url or LOCAL_BASE).rstrip("/")
    try:
        r = http.get(f"{base}/models", timeout=6)
    except Exception as e:
        return False, (f"nothing answered at {base} ({e.__class__.__name__}). "
                       "Start Ollama / LM Studio first."), None
    if r.status_code != 200:
        return False, f"{base} answered HTTP {r.status_code}", None
    try:
        data = r.json()
        ids = [str(m.get("id") or m.get("name") or "") for m in (data.get("data") or data.get("models") or [])]
    except Exception:
        ids = []
    ids = [i for i in ids if i]
    if preferred_model and preferred_model in ids:
        return True, f"server OK, model {preferred_model}", preferred_model
    if preferred_model and ids:
        return False, f"the server has no model called '{preferred_model}'. It has: {', '.join(ids[:8])}", None
    if ids:
        return True, f"server OK, using {ids[0]}", ids[0]
    return True, "server OK, but it lists no models — pull one first (e.g. `ollama pull qwen3:8b`)", None


# ─────────────────────────────────────────────────────────────────────────────
#  Building the settings (pure: returns what to write)
# ─────────────────────────────────────────────────────────────────────────────
def _local_cred(model: str, base_url: str = "") -> dict:
    return {"base_url": base_url or LOCAL_BASE, "api_key": "", "model": model or LOCAL_DEFAULT_MODEL}


def build_settings(preset: str, *, gemini_key: str = "", local_model: str = "", local_url: str = "",
                   whisper_model: str = "small") -> dict[str, Any]:
    """The arguments for `config_manager.save_providers` plus extras, for one preset."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {', '.join(PRESETS)}")
    if preset == "auto":
        return build_auto(gemini_key=gemini_key, local_model=local_model, local_url=local_url,
                          whisper_model=whisper_model)
    out: dict[str, Any] = {"creds": {}, "extra": {}, "notes": []}

    if preset == "gemini":
        out["mode"] = "gemini"
        out["extra"]["engine_auto"] = False
        return out

    if preset in ("backup", "local"):
        out["creds"][LOCAL_ID] = _local_cred(local_model, local_url)

    if preset == "backup":
        if not local_model:
            raise ValueError("the 'backup' preset needs a local model name (e.g. qwen3:8b)")
        out["sidecar"] = LOCAL_ID
        out["sidecar_first"] = False
        out["notes"].append("Gemini Live stays your assistant; your local model answers background "
                            "jobs when Gemini cannot.")
        out["extra"]["engine_auto"] = False
        return out

    # ── pipeline presets ─────────────────────────────────────────────────────
    if preset == "local":
        if not local_model:
            raise ValueError("the 'local' preset needs a local model name (e.g. qwen3:8b)")
        brain = LOCAL_ID
        out["sidecar"], out["sidecar_first"] = LOCAL_ID, True
    else:                                                      # gemini-voice
        if not gemini_key:
            raise ValueError("the 'gemini-voice' preset needs your Gemini key to be saved first")
        brain = GEMINI_COMPAT_ID
        out["creds"][GEMINI_COMPAT_ID] = {"base_url": GEMINI_COMPAT_BASE, "api_key": gemini_key,
                                          "model": GEMINI_COMPAT_MODEL}
        out["notes"].append("EXPERIMENTAL: Gemini through its chat-completions endpoint.")

    out["creds"][WHISPER_ID] = {"backend": "local", "model": whisper_model, "base_url": "", "api_key": ""}
    out["notes"].append(f"Speech-to-text: Whisper '{whisper_model}' on this computer (free).")
    out["notes"].append("Voice: Piper if installed, otherwise this computer's own voice.")
    out["mode"] = "pipeline"
    out["pipeline"] = {"stt": WHISPER_ID, "llm": brain, "tts": FAILOVER_TTS}
    out["creds"][FAILOVER_TTS] = {"chain": f"{PIPER_ID},{SYSTEM_TTS}"}
    out["extra"]["engine_auto"] = False
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Applying and reading back
# ─────────────────────────────────────────────────────────────────────────────
def can_start(data: dict) -> bool:
    """Can the app start with this config? It needs an OS name and ONE way to think: a Gemini
    key, or a pipeline whose brain is set (a local server needs no key; a hosted one does)."""
    if not isinstance(data, dict) or not data.get("os_system"):
        return False
    if str(data.get("gemini_api_key") or "").strip():
        return True
    prov = data.get("providers")
    if not isinstance(prov, dict) or str(prov.get("mode") or "").lower() != "pipeline":
        return False
    pipe = prov.get("pipeline") if isinstance(prov.get("pipeline"), dict) else {}
    brain = str(pipe.get("llm") or "").strip()
    if not brain:
        return False
    creds = prov.get("creds") if isinstance(prov.get("creds"), dict) else {}

    def usable(pid: str) -> bool:
        cfg = creds.get(pid) if isinstance(creds.get(pid), dict) else {}
        local = pid.split("@")[-1] in ("ollama", "lmstudio", "llamacpp", "vllm", "jan")
        return bool(str(cfg.get("api_key") or "").strip()) or local

    if brain.startswith("failover_"):
        chain = creds.get(brain, {}).get("chain", "") if isinstance(creds.get(brain), dict) else ""
        return any(usable(p.strip()) for p in str(chain).split(",") if p.strip())
    return usable(brain)


def apply_settings(settings: dict) -> None:
    """Write a build_settings() result in ONE config write, plus the OS name if missing."""
    from memory import config_manager as cm
    kwargs = {k: settings[k] for k in ("mode", "pipeline", "creds", "sidecar", "sidecar_first")
              if k in settings}
    cm.save_providers(**kwargs)
    patch = dict(settings.get("extra") or {})
    cfg = cm.load_api_keys()
    if not cfg.get("os_system"):
        patch["os_system"] = {"Windows": "windows", "Darwin": "mac"}.get(platform.system(), "linux")
    for k, v in list(patch.items()):
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            merged = dict(cfg[k]); merged.update(v); patch[k] = merged
    if patch:
        cm._patch_config(**patch)


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO: whatever you have, in any combination
# ─────────────────────────────────────────────────────────────────────────────
def build_auto(*, gemini_key: str = "", local_model: str = "", local_url: str = "",
               gemini_model: str = "", whisper_model: str = "small",
               faster_whisper: Optional[bool] = None, prefer: str = "speed") -> dict[str, Any]:
    """Settings for the 'auto' preset.

    Every part of the conversation gets a CHAIN of providers built from what exists, and each
    chain moves on to its next link by itself when one fails:

        brain   Gemini (through Google's chat-completions endpoint) → your local model
        ears    Whisper on this computer
        voice   Piper → this computer's voice

    One brain gives a chain of one and everything still works. A Gemini key keeps the app on
    Gemini Live (its own voice, and faster) unless `prefer="private"` asks for the all-local
    pipeline."""
    if not (gemini_key or local_model):
        raise ValueError("add a Gemini key or a local model name (or both)")
    prefer = normalise_prefer(prefer)
    if faster_whisper is None:
        from importlib.util import find_spec
        faster_whisper = find_spec("faster_whisper") is not None

    creds: dict[str, dict] = {}
    notes: list[str] = []

    # ── brain ────────────────────────────────────────────────────────────────
    llm_chain: list[str] = []
    if gemini_key:
        creds[GEMINI_COMPAT_ID] = {"base_url": GEMINI_COMPAT_BASE, "api_key": gemini_key,
                                   "model": gemini_model or GEMINI_COMPAT_MODEL}
        llm_chain.append(GEMINI_COMPAT_ID)
    if local_model:
        creds[LOCAL_ID] = _local_cred(local_model, local_url)
        llm_chain.append(LOCAL_ID)
    notes.append("Brain: " + " → ".join("Gemini" if p == GEMINI_COMPAT_ID else "local model" for p in llm_chain)
                 + (" (the second is the automatic spare)" if len(llm_chain) > 1 else ""))

    # ── ears ─────────────────────────────────────────────────────────────────
    creds[WHISPER_LOCAL] = {"backend": "local", "model": whisper_model, "base_url": "", "api_key": ""}
    stt_chain = [WHISPER_LOCAL]
    if not faster_whisper:
        notes.append("WARNING: faster-whisper is not installed, so the pipeline has no ears yet. "
                     "Run: pip install faster-whisper   (Gemini Live does not need it.)")
    notes.append("Ears: local Whisper")

    # ── voice ────────────────────────────────────────────────────────────────
    tts_chain = [PIPER_ID, SYSTEM_TTS]
    notes.append("Voice: Piper → computer voice")

    creds[FAILOVER_LLM] = {"chain": ",".join(llm_chain)}
    creds[FAILOVER_STT] = {"chain": ",".join(stt_chain)}
    creds[FAILOVER_TTS] = {"chain": ",".join(tts_chain)}

    # ── which engine ─────────────────────────────────────────────────────────
    live = bool(gemini_key) and prefer == "speed"
    out: dict[str, Any] = {"creds": creds, "notes": notes,
                           "extra": {"engine_auto": True, "engine_auto_prefer": prefer}}
    if live:
        out["mode"] = "gemini"
        notes.append("Engine: Gemini Live (its own voice, fastest).")
    else:
        out["mode"] = "pipeline"
        out["pipeline"] = {"stt": FAILOVER_STT, "llm": FAILOVER_LLM, "tts": FAILOVER_TTS}
        notes.append("Engine: ears → brain → voice (about 1.5–4 s per reply).")

    # background jobs: the local model is the spare when there is a Gemini key, the only brain when not
    if local_model:
        out["sidecar"], out["sidecar_first"] = LOCAL_ID, (not gemini_key)
    else:
        out["sidecar"], out["sidecar_first"] = "", False
    out["extra"]["image_gen"] = {"backends": (["gemini"] if gemini_key else []) + ["pollinations"]}
    return out


# Services this build no longer ships. A config written by an older build may still name them;
# scrub_removed() rewrites those names to something that exists so the app starts instead of
# reporting "no provider called 'elevenlabs'".
REMOVED_IDS = frozenset({"tts_openai", "whisper@openai", "openai_compat@openai"})


def scrub_removed() -> bool:
    """Drop every reference to a removed service from the saved config. Idempotent; returns True
    when something was rewritten."""
    from memory import config_manager as cm
    cfg = cm.load_api_keys()
    block = cfg.get("providers")
    patch: dict[str, Any] = {}
    changed = False
    if isinstance(block, dict):
        block = dict(block)
        creds = dict(block.get("creds") or {})
        for pid in list(creds):
            if pid in REMOVED_IDS:
                del creds[pid]; changed = True
        for fo in (FAILOVER_LLM, FAILOVER_STT, FAILOVER_TTS):
            c = creds.get(fo)
            if isinstance(c, dict) and c.get("chain"):
                kept = [x.strip() for x in str(c["chain"]).split(",") if x.strip() and x.strip() not in REMOVED_IDS]
                if ",".join(kept) != str(c["chain"]).replace(" ", ""):
                    creds[fo] = dict(c, chain=",".join(kept)); changed = True
        block["creds"] = creds
        pipe = dict(block.get("pipeline") or {})
        has_local = LOCAL_ID in creds
        for slot, fallback in (("tts", SYSTEM_TTS), ("stt", WHISPER_ID),
                               ("llm", LOCAL_ID if has_local else "")):
            if str(pipe.get(slot) or "") in REMOVED_IDS:
                pipe[slot] = fallback; changed = True
        if str(block.get("sidecar") or "") in REMOVED_IDS:
            block["sidecar"] = ""; block["sidecar_first"] = False; changed = True
        block["pipeline"] = pipe
        if changed:
            if (block.get("mode") == "pipeline" and not pipe.get("llm")
                    and str(cfg.get("gemini_api_key") or "").strip()):
                block["mode"] = "gemini"                 # nothing left to think with except Gemini
            patch["providers"] = block
    ig = cfg.get("image_gen")
    if isinstance(ig, dict) and "openai" in (ig.get("backends") or []):
        patch["image_gen"] = dict(ig, backends=[b for b in ig["backends"] if b != "openai"])
        changed = True
    if patch:
        cm._patch_config(**patch)
    return changed


def _auto_inputs() -> dict[str, Any]:
    """Everything build_auto needs, read from the saved config."""
    from memory import config_manager as cm
    cfg = cm.load_api_keys()
    creds = cm.get_providers().get("creds")
    creds = creds if isinstance(creds, dict) else {}

    def c(pid: str, key: str) -> str:
        v = creds.get(pid)
        return str(v.get(key) or "").strip() if isinstance(v, dict) else ""
    return {
        "gemini_key": str(cfg.get("gemini_api_key") or "").strip(),
        "local_model": c(LOCAL_ID, "model"),
        "local_url": c(LOCAL_ID, "base_url"),
        "gemini_model": c(GEMINI_COMPAT_ID, "model"),
        "whisper_model": c(WHISPER_LOCAL, "model") or "small",
        "prefer": normalise_prefer(str(cfg.get("engine_auto_prefer") or "speed")),
    }


def sync_auto(force: bool = False) -> bool:
    """Called at every start. If auto mode is on, rebuild the engine settings from what is saved
    NOW — so adding a local model or removing the Gemini key needs no command. Returns True if
    anything was rewritten.

    It stands down in two cases: auto mode is off, or you changed the engine or pipeline by hand
    in the settings panel since it last ran (then your choice wins and auto mode switches itself
    off, rather than overwriting it every start)."""
    from importlib.util import find_spec
    from memory import config_manager as cm

    try:
        scrub_removed()
    except Exception:
        pass
    cfg = cm.load_api_keys()
    if not cfg.get("engine_auto"):
        return False
    prov = cm.get_providers()
    applied = cfg.get("engine_auto_applied")
    if isinstance(applied, dict) and not force:
        if prov.get("mode") != applied.get("mode") or prov.get("pipeline") != applied.get("pipeline"):
            cm._patch_config(engine_auto=False)         # the user took over by hand
            return False

    inp = _auto_inputs()
    if not (inp["gemini_key"] or inp["local_model"]):
        return False                                    # nothing to build from; leave the config alone
    fw = find_spec("faster_whisper") is not None
    settings = build_auto(**inp, faster_whisper=fw)
    sig = _settings_sig(settings)
    if not force and cfg.get("engine_auto_sig") == sig:
        return False

    apply_settings(settings)
    record_auto_applied(sig)
    return True


def _settings_sig(settings: dict) -> str:
    """Fingerprint of what build_auto would write. Taken from the SETTINGS rather than the inputs
    so that filling in a default (a model name) does not look like a change next time."""
    import hashlib
    import json
    body = {k: settings.get(k) for k in ("mode", "pipeline", "creds", "sidecar", "sidecar_first")}
    body["image"] = (settings.get("extra") or {}).get("image_gen")
    return hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def record_auto_applied(sig: str = "") -> None:
    """Remember what auto mode wrote, so a LATER hand-edit of the engine can be told apart from it."""
    from importlib.util import find_spec
    from memory import config_manager as cm
    if not sig:
        inp = _auto_inputs()
        if inp["gemini_key"] or inp["local_model"]:
            sig = _settings_sig(build_auto(**inp, faster_whisper=find_spec("faster_whisper") is not None))
    now = cm.get_providers()
    cm._patch_config(engine_auto_sig=sig,
                     engine_auto_applied={"mode": now.get("mode"), "pipeline": now.get("pipeline")})


def apply_initial_keys(local_model: str = "", local_url: str = "", prefer: str = "speed") -> bool:
    """First-run dialog: save the local model if one was named and let auto mode decide the rest."""
    from memory import config_manager as cm
    if local_model.strip():
        cm.save_providers(creds={LOCAL_ID: _local_cred(local_model.strip(), local_url.strip())})
    cm._patch_config(engine_auto=True, engine_auto_prefer=normalise_prefer(prefer))
    return sync_auto(force=True)


def current_status() -> dict:
    """What is configured now, with every key masked."""
    from memory import config_manager as cm
    cfg = cm.load_api_keys()
    prov = cm.get_providers()
    creds = prov.get("creds") if isinstance(prov.get("creds"), dict) else {}
    keys = {"gemini": mask(str(cfg.get("gemini_api_key") or ""))}
    for pid, c in creds.items():
        if isinstance(c, dict) and c.get("api_key"):
            keys[pid] = mask(str(c["api_key"]))
    return {"mode": cm.get_provider_mode(), "pipeline": cm.get_pipeline(), "keys": keys,
            "engine_auto": bool(cfg.get("engine_auto")),
            "chains": {k: str((creds.get(k) or {}).get("chain") or "") for k in
                       (FAILOVER_LLM, FAILOVER_STT, FAILOVER_TTS) if isinstance(creds.get(k), dict)},
            "sidecar": cm.get_sidecar_provider(), "sidecar_first": cm.get_sidecar_first(),
            "can_start": can_start(cfg)}
