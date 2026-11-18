"""
Firebase Realtime Database + Cloud Firestore bidirectional memory & adaptive
command/voice training sync for JARVIS.

Uses the installed ``google-auth`` + ``requests`` packages directly against the
official REST APIs with ``config/firebase_service_account.json`` (no extra heavy
SDK required):
  - Firebase Realtime Database (RTDB): real-time command training shortcuts,
    phonetic voice aliases for unclear-speech recovery, and turn-by-turn
    communication training updates (plus background SSE stream listener).
  - Cloud Firestore (with automatic RTDB mirror fallback if Firestore hasn't
    been initialized yet): stores high-importance long-term memories & rules.
"""
from __future__ import annotations

from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import queue
import re
import sys
import threading
import time
from typing import Any, Callable
from urllib.parse import quote

import requests


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


SERVICE_ACCOUNT_PATH: Path = _base_dir() / "config" / "firebase_service_account.json"

_SCOPES = (
    "https://www.googleapis.com/auth/datastore",
    "https://www.googleapis.com/auth/firebase.database",
    "https://www.googleapis.com/auth/userinfo.email",
)

_TOKEN_LOCK = threading.Lock()
_CREDS: Any = None
_PROJECT_ID: str = ""
_RTDB_URL: str = ""
_FIRESTORE_AVAILABLE: bool | None = None
_FIRESTORE_WARNED = False

_WRITE_QUEUE: queue.Queue = queue.Queue(maxsize=256)
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()
_STREAM_STARTED = False
_ON_REMOTE_UPDATE: Callable[[str], None] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9\u0900-\u097F]+", "_", (text or "").strip().lower()).strip("_")
    return (s or "item")[:64]


def _phonetic_norm(text: str) -> str:
    """Normalize English/Hinglish text phonetically so slightly misheard voice matches."""
    s = re.sub(r"[^a-z0-9\u0900-\u097F\s]+", " ", (text or "").lower())
    s = re.sub(r"\b(?:jarvis|please|plz|zara|ek bar|bhai|yaar)\b", " ", s)
    s = s.replace("ph", "f").replace("v", "w").replace("z", "s").replace("q", "k")
    s = re.sub(r"([a-z])\1+", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def is_configured() -> bool:
    return SERVICE_ACCOUNT_PATH.is_file()


def _ensure_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        _WORKER_STARTED = True

        def _run() -> None:
            while True:
                fn, args, kwargs = _WRITE_QUEUE.get()
                try:
                    fn(*args, **kwargs)
                except Exception as exc:
                    print(f"[Firebase] sync worker error: {exc}")
                finally:
                    _WRITE_QUEUE.task_done()

        t = threading.Thread(target=_run, daemon=True, name="FirebaseWriteWorker")
        t.start()


def _enqueue(fn: Callable, *args, **kwargs) -> None:
    _ensure_worker()
    try:
        _WRITE_QUEUE.put_nowait((fn, args, kwargs))
    except queue.Full:
        pass


def _get_auth_headers() -> tuple[dict[str, str], str, str] | tuple[None, str, str]:
    global _CREDS, _PROJECT_ID, _RTDB_URL
    if not SERVICE_ACCOUNT_PATH.is_file():
        return None, "", ""
    with _TOKEN_LOCK:
        if _CREDS is None:
            from google.oauth2 import service_account

            raw = json.loads(SERVICE_ACCOUNT_PATH.read_text(encoding="utf-8"))
            _PROJECT_ID = str(raw.get("project_id") or "").strip()
            custom_rtdb = ""
            try:
                from memory.config_manager import load_api_keys
                custom_rtdb = str(load_api_keys().get("firebase_rtdb_url") or "").strip()
            except Exception:
                pass
            _RTDB_URL = (custom_rtdb or f"https://{_PROJECT_ID}-default-rtdb.firebaseio.com").rstrip("/")
            _CREDS = service_account.Credentials.from_service_account_file(
                str(SERVICE_ACCOUNT_PATH), scopes=list(_SCOPES)
            )
        if not _CREDS.valid:
            from google.auth.transport.requests import Request
            _CREDS.refresh(Request())
        return {"Authorization": f"Bearer {_CREDS.token}"}, _PROJECT_ID, _RTDB_URL


def get_status() -> dict[str, Any]:
    return {
        "configured": is_configured(),
        "project_id": _PROJECT_ID or (
            json.loads(SERVICE_ACCOUNT_PATH.read_text(encoding="utf-8")).get("project_id", "")
            if is_configured() else ""
        ),
        "rtdb_url": _RTDB_URL,
        "firestore_available": _FIRESTORE_AVAILABLE,
        "streaming": _STREAM_STARTED,
    }


# ---------------------------------------------------------------------------
# Realtime Database (RTDB) REST helpers
# ---------------------------------------------------------------------------

def rtdb_get(path: str, timeout: float = 6.0) -> Any:
    hdr, _pid, base = _get_auth_headers()
    if not hdr or not base:
        return None
    clean = path.strip("/")
    url = f"{base}/{clean}.json" if clean else f"{base}/.json"
    r = requests.get(url, headers=hdr, timeout=timeout)
    if r.status_code == 200:
        return r.json()
    return None


def rtdb_put(path: str, payload: Any, timeout: float = 6.0) -> bool:
    hdr, _pid, base = _get_auth_headers()
    if not hdr or not base:
        return False
    clean = path.strip("/")
    url = f"{base}/{clean}.json"
    r = requests.put(url, headers=hdr, json=payload, timeout=timeout)
    return 200 <= r.status_code < 300


def rtdb_patch(path: str, payload: dict[str, Any], timeout: float = 6.0) -> bool:
    hdr, _pid, base = _get_auth_headers()
    if not hdr or not base:
        return False
    clean = path.strip("/")
    url = f"{base}/{clean}.json"
    r = requests.patch(url, headers=hdr, json=payload, timeout=timeout)
    return 200 <= r.status_code < 300


def rtdb_delete(path: str, timeout: float = 6.0) -> bool:
    hdr, _pid, base = _get_auth_headers()
    if not hdr or not base:
        return False
    clean = path.strip("/")
    url = f"{base}/{clean}.json" if clean else f"{base}/.json"
    r = requests.delete(url, headers=hdr, timeout=timeout)
    return 200 <= r.status_code < 300


# ---------------------------------------------------------------------------
# Cloud Firestore REST helpers (with automatic RTDB mirror fallback)
# ---------------------------------------------------------------------------

def _to_fs_fields(doc: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for k, v in doc.items():
        ks = str(k)
        if isinstance(v, bool):
            fields[ks] = {"booleanValue": v}
        elif isinstance(v, int):
            fields[ks] = {"integerValue": str(v)}
        elif isinstance(v, float):
            fields[ks] = {"doubleValue": float(v)}
        elif isinstance(v, (list, tuple)):
            fields[ks] = {"arrayValue": {"values": [{"stringValue": str(x)} for x in v]}}
        else:
            fields[ks] = {"stringValue": str(v)}
    return fields


def _from_fs_fields(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, val in (fields or {}).items():
        if not isinstance(val, dict):
            continue
        if "stringValue" in val:
            out[k] = val["stringValue"]
        elif "booleanValue" in val:
            out[k] = bool(val["booleanValue"])
        elif "integerValue" in val:
            out[k] = int(val["integerValue"])
        elif "doubleValue" in val:
            out[k] = float(val["doubleValue"])
        elif "arrayValue" in val:
            out[k] = [x.get("stringValue", "") for x in val["arrayValue"].get("values", [])]
    return out


def firestore_set(collection: str, doc_id: str, doc: dict[str, Any], timeout: float = 6.0) -> bool:
    global _FIRESTORE_AVAILABLE, _FIRESTORE_WARNED
    hdr, pid, _base = _get_auth_headers()
    if not hdr or not pid:
        return False
    safe_id = _slug(doc_id)
    url = (
        f"https://firestore.googleapis.com/v1/projects/{pid}"
        f"/databases/(default)/documents/{quote(collection)}/{quote(safe_id)}"
    )
    try:
        r = requests.patch(url, headers=hdr, json={"fields": _to_fs_fields(doc)}, timeout=timeout)
        if 200 <= r.status_code < 300:
            _FIRESTORE_AVAILABLE = True
            return True
        if r.status_code == 404:
            _FIRESTORE_AVAILABLE = False
            if not _FIRESTORE_WARNED:
                _FIRESTORE_WARNED = True
                print("[Firebase] Firestore DB not provisioned (404) — mirroring important memories to RTDB + local DB.")
            return False
    except Exception:
        pass
    return False


def firestore_list(collection: str, page_size: int = 40, timeout: float = 6.0) -> list[dict[str, Any]]:
    global _FIRESTORE_AVAILABLE
    hdr, pid, _base = _get_auth_headers()
    if not hdr or not pid:
        return []
    url = (
        f"https://firestore.googleapis.com/v1/projects/{pid}"
        f"/databases/(default)/documents/{quote(collection)}?pageSize={int(page_size)}"
    )
    try:
        r = requests.get(url, headers=hdr, timeout=timeout)
        if r.status_code == 200:
            _FIRESTORE_AVAILABLE = True
            docs = r.json().get("documents") or []
            return [_from_fs_fields(d.get("fields", {})) for d in docs]
        if r.status_code == 404:
            _FIRESTORE_AVAILABLE = False
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# High-level Training & Memory APIs (local-first + cloud async)
# ---------------------------------------------------------------------------

def train_important_memory(
    key: str,
    value: str,
    category: str = "notes",
    importance: float = 0.9,
) -> None:
    """Save an important long-term memory locally AND to Firestore + RTDB."""
    from memory import local_db
    local_db.save_fact(key, value, category=category, confidence=importance)
    if not is_configured():
        return
    slug = _slug(key)
    payload = {
        "key": str(key).strip().lower()[:80],
        "value": str(value).strip()[:400],
        "category": str(category or "notes").strip().lower()[:40],
        "importance": float(importance),
        "updated_at": _now_iso(),
    }

    def _sync() -> None:
        firestore_set("jarvis_memories", slug, payload)
        rtdb_put(f"jarvis/important_memories/{slug}", payload)

    _enqueue(_sync)


def train_command(
    phrase: str,
    target_action: str,
    aliases: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Train a spoken command/shortcut & phonetic aliases locally + in Firebase RTDB/Firestore."""
    from memory import local_db

    p = re.sub(r"\s+", " ", (phrase or "").strip())
    act = (target_action or "").strip()
    if not p or not act:
        return {}
    al = [re.sub(r"\s+", " ", a.strip()) for a in (aliases or []) if a and a.strip()]
    local_db.save_trained_command(p, act, aliases=al, notes=notes)

    slug = _slug(p)
    payload = {
        "slug": slug,
        "phrase": p,
        "target_action": act,
        "aliases": al,
        "notes": (notes or "")[:240],
        "updated_at": _now_iso(),
    }
    if is_configured():
        def _sync() -> None:
            rtdb_put(f"jarvis/commands/{slug}", payload)
            for a in [p] + al:
                ap = _phonetic_norm(a)
                if ap:
                    rtdb_put(f"jarvis/voice_aliases/{_slug(ap)}", {
                        "spoken": a,
                        "canonical": p,
                        "target_action": act,
                        "updated_at": payload["updated_at"],
                    })
            firestore_set("jarvis_commands", slug, payload)

        _enqueue(_sync)
    return payload


def extract_explicit_training(text: str) -> tuple[str, str] | None:
    """Detect verbal command-training statements in English or Hinglish."""
    t = (text or "").strip()
    if not t or t.startswith("["):
        return None
    patterns = (
        re.compile(
            r"(?:\b(?:yaad rakhna|samajh lo|save kar lo|hamesha|remember)\b[,\s]*)?"
            r"\b(?:jab main|agar main|when i say|whenever i say|if i say)\s+['\"]?(.+?)['\"]?\s+"
            r"(?:bolu|kahun|boloon|kahu|say)\s*(?:to|tab|then|means|matlab|,)?\s+(.+)",
            re.I,
        ),
        re.compile(
            r"\b(?:train command|learn command|shortcut|command train(?: karo)?)\s*[:,-]?\s*['\"]?(.+?)['\"]?\s*"
            r"(?:->|=>|=|means|ka matlab|matlab|pe|par)\s*(.+)",
            re.I,
        ),
        re.compile(
            r"\b(?:meri awaaz|when unclear|agar clear na ho|remember|samajh lo).*?['\"](.+?)['\"]\s*"
            r"(?:means|ka matlab|matlab|samajhna|karna)\s+(.+)",
            re.I,
        ),
    )
    for pat in patterns:
        m = pat.search(t)
        if m:
            phr = m.group(1).strip(" .,:;\"'")
            act = m.group(2).strip(" .,:;\"'")
            if len(phr) >= 2 and len(act) >= 2:
                return phr, act
    return None


def resolve_unclear_command(spoken_text: str, min_score: float = 0.72) -> tuple[str, str, float] | None:
    """Fuzzy + phonetic matcher so unclear spoken audio matches trained commands."""
    from memory import local_db

    raw = (spoken_text or "").strip()
    if not raw or raw.startswith("[") or len(raw) < 2:
        return None
    norm_in = _phonetic_norm(raw)
    if not norm_in:
        return None

    cmds = local_db.get_trained_commands(limit=80)
    best: tuple[str, str, float] | None = None
    best_sc = 0.0
    for c in cmds:
        candidates = [c.get("phrase", "")] + list(c.get("aliases") or [])
        for cand in candidates:
            cn = _phonetic_norm(cand)
            if not cn:
                continue
            if norm_in == cn:
                sc = 1.0
            else:
                sc = SequenceMatcher(None, norm_in, cn).ratio()
                # Boost word-overlap for multi-word Hinglish phrases
                in_toks, cn_toks = set(norm_in.split()), set(cn.split())
                if in_toks and cn_toks:
                    jacc = len(in_toks & cn_toks) / float(len(in_toks | cn_toks))
                    sc = max(sc, 0.45 * sc + 0.55 * jacc)
            if sc > best_sc:
                best_sc = sc
                best = (c.get("phrase", ""), c.get("target_action", ""), float(sc))
    if best and best_sc >= float(min_score):
        return best
    return None


_TRIVIAL_PATTERN_RE = re.compile(
    r"^(?:hello|hi|hey|ok|okay|yes|no|haan|nahi|theek hai|wake up(?: me)?|activate|sleep(?: mode)?|jarvis)[.!?\s]*$",
    re.I,
)


def train_from_exchange(
    user_text: str,
    assistant_text: str = "",
    tools_used: list[str] | None = None,
    emotion: str = "neutral",
) -> None:
    """Learn explicit commands & tool-action habits without saving raw AI replies to RTDB."""
    del assistant_text  # Never upload raw AI reply text to Firebase training_turns
    u = (user_text or "").strip()
    if not u or u.startswith("["):
        return

    explicit = extract_explicit_training(u)
    if explicit:
        train_command(explicit[0], explicit[1], notes="Auto-learned from conversation")

    # Learn phrase -> tool habit only when meaningful action tools ran cleanly
    clean_tools = [str(t).strip() for t in (tools_used or []) if str(t).strip()]
    if not clean_tools or len(u) > 90 or _TRIVIAL_PATTERN_RE.match(u):
        return
    slug = re.sub(r"[^a-z0-9\u0900-\u097F]+", "_", u.lower()).strip("_")[:64]
    if not slug:
        return

    from memory import local_db
    action_hint = " -> ".join(clean_tools[:3])
    local_db.record_interaction_pattern(u, action_hint, emotion=emotion)

    if not is_configured():
        return

    ts = _now_iso()

    def _sync() -> None:
        rtdb_put(f"jarvis/learned_patterns/{slug}", {
            "phrase": u,
            "tools": clean_tools[:4],
            "hits": 1,
            "updated_at": ts,
        })

    _enqueue(_sync)


def pull_and_merge(timeout: float = 3.5) -> dict[str, int]:
    """Pull trained commands, patterns & important memories from RTDB + Firestore into local DB."""
    from memory import local_db

    counts = {"commands": 0, "memories": 0, "patterns": 0}
    if not is_configured():
        return counts

    try:
        cmds = rtdb_get("jarvis/commands", timeout=timeout)
        if isinstance(cmds, dict):
            for _k, item in cmds.items():
                if not isinstance(item, dict):
                    continue
                phr = str(item.get("phrase") or "").strip()
                act = str(item.get("target_action") or "").strip()
                if phr and act:
                    al = item.get("aliases")
                    if not isinstance(al, list):
                        al = []
                    local_db.save_trained_command(phr, act, aliases=al, notes=str(item.get("notes") or ""))
                    counts["commands"] += 1
    except Exception:
        pass

    try:
        fs_cmds = firestore_list("jarvis_commands", timeout=timeout)
        for item in fs_cmds:
            phr = str(item.get("phrase") or "").strip()
            act = str(item.get("target_action") or "").strip()
            if phr and act:
                al = item.get("aliases")
                if not isinstance(al, list):
                    al = []
                local_db.save_trained_command(phr, act, aliases=al, notes=str(item.get("notes") or ""))
                counts["commands"] += 1
    except Exception:
        pass

    # Important memories: Cloud Firestore first, then RTDB mirror
    try:
        fs_docs = firestore_list("jarvis_memories", timeout=timeout)
        for doc in fs_docs:
            k = str(doc.get("key") or "").strip()
            v = str(doc.get("value") or "").strip()
            if k and v:
                local_db.save_fact(k, v, category=str(doc.get("category") or "notes"))
                counts["memories"] += 1
    except Exception:
        pass

    try:
        rt_mems = rtdb_get("jarvis/important_memories", timeout=timeout)
        if isinstance(rt_mems, dict):
            for _k, doc in rt_mems.items():
                if not isinstance(doc, dict):
                    continue
                k = str(doc.get("key") or "").strip()
                v = str(doc.get("value") or "").strip()
                if k and v:
                    local_db.save_fact(k, v, category=str(doc.get("category") or "notes"))
                    counts["memories"] += 1
    except Exception:
        pass

    try:
        rt_pats = rtdb_get("jarvis/learned_patterns", timeout=timeout)
        if isinstance(rt_pats, dict):
            for _k, doc in rt_pats.items():
                if not isinstance(doc, dict):
                    continue
                phr = str(doc.get("phrase") or "").strip()
                tools = doc.get("tools") or []
                if phr and isinstance(tools, list) and tools:
                    local_db.record_interaction_pattern(phr, " -> ".join(str(t) for t in tools[:3]))
                    counts["patterns"] += 1
    except Exception:
        pass

    return counts


def start_realtime_listener(on_update: Callable[[str], None] | None = None) -> None:
    """Keep Firebase RTDB trained commands & memories synced in real time."""
    global _STREAM_STARTED, _ON_REMOTE_UPDATE
    if on_update is not None:
        _ON_REMOTE_UPDATE = on_update
    if not is_configured():
        return
    with _WORKER_LOCK:
        if _STREAM_STARTED:
            return
        _STREAM_STARTED = True

    def _stream_loop() -> None:
        # Initial sync on boot
        try:
            res = pull_and_merge()
            if _ON_REMOTE_UPDATE and (res["commands"] or res["memories"]):
                _ON_REMOTE_UPDATE(f"CLOUD SYNC · {res['commands']} CMD · {res['memories']} MEM")
        except Exception:
            pass

        while True:
            try:
                hdr, _pid, base = _get_auth_headers()
                if not hdr or not base:
                    time.sleep(10)
                    continue
                s_hdr = dict(hdr)
                s_hdr["Accept"] = "text/event-stream"
                url = f"{base}/jarvis/commands.json"
                with requests.get(url, headers=s_hdr, stream=True, timeout=(8, 70)) as resp:
                    if resp.status_code != 200:
                        time.sleep(15)
                        continue
                    event_name = ""
                    for raw_line in resp.iter_lines(decode_unicode=True):
                        if not raw_line:
                            continue
                        if raw_line.startswith("event:"):
                            event_name = raw_line[6:].strip()
                        elif raw_line.startswith("data:") and event_name in ("put", "patch"):
                            data_str = raw_line[5:].strip()
                            if data_str and data_str != "null":
                                try:
                                    payload = json.loads(data_str)
                                    d = payload.get("data") if isinstance(payload, dict) else None
                                    if isinstance(d, dict) and d.get("phrase") and d.get("target_action"):
                                        from memory import local_db
                                        local_db.save_trained_command(
                                            d["phrase"], d["target_action"], aliases=d.get("aliases") or []
                                        )
                                        if _ON_REMOTE_UPDATE:
                                            _ON_REMOTE_UPDATE(f"⚡ RTDB TRAINED · {str(d['phrase'])[:18]}")
                                except Exception:
                                    pass
            except Exception:
                time.sleep(10)

    t = threading.Thread(target=_stream_loop, daemon=True, name="FirebaseRTDBStream")
    t.start()

