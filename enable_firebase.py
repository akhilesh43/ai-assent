"""
Connect JARVIS to Firebase and prove it works.

    python enable_firebase.py                          # check the key already in config/
    python enable_firebase.py --key "C:\\path\\key.json" # copy a service-account key in, then check
    python enable_firebase.py --url https://NAME-default-rtdb.asia-southeast1.firebasedatabase.app

What it checks, in order, and what it tells you when one fails:
  1. the key file is a real service-account key (project id, private key present)
  2. google-auth / requests are installed
  3. Google accepts the key and hands back an access token
  4. Realtime Database: write → read → delete a test value. If the database lives in another
     region than the default URL (asia-southeast1, europe-west1 …) the right URL is found
     and saved to config/api_keys.json as "firebase_rtdb_url".
  5. Cloud Firestore (optional — memories mirror to the Realtime DB if it is not created)
  6. the SAME code path the app uses (memory/firebase_sync.py) reads the value back

Nothing secret is printed: no token, no private key — only the project id.
Everything trained with 'train_command', important memories and learned screen skills
(text only, never the pictures) sync through this connection.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

BASE = Path(__file__).resolve().parent
KEY_PATH = BASE / "config" / "firebase_service_account.json"
SCOPES = ["https://www.googleapis.com/auth/datastore",
          "https://www.googleapis.com/auth/firebase.database",
          "https://www.googleapis.com/auth/userinfo.email"]
REGIONS = ("asia-southeast1", "europe-west1")


def load_key(path: Path) -> dict:
    """The parsed key, or ValueError with a message a person can act on."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"No key file at {path}")
    except Exception as e:
        raise ValueError(f"That file is not valid JSON ({e})")
    if raw.get("type") != "service_account":
        raise ValueError("That is not a service-account key (it should say \"type\": \"service_account\"). "
                         "Firebase console → Project settings → Service accounts → Generate new private key.")
    for k in ("project_id", "private_key", "client_email"):
        if not raw.get(k):
            raise ValueError(f"The key is missing '{k}' — download a fresh one from the Firebase console.")
    return raw


def candidate_urls(project_id: str, configured: str = "", forced: str = "") -> list[str]:
    urls = [u.rstrip("/") for u in (forced, configured) if u]
    urls.append(f"https://{project_id}-default-rtdb.firebaseio.com")
    urls += [f"https://{project_id}-default-rtdb.{r}.firebasedatabase.app" for r in REGIONS]
    urls.append(f"https://{project_id}.firebaseio.com")
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def explain_status(status: int) -> str:
    return {
        401: "the key was rejected (wrong project, or the key was revoked)",
        403: "permission denied (give the service account the 'Firebase Admin' / 'Editor' role)",
        404: "no Realtime Database at this URL — in the Firebase console open Build → Realtime "
             "Database → 'Create database' once",
        307: "redirected — the database is in another region",
        423: "this database is locked/disabled in the Firebase console",
    }.get(status, f"HTTP {status}")


def probe_rtdb(http, headers: dict, base: str) -> tuple[bool, str]:
    stamp = {"ts": time.time(), "from": "enable_firebase.py"}
    try:
        r = http.put(f"{base}/jarvis_selftest/ping.json", headers=headers, json=stamp, timeout=10)
        if not (200 <= r.status_code < 300):
            return False, explain_status(r.status_code)
        g = http.get(f"{base}/jarvis_selftest/ping.json", headers=headers, timeout=10)
        if g.status_code != 200 or (g.json() or {}).get("from") != "enable_firebase.py":
            return False, "wrote the test value but could not read it back"
        http.delete(f"{base}/jarvis_selftest.json", headers=headers, timeout=10)
        return True, "write / read / delete OK"
    except Exception as e:
        return False, f"network error: {str(e)[:100]}"


def probe_firestore(http, headers: dict, project_id: str) -> tuple[bool, str]:
    url = (f"https://firestore.googleapis.com/v1/projects/{project_id}"
           f"/databases/(default)/documents/jarvis_selftest/ping")
    try:
        r = http.patch(url, headers=headers, json={"fields": {"from": {"stringValue": "enable_firebase.py"}}}, timeout=10)
        if r.status_code == 200:
            http.delete(url, headers=headers, timeout=10)
            return True, "write / delete OK"
        if r.status_code == 404:
            return False, "not created (optional) — memories will mirror to the Realtime DB instead"
        return False, explain_status(r.status_code)
    except Exception as e:
        return False, f"network error: {str(e)[:100]}"


def default_creds_factory(path: Path):
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    creds = service_account.Credentials.from_service_account_file(str(path), scopes=SCOPES)
    creds.refresh(Request())
    return creds.token


def save_rtdb_url(url: str, default_url: str) -> None:
    """Remember a non-default URL where firebase_sync will look for it."""
    from memory.config_manager import _patch_config
    _patch_config(firebase_rtdb_url="" if url == default_url else url)


def main(argv: Optional[list[str]] = None, *, http=None,
         creds_factory: Callable[[Path], str] = default_creds_factory,
         key_path: Path = KEY_PATH, out: Callable[[str], None] = print,
         save_url: Callable[[str, str], None] = save_rtdb_url,
         verify_app_path: bool = True) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--key", help="path to a service-account JSON to copy into config/")
    ap.add_argument("--url", help="Realtime Database URL, if you know it")
    a = ap.parse_args(argv)

    if http is None:
        try:
            import requests as http                                       # type: ignore
        except ImportError:
            out("✗ The 'requests' package is missing:  pip install requests google-auth")
            return 2

    if a.key:
        try:
            load_key(Path(a.key))
        except ValueError as e:
            out(f"✗ {e}")
            return 2
        key_path.parent.mkdir(parents=True, exist_ok=True)
        if Path(a.key).resolve() != key_path.resolve():
            shutil.copyfile(a.key, key_path)
        out(f"✓ Key copied to {key_path.relative_to(BASE) if key_path.is_relative_to(BASE) else key_path}")

    try:
        raw = load_key(key_path)
    except ValueError as e:
        out(f"✗ {e}\n  Put your key at config/firebase_service_account.json or run with --key PATH")
        return 2
    pid = raw["project_id"]
    out(f"✓ Key looks right — project '{pid}'")

    try:
        token = creds_factory(key_path)
    except ImportError:
        out("✗ google-auth is not installed:  pip install google-auth requests")
        return 2
    except Exception as e:
        msg = str(e)
        hint = (" Your computer's clock may be wrong — Google rejects keys when the time is off by minutes."
                if "invalid_grant" in msg or "Invalid JWT" in msg else "")
        out(f"✗ Google would not give an access token: {msg[:160]}.{hint}")
        return 1
    out("✓ Google accepted the key (access token received)")
    headers = {"Authorization": f"Bearer {token}"}

    configured = ""
    try:
        from memory.config_manager import load_api_keys
        configured = str(load_api_keys().get("firebase_rtdb_url") or "").strip()
    except Exception:
        pass
    default_url = f"https://{pid}-default-rtdb.firebaseio.com"
    working = ""
    last_note = ""
    for url in candidate_urls(pid, configured, (a.url or "").strip()):
        ok, note = probe_rtdb(http, headers, url)
        out(f"  {'✓' if ok else '·'} {url}  — {note}")
        if ok:
            working = url
            break
        last_note = note
    if not working:
        out(f"✗ Realtime Database: {last_note}")
        out("  Fix it in the Firebase console (Build → Realtime Database → Create database), "
            "then run this again — or pass the exact URL with --url.")
        return 1
    try:
        save_url(working, default_url)
        if working != default_url:
            out("✓ This database is in another region — its URL is saved in config/api_keys.json")
    except Exception as e:
        out(f"! Could not save the URL automatically ({e}). Add \"firebase_rtdb_url\": \"{working}\" yourself.")
    out("✓ Realtime Database connected")

    fs_ok, fs_note = probe_firestore(http, headers, pid)
    out(f"{'✓' if fs_ok else '·'} Firestore: {fs_note}")

    if verify_app_path:
        try:
            from memory import firebase_sync
            firebase_sync._CREDS = None                 # force it to re-read the saved URL
            firebase_sync._RTDB_URL = ""
            stamp = {"ok": True, "ts": time.time()}
            wrote = firebase_sync.rtdb_put("jarvis_selftest/app_path", stamp)
            back = firebase_sync.rtdb_get("jarvis_selftest/app_path")
            firebase_sync.rtdb_delete("jarvis_selftest")
            if wrote and isinstance(back, dict) and back.get("ok"):
                out("✓ JARVIS's own sync code can read and write it too")
            else:
                out("✗ The connection works, but JARVIS's own sync code could not use it — "
                    "restart and run this again; if it persists send me this output.")
                return 1
        except Exception as e:
            out(f"! Could not run the app-path check ({str(e)[:100]}).")

    out("\nDone. Restart JARVIS. From now on 'train_command' memories, important facts and learned "
        "screen skills (text only) sync here — and come back on a new PC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
