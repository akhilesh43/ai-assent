"""
phone_doctor.py — "Remote Control does not connect". Finds out why, in plain words.

    python phone_doctor.py            check this laptop (is the dashboard up, which addresses, firewall)
    python phone_doctor.py --serve    run ONLY the phone link and show a QR here in the terminal —
                                      a quick way to test pairing without starting all of JARVIS

The usual causes, in the order they happen:
  1. JARVIS is not running, or its PHONE LINK window was never opened
  2. the QR carries the address of a VPN / WSL / VirtualBox adapter, not your Wi-Fi
  3. Windows Firewall blocks the port (the phone says "timed out")
  4. another program already holds port 8765 (JARVIS then uses the next free one, up to 8775)
  5. phone on mobile data or guest Wi-Fi (guest networks usually isolate devices from each other)
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import socket
import sys
import urllib.request
from typing import Optional

PORT = 8765


def build_pair_url(ip: str, key: str, alts: Optional[list[str]] = None, port: int = PORT) -> str:
    url = f"http://{ip}:{port}/pair?key={key}"
    if alts:
        url += "&alt=" + ",".join(alts[:4])
    return url


def port_in_use(port: int = PORT) -> bool:
    """True when something already listens on the port (probed the way the dashboard itself does)."""
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def ping(ip: str, port: int = PORT, timeout: float = 2.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"http://{ip}:{port}/api/ping", timeout=timeout) as r:
            data = json.loads(r.read().decode())
            return data if data.get("jarvis") else None
    except Exception:
        return None


def missing_modules(names=("qrcode", "psutil")) -> list[str]:
    out = []
    for n in names:
        try:
            importlib.import_module(n)
        except Exception:
            out.append(n)
    return out


def ips() -> list[str]:
    try:
        from core.phone_link import lan_ips
        return lan_ips()
    except Exception:
        return []


def report(out=print) -> int:
    problems = 0
    out("JARVIS phone link — checking this laptop\n")

    miss = missing_modules()
    out("✓ The phone link needs nothing but Python (no fastapi / certificate any more)")
    if "qrcode" in miss:
        out('! qrcode is missing, so the QR window will be blank.   Fix:  pip install "qrcode[pil]"')

    addrs = ips()
    if not addrs:
        problems += 1
        out("✗ No private network address found. Is this laptop connected to Wi-Fi / LAN?")
    else:
        out("✓ This laptop's addresses: " + ", ".join(addrs))
        if len(addrs) > 1:
            out("  (more than one — the phone app tries each; a VPN or virtual adapter is usually the extra one)")

    up = ping("127.0.0.1")
    if up:
        out(f"✓ JARVIS phone link is running on port {PORT} (computer name: {up.get('name')})")
        for a in addrs:
            ok = ping(a)
            out(f"  {'✓' if ok else '✗'} reachable at http://{a}:{PORT}" + ("" if ok else "  ← blocked? see firewall below"))
            if not ok:
                problems += 1
    elif port_in_use():
        problems += 1
        out(f"✗ Port {PORT} is taken by another program, and it is not JARVIS.\n"
            "    Close the other program (or a second copy of JARVIS) and restart JARVIS.")
    else:
        out(f"– JARVIS phone link is not running now. Start JARVIS and open PHONE LINK (or run:  python phone_doctor.py --serve)")

    if sys.platform.startswith("win"):
        out("\nWindows Firewall (needed once; run in an ADMINISTRATOR terminal):\n"
            '    netsh advfirewall firewall add rule name="JARVIS phone" dir=in action=allow protocol=TCP localport=8765-8775 profile=any\n'
            "  And make sure your Wi-Fi is set to a PRIVATE network (Settings → Network → Wi-Fi → Properties).")
    out("\nOn the phone: same Wi-Fi as the laptop, VPN off, not a 'guest' network, not mobile data.")
    out("Quick test from the phone's browser: open  http://<laptop address>:8765/api/ping  — you should see {\"jarvis\": true …}.")
    out("\n" + ("Everything checked looks fine." if not problems else f"{problems} problem(s) found — fix the ✗ lines above."))
    return problems


async def _serve() -> None:
    from core.phone_link import PhoneLinkServer
    srv = PhoneLinkServer()
    if not srv.start():
        print(srv.bind_error)
        return
    info = srv.info()
    print(f"\nAddress to type : {info.get('manual')}    key: {info.get('key')}\nQR / link       : {info.get('url')}\n")
    try:
        import qrcode
        q = qrcode.QRCode(border=1)
        q.add_data(info["url"])
        q.make(fit=True)
        q.print_ascii(invert=True)
    except Exception:
        print('(pip install "qrcode[pil]" to see the QR here)')
    print("\nScan it with the JARVIS phone app. Ctrl+C to stop.\n")
    while True:
        await asyncio.sleep(3600)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="run only the phone server and show a QR")
    a = ap.parse_args(argv)
    if a.serve:
        try:
            asyncio.run(_serve())
        except KeyboardInterrupt:
            pass
        return 0
    return 1 if report() else 0


if __name__ == "__main__":
    sys.exit(main())
