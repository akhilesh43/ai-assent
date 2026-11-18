"""
ui_phonelink.py — the PHONE LINK window: a QR code to scan with the JARVIS phone app.

Replaces the old "Remote Control" window. It needs only Python and (for the picture) the `qrcode`
package; nothing about it depends on a web framework or a certificate. If `qrcode` is missing the
window still shows the address and key to type into the app.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Optional

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPainter
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

FONT = "Segoe UI"


def qr_matrix(text: str) -> Optional[list[list[bool]]]:
    """The QR as rows of dark/light modules, or None when the `qrcode` package is not installed."""
    try:
        import qrcode
        q = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
        q.add_data(text)
        q.make(fit=True)
        return q.get_matrix()
    except Exception:
        return None


class QrWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._m: Optional[list[list[bool]]] = None
        self.setFixedSize(280, 280)

    def set_data(self, text: str) -> bool:
        self._m = qr_matrix(text)
        self.update()
        return self._m is not None

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#ffffff"))
        if not self._m:
            p.setPen(QColor("#333333"))
            p.setFont(QFont(FONT, 9))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "QR needs the 'qrcode' package\n\npip install \"qrcode[pil]\"\n\nor type the address below\ninto the phone app")
            p.end()
            return
        n = len(self._m)
        cell = self.width() / n
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#000000"))
        for y, row in enumerate(self._m):
            for x, dark in enumerate(row):
                if dark:
                    p.drawRect(QRectF(x * cell, y * cell, cell + 0.6, cell + 0.6))
        p.end()


class PhoneLinkDialog(QDialog):
    """`provider()` returns PhoneLinkServer.info(): {ok, url, key, manual, ips, error}.
    `connected()` returns (bool, device name) — polled once a second so the window can say "Connected ✓"."""

    def __init__(self, pal, provider: Callable[[], Optional[dict]],
                 connected: Callable[[], tuple[bool, str]], firewall_bat: str = "", parent=None,
                 contact: Optional[Callable[[], Optional[dict]]] = None,
                 firewall_ok: Optional[Callable[[], bool]] = None,
                 fix_firewall: Optional[Callable[[str], bool]] = None):
        super().__init__(parent)
        self._pal, self._provider, self._connected, self._bat = pal, provider, connected, firewall_bat
        self._contact = contact or (lambda: None)
        self._firewall_ok = firewall_ok or (lambda: True)
        self._run_fix = fix_firewall
        self._info: dict = {}
        self._made = time.time()
        self.setWindowTitle("JARVIS — Phone link")
        self.setModal(False)
        self.setStyleSheet(f"QDialog {{ background: {pal.BG}; }} QLabel {{ background: transparent; color: {pal.TEXT}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(26, 22, 26, 22)
        lay.setSpacing(10)

        title = QLabel("◉  PHONE LINK  ·  v2")
        title.setFont(QFont(FONT, 15, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {pal.PRI};")
        lay.addWidget(title)
        sub = QLabel("Open the JARVIS app on your phone → scan this code.\nPhone and laptop must be on the SAME Wi-Fi.")
        sub.setFont(QFont(FONT, 9))
        sub.setStyleSheet(f"color: {pal.TEXT_MED};")
        lay.addWidget(sub)

        self.qr = QrWidget()
        qrrow = QHBoxLayout()
        qrrow.addStretch()
        qrrow.addWidget(self.qr)
        qrrow.addStretch()
        lay.addLayout(qrrow)

        self.status = QLabel("")
        self.status.setFont(QFont(FONT, 11, QFont.Weight.Bold))
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.status)

        self.firebase_status = QLabel("")
        self.firebase_status.setFont(QFont(FONT, 9, QFont.Weight.Bold))
        self.firebase_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.firebase_status)

        self.live = QLabel("")
        self.live.setWordWrap(True)
        self.live.setFont(QFont(FONT, 9))
        self.live.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.live)

        self.addr = QLabel("")
        self.addr.setFont(QFont("Consolas", 9))
        self.addr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.addr.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.addr.setStyleSheet(f"color: {pal.TEXT_MED};")
        lay.addWidget(self.addr)

        self.err = QLabel("")
        self.err.setWordWrap(True)
        self.err.setFont(QFont(FONT, 9))
        self.err.setStyleSheet(f"color: {pal.RED};")
        lay.addWidget(self.err)

        def btn(text: str, cb, accent: str = "PRI") -> QPushButton:
            b = QPushButton(text)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFont(QFont(FONT, 9, QFont.Weight.DemiBold))
            b.setStyleSheet(
                f"QPushButton {{ background: {pal.PANEL}; color: {getattr(pal, accent)}; border: 1px solid {pal.BORDER_B}; "
                f"border-radius: 8px; padding: 8px 14px; }} QPushButton:hover {{ background: {pal.PRI_GHO}; color: {pal.WHITE}; }}")
            b.clicked.connect(lambda _c=False: cb())
            return b

        row = QHBoxLayout()
        row.addWidget(btn("↻  New code", self.refresh, "GREEN"))
        row.addWidget(btn("⧉  Copy link", self._copy))
        row.addWidget(btn("🛡  Fix firewall", self._fix_firewall, "ACC2"))
        row.addStretch()
        row.addWidget(btn("Close", self.close, "TEXT_MED"))
        lay.addLayout(row)

        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._update_status)
        self.refresh()
        self._tick.start()
        # First time on Windows without the firewall rule: ask for it right away, once, instead of
        # leaving the user to discover why the phone cannot connect.
        QTimer.singleShot(400, self._auto_firewall)

    def refresh(self) -> None:
        try:
            self._info = self._provider() or {}
        except Exception as e:
            self._info = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self._made = time.time()
        url = self._info.get("url", "")
        drawn = self.qr.set_data(url) if url else False
        if url:
            ips = ", ".join(self._info.get("ips", []))
            self.addr.setText(f"Or type into the app:  {self._info.get('manual', '')}   key  {self._info.get('key', '')}\n"
                              f"This laptop: {ips}")
        else:
            self.addr.setText("")
        err = self._info.get("error", "")
        if url and not drawn:
            err = (err + "  " if err else "") + "Install the QR picture support:  pip install \"qrcode[pil]\""
        self.err.setText(err)
        self._update_status()

    def _auto_firewall(self) -> None:
        try:
            if self._run_fix and not self._firewall_ok():
                self.err.setText("Windows will now ask for administrator permission to let your phone in — click Yes.")
                self._fix_firewall()
        except Exception:
            pass

    def _update_status(self) -> None:
        pal = self._pal
        try:
            ok, name = self._connected()
        except Exception:
            ok, name = False, ""
        left = max(0, int(self._info.get("expires_in", 600) - (time.time() - self._made)))
        test = self._info.get("test_url", "")
        c = None
        try:
            c = self._contact()
        except Exception:
            pass

        fb = self._info.get("firebase")
        if fb:
            self.firebase_status.setText("Firebase Sync: Enabled")
            self.firebase_status.setStyleSheet(f"color: {pal.GREEN};")
        else:
            self.firebase_status.setText("Firebase Sync: Disabled")
            self.firebase_status.setStyleSheet(f"color: {pal.RED};")

        if c and c.get("ago", 999) < 120:
            self.live.setText(f"●  LIVE: your phone reached this laptop ({c['who']}, {c['ago']}s ago) — the network is fine.")
            self.live.setStyleSheet(f"color: {pal.GREEN};")
        elif test:
            self.live.setText(f"Network test — open this in your PHONE'S BROWSER:\n{test}\nThis line turns green the moment it arrives. "
                              "If it never does: press “Fix firewall”, and use the same Wi-Fi.")
            self.live.setStyleSheet(f"color: {pal.TEXT_DIM};")
        else:
            self.live.setText("")
        if ok:
            self.status.setText(f"✓  Phone connected: {name}")
            self.status.setStyleSheet(f"color: {pal.GREEN};")
        elif not self._info.get("url"):
            self.status.setText("Link not available")
            self.status.setStyleSheet(f"color: {pal.RED};")
        elif left == 0:
            self.status.setText("Code expired — press “New code”")
            self.status.setStyleSheet(f"color: {pal.ACC2};")
        else:
            self.status.setText(f"Waiting for the phone…   code valid {left // 60}:{left % 60:02d}")
            self.status.setStyleSheet(f"color: {pal.TEXT_MED};")

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self._info.get("url", ""))

    def _fix_firewall(self) -> None:
        if sys.platform.startswith("win") and self._bat and os.path.exists(self._bat):
            if not (self._run_fix and self._run_fix(self._bat)):          # elevated, one prompt
                os.startfile(self._bat)                   # type: ignore[attr-defined]  (the .bat asks by itself)
        else:
            self.err.setText("Windows only: run fix_phone_firewall.bat in the JARVIS folder.")

    def closeEvent(self, e):
        self._tick.stop()
        super().closeEvent(e)
