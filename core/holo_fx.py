"""
core/holo_fx.py — makes the head look like a hologram being PROJECTED, and makes the
assistant's state readable from across the room.

Two hooks, called from HudCanvas.paintEvent around the existing head:

    paint_back(...)   before the head : projector base + light cone, rising particles,
                                        a speaking aura that follows the voice
    paint_front(...)  after the head  : scanlines, a sweeping scan band, a rare glitch
                                        strip, and one ring per state:
                                          LISTENING  → ripples that grow with your voice
                                          THINKING   → three rotating arcs
                                          SLEEPING   → head dimmed, slow breathing ring

Everything is drawn from the wall clock (not a frame counter), so it looks the same
whether the HUD repaints at 60 Hz or the idle 20 Hz, and it keeps no state of its own.
Plain QPainter calls only — no new dependency, no OpenGL, no GPU.

Switch off with  "holo_fx": false  in config/api_keys.json (read once at start-up).
Any exception inside is swallowed by the caller: a cosmetic layer must never be able
to stop the HUD from painting.
"""
from __future__ import annotations

import math
import time

try:
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtGui import (QBrush, QColor, QLinearGradient, QPainterPath, QPen,
                             QRadialGradient)
    AVAILABLE = True
except Exception:                                   # no Qt (tests, headless)
    AVAILABLE = False


def _enabled() -> bool:
    try:
        from memory.config_manager import load_api_keys
        return bool(load_api_keys().get("holo_fx", True))
    except Exception:
        return True


ENABLED = AVAILABLE and _enabled()

_PARTICLES = 22
_STANDBY_STATES = ("SLEEPING", "STANDBY")


def _rand(i: int, salt: float) -> float:
    """Deterministic 0..1 'random' per particle, so nothing has to be stored."""
    v = math.sin(i * salt) * 43758.5453
    return v - math.floor(v)


def _alpha(c: "QColor", a: float) -> "QColor":
    out = QColor(c)
    out.setAlpha(int(max(0, min(255, a))))
    return out


def _intensity(st: dict) -> float:
    if st.get("muted"):
        return 0.35
    if st.get("state") in _STANDBY_STATES:
        return 0.45
    return 1.0


def _geometry(cy: float, r: float, span: float) -> tuple[float, float]:
    top = cy - r
    return top, top + span * r


def paint_back(p, cx: float, cy: float, r: float, acc: "QColor", st: dict) -> None:
    """Projector base, light cone, speaking aura and rising particles (behind the head)."""
    if not ENABLED or r < 20:
        return
    now = time.time()
    k = _intensity(st)
    span = float(st.get("span", 2.2))
    top, bottom = _geometry(cy, r, span)
    amp = max(0.0, min(1.0, float(st.get("amp", 0.0))))
    p.save()
    try:
        p.setPen(Qt.PenStyle.NoPen)

        # light cone rising from the projector
        cone = QPainterPath()
        cone.moveTo(QPointF(cx - r * 0.80, bottom))
        cone.lineTo(QPointF(cx - r * 0.32, top + r * 0.25))
        cone.lineTo(QPointF(cx + r * 0.32, top + r * 0.25))
        cone.lineTo(QPointF(cx + r * 0.80, bottom))
        cone.closeSubpath()
        g = QLinearGradient(QPointF(cx, bottom), QPointF(cx, top + r * 0.25))
        g.setColorAt(0.0, _alpha(acc, 46 * k))
        g.setColorAt(1.0, _alpha(acc, 0))
        p.setBrush(QBrush(g))
        p.drawPath(cone)

        # speaking aura — breathes with the voice
        if st.get("speaking") or amp > 0.05:
            rad = r * (1.15 + 0.25 * amp)
            ag = QRadialGradient(QPointF(cx, cy + r * 0.45), rad)
            ag.setColorAt(0.0, _alpha(acc, (30 + 120 * amp) * k))
            ag.setColorAt(1.0, _alpha(acc, 0))
            p.setBrush(QBrush(ag))
            p.drawEllipse(QPointF(cx, cy + r * 0.45), rad, rad)

        # projector base: glowing disc + two rings, the outer one pulsing
        by, ry = bottom - r * 0.03, r * 0.10
        bg = QRadialGradient(QPointF(cx, by), r * 0.95)
        bg.setColorAt(0.0, _alpha(acc, 95 * k))
        bg.setColorAt(1.0, _alpha(acc, 0))
        p.setBrush(QBrush(bg))
        p.drawEllipse(QPointF(cx, by), r * 0.95, ry * 1.6)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_alpha(acc, 120 * k), 1.2))
        p.drawEllipse(QPointF(cx, by), r * 0.62, ry * 0.9)
        pulse = 1.0 + 0.06 * math.sin(now * 2.2)
        p.setPen(QPen(_alpha(acc, 70 * k), 1.0))
        p.drawEllipse(QPointF(cx, by), r * 0.88 * pulse, ry * 1.3 * pulse)

        # rising particles
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(_PARTICLES):
            u, v = _rand(i, 127.1), _rand(i, 311.7)
            phase = (now * (0.05 + 0.07 * v) + u) % 1.0
            y = bottom - phase * span * r * 1.02
            x = cx + (u * 2.0 - 1.0) * r * 0.9 * (0.55 + 0.45 * phase) \
                + math.sin(now * 0.8 + i) * r * 0.03
            a = 130.0 * k * math.sin(math.pi * phase)
            if a < 4:
                continue
            p.setBrush(QBrush(_alpha(acc, a)))
            s = 1.1 + 1.6 * v
            p.drawEllipse(QPointF(x, y), s, s)
    finally:
        p.restore()


def paint_front(p, cx: float, cy: float, r: float, acc: "QColor", st: dict) -> None:
    """Scanlines, scan band, glitch strip and the per-state ring (over the head)."""
    if not ENABLED or r < 20:
        return
    now = time.time()
    k = _intensity(st)
    state = st.get("state", "")
    span = float(st.get("span", 2.2))
    top, bottom = _geometry(cy, r, span)
    amp = max(0.0, min(1.0, float(st.get("amp", 0.0))))
    bgc = st.get("bg")

    # ── clipped to the head's box: scanlines, band, glitch ───────────────────
    p.save()
    try:
        p.setClipRect(QRectF(cx - r * 1.05, top, r * 2.1, bottom - top))
        p.setPen(QPen(_alpha(acc, 15 * k), 1.0))
        y = top
        while y < bottom:
            p.drawLine(QPointF(cx - r * 1.05, y), QPointF(cx + r * 1.05, y))
            y += 4.0

        band_h = r * 0.24
        by = top + ((now * 0.33) % 1.0) * (bottom - top)
        g = QLinearGradient(QPointF(cx, by - band_h / 2), QPointF(cx, by + band_h / 2))
        g.setColorAt(0.0, _alpha(acc, 0))
        g.setColorAt(0.5, _alpha(acc, 58 * k))
        g.setColorAt(1.0, _alpha(acc, 0))
        p.fillRect(QRectF(cx - r * 1.05, by - band_h / 2, r * 2.1, band_h), QBrush(g))

        ph = now % 7.0                                   # a 0.12 s glitch every 7 s
        if ph < 0.12 and k > 0.5:
            gy = top + _rand(int(now // 7.0), 91.7) * (bottom - top)
            p.fillRect(QRectF(cx - r * 1.05, gy, r * 2.1, 2.0), _alpha(acc, 85))
            p.fillRect(QRectF(cx - r * 0.7, gy + 3.0, r * 1.2, 1.0), _alpha(acc, 55))
    finally:
        p.restore()

    # ── state rings, unclipped ───────────────────────────────────────────────
    mid = (top + bottom) / 2.0
    p.save()
    try:
        p.setBrush(Qt.BrushStyle.NoBrush)
        if state in _STANDBY_STATES:
            if bgc is not None:                          # dim the head
                rad = max(r * 1.05, (bottom - top) / 2.0)
                vg = QRadialGradient(QPointF(cx, mid), rad)
                vg.setColorAt(0.0, _alpha(bgc, 150))
                vg.setColorAt(0.8, _alpha(bgc, 120))
                vg.setColorAt(1.0, _alpha(bgc, 0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(vg))
                p.drawEllipse(QPointF(cx, mid), rad, rad)
                p.setBrush(Qt.BrushStyle.NoBrush)
            breath = 0.5 + 0.5 * math.sin(now * 1.2)
            rr = r * (1.10 + 0.03 * breath)
            p.setPen(QPen(_alpha(acc, 40 + 70 * breath), 1.4))
            p.drawEllipse(QPointF(cx, mid), rr, rr)
        elif state == "LISTENING" and not st.get("speaking"):
            for j in (0, 1):
                phase = ((now * 0.7) + j * 0.5) % 1.0
                rr = r * (1.08 + 0.35 * phase)
                a = 90.0 * (1.0 - phase) * (0.4 + amp * 1.6)
                if a < 3:
                    continue
                p.setPen(QPen(_alpha(acc, a), 1.5))
                p.drawEllipse(QPointF(cx, mid), rr, rr)
        elif state in ("THINKING", "PROCESSING"):
            rr = r * 1.28
            ang = (now * 140.0) % 360.0
            p.setPen(QPen(_alpha(acc, 150), 2.0))
            for j in range(3):
                p.drawArc(QRectF(cx - rr, mid - rr, rr * 2, rr * 2),
                          int(((ang + j * 120.0) % 360.0) * 16), 70 * 16)
    finally:
        p.restore()
