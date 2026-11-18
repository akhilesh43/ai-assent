"""
ui_workspace.py — the HUD's working tabs: OVERVIEW · TASKS · BRAIN · MIND.

They take over the centre of the window (the same stack the camera, video and browser use) and
hand it back to the face with one click. All numbers come from core/insights.py and all task
changes go through memory/tasks_store.py — the same two sources the phone dashboard uses.

SAFETY RULES THIS FILE FOLLOWS
  • Nothing here is allowed to take the app down. Every slot and timer callback is wrapped, and
    ui.py builds this module inside a try/except — if it cannot be built, the app starts exactly
    as it did before, without the tabs.
  • Reading a database and the OS counters happens on a worker thread; the result reaches the Qt
    thread through a signal. The Qt thread only draws.
  • Colours come from the `palette` object ui.py passes in (its class C), read at paint time, so
    the accent-colour picker recolours these tabs live like everything else.
"""
from __future__ import annotations

import math
import threading
from typing import Any, Callable, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

FONT = "Segoe UI"
MONO = "Consolas"

PAGES = (("overview", "▦  OVERVIEW"), ("tasks", "☑  TASKS"), ("brain", "◈  BRAIN"), ("mind", "☺  MIND"),
         ("phone", "📱  PHONE"))


def _safe(fn: Callable) -> Callable:
    """A slot that logs and carries on instead of raising into Qt."""
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:                                   # pragma: no cover - defensive
            print(f"[Workspace] {fn.__name__}: {type(e).__name__}: {e}")
    wrapper.__name__ = getattr(fn, "__name__", "slot")
    return wrapper


def _qc(hex_: str, alpha: int = 255) -> QColor:
    c = QColor(hex_)
    c.setAlpha(alpha)
    return c


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())


# ─────────────────────────────────────────────────────────────────────────────
#  Small painted widgets
# ─────────────────────────────────────────────────────────────────────────────
class Ring(QWidget):
    """A round gauge: value 0‥100 in the middle, a caption under it."""

    def __init__(self, pal, caption: str, color_attr: str = "PRI", parent=None):
        super().__init__(parent)
        self._pal, self._caption, self._color_attr = pal, caption, color_attr
        self._value, self._text = 0.0, "—"
        self.setMinimumSize(110, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_value(self, value: Optional[float], text: str = "") -> None:
        self._value = 0.0 if value is None else max(0.0, min(100.0, float(value)))
        self._text = text or ("—" if value is None else f"{value:.0f}%")
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        d = max(40, min(w - 12, h - 30))
        box = QRectF((w - d) / 2, 4, d, d)
        col = getattr(self._pal, self._color_attr, self._pal.PRI)
        if self._value >= 90 and self._color_attr in ("PRI", "ACC"):
            col = self._pal.RED
        track = QPen(_qc(self._pal.BORDER_B, 140), 7)
        track.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(track)
        p.drawArc(box, 0, 360 * 16)
        arc = QPen(_qc(col), 7)
        arc.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc)
        if self._value > 0:
            p.drawArc(box, 90 * 16, int(-self._value / 100.0 * 360 * 16))
        p.setPen(_qc(self._pal.WHITE))
        p.setFont(QFont(FONT, max(10, int(d / 5)), QFont.Weight.Bold))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, self._text)
        p.setPen(_qc(self._pal.TEXT_DIM))
        p.setFont(QFont(FONT, 8, QFont.Weight.DemiBold))
        p.drawText(QRectF(0, d + 6, w, 18), Qt.AlignmentFlag.AlignCenter, self._caption)
        p.end()


class BarChart(QWidget):
    """Vertical bars with a label under each. `series` is a list of (label, value[, value2])."""

    def __init__(self, pal, color_attr: str = "PRI", color2_attr: str = "ACC", parent=None):
        super().__init__(parent)
        self._pal, self._a, self._b = pal, color_attr, color2_attr
        self._series: list[tuple] = []
        self.setMinimumHeight(110)

    def set_series(self, series: list[tuple]) -> None:
        self._series = series
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        n = len(self._series)
        if n == 0:
            p.setPen(_qc(self._pal.TEXT_DIM))
            p.setFont(QFont(FONT, 9))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "no data yet")
            p.end()
            return
        top = max(1, max(sum(s[1:]) for s in self._series))
        slot = w / n
        bw = max(8.0, min(34.0, slot * 0.5))
        chart_h = h - 22
        for i, s in enumerate(self._series):
            x = i * slot + (slot - bw) / 2
            y = chart_h
            for j, v in enumerate(s[1:]):
                bh = chart_h * (v / top) * 0.92
                col = getattr(self._pal, self._a if j == 0 else self._b)
                g = QLinearGradient(0, y - bh, 0, y)
                g.setColorAt(0, _qc(col, 255))
                g.setColorAt(1, _qc(col, 110))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(g)
                if bh > 0.5:
                    p.drawRoundedRect(QRectF(x, y - bh, bw, bh), 3, 3)
                y -= bh
            p.setPen(_qc(self._pal.TEXT_DIM))
            p.setFont(QFont(FONT, 8))
            p.drawText(QRectF(i * slot, h - 18, slot, 16), Qt.AlignmentFlag.AlignCenter, str(s[0]))
            total = sum(s[1:])
            if total:
                p.setPen(_qc(self._pal.TEXT_MED))
                p.drawText(QRectF(i * slot, y - 16, slot, 14), Qt.AlignmentFlag.AlignCenter, str(int(total)))
        p.end()


class MeterBar(QWidget):
    """A labelled horizontal bar: label left, value right, fill 0‥1."""

    def __init__(self, pal, label: str, color_attr: str = "PRI", parent=None):
        super().__init__(parent)
        self._pal, self._label, self._color_attr = pal, label, color_attr
        self._frac, self._text = 0.0, ""
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set(self, frac: float, text: str = "", label: Optional[str] = None) -> None:
        self._frac = max(0.0, min(1.0, frac))
        self._text = text
        if label is not None:
            self._label = label
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        p.setFont(QFont(FONT, 8, QFont.Weight.DemiBold))
        p.setPen(_qc(self._pal.TEXT_MED))
        p.drawText(QRectF(0, 0, w * 0.6, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._label)
        p.setPen(_qc(self._pal.WHITE))
        p.drawText(QRectF(w * 0.4, 0, w * 0.6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self._text)
        track = QRectF(0, 18, w, 7)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_qc(self._pal.BAR_BG))
        p.drawRoundedRect(track, 3.5, 3.5)
        if self._frac > 0:
            col = getattr(self._pal, self._color_attr, self._pal.PRI)
            g = QLinearGradient(0, 0, w, 0)
            g.setColorAt(0, _qc(col, 150))
            g.setColorAt(1, _qc(col))
            p.setBrush(g)
            p.drawRoundedRect(QRectF(0, 18, max(7.0, w * self._frac), 7), 3.5, 3.5)
        p.end()


class Card(QFrame):
    """A titled panel with a big number: `Card(pal, "OPEN TASKS")` then `.set("7", "2 overdue")`."""

    def __init__(self, pal, title: str, accent_attr: str = "PRI", parent=None):
        super().__init__(parent)
        self._pal, self._accent = pal, accent_attr
        self.setMinimumHeight(86)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        self._title = QLabel(title)
        self._title.setFont(QFont(FONT, 8, QFont.Weight.DemiBold))
        self._value = QLabel("—")
        self._value.setFont(QFont(FONT, 20, QFont.Weight.Bold))
        self._sub = QLabel("")
        self._sub.setFont(QFont(FONT, 8))
        self._sub.setWordWrap(True)
        for l in (self._title, self._value, self._sub):
            lay.addWidget(l)
        self.restyle()

    def restyle(self) -> None:
        P = self._pal
        acc = getattr(P, self._accent, P.PRI)
        self.setObjectName("wsCard")
        self.setStyleSheet(
            f"QFrame#wsCard {{ background: {P.PANEL}; border: 1px solid {P.BORDER}; "
            f"border-left: 3px solid {acc}; border-radius: 10px; }}"
            f"QLabel {{ background: transparent; border: none; }}")
        self._title.setStyleSheet(f"color: {P.TEXT_DIM};")
        self._value.setStyleSheet(f"color: {acc};")
        self._sub.setStyleSheet(f"color: {P.TEXT_MED};")

    def set(self, value: str, sub: str = "") -> None:
        self._value.setText(value)
        self._sub.setText(sub)


def _section_label(pal, text: str) -> QLabel:
    l = QLabel(text)
    l.setFont(QFont(FONT, 9, QFont.Weight.Bold))
    l.setStyleSheet(f"color: {pal.TEXT_MED}; background: transparent; letter-spacing: 1px;")
    return l


def _panel(pal) -> QFrame:
    f = QFrame()
    f.setStyleSheet(f"QFrame {{ background: {pal.PANEL}; border: 1px solid {pal.BORDER}; border-radius: 10px; }}"
                    f"QLabel {{ background: transparent; border: none; }}")
    return f


def _scroll(pal, inner: QWidget) -> QScrollArea:
    s = QScrollArea()
    s.setWidgetResizable(True)
    s.setFrameShape(QFrame.Shape.NoFrame)
    s.setStyleSheet(
        f"QScrollArea {{ background: transparent; border: none; }}"
        f"QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}"
        f"QScrollBar::handle:vertical {{ background: {pal.BORDER_B}; border-radius: 4px; min-height: 24px; }}"
        f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}")
    inner.setStyleSheet("background: transparent;")
    s.setWidget(inner)
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  Pages
# ─────────────────────────────────────────────────────────────────────────────
class _Page(QWidget):
    def apply(self, snap: dict) -> None:           # overridden
        pass


def _err(sec: Any) -> bool:
    return not isinstance(sec, dict) or "error" in sec


class OverviewPage(_Page):
    def __init__(self, pal, parent=None):
        super().__init__(parent)
        self._pal = pal
        body = QWidget()
        root = QVBoxLayout(body)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)

        root.addWidget(_section_label(pal, "LIVE STATUS"))
        rings = QHBoxLayout()
        self.cpu = Ring(pal, "CPU", "PRI")
        self.mem = Ring(pal, "MEMORY", "ACC")
        self.brain = Ring(pal, "BRAIN LEVEL", "GREEN")
        self.tasks = Ring(pal, "TASKS DONE", "ACC2")
        for r in (self.cpu, self.mem, self.brain, self.tasks):
            rings.addWidget(r)
        root.addLayout(rings)

        grid = QGridLayout()
        grid.setSpacing(10)
        self.c_bond = Card(pal, "BOND", "ACC")
        self.c_mood = Card(pal, "MOOD NOW", "PRI")
        self.c_open = Card(pal, "OPEN TASKS", "ACC2")
        self.c_brain = Card(pal, "BRAIN LEVEL", "GREEN")
        for i, c in enumerate((self.c_bond, self.c_mood, self.c_open, self.c_brain)):
            grid.addWidget(c, 0, i)
        root.addLayout(grid)

        root.addWidget(_section_label(pal, "CONVERSATIONS — LAST 7 DAYS   (you ▪ JARVIS)"))
        self.chart = BarChart(pal, "PRI", "ACC")
        root.addWidget(self.chart)

        two = QHBoxLayout()
        two.setSpacing(12)
        left = _panel(pal)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 10, 14, 12)
        ll.addWidget(_section_label(pal, "HOW YOU'VE SOUNDED LATELY"))
        self.emo_box = QVBoxLayout()
        self.emo_box.setSpacing(2)
        ll.addLayout(self.emo_box)
        right = _panel(pal)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(14, 10, 14, 12)
        rl.addWidget(_section_label(pal, "WHAT IT KNOWS ABOUT YOU"))
        self.facts = QLabel("—")
        self.facts.setWordWrap(True)
        self.facts.setFont(QFont(FONT, 9))
        self.facts.setStyleSheet(f"color: {pal.TEXT};")
        rl.addWidget(self.facts)
        rl.addStretch()
        two.addWidget(left, 1)
        two.addWidget(right, 1)
        root.addLayout(two)
        root.addStretch()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scroll(pal, body))

    def apply(self, snap: dict) -> None:
        P = self._pal
        s = snap.get("system", {})
        if not _err(s):
            self.cpu.set_value(s.get("cpu"))
            self.mem.set_value(s.get("mem"))
        t = snap.get("tasks", {})
        if not _err(t):
            self.tasks.set_value(t.get("completion_pct", 0))
            self.c_open.set(str(t.get("open", 0)),
                            f"{t.get('overdue', 0)} overdue · {t.get('due_today', 0)} due today · "
                            f"{t.get('done_today', 0)} done today")
        b = snap.get("brain", {})
        if not _err(b):
            lv = b["level"]
            self.brain.set_value(lv["progress"] * 100, f"Lv {lv['level']}")
            self.c_brain.set(lv["name"], f"Lv {lv['level']} · {b['memories']} memories · {b['facts']} facts")
            self.facts.setText("\n".join(f"•  {f['key']}: {f['value']}" for f in b.get("top_facts", [])[:6])
                               or "Nothing learned yet — just talk to it.")
        m = snap.get("mind", {})
        if not _err(m):
            bd = m["bond"]
            self.c_bond.set(bd["name"], f"{bd['xp']} xp · {m['streak_days']}-day streak")
            self.c_mood.set(m["mood"].title(), m["personality"])
        a = snap.get("activity", {})
        if not _err(a):
            self.chart.set_series([(d["label"], d["user"], d["assistant"]) for d in a["days"]])
            _clear(self.emo_box)
            emo = a.get("emotions", {})
            tot = sum(emo.values()) or 1
            for name, n in list(emo.items())[:5]:
                bar = MeterBar(P, name.title(), "PRI")
                bar.set(n / tot, f"{100 * n // tot}%")
                self.emo_box.addWidget(bar)
            if not emo:
                self.emo_box.addWidget(QLabel("No emotion data yet."))


class _TaskCard(QFrame):
    def __init__(self, pal, task: dict, on_move, on_done, on_delete, parent=None):
        super().__init__(parent)
        pr = {1: pal.RED, 2: pal.PRI, 3: pal.TEXT_DIM}.get(task["priority"], pal.PRI)
        done = task["status"] == "done"
        self.setObjectName("wsTask")
        self.setStyleSheet(
            f"QFrame#wsTask {{ background: {pal.PANEL2}; border: 1px solid {pal.BORDER}; "
            f"border-left: 3px solid {pr}; border-radius: 8px; }}"
            f"QLabel {{ background: transparent; border: none; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 8, 8)
        lay.setSpacing(4)
        title = QLabel(task["title"])
        title.setWordWrap(True)
        title.setFont(QFont(FONT, 9, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {pal.TEXT_DIM if done else pal.WHITE};"
                            + ("text-decoration: line-through;" if done else ""))
        lay.addWidget(title)
        meta = []
        if task["due"]:
            meta.append(("⚠ overdue " if task["overdue"] else "⏰ ") + task["due"])
        meta.extend(f"#{t}" for t in task["tags"][:3])
        if meta:
            m = QLabel("   ".join(meta))
            m.setFont(QFont(FONT, 8))
            m.setStyleSheet(f"color: {pal.RED if task['overdue'] else pal.TEXT_DIM};")
            lay.addWidget(m)
        row = QHBoxLayout()
        row.setSpacing(4)

        def mk(text: str, tip: str, cb) -> QPushButton:
            b = QPushButton(text)
            b.setFixedSize(26, 22)
            b.setToolTip(tip)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {pal.TEXT_MED}; border: 1px solid {pal.BORDER}; "
                f"border-radius: 5px; font-size: 11px; }}"
                f"QPushButton:hover {{ color: {pal.WHITE}; border-color: {pal.PRI}; background: {pal.PRI_GHO}; }}")
            b.clicked.connect(_safe(lambda _c=False: cb()))
            return b
        tid = task["id"]
        st = task["status"]
        if st != "todo":
            row.addWidget(mk("◀", "move back", lambda: on_move(tid, "todo" if st == "doing" else "doing")))
        if st == "todo":
            row.addWidget(mk("▶", "start", lambda: on_move(tid, "doing")))
        if st != "done":
            row.addWidget(mk("✓", "mark done", lambda: on_done(tid)))
        row.addStretch()
        row.addWidget(mk("✕", "delete", lambda: on_delete(tid)))
        lay.addLayout(row)


class TasksPage(_Page):
    changed = pyqtSignal()

    def __init__(self, pal, parent=None):
        super().__init__(parent)
        self._pal = pal
        from memory import tasks_store
        self._store = tasks_store
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        stats = QHBoxLayout()
        stats.setSpacing(10)
        self.s_open = Card(pal, "OPEN", "PRI")
        self.s_over = Card(pal, "OVERDUE", "RED")
        self.s_today = Card(pal, "DONE TODAY", "GREEN")
        self.s_streak = Card(pal, "STREAK", "ACC2")
        for c in (self.s_open, self.s_over, self.s_today, self.s_streak):
            c.setMinimumHeight(72)
            stats.addWidget(c)
        root.addLayout(stats)

        add = QHBoxLayout()
        add.setSpacing(6)
        field = (f"QLineEdit, QComboBox {{ background: {pal.PANEL}; color: {pal.TEXT}; border: 1px solid {pal.BORDER}; "
                 f"border-radius: 7px; padding: 6px 10px; }}"
                 f"QLineEdit:focus, QComboBox:focus {{ border-color: {pal.PRI}; }}"
                 f"QComboBox QAbstractItemView {{ background: {pal.PANEL}; color: {pal.TEXT}; "
                 f"selection-background-color: {pal.PRI_GHO}; }}")
        self.title_in = QLineEdit()
        self.title_in.setPlaceholderText("New task…  (Enter to add)")
        self.due_in = QLineEdit()
        self.due_in.setPlaceholderText("due: kal / friday / 2026-12-01")
        self.due_in.setFixedWidth(190)
        self.prio = QComboBox()
        self.prio.addItems(["Normal", "High", "Low"])
        self.prio.setFixedWidth(90)
        for w in (self.title_in, self.due_in, self.prio):
            w.setStyleSheet(field)
            w.setFont(QFont(FONT, 9))
        btn = QPushButton("＋  ADD")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFont(QFont(FONT, 9, QFont.Weight.Bold))
        btn.setStyleSheet(
            f"QPushButton {{ background: {pal.PRI}; color: {pal.DARK}; border: none; border-radius: 7px; padding: 7px 16px; }}"
            f"QPushButton:hover {{ background: {pal.WHITE}; }}")
        btn.clicked.connect(_safe(lambda _c=False: self._add()))
        self.title_in.returnPressed.connect(_safe(self._add))
        self.due_in.returnPressed.connect(_safe(self._add))
        add.addWidget(self.title_in, 1)
        add.addWidget(self.due_in)
        add.addWidget(self.prio)
        add.addWidget(btn)
        root.addLayout(add)
        self.hint = QLabel("")
        self.hint.setFont(QFont(FONT, 8))
        self.hint.setStyleSheet(f"color: {pal.ACC2}; background: transparent;")
        root.addWidget(self.hint)

        cols = QHBoxLayout()
        cols.setSpacing(10)
        self._cols: dict[str, QVBoxLayout] = {}
        self._heads: dict[str, QLabel] = {}
        for key, name, accent in (("todo", "TO DO", pal.PRI), ("doing", "IN PROGRESS", pal.ACC2),
                                  ("done", "DONE", pal.GREEN)):
            col = _panel(pal)
            cl = QVBoxLayout(col)
            cl.setContentsMargins(10, 8, 10, 8)
            head = QLabel(name)
            head.setFont(QFont(FONT, 9, QFont.Weight.Bold))
            head.setStyleSheet(f"color: {accent};")
            self._heads[key] = head
            cl.addWidget(head)
            inner = QWidget()
            v = QVBoxLayout(inner)
            v.setContentsMargins(0, 0, 4, 0)
            v.setSpacing(6)
            v.addStretch()
            cl.addWidget(_scroll(pal, inner), 1)
            self._cols[key] = v
            cols.addWidget(col, 1)
        root.addLayout(cols, 1)

        clear = QPushButton("Clear finished")
        clear.setCursor(Qt.CursorShape.PointingHandCursor)
        clear.setStyleSheet(f"QPushButton {{ background: transparent; color: {pal.TEXT_DIM}; border: none; }}"
                            f"QPushButton:hover {{ color: {pal.WHITE}; }}")
        clear.clicked.connect(_safe(lambda _c=False: (self._store.clear_done(), self.reload())))
        root.addWidget(clear, 0, Qt.AlignmentFlag.AlignRight)
        self.reload()

    @_safe
    def _add(self) -> None:
        title = self.title_in.text().strip()
        if not title:
            return
        due = self.due_in.text().strip()
        pr = {"High": 1, "Normal": 2, "Low": 3}[self.prio.currentText()]
        t = self._store.add(title, due=due, priority=pr, source="hud")
        self.hint.setText("" if (t["due"] or not due) else f"Couldn't read “{due}” as a date — added without a deadline.")
        self.title_in.clear()
        self.due_in.clear()
        self.reload()

    def _move(self, tid: int, status: str) -> None:
        self._store.set_status(tid, status)
        self.reload()

    def _delete(self, tid: int) -> None:
        self._store.delete(tid)
        self.reload()

    @_safe
    def reload(self) -> None:
        tasks = self._store.list_tasks("all", limit=300)
        groups = {"todo": [], "doing": [], "done": []}
        for t in tasks:
            groups[t["status"]].append(t)
        groups["done"] = groups["done"][:25]
        for key, lay in self._cols.items():
            while lay.count() > 1:                              # keep the trailing stretch
                item = lay.takeAt(0)
                if item.widget() is not None:
                    item.widget().setParent(None)
                    item.widget().deleteLater()
            for t in groups[key]:
                lay.insertWidget(lay.count() - 1, _TaskCard(
                    self._pal, t, self._move, lambda i: self._move(i, "done"), self._delete))
            names = {"todo": "TO DO", "doing": "IN PROGRESS", "done": "DONE"}
            self._heads[key].setText(f"{names[key]}  ·  {len(groups[key])}")
        self.changed.emit()
        s = self._store.summary()
        self.s_open.set(str(s["open"]), f"{s['high_priority']} high priority")
        self.s_over.set(str(s["overdue"]), f"{s['due_today']} due today")
        self.s_today.set(str(s["done_today"]), f"{s['completion_pct']}% of all tasks")
        self.s_streak.set(f"{s['streak_days']} d", "days in a row")

    def apply(self, snap: dict) -> None:                       # tasks reload themselves; keep cheap
        pass


class BrainPage(_Page):
    def __init__(self, pal, parent=None):
        super().__init__(parent)
        self._pal = pal
        body = QWidget()
        root = QVBoxLayout(body)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(12)
        self.level = Ring(pal, "BRAIN LEVEL", "GREEN")
        self.level.setMinimumWidth(150)
        top.addWidget(self.level)
        grid = QGridLayout()
        grid.setSpacing(10)
        self.k = {}
        for i, (key, title, acc) in enumerate((
                ("memories", "MEMORIES", "PRI"), ("facts", "LEARNED FACTS", "GREEN"),
                ("turns", "CONVERSATION TURNS", "ACC"), ("trained", "TRAINED COMMANDS", "ACC2"),
                ("plugins", "SKILLS / PLUGINS", "PRI"), ("emotions", "EMOTIONS READ", "ACC"))):
            c = Card(pal, title, acc)
            self.k[key] = c
            grid.addWidget(c, i // 3, i % 3)
        top.addLayout(grid, 1)
        root.addLayout(top)

        cap = _panel(pal)
        cl = QVBoxLayout(cap)
        cl.setContentsMargins(14, 10, 14, 12)
        cl.addWidget(_section_label(pal, "CAPACITY"))
        self.m_store = MeterBar(pal, "Long-term storage", "GREEN")
        self.m_level = MeterBar(pal, "Progress to next brain level", "PRI")
        cl.addWidget(self.m_store)
        cl.addWidget(self.m_level)
        self.budget = QLabel("")
        self.budget.setFont(QFont(FONT, 8))
        self.budget.setWordWrap(True)
        self.budget.setStyleSheet(f"color: {pal.TEXT_DIM};")
        cl.addWidget(self.budget)
        root.addWidget(cap)

        two = QHBoxLayout()
        two.setSpacing(12)
        cats = _panel(pal)
        cv = QVBoxLayout(cats)
        cv.setContentsMargins(14, 10, 14, 12)
        cv.addWidget(_section_label(pal, "MEMORY BY CATEGORY"))
        self.cat_box = QVBoxLayout()
        self.cat_box.setSpacing(2)
        cv.addLayout(self.cat_box)
        cv.addStretch()
        habits = _panel(pal)
        hv = QVBoxLayout(habits)
        hv.setContentsMargins(14, 10, 14, 12)
        hv.addWidget(_section_label(pal, "TOP FACTS & HABITS"))
        self.habits = QLabel("—")
        self.habits.setWordWrap(True)
        self.habits.setFont(QFont(FONT, 9))
        self.habits.setStyleSheet(f"color: {pal.TEXT};")
        hv.addWidget(self.habits)
        hv.addStretch()
        two.addWidget(cats, 1)
        two.addWidget(habits, 1)
        root.addLayout(two)

        teach = _panel(pal)
        tv = QVBoxLayout(teach)
        tv.setContentsMargins(14, 10, 14, 12)
        tv.addWidget(_section_label(pal, "TEACH IT SOMETHING"))
        row = QHBoxLayout()
        field = (f"QLineEdit {{ background: {pal.PANEL2}; color: {pal.TEXT}; border: 1px solid {pal.BORDER}; "
                 f"border-radius: 7px; padding: 6px 10px; }} QLineEdit:focus {{ border-color: {pal.PRI}; }}")
        self.k_in = QLineEdit()
        self.k_in.setPlaceholderText("what (e.g. sister's name)")
        self.k_in.setFixedWidth(220)
        self.v_in = QLineEdit()
        self.v_in.setPlaceholderText("value (e.g. Riya)   — Enter to save")
        for w in (self.k_in, self.v_in):
            w.setStyleSheet(field)
            w.setFont(QFont(FONT, 9))
        self.v_in.returnPressed.connect(_safe(self._teach))
        b = QPushButton("REMEMBER")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFont(QFont(FONT, 9, QFont.Weight.Bold))
        b.setStyleSheet(f"QPushButton {{ background: {pal.GREEN}; color: {pal.DARK}; border: none; border-radius: 7px; padding: 7px 14px; }}"
                        f"QPushButton:hover {{ background: {pal.WHITE}; }}")
        b.clicked.connect(_safe(lambda _c=False: self._teach()))
        row.addWidget(self.k_in)
        row.addWidget(self.v_in, 1)
        row.addWidget(b)
        tv.addLayout(row)
        self.teach_msg = QLabel("")
        self.teach_msg.setFont(QFont(FONT, 8))
        self.teach_msg.setStyleSheet(f"color: {pal.GREEN};")
        tv.addWidget(self.teach_msg)
        root.addWidget(teach)
        root.addStretch()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scroll(pal, body))

    @_safe
    def _teach(self) -> None:
        from memory import memory_manager as mm
        k, v = self.k_in.text().strip(), self.v_in.text().strip()
        if not (k and v):
            self.teach_msg.setText("Fill both boxes.")
            return
        mm.remember(k, v, "notes")
        self.k_in.clear()
        self.v_in.clear()
        self.teach_msg.setText(f"Saved. It will remember: {k} → {v}")

    def apply(self, snap: dict) -> None:
        b = snap.get("brain", {})
        if _err(b):
            return
        lv = b["level"]
        self.level.set_value(lv["progress"] * 100, f"Lv {lv['level']}")
        self.level._caption = lv["name"].upper()
        self.k["memories"].set(str(b["memories"]), f"{len(b['by_category'])} categories")
        self.k["facts"].set(str(b["facts"]), "auto-learned from talk")
        self.k["turns"].set(str(b["conversation_turns"]), "saved for continuity")
        self.k["trained"].set(str(b["trained_commands"]), "voice shortcuts")
        self.k["plugins"].set(str(b["plugins"]), "drop-in skills")
        self.k["emotions"].set(str(b["emotions_logged"]), "mood readings")
        self.m_store.set(b["storage_pct"] / 100, f"{b['storage_chars']:,} / {b['storage_capacity']:,} chars")
        nxt = f"→ {lv['next']}" if lv["next"] else "max level"
        self.m_level.set(lv["progress"], f"score {lv['score']} {nxt}")
        pb = b["prompt_budget"]
        self.budget.setText(
            f"Each session it carries up to {pb['core_chars']:,} characters of what it knows (up to "
            f"{pb['per_category']} per category) plus an index of {pb['index_chars']:,} more it can look up "
            f"on demand. Everything else stays on disk — nothing is forgotten to make room.")
        _clear(self.cat_box)
        cats = sorted(b["by_category"].items(), key=lambda kv: -kv[1])
        top = max([n for _, n in cats] or [1])
        for name, n in cats[:7]:
            bar = MeterBar(self._pal, name.title(), "PRI")
            bar.set(n / top, str(n))
            self.cat_box.addWidget(bar)
        if not cats:
            self.cat_box.addWidget(QLabel("Empty — tell it things, or use the box below."))
        lines = [f"•  {f['key']}: {f['value']}   ×{f['hits']}" for f in b["top_facts"][:6]]
        lines += [f"↻  “{p['phrase']}” → {p['hint']}" for p in b["top_patterns"][:3]]
        self.habits.setText("\n".join(lines) or "Nothing yet.")


class MindPage(_Page):
    LABELS = {"warmth": ("Warmth", "ACC"), "humour": ("Humour", "ACC2"), "curiosity": ("Curiosity", "PRI"),
              "formality": ("Formality", "TEXT_MED"), "brevity": ("Brevity", "GREEN")}

    def __init__(self, pal, parent=None):
        super().__init__(parent)
        self._pal = pal
        self._updating = False
        from core import behavior
        self._bh = behavior
        body = QWidget()
        root = QVBoxLayout(body)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)

        cards = QHBoxLayout()
        cards.setSpacing(10)
        self.c_bond = Card(pal, "RELATIONSHIP", "ACC")
        self.c_mood = Card(pal, "MOOD IT'S HOLDING", "PRI")
        self.c_day = Card(pal, "RIGHT NOW", "ACC2")
        self.c_streak = Card(pal, "DAY STREAK", "GREEN")
        for c in (self.c_bond, self.c_mood, self.c_day, self.c_streak):
            cards.addWidget(c)
        root.addLayout(cards)

        bond = _panel(pal)
        bv = QVBoxLayout(bond)
        bv.setContentsMargins(14, 10, 14, 12)
        bv.addWidget(_section_label(pal, "BOND"))
        self.bond_bar = MeterBar(pal, "", "ACC")
        bv.addWidget(self.bond_bar)
        self.guidance = QLabel("")
        self.guidance.setWordWrap(True)
        self.guidance.setFont(QFont(FONT, 9))
        self.guidance.setStyleSheet(f"color: {pal.TEXT_MED};")
        bv.addWidget(self.guidance)
        root.addWidget(bond)

        tr = _panel(pal)
        tv = QVBoxLayout(tr)
        tv.setContentsMargins(14, 10, 14, 12)
        head = QHBoxLayout()
        head.addWidget(_section_label(pal, "PERSONALITY — learns from you, or set it yourself"))
        head.addStretch()
        reset = QPushButton("Reset")
        reset.setCursor(Qt.CursorShape.PointingHandCursor)
        reset.setStyleSheet(f"QPushButton {{ background: transparent; color: {pal.TEXT_DIM}; border: 1px solid {pal.BORDER}; "
                            f"border-radius: 5px; padding: 2px 10px; }} QPushButton:hover {{ color: {pal.WHITE}; }}")
        reset.clicked.connect(_safe(lambda _c=False: self._reset()))
        head.addWidget(reset)
        tv.addLayout(head)
        self.label = QLabel("")
        self.label.setFont(QFont(FONT, 12, QFont.Weight.Bold))
        self.label.setStyleSheet(f"color: {pal.PRI};")
        tv.addWidget(self.label)
        self.sliders: dict[str, QSlider] = {}
        self.vals: dict[str, QLabel] = {}
        for key, (name, acc) in self.LABELS.items():
            row = QHBoxLayout()
            n = QLabel(name)
            n.setFixedWidth(90)
            n.setFont(QFont(FONT, 9, QFont.Weight.DemiBold))
            n.setStyleSheet(f"color: {pal.TEXT_MED};")
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 100)
            s.setStyleSheet(
                f"QSlider::groove:horizontal {{ height: 6px; background: {pal.BAR_BG}; border-radius: 3px; }}"
                f"QSlider::sub-page:horizontal {{ background: {getattr(pal, acc)}; border-radius: 3px; }}"
                f"QSlider::handle:horizontal {{ background: {pal.WHITE}; width: 14px; margin: -5px 0; border-radius: 7px; }}")
            v = QLabel("0")
            v.setFixedWidth(34)
            v.setStyleSheet(f"color: {pal.WHITE};")
            s.valueChanged.connect(_safe(lambda val, k=key: self._on_slider(k, val)))
            self.sliders[key], self.vals[key] = s, v
            row.addWidget(n)
            row.addWidget(s, 1)
            row.addWidget(v)
            tv.addLayout(row)
        note = QLabel("These settings shape the replies: how warm, how funny, how brief, how formal. "
                      "Moving a slider takes effect from the next time the assistant connects.")
        note.setWordWrap(True)
        note.setFont(QFont(FONT, 8))
        note.setStyleSheet(f"color: {pal.TEXT_DIM};")
        tv.addWidget(note)
        root.addWidget(tr)
        root.addStretch()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scroll(pal, body))

    def _on_slider(self, key: str, val: int) -> None:
        self.vals[key].setText(str(val))
        if not self._updating:
            self._bh.set_trait(key, val / 100.0)

    def _reset(self) -> None:
        self._bh.reset_traits()
        self.apply({"mind": self._bh.snapshot()})

    def apply(self, snap: dict) -> None:
        m = snap.get("mind", {})
        if _err(m):
            return
        bd = m["bond"]
        self.c_bond.set(bd["name"], f"Level {bd['level']} · {bd['xp']} xp")
        self.c_mood.set(m["mood"].title(), "from how you've sounded")
        self.c_day.set(m["daypart"].title(), f"last chat {m['hours_since_last']}h ago")
        self.c_streak.set(f"{m['streak_days']} d", f"best {m['longest_streak']} d")
        self.bond_bar.set(bd["progress"], f"{bd['xp']} / {bd['next_at']} xp" if bd["next"] else "maximum",
                          label=f"{bd['name']}" + (f"  →  {bd['next']}" if bd["next"] else ""))
        self.guidance.setText("Current guidance to itself:  " + m["mood_guidance"])
        self.label.setText(m["personality"])
        focus = any(s.hasFocus() or s.isSliderDown() for s in self.sliders.values())
        if not focus:
            self._updating = True
            try:
                for k, s in self.sliders.items():
                    s.setValue(int(round(m["traits"][k] * 100)))
                    self.vals[k].setText(str(s.value()))
            finally:
                self._updating = False


class PhonePage(_Page):
    """The paired phone: status, quick actions, replies waiting for approval, notifications, teaching."""

    def __init__(self, pal, parent=None):
        super().__init__(parent)
        self._pal = pal
        self.on_pair: Optional[Callable[[], None]] = None
        from core.phone_bridge import get_bridge
        self._bridge = get_bridge
        self._sig = None
        body = QWidget()
        root = QVBoxLayout(body)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)

        cards = QHBoxLayout()
        cards.setSpacing(10)
        self.c_conn = Card(pal, "PHONE", "GREEN")
        self.c_batt = Card(pal, "BATTERY", "ACC2")
        self.c_call = Card(pal, "CALL", "ACC")
        self.c_learn = Card(pal, "LEARNED REPLIES", "PRI")
        for c in (self.c_conn, self.c_batt, self.c_call, self.c_learn):
            cards.addWidget(c)
        root.addLayout(cards)

        def btn(text, cb, accent="PRI"):
            b = QPushButton(text)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFont(QFont(FONT, 9, QFont.Weight.DemiBold))
            b.setStyleSheet(
                f"QPushButton {{ background: {pal.PANEL}; color: {getattr(pal, accent)}; border: 1px solid {pal.BORDER_B}; "
                f"border-radius: 8px; padding: 7px 14px; }}"
                f"QPushButton:hover {{ background: {pal.PRI_GHO}; color: {pal.WHITE}; }}")
            b.clicked.connect(_safe(lambda _c=False: cb()))
            return b

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(btn("📷  Pair phone (QR)", lambda: self.on_pair and self.on_pair(), "GREEN"))
        row.addWidget(btn("✆  Answer", lambda: self._do("answer_call"), "GREEN"))
        row.addWidget(btn("✖  End / decline", lambda: self._do("end_call"), "RED"))
        row.addWidget(btn("🔔  Find phone", lambda: self._do("find_phone"), "ACC2"))
        row.addWidget(btn("🔕  Stop", lambda: self._do("stop_find_phone"), "TEXT_MED"))
        row.addWidget(btn("🔦  Torch on", lambda: self._do("flashlight", {"on": True}), "ACC2"))
        row.addWidget(btn("Off", lambda: self._do("flashlight", {"on": False}), "TEXT_MED"))
        row.addStretch()
        root.addLayout(row)

        mode = QHBoxLayout()
        lab = QLabel("AUTO-REPLY to messages:")
        lab.setFont(QFont(FONT, 9, QFont.Weight.Bold))
        lab.setStyleSheet(f"color: {pal.TEXT_MED};")
        self.mode = QComboBox()
        self.mode.addItems(["ask", "auto", "ignore", "off"])
        self.mode.setFixedWidth(120)
        self.mode.setStyleSheet(
            f"QComboBox {{ background: {pal.PANEL}; color: {pal.TEXT}; border: 1px solid {pal.BORDER}; "
            f"border-radius: 7px; padding: 5px 10px; }}"
            f"QComboBox QAbstractItemView {{ background: {pal.PANEL}; color: {pal.TEXT}; "
            f"selection-background-color: {pal.PRI_GHO}; }}")
        self._mode_busy = False
        self.mode.currentTextChanged.connect(_safe(self._set_mode))
        hint = QLabel("ask = draft in your style, you approve with one tap · auto = answer on its own "
                      "(never groups, money, or anything private)")
        hint.setFont(QFont(FONT, 8))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {pal.TEXT_DIM};")
        mode.addWidget(lab)
        mode.addWidget(self.mode)
        mode.addWidget(hint, 1)
        root.addLayout(mode)

        root.addWidget(_section_label(pal, "WAITING FOR YOUR OK"))
        self.pend_box = QVBoxLayout()
        self.pend_box.setSpacing(8)
        root.addLayout(self.pend_box)

        root.addWidget(_section_label(pal, "PHONE NOTIFICATIONS"))
        self.note_box = QVBoxLayout()
        self.note_box.setSpacing(4)
        root.addLayout(self.note_box)

        teach = _panel(pal)
        tv = QVBoxLayout(teach)
        tv.setContentsMargins(14, 10, 14, 12)
        tv.addWidget(_section_label(pal, "TEACH IT HOW YOU REPLY"))
        r = QHBoxLayout()
        field = (f"QLineEdit {{ background: {pal.PANEL2}; color: {pal.TEXT}; border: 1px solid {pal.BORDER}; "
                 f"border-radius: 7px; padding: 6px 10px; }} QLineEdit:focus {{ border-color: {pal.PRI}; }}")
        self.t_who = QLineEdit(); self.t_who.setPlaceholderText("who (optional)"); self.t_who.setFixedWidth(130)
        self.t_in = QLineEdit(); self.t_in.setPlaceholderText("they say…")
        self.t_out = QLineEdit(); self.t_out.setPlaceholderText("you reply…  (Enter)")
        for w in (self.t_who, self.t_in, self.t_out):
            w.setStyleSheet(field); w.setFont(QFont(FONT, 9))
        self.t_out.returnPressed.connect(_safe(self._teach))
        r.addWidget(self.t_who); r.addWidget(self.t_in, 1); r.addWidget(self.t_out, 1)
        r.addWidget(btn("TEACH", self._teach, "GREEN"))
        tv.addLayout(r)
        self.t_msg = QLabel("")
        self.t_msg.setFont(QFont(FONT, 8))
        self.t_msg.setStyleSheet(f"color: {pal.GREEN};")
        tv.addWidget(self.t_msg)
        root.addWidget(teach)

        root.addWidget(_section_label(pal, "ACTIVITY"))
        self.log = QLabel("")
        self.log.setWordWrap(True)
        self.log.setFont(QFont(MONO, 8))
        self.log.setStyleSheet(f"color: {pal.TEXT_DIM};")
        root.addWidget(self.log)
        root.addStretch()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scroll(pal, body))

    # commands wait for the phone, so never on the Qt thread
    def _do(self, action: str, args: Optional[dict] = None) -> None:
        threading.Thread(target=lambda: self._bridge().send_command(action, args or {}),
                         daemon=True, name="PhoneCmd").start()

    def _set_mode(self, mode: str) -> None:
        if not self._mode_busy:
            self._bridge().set_mode(mode)

    def _teach(self) -> None:
        if self._bridge().teach(self.t_in.text() or "general", self.t_out.text(), self.t_who.text()):
            self.t_msg.setText("Learned. It will answer like that.")
            self.t_in.clear(); self.t_out.clear()
        else:
            self.t_msg.setText("Fill in what you reply (and optionally what they say).")

    def _approve(self, pid: str, edit: QLineEdit) -> None:
        text = edit.text().strip()
        threading.Thread(target=lambda: self._bridge().approve(pid, text or None), daemon=True).start()
        self._sig = None

    def _skip(self, pid: str) -> None:
        self._bridge().reject(pid)
        self._sig = None
        self.apply({})

    def _quick_reply(self, n: dict) -> None:
        def work():
            b = self._bridge()
            text = b.draft(n["text"], n["sender"] or n["title"], n["app"])
            if text:
                b.send_command("reply_notification", {"key": n["key"], "text": text})
        threading.Thread(target=work, daemon=True).start()

    def apply(self, snap: dict) -> None:
        P = self._pal
        s = self._bridge().snapshot()
        self.c_conn.set("Connected" if s["connected"] else "Not connected",
                        s["device"] or "press “Pair phone (QR)”")
        self.c_batt.set("—" if s["battery"] is None else f"{s['battery']}%", "charging" if s["charging"] else "")
        c = s["call"]
        self.c_call.set(c.get("state", "idle").title(), c.get("who", "") if c.get("state") == "ringing" else "")
        self.c_learn.set(str(s["learning"]["examples"]), f"{s['learning']['contacts']} contacts · mode {s['mode']}")
        self._mode_busy = True
        self.mode.setCurrentText(s["mode"])
        self._mode_busy = False
        self.log.setText("\n".join(x["text"] for x in s["log"][:8]) or "Nothing yet.")

        sig = (tuple(p["id"] for p in s["pending"]), tuple(n["key"] + n["text"] for n in s["notifications"][:8]))
        if sig == self._sig:
            return                                   # don't rebuild (and wipe a half-typed reply)
        self._sig = sig
        _clear(self.pend_box)
        for p in s["pending"][:6]:
            f = _panel(P)
            v = QVBoxLayout(f)
            v.setContentsMargins(12, 8, 12, 10)
            h = QLabel(f"{p['contact']}  ·  {p['app']}")
            h.setFont(QFont(FONT, 9, QFont.Weight.Bold))
            h.setStyleSheet(f"color: {P.ACC};")
            m = QLabel("“" + p["incoming"] + "”")
            m.setWordWrap(True)
            m.setStyleSheet(f"color: {P.TEXT};")
            e = QLineEdit(p["draft"])
            e.setPlaceholderText("Type your own reply" if p["needs_owner"] else "")
            e.setStyleSheet(f"QLineEdit {{ background: {P.PANEL2}; color: {P.WHITE}; border: 1px solid {P.BORDER_B}; "
                            f"border-radius: 7px; padding: 6px 10px; }}")
            r = QHBoxLayout()
            ok = QPushButton("✓  Send")
            no = QPushButton("Skip")
            for b, col in ((ok, P.GREEN), (no, P.TEXT_DIM)):
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.setStyleSheet(f"QPushButton {{ background: transparent; color: {col}; border: 1px solid {P.BORDER_B}; "
                                f"border-radius: 7px; padding: 5px 14px; }} QPushButton:hover {{ background: {P.PRI_GHO}; }}")
            ok.clicked.connect(_safe(lambda _c=False, i=p["id"], ed=e: self._approve(i, ed)))
            e.returnPressed.connect(_safe(lambda i=p["id"], ed=e: self._approve(i, ed)))
            no.clicked.connect(_safe(lambda _c=False, i=p["id"]: self._skip(i)))
            r.addWidget(ok); r.addWidget(no); r.addStretch()
            for w in (h, m, e):
                v.addWidget(w)
            v.addLayout(r)
            self.pend_box.addWidget(f)
        if not s["pending"]:
            q = QLabel("Nothing waiting.")
            q.setStyleSheet(f"color: {P.TEXT_DIM};")
            self.pend_box.addWidget(q)
        _clear(self.note_box)
        for n in s["notifications"][:8]:
            row = QHBoxLayout()
            t = QLabel(f"{n['app'] or n['package']}  ·  {n['sender'] or n['title']}:  {n['text'][:90]}")
            t.setStyleSheet(f"color: {P.TEXT_MED};")
            t.setFont(QFont(FONT, 9))
            row.addWidget(t, 1)
            if n["can_reply"]:
                rb = QPushButton("↩ reply")
                rb.setCursor(Qt.CursorShape.PointingHandCursor)
                rb.setStyleSheet(f"QPushButton {{ background: transparent; color: {P.PRI}; border: 1px solid {P.BORDER}; "
                                 f"border-radius: 6px; padding: 2px 10px; }} QPushButton:hover {{ background: {P.PRI_GHO}; }}")
                rb.clicked.connect(_safe(lambda _c=False, nn=n: self._quick_reply(nn)))
                row.addWidget(rb)
            w = QWidget(); w.setLayout(row)
            self.note_box.addWidget(w)
        if not s["notifications"]:
            q = QLabel("No notifications received yet.")
            q.setStyleSheet(f"color: {P.TEXT_DIM};")
            self.note_box.addWidget(q)


# ─────────────────────────────────────────────────────────────────────────────
#  The panel: nav + pages + refresh loop
# ─────────────────────────────────────────────────────────────────────────────
class WorkspacePanel(QWidget):
    """Hosts the four pages. `show_page("tasks")` switches; `closed` fires on the ✕ HUD button."""
    closed = pyqtSignal()
    _snap_ready = pyqtSignal(object)

    REFRESH_MS = 3000

    def __init__(self, palette, parent=None):
        super().__init__(parent)
        self._pal = palette
        self._busy = False
        self.setObjectName("wsRoot")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"QWidget#wsRoot {{ background: {palette.BG}; }}")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.stack = QStackedWidget()
        self.pages: dict[str, _Page] = {
            "overview": OverviewPage(palette),
            "tasks": TasksPage(palette),
            "brain": BrainPage(palette),
            "mind": MindPage(palette),
            "phone": PhonePage(palette),
        }
        for p in self.pages.values():
            self.stack.addWidget(p)
        root.addWidget(self.stack, 1)

        self._snap_ready.connect(_safe(self._apply))
        self._timer = QTimer(self)
        self._timer.setInterval(self.REFRESH_MS)
        self._timer.timeout.connect(_safe(self.refresh))

    @property
    def on_pair(self):
        return self.pages["phone"].on_pair

    @on_pair.setter
    def on_pair(self, fn) -> None:
        self.pages["phone"].on_pair = fn

    def show_page(self, key: str) -> None:
        if key not in self.pages:
            return
        self.stack.setCurrentWidget(self.pages[key])
        if key == "tasks":
            self.pages["tasks"].reload()
        self.refresh()

    def current_key(self) -> str:
        w = self.stack.currentWidget()
        return next((k for k, p in self.pages.items() if p is w), "overview")

    def showEvent(self, e):
        super().showEvent(e)
        self._timer.start()
        _safe(self.refresh)()

    def hideEvent(self, e):
        super().hideEvent(e)
        self._timer.stop()

    def refresh(self) -> None:
        if self._busy or not self.isVisible():
            return
        self._busy = True

        def work():
            try:
                from core import insights
                snap = insights.snapshot()
            except Exception as ex:                               # pragma: no cover - defensive
                snap = {"system": {"error": str(ex)}}
            self._snap_ready.emit(snap)
        threading.Thread(target=work, daemon=True, name="WorkspaceSnap").start()

    def _apply(self, snap: dict) -> None:
        self._busy = False
        self.pages[self.current_key()].apply(snap)
        if self.current_key() != "overview":                      # the ring row lives on overview only
            self.pages["overview"].apply(snap)


class NavBar(QWidget):
    """The strip under the header: HUD · OVERVIEW · TASKS · BRAIN · MIND."""
    picked = pyqtSignal(str)

    def __init__(self, palette, parent=None):
        super().__init__(parent)
        self._pal = palette
        self.setFixedHeight(38)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 4, 14, 4)
        lay.setSpacing(6)
        self._btns: dict[str, QPushButton] = {}
        for key, label in (("hud", "◉  HUD"),) + PAGES:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFont(QFont(FONT, 9, QFont.Weight.DemiBold))
            b.setFixedHeight(28)
            b.clicked.connect(_safe(lambda _c=False, k=key: self.picked.emit(k)))
            self._btns[key] = b
            lay.addWidget(b)
        lay.addStretch()
        self.restyle()
        self.select("hud")

    def restyle(self) -> None:
        P = self._pal
        self.setObjectName("wsNav")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"QWidget#wsNav {{ background: {P.DARK}; border-bottom: 1px solid {P.BORDER}; }}")
        for b in self._btns.values():
            b.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {P.TEXT_DIM}; border: 1px solid transparent; "
                f"border-radius: 14px; padding: 0 16px; }}"
                f"QPushButton:hover {{ color: {P.WHITE}; background: {P.PANEL}; }}"
                f"QPushButton:checked {{ color: {P.DARK}; background: {P.PRI}; font-weight: 700; }}")

    def select(self, key: str) -> None:
        for k, b in self._btns.items():
            b.setChecked(k == key)
