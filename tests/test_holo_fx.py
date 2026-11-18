"""python -m unittest tests.test_holo_fx   — runs the FX code against a recording fake of Qt,
so typos and bad arithmetic are caught without a display. (It cannot judge how it LOOKS.)"""
import importlib, os, sys, types, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Rec:
    """Stands in for any Qt object: remembers calls, accepts anything."""
    def __init__(self, *a, **k): self.a, self.k, self.calls = a, k, []
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        def f(*a, **k): self.calls.append((n, a)); return None
        return f
    def alpha(self): return getattr(self, "_al", 255)


class Color(Rec):
    def __init__(self, *a, **k):
        super().__init__(*a)
        self._al = a[0]._al if a and isinstance(a[0], Color) else 255
    def setAlpha(self, v):
        assert 0 <= v <= 255, v
        self._al = v


class Painter(Rec):
    def __init__(self):
        super().__init__(); self.depth = 0; self.draws = 0
    def save(self): self.depth += 1
    def restore(self): self.depth -= 1; assert self.depth >= 0
    def __getattr__(self, n):
        if n.startswith("draw") or n.startswith("fill"):
            def f(*a, **k): self.draws += 1
            return f
        return super().__getattr__(n)


def load():
    qc = types.ModuleType("PyQt6.QtCore")
    qc.QPointF = Rec; qc.QRectF = Rec
    qc.Qt = types.SimpleNamespace(PenStyle=types.SimpleNamespace(NoPen=0),
                                  BrushStyle=types.SimpleNamespace(NoBrush=0))
    qg = types.ModuleType("PyQt6.QtGui")
    for n in ("QBrush", "QLinearGradient", "QPainterPath", "QPen", "QRadialGradient"):
        setattr(qg, n, Rec)
    qg.QColor = Color
    pkg = types.ModuleType("PyQt6")
    sys.modules.update({"PyQt6": pkg, "PyQt6.QtCore": qc, "PyQt6.QtGui": qg})
    sys.modules.pop("core.holo_fx", None)
    return importlib.import_module("core.holo_fx")


class T(unittest.TestCase):
    def setUp(self):
        self.fx = load(); self.fx.ENABLED = True

    def run_states(self, **over):
        for state in ("LISTENING", "THINKING", "PROCESSING", "SLEEPING", "IDLE", "SPEAKING"):
            for amp in (0.0, 0.5, 1.0, 7.0):                 # 7.0: out-of-range input is clamped
                st = {"state": state, "speaking": state == "SPEAKING", "amp": amp,
                      "muted": False, "bg": Color(), "span": 2.2}
                st.update(over)
                p = Painter()
                self.fx.paint_back(p, 300.0, 200.0, 150.0, Color(), st)
                self.fx.paint_front(p, 300.0, 200.0, 150.0, Color(), st)
                self.assertEqual(p.depth, 0, f"save/restore unbalanced in {state}")
                self.assertGreater(p.draws, 0)

    def test_all_states_draw_and_balance_painter(self):
        self.run_states()

    def test_muted(self):
        self.run_states(muted=True)

    def test_tiny_or_disabled_draws_nothing(self):
        p = Painter(); self.fx.paint_back(p, 0, 0, 5.0, Color(), {}); self.fx.paint_front(p, 0, 0, 5.0, Color(), {})
        self.assertEqual(p.draws, 0)
        self.fx.ENABLED = False
        self.fx.paint_back(p, 0, 0, 150.0, Color(), {}); self.fx.paint_front(p, 0, 0, 150.0, Color(), {})
        self.assertEqual(p.draws, 0)

    def test_missing_keys_are_fine(self):
        p = Painter(); self.fx.paint_back(p, 300.0, 200.0, 150.0, Color(), {}); self.fx.paint_front(p, 300.0, 200.0, 150.0, Color(), {})
        self.assertEqual(p.depth, 0)

    def test_standby_is_dimmer_than_active(self):
        self.assertLess(self.fx._intensity({"state": "SLEEPING"}), self.fx._intensity({"state": "LISTENING"}))
        self.assertLess(self.fx._intensity({"muted": True}), 1.0)

    def test_painter_restored_even_if_a_draw_raises(self):
        class Bad(Painter):
            def drawPath(self, *a): raise RuntimeError("boom")
        p = Bad()
        with self.assertRaises(RuntimeError):
            self.fx.paint_back(p, 300.0, 200.0, 150.0, Color(), {"state": "LISTENING"})
        self.assertEqual(p.depth, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
