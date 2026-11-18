"""Test helper: lets `import main` work on a machine without PyQt6 / google-genai /
sounddevice / OpenCV etc. by serving MagicMock modules for anything missing.
Only for tests — never imported by the app."""
import importlib.abc, importlib.machinery, importlib.util, sys, types
from unittest import mock

_REAL_OK = {"numpy", "scipy", "requests", "PIL", "flask"}
_STUB_TOP = {"PyQt6", "google", "sounddevice", "cv2", "mss", "psutil", "pyautogui",
             "pyperclip", "playwright", "yt_dlp", "send2trash", "pptx", "qrcode",
             "cryptography", "fastapi", "uvicorn", "httpx", "bs4", "starlette",
             "multipart", "websockets", "pynput", "keyboard", "pystray", "win32api",
             "win32gui", "win32con", "pythoncom", "comtypes", "openwakeword",
             "faster_whisper", "ctranslate2", "onnxruntime", "piper", "firebase_admin"}


class _TypesModule(types.ModuleType):
    """google.genai.types: every attribute is a constructor returning a namespace of its
    keyword arguments, so `ThinkingConfig(thinking_budget=0).thinking_budget == 0`."""
    __path__ = []
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: types.SimpleNamespace(**k)


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        if spec.name == "google.genai.types":
            return _TypesModule(spec.name)
        m = mock.MagicMock(name=spec.name)
        m.__name__ = spec.name; m.__path__ = []; m.__spec__ = spec
        m.__loader__ = self; m.__file__ = f"<stub {spec.name}>"
        return m
    def exec_module(self, module):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    """Stubs a library only if it genuinely cannot be imported here. One that IS installed
    and works (cv2, psutil…) stays real, or tests would quietly exercise a MagicMock."""

    def __init__(self):
        self._decision: dict[str, bool] = {}          # top-level name -> True = stub it
        self._probing: set[str] = set()

    # `google` is a namespace package that exists here (protobuf…) while `google.genai` does not.
    _ALWAYS = {"google"}

    def _should_stub(self, top: str) -> bool:
        if top in self._ALWAYS:
            return True
        if top in self._decision:
            return self._decision[top]
        if importlib.machinery.PathFinder.find_spec(top) is None:
            self._decision[top] = True
            return True
        self._probing.add(top)
        try:
            importlib.import_module(top)              # e.g. sounddevice: present, but no PortAudio
            stub = False
        except Exception:
            for k in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
                sys.modules.pop(k, None)
            stub = True
        finally:
            self._probing.discard(top)
        self._decision[top] = stub
        return stub

    def find_spec(self, name, path, target=None):
        top = name.split(".")[0]
        if top not in _STUB_TOP or top in self._probing:
            return None
        if self._should_stub(top):
            return importlib.machinery.ModuleSpec(name, _Loader(), is_package=True)
        return None


def install():
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    # `ui` is a 9.5k-line Qt module; main only needs the name.
    if "ui" not in sys.modules:
        ui = types.ModuleType("ui"); ui.JarvisUI = mock.MagicMock(name="JarvisUI")
        sys.modules["ui"] = ui
