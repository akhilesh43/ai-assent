"""
Clipboard Smart Copilot plugin — explain, summarize, fix, translate, or
extract tasks from whatever text or code the user currently has copied.
"""
from __future__ import annotations

from core import gemini

PLUGIN = {
    "name": "clipboard_copilot",
    "tier": "extra",
    "brief": "Analyze, explain, translate, summarize, or fix copied clipboard text/code.",
    "description": (
        "Read the user's current clipboard content and explain it, summarize it, "
        "fix code or errors in it, translate it, or extract action items. "
        "Use when the user says 'explain what I copied', 'fix this copied error', "
        "'translate my clipboard', or 'summarize clipboard'."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "task": {
                "type": "STRING",
                "enum": ["explain", "summarize", "fix_code", "translate", "tasks"],
                "description": "What to do with the clipboard text.",
            },
            "instruction": {
                "type": "STRING",
                "description": "Optional extra direction (e.g. target language for translate).",
            },
        },
        "required": [],
    },
}


def _read_clipboard() -> str:
    try:
        import pyperclip
        return (pyperclip.paste() or "").strip()
    except Exception:
        return ""


def run(parameters: dict, player=None, session_memory=None) -> str:
    task = str((parameters or {}).get("task") or "explain").strip().lower()
    extra = str((parameters or {}).get("instruction") or "").strip()
    try:
        clip = _read_clipboard()
        if not clip:
            return "Your clipboard is empty right now."

        snippet = clip[:6000]
        prompts = {
            "explain": "Explain the following text or code clearly and concisely.",
            "summarize": "Summarize the key points of the following text in bullet points.",
            "fix_code": "Diagnose the bug or error below and provide the corrected code or fix.",
            "translate": f"Translate the following text{' (' + extra + ')' if extra else ' into clear English/Hindi as requested'}.",
            "tasks": "Extract a clean, actionable checklist of tasks from the following text.",
        }
        base_prompt = prompts.get(task, prompts["explain"])
        if extra and task != "translate":
            base_prompt = f"{base_prompt}\nUser instruction: {extra}"

        tier = gemini.SMART if task in ("fix_code", "explain") else gemini.FAST
        answer = gemini.text(f"{base_prompt}\n\n---\n{snippet}", tier=tier, timeout_ms=30_000)

        if not answer:
            lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
            preview = " ".join(lines[:3])[:220]
            answer = (
                f"Model unavailable right now. Clipboard preview ({len(clip)} chars): {preview}"
            )

        if player is not None:
            try:
                player.show_content(f"CLIPBOARD — {task.upper()}", answer)
            except Exception:
                pass

        first_line = answer.strip().splitlines()[0][:180]
        return first_line or "Clipboard analysis shown on screen."
    except Exception as e:
        return f"Clipboard copilot failed: {e}"

