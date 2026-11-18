"""
Productivity Copilot plugin — intelligent workspace context analysis, executive briefing,
task extraction, and system optimization recommendations via core.gemini and player
UI methods (show_review, show_content).
"""
from __future__ import annotations

import threading
import psutil

try:
    from core import gemini
except ImportError:
    try:
        import core.gemini as gemini
    except ImportError:
        gemini = None

PLUGIN = {
    "name": "productivity_copilot",
    "tier": "extra",
    "brief": "Analyze workspace context, generate executive briefings, extract tasks, and recommend system optimizations.",
    "description": (
        "Analyze current workspace context and running processes, generate an executive "
        "briefing or task checklist, and provide system optimization recommendations. "
        "Use when the user asks for a productivity check, executive briefing, workspace "
        "analysis, task extraction, or system optimization."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "task": {
                "type": "STRING",
                "enum": ["briefing", "tasks", "optimize", "analyze"],
                "description": "Productivity task to perform: briefing, tasks, optimize, or analyze.",
            },
            "focus": {
                "type": "STRING",
                "description": "Optional specific focus or project area for the analysis.",
            },
        },
        "required": [],
    },
}


def _get_workspace_metrics() -> dict:
    """Gather system resource usage and active process context safely using psutil."""
    metrics = {
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "processes": [],
    }
    try:
        metrics["cpu_percent"] = round(psutil.cpu_percent(interval=0.1), 1)
        mem = psutil.virtual_memory()
        metrics["ram_percent"] = round(mem.percent, 1)
        metrics["ram_used_gb"] = round(mem.used / (1024 ** 3), 1)
        metrics["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)

        procs = []
        for p in psutil.process_iter(["name", "cpu_percent", "memory_percent"]):
            try:
                info = p.info
                name = info.get("name")
                if name and name.lower() not in ("system idle process", "idle"):
                    cpu_p = info.get("cpu_percent") or 0.0
                    mem_p = info.get("memory_percent") or 0.0
                    procs.append((name, cpu_p, mem_p))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        procs.sort(key=lambda x: x[2], reverse=True)
        metrics["processes"] = procs[:8]
    except Exception:
        pass
    return metrics


def _analyze_and_update_ui(task: str, focus: str, player) -> None:
    """Offload LLM calls and context gathering to a background worker thread."""
    try:
        metrics = _get_workspace_metrics()
        proc_list = metrics["processes"]
        proc_summary = (
            ", ".join(f"{name} ({mem:.1f}% RAM)" for name, cpu, mem in proc_list)
            if proc_list
            else "No high-resource background processes detected"
        )

        focus_str = focus if focus else "General Workstation"
        ctx_prompt = (
            f"System State: CPU {metrics['cpu_percent']}%, RAM {metrics['ram_percent']}% "
            f"({metrics['ram_used_gb']}/{metrics['ram_total_gb']} GB).\n"
            f"Active Top Processes: {proc_summary}.\n"
            f"Focus Area: {focus_str}.\n"
            f"Requested Action: {task}.\n"
        )

        prompts = {
            "briefing": (
                "Generate a concise executive briefing on current workspace status, operational readiness, "
                "and active workload. List 3 to 5 key status findings or recommendations."
            ),
            "tasks": (
                "Extract a clean, prioritized checklist of actionable tasks and immediate next steps "
                "for the user based on the current workspace context and focus."
            ),
            "optimize": (
                "Analyze the system workload and resource usage below. Provide specific, actionable "
                "system optimization recommendations to improve performance and workflow efficiency."
            ),
            "analyze": (
                "Perform an intelligent workspace context analysis, identifying potential bottlenecks, "
                "distractions, or resource constraints."
            ),
        }

        instruction = prompts.get(task, prompts["briefing"])
        full_prompt = f"{instruction}\n\nWorkspace Context:\n{ctx_prompt}"

        answer = None
        if gemini is not None:
            try:
                answer = gemini.text(full_prompt, tier=gemini.SMART, timeout_ms=30_000)
            except Exception:
                answer = None

        if not answer:
            answer = (
                f"Workspace Analysis ({task.upper()}):\n"
                f"- CPU Utilization: {metrics['cpu_percent']}%\n"
                f"- RAM Utilization: {metrics['ram_percent']}% ({metrics['ram_used_gb']} GB / {metrics['ram_total_gb']} GB)\n"
                f"- Top Active Workload: {proc_summary}\n"
                f"- Status: Workstation operational. Focus area: {focus_str}"
            )

        if player is not None:
            try:
                if task == "briefing":
                    lines = [ln.strip("- *").strip() for ln in answer.splitlines() if ln.strip()]
                    findings = lines[:6] if lines else [
                        f"CPU Load: {metrics['cpu_percent']}%",
                        f"RAM Load: {metrics['ram_percent']}%",
                        f"Active focus: {focus_str}",
                    ]
                    player.show_review(
                        "EXECUTIVE BRIEFING",
                        f"Workspace status for {focus_str}",
                        findings,
                    )
                else:
                    player.show_content(f"PRODUCTIVITY COPILOT — {task.upper()}", answer)

                player.write_log(f"JARVIS: Productivity {task} completed.")
            except Exception as e:
                try:
                    player.write_log(f"ERR: Productivity Copilot UI update failed: {e}")
                except Exception:
                    pass
    except Exception as e:
        if player is not None:
            try:
                player.write_log(f"ERR: Productivity Copilot background analysis failed: {e}")
            except Exception:
                pass


def run(parameters: dict, player=None, session_memory=None) -> str:
    params = parameters or {}
    task = str(params.get("task") or "briefing").strip().lower()
    focus = str(params.get("focus") or "").strip()

    valid_tasks = ("briefing", "tasks", "optimize", "analyze")
    if task not in valid_tasks:
        task = "briefing"

    threading.Thread(
        target=_analyze_and_update_ui,
        args=(task, focus, player),
        daemon=True,
        name="productivity-copilot-worker",
    ).start()

    messages = {
        "briefing": "Preparing your executive workspace briefing now, sir.",
        "tasks": "Extracting your workspace tasks and action items, sir.",
        "optimize": "Analyzing system load and preparing optimization recommendations, sir.",
        "analyze": "Analyzing your workspace context and workload efficiency, sir.",
    }

    response = messages.get(task, "Analyzing workspace context now, sir.")
    if player is not None:
        try:
            player.write_log(f"JARVIS: {response}")
        except Exception:
            pass
    return response
