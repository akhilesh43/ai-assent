"""
run_cloud.py — Entrypoint for Cloud Web Hosting (Render, Railway, Heroku, Docker)
Runs the JARVIS Mark-LVI Web Dashboard & API without requiring a local desktop GUI (PyQt6/X11).
"""
import os
import sys

def run():
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 [CLOUD] Starting JARVIS Mark-LVI Cloud Server on port {port}...")
    try:
        from dashboard.server import DashboardServer
        import uvicorn

        server = DashboardServer()
        print(f"✅ [CLOUD] Dashboard loaded! Serving HUD and APIs on port {port}...")
        uvicorn.run(server.app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        print(f"❌ [CLOUD] Server failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run()

