"""
main.py — Entry point for the Meridian dashboard.

Run with:
    python main.py        (or ./run.sh dashboard)

The dashboard opens automatically in your default browser at
http://localhost:8050.

Hot-reload is ON. Two benefits:
  • edit dashboard.py / assets and the page refreshes itself, and
  • when you restart the server, any already-open tab auto-reloads and picks up
    the new code — so a stale tab can never keep calling a removed callback
    (the source of the "Callback function not found" 500s).
"""

import os
import threading
import time
import webbrowser

PORT = 8050
HOST = "127.0.0.1"
DEBUG = True  # enables hot-reload + client auto-refresh on restart


def _open_browser() -> None:
    # Give the server ~1.5s to bind its socket before we open the tab.
    time.sleep(1.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    # Import here so any startup errors (missing packages etc.) surface before
    # the browser opens.
    from dashboard import app

    # With the reloader on, this file runs twice (a supervisor + the worker).
    # Open the browser exactly once: in the worker (WERKZEUG_RUN_MAIN == "true")
    # when reloading, or unconditionally when the reloader is off.
    is_worker = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if is_worker or not DEBUG:
        print()
        print("  MERIDIAN — multi-asset long/short research dashboard")
        print("  " + "-" * 50)
        print(f"  Opening at   http://{HOST}:{PORT}")
        print("  Stop         Ctrl+C")
        print()
        threading.Thread(target=_open_browser, daemon=True).start()

    app.run(host=HOST, port=PORT, debug=DEBUG)
