"""
main.py — Entry point for the NullQuant dashboard.

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
# Hot-reload (Werkzeug auto-reloader) watches the source tree and runs a second
# Python process — handy while editing dashboard.py, but pure idle CPU if you're
# just using the dashboard. Off by default; opt in with NULLQUANT_DEBUG=1 when
# actively developing.
DEBUG = os.environ.get("NULLQUANT_DEBUG", "0") not in ("0", "", "false", "False")


def _open_browser() -> None:
    # Give the server ~1.5s to bind its socket before we open the tab.
    time.sleep(1.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    # Import here so any startup errors (missing packages etc.) surface before
    # the browser opens.
    from dashboard import app, warm_results

    # With the reloader on, this file runs twice (a supervisor + the worker).
    # Do startup work exactly once: in the worker (WERKZEUG_RUN_MAIN == "true")
    # when reloading, or unconditionally when the reloader is off.
    is_worker = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if is_worker or not DEBUG:
        print()
        print("  NULLQUANT — multi-asset long/short research dashboard")
        print("  " + "-" * 50)
        print(f"  Opening at   http://{HOST}:{PORT}")
        print("  Stop         Ctrl+C")
        print()
        print("  Preparing research… reuses the pipeline's cache when present.")
        print("  A cold cache fits the ML models once (~5 min); the page shows a")
        print("  spinner until ready — the server is responsive, not frozen.")
        print()
        # Warm the results off the request thread so the server binds and stays
        # responsive instead of blocking on the first page load.
        threading.Thread(target=warm_results, daemon=True).start()
        threading.Thread(target=_open_browser, daemon=True).start()

    app.run(host=HOST, port=PORT, debug=DEBUG)
