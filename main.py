"""
main.py — Entry point for Meridian.

Run with:
    python main.py

The dashboard opens automatically in your default browser at
http://localhost:8050 and auto-refreshes every 30 seconds.
"""

import threading
import time
import webbrowser

PORT = 8050
HOST = "127.0.0.1"


def _open_browser() -> None:
    # Give the Dash dev-server ~1.5s to bind its socket before we open the tab.
    time.sleep(1.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    # Import here so any startup errors (missing packages etc.) surface before
    # the browser opens.
    from dashboard import app

    print()
    print("  MERIDIAN")
    print("  BTC Trading Signal Dashboard")
    print("  " + "-" * 42)
    print(f"  Opening at        http://{HOST}:{PORT}")
    print("  Live refresh      every 30 seconds")
    print("  Stop              Ctrl+C")
    print()

    threading.Thread(target=_open_browser, daemon=True).start()

    app.run(
        host=HOST,
        port=PORT,
        debug=False,   # Set True for hot-reload during development
    )
