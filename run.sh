#!/usr/bin/env bash
# Activate the virtual environment and launch Meridian.
# Usage: ./run.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate
python main.py
