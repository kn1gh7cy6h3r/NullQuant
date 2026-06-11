#!/usr/bin/env bash
# NullQuant launcher.
#   ./run.sh            -> run the full research pipeline (data -> ablation -> report artifacts)
#   ./run.sh pipeline   -> same as above
#   ./run.sh tests      -> run the pytest suite
#   ./run.sh dashboard  -> launch the interactive dashboard
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

cmd="${1:-pipeline}"
case "$cmd" in
  pipeline)  python -m nullquant.pipeline "${@:2}" ;;
  tests)     python -m pytest tests/ -q ;;
  dashboard) python main.py ;;
  *) echo "usage: ./run.sh [pipeline|tests|dashboard]"; exit 1 ;;
esac
