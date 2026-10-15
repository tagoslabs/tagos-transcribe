#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "venv not found. run ./setup.sh first" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
exec python "$APP_DIR/voiceover.py" "$@"
