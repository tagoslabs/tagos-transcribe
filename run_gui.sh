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
# The shipped window calls itself com.appveyor.flet and paints data/app_icon.png.
export FLET_APP_ID=com.appveyor.flet
shopt -s nullglob
for root in "$HOME"/.flet/client/flet-desktop-*/flet; do
  if [ -f "$root/data/app_icon.png" ] && [ -f "$APP_DIR/assets/icons/256.png" ]; then
    cp -f "$APP_DIR/assets/icons/256.png" "$root/data/app_icon.png"
  fi
  for dest in "$root"/share/icons/hicolor/*/apps/com.appveyor.flet.png; do
    size="$(basename "$(dirname "$(dirname "$dest")")")"
    size="${size%x*}"
    src="$APP_DIR/assets/icons/${size}.png"
    if [ -f "$src" ]; then
      cp -f "$src" "$dest"
    fi
  done
done
shopt -u nullglob
exec python "$APP_DIR/gui.py" "$@"
