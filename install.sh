#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
DESKTOP_DIR="${DATA_HOME}/applications"
ICON_ROOT="${DATA_HOME}/icons"
BIN_DIR="${HOME}/.local/bin"
# The Flet window registers itself as com.appveyor.flet. The desktop file
# name has to be that id, or GNOME shows Flet's own picture in the dock.
DESKTOP_FILE="${DESKTOP_DIR}/com.appveyor.flet.desktop"
OLD_DESKTOP_FILE="${DESKTOP_DIR}/transcribe.desktop"
ICON_FILE="${ICON_ROOT}/transcribe.png"
CLI_LINK="${BIN_DIR}/transcribe"

echo "==> installing Transcribe"
echo "    app dir: ${APP_DIR}"

if [ ! -x "${APP_DIR}/run_gui.sh" ]; then
  echo "error: run_gui.sh not found in ${APP_DIR}" >&2
  exit 1
fi

echo "==> bundled Python, packages, ffmpeg, and the default model"
"${APP_DIR}/setup.sh"

mkdir -p "${DESKTOP_DIR}" "${ICON_ROOT}/hicolor/scalable/apps" "${BIN_DIR}"
cp "${APP_DIR}/assets/icons/512.png" "${ICON_FILE}"
cp "${APP_DIR}/assets/transcribe.svg" "${ICON_ROOT}/hicolor/scalable/apps/transcribe.svg"
for size in 16 24 32 48 64 128 256 512; do
  dest="${ICON_ROOT}/hicolor/${size}x${size}/apps"
  mkdir -p "${dest}"
  cp "${APP_DIR}/assets/icons/${size}.png" "${dest}/transcribe.png"
done

if [ -f "${OLD_DESKTOP_FILE}" ] && grep -F -x -q "Exec=${APP_DIR}/run_gui.sh" "${OLD_DESKTOP_FILE}"; then
  rm -f "${OLD_DESKTOP_FILE}"
fi

cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Transcribe
GenericName=Audio Transcription
Comment=Offline audio and video transcription
Exec=${APP_DIR}/run_gui.sh
TryExec=${APP_DIR}/run_gui.sh
Icon=${ICON_FILE}
StartupWMClass=com.appveyor.flet
Terminal=false
Categories=AudioVideo;Utility;Audio;
Keywords=transcribe;whisper;audio;video;subtitle;speech;
StartupNotify=true
EOF

cat > "${CLI_LINK}" <<EOF
#!/usr/bin/env bash
exec "${APP_DIR}/run_gui.sh" "\$@"
EOF
chmod +x "${CLI_LINK}"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${DESKTOP_DIR}" >/dev/null 2>&1 || true
fi

echo
echo "installed."
echo
echo "launch:"
echo "  - from app menu: Transcribe"
echo "  - from terminal: transcribe"
echo "  - directly:      ${APP_DIR}/run_gui.sh"
echo
echo "to remove: ${APP_DIR}/uninstall.sh   # asks first; deletes only this folder"
