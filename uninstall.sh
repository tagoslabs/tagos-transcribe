#!/usr/bin/env bash
# Removes this Transcribe folder and the menu launcher that points at it.
# Does not touch system packages, drivers, or any other directory.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(readlink -f "$APP_DIR")"
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/icons"
DESKTOP_FILES=(
  "${DESKTOP_DIR}/transcribe.desktop"
  "${DESKTOP_DIR}/com.appveyor.flet.desktop"
)
CLI_LINK="${HOME}/.local/bin/transcribe"
RUNNER="${APP_DIR}/run_gui.sh"

if [ ! -f "${APP_DIR}/gui.py" ] || [ ! -f "${APP_DIR}/setup.sh" ] || [ ! -f "${RUNNER}" ]; then
  echo "error: ${APP_DIR} is not a Transcribe folder" >&2
  exit 1
fi

case "${APP_DIR}" in
  / | /bin | /sbin | /usr | /usr/bin | /usr/lib | /usr/local | /etc | /var | /opt | /boot | /root | /home | /tmp | "${HOME}")
    echo "error: refusing to remove ${APP_DIR}" >&2
    exit 1
    ;;
esac

outside=()

icons_real="$(readlink -f "${ICONS_ROOT}")"
for desktop in "${DESKTOP_FILES[@]}"; do
  if [ -f "${desktop}" ] && grep -F -x -q "Exec=${RUNNER}" "${desktop}"; then
    outside+=("${desktop}")
    icon="$(grep -m1 '^Icon=' "${desktop}" | cut -d= -f2- || true)"
    if [ -n "${icon}" ] && [ -f "${icon}" ]; then
      icon_real="$(readlink -f "${icon}")"
      case "${icon_real}" in
        "${icons_real}"/transcribe.png | "${icons_real}"/transcribe.svg | "${icons_real}"/*/transcribe.png | "${icons_real}"/*/transcribe.svg)
          outside+=("${icon_real}")
          ;;
      esac
    fi
  fi
done
if [ -d "${ICONS_ROOT}" ]; then
  while IFS= read -r -d '' icon; do
    outside+=("${icon}")
  done < <(find "${ICONS_ROOT}" -type f \( -name 'transcribe.png' -o -name 'transcribe.svg' \) -print0)
fi

if [ -f "${CLI_LINK}" ] && grep -F -x -q "exec \"${RUNNER}\" \"\$@\"" "${CLI_LINK}"; then
  outside+=("${CLI_LINK}")
fi

echo "Удалить Transcribe?"
echo "Папка: ${APP_DIR}"
echo "В ней программа, свой Python, модели и ffmpeg."
if [ "${#outside[@]}" -eq 0 ]; then
  echo "Ярлыков этой копии в меню нет."
else
  echo "Снаружи папки будет снято только это:"
  printf '  %s\n' "${outside[@]}"
fi
echo "Системные пакеты, драйвер и другие папки не трогаются."
echo

confirmed=0
if [ "${1:-}" = "--yes" ]; then
  confirmed=1
elif [ -t 0 ]; then
  read -r -p "Напиши yes для удаления: " answer
  if [ "${answer}" = "yes" ] || [ "${answer}" = "да" ]; then
    confirmed=1
  fi
fi

if [ "${confirmed}" -ne 1 ]; then
  echo "Ничего не удалено."
  exit 0
fi

for path in "${outside[@]}"; do
  rm -f "${path}"
  echo "removed: ${path}"
done

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${DESKTOP_DIR}" >/dev/null 2>&1 || true
fi

cd /
rm -rf "${APP_DIR}"
echo "removed: ${APP_DIR}"
