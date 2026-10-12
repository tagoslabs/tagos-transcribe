#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/venv"
RUNTIME_DIR="$APP_DIR/runtime"
MODELS_DIR="$APP_DIR/models"
BIN_DIR="$APP_DIR/bin"

# One CPython build for every machine. The system Python is not used.
PY_RELEASE="20261009"
PY_VERSION="3.13.16+${PY_RELEASE}"

echo "==> setting up transcribe in $APP_DIR"

if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
  echo "error: curl and tar are required to download the bundled runtime" >&2
  exit 1
fi

python_triple() {
  case "$(uname -m)" in
    x86_64) echo "x86_64-unknown-linux-gnu" ;;
    aarch64 | arm64) echo "aarch64-unknown-linux-gnu" ;;
    *)
      echo "error: no bundled Python for $(uname -m)" >&2
      exit 1
      ;;
  esac
}

ensure_python() {
  local triple want url tmp
  triple="$(python_triple)"
  want="${PY_VERSION}-${triple}"
  if [ -x "$RUNTIME_DIR/bin/python3" ] && [ "$(cat "$RUNTIME_DIR/.bundle-version" 2>/dev/null || true)" = "$want" ]; then
    echo "==> bundled Python ${PY_VERSION}"
    return
  fi
  echo "==> downloading Python ${PY_VERSION} (${triple})"
  url="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_RELEASE}/cpython-${PY_VERSION}-${triple}-install_only.tar.gz"
  tmp="$(mktemp -d)"
  curl -fL --retry 3 "$url" | tar -xz -C "$tmp"
  rm -rf "$RUNTIME_DIR"
  mv "$tmp/python" "$RUNTIME_DIR"
  echo "$want" > "$RUNTIME_DIR/.bundle-version"
  rm -rf "$tmp"
}

ensure_venv() {
  local venv_python runtime_python
  runtime_python="$(readlink -f "$RUNTIME_DIR/bin/python3")"
  venv_python=""
  if [ -x "$VENV_DIR/bin/python" ]; then
    venv_python="$(readlink -f "$VENV_DIR/bin/python")"
  fi
  if [ "$venv_python" = "$runtime_python" ]; then
    echo "==> virtual environment already uses bundled Python"
    return
  fi
  echo "==> creating virtual environment from bundled Python"
  rm -rf "$VENV_DIR"
  "$RUNTIME_DIR/bin/python3" -m venv "$VENV_DIR"
}

ensure_ffmpeg() {
  mkdir -p "$BIN_DIR"
  if [ -x "$BIN_DIR/ffmpeg" ] && [ -x "$BIN_DIR/ffprobe" ]; then
    echo "==> bundled ffmpeg already present"
    return
  fi
  if [ "$(uname -m)" != "x86_64" ]; then
    if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
      echo "==> using system ffmpeg on $(uname -m): $(command -v ffmpeg)"
      return
    fi
    echo "error: bundled ffmpeg is built for x86_64. Install ffmpeg and ffprobe, then run ./setup.sh again." >&2
    exit 1
  fi
  echo "==> downloading ffmpeg"
  local tmp
  tmp="$(mktemp -d)"
  curl -fL --retry 3 "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz" \
    | tar -xJ -C "$tmp" --strip-components=1
  mv "$tmp/ffmpeg" "$BIN_DIR/ffmpeg"
  mv "$tmp/ffprobe" "$BIN_DIR/ffprobe"
  chmod +x "$BIN_DIR/ffmpeg" "$BIN_DIR/ffprobe"
  rm -rf "$tmp"
}

ensure_python
ensure_venv

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: --no-deps -r "$APP_DIR/requirements.lock"

echo "==> CUDA libraries for NVIDIA GPUs (skipped when there is no driver)"
python -c "from transcribe import ensure_cuda_libraries; print(ensure_cuda_libraries() or 'skipped')"

ensure_ffmpeg
mkdir -p "$MODELS_DIR"

echo "==> downloading default model 'small' (others download on first use in GUI)"
python - "$MODELS_DIR" <<'PY'
import sys
from pathlib import Path
from faster_whisper import WhisperModel

models_dir = Path(sys.argv[1])
models_dir.mkdir(parents=True, exist_ok=True)
WhisperModel("small", device="cpu", compute_type="int8", download_root=str(models_dir))
print("model ready in", models_dir)
PY

chmod +x "$APP_DIR/transcribe.py" "$APP_DIR/gui.py" "$APP_DIR/dub.py" \
  "$APP_DIR/run.sh" "$APP_DIR/run_gui.sh" "$APP_DIR/dub.sh" \
  "$APP_DIR/install.sh" "$APP_DIR/uninstall.sh"

echo
echo "setup complete."
echo "gui:     ./run_gui.sh"
echo "install: ./install.sh   # ярлык в меню приложений"
echo "cli:     ./run.sh your_file.mp4"
echo "or:      ./run.sh -r /path/to/folder"
echo "dub:     ./dub.sh your_video.mp4"
echo
echo "Python, packages, and ffmpeg live in this folder."
echo "Whisper 'small' is downloaded now."
echo "Translation and Piper voices download on first use into models/."
