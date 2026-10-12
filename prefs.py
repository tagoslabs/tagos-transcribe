"""Choices the GUI should still have after a restart."""

from __future__ import annotations

import json
from pathlib import Path

PREFS_FILE = Path(__file__).resolve().parent / "ui_prefs.json"


def load_prefs() -> dict:
    try:
        raw = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def save_pref(key: str, value: str) -> None:
    data = load_prefs()
    data[key] = value
    try:
        PREFS_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def choice(key: str, allowed: set[str], default: str) -> str:
    value = load_prefs().get(key)
    if isinstance(value, str) and value in allowed:
        return value
    return default
