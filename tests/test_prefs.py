import tempfile
import unittest
from pathlib import Path

import prefs


class PrefsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original = prefs.PREFS_FILE
        self._dir = tempfile.TemporaryDirectory()
        prefs.PREFS_FILE = Path(self._dir.name) / "ui_prefs.json"

    def tearDown(self) -> None:
        prefs.PREFS_FILE = self._original
        self._dir.cleanup()

    def test_choice_returns_saved_value(self) -> None:
        prefs.save_pref("device", "cuda")
        self.assertEqual(prefs.choice("device", {"cpu", "cuda"}, "cpu"), "cuda")

    def test_choice_ignores_unknown_value(self) -> None:
        prefs.save_pref("device", "tpu")
        self.assertEqual(prefs.choice("device", {"cpu", "cuda"}, "cpu"), "cpu")

    def test_missing_file_uses_default(self) -> None:
        self.assertEqual(prefs.choice("model", {"small"}, "small"), "small")
