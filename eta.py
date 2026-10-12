"""Countdown of a budget fixed before the job starts.

The budget is the file length run through the machine's measured speed.
Nothing inside the job is allowed to change it, so the number cannot jump.
"""

from __future__ import annotations

import math
import threading


def display_seconds(seconds: float) -> int:
    """Whole seconds still on the clock. A fraction of a second stays visible."""
    if seconds <= 0:
        return 0
    return math.ceil(seconds - 1e-9)


class EtaClock:
    def __init__(self, budget: float | None) -> None:
        if budget is None or budget < 0:
            self.budget = None
        else:
            self.budget = float(budget)
        self._lock = threading.Lock()
        self._started: float | None = None

    def start(self, now: float) -> None:
        with self._lock:
            if self._started is None:
                self._started = float(now)

    def remaining(self, now: float) -> float | None:
        with self._lock:
            if self.budget is None:
                return None
            if self._started is None:
                return self.budget
            left = self.budget - (float(now) - self._started)
            return left if left > 0 else 0.0
