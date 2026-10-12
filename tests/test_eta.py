"""The job clock is the budget counted down, one second per second.

The budget is fixed when the file length is known. Readings from the
transcription are not allowed to move it.
"""

import unittest

from eta import EtaClock, display_seconds

# The gimbal file: 14:00.441 of audio finished in 72.94s on this machine.
# Startup 6.37s, then the rest of the wall time spread across the file.
GIMBAL_AUDIO = 840.441
GIMBAL_FINISH = 72.94
GIMBAL_STARTUP = 6.37
GIMBAL_RATE = (GIMBAL_FINISH - GIMBAL_STARTUP) / GIMBAL_AUDIO

BUDGETS = [12.5, 20, 32, 60, 72.94, 73, 90, 120, 180, 300, 600, 840.441, 900, 1800, 3600]
STEPS = range(0, 41)


def _case(budget: float, step: int) -> None:
    elapsed = budget * step / 40
    clock = EtaClock(budget)
    clock.start(0.0)
    remaining = clock.remaining(elapsed)
    expected = budget - elapsed
    if remaining is None or abs(remaining - expected) > 1e-9:
        raise AssertionError(
            f"countdown moved: budget={budget} elapsed={elapsed} "
            f"left={remaining} expected={expected}"
        )
    if step < 40 and display_seconds(remaining) <= 0:
        raise AssertionError(f"unfinished budget drawn as zero: {budget} at step {step}")


def _make_case(budget: float, step: int):
    def test(self) -> None:
        _case(budget, step)

    test.__name__ = f"test_budget_{budget}_step_{step}"
    return test


class EtaFormulaTests(unittest.TestCase):
    def test_gimbal_file_is_one_minute_thirteen_not_thirty_seconds(self) -> None:
        budget = GIMBAL_STARTUP + GIMBAL_RATE * GIMBAL_AUDIO
        self.assertAlmostEqual(budget, GIMBAL_FINISH)
        self.assertGreater(budget, 60.0)
        self.assertNotAlmostEqual(budget, 0.03786557029581909 * GIMBAL_AUDIO, delta=5)
        clock = EtaClock(budget)
        self.assertAlmostEqual(clock.remaining(0.0), GIMBAL_FINISH)
        clock.start(0.0)
        self.assertAlmostEqual(clock.remaining(0.0), GIMBAL_FINISH)
        self.assertAlmostEqual(clock.remaining(10.0), GIMBAL_FINISH - 10.0)
        self.assertEqual(display_seconds(budget), 73)
        self.assertEqual(clock.remaining(GIMBAL_FINISH), 0.0)

    def test_zero_only_when_the_budget_is_spent(self) -> None:
        clock = EtaClock(100.0)
        clock.start(0.0)
        self.assertGreater(clock.remaining(99.9), 0.0)
        self.assertEqual(clock.remaining(100.0), 0.0)
        self.assertEqual(clock.remaining(140.0), 0.0)

    def test_before_start_the_whole_budget_is_shown(self) -> None:
        clock = EtaClock(72.94)
        self.assertAlmostEqual(clock.remaining(500.0), 72.94)

    def test_no_budget_shows_nothing(self) -> None:
        clock = EtaClock(None)
        clock.start(0.0)
        self.assertIsNone(clock.remaining(1.0))

    def test_case_count_is_in_the_hundreds(self) -> None:
        names = [name for name in dir(EtaFormulaTests) if name.startswith("test_budget_")]
        self.assertGreaterEqual(len(names), 200)


for _budget in BUDGETS:
    for _step in STEPS:
        _test = _make_case(_budget, _step)
        setattr(EtaFormulaTests, _test.__name__, _test)
