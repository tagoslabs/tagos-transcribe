import tempfile
import unittest
from pathlib import Path

import numpy as np

import benchmark
from faster_whisper.vad import get_speech_timestamps


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original = benchmark.BENCHMARK_FILE
        self._dir = tempfile.TemporaryDirectory()
        benchmark.BENCHMARK_FILE = Path(self._dir.name) / "benchmarks.json"

    def tearDown(self) -> None:
        benchmark.BENCHMARK_FILE = self._original
        self._dir.cleanup()

    def test_sample_is_detected_as_speech(self) -> None:
        audio = benchmark.synthesize_speech(2.0)
        spans = get_speech_timestamps(audio, sampling_rate=benchmark.SAMPLE_RATE)
        speech = sum(item["end"] - item["start"] for item in spans) / benchmark.SAMPLE_RATE
        self.assertGreater(speech, 1.5)

    def test_estimate_uses_saved_rate(self) -> None:
        benchmark.remember("small", "cpu", "int8", audio_seconds=30, wall_seconds=60)
        self.assertEqual(benchmark.estimate_seconds(10, "small", "cpu", "int8"), 20)
        self.assertIsNone(benchmark.estimate_seconds(10, "small", "cuda", "int8"))

    def test_gimbal_budget_is_the_measured_job_not_the_clip(self) -> None:
        audio = 840.441
        finish = 72.94
        startup = 6.37
        benchmark.remember(
            "small", "cuda", "int8", audio, finish, startup=startup
        )
        estimate = benchmark.estimate_seconds(audio, "small", "cuda", "int8")
        self.assertAlmostEqual(estimate, finish)
        self.assertGreater(estimate, 60)
        self.assertNotAlmostEqual(estimate, 0.03786557029581909 * audio, delta=5)

    def test_later_passes_do_not_move_the_rate(self) -> None:
        benchmark.remember("small", "cpu", "int8", audio_seconds=30, wall_seconds=60)
        benchmark.remember("small", "cpu", "int8", audio_seconds=30, wall_seconds=30)
        self.assertEqual(benchmark.rate_for("small", "cpu", "int8"), 2.0)

    def test_tiny_pass_is_ignored(self) -> None:
        benchmark.remember("small", "cpu", "int8", audio_seconds=0.2, wall_seconds=0.1)
        self.assertIsNone(benchmark.rate_for("small", "cpu", "int8"))

    def test_cpu_and_gpu_are_separate(self) -> None:
        benchmark.remember("small", "cpu", "int8", 30, 90)
        benchmark.remember("small", "cuda", "int8", 30, 15)
        self.assertAlmostEqual(benchmark.rate_for("small", "cpu", "int8"), 3.0)
        self.assertAlmostEqual(benchmark.rate_for("small", "cuda", "int8"), 0.5)
        self.assertEqual(len(audio := benchmark.synthesize_speech(0.2)), int(0.2 * benchmark.SAMPLE_RATE))
        self.assertEqual(audio.dtype, np.float32)
