import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dub import (
    DEFAULT_DUCK,
    DEFAULT_MAX_SPEED,
    DEFAULT_THREADS,
    DEFAULT_VOICE,
    DEFAULT_VOICES,
    MIN_LENGTH_SCALE,
    PIPER_VOICES,
    TARGET_LANGUAGES,
    TRANSLATOR_THREAD_CAP,
    apply_atempo,
    atempo_chain,
    default_thread_count,
    default_voice_for,
    fit_clip_to_slot,
    main,
    parse_args,
    resegment,
    resolve_voice,
    segment_slot,
    select_translator,
    _options_from_args,
)


class TimingTests(unittest.TestCase):
    def test_defaults_are_the_prototype_fit(self) -> None:
        self.assertEqual(MIN_LENGTH_SCALE, 0.8)
        self.assertEqual(DEFAULT_MAX_SPEED, 1.6)
        self.assertEqual(DEFAULT_DUCK, 0.2)
        self.assertEqual(TRANSLATOR_THREAD_CAP, 4)
        self.assertEqual(default_thread_count(), 1)
        self.assertEqual(DEFAULT_THREADS, 1)
        self.assertEqual(DEFAULT_VOICE, "ru_RU-dmitri-medium")

    def test_phrase_may_use_the_pause_before_the_next_line(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "a"},
            {"start": 3.0, "end": 4.0, "text": "b"},
        ]
        start, slot = segment_slot(0, segments, cursor=0.0, total_duration=10.0)
        self.assertEqual(start, 0.0)
        self.assertEqual(slot, 3.0)

    def test_next_phrase_waits_until_the_previous_voice_ends(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "a"},
            {"start": 1.2, "end": 2.0, "text": "b"},
        ]
        start, slot = segment_slot(1, segments, cursor=2.5, total_duration=10.0)
        self.assertEqual(start, 2.5)
        self.assertAlmostEqual(slot, 7.5)

    def test_short_phrase_is_not_sped_up(self) -> None:
        sample_rate = 16000
        audio = np.full(int(sample_rate * 0.4), 0.2, np.float32)

        def unused(*_args, **_kwargs):
            raise AssertionError("speedup should not run")

        fitted, speed = fit_clip_to_slot(
            audio,
            sample_rate,
            1.0,
            unused,
            Path("."),
            DEFAULT_MAX_SPEED,
            atempo_fn=unused,
        )
        self.assertEqual(speed, 1.0)
        self.assertEqual(len(fitted), len(audio))

    def test_mild_overrun_uses_piper_length_scale_before_atempo(self) -> None:
        sample_rate = 16000
        base = int(sample_rate * 1.2)
        audio = np.full(base, 0.2, np.float32)
        scales: list[float] = []

        def resynth(length_scale: float) -> np.ndarray:
            scales.append(length_scale)
            return np.full(max(int(round(base * length_scale)), 1), 0.2, np.float32)

        def atempo(*_args, **_kwargs):
            raise AssertionError("atempo is only for what Piper could not absorb")

        _fitted, speed = fit_clip_to_slot(
            audio,
            sample_rate,
            1.0,
            resynth,
            Path("."),
            DEFAULT_MAX_SPEED,
            atempo_fn=atempo,
        )
        self.assertAlmostEqual(scales[0], 1 / 1.2, places=5)
        self.assertGreaterEqual(scales[0], MIN_LENGTH_SCALE)
        self.assertAlmostEqual(speed, 1.2, places=5)

    def test_long_phrase_caps_total_speed_at_1_6(self) -> None:
        sample_rate = 16000
        base = sample_rate * 2
        audio = np.full(base, 0.2, np.float32)
        scales: list[float] = []
        factors: list[float] = []

        def resynth(length_scale: float) -> np.ndarray:
            scales.append(length_scale)
            return np.full(max(int(round(base * length_scale)), 1), 0.2, np.float32)

        def atempo(samples, _sr, factor, _tmp):
            factors.append(factor)
            count = max(int(round(len(samples) / factor)), 1)
            return np.full(count, 0.2, np.float32)

        _fitted, speed = fit_clip_to_slot(
            audio,
            sample_rate,
            0.5,
            resynth,
            Path("."),
            1.6,
            atempo_fn=atempo,
        )
        self.assertAlmostEqual(scales[0], 0.8, places=5)
        self.assertEqual(len(factors), 1)
        self.assertAlmostEqual(factors[0], 1.28, places=3)
        self.assertAlmostEqual(speed, 1.6, places=3)
        self.assertLessEqual(speed, 1.6 + 1e-6)

    def test_atempo_chain_is_a_single_stage_under_the_default_cap(self) -> None:
        self.assertEqual(atempo_chain(1.28), "atempo=1.2800")
        self.assertEqual(atempo_chain(1.6), "atempo=1.6000")
        self.assertEqual(atempo_chain(3.0), "atempo=2.0000,atempo=1.5000")

    def test_ffmpeg_atempo_shortens_audio(self) -> None:
        sample_rate = 16000
        audio = np.full(sample_rate, 0.2, np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            sped = apply_atempo(audio, sample_rate, 1.25, Path(tmp))
        duration = len(sped) / sample_rate
        self.assertGreater(duration, 0.65)
        self.assertLess(duration, 0.95)


class ResegmentTests(unittest.TestCase):
    def test_sentence_and_pause_splits(self) -> None:
        segments = [
            {
                "start": 0.0,
                "end": 2.4,
                "text": "ignored",
                "words": [
                    (0.0, 0.4, " Hola"),
                    (0.4, 0.9, " mundo."),
                    (1.0, 1.6, " Adiós."),
                ],
            }
        ]
        parts = resegment(segments)
        self.assertEqual([part["text"] for part in parts], ["Hola mundo.", "Adiós."])

        paused = resegment(
            [
                {
                    "start": 0.0,
                    "end": 2.4,
                    "text": "x",
                    "words": [(0.0, 0.4, " Hola"), (2.0, 2.4, " mundo")],
                }
            ]
        )
        self.assertEqual(len(paused), 2)

    def test_long_unpunctuated_line_splits_on_a_short_pause(self) -> None:
        words = []
        cursor = 0.0
        while cursor < 9.0:
            words.append((cursor, cursor + 0.4, " слово"))
            cursor += 0.5
        words.append((cursor + 0.3, cursor + 0.7, " дальше"))
        parts = resegment(
            [{"start": 0.0, "end": words[-1][1], "text": "x", "words": words}],
            max_dur=8.0,
        )
        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(any(part["text"].endswith("дальше") for part in parts))

    def test_no_words_keeps_the_whisper_segments(self) -> None:
        original = [{"start": 0.0, "end": 1.0, "text": "hola", "words": []}]
        self.assertIs(resegment(original), original)


class TranslatorChoiceTests(unittest.TestCase):
    def test_auto_uses_argos_only_for_english_to_russian(self) -> None:
        self.assertEqual(select_translator("auto", "en", "ru"), "argos")
        self.assertEqual(select_translator("auto", "es", "ru"), "nllb")
        self.assertEqual(select_translator("auto", "en", "de"), "nllb")
        self.assertEqual(select_translator("nllb", "en", "ru"), "nllb")
        with self.assertRaises(ValueError):
            select_translator("argos", "es", "ru")


class VoiceSelectionTests(unittest.TestCase):
    def test_every_target_language_has_a_matching_piper_voice(self) -> None:
        self.assertEqual(set(DEFAULT_VOICES), {code for code, _label in TARGET_LANGUAGES})
        for code, _label in TARGET_LANGUAGES:
            voice = default_voice_for(code)
            self.assertIn(voice, PIPER_VOICES)
            self.assertEqual(PIPER_VOICES[voice]["lang"], code)

    def test_known_defaults(self) -> None:
        self.assertEqual(default_voice_for("ru"), "ru_RU-dmitri-medium")
        self.assertEqual(default_voice_for("es"), "es_ES-davefx-medium")
        self.assertEqual(default_voice_for("en"), "en_US-lessac-medium")
        self.assertEqual(default_voice_for("de"), "de_DE-thorsten-medium")
        self.assertEqual(default_voice_for("fr"), "fr_FR-siwis-medium")
        self.assertEqual(DEFAULT_VOICE, "ru_RU-dmitri-medium")

    def test_missing_voice_does_not_fall_back_to_russian(self) -> None:
        with self.assertRaises(ValueError) as caught:
            default_voice_for("xx")
        message = str(caught.exception)
        self.assertIn("no Piper voice", message)
        self.assertIn("'xx'", message)
        self.assertIn("another language", message)

    def test_explicit_voice_overrides_the_target_default(self) -> None:
        self.assertEqual(resolve_voice("es", None), "es_ES-davefx-medium")
        self.assertEqual(resolve_voice("es", "  "), "es_ES-davefx-medium")
        self.assertEqual(resolve_voice("es", "en_US-lessac-medium"), "en_US-lessac-medium")
        self.assertEqual(resolve_voice("xx", "ru_RU-ruslan-medium"), "ru_RU-ruslan-medium")

    def test_cli_target_selects_the_voice(self) -> None:
        spanish = _options_from_args(parse_args(["clip.mp4", "--target", "es"]))
        self.assertEqual(spanish.target_language, "es")
        self.assertEqual(spanish.voice, "es_ES-davefx-medium")

        english = _options_from_args(parse_args(["clip.mp4", "--target", "en"]))
        self.assertEqual(english.voice, "en_US-lessac-medium")

        override = _options_from_args(
            parse_args(["clip.mp4", "--target", "es", "--voice", "en_GB-alan-medium"])
        )
        self.assertEqual(override.target_language, "es")
        self.assertEqual(override.voice, "en_GB-alan-medium")

    def test_cli_rejects_a_target_with_no_piper_voice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "clip.wav"
            source.write_bytes(b"not audio")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main([str(source), "--target", "xx"])
        self.assertEqual(code, 1)
        message = stderr.getvalue()
        self.assertIn("no Piper voice", message)
        self.assertIn("'xx'", message)
        self.assertNotIn("transcribe", message.lower())


if __name__ == "__main__":
    unittest.main()
