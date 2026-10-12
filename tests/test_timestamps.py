import tempfile
import unittest
from pathlib import Path

from timestamps import (
    apply_srt_cues,
    format_srt_timestamp,
    format_transcript_preview,
    load_segments,
    parse_srt,
    parse_srt_timestamp,
    resolve_export_path,
    segments_payload,
    segments_to_srt,
    segments_to_timed_text,
    dump_segments,
)


class TimestampFormatTests(unittest.TestCase):
    def test_srt_timestamp_roundtrip(self) -> None:
        self.assertEqual(format_srt_timestamp(1.5), "00:00:01,500")
        self.assertEqual(format_srt_timestamp(3661.007), "01:01:01,007")
        self.assertAlmostEqual(parse_srt_timestamp("01:01:01,007"), 3661.007, places=3)
        self.assertAlmostEqual(parse_srt_timestamp("00:00:01.500"), 1.5, places=3)

    def test_preview_without_timestamps_matches_the_old_window(self) -> None:
        preview = format_transcript_preview("a.mp4", "ru", "привет мир", [], False)
        self.assertEqual(preview, "Файл: a.mp4\nЯзык: ru\n\nпривет мир")
        unknown = format_transcript_preview("a.mp4", None, "", [], False)
        self.assertEqual(unknown, "Файл: a.mp4\nЯзык: не определён\n\n")

    def test_timed_text_and_raw_srt(self) -> None:
        segments = [{"start": 0.34, "end": 1.2, "text": "Hola"}]
        self.assertEqual(
            segments_to_timed_text(segments),
            "[00:00:00,340 --> 00:00:01,200] Hola\n",
        )
        srt = segments_to_srt(segments)
        self.assertIn("00:00:00,340 --> 00:00:01,200", srt)
        self.assertIn("Hola", srt)

    def test_extended_cue_uses_pause_but_not_the_next_line(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "hi", "translation": "привет"},
            {"start": 5.0, "end": 5.2, "text": "x", "translation": "я" * 60},
        ]
        srt = segments_to_srt(
            segments,
            text_key="translation",
            total_duration=20,
            extend_cues=True,
        )
        self.assertIn("00:00:00,000 --> 00:00:01,000", srt)
        self.assertIn("00:00:05,000 --> 00:00:09,000", srt)

    def test_srt_import_keeps_speech_timings_when_counts_match(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "hola", "translation": "старое"},
            {"start": 2.0, "end": 3.0, "text": "adios", "translation": "тоже"},
        ]
        cues = parse_srt(
            "1\n00:00:00,000 --> 00:00:04,000\nПривет\n\n"
            "2\n00:00:04,000 --> 00:00:08,000\nПока\n"
        )
        how = apply_srt_cues(segments, cues, field="translation")
        self.assertEqual(how, "text")
        self.assertEqual(segments[0]["translation"], "Привет")
        self.assertEqual(segments[0]["end"], 1.0)
        self.assertEqual(segments[1]["text"], "adios")

    def test_json_roundtrip_accepts_prototype_ru_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.segments.json"
            dump_segments(
                path,
                {
                    "language": "es",
                    "segments": [
                        {"start": 0.0, "end": 1.25, "text": "Hola", "ru": "Привет"},
                    ],
                },
            )
            loaded = load_segments(path)
        self.assertEqual(loaded["segments"][0]["translation"], "Привет")
        self.assertEqual(loaded["target_language"], "ru")

    def test_payload_and_export_path(self) -> None:
        payload = segments_payload(
            [{"start": 1, "end": 2, "text": "a", "translation": "б"}],
            language="es",
            target_language="ru",
            source="video.mp4",
        )
        self.assertEqual(payload["segments"][0]["translation"], "б")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            one = resolve_export_path(str(directory), Path("lecture.mp4"), ".srt", multiple=True)
            self.assertEqual(one.name, "lecture.srt")
            single = resolve_export_path(
                str(directory / "out.srt"),
                Path("lecture.mp4"),
                ".srt",
                multiple=False,
            )
            self.assertEqual(single.name, "out.srt")
            with self.assertRaises(ValueError):
                resolve_export_path(str(directory / "out.srt"), Path("a.mp4"), ".srt", multiple=True)


class CliDefaultTests(unittest.TestCase):
    def test_transcribe_cli_stays_print_only(self) -> None:
        from transcribe import parse_args

        args = parse_args(["clip.mp4"])
        self.assertEqual(args.model, "small")
        self.assertIsNone(args.language)
        self.assertFalse(args.timestamps)
        self.assertIsNone(args.save_srt)
        self.assertIsNone(args.save_timed_text)
        self.assertFalse(args.recursive)
        self.assertEqual(args.device, "auto")

    def test_compute_type_follows_the_device(self) -> None:
        from transcribe import resolve_compute_type

        self.assertEqual(resolve_compute_type("float32", "cpu"), "float32")
        self.assertEqual(resolve_compute_type("auto", "cpu"), "int8")
        self.assertIn(resolve_compute_type("auto", "cuda"), {"float16", "int8_float16", "int8", "int8_float32", "float32"})

    def test_cublas_pin_follows_the_driver(self) -> None:
        from transcribe import cublas_version_for_driver

        self.assertIsNone(cublas_version_for_driver(None))
        self.assertIsNone(cublas_version_for_driver((545, 29)))
        self.assertEqual(cublas_version_for_driver((550, 163)), "12.4.5.8")
        self.assertEqual(cublas_version_for_driver((560, 28)), "12.6.4.1")
        self.assertEqual(cublas_version_for_driver((570, 26)), "12.8.5.5")
        self.assertEqual(cublas_version_for_driver((575, 51)), "12.9.2.10")

    def test_dub_cli_defaults_match_the_requested_mix(self) -> None:
        from dub import parse_args

        args = parse_args(["video.mp4"])
        from dub import default_thread_count

        self.assertEqual(args.duck, 0.2)
        self.assertEqual(args.max_speed, 1.6)
        self.assertEqual(args.threads, default_thread_count())
        self.assertLessEqual(args.threads, 4)
        self.assertEqual(args.voice, "ru_RU-dmitri-medium")
        self.assertEqual(args.target, "ru")
        self.assertEqual(args.translator, "auto")
        self.assertFalse(args.embed_subs)
        self.assertFalse(args.no_original_track)
        self.assertIsNone(args.segments)


if __name__ == "__main__":
    unittest.main()
