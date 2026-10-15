import re
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import voiceover
from voiceover import (
    VoiceoverInput,
    cues_to_segments,
    export_audio,
    installed_voices,
    preview_text,
    read_input,
    speed_to_length_scale,
    split_chunks,
    strip_timecodes,
    synthesize_plain,
    transcript_to_input,
    voice_language,
    voices_for_language,
)


class FakeSynth:
    def __init__(self, rate=1000):
        self.rate = rate
        self.calls = []

    def __call__(self, text, length_scale):
        self.calls.append((text, length_scale))
        return np.full(int(len(text) * 10 * length_scale), 0.5, np.float32)


class ChunkTests(unittest.TestCase):
    def test_sentences_and_paragraph_pauses(self):
        chunks = split_chunks("Привет. Как дела?\n\nВторой абзац!")
        self.assertEqual([c[0] for c in chunks], ["Привет.", "Как дела?", "Второй абзац!"])
        self.assertEqual(chunks[0][1], voiceover.SENTENCE_PAUSE)
        self.assertEqual(chunks[1][1], voiceover.PARAGRAPH_PAUSE)
        self.assertEqual(chunks[-1][1], 0.0)

    def test_blank_text_has_no_chunks(self):
        self.assertEqual(split_chunks("  \n\n "), [])

    def test_line_breaks_inside_paragraph_join(self):
        self.assertEqual(split_chunks("one\ntwo"), [("one two", 0.0)])

    def test_preview_is_short(self):
        text = "Short one. " * 100
        self.assertLessEqual(len(preview_text(text)), voiceover.PREVIEW_CHARS)
        self.assertTrue(preview_text(text).startswith("Short one."))


class SpeedTests(unittest.TestCase):
    def test_length_scale(self):
        self.assertAlmostEqual(speed_to_length_scale(2.0), 0.5)
        self.assertAlmostEqual(speed_to_length_scale(1.0), 1.0)

    def test_out_of_range(self):
        for bad in (0.1, 3.0):
            with self.assertRaises(ValueError):
                speed_to_length_scale(bad)


class InputTests(unittest.TestCase):
    def test_reads_srt_with_cues(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.srt"
            path.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:04,500\nWorld\n",
                            encoding="utf-8")
            source = read_input(path)
        self.assertTrue(source.timed)
        self.assertEqual(source.text, "Hello\nWorld")
        self.assertEqual(source.cues[1]["start"], 3.0)

    def test_reads_txt_and_strips_transcribe_timecodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            path.write_text("\ufeff[00:00:01,200 --> 00:00:04,000] Привет.\nplain line\n", encoding="utf-8")
            source = read_input(path)
        self.assertFalse(source.timed)
        self.assertEqual(source.text, "Привет.\nplain line")

    def test_strip_timecodes_leaves_brackets_in_text(self):
        self.assertEqual(strip_timecodes("a [note] b"), "a [note] b")

    def test_transcript_to_input_keeps_timings(self):
        segments = [SimpleNamespace(start=0.5, end=1.5, text=" Hi. "), SimpleNamespace(start=2, end=3, text=" ")]
        source = transcript_to_input(segments)
        self.assertEqual(source.text, "Hi.")
        self.assertEqual(source.cues, [{"start": 0.5, "end": 1.5, "text": "Hi."}])

    def test_cues_to_segments_feeds_dub_synth(self):
        segs = cues_to_segments([{"start": 1, "end": 2, "text": "a"}, {"start": 3, "end": 4, "text": ""}])
        self.assertEqual(segs, [{"start": 1.0, "end": 2.0, "text": "a", "translation": "a"}])


class VoiceListTests(unittest.TestCase):
    def test_installed_needs_onnx_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            piper = Path(tmp) / "piper"
            piper.mkdir()
            (piper / "en_GB-alan-medium.onnx").write_bytes(b"")
            (piper / "en_GB-alan-medium.onnx.json").write_text("{}")
            (piper / "ru_RU-half-medium.onnx").write_bytes(b"")
            self.assertEqual(installed_voices(Path(tmp)), ["en_GB-alan-medium"])
            self.assertEqual(voices_for_language("en", Path(tmp))[:2], ["en_GB-alan-medium", "en_US-lessac-medium"])
            self.assertEqual(voices_for_language("ru", Path(tmp)), ["ru_RU-dmitri-medium", "ru_RU-ruslan-medium", "ru_RU-irina-medium", "ru_RU-denis-medium"])

    def test_missing_dir(self):
        self.assertEqual(installed_voices(Path("/nonexistent/x")), [])

    def test_voice_language(self):
        self.assertEqual(voice_language("ru_RU-dmitri-medium"), "ru")
        self.assertEqual(voice_language("de_DE-custom-low"), "de")


class SynthTests(unittest.TestCase):
    def test_plain_concatenates_with_pauses(self):
        synth = FakeSynth()
        audio, rate = synthesize_plain("Ab. Cd.", "x", speed=2.0, synth=synth, sample_rate=1000)
        self.assertEqual(rate, 1000)
        self.assertEqual([c[1] for c in synth.calls], [0.5, 0.5])
        expected = 15 + int(voiceover.SENTENCE_PAUSE * 1000) + 15
        self.assertEqual(len(audio), expected)
        self.assertAlmostEqual(float(np.abs(audio).max()), 0.95, places=5)

    def test_plain_empty_raises(self):
        with self.assertRaises(ValueError):
            synthesize_plain("   ", "x", synth=FakeSynth(), sample_rate=1000)

    def test_timed_uses_dub_track(self):
        captured = {}

        def fake_track(segments, voice, models_dir, total, tmp, max_speed, **kw):
            captured.update(segments=segments, total=total, kw=kw)
            return np.zeros(10, np.float32), 22050, []

        original = voiceover.synthesize_track
        voiceover.synthesize_track = fake_track
        try:
            voiceover.synthesize_timed([{"start": 1, "end": 2, "text": "a"}, {"start": 3, "end": 5, "text": "b"}],
                                       "v", speed=1.25)
        finally:
            voiceover.synthesize_track = original
        self.assertEqual(captured["total"], 5)
        self.assertAlmostEqual(captured["kw"]["length_scale"], 0.8)
        self.assertEqual(len(captured["segments"]), 2)


class ExportTests(unittest.TestCase):
    def test_wav_and_mp3(self):
        audio = np.sin(np.linspace(0, 400, 8000)).astype(np.float32) * 0.5
        with tempfile.TemporaryDirectory() as tmp:
            wav = export_audio(audio, 8000, Path(tmp) / "a.wav")
            with wave.open(str(wav)) as w:
                self.assertEqual((w.getframerate(), w.getnchannels(), w.getnframes()), (8000, 1, 8000))
            mp3 = export_audio(audio, 8000, Path(tmp) / "a.mp3")
            head = mp3.read_bytes()[:3]
            self.assertTrue(head == b"ID3" or head[0] == 0xFF)

    def test_bad_suffix(self):
        with self.assertRaises(ValueError):
            export_audio(np.zeros(1, np.float32), 8000, Path("/tmp/x.ogg"))


if __name__ == "__main__":
    unittest.main()
