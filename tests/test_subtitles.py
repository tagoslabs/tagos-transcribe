import re
import unittest

from timestamps import (
    SUB_MAX_DURATION,
    SUB_MAX_LINE,
    SUB_MAX_LINES,
    parse_srt,
    segments_to_subtitles,
    wrap_subtitle_lines,
)

CUE = re.compile(r"^(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.+\n?)+)$")


def validate(srt: str):
    """Strict SubRip check: what YouTube and players accept."""
    assert srt.endswith("\n")
    blocks = srt.strip("\n").split("\n\n")
    previous_end = -1.0
    cues = []
    for number, block in enumerate(blocks, 1):
        match = CUE.match(block)
        assert match, f"bad block: {block!r}"
        assert int(match.group(1)) == number
        start = parse_srt(f"1\n{match.group(2)} --> {match.group(3)}\nx\n")[0]
        assert start["end"] > start["start"], block
        assert start["start"] >= previous_end, block
        previous_end = start["end"]
        lines = match.group(4).strip("\n").split("\n")
        cues.append((start["start"], start["end"], lines))
    return cues


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


class SubtitleFormatTests(unittest.TestCase):
    def test_basic_cues_are_valid(self):
        srt = segments_to_subtitles([seg(0, 2, "Hello."), seg(2.5, 4, "Привет, мир.")])
        cues = validate(srt)
        self.assertEqual(srt.split("\n")[1], "00:00:00,000 --> 00:00:02,000")
        self.assertEqual(cues[1][2], ["Привет, мир."])

    def test_hours_and_millis(self):
        srt = segments_to_subtitles([seg(3723.456, 3725.0, "x")])
        self.assertIn("01:02:03,456 --> 01:02:05,000", srt)

    def test_long_text_wrapped_and_split(self):
        text = " ".join(["word"] * 60)
        cues = validate(segments_to_subtitles([seg(0, 20, text)]))
        self.assertGreater(len(cues), 1)
        for start, end, lines in cues:
            self.assertLessEqual(len(lines), SUB_MAX_LINES)
            self.assertTrue(all(len(line) <= SUB_MAX_LINE for line in lines))
            self.assertLessEqual(end - start, SUB_MAX_DURATION + 1e-6)
        self.assertEqual(" ".join(" ".join(c[2]) for c in cues), text)

    def test_overlaps_removed(self):
        cues = validate(segments_to_subtitles([seg(0, 5, "a"), seg(3, 6, "b")]))
        self.assertLess(cues[0][1], cues[1][0])

    def test_min_duration(self):
        cues = validate(segments_to_subtitles([seg(1, 1.1, "Ok."), seg(10, 11, "b")]))
        self.assertGreaterEqual(cues[0][1] - cues[0][0], 0.69)

    def test_zero_length_never_inverted(self):
        validate(segments_to_subtitles([seg(1, 1, "a"), seg(1, 1, "b")]))

    def test_empty_dropped_and_renumbered(self):
        srt = segments_to_subtitles([seg(0, 1, "a"), seg(1, 2, "  "), seg(2, 3, "c")])
        self.assertEqual([c[2] for c in validate(srt)], [["a"], ["c"]])

    def test_no_segments(self):
        self.assertEqual(segments_to_subtitles([]), "")

    def test_round_trip_parse(self):
        srt = segments_to_subtitles([seg(0, 2, "one two"), seg(3, 4, "three")])
        self.assertEqual([c["text"] for c in parse_srt(srt)], ["one two", "three"])

    def test_no_split_keeps_cue_count(self):
        segments = [seg(0, 30, " ".join(["long"] * 40)), seg(31, 32, "b")]
        cues = validate(segments_to_subtitles(segments, max_duration=None))
        self.assertEqual(len(cues), 2)

    def test_objects_like_whisper_segments(self):
        class S:
            def __init__(self, start, end, text):
                self.start, self.end, self.text = start, end, text

        validate(segments_to_subtitles([S(0.0, 1.5, " Hi there ")]))

    def test_wrap(self):
        self.assertEqual(wrap_subtitle_lines("aa bb cc", 5), ["aa bb", "cc"])
        self.assertEqual(wrap_subtitle_lines("x" * 50, 10), ["x" * 50])


if __name__ == "__main__":
    unittest.main()
