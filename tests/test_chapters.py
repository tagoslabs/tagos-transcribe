import re
import unittest

from chapters import (
    MIN_CHAPTER,
    ChapterError,
    build_chapters,
    chapters_text,
    format_chapter_time,
    make_title,
)

LINE = re.compile(r"^(\d+:)?\d{1,2}:\d{2} \S.*$")


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def talk():
    topics = [
        "Cooking pasta needs salted water and fresh tomatoes.",
        "Bicycles need chain oil and tire pressure checks.",
        "Planets orbit stars because gravity bends space.",
        "Gardens need compost, seeds and regular watering.",
    ]
    rows, t = [], 0.0
    for topic in topics:
        for _ in range(8):
            rows.append(seg(t, t + 4.0, topic))
            t += 4.5
        t += 3.0  # pause between topics
    return rows


def to_seconds(stamp):
    parts = [int(p) for p in stamp.split(":")]
    total = 0
    for p in parts:
        total = total * 60 + p
    return total


class ChapterTests(unittest.TestCase):
    def test_youtube_rules(self):
        text = chapters_text(talk())
        lines = text.strip().split("\n")
        self.assertGreaterEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("0:00 "))
        for line in lines:
            self.assertRegex(line, LINE)
        times = [to_seconds(line.split(" ")[0]) for line in lines]
        self.assertEqual(times, sorted(times))
        for a, b in zip(times, times[1:]):
            self.assertGreaterEqual(b - a, MIN_CHAPTER)

    def test_cuts_at_topic_shifts(self):
        chapters = build_chapters(talk())
        titles = [title for _, title in chapters]
        self.assertTrue(any(t.startswith("Bicycles") for t in titles))
        self.assertTrue(any(t.startswith("Planets") for t in titles))

    def test_too_short(self):
        with self.assertRaises(ChapterError):
            build_chapters([seg(0, 5, "hi"), seg(5, 10, "there")])
        with self.assertRaises(ChapterError):
            build_chapters([])

    def test_few_long_segments_fallback(self):
        chapters = build_chapters([seg(0, 15, "alpha one"), seg(15, 30, "beta two"), seg(30, 45, "gamma three")])
        self.assertEqual([s for s, _ in chapters], [0.0, 15, 30])

    def test_time_format(self):
        self.assertEqual(format_chapter_time(0), "0:00")
        self.assertEqual(format_chapter_time(65.9), "1:05")
        self.assertEqual(format_chapter_time(3725), "1:02:05")

    def test_title(self):
        self.assertEqual(make_title("so, um, welcome to the channel. Today we cook."), "Welcome to the channel")
        self.assertEqual(make_title("ну вот, сегодня поговорим про сад"), "Сегодня поговорим про сад")
        self.assertLessEqual(len(make_title("x" * 10 + " " + "y" * 60)), 48)


if __name__ == "__main__":
    unittest.main()
