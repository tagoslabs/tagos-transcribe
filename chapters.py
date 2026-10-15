"""Offline YouTube chapters from a timed transcript.

No model and no network: boundaries come from pauses and word-overlap drops
between the text before and after each segment. YouTube needs the first
chapter at 0:00, at least three chapters, and each at least 10 seconds long.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from timestamps import segment_end, segment_field, segment_start

MIN_CHAPTER = 10.0
MIN_CHAPTERS = 3
TARGET_CHAPTER = 120.0
MAX_CHAPTERS = 15
TITLE_WORDS = 6
TITLE_CHARS = 48
WINDOW = 4

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
STOPWORDS = set(
    "the a an and or but so of to in on at for with is are was were be it this that these those i you we "
    "he she they my your our um uh like just okay ok well yeah yes no not do does did have has had there here "
    "и в во не что он она они я ты мы вы на с со как а то это так но по за из у к же ну вот да нет бы ли "
    "его её их мне меня там тут ещё уже или если чтобы".split()
)
FILLER_START = re.compile(r"^(?:(?:so|and|but|okay|ok|well|um|uh|now|right|итак|так|ну|и|а|вот|короче)[,\s]+)+",
                          re.IGNORECASE)


class ChapterError(ValueError):
    """The transcript is too short for YouTube's chapter rules."""


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if len(w) > 2 and w.lower() not in STOPWORDS}


def format_chapter_time(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def make_title(text: str) -> str:
    text = FILLER_START.sub("", " ".join(text.split())).strip(" ,.;:-–—")
    sentence = re.split(r"(?<=[.!?…])\s", text, maxsplit=1)[0].rstrip(".!?… ,;:")
    words = sentence.split()
    words = words[:TITLE_WORDS]
    while len(words) > 1 and words[-1].lower().strip(",;:") in STOPWORDS:
        words.pop()
    title = " ".join(words).rstrip(",;:")
    if len(title) > TITLE_CHARS:
        title = title[:TITLE_CHARS].rsplit(" ", 1)[0]
    if not title:
        return ""
    return title[0].upper() + title[1:]


def boundary_scores(rows: list[tuple[float, float, str]]) -> list[float]:
    """Score for starting a chapter at each segment (index 0 is unused)."""
    scores = [0.0] * len(rows)
    for i in range(1, len(rows)):
        gap = max(rows[i][0] - rows[i - 1][1], 0.0)
        before = set().union(*(_words(r[2]) for r in rows[max(0, i - WINDOW):i]))
        after = set().union(*(_words(r[2]) for r in rows[i:i + WINDOW]))
        union = before | after
        overlap = len(before & after) / len(union) if union else 1.0
        scores[i] = min(gap, 3.0) / 3.0 + (1.0 - overlap)
    return scores


def build_chapters(segments: Iterable[Any], *, intro_title: str = "Intro") -> list[tuple[float, str]]:
    rows = [(segment_start(s), segment_end(s), segment_field(s).strip()) for s in segments]
    rows = [r for r in rows if r[2]]
    if not rows:
        raise ChapterError("empty transcript")
    duration = max(r[1] for r in rows)
    if duration < MIN_CHAPTER * MIN_CHAPTERS:
        raise ChapterError(f"needs at least {int(MIN_CHAPTER * MIN_CHAPTERS)} s of speech")
    wanted = int(min(MAX_CHAPTERS, max(MIN_CHAPTERS, round(duration / TARGET_CHAPTER))))
    scores = boundary_scores(rows)
    starts = [0.0]
    picked = [0]
    for i in sorted(range(1, len(rows)), key=lambda i: -scores[i]):
        if len(picked) >= wanted:
            break
        t = rows[i][0]
        if duration - t < MIN_CHAPTER or any(abs(t - s) < MIN_CHAPTER for s in starts):
            continue
        starts.append(t)
        picked.append(i)
    if len(picked) < MIN_CHAPTERS:
        # Too few segments to cut at good spots: cut at even time steps instead.
        step = duration / MIN_CHAPTERS
        picked = [0]
        for k in range(1, MIN_CHAPTERS):
            index = min(range(len(rows)), key=lambda i: abs(rows[i][0] - k * step))
            if index not in picked:
                picked.append(index)
        starts = [0.0] + [rows[i][0] for i in picked[1:]]
        if len(picked) < MIN_CHAPTERS or any(b - a < MIN_CHAPTER for a, b in zip(sorted(starts), sorted(starts)[1:])):
            raise ChapterError("not enough separate segments for 3 chapters")
    picked.sort()
    chapters: list[tuple[float, str]] = []
    for n, index in enumerate(picked):
        start = 0.0 if n == 0 else rows[index][0]
        end_index = picked[n + 1] if n + 1 < len(picked) else len(rows)
        text = " ".join(r[2] for r in rows[index:end_index])
        title = make_title(text) or f"Part {n + 1}"
        if n == 0 and not title:
            title = intro_title
        chapters.append((start, title))
    return chapters


def chapters_text(segments: Iterable[Any]) -> str:
    """Lines ready to paste into a YouTube description: ``0:00 Title``."""
    return "\n".join(f"{format_chapter_time(s)} {title}" for s, title in build_chapters(segments)) + "\n"
