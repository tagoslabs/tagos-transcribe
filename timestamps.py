"""Timestamp, SRT, and segment-file helpers.

Callers decide whether anything is written. Importing this module does not
touch the disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def segment_start(segment: Any) -> float:
    if isinstance(segment, dict):
        return float(segment["start"])
    return float(segment.start)


def segment_end(segment: Any) -> float:
    if isinstance(segment, dict):
        return float(segment["end"])
    return float(segment.end)


def segment_field(segment: Any, key: str = "text") -> str:
    if isinstance(segment, dict):
        if key == "translation":
            value = segment.get("translation")
            if value is None or value == "":
                value = segment.get("ru") or ""
            return str(value)
        return str(segment.get(key) or "")
    if key == "text":
        return str(segment.text or "")
    return ""


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(float(seconds) * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt_timestamp(value: str) -> float:
    text = value.strip().replace(".", ",")
    hours, minutes, rest = text.split(":")
    seconds, millis = rest.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def segments_to_timed_text(segments: Iterable[Any], *, text_key: str = "text") -> str:
    lines: list[str] = []
    for segment in segments:
        text = segment_field(segment, text_key).strip()
        if not text:
            continue
        start = format_srt_timestamp(segment_start(segment))
        end = format_srt_timestamp(segment_end(segment))
        lines.append(f"[{start} --> {end}] {text}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def segments_to_srt(
    segments: Iterable[Any],
    *,
    text_key: str = "text",
    total_duration: float | None = None,
    extend_cues: bool = False,
    skip_empty: bool = False,
) -> str:
    """Serialize cues.

    extend_cues matches the prototype subtitle timing: keep a line on screen
    long enough to read (~15 characters/second) without running into the next cue.
    Speech timings themselves are unchanged; this only affects the SRT end time.
    """
    rows = list(segments)
    blocks: list[str] = []
    for index, segment in enumerate(rows):
        text = segment_field(segment, text_key).strip()
        if skip_empty and not text:
            continue
        start = segment_start(segment)
        end = segment_end(segment)
        if extend_cues:
            if index + 1 < len(rows):
                nxt = segment_start(rows[index + 1]) - 0.05
            elif total_duration is not None:
                nxt = float(total_duration)
            else:
                nxt = end
            readable = start + (len(text) / 15.0 if text else 0.0)
            end = max(end, min(nxt, readable))
        if end < start:
            end = start
        blocks.append(
            f"{len(blocks) + 1}\n"
            f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
            f"{text}\n"
        )
    if not blocks:
        return ""
    return "\n".join(blocks) + "\n"


def format_transcript_preview(
    source_name: str,
    language: str | None,
    text: str,
    segments: Iterable[Any] | None = None,
    with_timestamps: bool = False,
    file_label: str = "Файл",
    language_label: str = "Язык",
    unknown_language: str = "не определён",
) -> str:
    """Preview shown in the transcription window.

    With timestamps off, the body is the plain transcript. That is the
    historical preview, including the header lines.
    """
    header = f"{file_label}: {source_name}\n{language_label}: {language or unknown_language}\n\n"
    if with_timestamps:
        body = segments_to_timed_text(segments or [], text_key="text").rstrip("\n")
    else:
        body = text
    return header + body


def segments_payload(
    segments: Iterable[Any],
    *,
    language: str | None,
    target_language: str | None,
    source: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        if isinstance(segment, dict):
            text = str(segment.get("text") or "")
            translation = segment.get("translation")
            if translation is None:
                translation = segment.get("ru") or ""
            start = float(segment["start"])
            end = float(segment["end"])
        else:
            text = str(segment.text or "")
            translation = ""
            start = float(segment.start)
            end = float(segment.end)
        item: dict[str, Any] = {
            "start": start,
            "end": end,
            "text": text,
            "translation": str(translation or ""),
        }
        rows.append(item)
    payload: dict[str, Any] = {
        "language": language,
        "target_language": target_language,
        "segments": rows,
    }
    if source:
        payload["source"] = source
    return payload


def dump_segments(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def load_segments(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"segments": data}
    if "segments" not in data or not isinstance(data["segments"], list):
        raise ValueError(f"{path} has no segment list")
    segments: list[dict[str, Any]] = []
    for index, segment in enumerate(data["segments"], start=1):
        if not isinstance(segment, dict):
            raise ValueError(f"segment {index} is not an object")
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"segment {index} needs numeric start and end") from exc
        translation = segment.get("translation")
        if translation is None:
            translation = segment.get("ru") or ""
        segments.append(
            {
                "start": start,
                "end": end,
                "text": str(segment.get("text") or ""),
                "translation": str(translation or ""),
            }
        )
    return {
        "language": data.get("language"),
        "target_language": data.get("target_language") or "ru",
        "source": data.get("source"),
        "segments": segments,
    }


_SRT_TIME = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


def parse_srt(text: str) -> list[dict[str, Any]]:
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        time_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        match = _SRT_TIME.search(lines[time_index])
        if not match:
            continue
        body = "\n".join(lines[time_index + 1 :]).strip()
        cues.append(
            {
                "start": parse_srt_timestamp(match.group("start")),
                "end": parse_srt_timestamp(match.group("end")),
                "text": body,
            }
        )
    if not cues:
        raise ValueError("no cues found in SRT")
    return cues


def apply_srt_cues(
    segments: list[dict[str, Any]],
    cues: list[dict[str, Any]],
    *,
    field: str = "translation",
) -> str:
    """Update translated lines when the cue count matches.

    SRT ends from the dubbing exporter are display times, not speech ends.
    Matching by index keeps the speech timings used for voice-over.
    Returns 'text' when only the field was replaced, or 'timings' when the
    cue list replaced the whole segment list.
    """
    if field not in {"translation", "text"}:
        raise ValueError(field)
    if len(cues) == len(segments) and segments:
        for segment, cue in zip(segments, cues):
            segment[field] = cue["text"]
        return "text"
    replaced: list[dict[str, Any]] = []
    for cue in cues:
        replaced.append(
            {
                "start": cue["start"],
                "end": cue["end"],
                "text": cue["text"] if field == "text" else "",
                "translation": cue["text"] if field == "translation" else "",
            }
        )
    segments.clear()
    segments.extend(replaced)
    return "timings"


def resolve_export_path(
    requested: str,
    source: Path,
    suffix: str,
    *,
    multiple: bool,
) -> Path:
    path = Path(requested).expanduser()
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    looks_like_file = path.suffix.lower() == suffix.lower()
    if multiple and looks_like_file:
        raise ValueError(
            f"{requested} is a single file, but several inputs were given. "
            "Pass a directory instead."
        )
    if looks_like_file and not multiple:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{source.stem}{suffix}"
