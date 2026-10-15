#!/usr/bin/env python3
"""Offline voice-over: text or subtitles to speech with an installed Piper voice.

Plain text is split into sentences and spoken back to back with short pauses.
An .srt can be spoken the same way, or as a timed track where every cue starts
at its subtitle time. The timed path is the dubbing synthesizer from ``dub.py``
(same slot fitting and speed cap), so both modes share one TTS pipeline.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dub import (
    DEFAULT_MAX_SPEED,
    DEFAULT_VOICES,
    PIPER_VOICES,
    CancelCallback,
    ProgressCallback,
    check_cancel,
    load_piper_voice,
    make_synth,
    synthesize_track,
    trim_silence,
    write_wav,
)
from timestamps import parse_srt, segment_end, segment_field, segment_start
from transcribe import MODELS_DIR, find_ffmpeg

SENTENCE_PAUSE = 0.35
PARAGRAPH_PAUSE = 0.8
MIN_SPEED = 0.5
MAX_SPEED = 2.0
PREVIEW_CHARS = 220
_SENTENCE_END = re.compile(r"(?<=[.!?…。！？])\s+")


@dataclass
class VoiceoverInput:
    text: str
    cues: list[dict] | None = None  # set when the source was an .srt

    @property
    def timed(self) -> bool:
        return bool(self.cues)


def voice_language(voice_name: str) -> str:
    known = PIPER_VOICES.get(voice_name)
    if known:
        return known["lang"]
    return voice_name.split("_", 1)[0].split("-", 1)[0].lower()


def installed_voices(models_dir: Path = MODELS_DIR) -> list[str]:
    """Piper voices present on disk (model and config both downloaded)."""
    directory = Path(models_dir) / "piper"
    if not directory.is_dir():
        return []
    names = [
        onnx.stem
        for onnx in directory.glob("*.onnx")
        if onnx.with_name(onnx.name + ".json").is_file()
    ]
    return sorted(names)


def voices_for_language(language: str, models_dir: Path = MODELS_DIR) -> list[str]:
    """Installed voices first; the catalog default is added (downloads on first use)."""
    voices = [name for name in installed_voices(models_dir) if voice_language(name) == language]
    default = DEFAULT_VOICES.get(language)
    if default and default not in voices:
        voices.append(default)
    return voices


def voice_label(voice_name: str) -> str:
    known = PIPER_VOICES.get(voice_name)
    return known["label"] if known else voice_name


def speed_to_length_scale(speed: float) -> float:
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}")
    return 1.0 / speed


def read_input(path: Path) -> VoiceoverInput:
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() == ".srt":
        cues = parse_srt(raw)
        return VoiceoverInput(text="\n".join(cue["text"] for cue in cues), cues=cues)
    return VoiceoverInput(text=strip_timecodes(raw))


_TIMED_LINE = re.compile(r"^\s*\[\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:\s*-->?\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)?\]\s*")


def strip_timecodes(text: str) -> str:
    """Drop leading ``[00:01 --> 00:04]``-style stamps from Transcribe's timed text."""
    return "\n".join(_TIMED_LINE.sub("", line) for line in text.splitlines())


def split_chunks(text: str) -> list[tuple[str, float]]:
    """Sentences to speak and the pause after each one."""
    chunks: list[tuple[str, float]] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if p.strip()]
    for paragraph in paragraphs:
        flat = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        sentences = [s.strip() for s in _SENTENCE_END.split(flat) if s.strip()]
        for sentence in sentences:
            chunks.append((sentence, SENTENCE_PAUSE))
        if chunks:
            chunks[-1] = (chunks[-1][0], PARAGRAPH_PAUSE)
    if chunks:
        chunks[-1] = (chunks[-1][0], 0.0)
    return chunks


def preview_text(text: str, limit: int = PREVIEW_CHARS) -> str:
    chunks = split_chunks(text)
    picked: list[str] = []
    for sentence, _pause in chunks:
        if picked and len(" ".join(picked + [sentence])) > limit:
            break
        picked.append(sentence)
    joined = " ".join(picked)
    return joined[:limit] if len(joined) > limit else joined


def transcript_to_input(segments) -> VoiceoverInput:
    """A Transcribe result as Voiceover input: one line per segment, timings kept."""
    cues = [
        {"start": segment_start(seg), "end": segment_end(seg), "text": segment_field(seg).strip()}
        for seg in segments
        if segment_field(seg).strip()
    ]
    return VoiceoverInput(text="\n".join(c["text"] for c in cues), cues=cues or None)


def cues_to_segments(cues: list[dict]) -> list[dict]:
    return [
        {"start": float(c["start"]), "end": float(c["end"]), "text": c["text"], "translation": c["text"]}
        for c in cues
        if str(c.get("text", "")).strip()
    ]


def synthesize_plain(
    text: str,
    voice_name: str,
    *,
    speed: float = 1.0,
    models_dir: Path = MODELS_DIR,
    on_progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
    synth=None,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    chunks = split_chunks(text)
    if not chunks:
        raise ValueError("no text to speak")
    length_scale = speed_to_length_scale(speed)
    with tempfile.TemporaryDirectory(prefix="voiceover-") as tmp:
        if synth is None:
            voice = load_piper_voice(voice_name, models_dir, on_progress=on_progress, cancel=cancel)
            sample_rate = int(voice.config.sample_rate)
            synth = make_synth(voice, Path(tmp))
        assert sample_rate
        parts: list[np.ndarray] = []
        for index, (sentence, pause) in enumerate(chunks):
            check_cancel(cancel)
            parts.append(trim_silence(synth(sentence, length_scale)))
            if pause:
                parts.append(np.zeros(int(pause * sample_rate), np.float32))
            if on_progress:
                on_progress("tts", (index + 1) / len(chunks), sentence)
    audio = np.concatenate(parts).astype(np.float32)
    peak = float(np.abs(audio).max() or 1.0)
    return audio / peak * 0.95, sample_rate


def synthesize_timed(
    cues: list[dict],
    voice_name: str,
    *,
    speed: float = 1.0,
    max_speed: float = DEFAULT_MAX_SPEED,
    models_dir: Path = MODELS_DIR,
    on_progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
) -> tuple[np.ndarray, int, list[dict]]:
    """Each cue starts at its subtitle time; long lines are fitted like in Dub."""
    segments = cues_to_segments(cues)
    if not segments:
        raise ValueError("no subtitle text to speak")
    total = max(float(c["end"]) for c in cues)
    with tempfile.TemporaryDirectory(prefix="voiceover-") as tmp:
        return synthesize_track(
            segments,
            voice_name,
            models_dir,
            total,
            Path(tmp),
            max_speed,
            on_progress=on_progress,
            cancel=cancel,
            length_scale=speed_to_length_scale(speed),
        )


def export_audio(audio: np.ndarray, sample_rate: int, output: Path, ffmpeg: str | None = None) -> Path:
    """Write WAV, or MP3 through ffmpeg (already required by the app)."""
    output = Path(output)
    suffix = output.suffix.lower()
    if suffix == ".wav":
        write_wav(output, audio, sample_rate)
        return output
    if suffix != ".mp3":
        raise ValueError(f"unsupported output format '{output.suffix}', use .wav or .mp3")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voiceover-") as tmp:
        wav = Path(tmp) / "voice.wav"
        write_wav(wav, audio, sample_rate)
        command = [ffmpeg or find_ffmpeg(), "-v", "error", "-y", "-i", str(wav),
                   "-codec:a", "libmp3lame", "-q:a", "3", str(output)]
        result = subprocess.run(command, capture_output=True)
    if result.returncode != 0 or not output.is_file():
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg mp3 export failed: {detail}")
    return output


def render_voiceover(
    source: VoiceoverInput,
    voice_name: str,
    output: Path,
    *,
    speed: float = 1.0,
    timed: bool = False,
    models_dir: Path = MODELS_DIR,
    on_progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
) -> Path:
    if timed and source.cues:
        audio, rate, _ = synthesize_timed(source.cues, voice_name, speed=speed, models_dir=models_dir,
                                          on_progress=on_progress, cancel=cancel)
    else:
        audio, rate = synthesize_plain(source.text, voice_name, speed=speed, models_dir=models_dir,
                                       on_progress=on_progress, cancel=cancel)
    return export_audio(audio, rate, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline text-to-speech with a Piper voice.")
    parser.add_argument("input", nargs="?", help=".txt or .srt file (omit and use --text)")
    parser.add_argument("--text", help="text to speak")
    parser.add_argument("-o", "--output", required=True, help="output .wav or .mp3")
    parser.add_argument("--lang", default="ru", help="language code (picks the default voice)")
    parser.add_argument("--voice", help="Piper voice id, e.g. en_US-lessac-medium")
    parser.add_argument("--speed", type=float, default=1.0, help=f"{MIN_SPEED}..{MAX_SPEED}")
    parser.add_argument("--timed", action="store_true", help="for .srt: start each line at its cue time")
    parser.add_argument("--list-voices", action="store_true", help="print installed voices and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--list-voices" in argv:
        for name in installed_voices():
            print(f"{name}\t{voice_label(name)}")
        return 0
    args = build_parser().parse_args(argv)
    if args.text:
        source = VoiceoverInput(text=args.text)
    elif args.input:
        source = read_input(Path(args.input))
    else:
        print("give an input file or --text", file=sys.stderr)
        return 2
    voice = args.voice or DEFAULT_VOICES.get(args.lang)
    if not voice:
        print(f"no default Piper voice for '{args.lang}', pass --voice", file=sys.stderr)
        return 2
    out = render_voiceover(source, voice, Path(args.output), speed=args.speed, timed=args.timed)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
