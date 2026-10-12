"""One timed pass per device and model, then estimates for real media.

The pass follows the same steps as a real file: extract the soundtrack
from a short video, then transcribe it to the end. Silence would be
skipped by the voice detector and the timing would look instant.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_SECONDS = 30.0
SAMPLE_RATE = 16000
BENCHMARK_FILE = Path(__file__).resolve().parent / "benchmarks.json"
SAMPLE_WAV = Path(__file__).resolve().parent / "benchmark-sample.wav"
SAMPLE_VIDEO = Path(__file__).resolve().parent / "benchmark-sample.mp4"


class BenchmarkCancelled(Exception):
    pass


def profile_key(model: str, device: str, compute: str, threads: int = 1) -> str:
    # "3" is startup plus seconds of work per second of audio, taken from a
    # real file. Older keys were a short synthetic clip and are left unused.
    return f"3|{model}|{device}|{compute}|{max(int(threads), 1)}"


def _load() -> dict:
    try:
        raw = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _save(data: dict) -> None:
    try:
        BENCHMARK_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _factors(
    model: str, device: str, compute: str, threads: int = 1
) -> tuple[float, float] | None:
    """Fixed startup seconds, then wall seconds per second of audio."""
    row = _load().get(profile_key(model, device, compute, threads))
    if not isinstance(row, dict):
        return None
    rate = row.get("wall_per_audio")
    if not isinstance(rate, (int, float)) or rate <= 0:
        return None
    startup = row.get("startup", 0.0)
    if not isinstance(startup, (int, float)) or startup < 0:
        startup = 0.0
    return float(startup), float(rate)


def rate_for(model: str, device: str, compute: str, threads: int = 1) -> float | None:
    """Wall-clock seconds spent per second of audio. Missing until measured."""
    factors = _factors(model, device, compute, threads)
    if factors is None:
        return None
    return factors[1]


def estimate_seconds(
    audio_seconds: float,
    model: str,
    device: str,
    compute: str,
    threads: int = 1,
) -> float | None:
    factors = _factors(model, device, compute, threads)
    if factors is None or audio_seconds <= 0:
        return None
    startup, rate = factors
    return startup + rate * float(audio_seconds)


def remember(
    model: str,
    device: str,
    compute: str,
    audio_seconds: float,
    wall_seconds: float,
    threads: int = 1,
    startup: float = 0.0,
) -> None:
    if audio_seconds < 1 or wall_seconds < 0.2:
        return
    data = _load()
    key = profile_key(model, device, compute, threads)
    previous = data.get(key)
    if isinstance(previous, dict) and previous.get("wall_per_audio"):
        return
    if startup < 0 or startup >= wall_seconds:
        startup = 0.0
    data[key] = {
        "startup": float(startup),
        "wall_per_audio": (wall_seconds - startup) / audio_seconds,
    }
    _save(data)


def synthesize_speech(seconds: float) -> np.ndarray:
    """Voiced syllables. The voice detector treats this as speech, unlike silence."""
    seconds = max(float(seconds), 0.2)
    count = int(SAMPLE_RATE * seconds)
    timeline = np.arange(count) / SAMPLE_RATE
    syllable = np.clip(0.5 * (1 + np.sin(2 * np.pi * 4.2 * timeline - np.pi / 2)), 0, 1) ** 1.6
    fundamental = 140 + 25 * np.sin(2 * np.pi * 0.7 * timeline)
    phase = np.cumsum(fundamental) / SAMPLE_RATE * 2 * np.pi
    excitation = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    source = excitation * syllable

    def resonate(frequency: float, bandwidth: float, amplitude: float) -> np.ndarray:
        formed = np.zeros(count)
        omega = 2 * np.pi * frequency / SAMPLE_RATE
        radius = np.exp(-np.pi * bandwidth / SAMPLE_RATE)
        first = 2 * radius * np.cos(omega)
        second = -(radius * radius)
        for index in range(2, count):
            formed[index] = source[index] + first * formed[index - 1] + second * formed[index - 2]
        peak = float(np.max(np.abs(formed)) or 1.0)
        return amplitude * formed / peak

    audio = resonate(700, 80, 1.0) + resonate(1200, 100, 0.6) + resonate(2500, 150, 0.3)
    audio = audio / (float(np.max(np.abs(audio))) or 1.0) * 0.6
    return np.ascontiguousarray(audio, dtype=np.float32)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    samples = np.clip(audio, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0


def speech_sample(seconds: float = SAMPLE_SECONDS) -> np.ndarray:
    if seconds == SAMPLE_SECONDS and SAMPLE_WAV.is_file():
        cached = _read_wav(SAMPLE_WAV)
        if len(cached) >= int(SAMPLE_RATE * SAMPLE_SECONDS * 0.9):
            return cached
    audio = synthesize_speech(seconds)
    if seconds == SAMPLE_SECONDS:
        try:
            _write_wav(SAMPLE_WAV, audio)
        except OSError:
            pass
    return audio


def ensure_sample_video(ffmpeg: str) -> Path:
    """A 30-second picture with the speech on the soundtrack, cached after the first build."""
    if SAMPLE_VIDEO.is_file() and SAMPLE_VIDEO.stat().st_size > 1000:
        return SAMPLE_VIDEO
    speech_sample(SAMPLE_SECONDS)
    command = [
        ffmpeg,
        "-y",
        "-v", "error",
        "-f", "lavfi",
        "-i", "color=c=black:s=1280x720:r=30",
        "-i", str(SAMPLE_WAV),
        "-t", f"{SAMPLE_SECONDS:.3f}",
        "-shortest",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-b:a", "128k",
        str(SAMPLE_VIDEO),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not SAMPLE_VIDEO.is_file():
        detail = result.stderr.strip() or "could not build the sample video"
        raise RuntimeError(detail)
    return SAMPLE_VIDEO


def measure(model, cancel=None) -> tuple[float, float]:
    """Time extract plus transcription, the same steps as a real file.

    Returns audio seconds and wall seconds.
    """
    import time

    from transcribe import extract_audio, find_ffmpeg, waveform_from_wav

    source = ensure_sample_video(find_ffmpeg())
    if cancel and cancel():
        raise BenchmarkCancelled()
    started = time.monotonic()
    wav_path = extract_audio(source, find_ffmpeg())
    try:
        if cancel and cancel():
            raise BenchmarkCancelled()
        try:
            segments, info = model.transcribe(str(wav_path), language="en", vad_filter=True)
        except TypeError as exc:
            if "metadata_errors" not in str(exc):
                raise
            segments, info = model.transcribe(
                waveform_from_wav(wav_path),
                language="en",
                vad_filter=True,
            )
        for _segment in segments:
            if cancel and cancel():
                raise BenchmarkCancelled()
        audio_seconds = float(info.duration or 0.0) or SAMPLE_SECONDS
    finally:
        shutil.rmtree(wav_path.parent, ignore_errors=True)
    wall = time.monotonic() - started
    if wall < 0.2:
        raise RuntimeError("benchmark finished too quickly to trust")
    return audio_seconds, wall
