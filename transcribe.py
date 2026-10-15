#!/usr/bin/env python3
"""Portable audio/video transcription tool using faster-whisper."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
BIN_DIR = APP_DIR / "bin"
DEFAULT_MODEL = "small"
AVAILABLE_MODELS = {
    "small": {
        "label": "Small — быстро",
        "hint": "~460 MB, хорошо для чистых записей",
    },
    "medium": {
        "label": "Medium — точнее",
        "hint": "~1.5 GB, лучше на шумных записях",
    },
    "large-v3": {
        "label": "Large v3 — максимум",
        "hint": "~3 GB, лучшее качество, медленнее",
    },
}
SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".m4v",
}


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    source: Path
    language: str | None
    text: str
    segments: list[Segment]


def find_ffmpeg() -> str:
    local = BIN_DIR / "ffmpeg"
    if local.is_file() and local.stat().st_mode & 0o111:
        return str(local)
    system = shutil.which("ffmpeg")
    if system:
        return system
    raise FileNotFoundError(
        "ffmpeg not found. Run ./setup.sh or install ffmpeg system-wide."
    )


def find_ffprobe() -> str:
    local = BIN_DIR / "ffprobe"
    if local.is_file() and local.stat().st_mode & 0o111:
        return str(local)
    system = shutil.which("ffprobe")
    if system:
        return system
    ffmpeg = find_ffmpeg()
    sibling = Path(ffmpeg).with_name("ffprobe")
    if sibling.is_file() and sibling.stat().st_mode & 0o111:
        return str(sibling)
    raise FileNotFoundError(
        "ffprobe not found. Run ./setup.sh or install ffmpeg system-wide."
    )


def get_audio_duration(audio_path: Path, ffprobe: str) -> float:
    command = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return max(float(result.stdout.strip()), 0.0)
    except ValueError:
        return 0.0


def format_duration(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def is_media_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def collect_inputs(paths: Iterable[Path], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            if is_media_file(path):
                files.append(path.resolve())
            else:
                print(f"skip: unsupported file {path}", file=sys.stderr)
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            for item in sorted(iterator):
                if is_media_file(item):
                    files.append(item.resolve())
        else:
            print(f"skip: not found {path}", file=sys.stderr)
    return files


def extract_audio(source: Path, ffmpeg: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="transcribe-"))
    output = temp_dir / "audio.wav"
    command = [
        ffmpeg,
        "-y",
        "-i", str(source),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            f"ffmpeg failed for {source.name}:\n{result.stderr.strip()}"
        )
    return output


def normalize_model_name(model_name: str) -> str:
    if model_name not in AVAILABLE_MODELS:
        supported = ", ".join(AVAILABLE_MODELS)
        raise ValueError(f"unknown model '{model_name}'. supported: {supported}")
    return model_name


def model_info_label(model_name: str, device: str = "cpu", prefix: str = "Модель") -> str:
    model_name = normalize_model_name(model_name)
    resolved = resolve_device(device)
    compute = resolve_compute_type("auto", resolved)
    device_name = "GPU" if resolved == "cuda" else "CPU"
    return f"{prefix}: {model_name}  •  {device_name}  •  {compute}"


def load_model(
    model_name: str,
    device: str,
    compute_type: str,
    quiet: bool = False,
    cpu_threads: int = 1,
):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "faster-whisper is not installed. Run ./setup.sh first."
        ) from exc

    model_name = normalize_model_name(model_name)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        prepare_cuda_device()
    if not quiet:
        print(f"loading model '{model_name}' ({device}, {compute_type})...")
    # One CPU thread by default. On this class of CPU, 2+ int8 threads made
    # CTranslate2 nondeterministic and sometimes emitted garbage tokens.
    # Dubbing can pass a higher count; translation is still capped at 4.
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=max(int(cpu_threads), 1),
        download_root=str(MODELS_DIR),
    )


def waveform_from_wav(path: Path):
    """Read the 16-bit wav ffmpeg just wrote, without PyAV.

    faster-whisper opens files with ``av.open(..., metadata_errors="ignore")``.
    Some PyAV builds reject that argument. The samples are the same wav.
    """
    import numpy as np
    import wave

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        if handle.getsampwidth() != 2:
            raise RuntimeError(f"expected 16-bit audio, got {path.name}")
        frames = handle.readframes(handle.getnframes())
    audio = np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def transcribe_file(
    model,
    source: Path,
    ffmpeg: str,
    language: str | None,
    quiet: bool = False,
    on_progress: Callable[[float, float, float, str], None] | None = None,
) -> TranscriptionResult:
    if not quiet:
        print(f"\n=== {source.name} ===")

    if on_progress:
        on_progress(0.0, 0.0, 0.0, "extract")

    audio_path = extract_audio(source, ffmpeg)
    temp_dir = audio_path.parent

    try:
        if on_progress:
            on_progress(0.0, 0.0, 0.0, "transcribe")

        try:
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=language,
                vad_filter=True,
            )
        except TypeError as exc:
            if "metadata_errors" not in str(exc):
                raise
            segments_iter, info = model.transcribe(
                waveform_from_wav(audio_path),
                language=language,
                vad_filter=True,
            )

        duration = float(info.duration or 0.0)
        if duration <= 0:
            try:
                duration = get_audio_duration(audio_path, find_ffprobe())
            except FileNotFoundError:
                duration = 0.0

        if on_progress:
            on_progress(0.0, 0.0, duration, "transcribe")

        segments: list[Segment] = []
        last_reported = -1.0
        stop_timer = threading.Event()
        best_fraction = 0.0

        def time_estimator() -> None:
            nonlocal best_fraction
            start = time.monotonic()
            while not stop_timer.wait(0.2):
                if duration <= 0 or not on_progress:
                    continue
                elapsed = time.monotonic() - start
                estimated = min(elapsed / duration, 0.95)
                if estimated > best_fraction:
                    best_fraction = estimated
                    on_progress(estimated, 0.0, duration, "pulse")

        estimator_thread: threading.Thread | None = None
        if on_progress and duration > 0:
            estimator_thread = threading.Thread(target=time_estimator, daemon=True)
            estimator_thread.start()

        try:
            for seg in segments_iter:
                text = seg.text.strip()
                if text:
                    segments.append(
                        Segment(start=seg.start, end=seg.end, text=text)
                    )

                if on_progress:
                    if duration > 0:
                        fraction = min(max(seg.end / duration, 0.0), 0.99)
                        current_seconds = seg.end
                    else:
                        fraction = min(len(segments) / max(len(segments) + 2, 1), 0.95)
                        current_seconds = 0.0

                    fraction = max(fraction, best_fraction)
                    if fraction - last_reported >= 0.01 or fraction >= 0.99:
                        on_progress(fraction, current_seconds, duration, "transcribe")
                        last_reported = fraction
                        best_fraction = fraction
        finally:
            stop_timer.set()
            if estimator_thread is not None:
                estimator_thread.join(timeout=1)

        full_text = " ".join(segment.text for segment in segments)
        detected_language = info.language

        if on_progress:
            on_progress(1.0, duration, duration, "done")

        if not quiet:
            print(f"language: {detected_language}")
            print(full_text.strip())

        return TranscriptionResult(
            source=source,
            language=detected_language,
            text=full_text.strip(),
            segments=segments,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio and video files to text (offline).",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Files or directories to transcribe",
    )
    parser.add_argument(
        "-m", "--model",
        choices=list(AVAILABLE_MODELS),
        default=DEFAULT_MODEL,
        help="Whisper model (default: small). Others download on first use.",
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Language code, e.g. ru, en. Auto-detect if omitted.",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively scan directories",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="Compute type: auto, int8, float16, etc. (default: auto)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Print each segment with timecodes. Default: plain text only.",
    )
    parser.add_argument(
        "--save-srt",
        metavar="PATH",
        help="Write a SubRip .srt file or a directory of them. Opt-in; nothing is saved otherwise.",
    )
    parser.add_argument(
        "--save-timed-text",
        metavar="PATH",
        help="Write plain text with timestamps. Opt-in; nothing is saved otherwise.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_compute_type(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    # GTX 10xx and other older cards advertise CUDA but reject float16.
    preference = (
        ("float16", "int8_float16", "int8", "int8_float32", "float32")
        if device == "cuda"
        else ("int8", "int8_float32", "int16", "float32")
    )
    try:
        import ctranslate2

        supported = set(ctranslate2.get_supported_compute_types(device))
    except Exception:
        return "int8"
    for candidate in preference:
        if candidate in supported:
            return candidate
    return "int8"


# Newest cuBLAS 12 build each NVIDIA driver can load.
# Minimum driver versions are from the CUDA toolkit release notes.
# A newer library than the driver supports fails with "cannot be loaded".
_CUBLAS_BY_DRIVER = (
    ((575, 51), "12.9.2.10"),
    ((570, 26), "12.8.5.5"),
    ((560, 28), "12.6.4.1"),
    ((555, 42), "12.5.3.2"),
    ((550, 54), "12.4.5.8"),
)


def cublas_version_for_driver(version: tuple[int, int] | None) -> str | None:
    if version is None:
        return None
    for minimum, pin in _CUBLAS_BY_DRIVER:
        if version >= minimum:
            return pin
    return None


def nvidia_driver_version() -> tuple[int, int] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    line = output.strip().splitlines()
    if not line:
        return None
    major_text, _, rest = line[0].strip().partition(".")
    minor_text = rest.split(".", 1)[0]
    try:
        return int(major_text), int(minor_text or "0")
    except ValueError:
        return None


def _cublas_library_path() -> Path | None:
    try:
        from importlib.metadata import PackageNotFoundError, distribution
    except ImportError:
        return None
    try:
        dist = distribution("nvidia-cublas-cu12")
    except PackageNotFoundError:
        return None
    candidate = Path(str(dist.locate_file("nvidia/cublas/lib/libcublas.so.12")))
    if candidate.is_file():
        return candidate
    for file in dist.files or []:
        if Path(str(file)).name == "libcublas.so.12":
            path = Path(str(dist.locate_file(file)))
            if path.is_file():
                return path
    return None


def preload_cuda_libraries() -> bool:
    """Load the pip-installed cuBLAS before CTranslate2 calls dlopen("libcublas.so.12")."""
    path = _cublas_library_path()
    if path is None:
        return False
    import ctypes

    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    return True


def ensure_cuda_libraries() -> str | None:
    """Install the cuBLAS build this driver can load. None means no usable NVIDIA driver."""
    wanted = cublas_version_for_driver(nvidia_driver_version())
    if wanted is None:
        return None
    installed = None
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        version = None
        PackageNotFoundError = Exception
    if version is not None:
        try:
            installed = version("nvidia-cublas-cu12")
        except PackageNotFoundError:
            installed = None
    if installed != wanted:
        spec = f"nvidia-cublas-cu12=={wanted}"
        print(f"==> installing CUDA library {spec}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
    return wanted


def prepare_cuda_device() -> None:
    if ensure_cuda_libraries() is None:
        raise RuntimeError(
            "Видеокарта не запустится: нужен драйвер NVIDIA 550 или новее. Выбери процессор."
        )
    if not preload_cuda_libraries():
        raise RuntimeError(
            "Не удалось загрузить libcublas.so.12. Проверь интернет и запусти ./setup.sh."
        )


def _print_transcript(result: TranscriptionResult, timestamps: bool) -> None:
    from timestamps import segments_to_timed_text

    print(f"\n=== {result.source.name} ===")
    print(f"language: {result.language}")
    if timestamps:
        sys.stdout.write(segments_to_timed_text(result.segments))
    else:
        print(result.text.strip())


def _export_transcripts(args: argparse.Namespace, files: list[Path], result: TranscriptionResult) -> None:
    from timestamps import resolve_export_path, segments_to_subtitles, segments_to_timed_text

    multiple = len(files) > 1
    if args.save_srt:
        path = resolve_export_path(args.save_srt, result.source, ".srt", multiple=multiple)
        path.write_text(segments_to_subtitles(result.segments, text_key="text"), encoding="utf-8")
        print(f"saved: {path}", file=sys.stderr)
    if args.save_timed_text:
        path = resolve_export_path(
            args.save_timed_text,
            result.source,
            ".txt",
            multiple=multiple,
        )
        path.write_text(segments_to_timed_text(result.segments), encoding="utf-8")
        print(f"saved: {path}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    files = collect_inputs(args.inputs, args.recursive)
    if not files:
        print("no supported media files found", file=sys.stderr)
        return 1

    try:
        ffmpeg = find_ffmpeg()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    compute_type = resolve_compute_type(args.compute_type, device)
    model = load_model(args.model, device, compute_type)

    export = bool(args.timestamps or args.save_srt or args.save_timed_text)
    errors = 0
    for source in files:
        try:
            if export:
                result = transcribe_file(
                    model,
                    source,
                    ffmpeg,
                    args.language,
                    quiet=True,
                )
                _print_transcript(result, args.timestamps)
                _export_transcripts(args, files, result)
            else:
                transcribe_file(
                    model,
                    source,
                    ffmpeg,
                    args.language,
                )
        except Exception as exc:
            errors += 1
            print(f"error: {source.name}: {exc}", file=sys.stderr)

    print(f"\ndone: {len(files) - errors}/{len(files)} files")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
