#!/usr/bin/env python3
"""Offline dubbing: timecodes, translation, review file, Piper voice-over.

The timing fit matches the working prototype:

* each phrase is placed at its start time;
* it may use the pause before the next phrase;
* if it is still too long, Piper ``length_scale`` goes down to 0.8 first;
* the rest is ffmpeg ``atempo``, and the total speedup is capped (default 1.6x);
* the original audio is ducked (default 20%) under the new voice;
* the dubbed mix is the default audio track and the original is kept as track 2;
* subtitles are not embedded unless asked.

faster-whisper and the translator default to 4 CPU threads. 8 threads produced
corrupted CTranslate2 output on one 8-core machine. Audio is decoded with
ffmpeg into a numpy array because opening the file inside faster-whisper
raised ``av.open`` in that environment.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from timestamps import (
    dump_segments,
    load_segments,
    segments_payload,
    segments_to_srt,
    segments_to_timed_text,
)
from transcribe import (
    AVAILABLE_MODELS,
    MODELS_DIR,
    find_ffmpeg,
    find_ffprobe,
    get_audio_duration,
    load_model,
    normalize_model_name,
    resolve_compute_type,
    resolve_device,
)

HF = "https://huggingface.co"
NLLB_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
NLLB_FILES = (
    "config.json",
    "model.bin",
    "sentencepiece.bpe.model",
    "shared_vocabulary.txt",
    "special_tokens_map.json",
)
ARGOS_URL = "https://argos-net.com/v1/translate-en_ru-1_9.argosmodel"
NLLB_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "pl": "pol_Latn",
    "uk": "ukr_Cyrl",
    "tr": "tur_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "nl": "nld_Latn",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "sv": "swe_Latn",
    "fi": "fin_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "ru": "rus_Cyrl",
    "el": "ell_Grek",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "da": "dan_Latn",
    "no": "nob_Latn",
    "nb": "nob_Latn",
    "id": "ind_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "bg": "bul_Cyrl",
    "sk": "slk_Latn",
    "hr": "hrv_Latn",
    "sr": "srp_Cyrl",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "et": "est_Latn",
    "fa": "pes_Arab",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "ms": "zsm_Latn",
}
ISO639_2 = {
    "ru": "rus",
    "en": "eng",
    "es": "spa",
    "de": "deu",
    "fr": "fra",
    "it": "ita",
    "pt": "por",
    "pl": "pol",
    "uk": "ukr",
    "zh": "zho",
    "ja": "jpn",
    "ko": "kor",
    "tr": "tur",
    "nl": "nld",
    "ar": "ara",
    "cs": "ces",
    "sv": "swe",
    "fi": "fin",
    "hi": "hin",
    "he": "heb",
    "ca": "cat",
}

# Piper voices that exist in rhasspy/piper-voices. The id is the filename stem.
# Russian is first because it is the default dubbing target.
PIPER_VOICES: dict[str, dict[str, str]] = {
    "ru_RU-dmitri-medium": {"lang": "ru", "label": "Дмитрий (ru, medium)"},
    "ru_RU-ruslan-medium": {"lang": "ru", "label": "Руслан (ru, medium)"},
    "ru_RU-irina-medium": {"lang": "ru", "label": "Ирина (ru, medium)"},
    "ru_RU-denis-medium": {"lang": "ru", "label": "Денис (ru, medium)"},
    "en_US-lessac-medium": {"lang": "en", "label": "Lessac (en_US, medium)"},
    "en_GB-alan-medium": {"lang": "en", "label": "Alan (en_GB, medium)"},
    "es_ES-davefx-medium": {"lang": "es", "label": "Davefx (es_ES, medium)"},
    "de_DE-thorsten-medium": {"lang": "de", "label": "Thorsten (de, medium)"},
    "fr_FR-siwis-medium": {"lang": "fr", "label": "Siwis (fr, medium)"},
    "uk_UA-ukrainian_tts-medium": {"lang": "uk", "label": "Ukrainian TTS (uk, medium)"},
    "it_IT-riccardo-x_low": {"lang": "it", "label": "Riccardo (it, x_low)"},
    "pl_PL-gosia-medium": {"lang": "pl", "label": "Gosia (pl, medium)"},
    "pt_BR-faber-medium": {"lang": "pt", "label": "Faber (pt_BR, medium)"},
    "zh_CN-huayan-medium": {"lang": "zh", "label": "Huayan (zh_CN, medium)"},
    "tr_TR-dfki-medium": {"lang": "tr", "label": "DFKI (tr, medium)"},
    "nl_NL-mls-medium": {"lang": "nl", "label": "MLS (nl, medium)"},
    "cs_CZ-jirka-medium": {"lang": "cs", "label": "Jirka (cs, medium)"},
    "ar_JO-kareem-medium": {"lang": "ar", "label": "Kareem (ar, medium)"},
    "sv_SE-nst-medium": {"lang": "sv", "label": "NST (sv, medium)"},
    "fi_FI-harri-medium": {"lang": "fi", "label": "Harri (fi, medium)"},
    "ca_ES-upc_ona-medium": {"lang": "ca", "label": "Ona (ca, medium)"},
}

DEFAULT_VOICES = {
    "ru": "ru_RU-dmitri-medium",
    "en": "en_US-lessac-medium",
    "es": "es_ES-davefx-medium",
    "de": "de_DE-thorsten-medium",
    "fr": "fr_FR-siwis-medium",
    "uk": "uk_UA-ukrainian_tts-medium",
    "it": "it_IT-riccardo-x_low",
    "pl": "pl_PL-gosia-medium",
    "pt": "pt_BR-faber-medium",
    "zh": "zh_CN-huayan-medium",
    "tr": "tr_TR-dfki-medium",
    "nl": "nl_NL-mls-medium",
    "cs": "cs_CZ-jirka-medium",
    "ar": "ar_JO-kareem-medium",
    "sv": "sv_SE-nst-medium",
    "fi": "fi_FI-harri-medium",
    "ca": "ca_ES-upc_ona-medium",
}

TARGET_LANGUAGES = (
    ("ru", "Русский"),
    ("en", "English"),
    ("es", "Español"),
    ("de", "Deutsch"),
    ("fr", "Français"),
    ("it", "Italiano"),
    ("pt", "Português"),
    ("pl", "Polski"),
    ("uk", "Українська"),
    ("zh", "中文"),
    ("tr", "Türkçe"),
    ("nl", "Nederlands"),
    ("cs", "Čeština"),
    ("ar", "العربية"),
    ("ja", "日本語"),
    ("ko", "한국어"),
    ("hi", "हिन्दी"),
    ("sv", "Svenska"),
    ("fi", "Suomi"),
    ("he", "עברית"),
    ("ca", "Català"),
)

DEFAULT_DUCK = 0.2
DEFAULT_MAX_SPEED = 1.6
MIN_LENGTH_SCALE = 0.8
DEFAULT_VOICE = "ru_RU-dmitri-medium"
DEFAULT_TARGET = "ru"
# 8 threads corrupted faster-whisper and CTranslate2 on an 8-core machine; 4 was fine there.
# On the 4-core CPU used to test this app, 2 and 4 threads were nondeterministic and
# sometimes wrote garbage tokens; 1 thread repeated the same clean transcript.
# 4 stays available (--threads / the GUI field) for machines where it is stable.
TRANSLATOR_THREAD_CAP = 4


def default_thread_count() -> int:
    return 1


DEFAULT_THREADS = default_thread_count()

ProgressCallback = Callable[[str, float, str], None]
CancelCallback = Callable[[], None]


class DubCancelled(Exception):
    """Raised when the user cancels between stages or segments."""


@dataclass
class DubOptions:
    model: str = "small"
    language: str | None = None
    target_language: str = DEFAULT_TARGET
    translator: str = "auto"
    voice: str = DEFAULT_VOICE
    duck: float = DEFAULT_DUCK
    max_speed: float = DEFAULT_MAX_SPEED
    threads: int = DEFAULT_THREADS
    embed_subs: bool = False
    keep_original_track: bool = True
    device: str = "auto"
    compute_type: str = "auto"
    models_dir: Path = field(default_factory=lambda: MODELS_DIR)


@dataclass
class TimedTranscript:
    source: Path
    language: str | None
    duration: float
    segments: list[dict]


def check_cancel(cancel: CancelCallback | None) -> None:
    if cancel is not None and cancel():
        raise DubCancelled()


def nllb_code(language: str) -> str:
    if language in NLLB_CODES.values():
        return language
    try:
        return NLLB_CODES[language]
    except KeyError as exc:
        supported = ", ".join(sorted(NLLB_CODES))
        raise ValueError(
            f"no NLLB code for language '{language}'. Supported: {supported}"
        ) from exc


def select_translator(requested: str, source_language: str, target_language: str) -> str:
    if requested not in {"auto", "nllb", "argos"}:
        raise ValueError("translator must be auto, nllb, or argos")
    if requested == "auto":
        if source_language == "en" and target_language == "ru":
            return "argos"
        return "nllb"
    if requested == "argos" and not (source_language == "en" and target_language == "ru"):
        raise ValueError(
            "Argos is only bundled for English -> Russian. "
            "Use translator 'nllb' or 'auto' for other pairs."
        )
    return requested


def default_voice_for(language: str) -> str:
    return DEFAULT_VOICES.get(language, DEFAULT_VOICE)


def iso639_2(language: str | None) -> str:
    if not language:
        return "und"
    return ISO639_2.get(language, language)


def download(
    url: str,
    destination: Path,
    *,
    on_progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
) -> Path:
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if on_progress:
        on_progress("download", 0.0, f"download {destination.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "transcribe-dub/1.0"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            read = 0
            while True:
                check_cancel(cancel)
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                read += len(chunk)
                if on_progress and total:
                    on_progress("download", min(read / total, 0.99), destination.name)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if not partial.is_file() or partial.stat().st_size <= 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"empty download: {url}")
    os.replace(partial, destination)
    if on_progress:
        on_progress("download", 1.0, destination.name)
    return destination


def load_audio_array(path: Path, ffmpeg: str, sample_rate: int = 16000) -> np.ndarray:
    """Decode with ffmpeg. Avoids faster-whisper's av.open file loader."""
    command = [
        ffmpeg,
        "-v", "error",
        "-i", str(path),
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {path.name}: {detail}")
    audio = np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0
    return np.ascontiguousarray(audio)


def probe_has_stream(path: Path, ffprobe: str, kind: str) -> bool:
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", kind,
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def resegment(segments: list[dict], max_dur: float = 8.0, gap: float = 0.8) -> list[dict]:
    """Rebuild units from word timestamps.

    Split at sentence ends, at long pauses, or — when a line is already long
    and has no punctuation, which ``medium`` sometimes returns — at a short pause.
    """
    words = [word for segment in segments for word in segment.get("words") or []]
    if not words:
        return segments
    output: list[dict] = []
    current: list[tuple] = []

    def flush() -> None:
        if not current:
            return
        output.append(
            {
                "start": current[0][0],
                "end": current[-1][1],
                "text": "".join(word[2] for word in current).strip(),
                "words": list(current),
            }
        )
        current.clear()

    for word in words:
        long_line = bool(current) and word[1] - current[0][0] > max_dur
        if current and (
            word[0] - current[-1][1] > gap
            or word[1] - current[0][0] > max_dur + 3
            or (long_line and word[0] - current[-1][1] > 0.25)
        ):
            flush()
        current.append(word)
        token = word[2].strip()
        duration = word[1] - current[0][0]
        if re.search(r"[.!?…]['\"»)]?$", token) or (
            duration > max_dur and token.endswith((",", ";", ":"))
        ):
            flush()
    flush()
    return output


def transcribe_timed(
    source: Path,
    options: DubOptions,
    *,
    on_progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
    ffmpeg: str | None = None,
) -> TimedTranscript:
    check_cancel(cancel)
    ffmpeg = ffmpeg or find_ffmpeg()
    ffprobe = find_ffprobe()
    duration = get_audio_duration(source, ffprobe)
    if on_progress:
        on_progress("extract", 0.0, source.name)
    audio = load_audio_array(source, ffmpeg)
    if duration <= 0:
        duration = len(audio) / 16000.0
    check_cancel(cancel)
    if on_progress:
        on_progress("transcribe", 0.0, source.name)

    device = resolve_device(options.device)
    compute_type = resolve_compute_type(options.compute_type, device)
    model_name = normalize_model_name(options.model)
    threads = max(int(options.threads), 1)
    model = load_model(
        model_name,
        device,
        compute_type,
        quiet=True,
        cpu_threads=threads,
    )
    try:
        language = options.language or None
        if on_progress:
            on_progress("transcribe", 0.0, "clock")
        segments_iter, info = model.transcribe(
            audio,
            language=language,
            vad_filter=True,
            word_timestamps=True,
            beam_size=5,
        )
        detected = info.language or language
        info_duration = float(info.duration or 0.0)
        if info_duration > 0:
            duration = info_duration
        raw: list[dict] = []
        for segment in segments_iter:
            check_cancel(cancel)
            words = [item for item in (segment.words or []) if item.word and item.word.strip()]
            start = words[0].start if words else segment.start
            end = words[-1].end if words else segment.end
            raw.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "text": (segment.text or "").strip(),
                    "words": [(float(item.start), float(item.end), item.word) for item in words],
                }
            )
            if on_progress and duration > 0:
                on_progress(
                    "transcribe",
                    min(max(float(end) / duration, 0.0), 0.99),
                    source.name,
                )
    finally:
        del model
        import gc
        gc.collect()

    segments = resegment(raw)
    for segment in segments:
        segment.pop("words", None)
        segment["translation"] = ""
    if on_progress:
        on_progress("transcribe", 1.0, detected or "")
    return TimedTranscript(
        source=source,
        language=detected,
        duration=float(duration),
        segments=segments,
    )


class NLLB:
    def __init__(
        self,
        models_dir: Path,
        source_language: str,
        target_language: str,
        threads: int,
        *,
        on_progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> None:
        import ctranslate2
        import sentencepiece as spm

        directory = Path(models_dir) / "nllb"
        for name in NLLB_FILES:
            download(
                f"{HF}/{NLLB_REPO}/resolve/main/{name}",
                directory / name,
                on_progress=on_progress,
                cancel=cancel,
            )
        intra = max(1, min(int(threads), TRANSLATOR_THREAD_CAP))
        self.translator = ctranslate2.Translator(
            str(directory),
            device="cpu",
            compute_type="int8",
            intra_threads=intra,
        )
        self.sp = spm.SentencePieceProcessor(model_file=str(directory / "sentencepiece.bpe.model"))
        self.source = nllb_code(source_language)
        self.target = nllb_code(target_language)
        self.name = "NLLB-200 distilled 600M (CTranslate2 int8)"

    def __call__(self, texts: list[str], batch_size: int = 16) -> list[str]:
        if len(texts) > batch_size:
            merged: list[str] = []
            for offset in range(0, len(texts), batch_size):
                merged.extend(self(texts[offset:offset + batch_size], batch_size))
            return merged
        tokens = [
            [self.source] + self.sp.encode(text, out_type=str) + ["</s>"]
            for text in texts
        ]
        results = self.translator.translate_batch(
            tokens,
            target_prefix=[[self.target]] * len(tokens),
            beam_size=4,
            max_decoding_length=256,
            no_repeat_ngram_size=4,
        )
        decoded: list[str] = []
        for result in results:
            hypothesis = [token for token in result.hypotheses[0] if token != self.target]
            decoded.append(self.sp.decode(hypothesis))
        return decoded


class Argos:
    def __init__(
        self,
        models_dir: Path,
        source_language: str,
        target_language: str,
        threads: int,
        *,
        on_progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> None:
        import ctranslate2
        import sentencepiece as spm

        if source_language != "en" or target_language != "ru":
            raise ValueError("Argos has no bundled model except English -> Russian")
        models_dir = Path(models_dir)
        directory = models_dir / "translate-en_ru-1_9"
        if not directory.is_dir():
            archive = download(
                ARGOS_URL,
                models_dir / "en_ru.argosmodel",
                on_progress=on_progress,
                cancel=cancel,
            )
            with zipfile.ZipFile(archive) as package:
                package.extractall(models_dir)
            archive.unlink(missing_ok=True)
        if not directory.is_dir():
            raise RuntimeError(
                f"Argos package did not contain {directory.name}. "
                "Delete the partial download in models/ and try again."
            )
        intra = max(1, min(int(threads), TRANSLATOR_THREAD_CAP))
        self.translator = ctranslate2.Translator(
            str(directory / "model"),
            device="cpu",
            intra_threads=intra,
        )
        self.sp = spm.SentencePieceProcessor(model_file=str(directory / "sentencepiece.model"))
        self.name = "Argos Translate en->ru 1.9 (CTranslate2)"

    def __call__(self, texts: list[str], batch_size: int = 16) -> list[str]:
        if len(texts) > batch_size:
            merged: list[str] = []
            for offset in range(0, len(texts), batch_size):
                merged.extend(self(texts[offset:offset + batch_size], batch_size))
            return merged
        tokens = [self.sp.encode(text, out_type=str) for text in texts]
        results = self.translator.translate_batch(
            tokens,
            beam_size=4,
            max_decoding_length=256,
        )
        return [self.sp.decode(result.hypotheses[0]) for result in results]


def build_translator(
    options: DubOptions,
    source_language: str,
    *,
    on_progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
):
    kind = select_translator(options.translator, source_language, options.target_language)
    threads = max(int(options.threads), 1)
    if kind == "argos":
        return Argos(
            options.models_dir,
            source_language,
            options.target_language,
            threads,
            on_progress=on_progress,
            cancel=cancel,
        )
    return NLLB(
        options.models_dir,
        source_language,
        options.target_language,
        threads,
        on_progress=on_progress,
        cancel=cancel,
    )


def translate_segments(
    segments: list[dict],
    options: DubOptions,
    source_language: str,
    *,
    on_progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
) -> str:
    check_cancel(cancel)
    if not source_language:
        raise ValueError("source language is unknown; set it and transcribe again")
    if not segments:
        raise ValueError("no segments to translate")
    if on_progress:
        on_progress("translate", 0.0, source_language)
    translator = build_translator(
        options,
        source_language,
        on_progress=on_progress,
        cancel=cancel,
    )
    try:
        texts = [str(segment.get("text") or "") for segment in segments]
        # Translate in batches so a long video can report progress and cancel.
        batch_size = 16
        translated: list[str] = []
        for offset in range(0, len(texts), batch_size):
            check_cancel(cancel)
            translated.extend(translator(texts[offset:offset + batch_size], batch_size))
            if on_progress:
                on_progress(
                    "translate",
                    min((offset + batch_size) / max(len(texts), 1), 0.99),
                    translator.name,
                )
        if len(translated) != len(segments):
            raise RuntimeError("translator returned a different number of lines")
        for segment, line in zip(segments, translated):
            segment["translation"] = (line or "").strip()
    finally:
        del translator
    if on_progress:
        on_progress("translate", 1.0, "")
    return kind_name(options, source_language)


def kind_name(options: DubOptions, source_language: str) -> str:
    kind = select_translator(options.translator, source_language, options.target_language)
    if kind == "argos":
        return "Argos Translate en->ru 1.9 (CTranslate2)"
    return "NLLB-200 distilled 600M (CTranslate2 int8)"


def piper_voice_urls(voice_name: str) -> tuple[str, str]:
    parts = voice_name.split("-")
    if len(parts) < 3:
        raise ValueError(
            f"unexpected Piper voice id '{voice_name}'. "
            "Expected a name like ru_RU-dmitri-medium."
        )
    language = parts[0]
    quality = parts[-1]
    speaker = "-".join(parts[1:-1])
    family = language.split("_")[0].lower()
    base = (
        f"{HF}/rhasspy/piper-voices/resolve/main/"
        f"{family}/{language}/{speaker}/{quality}/{voice_name}"
    )
    return base + ".onnx", base + ".onnx.json"


def ensure_piper_voice(
    voice_name: str,
    models_dir: Path,
    *,
    on_progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
) -> Path:
    onnx_url, config_url = piper_voice_urls(voice_name)
    directory = Path(models_dir) / "piper"
    onnx = download(onnx_url, directory / f"{voice_name}.onnx", on_progress=on_progress, cancel=cancel)
    download(config_url, directory / f"{voice_name}.onnx.json", on_progress=on_progress, cancel=cancel)
    return onnx


def trim_silence(audio: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    nonzero = np.flatnonzero(np.abs(audio) > threshold)
    if nonzero.size == 0:
        return audio
    return audio[nonzero[0]:nonzero[-1] + 1]


def atempo_chain(factor: float) -> str:
    """ffmpeg atempo stages. A single stage when 0.5..2.0, which covers the 1.6x cap."""
    if factor <= 0:
        raise ValueError(f"atempo factor must be positive, got {factor}")
    stages: list[float] = []
    remaining = float(factor)
    while remaining > 2.0 + 1e-6:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5 - 1e-6:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return ",".join(f"atempo={stage:.4f}" for stage in stages)


def apply_atempo(
    audio: np.ndarray,
    sample_rate: int,
    factor: float,
    tmp_dir: Path,
    ffmpeg: str | None = None,
) -> np.ndarray:
    ffmpeg = ffmpeg or find_ffmpeg()
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    source = tmp_dir / "a.f32"
    target = tmp_dir / "b.f32"
    np.ascontiguousarray(audio, dtype=np.float32).tofile(source)
    command = [
        ffmpeg,
        "-v", "error",
        "-y",
        "-f", "f32le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", str(source),
        "-filter:a", atempo_chain(factor),
        "-f", "f32le",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0 or not target.is_file():
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg atempo failed: {detail}")
    sped = np.fromfile(target, dtype=np.float32)
    return sped


def fit_clip_to_slot(
    audio: np.ndarray,
    sample_rate: int,
    slot: float,
    resynth,
    tmp_dir: Path,
    max_speed: float = DEFAULT_MAX_SPEED,
    *,
    atempo_fn: Callable[..., np.ndarray] = apply_atempo,
) -> tuple[np.ndarray, float]:
    """Fit one phrase into ``slot`` seconds.

    Order, matching the prototype: natural Piper audio, then Piper again with
    ``length_scale`` down to 0.8, then ffmpeg atempo. Total speed stays <= max_speed.
    """
    if max_speed < 1.0:
        raise ValueError("max speed must be at least 1.0")
    if slot <= 0:
        raise ValueError("slot must be positive")
    audio = trim_silence(audio)
    ratio = (len(audio) / sample_rate) / slot
    speed = 1.0
    if ratio > 1.0:
        length_scale = max(1.0 / ratio, MIN_LENGTH_SCALE)
        audio = trim_silence(resynth(length_scale))
        ratio2 = (len(audio) / sample_rate) / slot
        speed = 1.0 / length_scale
        if ratio2 > 1.0:
            atempo = min(ratio2, max_speed / speed)
            audio = atempo_fn(audio, sample_rate, atempo, tmp_dir)
            speed *= atempo
    return audio, speed


def segment_slot(
    index: int,
    segments: list[dict],
    cursor: float,
    total_duration: float,
) -> tuple[float, float]:
    """Start used for placement, and the slot up to the next phrase (the pause included)."""
    segment = segments[index]
    if index + 1 < len(segments):
        nxt = float(segments[index + 1]["start"])
    else:
        nxt = float(total_duration)
    start = max(float(segment["start"]), cursor)
    slot = max(0.3, min(nxt, float(total_duration)) - start)
    return start, slot


def _spoken_text(segment: dict) -> str:
    raw = segment.get("translation")
    if raw is None or raw == "":
        raw = segment.get("ru") or ""
    spoken = re.sub(r"\[[^\]]*\]", "", str(raw)).strip()
    return spoken


def synthesize_track(
    segments: list[dict],
    voice_name: str,
    models_dir: Path,
    total_duration: float,
    tmp_dir: Path,
    max_speed: float = DEFAULT_MAX_SPEED,
    *,
    on_progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
    ffmpeg: str | None = None,
) -> tuple[np.ndarray, int, list[dict]]:
    from piper import PiperVoice
    from piper import SynthesisConfig

    check_cancel(cancel)
    # Same filter as the prototype: skip blank lines and [неразборчиво], then
    # the slot runs up to the next line that is actually spoken.
    spoken_segments: list[dict] = []
    for segment in segments:
        spoken = _spoken_text(segment)
        if not spoken or spoken == "[неразборчиво]":
            continue
        spoken_segments.append(segment)
    if not spoken_segments:
        raise ValueError("no translated lines to speak")

    if on_progress:
        on_progress("tts", 0.0, voice_name)
    onnx = ensure_piper_voice(
        voice_name,
        models_dir,
        on_progress=on_progress,
        cancel=cancel,
    )
    voice = PiperVoice.load(str(onnx))
    sample_rate = int(voice.config.sample_rate)
    track = np.zeros(int((total_duration + 30.0) * sample_rate), np.float32)
    cursor = 0.0
    report: list[dict] = []
    ffmpeg = ffmpeg or find_ffmpeg()
    tmp_dir = Path(tmp_dir)

    def synth(text: str, length_scale: float) -> np.ndarray:
        wav_path = tmp_dir / "seg.wav"
        with wave.open(str(wav_path), "wb") as wav_file:
            voice.synthesize_wav(
                text,
                wav_file,
                syn_config=SynthesisConfig(length_scale=length_scale),
            )
        with wave.open(str(wav_path), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            channels = wav_file.getnchannels()
        samples = np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return np.ascontiguousarray(samples, dtype=np.float32)

    for index, segment in enumerate(spoken_segments):
        text = _spoken_text(segment)
        check_cancel(cancel)
        start, slot = segment_slot(index, spoken_segments, cursor, total_duration)
        natural = synth(text, 1.0)

        def resynth(length_scale: float, phrase: str = text) -> np.ndarray:
            return synth(phrase, length_scale)

        fitted, speed = fit_clip_to_slot(
            natural,
            sample_rate,
            slot,
            resynth,
            tmp_dir,
            max_speed,
            atempo_fn=lambda audio, sr, factor, folder: apply_atempo(
                audio, sr, factor, folder, ffmpeg
            ),
        )
        origin = int(start * sample_rate)
        end = origin + len(fitted)
        if end > len(track):
            extra = np.zeros(end - len(track) + sample_rate, np.float32)
            track = np.concatenate([track, extra])
        track[origin:end] += fitted
        cursor = start + len(fitted) / sample_rate
        if index + 1 < len(spoken_segments):
            nxt = float(spoken_segments[index + 1]["start"])
        else:
            nxt = total_duration
        overflow = cursor - min(nxt, total_duration)
        report.append(
            {
                "index": index,
                "start": float(segment["start"]),
                "placed": start,
                "slot": slot,
                "duration": len(fitted) / sample_rate,
                "speed": speed,
                "overflow": overflow,
                "text": text,
            }
        )
        if on_progress:
            on_progress("tts", (index + 1) / len(spoken_segments), text)

    track = track[: int(max(total_duration, cursor) * sample_rate)]
    peak = float(np.abs(track).max() or 1.0)
    track = track / peak * 0.95
    if on_progress:
        on_progress("tts", 1.0, voice_name)
    return track, sample_rate, report


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def mix_and_mux(
    source: Path,
    voice_wav: Path,
    output: Path,
    *,
    duck: float = DEFAULT_DUCK,
    keep_original_track: bool = True,
    embed_subs: bool = False,
    subtitles: Path | None = None,
    target_language: str = DEFAULT_TARGET,
    source_language: str | None = None,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    crf: int | None = None,
    maxrate: str | None = None,
    threads: int = DEFAULT_THREADS,
    audio_bitrates: str = "128k,96k",
) -> None:
    if duck < 0:
        raise ValueError("duck volume must be >= 0")
    ffmpeg = ffmpeg or find_ffmpeg()
    ffprobe = ffprobe or find_ffprobe()
    if not probe_has_stream(source, ffprobe, "a"):
        raise RuntimeError(f"{source.name} has no audio stream")
    has_video = probe_has_stream(source, ffprobe, "v")
    if embed_subs and (not subtitles or not subtitles.is_file()):
        raise ValueError("embed_subs requires a subtitle file")
    if embed_subs and not has_video:
        raise RuntimeError("cannot embed subtitles: the input has no video stream")

    rates = [item.strip() for item in audio_bitrates.split(",") if item.strip()]
    if not rates:
        rates = ["128k", "96k"]
    mix_rate = rates[0]
    original_rate = rates[-1]
    mix_language = iso639_2(target_language)
    original_language = iso639_2(source_language)
    title = "Russian voice-over" if target_language == "ru" else f"{target_language} voice-over"

    filter_complex = (
        f"[0:a:0]volume={duck}[bg];"
        "[1:a]aresample=48000,pan=stereo|c0=c0|c1=c0[vo];"
        "[bg][vo]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[mix]"
    )
    command = [
        ffmpeg,
        "-v", "error",
        "-y",
        "-i", str(source),
        "-i", str(voice_wav),
    ]
    if embed_subs and subtitles is not None:
        command.extend(["-i", str(subtitles)])
    command.extend(["-filter_complex", filter_complex])
    if has_video:
        command.extend(["-map", "0:v:0"])
    command.extend(["-map", "[mix]"])
    if keep_original_track:
        command.extend(["-map", "0:a:0"])
    if embed_subs and subtitles is not None:
        command.extend(
            [
                "-map", "2:s:0",
                "-c:s", "mov_text",
                "-metadata:s:s:0", f"language={mix_language}",
            ]
        )
    if crf is not None:
        command.extend(
            [
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", str(crf),
                "-threads", str(max(int(threads), 1)),
            ]
        )
        if maxrate:
            buf = str(2 * int(str(maxrate).rstrip("k"))) + "k"
            command.extend(["-maxrate", str(maxrate), "-bufsize", buf])
    elif has_video:
        command.extend(["-c:v", "copy"])
    command.extend(
        [
            "-c:a", "aac",
            "-b:a:0", mix_rate,
            "-metadata:s:a:0", f"language={mix_language}",
            "-metadata:s:a:0", f"title={title}",
            "-disposition:a:0", "default",
        ]
    )
    if keep_original_track:
        command.extend(
            [
                "-b:a:1", original_rate,
                "-metadata:s:a:1", f"language={original_language}",
                "-metadata:s:a:1", "title=Original",
                "-disposition:a:1", "0",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    command.extend(["-movflags", "+faststart", str(output)])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg mux failed")


def render_dub(
    source: Path,
    segments: list[dict],
    *,
    duration: float,
    options: DubOptions,
    output: Path,
    source_language: str | None,
    on_progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
    work_dir: Path | None = None,
    keep_work: bool = False,
    crf: int | None = None,
    maxrate: str | None = None,
    audio_bitrates: str = "128k,96k",
) -> list[dict]:
    check_cancel(cancel)
    if duration <= 0:
        duration = get_audio_duration(source, find_ffprobe())
    if duration <= 0:
        raise RuntimeError("could not read media duration")
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="dub-"))
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
    try:
        if on_progress:
            on_progress("tts", 0.0, options.voice)
        track, sample_rate, report = synthesize_track(
            segments,
            options.voice,
            options.models_dir,
            duration,
            work_dir,
            options.max_speed,
            on_progress=on_progress,
            cancel=cancel,
        )
        voice_wav = work_dir / "voice.wav"
        write_wav(voice_wav, track, sample_rate)
        del track
        subtitles = None
        if options.embed_subs:
            subtitles = work_dir / "subs.srt"
            subtitles.write_text(
                segments_to_srt(
                    segments,
                    text_key="translation",
                    total_duration=duration,
                    extend_cues=True,
                ),
                encoding="utf-8",
            )
        if on_progress:
            on_progress("mux", 0.2, output.name)
        mix_and_mux(
            source,
            voice_wav,
            output,
            duck=options.duck,
            keep_original_track=options.keep_original_track,
            embed_subs=options.embed_subs,
            subtitles=subtitles,
            target_language=options.target_language,
            source_language=source_language,
            crf=crf,
            maxrate=maxrate,
            threads=options.threads,
            audio_bitrates=audio_bitrates,
        )
        if on_progress:
            on_progress("mux", 1.0, str(output))
        return report
    finally:
        if not keep_work:
            shutil.rmtree(work_dir, ignore_errors=True)


def write_sidecar_files(
    outdir: Path,
    base: str,
    segments: list[dict],
    *,
    language: str | None,
    target_language: str,
    duration: float,
    source: Path,
    include_translation: bool,
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    language = language or "und"
    payload = segments_payload(
        segments,
        language=language,
        target_language=target_language,
        source=str(source),
    )
    # Keep the prototype key so an edited file can be resumed by either name.
    if target_language == "ru":
        for item, segment in zip(payload["segments"], segments):
            item["ru"] = segment.get("translation") or ""
    json_path = outdir / f"{base}.segments.json"
    dump_segments(json_path, payload)
    written.append(json_path)

    source_srt = outdir / f"{base}.{language}.srt"
    source_srt.write_text(
        segments_to_srt(segments, text_key="text", total_duration=duration, extend_cues=True),
        encoding="utf-8",
    )
    written.append(source_srt)
    source_txt = outdir / f"{base}.{language}.txt"
    source_txt.write_text(segments_to_timed_text(segments, text_key="text"), encoding="utf-8")
    written.append(source_txt)

    if include_translation:
        target_srt = outdir / f"{base}.{target_language}.srt"
        target_srt.write_text(
            segments_to_srt(
                segments,
                text_key="translation",
                total_duration=duration,
                extend_cues=True,
            ),
            encoding="utf-8",
        )
        written.append(target_srt)
        target_txt = outdir / f"{base}.{target_language}.txt"
        target_txt.write_text(
            segments_to_timed_text(segments, text_key="translation"),
            encoding="utf-8",
        )
        written.append(target_txt)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dub a video offline: transcribe with timecodes, translate, "
            "optionally pause for edits, then mix a Piper voice-over."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ./dub.sh video.mp4\n"
            "  ./dub.sh video.mp4 --stop-after-translate\n"
            "  ./dub.sh video.mp4 --segments video.segments.json\n"
            "  ./dub.sh video.mp4 --stop-after-transcribe\n"
            "  ./dub.sh video.mp4 --target en --voice en_US-lessac-medium --duck 0.3 --max-speed 1.4\n"
        ),
    )
    parser.add_argument("input", type=Path, help="Video or audio file")
    parser.add_argument("--lang", default="auto", help="Source language: auto, es, en, ru, ...")
    parser.add_argument(
        "--model",
        choices=list(AVAILABLE_MODELS),
        default="small",
        help="faster-whisper model (default: small)",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Translation language (default: ru)",
    )
    parser.add_argument(
        "--translator",
        default="auto",
        choices=["auto", "nllb", "argos"],
        help="auto = Argos for English->Russian, NLLB-200 otherwise",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help="Piper voice id (default: ru_RU-dmitri-medium)",
    )
    parser.add_argument(
        "--duck",
        type=float,
        default=DEFAULT_DUCK,
        help="Original audio volume under the voice-over, 0.2 = 20%% (default: 0.2)",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=DEFAULT_MAX_SPEED,
        help="Maximum total phrase speedup (default: 1.6)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=(
            "CPU threads (default: 1). "
            "4 was stable on one 8-core machine and faster; 2+ threads corrupted output on a 4-core CPU. "
            "Values above 4 are capped for translation."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory for Whisper, NLLB, Argos, and Piper files",
    )
    parser.add_argument(
        "--embed-subs",
        action="store_true",
        help="Embed the translated SRT in the video (default: off)",
    )
    parser.add_argument(
        "--no-original-track",
        action="store_true",
        help="Do not keep the original audio as a second track (default: keep it)",
    )
    parser.add_argument(
        "--stop-after-transcribe",
        action="store_true",
        help="Write timecoded source SRT, text, and segments JSON, then stop",
    )
    parser.add_argument(
        "--stop-after-translate",
        action="store_true",
        help="Write segments JSON and SRT so the translation can be edited, then stop",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=None,
        help="Resume from an edited segments JSON (skips transcription and translation)",
    )
    parser.add_argument("--keep-tmp", action="store_true", help="Keep the temporary voice wav")
    parser.add_argument("--crf", type=int, default=None, help="Re-encode video with libx264 at this CRF")
    parser.add_argument("--maxrate", default=None, help="Optional x264 maxrate used with --crf, e.g. 400k")
    parser.add_argument("--abr", default="128k,96k", help="Audio bitrates: mix, original track")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Whisper device (default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="Whisper compute type (default: auto)",
    )
    parser.add_argument("-o", "--outdir", type=Path, default=None, help="Output directory")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _options_from_args(args: argparse.Namespace) -> DubOptions:
    language = None if args.lang in {None, "", "auto"} else args.lang
    return DubOptions(
        model=args.model,
        language=language,
        target_language=args.target,
        translator=args.translator,
        voice=args.voice,
        duck=args.duck,
        max_speed=args.max_speed,
        threads=args.threads,
        embed_subs=bool(args.embed_subs),
        keep_original_track=not bool(args.no_original_track),
        device=args.device,
        compute_type=args.compute_type,
        models_dir=Path(args.models_dir),
    )


def _print_segments(segments: Iterable[dict], key: str) -> None:
    for segment in segments:
        text = str(segment.get(key) or "").replace("\n", " ")
        print(f"  {float(segment['start']):7.2f}-{float(segment['end']):7.2f} {text}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.input.expanduser().resolve()
    if not source.is_file():
        print(f"not found: {source}", file=sys.stderr)
        return 1
    if args.duck < 0:
        print("duck must be >= 0", file=sys.stderr)
        return 1
    if args.max_speed < 1.0:
        print("max-speed must be >= 1", file=sys.stderr)
        return 1
    if args.threads < 1:
        print("threads must be >= 1", file=sys.stderr)
        return 1
    cpu = os.cpu_count() or 1
    if args.threads > TRANSLATOR_THREAD_CAP or args.threads >= cpu:
        print(
            "warning: more than 4 threads, or one thread per CPU core, "
            "corrupted faster-whisper/CTranslate2 output on some machines. "
            f"Translation still uses at most {TRANSLATOR_THREAD_CAP}.",
            file=sys.stderr,
        )

    options = _options_from_args(args)
    outdir = Path(args.outdir) if args.outdir else source.parent
    base = source.stem
    started = time.time()
    last_download: dict[str, int | str] = {"name": "", "percent": -1}

    def log(stage: str, fraction: float, detail: str) -> None:
        if stage != "download":
            return
        percent = int(fraction * 100)
        if last_download["name"] == detail and last_download["percent"] == percent:
            return
        last_download["name"] = detail
        last_download["percent"] = percent
        print(f"[download] {percent}% {detail}", flush=True)

    try:
        if args.segments:
            loaded = load_segments(args.segments)
            segments = loaded["segments"]
            language = loaded.get("language") or options.language
            if loaded.get("target_language"):
                options.target_language = str(loaded["target_language"])
            duration = get_audio_duration(source, find_ffprobe())
            print(f"[resume] {len(segments)} segments from {args.segments}", flush=True)
        else:
            print("[transcribe] ...", flush=True)
            transcript = transcribe_timed(source, options, on_progress=log)
            language = transcript.language
            duration = transcript.duration
            segments = transcript.segments
            print(
                f"[transcribe] language={language} segments={len(segments)} "
                f"duration={duration:.1f}s",
                flush=True,
            )
            _print_segments(segments, "text")
            if args.stop_after_transcribe:
                written = write_sidecar_files(
                    outdir,
                    base,
                    segments,
                    language=language,
                    target_language=options.target_language,
                    duration=duration,
                    source=source,
                    include_translation=False,
                )
                for path in written:
                    print(f"Wrote {path}")
                return 0

            print(f"[translate] {language} -> {options.target_language}", flush=True)
            engine = translate_segments(segments, options, language or "", on_progress=log)
            print(f"[translate] {engine}", flush=True)
            _print_segments(segments, "translation")
            written = write_sidecar_files(
                outdir,
                base,
                segments,
                language=language,
                target_language=options.target_language,
                duration=duration,
                source=source,
                include_translation=True,
            )
            for path in written:
                print(f"Wrote {path}")
            if args.stop_after_translate:
                json_path = outdir / f"{base}.segments.json"
                print(
                    f"Stopped after translation. Edit 'translation' in {json_path}, "
                    f"then rerun with --segments {json_path}"
                )
                return 0

        output = outdir / f"{base}.{options.target_language}.mp4"
        print(
            f"[voice] {options.voice} duck={options.duck} max_speed={options.max_speed} "
            f"embed_subs={options.embed_subs} original_track={options.keep_original_track}",
            flush=True,
        )
        report = render_dub(
            source,
            segments,
            duration=duration,
            options=options,
            output=output,
            source_language=language,
            on_progress=log,
            work_dir=outdir / f".{base}.dubtmp",
            keep_work=bool(args.keep_tmp),
            crf=args.crf,
            maxrate=args.maxrate,
            audio_bitrates=args.abr,
        )
        for item in report:
            overflow = item["overflow"]
            line = (
                f"  seg{item['index']:02d} at {item['placed']:6.2f}s "
                f"slot {item['slot']:5.2f}s tts {item['duration']:5.2f}s "
                f"speed x{item['speed']:.2f}"
            )
            if overflow > 0.05:
                line += f"  OVERFLOW {overflow:.2f}s"
            print(line, flush=True)
        print(f"Wrote {output}")
        print(f"done in {time.time() - started:.1f}s")
    except DubCancelled:
        print("cancelled", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
