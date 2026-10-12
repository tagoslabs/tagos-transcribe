# Tagos Transcribe

**by Tagos Labs**

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/tagoslabs)

Offline transcription, translation and voice-over dubbing for local audio and video files.
Nothing is sent to the cloud: models are downloaded once and then everything runs on your machine.

## What it does

- **Offline transcription** of audio and video (faster-whisper: `small`, `medium`, `large-v3`), CPU or NVIDIA GPU.
- **Timestamps / SRT**: plain text, timestamped text, or SubRip `.srt` subtitles.
- **Offline translation**: Argos Translate for English→Russian, NLLB-200 for other pairs.
- **Review / editing**: stop after transcription or translation, edit the segments, then continue.
- **Voice-over dubbing**: a Piper TTS voice is mixed over the original audio, aligned to the timecodes.

Video input is any local file. The app has no built-in downloading.

## Install (Linux)

```bash
./setup.sh      # downloads a bundled Python runtime, creates venv/, installs requirements.lock, fetches the small model
./install.sh    # optional: adds "Tagos Transcribe" to the app menu and a `transcribe` command
./uninstall.sh  # removes this folder and its menu shortcut (asks first)
```

Requires `curl`, `tar` and `ffmpeg`. If you move the folder, run `./install.sh` again.

## Usage

```bash
./run_gui.sh                                   # desktop GUI (transcribe + dub tabs)
./run.sh video.mp4                             # transcribe to stdout
./run.sh -m medium -l en --save-srt out/ talk.mp4
./run.sh --timestamps --save-timed-text out.txt audio.wav
./dub.sh video.mp4 --lang en --target ru -o out/   # dub into Russian
./dub.sh video.mp4 --stop-after-translate          # review/edit segments, then rerun with --segments
./dub.sh --help
```

### Dubbing defaults

| Setting | Default | Flag |
|---|---|---|
| Max phrase speed-up | 1.6x | `--max-speed` |
| Original audio under voice-over | 20% | `--duck 0.2` |
| Original audio kept as 2nd track | yes | `--no-original-track` to drop |
| Subtitles burned/embedded | no | `--embed-subs` to embed the SRT |
| Target language / voice | `ru` / `ru_RU-dmitri-medium` | `--target`, `--voice` |

## License

Only the source code in this repository is MIT licensed (see `LICENSE`). **Not everything the app uses is MIT:**

- the `piper-tts` Python dependency (TTS engine) is **GPL-3.0-or-later**;
- the NLLB-200 translation model is **CC-BY-NC 4.0 (non-commercial)**;
- other models have their own licenses (table below).

Models are **not** included in this repository; they are downloaded on first use into `models/`. If you redistribute a bundled build, you must comply with the licenses of every dependency and model it contains.

### Models and their licenses

| Component | Used for | License |
|---|---|---|
| OpenAI Whisper via faster-whisper (Systran CTranslate2 conversions) | transcription | MIT |
| **NLLB-200 distilled 600M** (Meta; CT2 int8 conversion `JustFrederik/nllb-200-distilled-600M-ct2-int8`) | translation for all pairs except en→ru | **CC-BY-NC 4.0 — non-commercial use only** |
| Argos Translate `translate-en_ru` 1.9 | en→ru translation (default for that pair) | **no explicit license** in the package (README lists OPUS training corpora only; [argos-translate#507](https://github.com/argosopentech/argos-translate/issues/507)). Argos states its own model work is MIT/CC0 ([#76](https://github.com/argosopentech/argos-translate/issues/76)); individual corpora may carry their own terms |
| Piper voice `ru_RU-dmitri-medium` (default) | TTS voice-over | dataset CC0, but **fine-tuned from `en_US-lessac`** ([MODEL_CARD](https://huggingface.co/rhasspy/piper-voices/blob/main/ru/ru_RU/dmitri/medium/MODEL_CARD)) |
| Piper base voice `en_US-lessac` | base of the Dmitri voice | Blizzard 2013 Lessac data: **research-only licence, commercial use excluded** ([MODEL_CARD](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/lessac/medium/MODEL_CARD), [licence](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html)) |
| Other Piper voices | `--voice` | per voice — see each voice's `MODEL_CARD` |
| piper-tts (Python package) | TTS engine | GPL-3.0-or-later |

> **Non-commercial notice:** translations produced with NLLB-200 are subject to CC-BY-NC 4.0. Do not use the NLLB path commercially. For English→Russian the app uses Argos by default (`--translator auto`).
>
> The default Russian voice (`ru_RU-dmitri-medium`) is derived from `en_US-lessac`, whose training data is licensed for research only. For commercial use, choose a Piper voice whose MODEL_CARD lists a permissive dataset license and that is not fine-tuned from lessac.

## Responsible use

Users are responsible for complying with recording-consent, privacy and copyright laws in their jurisdiction when transcribing, translating or dubbing any material.

## Development

```bash
venv/bin/python -m unittest discover -s tests
```

## Links

- Website: https://tagoslabs.com
- Reddit: https://www.reddit.com/user/tagoslabs
- Support: Buy Me a Coffee — https://buymeacoffee.com/tagoslabs

---

## По-русски

**Tagos Transcribe** от Tagos Labs — офлайн-транскрибация аудио и видео, таймкоды и SRT, офлайн-перевод (Argos для en→ru, NLLB-200 для остальных пар), правка сегментов и закадровая озвучка голосом Piper.

```bash
./setup.sh && ./run_gui.sh        # GUI
./run.sh video.mp4                # текст в терминал
./dub.sh video.mp4 --target ru    # озвучка
```

Под MIT распространяется только исходный код. Зависимость piper-tts — GPL-3.0, модель NLLB-200 — CC-BY-NC 4.0 (только некоммерческое использование). Пользователь сам отвечает за соблюдение законов о согласии на запись и авторском праве.
