"""Dubbing screen: transcribe, translate, edit, voice-over, export."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import flet as ft

import look
from dub import (
    DEFAULT_DUCK,
    DEFAULT_MAX_SPEED,
    DEFAULT_THREADS,
    DEFAULT_VOICE,
    DEFAULT_VOICES,
    PIPER_VOICES,
    TARGET_LANGUAGES,
    TRANSLATOR_THREAD_CAP,
    DubCancelled,
    DubOptions,
    render_dub,
    transcribe_timed,
    translate_segments,
)
from timestamps import (
    apply_srt_cues,
    dump_segments,
    format_srt_timestamp,
    load_segments,
    parse_srt,
    segments_payload,
    segments_to_subtitles,
    segments_to_timed_text,
)
from i18n import all_values, no_file_labels, t
from transcribe import (
    find_ffprobe,
    get_audio_duration,
    resolve_compute_type,
    resolve_device,
)


def stage_label(stage: str) -> str:
    return t(f"stage_{stage}")


def build_dub_view(
    page: ft.Page,
    *,
    get_device,
    get_model,
    get_language,
    on_busy,
    ui_hooks: dict | None = None,
    source_choices=None,
    initial_source: str = "auto",
    on_source_changed=None,
) -> ft.Column:
    file_picker = ft.FilePicker()
    state: dict = {
        "video": None,
        "language": None,
        "duration": 0.0,
        "segments": [],
        "done": False,
        "output_name": "",
    }
    translation_fields: list[ft.TextField] = []
    cancel_event = threading.Event()
    running = {"value": False}
    last_paint = {"time": 0.0, "stage": ""}

    def run_on_ui(callback) -> None:
        page.session.connection.loop.call_soon_threadsafe(callback)

    def show_message(text: str, error: bool = False) -> None:
        def apply() -> None:
            page.open(
                ft.SnackBar(
                    content=ft.Text(text),
                    bgcolor=look.DANGER if error else look.RAISED,
                )
            )

        run_on_ui(apply)

    def set_running(active: bool) -> None:
        def apply() -> None:
            running["value"] = active
            for control in (
                pick_btn,
                source_dropdown,
                transcribe_btn,
                translate_btn,
                voice_btn,
                target_dropdown,
                translator_dropdown,
                voice_dropdown,
                custom_voice,
                threads_field,
                embed_check,
                original_check,
                duck_slider,
                speed_slider,
            ):
                control.disabled = active
            cancel_btn.disabled = not active
            progress_footer.visible = active
            on_busy(active)
            page.update()

        run_on_ui(apply)

    def append_log(line: str) -> None:
        def apply() -> None:
            current = log_box.value or ""
            log_box.value = f"{current}\n{line}".strip()
            page.update()

        run_on_ui(apply)

    def dub_threads() -> int:
        try:
            return max(int(str(threads_field.value).strip()), 1)
        except (TypeError, ValueError):
            return DEFAULT_THREADS

    def video_duration() -> float:
        video = state["video"]
        if video is None:
            return 0.0
        try:
            return get_audio_duration(video, find_ffprobe())
        except (OSError, FileNotFoundError):
            return 0.0

    def report_progress(stage: str, fraction: float, detail: str) -> None:
        now = time.monotonic()
        if (
            fraction < 1.0
            and stage == last_paint["stage"]
            and now - last_paint["time"] < 0.15
        ):
            return
        changed_stage = stage != last_paint["stage"]
        if changed_stage:
            last_paint["started"] = now
        last_paint["time"] = now
        last_paint["stage"] = stage

        def apply() -> None:
            if stage == "benchmark":
                return
            label = stage_label(stage)
            status_text.value = f"{label}: {detail}" if detail else label
            progress_bar.value = min(max(float(fraction), 0.0), 1.0)
            percent = int(min(max(fraction, 0.0), 1.0) * 100)
            progress_label.value = f"{percent}%"
            page.update()

        run_on_ui(apply)
        if changed_stage:
            if stage != "benchmark":
                extra = f" — {detail}" if detail and stage == "download" else ""
                append_log(f"{stage_label(stage)}{extra}")

    def edited_segments() -> list[dict]:
        rows: list[dict] = []
        segments = state["segments"]
        for index, segment in enumerate(segments):
            translation = segment.get("translation") or ""
            if index < len(translation_fields):
                translation = translation_fields[index].value or ""
                segment["translation"] = translation
            rows.append(
                {
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": segment.get("text") or "",
                    "translation": translation,
                }
            )
        return rows

    def rebuild_review() -> None:
        review.controls.clear()
        translation_fields.clear()
        segments = state["segments"]
        if not segments:
            review.controls.append(
                ft.Text(
                    t("review_empty"),
                    color=look.MUTE,
                )
            )
        for segment in segments:
            translated = ft.TextField(
                value=segment.get("translation") or "",
                multiline=True,
                min_lines=2,
                max_lines=5,
                text_size=13,
                expand=True,
                label=t("translation"),
            )
            translation_fields.append(translated)
            original = ft.TextField(
                value=segment.get("text") or "",
                multiline=True,
                min_lines=2,
                max_lines=5,
                text_size=13,
                read_only=True,
                expand=True,
                label=t("original"),
            )
            review.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                f"{format_srt_timestamp(segment['start'])}  →  "
                                f"{format_srt_timestamp(segment['end'])}",
                                size=12,
                                color=look.MUTE,
                            ),
                            ft.Row(
                                [original, translated],
                                vertical_alignment=ft.CrossAxisAlignment.START,
                            ),
                        ],
                        spacing=4,
                    ),
                    bgcolor=look.RAISED,
                    border=look.edge(),
                    border_radius=8,
                    padding=12,
                )
            )
        # parent is None until the view is mounted; .page raises in that case.
        if review.parent is not None:
            page.update()

    def current_options() -> DubOptions:
        try:
            threads = int(str(threads_field.value).strip())
        except ValueError as exc:
            raise ValueError(t("threads_int")) from exc
        if threads < 1:
            raise ValueError(t("threads_min"))
        custom = (custom_voice.value or "").strip()
        voice = custom or (voice_dropdown.value or DEFAULT_VOICE)
        duck = float(duck_slider.value if duck_slider.value is not None else DEFAULT_DUCK * 100)
        speed = float(speed_slider.value if speed_slider.value is not None else DEFAULT_MAX_SPEED)
        if speed < 1.0:
            raise ValueError(t("speed_min"))
        return DubOptions(
            model=get_model(),
            language=get_language(),
            device=get_device(),
            target_language=target_dropdown.value or "ru",
            translator=translator_dropdown.value or "auto",
            voice=voice,
            duck=duck / 100.0,
            max_speed=speed,
            threads=threads,
            embed_subs=bool(embed_check.value),
            keep_original_track=bool(original_check.value),
        )

    def fail(message: str) -> None:
        append_log(t("log_error", error=message))
        set_running(False)

        def apply() -> None:
            status_text.value = t("status_error")
            page.update()

        run_on_ui(apply)
        show_message(message, error=True)

    def transcribe_worker(options: DubOptions) -> None:
        try:
            result = transcribe_timed(
                state["video"],
                options,
                on_progress=report_progress,
                cancel=cancel_event.is_set,
            )
        except DubCancelled:
            append_log(t("stopped"))
            set_running(False)
            return
        except Exception as exc:
            fail(str(exc))
            return

        def apply() -> None:
            state["language"] = result.language
            state["duration"] = result.duration
            state["segments"] = result.segments
            state["done"] = False
            rebuild_review()
            status_text.value = t("status_lines_ready", n=len(result.segments))
            progress_bar.value = 1
            progress_label.value = "100%"
            paint_step()
            page.update()

        run_on_ui(apply)
        append_log(
            t(
                "log_lines",
                language=result.language or t("unknown_language"),
                n=len(result.segments),
                seconds=f"{result.duration:.1f}",
            )
        )
        set_running(False)

    def translate_worker(options: DubOptions, segments: list[dict]) -> None:
        try:
            engine = translate_segments(
                segments,
                options,
                state["language"] or options.language or "",
                on_progress=report_progress,
                cancel=cancel_event.is_set,
            )
        except DubCancelled:
            append_log(t("stopped"))
            set_running(False)
            return
        except Exception as exc:
            fail(str(exc))
            return

        def apply() -> None:
            state["segments"] = segments
            state["done"] = False
            rebuild_review()
            status_text.value = t("status_translation_ready")
            progress_bar.value = 1
            progress_label.value = "100%"
            paint_step()
            page.update()

        run_on_ui(apply)
        append_log(engine)
        set_running(False)

    def voice_worker(options: DubOptions, segments: list[dict], output: Path) -> None:
        try:
            report = render_dub(
                state["video"],
                segments,
                duration=float(state["duration"] or 0.0),
                options=options,
                output=output,
                source_language=state["language"],
                on_progress=report_progress,
                cancel=cancel_event.is_set,
            )
        except DubCancelled:
            append_log(t("stopped"))
            set_running(False)
            return
        except Exception as exc:
            fail(str(exc))
            return
        for item in report:
            note = ""
            if item["overflow"] > 0.05:
                note = t("overflow", seconds=f"{item['overflow']:.2f}")
            append_log(
                t(
                    "voice_line",
                    index=f"{item['index'] + 1:02d}",
                    slot=f"{item['slot']:.2f}",
                    duration=f"{item['duration']:.2f}",
                    speed=f"{item['speed']:.2f}",
                    note=note,
                )
            )
        append_log(t("video_saved", path=output))

        def apply() -> None:
            state["done"] = True
            state["output_name"] = output.name
            status_text.value = t("dub_ready")
            progress_bar.value = 1
            progress_label.value = "100%"
            paint_step()
            page.update()

        run_on_ui(apply)
        show_message(t("saved_name", name=output.name))
        set_running(False)

    def require_idle() -> bool:
        if running["value"]:
            show_message(t("wait"), error=True)
            return False
        return True

    def on_transcribe(_=None) -> None:
        if not require_idle():
            return
        if state["video"] is None:
            show_message(t("pick_video_first"), error=True)
            return
        try:
            options = current_options()
        except ValueError as exc:
            show_message(str(exc), error=True)
            return
        cancel_event.clear()
        last_paint["stage"] = ""
        last_paint["started"] = time.monotonic()
        state["duration"] = video_duration()
        log_box.value = ""
        progress_bar.value = 0
        progress_label.value = "0%"
        status_text.value = t("status_transcribe")
        running["value"] = True
        set_running(True)
        page.run_thread(lambda: transcribe_worker(options))

    def on_translate(_=None) -> None:
        if not require_idle():
            return
        if not state["segments"]:
            show_message(t("transcribe_first"), error=True)
            return
        if not (state["language"] or get_language()):
            show_message(t("need_source_language"), error=True)
            return
        try:
            options = current_options()
        except ValueError as exc:
            show_message(str(exc), error=True)
            return
        if state["language"] is None and options.language:
            state["language"] = options.language
        segments = edited_segments()
        cancel_event.clear()
        last_paint["stage"] = ""
        status_text.value = t("status_translate")
        set_running(True)
        page.run_thread(lambda: translate_worker(options, segments))

    async def on_voice(_=None) -> None:
        if not require_idle():
            return
        if state["video"] is None:
            show_message(t("pick_video_first"), error=True)
            return
        if not state["segments"]:
            show_message(t("transcribe_and_translate"), error=True)
            return
        suggested_target = target_dropdown.value or "ru"
        suggested = f"{state['video'].stem}.{suggested_target}.mp4"
        path = await save_path(
            file_name=suggested,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["mp4"],
        )
        if not path:
            return
        segments = edited_segments()
        if not any(row["translation"].strip() for row in segments):
            show_message(t("nothing_to_voice"), error=True)
            return
        try:
            options = current_options()
        except ValueError as exc:
            show_message(str(exc), error=True)
            return
        cancel_event.clear()
        last_paint["stage"] = ""
        status_text.value = t("status_voice")
        set_running(True)
        page.run_thread(lambda: voice_worker(options, segments, Path(path)))

    def on_cancel(_=None) -> None:
        cancel_event.set()
        status_text.value = t("status_cancel")
        page.update()

    async def pick_paths(**kwargs):
        try:
            return await file_picker.pick_files(**kwargs)
        except Exception as exc:
            show_message(t("pick_file_failed", error=exc), error=True)
            return None

    async def save_path(**kwargs):
        try:
            return await file_picker.save_file(**kwargs)
        except Exception as exc:
            show_message(t("save_failed", error=exc), error=True)
            return None

    async def on_pick_video(_=None) -> None:
        if not require_idle():
            return
        picked = await pick_paths(allow_multiple=False)
        if not picked or not picked[0].path:
            return
        had_lines = bool(state["segments"])
        state["video"] = Path(picked[0].path)
        state["language"] = None
        state["segments"] = []
        state["done"] = False
        state["output_name"] = ""
        state["duration"] = video_duration()
        log_box.value = ""
        progress_bar.value = 0
        progress_label.value = "0%"
        video_label.value = state["video"].name
        video_path_label.value = str(state["video"].parent)
        rebuild_review()
        paint_step()
        if had_lines:
            step_hint.value = t("new_video_reset")
        page.update()

    def payload():
        video = state["video"]
        return segments_payload(
            edited_segments(),
            language=state["language"],
            target_language=target_dropdown.value or "ru",
            source=str(video) if video else None,
        )

    async def on_export_json(_=None) -> None:
        if not state["segments"]:
            show_message(t("no_lines"))
            return
        stem = state["video"].stem if state["video"] else "segments"
        path = await save_path(
            file_name=f"{stem}.segments.json",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["json"],
        )
        if not path:
            return
        body = payload()
        if (target_dropdown.value or "ru") == "ru":
            for item in body["segments"]:
                item["ru"] = item["translation"]
        dump_segments(Path(path), body)
        show_message(t("json_saved"))

    async def on_export_srt(key: str, suffix: str) -> None:
        if not state["segments"]:
            show_message(t("no_lines"))
            return
        rows = edited_segments()
        stem = state["video"].stem if state["video"] else "subtitles"
        path = await save_path(
            file_name=f"{stem}.{suffix}.srt",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["srt"],
        )
        if not path:
            return
        Path(path).write_text(
            segments_to_subtitles(rows, text_key=key, max_duration=None),
            encoding="utf-8",
        )
        show_message(t("srt_saved"))

    async def on_export_translation_srt(_=None) -> None:
        await on_export_srt("translation", target_dropdown.value or "ru")

    async def on_export_source_srt(_=None) -> None:
        language = state["language"] or "src"
        await on_export_srt("text", language)

    async def on_export_timed(_=None) -> None:
        if not state["segments"]:
            show_message(t("no_lines"))
            return
        rows = edited_segments()
        key = "translation" if any(row["translation"].strip() for row in rows) else "text"
        stem = state["video"].stem if state["video"] else "timed"
        path = await save_path(
            file_name=f"{stem}.txt",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["txt"],
        )
        if not path:
            return
        Path(path).write_text(segments_to_timed_text(rows, text_key=key), encoding="utf-8")
        show_message(t("timed_saved"))

    async def on_import_json(_=None) -> None:
        if not require_idle():
            return
        picked = await pick_paths(
            allow_multiple=False,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["json"],
        )
        if not picked or not picked[0].path:
            return
        try:
            loaded = load_segments(Path(picked[0].path))
        except (OSError, ValueError) as exc:
            show_message(str(exc), error=True)
            return
        state["segments"] = loaded["segments"]
        if loaded.get("language"):
            state["language"] = loaded["language"]
        target = loaded.get("target_language")
        if target and any(code == target for code, _label in TARGET_LANGUAGES):
            target_dropdown.value = target
        state["done"] = False
        rebuild_review()
        paint_step()
        append_log(t("json_imported", n=len(state["segments"])))
        show_message(t("segments_loaded"))
        page.update()

    async def on_import_srt(_=None) -> None:
        if not require_idle():
            return
        picked = await pick_paths(
            allow_multiple=False,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["srt"],
        )
        if not picked or not picked[0].path:
            return
        try:
            cues = parse_srt(Path(picked[0].path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            show_message(str(exc), error=True)
            return
        how = apply_srt_cues(state["segments"], cues, field="translation")
        state["done"] = False
        rebuild_review()
        paint_step()
        page.update()
        if how == "text":
            show_message(t("srt_text_only"))
        else:
            show_message(t("srt_replaced_all"))

    def on_target_changed(_=None) -> None:
        if (custom_voice.value or "").strip():
            return
        voice = DEFAULT_VOICES.get(target_dropdown.value or "")
        if voice and voice in PIPER_VOICES:
            voice_dropdown.value = voice
            voice_hint.value = t("voice_hint_matched")
        else:
            voice_hint.value = t("voice_hint_missing")
        paint_step()
        page.update()

    def on_duck_changed(_=None) -> None:
        duck_value.value = f"{int(round(float(duck_slider.value or 0)))}%"
        page.update()

    def on_speed_changed(_=None) -> None:
        speed_value.value = f"{float(speed_slider.value or DEFAULT_MAX_SPEED):.1f}×"
        page.update()

    video_label = ft.Text(t("no_file"), weight=ft.FontWeight.W_500, color=look.IVORY)
    video_path_label = ft.Text("", size=12, color=look.MUTE)
    pick_btn = ft.Button(
        t("pick_video"),
        icon=ft.Icons.VIDEO_FILE_OUTLINED,
        style=look.quiet_style(),
        on_click=on_pick_video,
    )
    threads_field = ft.TextField(
        value=str(DEFAULT_THREADS),
        width=90,
        label=t("threads"),
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    target_dropdown = ft.Dropdown(
        value="ru",
        options=[ft.dropdown.Option(key=code, text=label) for code, label in TARGET_LANGUAGES],
        on_select=on_target_changed,
    )
    translator_dropdown = ft.Dropdown(
        value="auto",
        options=[
            ft.dropdown.Option(key="auto", text=t("translator_auto")),
            ft.dropdown.Option(key="nllb", text=t("translator_nllb")),
            ft.dropdown.Option(key="argos", text=t("translator_argos")),
        ],
    )
    voice_dropdown = ft.Dropdown(
        value=DEFAULT_VOICE,
        options=[
            ft.dropdown.Option(key=voice_id, text=info["label"])
            for voice_id, info in PIPER_VOICES.items()
        ],
    )
    custom_voice = ft.TextField(
        label=t("custom_voice"),
        hint_text=t("custom_voice_hint"),
    )
    voice_hint = ft.Text(
        t("voice_hint"),
        size=12,
        color=look.MUTE,
    )
    duck_slider = ft.Slider(min=0, max=100, value=20, divisions=20, on_change=on_duck_changed)
    duck_value = ft.Text("20%", size=12, color=look.MUTE)
    speed_slider = ft.Slider(
        min=1,
        max=2.5,
        value=DEFAULT_MAX_SPEED,
        divisions=15,
        round=1,
        on_change=on_speed_changed,
    )
    speed_value = ft.Text("1.6×", size=12, color=look.MUTE)
    embed_check = ft.Checkbox(label=t("embed_subs"), value=False)
    original_check = ft.Checkbox(label=t("keep_original"), value=True)
    status_text = ft.Text(t("pick_video_first"), color=look.MUTE, size=13)
    progress_bar = ft.ProgressBar(
        value=0,
        bar_height=3,
        border_radius=2,
        color=look.ACCENT,
        bgcolor=look.LINE,
    )
    progress_label = ft.Text("0%", size=13, color=look.IVORY, weight=ft.FontWeight.W_500)
    log_box = ft.TextField(
        label=t("log"),
        multiline=True,
        min_lines=5,
        max_lines=8,
        read_only=True,
    )
    review = ft.ListView(expand=True, spacing=8, padding=4)
    transcribe_btn = ft.FilledButton(
        t("dub_transcribe"),
        icon=ft.Icons.SUBTITLES_OUTLINED,
        style=look.primary_style(),
        on_click=on_transcribe,
    )
    translate_btn = ft.FilledButton(
        t("dub_translate"),
        icon=ft.Icons.TRANSLATE,
        style=look.quiet_style(),
        on_click=on_translate,
    )
    voice_btn = ft.FilledButton(
        t("dub_voice"),
        icon=ft.Icons.RECORD_VOICE_OVER_OUTLINED,
        style=look.primary_style(),
        on_click=on_voice,
    )
    cancel_btn = ft.OutlinedButton(
        t("cancel"),
        icon=ft.Icons.STOP_ROUNDED,
        disabled=True,
        on_click=on_cancel,
    )
    dub_title = ft.Text(t("dub_title"), size=18, weight=ft.FontWeight.W_500, color=look.IVORY)
    step_hint = ft.Text(
        t("dub_step_pick"),
        size=13,
        color=look.MUTE,
    )
    threads_hint = ft.Text(
        t("threads_hint", threads=DEFAULT_THREADS, cap=TRANSLATOR_THREAD_CAP),
        size=12,
        color=look.MUTE,
    )
    label_target = ft.Text(t("target_language"), size=12, color=look.MUTE)
    label_translator = ft.Text(t("translator"), size=12, color=look.MUTE)
    label_voice = ft.Text(t("voice"), size=12, color=look.MUTE)
    duck_label = ft.Text(t("duck"), size=13)
    speed_label = ft.Text(t("speed_limit"), size=13)

    transcribe_block = ft.Column([transcribe_btn], tight=True, spacing=8)
    translate_block = ft.Column(
        [label_target, target_dropdown, translate_btn],
        tight=True,
        spacing=8,
    )
    voice_block = ft.Column(
        [
            label_voice,
            voice_dropdown,
            custom_voice,
            voice_hint,
            ft.Row(
                [duck_label, duck_value],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            duck_slider,
            ft.Row(
                [speed_label, speed_value],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            speed_slider,
            embed_check,
            original_check,
            voice_btn,
        ],
        tight=True,
        spacing=8,
    )
    def dub_source_options() -> list[ft.dropdown.Option]:
        if source_choices is None:
            return [ft.dropdown.Option(key="auto", text=t("auto"))]
        return [ft.dropdown.Option(key=key, text=label) for key, label in source_choices()]

    def on_dub_source(_=None) -> None:
        if on_source_changed is not None:
            on_source_changed(source_dropdown.value or "auto")

    source_dropdown = ft.Dropdown(
        value=initial_source,
        options=dub_source_options(),
        on_select=on_dub_source,
    )
    dub_source_label = ft.Text(t("source_language"), size=12, color=look.MUTE)
    source_block = ft.Column(
        [dub_source_label, source_dropdown],
        spacing=8,
        tight=True,
        visible=False,
    )
    controls = ft.Column(
        [
            dub_title,
            step_hint,
            pick_btn,
            video_label,
            video_path_label,
            source_block,
            transcribe_block,
            translate_block,
            voice_block,
        ],
        spacing=12,
        scroll=ft.ScrollMode.AUTO,
        expand=True,
    )

    review_title = ft.Text(t("review"), size=18, weight=ft.FontWeight.W_500, color=look.IVORY)
    review_blurb = ft.Text(
        t("review_blurb"),
        size=12,
        color=look.MUTE,
    )
    lines_file_label = ft.Text(t("lines_file"), size=13, color=look.IVORY)
    export_json_item = ft.PopupMenuItem(content=t("export_json"), on_click=on_export_json)
    export_translation_item = ft.PopupMenuItem(content=t("export_translation_srt"), on_click=on_export_translation_srt)
    export_source_item = ft.PopupMenuItem(content=t("export_source_srt"), on_click=on_export_source_srt)
    export_timed_item = ft.PopupMenuItem(content=t("save_timed"), on_click=on_export_timed)
    import_json_item = ft.PopupMenuItem(content=t("import_json"), on_click=on_import_json)
    import_srt_item = ft.PopupMenuItem(content=t("import_srt"), on_click=on_import_srt)
    lines_menu = ft.PopupMenuButton(
        content=ft.Row(
            [
                ft.Icon(ft.Icons.FOLDER_OUTLINED, size=16, color=look.IVORY),
                lines_file_label,
            ],
            tight=True,
            spacing=6,
        ),
        tooltip=t("lines_file"),
        bgcolor=look.PANEL,
        style=look.quiet_style(),
        items=[
            export_json_item,
            export_translation_item,
            export_source_item,
            export_timed_item,
            import_json_item,
            import_srt_item,
        ],
    )

    review_pane = ft.Column(
        [
            ft.Row(
                [
                    ft.Column([review_title, review_blurb], expand=True, spacing=2),
                    lines_menu,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            ft.Container(content=review, expand=True),
        ],
        spacing=8,
        expand=True,
    )
    review_wrap = ft.Container(
        content=review_pane,
        expand=6,
        bgcolor=look.PANEL,
        border=look.edge(),
        border_radius=12,
        padding=20,
        visible=False,
    )
    progress_footer = ft.Container(
        visible=False,
        border=ft.Border.only(top=ft.BorderSide(1, look.LINE)),
        padding=ft.Padding.only(top=14),
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Container(content=status_text, expand=True),
                        progress_label,
                        cancel_btn,
                    ],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                progress_bar,
            ],
            spacing=8,
        ),
    )

    def paint_step() -> None:
        video = state["video"] is not None
        lines = bool(state["segments"])
        translated = any((segment.get("translation") or "").strip() for segment in state["segments"])
        done = bool(state["done"])
        pick_btn.content = t("another_video") if video else t("pick_video")
        pick_btn.style = look.quiet_style() if video else look.primary_style()
        video_label.visible = video
        video_path_label.visible = video
        source_block.visible = video
        transcribe_block.visible = video and not lines
        translate_block.visible = lines and not translated
        voice_block.visible = translated
        custom_voice.visible = voice_hint.value in all_values("voice_hint_missing")
        review_wrap.visible = lines
        if not video:
            step_hint.value = t("dub_step_pick")
        elif not lines:
            step_hint.value = t("dub_step_transcribe")
        elif not translated:
            step_hint.value = t("dub_step_translate")
        elif done:
            step_hint.value = t("dub_step_done", name=state["output_name"])
        else:
            step_hint.value = t("dub_step_voice")

    def apply_ui() -> None:
        dub_title.value = t("dub_title")
        dub_source_label.value = t("source_language")
        source_value = source_dropdown.value
        source_dropdown.options = dub_source_options()
        source_dropdown.value = source_value
        pick_btn.content = t("pick_video")
        threads_field.label = t("threads")
        threads_hint.value = t("threads_hint", threads=DEFAULT_THREADS, cap=TRANSLATOR_THREAD_CAP)
        transcribe_btn.content = t("dub_transcribe")
        label_target.value = t("target_language")
        label_translator.value = t("translator")
        translate_btn.content = t("dub_translate")
        label_voice.value = t("voice")
        custom_voice.label = t("custom_voice")
        custom_voice.hint_text = t("custom_voice_hint")
        if voice_hint.value in all_values("voice_hint"):
            voice_hint.value = t("voice_hint")
        elif voice_hint.value in all_values("voice_hint_matched"):
            voice_hint.value = t("voice_hint_matched")
        elif voice_hint.value in all_values("voice_hint_missing"):
            voice_hint.value = t("voice_hint_missing")
        duck_label.value = t("duck")
        speed_label.value = t("speed_limit")
        embed_check.label = t("embed_subs")
        original_check.label = t("keep_original")
        voice_btn.content = t("dub_voice")
        cancel_btn.content = t("cancel")
        log_box.label = t("log")
        review_title.value = t("review")
        review_blurb.value = t("review_blurb")
        lines_file_label.value = t("lines_file")
        lines_menu.tooltip = t("lines_file")
        export_json_item.content = t("export_json")
        export_translation_item.content = t("export_translation_srt")
        export_source_item.content = t("export_source_srt")
        export_timed_item.content = t("save_timed")
        import_json_item.content = t("import_json")
        import_srt_item.content = t("import_srt")
        translator_dropdown.options = [
            ft.dropdown.Option(key="auto", text=t("translator_auto")),
            ft.dropdown.Option(key="nllb", text=t("translator_nllb")),
            ft.dropdown.Option(key="argos", text=t("translator_argos")),
        ]
        if video_label.value in no_file_labels():
            video_label.value = t("no_file")
        if status_text.value in all_values("pick_video_first"):
            status_text.value = t("pick_video_first")
        rebuild_review()
        paint_step()

    def set_source(key: str) -> None:
        source_dropdown.value = key or "auto"

    def set_source_locked(locked: bool) -> None:
        source_dropdown.disabled = locked

    if ui_hooks is not None:
        ui_hooks["apply"] = apply_ui
        ui_hooks["set_source"] = set_source
        ui_hooks["set_source_locked"] = set_source_locked

    rebuild_review()
    paint_step()

    return ft.Column(
        [
            ft.Row(
                [
                    ft.Container(content=controls, expand=5),
                    review_wrap,
                ],
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                spacing=16,
            ),
            progress_footer,
        ],
        expand=True,
        spacing=0,
        visible=False,
    )
