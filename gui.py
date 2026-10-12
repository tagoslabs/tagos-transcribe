#!/usr/bin/env python3
"""Native desktop GUI for the transcribe tool."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import flet as ft

from dub_ui import build_dub_view
from i18n import (
    get_lang,
    idle_statuses,
    load_lang,
    no_file_labels,
    preview_placeholders,
    set_lang,
    t,
)
import branding
import look
from prefs import choice, save_pref
from timestamps import format_transcript_preview, segments_to_srt, segments_to_timed_text
from transcribe import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    TranscriptionResult,
    collect_inputs,
    find_ffmpeg,
    is_media_file,
    load_model,
    model_info_label,
    resolve_compute_type,
    resolve_device,
    transcribe_file,
)

SOURCE_LANGS = (
    ("auto", None, None),
    ("ru", "ru", "Русский"),
    ("en", "en", "English"),
    ("de", "de", "Deutsch"),
    ("fr", "fr", "Français"),
    ("es", "es", "Español"),
    ("it", "it", "Italiano"),
    ("pt", "pt", "Português"),
    ("zh", "zh", "中文"),
    ("ja", "ja", "日本語"),
    ("ko", "ko", "한국어"),
    ("uk", "uk", "Українська"),
    ("pl", "pl", "Polski"),
    ("tr", "tr", "Türkçe"),
    ("nl", "nl", "Nederlands"),
)


class TranscriptionSession:
    def __init__(self) -> None:
        self.model = None
        self.loaded_model_name: str | None = None
        self.loaded_device: str | None = None
        self.ffmpeg: str | None = None
        self.device = "cpu"
        self.compute_type = resolve_compute_type("auto", self.device)
        self.cancel_event = threading.Event()
        self.is_running = False


def prepare_window(page: ft.Page) -> None:
    """Apply the dark frame before the desktop window is shown."""
    page.title = branding.APP_NAME
    look.mount(page)
    page.padding = 0
    page.window.title_bar_hidden = True
    page.window.frameless = True
    page.window.brightness = ft.Brightness.DARK
    page.window.bgcolor = look.INK
    page.window.width = 1180
    page.window.height = 840
    page.window.min_width = 860
    page.window.min_height = 600
    page.window.visible = False


def build_footer() -> ft.Control:
    """About line: product name and public links from branding.py."""
    items: list[ft.Control] = [
        ft.Text(branding.TAGLINE, size=12, color=look.IVORY)
    ]
    for label, url in branding.LINKS:
        if branding.is_placeholder(url):
            items.append(ft.Text(f"{label}: {url}", size=12, color=look.IVORY))
        else:
            items.append(ft.TextButton(label, url=url))
    return ft.Container(
        padding=ft.Padding.only(left=28, right=28, bottom=10),
        content=ft.Row(items, spacing=16, wrap=True),
    )


def main(page: ft.Page) -> None:
    set_lang(load_lang())
    prepare_window(page)

    saved_device = choice("device", {"cpu", "cuda"}, "cpu")
    saved_model = choice("model", set(AVAILABLE_MODELS), DEFAULT_MODEL)
    saved_source = choice("source_language", {key for key, _code, _name in SOURCE_LANGS}, "auto")
    session = TranscriptionSession()
    session.device = saved_device
    session.compute_type = resolve_compute_type("auto", session.device)
    dub_busy = {"value": False}
    device_dropdown = ft.Dropdown(
        value=saved_device,
        options=[
            ft.dropdown.Option(key="cpu", text=t("cpu")),
            ft.dropdown.Option(key="cuda", text=t("gpu")),
        ],
    )
    ui_lang_dropdown = ft.Dropdown(
        value=get_lang(),
        options=[
            ft.dropdown.Option(key="ru", text="Русский"),
            ft.dropdown.Option(key="en", text="English"),
        ],
    )

    def refresh_shared_lock() -> None:
        locked = session.is_running or dub_busy["value"]
        device_dropdown.disabled = locked
        model_dropdown.disabled = locked
        language_dropdown.disabled = locked
        lock_dub_source = dub_hooks.get("set_source_locked")
        if lock_dub_source is not None:
            lock_dub_source(locked)

    file_paths: list[Path] = []
    last_preview_text = ""
    last_result: TranscriptionResult | None = None

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

    def set_running(running: bool) -> None:
        def apply() -> None:
            session.is_running = running
            start_btn.disabled = running
            cancel_btn.disabled = not running
            add_files_btn.disabled = running
            add_folder_btn.disabled = running
            clear_btn.disabled = running
            reset_btn.disabled = running
            refresh_shared_lock()
            recursive_check.disabled = running
            timestamps_check.disabled = running
            save_srt_btn.disabled = running
            save_timed_btn.disabled = running
            sync_workspace()
            page.update()

        run_on_ui(apply)

    def refresh_file_list() -> None:
        if not file_paths:
            files_body.content = empty_state
        else:
            file_list.controls.clear()
            for path in file_paths:
                file_list.controls.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Icon(ft.Icons.AUDIO_FILE_OUTLINED, color=look.MUTE, size=18),
                                ft.Column(
                                    [
                                        ft.Text(path.name, weight=ft.FontWeight.W_500, color=look.IVORY),
                                        ft.Text(
                                            str(path.parent),
                                            size=12,
                                            color=look.MUTE,
                                            max_lines=1,
                                            overflow=ft.TextOverflow.ELLIPSIS,
                                        ),
                                    ],
                                    expand=True,
                                    spacing=2,
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.CLOSE,
                                    tooltip=t("remove"),
                                    on_click=lambda e, p=path: remove_file(p),
                                    disabled=session.is_running,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.START,
                        ),
                        bgcolor=look.RAISED,
                        border=look.edge(),
                        border_radius=8,
                        padding=12,
                    )
                )
            files_body.content = file_list
        source_block.visible = bool(file_paths)
        sync_workspace()
        page.update()

    def add_paths(paths: list[Path]) -> None:
        existing = {path.resolve() for path in file_paths}
        for path in paths:
            resolved = path.resolve()
            if resolved not in existing and is_media_file(resolved):
                file_paths.append(resolved)
                existing.add(resolved)
        refresh_file_list()

    def remove_file(path: Path) -> None:
        nonlocal file_paths
        file_paths = [item for item in file_paths if item != path]
        refresh_file_list()

    def clear_files(_=None) -> None:
        nonlocal file_paths
        if session.is_running:
            show_message(t("wait_or_cancel"), error=True)
            return
        file_paths.clear()
        refresh_file_list()

    def reset_session(_=None) -> None:
        nonlocal file_paths, last_preview_text, last_result
        if session.is_running:
            show_message(t("wait_or_cancel"), error=True)
            return

        file_paths.clear()
        last_preview_text = ""
        last_result = None
        session.cancel_event.clear()
        preview_box.value = t("preview_placeholder")
        log_box.value = ""
        progress_bar.value = 0
        progress_label.value = "0%"
        status_text.value = t("status_ready")
        refresh_file_list()
        page.update()
        show_message(t("reset_done"))

    file_picker = ft.FilePicker()
    clipboard = ft.Clipboard()

    async def on_add_files(_=None) -> None:
        try:
            picked = await file_picker.pick_files(allow_multiple=True)
        except Exception as exc:
            show_message(t("pick_file_failed", error=exc), error=True)
            return
        if picked:
            add_paths([Path(item.path) for item in picked if item.path])

    async def on_add_folder(_=None) -> None:
        try:
            folder = await file_picker.get_directory_path()
        except Exception as exc:
            show_message(t("pick_folder_failed", error=exc), error=True)
            return
        if folder:
            add_paths(collect_inputs([Path(folder)], recursive_check.value))

    status_text = ft.Text(t("status_ready"), color=look.MUTE, size=13)
    progress_bar = ft.ProgressBar(
        value=0,
        bar_height=3,
        border_radius=2,
        color=look.ACCENT,
        bgcolor=look.LINE,
    )
    progress_label = ft.Text("0%", size=13, color=look.IVORY, weight=ft.FontWeight.W_500)
    preview_box = ft.TextField(
        label=t("result"),
        multiline=True,
        min_lines=5,
        max_lines=12,
        read_only=True,
        expand=True,
        value=t("preview_placeholder"),
    )
    log_box = ft.TextField(
        label=t("log"),
        multiline=True,
        min_lines=4,
        max_lines=8,
        read_only=True,
        expand=True,
    )

    source_key = {"value": saved_source}

    def source_language_code() -> str | None:
        key = source_key["value"] or "auto"
        for item_key, code, _name in SOURCE_LANGS:
            if item_key == key:
                return code
        return None

    def source_choices() -> list[tuple[str, str]]:
        return [
            (key, t("auto") if key == "auto" else name)
            for key, _code, name in SOURCE_LANGS
        ]

    def apply_source_key(key: str, *, from_dub: bool = False) -> None:
        selected = key or "auto"
        if not any(item_key == selected for item_key, _code, _name in SOURCE_LANGS):
            return
        source_key["value"] = selected
        language_dropdown.value = selected
        save_pref("source_language", selected)
        if not from_dub:
            set_dub_source = dub_hooks.get("set_source")
            if set_dub_source is not None:
                set_dub_source(selected)

    language_dropdown = ft.Dropdown(
        value=saved_source,
        options=[
            ft.dropdown.Option(
                key=key,
                text=t("auto") if key == "auto" else name,
            )
            for key, _code, name in SOURCE_LANGS
        ],
    )
    label_source = ft.Text(t("source_language"), size=12, color=look.MUTE)
    source_block = ft.Column(
        [label_source, language_dropdown],
        spacing=8,
        tight=True,
        visible=False,
    )
    model_dropdown = ft.Dropdown(
        value=saved_model,
        options=[
            ft.dropdown.Option(
                key=model_id,
                text=t(f"model_{model_id}"),
            )
            for model_id, info in AVAILABLE_MODELS.items()
        ],
    )
    model_hint = ft.Text(
        t(f"model_{saved_model}_hint"),
        size=12,
        color=look.MUTE,
    )
    device_info = ft.Text(
        model_info_label(saved_model, session.device, t("model")),
        size=12,
        color=look.MUTE,
    )
    recursive_check = ft.Checkbox(label=t("scan_subfolders"), value=False)
    timestamps_check = ft.Checkbox(label=t("timestamps_preview"), value=False)
    last_progress_paint = 0.0

    def paint_progress(
        file_index: int,
        total_files: int,
        file_fraction: float,
        source_name: str,
        current_seconds: float = 0.0,
        total_seconds: float = 0.0,
        stage: str = "transcribe",
    ) -> None:
        overall = ((file_index - 1) + file_fraction) / total_files if total_files else 0
        percent = int(min(max(file_fraction, 0.0), 1.0) * 100)
        progress_bar.value = min(max(overall, 0.0), 1.0)

        if stage == "extract":
            status_text.value = t("extracting", name=source_name)
        else:
            status_text.value = t("transcribing", name=source_name)
        progress_label.value = f"{percent}%"

        page.update()

    def schedule_progress(
        file_index: int,
        total_files: int,
        file_fraction: float,
        source_name: str,
        current_seconds: float = 0.0,
        total_seconds: float = 0.0,
        stage: str = "transcribe",
    ) -> None:
        nonlocal last_progress_paint
        now = time.monotonic()
        if (
            file_fraction < 1.0
            and stage not in {"done", "extract"}
            and now - last_progress_paint < 0.15
        ):
            return
        last_progress_paint = now

        def apply() -> None:
            paint_progress(
                file_index,
                total_files,
                file_fraction,
                source_name,
                current_seconds,
                total_seconds,
                stage,
            )

        page.session.connection.loop.call_soon_threadsafe(apply)

    def set_status(text: str) -> None:
        def apply() -> None:
            status_text.value = text
            page.update()

        run_on_ui(apply)

    def set_log(text: str) -> None:
        def apply() -> None:
            log_box.value = text
            page.update()

        run_on_ui(apply)

    def append_log(line: str) -> None:
        def apply() -> None:
            current = log_box.value or ""
            log_box.value = f"{current}\n{line}".strip()
            page.update()

        run_on_ui(apply)

    def preview_for(result: TranscriptionResult) -> str:
        return format_transcript_preview(
            result.source.name,
            result.language,
            result.text,
            result.segments,
            with_timestamps=bool(timestamps_check.value),
            file_label=t("file"),
            language_label=t("language"),
            unknown_language=t("unknown_language"),
        )

    def set_preview(text: str, result: TranscriptionResult | None = None) -> None:
        nonlocal last_preview_text, last_result

        def apply() -> None:
            nonlocal last_preview_text, last_result
            if result is not None:
                last_result = result
                text_value = preview_for(result)
            else:
                text_value = text
            last_preview_text = text_value
            preview_box.value = text_value
            sync_workspace()
            page.update()

        run_on_ui(apply)

    def finish_transcription(
        preview: str,
        done: int,
        total: int,
        errors: int,
        cancelled: bool,
    ) -> None:
        def apply() -> None:
            nonlocal last_preview_text

            if cancelled:
                status_text.value = t("cancelled_count", done=done, total=total)
            elif errors:
                status_text.value = t("done_errors", done=done, total=total)
            else:
                status_text.value = t("done_count", done=done, total=total)

            final_preview = preview or t("empty_transcript")
            last_preview_text = final_preview
            preview_box.value = final_preview

            session.is_running = False
            start_btn.disabled = False
            cancel_btn.disabled = True
            add_files_btn.disabled = False
            add_folder_btn.disabled = False
            clear_btn.disabled = False
            reset_btn.disabled = False
            refresh_shared_lock()
            recursive_check.disabled = False
            timestamps_check.disabled = False
            save_srt_btn.disabled = False
            save_timed_btn.disabled = False
            refresh_file_list()
            page.update()

            if cancelled:
                return
            if errors:
                page.open(
                    ft.SnackBar(
                        content=ft.Text(t("processed_errors", done=done, total=total, errors=errors)),
                        bgcolor=look.DANGER,
                    )
                )
            else:
                page.open(
                    ft.SnackBar(
                        content=ft.Text(t("processed_all", total=total)),
                        bgcolor=look.RAISED,
                    )
                )

        run_on_ui(apply)

    def transcription_worker() -> None:
        if not file_paths:
            set_running(False)
            show_message(t("add_one_file"), error=True)
            return

        language = source_language_code()
        selected_model = model_dropdown.value

        try:
            if session.ffmpeg is None:
                set_status(t("prepare_ffmpeg"))
                session.ffmpeg = find_ffmpeg()
            session.device = resolve_device(device_dropdown.value or "cpu")
            session.compute_type = resolve_compute_type("auto", session.device)
            if (
                session.model is None
                or session.loaded_model_name != selected_model
                or session.loaded_device != session.device
            ):
                set_status(t("loading_model", name=selected_model))
                session.model = load_model(
                    selected_model,
                    session.device,
                    session.compute_type,
                    quiet=True,
                )
                session.loaded_model_name = selected_model
                session.loaded_device = session.device
                run_on_ui(
                    lambda m=selected_model, device=session.device: (
                        setattr(device_info, "value", model_info_label(m, device, t("model"))),
                        page.update(),
                    )
                )
        except Exception as exc:
            set_status(t("error_prefix", error=exc))
            set_running(False)
            show_message(str(exc), error=True)
            return

        set_log(model_info_label(selected_model, session.device, t("model")))

        errors = 0
        total = len(file_paths)
        preview = ""

        for index, source in enumerate(file_paths, start=1):
            if session.cancel_event.is_set():
                append_log(t("stopped_by_user"))
                break

            append_log(f"[{index}/{total}] {source.name}")

            def make_progress_callback(file_index: int, source_name: str):
                def report(
                    fraction: float,
                    current_seconds: float,
                    total_seconds: float,
                    stage: str,
                ) -> None:
                    schedule_progress(
                        file_index,
                        total,
                        fraction,
                        source_name,
                        current_seconds,
                        total_seconds,
                        stage,
                    )

                return report

            try:
                result = transcribe_file(
                    session.model,
                    source,
                    session.ffmpeg,
                    language,
                    quiet=True,
                    on_progress=make_progress_callback(index, source.name),
                )
                run_on_ui(
                    lambda i=index, name=source.name: paint_progress(
                        i, total, 1.0, name, stage="done"
                    )
                )
                preview = preview_for(result)
                set_preview(preview, result)
                append_log(t("log_done", language=result.language))
            except Exception as exc:
                errors += 1
                append_log(t("log_error", error=exc))

        if not session.cancel_event.is_set():
            run_on_ui(lambda: paint_progress(total, total, 1.0, t("done_word"), stage="done"))

        done = total - errors
        finish_transcription(
            preview,
            done,
            total,
            errors,
            session.cancel_event.is_set(),
        )

    def start_transcription(_=None) -> None:
        if session.is_running:
            return
        session.cancel_event.clear()
        last_progress_paint = 0.0
        log_box.value = ""
        progress_bar.value = 0
        progress_label.value = "0%"
        session.is_running = True
        set_running(True)
        page.run_thread(transcription_worker)

    def cancel_transcription(_=None) -> None:
        session.cancel_event.set()
        status_text.value = t("cancel_after_file")
        page.update()

    def copy_preview(_=None) -> None:
        if not last_preview_text:
            show_message(t("nothing_to_copy"))
            return

        async def copy_to_clipboard() -> None:
            await clipboard.set(last_preview_text)
            show_message(t("copied"))

        page.run_task(copy_to_clipboard)

    def on_timestamps_changed(_=None) -> None:
        nonlocal last_preview_text
        if last_result is None:
            return
        last_preview_text = preview_for(last_result)
        preview_box.value = last_preview_text
        page.update()

    timestamps_check.on_change = on_timestamps_changed

    async def on_save_srt(_=None) -> None:
        if last_result is None:
            show_message(t("nothing_to_save"))
            return
        try:
            path = await file_picker.save_file(
                file_name=f"{last_result.source.stem}.srt",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["srt"],
            )
        except Exception as exc:
            show_message(t("save_failed", error=exc), error=True)
            return
        if not path:
            return
        Path(path).write_text(segments_to_srt(last_result.segments), encoding="utf-8")
        show_message(t("srt_saved"))

    async def on_save_timed(_=None) -> None:
        if last_result is None:
            show_message(t("nothing_to_save"))
            return
        try:
            path = await file_picker.save_file(
                file_name=f"{last_result.source.stem}.txt",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["txt"],
            )
        except Exception as exc:
            show_message(t("save_failed", error=exc), error=True)
            return
        if not path:
            return
        Path(path).write_text(
            segments_to_timed_text(last_result.segments),
            encoding="utf-8",
        )
        show_message(t("timed_saved"))

    def on_model_changed(_=None) -> None:
        if session.is_running:
            return
        session.model = None
        session.loaded_model_name = None
        session.loaded_device = None
        selected = model_dropdown.value
        model_hint.value = t(f"model_{selected}_hint")
        device_info.value = model_info_label(selected, session.device, t("model"))
        if selected in AVAILABLE_MODELS:
            save_pref("model", selected)
        page.update()

    def on_source_language_changed(_=None) -> None:
        apply_source_key(language_dropdown.value or "auto")

    def on_device_changed(_=None) -> None:
        if session.is_running:
            return
        session.device = resolve_device(device_dropdown.value or "cpu")
        session.compute_type = resolve_compute_type("auto", session.device)
        session.model = None
        session.loaded_model_name = None
        session.loaded_device = None
        device_info.value = model_info_label(model_dropdown.value or DEFAULT_MODEL, session.device, t("model"))
        if session.device in {"cpu", "cuda"}:
            save_pref("device", session.device)
        page.update()

    model_dropdown.on_select = on_model_changed
    language_dropdown.on_select = on_source_language_changed
    device_dropdown.on_select = on_device_changed

    add_files_btn = ft.Button(
        t("add_files"),
        icon=ft.Icons.FILE_OPEN_OUTLINED,
        on_click=on_add_files,
    )
    add_folder_btn = ft.OutlinedButton(
        t("add_folder"),
        icon=ft.Icons.FOLDER_OPEN_OUTLINED,
        on_click=on_add_folder,
    )
    clear_btn = ft.TextButton(t("clear_list"), on_click=clear_files)
    reset_btn = ft.OutlinedButton(
        t("reset"),
        icon=ft.Icons.RESTART_ALT,
        on_click=reset_session,
    )
    start_btn = ft.FilledButton(
        t("start"),
        icon=ft.Icons.PLAY_ARROW_ROUNDED,
        height=44,
        style=look.primary_style(),
        on_click=start_transcription,
    )
    cancel_btn = ft.OutlinedButton(
        t("cancel"),
        icon=ft.Icons.STOP_ROUNDED,
        height=44,
        disabled=True,
        on_click=cancel_transcription,
    )

    file_list = ft.ListView(expand=True, spacing=8, padding=4)

    empty_hint = ft.Text(
        t("empty_files"),
        color=look.MUTE,
        text_align=ft.TextAlign.CENTER,
    )
    empty_state = ft.Container(
        content=empty_hint,
        alignment=ft.Alignment.CENTER,
        expand=True,
    )

    files_body = ft.Container(
        content=empty_state,
        expand=True,
        bgcolor=look.INK,
        border=look.edge(),
        border_radius=8,
        padding=8,
    )

    files_title = ft.Text(t("files"), size=18, weight=ft.FontWeight.W_500, color=look.IVORY)
    files_hint = ft.Text(
        t("formats"),
        size=12,
        color=look.MUTE,
    )
    files_header = ft.Column(
        [
            files_title,
            ft.Row(
                [add_files_btn, add_folder_btn, clear_btn, reset_btn],
                spacing=8,
                wrap=True,
                run_spacing=8,
            ),
            recursive_check,
            files_hint,
            source_block,
        ],
        spacing=8,
        tight=True,
    )

    files_card = ft.Container(
        content=ft.Column(
            [files_header, files_body],
            expand=True,
            spacing=10,
        ),
        height=360,
        bgcolor=look.PANEL,
        border=look.edge(),
        border_radius=12,
        padding=20,
    )

    progress_title = ft.Text(t("progress"), size=18, weight=ft.FontWeight.W_500, color=look.IVORY)
    progress_card = ft.Container(
        visible=False,
        content=ft.Column(
            [
                progress_title,
                status_text,
                progress_bar,
                progress_label,
                cancel_btn,
            ],
            spacing=10,
            tight=True,
        ),
        bgcolor=look.PANEL,
        border=look.edge(),
        border_radius=12,
        padding=20,
    )

    save_srt_btn = ft.OutlinedButton(
        t("save_srt"),
        icon=ft.Icons.SUBTITLES_OUTLINED,
        on_click=on_save_srt,
    )
    save_timed_btn = ft.OutlinedButton(
        t("save_timed"),
        on_click=on_save_timed,
    )
    copy_btn = ft.OutlinedButton(
        t("copy"),
        icon=ft.Icons.CONTENT_COPY_OUTLINED,
        on_click=copy_preview,
    )

    preview_card = ft.Container(
        content=preview_box,
        expand=7,
        height=280,
        bgcolor=look.PANEL,
        border=look.edge(),
        border_radius=12,
        padding=16,
    )

    log_card = ft.Container(
        content=log_box,
        expand=3,
        height=280,
        bgcolor=look.PANEL,
        border=look.edge(),
        border_radius=12,
        padding=16,
    )

    output_block = ft.Column(
        [
            ft.Row(
                [
                    timestamps_check,
                    copy_btn,
                    save_srt_btn,
                    save_timed_btn,
                ],
                spacing=8,
                wrap=True,
                run_spacing=8,
            ),
            ft.Row(
                [preview_card, log_card],
                vertical_alignment=ft.CrossAxisAlignment.START,
                spacing=16,
            ),
        ],
        visible=False,
        spacing=12,
    )

    def sync_workspace() -> None:
        has_files = bool(file_paths)
        running = session.is_running
        has_result = last_result is not None
        has_log = bool((log_box.value or "").strip())
        clear_btn.visible = has_files
        reset_btn.visible = has_files or has_result or has_log
        start_btn.visible = has_files and not running
        progress_card.visible = running or (
            not has_result and status_text.value not in idle_statuses()
        )
        output_block.visible = has_result or has_log

    transcribe_view = ft.Column(
        [
            files_card,
            start_btn,
            progress_card,
            output_block,
        ],
        expand=True,
        spacing=16,
        scroll=ft.ScrollMode.AUTO,
    )
    dub_hooks: dict = {}

    def on_dub_busy(active: bool) -> None:
        dub_busy["value"] = active
        refresh_shared_lock()

    dub_view = build_dub_view(
        page,
        get_device=lambda: device_dropdown.value or "cpu",
        get_model=lambda: model_dropdown.value or DEFAULT_MODEL,
        get_language=source_language_code,
        on_busy=on_dub_busy,
        ui_hooks=dub_hooks,
        source_choices=source_choices,
        initial_source=saved_source,
        on_source_changed=lambda key: apply_source_key(key, from_dub=True),
    )

    def paint_nav(text_mode: bool) -> None:
        transcribe_mode_btn.style = look.quiet_style(text_mode)
        dub_mode_btn.style = look.quiet_style(not text_mode)

    def show_transcribe_mode(_=None) -> None:
        transcribe_view.visible = True
        dub_view.visible = False
        paint_nav(True)
        page.update()

    def show_dub_mode(_=None) -> None:
        transcribe_view.visible = False
        dub_view.visible = True
        paint_nav(False)
        page.update()

    transcribe_mode_btn = ft.FilledButton(
        t("mode_text"),
        icon=ft.Icons.SUBTITLES_OUTLINED,
        style=look.quiet_style(True),
        on_click=show_transcribe_mode,
    )
    dub_mode_btn = ft.FilledButton(
        t("mode_dub"),
        icon=ft.Icons.RECORD_VOICE_OVER_OUTLINED,
        style=look.quiet_style(False),
        on_click=show_dub_mode,
    )
    menu_blurb = ft.Text(
        t("menu_blurb"),
        size=12,
        color=look.MUTE,
    )
    label_task = ft.Text(t("task"), size=12, color=look.MUTE)
    label_model = ft.Text(t("model"), size=12, color=look.MUTE)
    label_device = ft.Text(t("device"), size=12, color=look.MUTE)
    label_ui = ft.Text(t("interface"), size=12, color=look.MUTE)

    def open_settings(_=None) -> None:
        page.show_dialog(settings_dialog)

    def close_settings(_=None) -> None:
        page.pop_dialog()

    settings_title = ft.Text(t("settings"), size=18, weight=ft.FontWeight.W_500, color=look.IVORY)
    close_settings_btn = ft.TextButton(t("close"), on_click=close_settings)
    settings_btn = ft.IconButton(
        icon=ft.Icons.SETTINGS_OUTLINED,
        icon_color=look.MUTE,
        icon_size=18,
        width=36,
        height=36,
        tooltip=t("settings"),
        on_click=open_settings,
    )

    def chrome_button(icon, tooltip, on_click):
        return ft.IconButton(
            icon=icon,
            icon_color=look.MUTE,
            icon_size=16,
            width=36,
            height=36,
            tooltip=tooltip,
            on_click=on_click,
        )

    def request_close(_=None) -> None:
        page.run_task(page.window.close)

    def request_minimize(_=None) -> None:
        page.window.minimized = True
        page.update()

    def request_maximize(_=None) -> None:
        page.window.maximized = not page.window.maximized
        sync_window_buttons()
        page.update()

    def sync_window_buttons(_=None) -> None:
        maximized = bool(page.window.maximized)
        maximize_btn.icon = ft.Icons.FILTER_NONE if maximized else ft.Icons.CROP_SQUARE
        maximize_btn.tooltip = t("window_restore") if maximized else t("window_maximize")

    def on_window_event(event: ft.WindowEvent) -> None:
        if event.type in (ft.WindowEventType.MAXIMIZE, ft.WindowEventType.UNMAXIMIZE):
            sync_window_buttons()
            page.update()

    minimize_btn = chrome_button(ft.Icons.HORIZONTAL_RULE, t("window_minimize"), request_minimize)
    maximize_btn = chrome_button(ft.Icons.CROP_SQUARE, t("window_maximize"), request_maximize)
    close_window_btn = chrome_button(ft.Icons.CLOSE, t("close"), request_close)
    page.window.on_event = on_window_event

    title_bar = ft.Container(
        height=44,
        bgcolor=look.INK,
        border=ft.Border.only(bottom=ft.BorderSide(1, look.LINE)),
        padding=ft.Padding.only(right=8),
        content=ft.Row(
            [
                ft.WindowDragArea(
                    expand=True,
                    content=ft.Container(
                        height=44,
                        alignment=ft.Alignment.CENTER_LEFT,
                        padding=ft.Padding.only(left=22),
                        content=ft.Text(
                            branding.APP_NAME,
                            size=15,
                            weight=ft.FontWeight.W_500,
                            color=look.IVORY,
                        ),
                    ),
                ),
                settings_btn,
                minimize_btn,
                maximize_btn,
                close_window_btn,
            ],
            spacing=2,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )
    settings_dialog = ft.AlertDialog(
        modal=True,
        title=settings_title,
        content=ft.Column(
            [
                label_ui,
                ui_lang_dropdown,
                label_device,
                device_dropdown,
            ],
            tight=True,
            width=320,
            spacing=10,
        ),
        actions=[close_settings_btn],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    menu = ft.Container(
        width=268,
        bgcolor=look.INK,
        border=ft.Border.only(right=ft.BorderSide(1, look.LINE)),
        padding=ft.Padding.only(right=22),
        content=ft.Column(
            [
                menu_blurb,
                ft.Divider(height=16, color=look.LINE),
                label_task,
                transcribe_mode_btn,
                dub_mode_btn,
                ft.Divider(height=16, color=look.LINE),
                label_model,
                model_dropdown,
                model_hint,
                device_info,
            ],
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        ),
    )

    def apply_ui_language(_=None) -> None:
        nonlocal last_preview_text
        set_lang(ui_lang_dropdown.value or "en")
        menu_blurb.value = t("menu_blurb")
        label_ui.value = t("interface")
        settings_title.value = t("settings")
        settings_btn.tooltip = t("settings")
        minimize_btn.tooltip = t("window_minimize")
        close_window_btn.tooltip = t("close")
        sync_window_buttons()
        close_settings_btn.content = t("close")
        label_task.value = t("task")
        label_model.value = t("model")
        label_source.value = t("source_language")
        label_device.value = t("device")
        transcribe_mode_btn.content = t("mode_text")
        dub_mode_btn.content = t("mode_dub")
        files_title.value = t("files")
        files_hint.value = t("formats")
        empty_hint.value = t("empty_files")
        progress_title.value = t("progress")
        add_files_btn.content = t("add_files")
        add_folder_btn.content = t("add_folder")
        clear_btn.content = t("clear_list")
        reset_btn.content = t("reset")
        start_btn.content = t("start")
        cancel_btn.content = t("cancel")
        copy_btn.content = t("copy")
        save_srt_btn.content = t("save_srt")
        save_timed_btn.content = t("save_timed")
        recursive_check.label = t("scan_subfolders")
        timestamps_check.label = t("timestamps_preview")
        preview_box.label = t("result")
        log_box.label = t("log")
        if preview_box.value in preview_placeholders():
            preview_box.value = t("preview_placeholder")
        if status_text.value in idle_statuses():
            status_text.value = t("status_ready")
        device_value = device_dropdown.value
        source_value = source_key["value"]
        model_value = model_dropdown.value
        device_dropdown.options = [
            ft.dropdown.Option(key="cpu", text=t("cpu")),
            ft.dropdown.Option(key="cuda", text=t("gpu")),
        ]
        device_dropdown.value = device_value
        language_dropdown.options = [
            ft.dropdown.Option(
                key=key,
                text=t("auto") if key == "auto" else name,
            )
            for key, _code, name in SOURCE_LANGS
        ]
        language_dropdown.value = source_value
        set_dub_source = dub_hooks.get("set_source")
        if set_dub_source is not None:
            set_dub_source(source_value)
        model_dropdown.options = [
            ft.dropdown.Option(key=model_id, text=t(f"model_{model_id}"))
            for model_id in AVAILABLE_MODELS
        ]
        model_dropdown.value = model_value
        selected = model_dropdown.value or DEFAULT_MODEL
        model_hint.value = t(f"model_{selected}_hint")
        device_info.value = model_info_label(selected, session.device, t("model"))
        if last_result is not None:
            last_preview_text = preview_for(last_result)
            preview_box.value = last_preview_text
        apply_dub = dub_hooks.get("apply")
        if apply_dub is not None:
            apply_dub()
        refresh_file_list()

    ui_lang_dropdown.on_select = apply_ui_language

    page.add(
        ft.Column(
            [
                title_bar,
                ft.Container(
                    expand=True,
                    padding=28,
                    content=ft.Row(
                        [
                            menu,
                            ft.Column(
                                [transcribe_view, dub_view],
                                expand=True,
                                spacing=0,
                            ),
                        ],
                        expand=True,
                        spacing=28,
                        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                    ),
                ),
                build_footer(),
            ],
            expand=True,
            spacing=0,
        )
    )

    page.window.visible = True
    refresh_file_list()


def run() -> None:
    os.environ["FLET_APP_ID"] = "com.appveyor.flet"
    ft.run(main, before_main=prepare_window, view=ft.AppView.FLET_APP_HIDDEN)


if __name__ == "__main__":
    run()
