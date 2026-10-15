"""Voiceover screen: text or subtitles to speech with an installed Piper voice."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import flet as ft

import look
from dub import DEFAULT_TARGET, TARGET_LANGUAGES, DubCancelled
from i18n import t
from voiceover import (
    MAX_SPEED,
    MIN_SPEED,
    VoiceoverInput,
    preview_text,
    read_input,
    render_voiceover,
    voice_label,
    voices_for_language,
)

PREVIEW_FILE = Path(tempfile.gettempdir()) / "tagos-voiceover-preview.wav"


def voiceover_languages() -> list[tuple[str, str]]:
    """Target languages that have at least one Piper voice."""
    return [(code, name) for code, name in TARGET_LANGUAGES if voices_for_language(code)]


def play_file(path: Path) -> bool:
    player = shutil.which("ffplay")
    if not player:
        return False
    subprocess.Popen(
        [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def build_voiceover_view(page: ft.Page, *, on_busy, ui_hooks: dict | None = None) -> ft.Column:
    file_picker = ft.FilePicker()
    state: dict = {"cues": None, "name": ""}
    cancel_event = threading.Event()
    running = {"value": False}

    def run_on_ui(callback) -> None:
        page.session.connection.loop.call_soon_threadsafe(callback)

    def show_message(text: str, error: bool = False) -> None:
        run_on_ui(lambda: page.open(ft.SnackBar(content=ft.Text(text),
                                                bgcolor=look.DANGER if error else look.RAISED)))

    title = ft.Text(t("vo_title"), size=22, weight=ft.FontWeight.W_500, color=look.IVORY)
    blurb = ft.Text(t("vo_blurb"), size=12, color=look.MUTE)
    text_box = ft.TextField(multiline=True, min_lines=12, max_lines=18, label=t("vo_text"),
                            hint_text=t("vo_text_hint"), expand=True)
    source_label = ft.Text("", size=12, color=look.MUTE)
    lang_label = ft.Text(t("language"), size=12, color=look.MUTE)
    voice_title = ft.Text(t("voice"), size=12, color=look.MUTE)
    lang_dropdown = ft.Dropdown(
        width=260,
        value=DEFAULT_TARGET,
        options=[ft.dropdown.Option(key=c, text=n) for c, n in voiceover_languages()],
    )
    voice_dropdown = ft.Dropdown(width=260)
    speed_label = ft.Text(t("vo_speed"), size=12, color=look.MUTE)
    speed_value = ft.Text("1.0×", size=12, color=look.MUTE)
    speed_slider = ft.Slider(min=MIN_SPEED, max=MAX_SPEED, value=1.0, divisions=15, round=1)
    timed_check = ft.Checkbox(label=t("vo_timed"), value=True, visible=False)
    status_text = ft.Text("", size=12, color=look.MUTE)
    progress_bar = ft.ProgressBar(value=0, visible=False)

    def refresh_voices() -> None:
        voices = voices_for_language(lang_dropdown.value or DEFAULT_TARGET)
        voice_dropdown.options = [ft.dropdown.Option(key=v, text=voice_label(v)) for v in voices]
        if voice_dropdown.value not in voices:
            voice_dropdown.value = voices[0] if voices else None

    def set_cues(cues: list[dict] | None, name: str) -> None:
        state["cues"] = cues
        state["name"] = name
        timed_check.visible = bool(cues)
        source_label.value = t("vo_source_srt", name=name, count=len(cues)) if cues else (
            t("vo_source_file", name=name) if name else "")

    def load_input(source: VoiceoverInput, name: str, language: str | None = None) -> None:
        """Entry point for other modes (Transcribe 'Send to Voiceover')."""
        text_box.value = source.text
        set_cues(source.cues, name)
        if language and any(o.key == language for o in lang_dropdown.options):
            lang_dropdown.value = language
            refresh_voices()

    def current_source() -> VoiceoverInput:
        text = text_box.value or ""
        cues = state["cues"]
        if cues and text != "\n".join(c["text"] for c in cues):
            # The text was edited. Keep timings only if the line count still matches.
            lines = [line for line in text.split("\n")]
            if len(lines) == len(cues):
                cues = [dict(c, text=line) for c, line in zip(cues, lines)]
            else:
                cues = None
        return VoiceoverInput(text=text, cues=cues)

    def set_running(active: bool) -> None:
        def apply() -> None:
            running["value"] = active
            for control in (open_btn, preview_btn, export_btn, lang_dropdown, voice_dropdown,
                            speed_slider, timed_check, text_box):
                control.disabled = active
            cancel_btn.visible = active
            progress_bar.visible = active
            on_busy(active)
            page.update()

        run_on_ui(apply)

    def report(stage: str, fraction: float, detail: str) -> None:
        def apply() -> None:
            progress_bar.value = fraction
            status_text.value = t("stage_" + stage) if stage in {"tts", "download"} else detail
            page.update()

        run_on_ui(apply)

    def worker(source: VoiceoverInput, output: Path, timed: bool, preview: bool) -> None:
        try:
            render_voiceover(source, voice_dropdown.value, output, speed=float(speed_slider.value),
                             timed=timed, on_progress=report, cancel=cancel_event.is_set)
        except DubCancelled:
            run_on_ui(lambda: setattr(status_text, "value", t("stopped")))
            set_running(False)
            return
        except Exception as exc:
            show_message(t("error_prefix", error=exc), error=True)
            set_running(False)
            return
        if preview:
            if not play_file(output):
                show_message(t("vo_preview_saved", path=output))
            msg = t("vo_previewing")
        else:
            msg = t("saved_name", name=output.name)
            show_message(msg)

        def done() -> None:
            status_text.value = msg
            page.update()

        run_on_ui(done)
        set_running(False)

    def start(output: Path, preview: bool) -> None:
        if running["value"]:
            show_message(t("wait"), error=True)
            return
        source = current_source()
        if not source.text.strip():
            show_message(t("vo_no_text"), error=True)
            return
        if not voice_dropdown.value:
            show_message(t("vo_no_voice"), error=True)
            return
        timed = bool(timed_check.visible and timed_check.value and source.cues)
        if preview:
            source = VoiceoverInput(text=preview_text(source.text))
            timed = False
        cancel_event.clear()
        set_running(True)
        page.run_thread(lambda: worker(source, output, timed, preview))

    async def on_open(_=None) -> None:
        try:
            picked = await file_picker.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.CUSTOM,
                                                  allowed_extensions=["txt", "srt"])
        except Exception as exc:
            show_message(t("pick_file_failed", error=exc), error=True)
            return
        if not picked or not picked[0].path:
            return
        path = Path(picked[0].path)
        try:
            source = read_input(path)
        except Exception as exc:
            show_message(t("error_prefix", error=exc), error=True)
            return
        load_input(source, path.name)
        page.update()

    async def on_export(_=None) -> None:
        stem = Path(state["name"]).stem if state["name"] else "voiceover"
        try:
            path = await file_picker.save_file(file_name=f"{stem}.wav", file_type=ft.FilePickerFileType.CUSTOM,
                                               allowed_extensions=["wav", "mp3"])
        except Exception as exc:
            show_message(t("save_failed", error=exc), error=True)
            return
        if not path:
            return
        output = Path(path)
        if output.suffix.lower() not in {".wav", ".mp3"}:
            output = output.with_suffix(".wav")
        start(output, preview=False)

    def on_lang(_=None) -> None:
        refresh_voices()
        page.update()

    def on_speed(_=None) -> None:
        speed_value.value = f"{float(speed_slider.value):.1f}×"
        page.update()

    def on_text_change(_=None) -> None:
        if state["cues"] is not None and not text_box.value:
            set_cues(None, "")
            page.update()

    lang_dropdown.on_select = on_lang
    speed_slider.on_change = on_speed
    text_box.on_change = on_text_change
    open_btn = ft.OutlinedButton(t("vo_open"), icon=ft.Icons.FOLDER_OPEN_OUTLINED, on_click=on_open)
    preview_btn = ft.OutlinedButton(t("vo_preview"), icon=ft.Icons.PLAY_ARROW_OUTLINED,
                                    on_click=lambda _: start(PREVIEW_FILE, preview=True))
    export_btn = ft.FilledButton(t("vo_export"), icon=ft.Icons.SAVE_ALT_OUTLINED,
                                 style=look.primary_style(), on_click=on_export)
    cancel_btn = ft.OutlinedButton(t("cancel"), visible=False, on_click=lambda _: cancel_event.set())
    refresh_voices()

    def apply_ui() -> None:
        title.value = t("vo_title")
        blurb.value = t("vo_blurb")
        text_box.label = t("vo_text")
        text_box.hint_text = t("vo_text_hint")
        lang_label.value = t("language")
        voice_title.value = t("voice")
        speed_label.value = t("vo_speed")
        timed_check.label = t("vo_timed")
        open_btn.content = t("vo_open")
        preview_btn.content = t("vo_preview")
        export_btn.content = t("vo_export")
        cancel_btn.content = t("cancel")
        set_cues(state["cues"], state["name"])

    if ui_hooks is not None:
        ui_hooks["apply"] = apply_ui
        ui_hooks["load"] = load_input
        ui_hooks["refresh_voices"] = refresh_voices

    settings = ft.Container(
        width=300,
        bgcolor=look.PANEL, border=look.edge(), border_radius=12, padding=20,
        content=ft.Column(
            [lang_label, lang_dropdown, voice_title, voice_dropdown,
             ft.Row([speed_label, speed_value], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
             speed_slider, timed_check, ft.Divider(height=12, color=look.LINE),
             preview_btn, export_btn, cancel_btn, progress_bar, status_text],
            spacing=10, scroll=ft.ScrollMode.AUTO,
        ),
    )
    editor = ft.Container(
        expand=True, bgcolor=look.PANEL, border=look.edge(), border_radius=12, padding=20,
        content=ft.Column([ft.Row([open_btn, source_label], spacing=12), text_box], spacing=10, expand=True),
    )
    return ft.Column(
        [title, blurb, ft.Row([editor, settings], expand=True, spacing=16,
                              vertical_alignment=ft.CrossAxisAlignment.STRETCH)],
        expand=True, spacing=8, visible=False,
    )

