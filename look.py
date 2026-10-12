"""Quiet dark surfaces. One warm accent, hairline borders, no default blue."""

from __future__ import annotations

import flet as ft

INK = "#0C0D10"
PANEL = "#14161B"
RAISED = "#1C1F26"
LINE = "#2E323A"
IVORY = "#F3F0E8"
MUTE = "#9C978E"
ACCENT = "#C6A36A"
DANGER = "#C4564A"


def theme() -> ft.Theme:
    scheme = ft.ColorScheme(
        primary=ACCENT,
        on_primary=INK,
        primary_container=RAISED,
        on_primary_container=IVORY,
        secondary=MUTE,
        on_secondary=INK,
        surface=PANEL,
        on_surface=IVORY,
        on_surface_variant=MUTE,
        surface_container_lowest=INK,
        surface_container_low=INK,
        surface_container=PANEL,
        surface_container_high=RAISED,
        surface_container_highest=RAISED,
        surface_bright=RAISED,
        outline=LINE,
        outline_variant=LINE,
        error=DANGER,
        on_error=IVORY,
    )
    return ft.Theme(color_scheme=scheme)


def mount(page: ft.Page) -> None:
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = INK
    built = theme()
    page.theme = built
    page.dark_theme = built
    page.padding = 28


def edge() -> ft.Border:
    return ft.Border.all(1, LINE)


def primary_style() -> ft.ButtonStyle:
    return ft.ButtonStyle(
        bgcolor=ACCENT,
        color=INK,
        shape=ft.RoundedRectangleBorder(radius=8),
    )


def quiet_style(active: bool = False) -> ft.ButtonStyle:
    return ft.ButtonStyle(
        bgcolor=ACCENT if active else "transparent",
        color=INK if active else IVORY,
        side=ft.BorderSide(1, ACCENT if active else LINE),
        shape=ft.RoundedRectangleBorder(radius=8),
    )
