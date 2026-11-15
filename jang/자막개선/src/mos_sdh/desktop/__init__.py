"""Pure Native Windows Qt6 (PySide6) Desktop Application for MoS-SDH Studio."""

from __future__ import annotations

__all__ = ["launch_desktop_app"]


def launch_desktop_app(argv: list[str] | None = None) -> int:
    from mos_sdh.desktop.__main__ import main

    return main(argv)
