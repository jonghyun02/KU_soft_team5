"""Native Windows Qt6 Desktop Entrypoint for MoS-SDH Studio."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import sys


def _set_windows_app_id() -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MoSSDH.Studio.Desktop.1.0")
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MoS-SDH Studio Native Windows Desktop App")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional mos_report.json path to load immediately on startup.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Optional PNG path to grab a window screenshot before auto-exit.",
    )
    parser.add_argument(
        "--select-segment",
        type=int,
        default=0,
        help="Optional segment_id to select on startup.",
    )
    parser.add_argument(
        "--smoke-exit-ms",
        type=int,
        default=0,
        help="Optional milliseconds after which the window closes automatically (for smoke tests).",
    )
    args, qt_extra = parser.parse_known_args(argv)

    # IMPORTANT on Windows: bind CUDA DLLs and import torch BEFORE PySide6 loads Qt DLLs
    from mos_sdh.config import get_default_cache_dir
    from mos_sdh.core.vram import bind_windows_cuda_dlls

    bind_windows_cuda_dlls()
    import torch  # noqa: F401

    _set_windows_app_id()

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from mos_sdh.desktop.main_window import MoSSDHMainWindow
    from mos_sdh.desktop.theme import STUDIO_DARK_QSS, ensure_studio_ico

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication([sys.argv[0], *qt_extra])
    app.setApplicationName("MoS-SDH Studio")
    app.setOrganizationName("MoS-SDH")
    app.setStyleSheet(STUDIO_DARK_QSS)

    local_ico = Path("./assets/mos_sdh.ico")
    ico_path = ensure_studio_ico(local_ico if local_ico.parent.exists() else (get_default_cache_dir() / "mos_sdh.ico"))
    app.setWindowIcon(QIcon(str(ico_path)))

    win = MoSSDHMainWindow()
    win.setWindowIcon(QIcon(str(ico_path)))
    if args.report is not None and args.report.exists():
        win.load_project_from_report(args.report)
    if args.select_segment > 0:
        win.select_segment(args.select_segment, seek_video=True)

    win.show()

    if args.smoke_exit_ms > 0:
        def _finish_smoke() -> None:
            if args.screenshot is not None:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                win.grab().save(str(args.screenshot), "PNG")
            win.close()
            app.quit()

        QTimer.singleShot(args.smoke_exit_ms, _finish_smoke)

    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
