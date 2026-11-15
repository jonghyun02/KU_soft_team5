"""Background QThread worker executing the 5-Stage GPU MoS pipeline without blocking the Qt6 UI."""

from __future__ import annotations

from pathlib import Path
import traceback
from PySide6.QtCore import QThread, Signal

from mos_sdh.config import AnalysisConfig
from mos_sdh.exporters.report_exporter import export_reports
from mos_sdh.exporters.subtitle_exporter import export_subtitles
from mos_sdh.pipeline import run_full_pipeline


class PipelineWorker(QThread):
    """Run Stages 1 through 5 sequentially on local GPU in a dedicated worker thread."""

    status_changed = Signal(str)
    finished_ok = Signal(object, object)  # (MoSDocument, Path out_dir)
    failed = Signal(str)

    def __init__(
        self,
        input_media: Path,
        out_dir: Path,
        config: AnalysisConfig,
        existing_subtitle: Path | None = None,
    ) -> None:
        super().__init__()
        self.input_media = input_media
        self.out_dir = out_dir
        self.config = config
        self.existing_subtitle = existing_subtitle

    def run(self) -> None:
        try:
            self.status_changed.emit(f"5-Stage GPU 화법(MoS) 분석 실행 중: {self.input_media.name} …")
            doc = run_full_pipeline(
                input_media=self.input_media,
                config=self.config,
                existing_subtitle=self.existing_subtitle,
            )
            self.status_changed.emit("배리어프리 자막(.srt/.vtt/.ass) 및 진단 리포트 저장 중…")
            export_subtitles(doc, self.out_dir)
            export_reports(doc, self.out_dir)
            self.finished_ok.emit(doc, self.out_dir)
        except Exception as exc:
            tb = traceback.format_exc(limit=4)
            self.failed.emit(f"{exc}\n\n{tb}")
