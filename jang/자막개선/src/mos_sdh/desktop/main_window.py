"""Main Qt6 Broadcast NLE Studio Window integrating all 4 quadrants of MoS-SDH Studio."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import html
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QFont, QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from mos_sdh import __version__
from mos_sdh.config import AnalysisConfig, get_default_cache_dir
from mos_sdh.desktop.dialogs import NewAnalysisDialog, SWERCalculatorDialog
from mos_sdh.desktop.macro_charts import MacroDashboardPanel
from mos_sdh.desktop.theme import SPEAKER_HEX_PALETTE, STRATEGY_KO_LABELS
from mos_sdh.desktop.timeline_widget import MultitrackTimelinePanel
from mos_sdh.desktop.video_canvas import VideoPlayerPanel
from mos_sdh.exporters.report_exporter import export_reports
from mos_sdh.exporters.subtitle_exporter import export_subtitles
from mos_sdh.schemas.document import MoSDocument, SegmentMoS
from mos_sdh.stage5_llm.overstylization_pruner import prune_overstylized_cues


def _fmt_ms(ms: int) -> str:
    tot_s, milli = divmod(max(0, int(ms)), 1000)
    m, s = divmod(tot_s, 60)
    return f"{m:02d}:{s:02d}.{milli:03d}"


class MoSSDHMainWindow(QMainWindow):
    """Pure Native Windows Qt6 Studio for 5-Stage Manner-of-Speech Analysis & Barrier-Free Subtitling."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"MoS-SDH Studio v{__version__} — 시청각 화법(MoS) 배리어프리 자막 워크스테이션")
        self.resize(1460, 920)
        self.setMinimumSize(1120, 720)

        self.current_doc: MoSDocument | None = None
        self.current_out_dir: Path | None = None
        self.selected_segment_id: int | None = None
        self._active_filter_tag: str = "all"
        self._active_speaker_filter: str = ""
        self._suppress_combo_signal: bool = False
        self._suppress_table_signal: bool = False
        self._discovered_projects: list[tuple[str, Path]] = []

        self._build_menus()
        self._build_toolbar()
        self._build_central_workspace()
        self._build_statusbar()

        self.refresh_project_list(auto_load_first=False)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_menus(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("파일(&F)")
        act_new = QAction("새 영상 5-Stage GPU 분석…", self)
        act_new.setShortcut(QKeySequence("Ctrl+O"))
        act_new.triggered.connect(self.open_new_analysis_dialog)
        file_menu.addAction(act_new)

        act_open_json = QAction("기존 분석 리포트(mos_report.json) 열기…", self)
        act_open_json.triggered.connect(self.open_external_report_dialog)
        file_menu.addAction(act_open_json)

        file_menu.addSeparator()
        for label, fmt in (
            ("표준 배리어프리 자막 (.SRT) 다른 이름으로 저장…", "srt"),
            ("웹 표준 자막 (.VTT) 다른 이름으로 저장…", "vtt"),
            ("시각 스타일 자막 (.ASS) 다른 이름으로 저장…", "ass"),
            ("정량 진단 리포트 (.MD) 다른 이름으로 저장…", "md"),
            ("통합 스키마 (.JSON) 다른 이름으로 저장…", "json"),
        ):
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, f=fmt: self.save_export_as(f))
            file_menu.addAction(act)

        file_menu.addSeparator()
        act_open_dir = QAction("현재 산출물 폴더 탐색기에서 열기", self)
        act_open_dir.triggered.connect(self.open_current_output_folder)
        file_menu.addAction(act_open_dir)

        file_menu.addSeparator()
        act_save_manual = QAction("선택 세그먼트 수동 수정 저장", self)
        act_save_manual.setShortcut(QKeySequence("Ctrl+S"))
        act_save_manual.triggered.connect(self._on_manual_save_clicked)
        file_menu.addAction(act_save_manual)

        act_quit = QAction("종료(&X)", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        tools_menu = mb.addMenu("도구(&T)")
        act_swer = QAction("한국어 공백 정규화 sWER 계산기 (Bang et al., 2020)…", self)
        act_swer.triggered.connect(self.open_swer_dialog)
        tools_menu.addAction(act_swer)

        act_play = QAction("비디오 재생 / 일시정지 (Space)", self)
        act_play.triggered.connect(lambda: self.video_panel.toggle_play_pause())
        tools_menu.addAction(act_play)

    def _build_toolbar(self) -> None:
        tb = QToolBar("메인 스튜디오 툴바", self)
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        lbl_proj = QLabel("프로젝트:")
        lbl_proj.setObjectName("MutedLabel")
        tb.addWidget(lbl_proj)

        self.cmb_projects = QComboBox()
        self.cmb_projects.setMinimumWidth(210)
        self.cmb_projects.currentIndexChanged.connect(self._on_project_combo_changed)
        tb.addWidget(self.cmb_projects)

        self.btn_new = QPushButton("+ 새 영상 5-Stage 분석")
        self.btn_new.setObjectName("PrimaryBtn")
        self.btn_new.clicked.connect(self.open_new_analysis_dialog)
        tb.addWidget(self.btn_new)

        self.btn_open_report = QPushButton("기존 리포트 열기…")
        self.btn_open_report.clicked.connect(self.open_external_report_dialog)
        tb.addWidget(self.btn_open_report)

        btn_swer = QPushButton("sWER 계산기")
        btn_swer.clicked.connect(self.open_swer_dialog)
        tb.addWidget(btn_swer)

        tb.addSeparator()

        for text, fmt, obj_name in (
            (".SRT 저장", "srt", ""),
            (".VTT 저장", "vtt", ""),
            (".ASS 스타일 저장", "ass", "AssBtn"),
            ("리포트(.MD)", "md", ""),
        ):
            b = QPushButton(text)
            if obj_name:
                b.setObjectName(obj_name)
            b.clicked.connect(lambda checked=False, f=fmt: self.save_export_as(f))
            tb.addWidget(b)

        btn_dir = QPushButton("폴더 열기")
        btn_dir.clicked.connect(self.open_current_output_folder)
        tb.addWidget(btn_dir)

    def _build_central_workspace(self) -> None:
        self.main_vsplit = QSplitter(Qt.Orientation.Vertical, self)
        self.main_vsplit.setChildrenCollapsible(False)
        self.main_vsplit.setHandleWidth(5)

        # 1. Upper Split: Quadrant 1 (Video + Live ASS/OCR) | Quadrant 2 (Relation Graph + KPIs)
        self.top_hsplit = QSplitter(Qt.Orientation.Horizontal)
        self.top_hsplit.setChildrenCollapsible(False)
        self.top_hsplit.setHandleWidth(5)

        self.video_panel = VideoPlayerPanel(self)
        self.macro_panel = MacroDashboardPanel(self)
        self.top_hsplit.addWidget(self.video_panel)
        self.top_hsplit.addWidget(self.macro_panel)
        self.top_hsplit.setStretchFactor(0, 54)
        self.top_hsplit.setStretchFactor(1, 46)

        # 2. Middle Band: 4-Track Synchronized QPainter Timeline
        self.timeline_panel = MultitrackTimelinePanel(self)
        self.timeline_panel.setMinimumHeight(138)
        self.timeline_panel.setMaximumHeight(168)

        # 3. Lower Split: Bottom-Left Diff Table | Bottom-Right Multi-Candidate & Evidence Inspector
        self.bot_hsplit = QSplitter(Qt.Orientation.Horizontal)
        self.bot_hsplit.setChildrenCollapsible(False)
        self.bot_hsplit.setHandleWidth(5)

        self.diff_panel = self._create_diff_table_panel()
        self.inspector_panel = self._create_inspector_panel()
        self.bot_hsplit.addWidget(self.diff_panel)
        self.bot_hsplit.addWidget(self.inspector_panel)
        self.bot_hsplit.setStretchFactor(0, 55)
        self.bot_hsplit.setStretchFactor(1, 45)

        self.main_vsplit.addWidget(self.top_hsplit)
        self.main_vsplit.addWidget(self.timeline_panel)
        self.main_vsplit.addWidget(self.bot_hsplit)
        self.main_vsplit.setSizes([380, 145, 345])

        container = QWidget(self)
        c_lay = QVBoxLayout(container)
        c_lay.setContentsMargins(6, 6, 6, 6)
        c_lay.setSpacing(0)
        c_lay.addWidget(self.main_vsplit)
        self.setCentralWidget(container)

        # Cross-panel signal wiring
        self.video_panel.time_updated.connect(self.timeline_panel.canvas.set_playhead_ms)
        self.timeline_panel.canvas.seek_requested.connect(self.video_panel.seek_ms)
        self.timeline_panel.canvas.segment_clicked.connect(lambda sid: self.select_segment(sid, seek_video=True))
        self.macro_panel.scene_seek_requested.connect(self.video_panel.seek_ms)
        self.macro_panel.speaker_filter_toggled.connect(self._on_speaker_filter_toggled)
        self.macro_panel.prune_requested.connect(self.apply_pruning)

    def _create_diff_table_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setObjectName("StudioPanel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 7, 8, 7)
        lay.setSpacing(6)

        # Header + Filter Bar
        flt_row = QHBoxLayout()
        flt_row.setSpacing(4)
        title = QLabel("세그먼트 Diff 비교 · 화법 필터")
        title.setObjectName("PanelTitle")
        flt_row.addWidget(title)
        flt_row.addSpacing(6)

        self.filter_btn_group = QButtonGroup(self)
        self.filter_btn_group.setExclusive(True)
        filter_specs = [
            ("all", "전체"),
            ("irony", "⚡ 반어법"),
            ("mismatch", "🔴 화계오류"),
            ("shift", "🟠 극적전환"),
            ("ocr", "⬆ OCR회피"),
            ("emphasis", "★ 강한강조"),
        ]
        for key, txt in filter_specs:
            btn = QPushButton(txt)
            btn.setCheckable(True)
            btn.setProperty("filter_key", key)
            btn.setStyleSheet("padding:3px 8px;font-size:11px;")
            if key == "all":
                btn.setChecked(True)
            btn.clicked.connect(lambda checked=False, k=key: self._set_tag_filter(k))
            self.filter_btn_group.addButton(btn)
            flt_row.addWidget(btn)

        flt_row.addStretch(1)
        self.inp_search = QLineEdit()
        self.inp_search.setPlaceholderText("대사·화자·진단 검색…")
        self.inp_search.setFixedWidth(165)
        self.inp_search.textChanged.connect(lambda _: self._populate_diff_table())
        flt_row.addWidget(self.inp_search)
        lay.addLayout(flt_row)

        # 7-Column Diff Table
        self.diff_table = QTableWidget(0, 7, panel)
        self.diff_table.setHorizontalHeaderLabels(
            ["#", "시간", "화자·위치", "기존·원음 대사", "채택된 배리어프리 자막", "MoS 진단", "CPS"]
        )
        self.diff_table.setAlternatingRowColors(True)
        self.diff_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.diff_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.diff_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.diff_table.verticalHeader().setVisible(False)
        self.diff_table.verticalHeader().setDefaultSectionSize(32)

        hh = self.diff_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        self.diff_table.setColumnWidth(0, 36)
        self.diff_table.setColumnWidth(1, 76)
        self.diff_table.setColumnWidth(2, 92)
        self.diff_table.setColumnWidth(5, 138)
        self.diff_table.setColumnWidth(6, 54)

        self.diff_table.itemSelectionChanged.connect(self._on_diff_row_selected)
        lay.addWidget(self.diff_table, stretch=1)
        return panel

    def _create_inspector_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setObjectName("StudioPanel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 7, 8, 7)
        lay.setSpacing(6)

        # Header + Global Speaker Rename
        hdr = QHBoxLayout()
        self.lbl_insp_title = QLabel("세그먼트 정밀 인스펙터 · 다중 후보 채택")
        self.lbl_insp_title.setObjectName("PanelTitle")
        hdr.addWidget(self.lbl_insp_title)
        hdr.addStretch(1)

        lbl_spk = QLabel("화자명:")
        lbl_spk.setObjectName("MutedLabel")
        self.inp_speaker = QLineEdit()
        self.inp_speaker.setFixedWidth(96)
        self.inp_speaker.setPlaceholderText("예: 민수")
        self.btn_rename_spk = QPushButton("일괄 변경")
        self.btn_rename_spk.clicked.connect(self._on_rename_speaker_clicked)
        hdr.addWidget(lbl_spk)
        hdr.addWidget(self.inp_speaker)
        hdr.addWidget(self.btn_rename_spk)
        lay.addLayout(hdr)

        # Scrollable Inspector Body
        scroll = QScrollArea(panel)
        scroll.setWidgetResizable(True)
        body = QWidget()
        self.insp_body_lay = QVBoxLayout(body)
        self.insp_body_lay.setContentsMargins(2, 2, 4, 2)
        self.insp_body_lay.setSpacing(7)

        # 1. 2x2 Quantitative MoS Evidence Grid
        ev_grid = QGridLayout()
        ev_grid.setSpacing(5)
        self._ev_labels: list[QLabel] = []
        ev_titles = [
            "시각 ASD · 표정 · OCR 공간",
            "Trimodal CmGI 정서 · 반어법",
            "한국어 3축 경어법 · 6화계(CAPE)",
            "운율(F0·dB·Z) · <toe_k> 강조 단어",
        ]
        for idx, et in enumerate(ev_titles):
            card = QFrame()
            card.setObjectName("SubCard")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(8, 6, 8, 6)
            cl.setSpacing(3)
            t_lbl = QLabel(et)
            t_lbl.setStyleSheet("color:#93C5FD;font-size:11px;font-weight:bold;")
            v_lbl = QLabel("-")
            v_lbl.setWordWrap(True)
            v_lbl.setTextFormat(Qt.TextFormat.RichText)
            v_lbl.setStyleSheet("font-size:11px;line-height:1.35;")
            cl.addWidget(t_lbl)
            cl.addWidget(v_lbl)
            self._ev_labels.append(v_lbl)
            ev_grid.addWidget(card, idx // 2, idx % 2)
        self.insp_body_lay.addLayout(ev_grid)

        # 2. Multi-Candidate Container
        lbl_cand_hdr = QLabel("Stage 5 AI 다중 자막 후보 (de Rivero et al. C_i 의미 가드 · 조화 점수 O_i 랭킹)")
        lbl_cand_hdr.setStyleSheet("color:#CBD5E1;font-weight:bold;font-size:11px;margin-top:2px;")
        self.insp_body_lay.addWidget(lbl_cand_hdr)

        self.cand_container = QWidget()
        self.cand_vlay = QVBoxLayout(self.cand_container)
        self.cand_vlay.setContentsMargins(0, 0, 0, 0)
        self.cand_vlay.setSpacing(6)
        self.insp_body_lay.addWidget(self.cand_container)

        # 3. Manual .SRT / .ASS Fine Editor
        man_card = QFrame()
        man_card.setObjectName("SubCard")
        ml = QVBoxLayout(man_card)
        ml.setContentsMargins(8, 7, 8, 7)
        ml.setSpacing(5)
        mh = QHBoxLayout()
        mh.addWidget(QLabel("수동 자막 미세 편집 (.SRT 표준 지문 / .ASS 시각 태그)"))
        mh.addStretch(1)
        self.btn_manual_save = QPushButton("수동 수정 저장 (Ctrl+S)")
        self.btn_manual_save.setObjectName("AccentBtn")
        self.btn_manual_save.clicked.connect(self._on_manual_save_clicked)
        mh.addWidget(self.btn_manual_save)
        ml.addLayout(mh)

        r_srt = QHBoxLayout()
        l_s = QLabel(".SRT:")
        l_s.setFixedWidth(38)
        l_s.setObjectName("MutedLabel")
        self.inp_manual_srt = QLineEdit()
        r_srt.addWidget(l_s)
        r_srt.addWidget(self.inp_manual_srt, stretch=1)
        ml.addLayout(r_srt)

        r_ass = QHBoxLayout()
        l_a = QLabel(".ASS:")
        l_a.setFixedWidth(38)
        l_a.setObjectName("MutedLabel")
        self.inp_manual_ass = QLineEdit()
        r_ass.addWidget(l_a)
        r_ass.addWidget(self.inp_manual_ass, stretch=1)
        ml.addLayout(r_ass)

        self.insp_body_lay.addWidget(man_card)
        self.insp_body_lay.addStretch(1)

        scroll.setWidget(body)
        lay.addWidget(scroll, stretch=1)
        return panel

    def _build_statusbar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self.lbl_status = QLabel("준비 완료 — 로컬 순수 네이티브 Qt6 데스크톱 모드 (웹 브라우저·외부 서버 불필요)")
        sb.addWidget(self.lbl_status, 1)
        self.lbl_gpu_badge = QLabel(self._detect_gpu_badge_text())
        self.lbl_gpu_badge.setObjectName("GpuBadge")
        sb.addPermanentWidget(self.lbl_gpu_badge)

    # ------------------------------------------------------------------
    # Project Discovery & Loading
    # ------------------------------------------------------------------
    def _detect_gpu_badge_text(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return f"GPU: {name} · {vram:.1f} GB · CUDA 가속"
        except Exception:
            pass
        return "CPU 폴백 모드"

    def refresh_project_list(self, auto_load_first: bool = False) -> None:
        found: list[tuple[str, Path]] = []
        seen_paths: set[Path] = set()

        search_roots = [Path("./output"), get_default_cache_dir() / "projects"]
        for root in search_roots:
            if not root.exists():
                continue
            for rep in root.rglob("mos_report.json"):
                resolved = rep.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                parent_name = rep.parent.name
                label = f"{parent_name} ({rep.parent.parent.name}/{parent_name})"
                found.append((label, resolved))

        found.sort(key=lambda item: item[0])
        self._discovered_projects = found

        self._suppress_combo_signal = True
        self.cmb_projects.clear()
        self.cmb_projects.addItem("프로젝트 선택…", "")
        for lbl, p in self._discovered_projects:
            self.cmb_projects.addItem(lbl, str(p))
        self._suppress_combo_signal = False

        if auto_load_first and self._discovered_projects:
            self.load_project_from_report(self._discovered_projects[0][1])

    def load_project_from_report(self, report_path: Path) -> None:
        if not report_path.exists():
            QMessageBox.warning(self, "파일 없음", f"리포트 파일을 찾을 수 없습니다:\n{report_path}")
            return
        doc = MoSDocument.model_validate_json(report_path.read_text(encoding="utf-8"))
        self.load_document(doc, report_path.parent, label=report_path.parent.name)

    def load_document(self, doc: MoSDocument, out_dir: Path, label: str | None = None) -> None:
        self.current_doc = doc
        self.current_out_dir = out_dir.resolve()
        self.current_out_dir.mkdir(parents=True, exist_ok=True)
        self._active_speaker_filter = ""

        # Sync project selector combo
        rep_str = str((self.current_out_dir / "mos_report.json").resolve())
        self._suppress_combo_signal = True
        existing_idx = self.cmb_projects.findData(rep_str)
        if existing_idx < 0:
            disp = label or self.current_out_dir.name
            self.cmb_projects.insertItem(0, disp, rep_str)
            self.cmb_projects.setCurrentIndex(0)
        else:
            self.cmb_projects.setCurrentIndex(existing_idx)
        self._suppress_combo_signal = False

        # Populate all 4 quadrants
        self.video_panel.load_project_media(doc)
        self.macro_panel.update_from_document(doc, active_speaker=None)
        first_seg_id = doc.segments[0].segment_id if doc.segments else None
        self.timeline_panel.canvas.set_document(doc, selected_seg_id=first_seg_id)
        self._populate_diff_table()

        if first_seg_id is not None:
            # Prefer selecting an impactful segment if present on first load, else #1
            self.select_segment(first_seg_id, seek_video=True)

        self.lbl_status.setText(
            f"프로젝트 로드 완료: {Path(doc.input_media_path).name}  |  "
            f"{len(doc.scenes)}개 씬 · {len(doc.segments)}개 세그먼트  |  산출물 경로: {self.current_out_dir}"
        )

    # ------------------------------------------------------------------
    # Segment Filtering, Selection & Editing
    # ------------------------------------------------------------------
    def _set_tag_filter(self, key: str) -> None:
        self._active_filter_tag = key
        self._populate_diff_table()

    def _on_speaker_filter_toggled(self, speaker_id: str) -> None:
        if speaker_id and self._active_speaker_filter == speaker_id:
            self._active_speaker_filter = ""
        else:
            self._active_speaker_filter = speaker_id
        if self.current_doc is not None:
            self.macro_panel.update_from_document(
                self.current_doc,
                active_speaker=self._active_speaker_filter or None,
            )
        self._populate_diff_table()

    def _segment_matches_filters(self, seg: SegmentMoS) -> bool:
        if self._active_speaker_filter and seg.speaker_id != self._active_speaker_filter:
            return False

        tag = self._active_filter_tag
        if tag == "irony" and not seg.emotion_irony.irony_detected:
            return False
        if tag == "mismatch" and seg.honorific.consistency_status != "mismatch_error":
            return False
        if tag == "shift" and seg.honorific.consistency_status != "dramatic_shift":
            return False
        if tag == "ocr" and not seg.visual.avoid_bottom_region:
            return False
        if tag == "emphasis" and max((w.emphasis_level for w in seg.words), default=0) < 3:
            return False

        q = self.inp_search.text().strip().lower()
        if q:
            haystack = " ".join(
                [
                    seg.speaker_id,
                    seg.raw_transcript,
                    seg.existing_subtitle or "",
                    seg.sdh_srt_text,
                    seg.emotion_irony.dominant_emotion,
                    seg.honorific.speech_level,
                ]
            ).lower()
            if q not in haystack:
                return False
        return True

    def _populate_diff_table(self) -> None:
        self._suppress_table_signal = True
        self.diff_table.setRowCount(0)
        if self.current_doc is None:
            self._suppress_table_signal = False
            return

        speakers = sorted({s.speaker_id for s in self.current_doc.segments})
        filtered = [s for s in self.current_doc.segments if self._segment_matches_filters(s)]
        self.diff_table.setRowCount(len(filtered))

        sel_row_to_restore = -1
        for row_idx, seg in enumerate(filtered):
            if seg.segment_id == self.selected_segment_id:
                sel_row_to_restore = row_idx

            it_id = QTableWidgetItem(f"#{seg.segment_id}")
            it_id.setData(Qt.ItemDataRole.UserRole, seg.segment_id)
            it_id.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            it_time = QTableWidgetItem(_fmt_ms(seg.start_ms)[:8])
            it_time.setFont(QFont("Consolas", 9))

            pos_tag = "화면밖" if seg.visual.on_screen_speaker is False else seg.visual.speaker_zone
            it_spk = QTableWidgetItem(f"{seg.speaker_id} ({pos_tag})")
            spk_idx = speakers.index(seg.speaker_id) if seg.speaker_id in speakers else 0
            it_spk.setForeground(QColor(SPEAKER_HEX_PALETTE[spk_idx % len(SPEAKER_HEX_PALETTE)]))

            orig_txt = seg.existing_subtitle if seg.existing_subtitle else seg.raw_transcript
            it_orig = QTableWidgetItem(orig_txt)
            it_orig.setToolTip(orig_txt)

            it_sdh = QTableWidgetItem(seg.sdh_srt_text)
            it_sdh.setToolTip(seg.sdh_srt_text)
            it_sdh.setForeground(QColor("#F8FAFC"))

            diag_parts: list[str] = [seg.honorific.speech_level]
            diag_col = "#94A3B8"
            if seg.emotion_irony.irony_detected:
                diag_parts.append("⚡반어")
                diag_col = "#FCD34D"
            if seg.honorific.consistency_status == "mismatch_error":
                diag_parts.append("🔴오류")
                diag_col = "#FDA4AF"
            elif seg.honorific.consistency_status == "dramatic_shift":
                diag_parts.append("🟠전환")
                diag_col = "#FDBA74"
            if seg.visual.avoid_bottom_region:
                diag_parts.append("⬆OCR")
                if diag_col == "#94A3B8":
                    diag_col = "#6EE7B7"
            it_diag = QTableWidgetItem(" · ".join(diag_parts))
            it_diag.setForeground(QColor(diag_col))

            it_cps = QTableWidgetItem(f"{seg.cps:.1f}")
            it_cps.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            it_cps.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            it_cps.setForeground(QColor("#FB7185") if seg.cps_exceeded else QColor("#34D399"))

            for col_i, item in enumerate((it_id, it_time, it_spk, it_orig, it_sdh, it_diag, it_cps)):
                self.diff_table.setItem(row_idx, col_i, item)

        if sel_row_to_restore >= 0:
            self.diff_table.selectRow(sel_row_to_restore)
        self._suppress_table_signal = False

    def _on_diff_row_selected(self) -> None:
        if self._suppress_table_signal:
            return
        items = self.diff_table.selectedItems()
        if not items:
            return
        seg_id = items[0].data(Qt.ItemDataRole.UserRole)
        if seg_id is not None:
            self.select_segment(int(seg_id), seek_video=True)

    def select_segment(self, segment_id: int, seek_video: bool = True) -> None:
        if self.current_doc is None:
            return
        seg = next((s for s in self.current_doc.segments if s.segment_id == segment_id), None)
        if seg is None:
            return

        self.selected_segment_id = segment_id
        self.timeline_panel.canvas.set_selected_segment(segment_id)

        # Highlight matching row in diff_table without re-triggering signal
        self._suppress_table_signal = True
        for r in range(self.diff_table.rowCount()):
            it = self.diff_table.item(r, 0)
            if it and it.data(Qt.ItemDataRole.UserRole) == segment_id:
                self.diff_table.selectRow(r)
                break
        self._suppress_table_signal = False

        if seek_video:
            self.video_panel.seek_ms(seg.start_ms + 35)
        else:
            self.video_panel.refresh_overlay()

        self._render_inspector_for_segment(seg)

    def _render_inspector_for_segment(self, seg: SegmentMoS) -> None:
        self.lbl_insp_title.setText(
            f"세그먼트 #{seg.segment_id} 정밀 인스펙터 ({_fmt_ms(seg.start_ms)} ~ {_fmt_ms(seg.end_ms)})"
        )
        self.inp_speaker.setText(seg.speaker_id)
        self.inp_manual_srt.setText(seg.sdh_srt_text)
        self.inp_manual_ass.setText(seg.sdh_ass_text)

        # 1. Visual ASD / FER / OCR Evidence
        asd_str = (
            "<b style='color:#FBBF24;'>화면 밖 발화 (Off-Screen)</b>"
            if seg.visual.on_screen_speaker is False
            else f"화면 내 발화 (구역: <b>{html.escape(seg.visual.speaker_zone)}</b> · 구순 활성 {seg.visual.lip_activity_score:.2f})"
        )
        ocr_str = (
            f"<span style='color:#6EE7B7;'>하단 OCR 충돌 감지 → 상단 {{\\an8}} 회피 ({html.escape(', '.join(seg.visual.ocr_texts))})</span>"
            if seg.visual.avoid_bottom_region
            else (html.escape(", ".join(seg.visual.ocr_texts)) if seg.visual.ocr_texts else "하단 자막 충돌 없음")
        )
        self._ev_labels[0].setText(
            f"{asd_str}<br/>표정(FER): <b>{html.escape(seg.visual.facial_expression or '미감지')}</b><br/>OCR: {ocr_str}"
        )

        # 2. Trimodal CmGI Emotion & Irony Evidence
        ei = seg.emotion_irony
        gates = ei.gate_weights_mean
        g_txt = (
            f"T={gates.get('text', 0.33):.2f} · A={gates.get('audio', 0.33):.2f} · V={gates.get('visual', 0.33):.2f}"
            if gates
            else "기본 균등 게이트"
        )
        irony_badge = (
            f"<b style='color:#FCD34D;'>⚡ 반어법 감지 (I={ei.irony_score:.2f})</b>"
            if ei.irony_detected
            else f"일치 (I={ei.irony_score:.2f})"
        )
        self._ev_labels[1].setText(
            f"주류 정서: <b>{html.escape(ei.dominant_emotion)}</b> | {irony_badge}<br/>"
            f"어휘 극성 {ei.lexical_polarity:+.2f} vs 시청각 극성 {ei.audiovisual_polarity:+.2f}<br/>"
            f"CmGI 모달리티 게이트: <span style='color:#94A3B8;'>{g_txt}</span>"
        )

        # 3. Korean 3-Axis Honorifics & 6-Speech-Level Consistency (CAPE)
        hon = seg.honorific
        st_map = {
            "consistent": "<span style='color:#34D399;'>문맥 일치 (Consistent)</span>",
            "mismatch_error": "<b style='color:#FB7185;'>🔴 화계 오인식 의심 (CAPE 교정 권고)</b>",
            "dramatic_shift": "<b style='color:#FDBA74;'>🟠 고각성 의도적 화계 전환</b>",
            "insufficient_context": "<span style='color:#94A3B8;'>초기 맥락 수집</span>",
        }
        cape_note = (
            f"<br/>교정 제안: <b style='color:#38BDF8;'>{html.escape(hon.suggested_correction)}</b>"
            if hon.suggested_correction
            else ""
        )
        self._ev_labels[2].setText(
            f"화계: <b>{html.escape(hon.speech_level)}</b> (기대 화계: {html.escape(hon.expected_speech_level or '-')})<br/>"
            f"주체높임: {'O' if hon.subject_honorific else 'X'} · 객체높임: {'O' if hon.object_honorific else 'X'} | "
            f"{st_map.get(hon.consistency_status, hon.consistency_status)}{cape_note}"
        )

        # 4. Prosody & Word-Level <toe_k> Emphasis Chips
        word_chips: list[str] = []
        for w in seg.words:
            ew = html.escape(w.text)
            if w.emphasis_level >= 3:
                word_chips.append(
                    f"<span style='background:#451A03;color:#FDE047;padding:1px 4px;border-radius:3px;font-weight:bold;'>"
                    f"{ew}(★e{w.emphasis_level})</span>"
                )
            elif w.emphasis_level >= 1:
                word_chips.append(f"<span style='color:#93C5FD;font-weight:bold;'>{ew}(e{w.emphasis_level})</span>")
            else:
                word_chips.append(ew)
        chips_html = " ".join(word_chips) if word_chips else html.escape(seg.raw_transcript)
        self._ev_labels[3].setText(
            f"평균 F0: <b>{seg.f0_mean_hz:.1f} Hz</b> · 에너지: <b>{seg.energy_mean_db:.1f} dB</b> · "
            f"각성 Z: <b>{seg.arousal_zscore:+.2f}</b><br/>{chips_html}"
        )

        # Rebuild Multi-Candidate Cards
        while self.cand_vlay.count():
            item = self.cand_vlay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for cand in seg.candidates:
            c_frame = QFrame()
            c_frame.setObjectName("CandCardSelected" if cand.selected else "CandCard")
            cl = QVBoxLayout(c_frame)
            cl.setContentsMargins(9, 7, 9, 7)
            cl.setSpacing(4)

            top_r = QHBoxLayout()
            strat_title = STRATEGY_KO_LABELS.get(cand.strategy, cand.strategy)
            lbl_st = QLabel(f"후보 #{cand.candidate_id} · {strat_title}")
            lbl_st.setStyleSheet("font-weight:bold;color:#F8FAFC;font-size:11px;")
            top_r.addWidget(lbl_st)
            top_r.addStretch(1)

            guard_txt = "가드 통과" if cand.passed_guard else "C_i 기준 미달"
            guard_col = "#34D399" if cand.passed_guard else "#FB7185"
            lbl_g = QLabel(guard_txt)
            lbl_g.setStyleSheet(f"color:{guard_col};font-size:10px;font-weight:bold;")
            top_r.addWidget(lbl_g)

            btn_pick = QPushButton("✓ 현재 채택됨" if cand.selected else "이 후보 채택")
            if cand.selected:
                btn_pick.setEnabled(False)
            else:
                btn_pick.setObjectName("PrimaryBtn")
                btn_pick.clicked.connect(
                    lambda checked=False, s_id=seg.segment_id, c_id=cand.candidate_id: self.select_candidate(s_id, c_id)
                )
            btn_pick.setStyleSheet("padding:3px 9px;font-size:11px;")
            top_r.addWidget(btn_pick)
            cl.addLayout(top_r)

            lbl_srt = QLabel(cand.srt_text)
            lbl_srt.setWordWrap(True)
            lbl_srt.setStyleSheet("color:#FFFFFF;font-size:12px;font-weight:600;padding:2px 0;")
            cl.addWidget(lbl_srt)

            # Score Bars Row (C_i, S_MoS, O_i)
            sc_row = QHBoxLayout()
            sc_row.setSpacing(8)
            for m_name, m_val in (
                ("의미 보존 C_i", cand.semantic_ci),
                ("화법 충실도 S_MoS", cand.mos_fidelity),
                ("조화 점수 O_i", cand.harmonic_oi),
            ):
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(int(round(min(1.0, max(0.0, m_val)) * 100)))
                bar.setFormat(f"{m_name}: {m_val:.3f}")
                bar.setFixedHeight(14)
                sc_row.addWidget(bar)
            cl.addLayout(sc_row)

            self.cand_vlay.addWidget(c_frame)

    def select_candidate(self, segment_id: int, candidate_id: int) -> None:
        if self.current_doc is None or self.current_out_dir is None:
            return
        seg = next((s for s in self.current_doc.segments if s.segment_id == segment_id), None)
        if seg is None:
            return
        target = next((c for c in seg.candidates if c.candidate_id == candidate_id), None)
        if target is None:
            return

        for c in seg.candidates:
            c.selected = c.candidate_id == candidate_id

        seg.sdh_srt_text = target.srt_text
        seg.sdh_ass_text = target.ass_text
        seg.stage_direction_pruned = False

        self._recompute_metrics_and_persist()
        self._populate_diff_table()
        self._render_inspector_for_segment(seg)
        self.video_panel.refresh_overlay()
        self.macro_panel.update_from_document(self.current_doc, active_speaker=self._active_speaker_filter or None)
        self.lbl_status.setText(
            f"세그먼트 #{segment_id} → 후보 #{candidate_id} 채택 완료 및 .SRT/.VTT/.ASS 자동 갱신 저장됨"
        )

    def _on_manual_save_clicked(self) -> None:
        if self.current_doc is None or self.selected_segment_id is None:
            return
        new_srt = self.inp_manual_srt.text().strip()
        new_ass = self.inp_manual_ass.text().strip()
        if not new_srt:
            QMessageBox.warning(self, "빈 자막", ".SRT 자막 텍스트를 비워둘 수 없습니다.")
            return
        self.apply_manual_edit(self.selected_segment_id, new_srt, new_ass or new_srt)

    def apply_manual_edit(self, segment_id: int, new_srt: str, new_ass: str) -> None:
        if self.current_doc is None:
            return
        seg = next((s for s in self.current_doc.segments if s.segment_id == segment_id), None)
        if seg is None:
            return
        seg.sdh_srt_text = new_srt
        seg.sdh_ass_text = new_ass
        self._recompute_metrics_and_persist()
        self._populate_diff_table()
        self._render_inspector_for_segment(seg)
        self.video_panel.refresh_overlay()
        self.macro_panel.update_from_document(self.current_doc, active_speaker=self._active_speaker_filter or None)
        self.lbl_status.setText(f"세그먼트 #{segment_id} 수동 자막 편집이 저장되어 .SRT/.VTT/.ASS 파일에 반영되었습니다.")

    def _on_rename_speaker_clicked(self) -> None:
        if self.current_doc is None or self.selected_segment_id is None:
            return
        seg = next((s for s in self.current_doc.segments if s.segment_id == self.selected_segment_id), None)
        if seg is None:
            return
        new_name = self.inp_speaker.text().strip()
        if not new_name or new_name == seg.speaker_id:
            return
        self.rename_speaker_globally(seg.speaker_id, new_name)

    def rename_speaker_globally(self, old_speaker: str, new_speaker: str) -> None:
        if self.current_doc is None:
            return
        for seg in self.current_doc.segments:
            if seg.speaker_id == old_speaker:
                seg.speaker_id = new_speaker
                seg.sdh_srt_text = seg.sdh_srt_text.replace(f"[{old_speaker}", f"[{new_speaker}", 1)
                seg.sdh_ass_text = seg.sdh_ass_text.replace(f"[{old_speaker}", f"[{new_speaker}", 1)
                for cand in seg.candidates:
                    cand.srt_text = cand.srt_text.replace(f"[{old_speaker}", f"[{new_speaker}", 1)
                    cand.ass_text = cand.ass_text.replace(f"[{old_speaker}", f"[{new_speaker}", 1)

        for sc in self.current_doc.scenes:
            for rel in sc.inferred_relations:
                if rel.from_speaker == old_speaker:
                    rel.from_speaker = new_speaker
                if rel.to_speaker == old_speaker:
                    rel.to_speaker = new_speaker

        if self._active_speaker_filter == old_speaker:
            self._active_speaker_filter = new_speaker

        self._recompute_metrics_and_persist()
        self._populate_diff_table()
        self.timeline_panel.canvas.update()
        self.video_panel.refresh_overlay()
        self.macro_panel.update_from_document(self.current_doc, active_speaker=self._active_speaker_filter or None)
        if self.selected_segment_id is not None:
            cur = next((s for s in self.current_doc.segments if s.segment_id == self.selected_segment_id), None)
            if cur is not None:
                self._render_inspector_for_segment(cur)
        self.lbl_status.setText(f"화자명 일괄 변경 완료: '{old_speaker}' → '{new_speaker}'")

    def apply_pruning(self, max_cue_density_ratio: float, recommended_max_cps: float) -> None:
        if self.current_doc is None:
            return
        # Restore selected candidate baseline first so raising the slider also restores previously pruned cues
        for seg in self.current_doc.segments:
            sel = next((c for c in seg.candidates if c.selected), None)
            if sel is not None:
                seg.sdh_srt_text = sel.srt_text
                seg.sdh_ass_text = sel.ass_text
                seg.stage_direction_pruned = False

        cfg = AnalysisConfig(
            max_cue_density_ratio=max_cue_density_ratio,
            recommended_max_cps=recommended_max_cps,
        )
        prune_overstylized_cues(self.current_doc.segments, cfg)
        self._recompute_metrics_and_persist(recommended_max_cps=recommended_max_cps)
        self._populate_diff_table()
        self.video_panel.refresh_overlay()
        self.macro_panel.update_from_document(self.current_doc, active_speaker=self._active_speaker_filter or None)
        if self.selected_segment_id is not None:
            cur = next((s for s in self.current_doc.segments if s.segment_id == self.selected_segment_id), None)
            if cur is not None:
                self._render_inspector_for_segment(cur)
        self.lbl_status.setText(
            f"인지 피로 억제 기준 재적용 완료 (지문 상한 {max_cue_density_ratio * 100:.0f}% · 권장 최대 {recommended_max_cps:.0f} CPS)"
        )

    def _recompute_metrics_and_persist(self, recommended_max_cps: float = 14.0) -> dict[str, Path]:
        if self.current_doc is None or self.current_out_dir is None:
            return {}
        doc = self.current_doc
        if doc.segments:
            for seg in doc.segments:
                dur_s = max(0.5, (seg.end_ms - seg.start_ms) / 1000.0)
                vis_chars = len(re.sub(r"\s+", "", seg.sdh_srt_text or seg.raw_transcript))
                seg.cps = round(vis_chars / dur_s, 2)
                seg.cps_exceeded = seg.cps > recommended_max_cps

            valid_cands = [
                c
                for s in doc.segments
                for c in s.candidates
                if c.selected
            ]
            if valid_cands:
                doc.metrics.mean_semantic_ci = round(sum(c.semantic_ci for c in valid_cands) / len(valid_cands), 4)
                doc.metrics.mean_harmonic_oi = round(sum(c.harmonic_oi for c in valid_cands) / len(valid_cands), 4)

            doc.metrics.mean_cps = round(sum(s.cps for s in doc.segments) / len(doc.segments), 2)
            cps_ok = sum(1 for s in doc.segments if not s.cps_exceeded)
            doc.metrics.cps_compliance_rate = round((cps_ok / len(doc.segments)) * 100.0, 2)
            has_dir = sum(1 for s in doc.segments if "(" in s.sdh_srt_text)
            doc.metrics.cue_attachment_ratio = round(has_dir / len(doc.segments), 4)
            doc.metrics.speaker_count = len({s.speaker_id for s in doc.segments})

        sub_files = export_subtitles(doc, self.current_out_dir)
        rep_files = export_reports(doc, self.current_out_dir)
        return {**sub_files, **rep_files}

    # ------------------------------------------------------------------
    # Toolbar / Dialog Actions & Native File Exports
    # ------------------------------------------------------------------
    def _on_project_combo_changed(self, index: int) -> None:
        if self._suppress_combo_signal or index < 0:
            return
        rep_str = self.cmb_projects.itemData(index)
        if rep_str:
            self.load_project_from_report(Path(str(rep_str)))

    def open_new_analysis_dialog(self) -> None:
        dlg = NewAnalysisDialog(self)
        if dlg.exec() == NewAnalysisDialog.DialogCode.Accepted and dlg.result_doc and dlg.result_out_dir:
            self.refresh_project_list(auto_load_first=False)
            self.load_document(dlg.result_doc, dlg.result_out_dir)

    def open_external_report_dialog(self) -> None:
        start_dir = str(Path("./output").resolve()) if Path("./output").exists() else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "MoS 분석 리포트 JSON 열기",
            start_dir,
            "MoS Report JSON (mos_report.json *.json)",
        )
        if path:
            self.load_project_from_report(Path(path))

    def open_swer_dialog(self) -> None:
        dlg = SWERCalculatorDialog(self)
        dlg.exec()

    def save_export_as(self, fmt: str) -> None:
        if self.current_doc is None or self.current_out_dir is None:
            QMessageBox.information(self, "안내", "먼저 프로젝트를 로드하거나 새 영상을 분석해 주세요.")
            return

        files = self._recompute_metrics_and_persist()
        spec_map = {
            "srt": ("srt", "prototype_sdh.srt", "SubRip 자막 (*.srt)"),
            "vtt": ("vtt", "prototype_sdh.vtt", "WebVTT 자막 (*.vtt)"),
            "ass": ("ass", "prototype_sdh.ass", "SubStation Alpha 스타일 자막 (*.ass)"),
            "md": ("markdown", "mos_report.md", "Markdown 진단 리포트 (*.md)"),
            "json": ("json", "mos_report.json", "MoS JSON 스키마 (*.json)"),
        }
        key, default_name, flt = spec_map[fmt]
        src_path = files[key]

        dest, _ = QFileDialog.getSaveFileName(
            self,
            f"{default_name} 다른 이름으로 저장",
            str((self.current_out_dir / default_name).resolve()),
            flt,
        )
        if dest:
            dest_p = Path(dest)
            if dest_p.resolve() != src_path.resolve():
                shutil.copy2(src_path, dest_p)
            self.lbl_status.setText(f"파일 저장 완료: {dest_p}")

    def open_current_output_folder(self) -> None:
        target = self.current_out_dir or Path("./output").resolve()
        target.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(str(target))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Keyboard Ergonomics
    # ------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        focused = QApplication.focusWidget()
        if event.key() == Qt.Key.Key_Space and not isinstance(focused, (QLineEdit, QPlainTextEdit)):
            self.video_panel.toggle_play_pause()
            event.accept()
            return
        super().keyPressEvent(event)
