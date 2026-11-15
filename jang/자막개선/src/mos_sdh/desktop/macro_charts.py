"""Quadrant 2: Native QPainter 6-Speech-Level Relationship Graph, Scene Cards, Score_dist & VRAM Panel."""

from __future__ import annotations

import math
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mos_sdh.desktop.theme import SPEAKER_HEX_PALETTE
from mos_sdh.schemas.document import MoSDocument, SpeakerRelation


def _fmt_ms(ms: int) -> str:
    tot_s, milli = divmod(max(0, int(ms)), 1000)
    m, s = divmod(tot_s, 60)
    return f"{m:02d}:{s:02d}.{milli:03d}"


class RelationGraphWidget(QWidget):
    """Custom QPainter directional graph of speaker social relationships & 6-speech-level priors."""

    speaker_clicked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(145)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._doc: MoSDocument | None = None
        self._active_speaker: str | None = None
        self._node_centers: dict[str, QPointF] = {}
        self._real_speakers: set[str] = set()

    def set_document(self, doc: MoSDocument | None, active_speaker: str | None = None) -> None:
        self._doc = doc
        self._active_speaker = active_speaker
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w = float(self.width())
        h = float(self.height())
        p.fillRect(QRectF(0, 0, w, h), QColor("#0B101B"))
        p.setPen(QPen(QColor("#222E44"), 1.0))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 6.0, 6.0)

        self._node_centers.clear()
        if self._doc is None or not self._doc.segments:
            p.setPen(QColor("#64748B"))
            p.setFont(QFont("Malgun Gothic", 9))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "인물 관계 그래프 대기 중")
            p.end()
            return

        speakers = sorted({s.speaker_id for s in self._doc.segments})
        self._real_speakers = set(speakers)

        relations: list[SpeakerRelation] = []
        for sc in self._doc.scenes:
            for r in sc.inferred_relations:
                if not any(x.from_speaker == r.from_speaker and x.to_speaker == r.to_speaker for x in relations):
                    relations.append(r)

        nodes = list(speakers)
        for r in relations:
            if r.from_speaker not in nodes:
                nodes.append(r.from_speaker)
            if r.to_speaker not in nodes:
                nodes.append(r.to_speaker)
        if len(nodes) == 1:
            nodes.append("대화 상대")

        cx, cy = w * 0.5, h * 0.52
        rx = min(175.0, w * 0.34)
        ry = min(46.0, h * 0.30)

        for idx, name in enumerate(nodes):
            ang = math.pi + (idx * 2.0 * math.pi) / max(1, len(nodes))
            nx = cx + rx * math.cos(ang)
            ny = cy if len(nodes) <= 2 else cy + ry * math.sin(ang)
            self._node_centers[name] = QPointF(nx, ny)

        if not relations:
            dom = self._doc.segments[0].honorific.speech_level
            relations = [
                SpeakerRelation(
                    from_speaker=nodes[0],
                    to_speaker=nodes[1],
                    relation_ko="주류 화계",
                    prior_speech_level=dom,
                )
            ]

        # Curved edges + arrowheads + speech-level badge
        for i, rel in enumerate(relations):
            p1 = self._node_centers.get(rel.from_speaker, QPointF(90, cy))
            p2 = self._node_centers.get(rel.to_speaker, QPointF(w - 90, cy))
            dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
            dist = math.hypot(dx, dy) or 1.0
            pad = 28.0
            sx, sy = p1.x() + (dx / dist) * pad, p1.y() + (dy / dist) * pad
            ex, ey = p2.x() - (dx / dist) * pad, p2.y() - (dy / dist) * pad
            curve = -20.0 if i % 2 == 0 else 20.0
            mx, my = (sx + ex) * 0.5, (sy + ey) * 0.5 + curve

            path = QPainterPath(QPointF(sx, sy))
            path.quadTo(QPointF(mx, my), QPointF(ex, ey))
            p.setPen(QPen(QColor("#38BDF8"), 1.8))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

            # Arrowhead at (ex, ey)
            ang_end = math.atan2(ey - my, ex - mx)
            ah = 7.5
            a1 = QPointF(ex - ah * math.cos(ang_end - 0.45), ey - ah * math.sin(ang_end - 0.45))
            a2 = QPointF(ex - ah * math.cos(ang_end + 0.45), ey - ah * math.sin(ang_end + 0.45))
            p.setBrush(QColor("#38BDF8"))
            p.drawPolygon([QPointF(ex, ey), a1, a2])

            badge_txt = f" {rel.prior_speech_level} · {rel.relation_ko[:12]} "
            p.setFont(QFont("Malgun Gothic", 8, QFont.Weight.Bold))
            bw = p.fontMetrics().horizontalAdvance(badge_txt) + 10
            br = QRectF(mx - bw * 0.5, my - 9.0, bw, 18.0)
            p.setPen(QPen(QColor("#334155"), 1.0))
            p.setBrush(QColor("#0F172A"))
            p.drawRoundedRect(br, 4.0, 4.0)
            p.setPen(QColor("#FDE68A"))
            p.drawText(br, Qt.AlignmentFlag.AlignCenter, badge_txt)

        # Speaker nodes
        p.setFont(QFont("Malgun Gothic", 9, QFont.Weight.Bold))
        for idx, name in enumerate(nodes):
            pt = self._node_centers[name]
            is_real = name in self._real_speakers
            is_sel = name == self._active_speaker
            ring_hex = (
                "#FBBF24"
                if is_sel
                else (SPEAKER_HEX_PALETTE[idx % len(SPEAKER_HEX_PALETTE)] if is_real else "#475569")
            )
            p.setPen(QPen(QColor(ring_hex), 2.8 if is_sel else 2.0))
            p.setBrush(QColor("#162136"))
            p.drawEllipse(pt, 24.0, 24.0)
            p.setPen(QColor("#FFFFFF"))
            p.drawText(QRectF(pt.x() - 32, pt.y() - 12, 64, 24), Qt.AlignmentFlag.AlignCenter, name)

        p.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        for name, center in self._node_centers.items():
            if name in self._real_speakers and math.hypot(pos.x() - center.x(), pos.y() - center.y()) <= 26.0:
                self.speaker_clicked.emit(name)
                return


class ScoreDistChartWidget(QWidget):
    """Custom QPainter 5-bin Score_dist bar chart (Sun et al., 2024 Eq. 3)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(68)
        self._bins: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]

    def set_bins(self, bins: list[float]) -> None:
        self._bins = list(bins[:5]) if len(bins) >= 5 else [0.0] * 5
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w = float(self.width())
        h = float(self.height())
        col_w = w / 5.0
        max_bar_h = h - 20.0

        p.setFont(QFont("Consolas", 8))
        for i, val in enumerate(self._bins):
            cx = i * col_w
            bh = max(4.0, (min(100.0, max(0.0, val)) / 100.0) * max_bar_h)
            bar_r = QRectF(cx + 8.0, h - 17.0 - bh, max(12.0, col_w - 16.0), bh)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#38BDF8") if val > 0 else QColor("#1E293B"))
            p.drawRoundedRect(bar_r, 3.0, 3.0)

            p.setPen(QColor("#94A3B8"))
            p.drawText(
                QRectF(cx, h - 16.0, col_w, 16.0),
                Qt.AlignmentFlag.AlignCenter,
                f"Q{i + 1}({int(round(val))})",
            )
        p.end()


class MacroDashboardPanel(QFrame):
    """Quadrant 2 panel with Tabs: [인물 6화계 관계도 · 씬 맥락] and [거시 품질 · Score_dist · VRAM]."""

    scene_seek_requested = Signal(int)
    speaker_filter_toggled = Signal(str)
    prune_requested = Signal(float, float)  # (max_cue_density_ratio, recommended_max_cps)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StudioPanel")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 7, 8, 7)
        root.setSpacing(5)

        self.tabs = QTabWidget(self)
        root.addWidget(self.tabs, stretch=1)

        # --- Tab 1: Relations & Scene Context ---
        tab_rel = QWidget()
        rel_lay = QVBoxLayout(tab_rel)
        rel_lay.setContentsMargins(6, 6, 6, 6)
        rel_lay.setSpacing(6)

        rel_top = QHBoxLayout()
        lbl_hint = QLabel("화자 노드를 클릭하면 좌하단 대사 표를 해당 화자로 필터링합니다.")
        lbl_hint.setObjectName("MutedLabel")
        self.btn_clear_spk = QPushButton("화자 필터 해제 ✕")
        self.btn_clear_spk.setVisible(False)
        self.btn_clear_spk.clicked.connect(lambda: self.speaker_filter_toggled.emit(""))
        rel_top.addWidget(lbl_hint)
        rel_top.addStretch(1)
        rel_top.addWidget(self.btn_clear_spk)
        rel_lay.addLayout(rel_top)

        self.rel_graph = RelationGraphWidget(tab_rel)
        self.rel_graph.speaker_clicked.connect(self.speaker_filter_toggled.emit)
        rel_lay.addWidget(self.rel_graph)

        self.scene_list = QListWidget(tab_rel)
        self.scene_list.setStyleSheet(
            "QListWidget { background:#0B101B; border:1px solid #222E44; border-radius:5px; padding:3px; }"
            "QListWidget::item { padding:5px; border-bottom:1px solid #1A2436; }"
            "QListWidget::item:selected { background:#1E3252; }"
        )
        self.scene_list.itemClicked.connect(self._on_scene_clicked)
        rel_lay.addWidget(self.scene_list, stretch=1)

        self.tabs.addTab(tab_rel, "인물 6화계 관계도 · 씬 맥락")

        # --- Tab 2: Macro Quality, Score_dist, Pruner & VRAM ---
        tab_qual_scroll = QScrollArea()
        tab_qual_scroll.setWidgetResizable(True)
        tab_qual = QWidget()
        qual_lay = QVBoxLayout(tab_qual)
        qual_lay.setContentsMargins(6, 6, 6, 6)
        qual_lay.setSpacing(6)

        # 6 KPI Cards Grid
        kpi_grid = QGridLayout()
        kpi_grid.setSpacing(5)
        self._kpi_val_labels: list[QLabel] = []
        kpi_titles = [
            ("6화계 문맥 일치율", "#38BDF8"),
            ("의미 보존 C_i · 조화 O_i", "#34D399"),
            ("반어법 · 화계오류 · 전환", "#FBBF24"),
            ("기존 자막 공백정규화 sWER", "#C084FC"),
            ("평균 CPS · 지문 밀도", "#F8FAFC"),
            ("화면밖 ASD · OCR 상단회피", "#6EE7B7"),
        ]
        for idx, (kt, col) in enumerate(kpi_titles):
            card = QFrame()
            card.setObjectName("SubCard")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(7, 5, 7, 5)
            cl.setSpacing(2)
            lt = QLabel(kt)
            lt.setObjectName("MutedLabel")
            lv = QLabel("-")
            lv.setStyleSheet(f"color:{col};font-size:13px;font-weight:bold;font-family:Consolas;")
            cl.addWidget(lt)
            cl.addWidget(lv)
            self._kpi_val_labels.append(lv)
            kpi_grid.addWidget(card, idx // 3, idx % 3)
        qual_lay.addLayout(kpi_grid)

        # 5-Bin Score_dist
        dist_card = QFrame()
        dist_card.setObjectName("SubCard")
        dl = QVBoxLayout(dist_card)
        dl.setContentsMargins(8, 6, 8, 6)
        dh = QHBoxLayout()
        dh.addWidget(QLabel("5구간 화법 강도 분포 (Score_dist · Sun et al., 2024)"))
        self.lbl_score_text = QLabel("Score_text: 0.0 / 100")
        self.lbl_score_text.setStyleSheet("color:#38BDF8;font-family:Consolas;font-weight:bold;")
        dh.addStretch(1)
        dh.addWidget(self.lbl_score_text)
        dl.addLayout(dh)
        self.score_dist_chart = ScoreDistChartWidget(dist_card)
        dl.addWidget(self.score_dist_chart)
        qual_lay.addWidget(dist_card)

        # Over-stylization & CPS Pruner Controls
        prune_card = QFrame()
        prune_card.setObjectName("SubCard")
        pl = QVBoxLayout(prune_card)
        pl.setContentsMargins(8, 6, 8, 6)
        ph = QHBoxLayout()
        ph.addWidget(QLabel("과잉 지문 밀도 및 CPS 제어 (인지 피로 억제기)"))
        btn_prune = QPushButton("억제 기준 재적용")
        btn_prune.setObjectName("AccentBtn")
        btn_prune.clicked.connect(self._emit_prune)
        ph.addStretch(1)
        ph.addWidget(btn_prune)
        pl.addLayout(ph)

        sl_row = QHBoxLayout()
        self.lbl_cue_cap = QLabel("지문 밀도 상한: 45%")
        self.lbl_cue_cap.setObjectName("MutedLabel")
        self.sld_cue_cap = QSlider(Qt.Orientation.Horizontal)
        self.sld_cue_cap.setRange(15, 100)
        self.sld_cue_cap.setValue(45)
        self.sld_cue_cap.valueChanged.connect(lambda v: self.lbl_cue_cap.setText(f"지문 밀도 상한: {v}%"))

        self.lbl_cps_cap = QLabel("권장 최대 CPS: 14자/초")
        self.lbl_cps_cap.setObjectName("MutedLabel")
        self.sld_cps_cap = QSlider(Qt.Orientation.Horizontal)
        self.sld_cps_cap.setRange(8, 24)
        self.sld_cps_cap.setValue(14)
        self.sld_cps_cap.valueChanged.connect(lambda v: self.lbl_cps_cap.setText(f"권장 최대 CPS: {v}자/초"))

        c1_l = QVBoxLayout()
        c1_l.addWidget(self.lbl_cue_cap)
        c1_l.addWidget(self.sld_cue_cap)
        c2_l = QVBoxLayout()
        c2_l.addWidget(self.lbl_cps_cap)
        c2_l.addWidget(self.sld_cps_cap)
        sl_row.addLayout(c1_l)
        sl_row.addLayout(c2_l)
        pl.addLayout(sl_row)
        qual_lay.addWidget(prune_card)

        # VRAM Telemetry Card
        vram_card = QFrame()
        vram_card.setObjectName("SubCard")
        self.vram_lay = QVBoxLayout(vram_card)
        self.vram_lay.setContentsMargins(8, 6, 8, 6)
        self.vram_lay.setSpacing(4)
        self.vram_lay.addWidget(QLabel("로컬 GPU 순차 스테이지 피크 VRAM 회수 기록"))
        qual_lay.addWidget(vram_card)
        qual_lay.addStretch(1)

        tab_qual_scroll.setWidget(tab_qual)
        self.tabs.addTab(tab_qual_scroll, "거시 품질 · Score_dist · VRAM")

    def update_from_document(self, doc: MoSDocument, active_speaker: str | None = None) -> None:
        self.btn_clear_spk.setVisible(bool(active_speaker))
        self.rel_graph.set_document(doc, active_speaker=active_speaker)

        # Scenes list
        self.scene_list.clear()
        for sc in doc.scenes:
            ocr_str = f" | OCR: {', '.join(sc.ocr_summary)}" if sc.ocr_summary else ""
            item = QListWidgetItem(
                f"[씬 #{sc.scene_id}] {_fmt_ms(sc.start_ms)}~{_fmt_ms(sc.end_ms)} — {sc.description_ko}{ocr_str}"
            )
            item.setData(Qt.ItemDataRole.UserRole, sc.start_ms)
            self.scene_list.addItem(item)

        # KPIs
        m = doc.metrics
        swer_txt = (
            f"{m.swer_comparison.swer_percent:.1f}% (WER {m.swer_comparison.wer_percent:.1f}%)"
            if m.swer_comparison
            else "영상 단독(Mode A)"
        )
        vals = [
            f"{m.honorific_consistency_rate:.1f}%",
            f"{m.mean_semantic_ci:.2f} / {m.mean_harmonic_oi:.2f}",
            f"{m.irony_segment_count} / {m.mismatch_error_count} / {m.dramatic_shift_count}건",
            swer_txt,
            f"{m.mean_cps:.1f}자/s · {m.cue_attachment_ratio * 100:.0f}%",
            f"{m.off_screen_segment_count}건 / {m.ocr_collision_avoided_count}건",
        ]
        for lbl, v in zip(self._kpi_val_labels, vals):
            lbl.setText(v)

        self.lbl_score_text.setText(f"Score_text: {m.score_text_intensity:.1f} / 100")
        self.score_dist_chart.set_bins(m.score_dist_5bin)

        # Rebuild VRAM rows
        while self.vram_lay.count() > 1:
            child = self.vram_lay.takeAt(1)
            if child.widget():
                child.widget().deleteLater()

        if not doc.vram_telemetry:
            none_lbl = QLabel("기록된 GPU 스테이지 텔레메트리가 없습니다.")
            none_lbl.setObjectName("MutedLabel")
            self.vram_lay.addWidget(none_lbl)
        else:
            max_mb = max(1024.0, *(t.peak_allocated_mb for t in doc.vram_telemetry))
            for t in doc.vram_telemetry:
                row_w = QWidget()
                rl = QHBoxLayout(row_w)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.setSpacing(6)
                s_lbl = QLabel(t.stage_name.replace("Stage ", ""))
                s_lbl.setFixedWidth(185)
                s_lbl.setObjectName("MutedLabel")
                bar = QProgressBar()
                bar.setRange(0, int(max_mb))
                bar.setValue(int(t.peak_allocated_mb))
                bar.setFormat(f"피크 {t.peak_allocated_mb:.0f} MB → 회수 후 {t.allocated_after_mb:.0f} MB")
                bar.setFixedHeight(15)
                rl.addWidget(s_lbl)
                rl.addWidget(bar, stretch=1)
                self.vram_lay.addWidget(row_w)

    def _on_scene_clicked(self, item: QListWidgetItem) -> None:
        ms = item.data(Qt.ItemDataRole.UserRole)
        if ms is not None:
            self.scene_seek_requested.emit(int(ms))

    def _emit_prune(self) -> None:
        self.prune_requested.emit(self.sld_cue_cap.value() / 100.0, float(self.sld_cps_cap.value()))
