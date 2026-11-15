"""Quadrant 1: Native Qt6 Multimedia Video Player + QPainter Live .ASS/.SRT & OCR Debug Overlay."""

from __future__ import annotations

import html
from pathlib import Path
import re
from PySide6.QtCore import QRectF, QSizeF, Qt, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
    QTextDocument,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from mos_sdh.desktop.theme import SPEAKER_HEX_PALETTE
from mos_sdh.schemas.document import MoSDocument, SegmentMoS


def _fmt_ms(ms: int) -> str:
    safe = max(0, int(ms))
    tot_s, milli = divmod(safe, 1000)
    m, s = divmod(tot_s, 60)
    return f"{m:02d}:{s:02d}.{milli:03d}"


class OverlayVideoGraphicsView(QGraphicsView):
    """Hardware-composited QGraphicsView painting live .ASS/.SRT subtitles & OCR debug boxes over video."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QColor("#04070C"))

        self.video_item = QGraphicsVideoItem()
        self.video_item.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self._scene.addItem(self.video_item)

        self._doc: MoSDocument | None = None
        self._current_ms: int = 0
        self._sub_mode: str = "ass"  # 'ass' | 'srt' | 'raw'
        self._show_debug: bool = True
        self._has_media: bool = False

    def set_document(self, doc: MoSDocument | None, has_media: bool) -> None:
        self._doc = doc
        self._has_media = has_media
        self.viewport().update()

    def set_playback_time(self, ms: int) -> None:
        self._current_ms = ms
        self.viewport().update()

    def set_subtitle_mode(self, mode: str) -> None:
        self._sub_mode = mode
        self.viewport().update()

    def set_debug_enabled(self, enabled: bool) -> None:
        self._show_debug = enabled
        self.viewport().update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        w = max(10.0, float(self.viewport().width()))
        h = max(10.0, float(self.viewport().height()))
        self._scene.setSceneRect(0.0, 0.0, w, h)
        self.video_item.setSize(QSizeF(w, h))
        self.video_item.setPos(0.0, 0.0)

    def _active_segment(self) -> SegmentMoS | None:
        if self._doc is None:
            return None
        t = self._current_ms
        for seg in self._doc.segments:
            if seg.start_ms <= t <= max(seg.start_ms + 300, seg.end_ms):
                return seg
        return None

    def _video_content_rect(self) -> QRectF:
        vw = float(self.viewport().width())
        vh = float(self.viewport().height())
        nat = self.video_item.nativeSize()
        nw = nat.width() if nat.width() > 0 else 16.0
        nh = nat.height() if nat.height() > 0 else 9.0
        view_ratio = vw / max(1.0, vh)
        vid_ratio = nw / max(1.0, nh)

        if vid_ratio > view_ratio:
            h = vw / vid_ratio
            return QRectF(0.0, (vh - h) * 0.5, vw, h)
        w = vh * vid_ratio
        return QRectF((vw - w) * 0.5, 0.0, w, vh)

    def _speaker_hex(self, speaker_id: str) -> str:
        if self._doc is None:
            return SPEAKER_HEX_PALETTE[0]
        speakers = sorted({s.speaker_id for s in self._doc.segments})
        idx = speakers.index(speaker_id) if speaker_id in speakers else 0
        return SPEAKER_HEX_PALETTE[idx % len(SPEAKER_HEX_PALETTE)]

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        painter.save()
        vp_w = float(self.viewport().width())
        vp_h = float(self.viewport().height())

        if self._doc is None:
            painter.setPen(QColor("#94A3B8"))
            painter.setFont(QFont("Malgun Gothic", 11))
            painter.drawText(
                QRectF(0, 0, vp_w, vp_h),
                Qt.AlignmentFlag.AlignCenter,
                "상단 툴바의 [+ 새 영상 5-Stage 분석] 또는 [기존 리포트 열기]를 눌러 시작하세요.",
            )
            painter.restore()
            return

        vrect = self._video_content_rect()
        seg = self._active_segment()

        # 1. Multimodal Debug Overlays (Scene OCR Dashed Boxes + Top HUD Pills)
        if self._show_debug:
            self._paint_debug_hud_and_ocr(painter, vrect, seg)

        # 2. Live Subtitle Overlay (.ASS Rich Typography / .SRT / Raw)
        if seg is not None:
            self._paint_subtitle_box(painter, vrect, seg)

        painter.restore()

    def _paint_debug_hud_and_ocr(
        self,
        painter: QPainter,
        vrect: QRectF,
        seg: SegmentMoS | None,
    ) -> None:
        if self._doc is None:
            return

        # Yellow dashed OCR bounding boxes
        if seg is not None and seg.visual.ocr_boxes:
            dash_pen = QPen(QColor("#FACC15"), 2.0, Qt.PenStyle.DashLine)
            painter.setFont(QFont("Malgun Gothic", 8, QFont.Weight.Bold))
            for b in seg.visual.ocr_boxes:
                bx = vrect.left() + b.x_min * vrect.width()
                by = vrect.top() + b.y_min * vrect.height()
                bw = max(20.0, (b.x_max - b.x_min) * vrect.width())
                bh = max(14.0, (b.y_max - b.y_min) * vrect.height())
                box_r = QRectF(bx, by, bw, bh)

                painter.fillRect(box_r, QColor(250, 204, 21, 28))
                painter.setPen(dash_pen)
                painter.drawRect(box_r)

                lbl_txt = f" OCR: {b.text} ({int(round(b.confidence * 100))}%) "
                fm = painter.fontMetrics()
                lw = fm.horizontalAdvance(lbl_txt) + 6
                lh = fm.height() + 4
                # Place tag below the box if the OCR box sits near the top HUD row
                lbl_y = (by + bh + 3.0) if (by - lh - 2.0 < vrect.top() + 36.0) else (by - lh - 2.0)
                lbl_r = QRectF(bx, lbl_y, lw, lh)
                painter.fillRect(lbl_r, QColor("#FACC15"))
                painter.setPen(QColor("#0F172A"))
                painter.drawText(lbl_r, Qt.AlignmentFlag.AlignCenter, lbl_txt)

        # Top HUD Pills
        hud_font = QFont("Malgun Gothic", 8, QFont.Weight.Bold)
        painter.setFont(hud_font)
        fm = painter.fontMetrics()

        cur_scene = next(
            (sc for sc in self._doc.scenes if sc.start_ms <= self._current_ms <= sc.end_ms),
            self._doc.scenes[0] if self._doc.scenes else None,
        )
        left_badges: list[tuple[str, str, str]] = []
        if cur_scene is not None:
            left_badges.append((f"씬 #{cur_scene.scene_id}", "#0F172A", "#E2E8F0"))
        if seg is not None:
            asd_txt = (
                "ASD: 화면 밖 발화"
                if seg.visual.on_screen_speaker is False
                else f"ASD: 화면 내 ({seg.visual.speaker_zone})"
            )
            left_badges.append((asd_txt, "#0F172A", "#CBD5E1"))
            if seg.visual.facial_expression:
                left_badges.append((f"표정: {seg.visual.facial_expression}", "#0F172A", "#93C5FD"))

        cx = vrect.left() + 10.0
        cy = vrect.top() + 10.0
        for txt, bg_hex, fg_hex in left_badges:
            bw = fm.horizontalAdvance(txt) + 14
            bh = fm.height() + 6
            r = QRectF(cx, cy, bw, bh)
            painter.setPen(QPen(QColor("#334155"), 1.0))
            painter.setBrush(QColor(bg_hex))
            painter.drawRoundedRect(r, 4.0, 4.0)
            painter.setPen(QColor(fg_hex))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, txt)
            cx += bw + 6.0

        if seg is not None:
            right_badges: list[tuple[str, str, str]] = [
                (
                    f"정서: {seg.emotion_irony.dominant_emotion} · 각성 Z={seg.arousal_zscore:+.2f}",
                    "#0F172A",
                    "#F8FAFC",
                )
            ]
            if seg.emotion_irony.irony_detected:
                right_badges.append(
                    (f"⚡ 반어법 감지 (I={seg.emotion_irony.irony_score:.2f})", "#451A03", "#FCD34D")
                )
            if seg.visual.avoid_bottom_region:
                right_badges.append(("⬆ 하단 OCR 충돌 → 상단 {\\an8} 회피", "#064E3B", "#6EE7B7"))

            rx = vrect.right() - 10.0
            for txt, bg_hex, fg_hex in reversed(right_badges):
                bw = fm.horizontalAdvance(txt) + 14
                bh = fm.height() + 6
                rx -= bw
                r = QRectF(rx, cy, bw, bh)
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.setBrush(QColor(bg_hex))
                painter.drawRoundedRect(r, 4.0, 4.0)
                painter.setPen(QColor(fg_hex))
                painter.drawText(r, Qt.AlignmentFlag.AlignCenter, txt)
                rx -= 6.0

    def _paint_subtitle_box(
        self,
        painter: QPainter,
        vrect: QRectF,
        seg: SegmentMoS,
    ) -> None:
        is_ass = self._sub_mode == "ass"
        top_avoid = is_ass and seg.visual.avoid_bottom_region
        zone = seg.visual.speaker_zone if is_ass else "center"

        html_body = self._build_subtitle_html(seg)
        td = QTextDocument()
        td.setDefaultFont(QFont("Malgun Gothic", 13, QFont.Weight.DemiBold))
        max_w = max(200.0, vrect.width() * 0.84)
        td.setTextWidth(max_w)
        td.setHtml(html_body)

        ideal_w = min(max_w, td.idealWidth() + 18.0)
        td.setTextWidth(ideal_w)
        box_h = td.size().height() + 10.0

        if zone == "left":
            bx = vrect.left() + vrect.width() * 0.06
        elif zone == "right":
            bx = vrect.right() - vrect.width() * 0.06 - ideal_w
        else:
            bx = vrect.center().x() - ideal_w * 0.5

        by = (
            vrect.top() + 42.0
            if top_avoid
            else vrect.bottom() - box_h - 32.0
        )

        bg_rect = QRectF(bx - 8.0, by - 4.0, ideal_w + 16.0, box_h + 8.0)
        painter.setPen(QPen(QColor(255, 255, 255, 42), 1.0))
        painter.setBrush(QColor(6, 9, 16, 212))
        painter.drawRoundedRect(bg_rect, 7.0, 7.0)

        painter.save()
        painter.translate(bx, by)
        td.drawContents(painter)
        painter.restore()

    def _build_subtitle_html(self, seg: SegmentMoS) -> str:
        if self._sub_mode == "raw":
            raw = html.escape(seg.existing_subtitle or seg.raw_transcript)
            return f"<div style='color:#E2E8F0;text-align:center;'>{raw}</div>"

        if self._sub_mode == "srt":
            srt = html.escape(seg.sdh_srt_text)
            return f"<div style='color:#FFFFFF;text-align:center;'>{srt}</div>"

        spk_hex = self._speaker_hex(seg.speaker_id)
        spk_lbl = (
            f"[{seg.speaker_id}(화면 밖)]"
            if seg.visual.on_screen_speaker is False
            else f"[{seg.speaker_id}]"
        )
        parts: list[str] = [f"<span style='color:{spk_hex};font-weight:700;'>{html.escape(spk_lbl)}</span>"]

        if seg.emotion_irony.irony_detected:
            parts.append("<span style='color:#FFD166;font-style:italic;'>(비꼬는 어조로)</span>")
        elif seg.honorific.consistency_status == "dramatic_shift":
            note = "격앙되어 반말로" if seg.honorific.is_polite_group is False else "어조 전환"
            parts.append(f"<span style='color:#FB923C;font-style:italic;'>({html.escape(note)})</span>")

        if seg.words and seg.honorific.consistency_status != "mismatch_error":
            w_html: list[str] = []
            for w in seg.words:
                esc_w = html.escape(w.text)
                if w.emphasis_level >= 4:
                    w_html.append(f"<span style='font-size:125%;font-weight:800;color:#FEF08A;'>{esc_w}</span>")
                elif w.emphasis_level == 3:
                    w_html.append(f"<span style='font-size:116%;font-weight:800;text-decoration:underline;'>{esc_w}</span>")
                elif w.emphasis_level == 2:
                    w_html.append(f"<span style='font-size:108%;font-weight:700;'>{esc_w}</span>")
                else:
                    w_html.append(f"<span style='color:{spk_hex};'>{esc_w}</span>")
            parts.append(" ".join(w_html))
        else:
            clean = re.sub(r"\{[^}]*\}", "", seg.sdh_ass_text or seg.sdh_srt_text)
            clean = re.sub(r"^\[[^\]]+\]\s*", "", clean)
            clean = re.sub(r"^\([^)]+\)\s*", "", clean)
            parts.append(f"<span style='color:{spk_hex};'>{html.escape(clean)}</span>")

        if seg.cues:
            cue_str = ", ".join(c.label_ko for c in seg.cues)
            parts.append(f"<span style='color:#93C5FD;font-style:italic;font-size:90%;'>({html.escape(cue_str)})</span>")

        return f"<div style='text-align:center;'>{' '.join(parts)}</div>"


class VideoPlayerPanel(QFrame):
    """Complete Quadrant 1 panel: header controls + OverlayVideoGraphicsView + native transport bar."""

    time_updated = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StudioPanel")

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.85)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)

        self._duration_ms: int = 0
        self._slider_dragging: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(6)

        # Header Row
        hdr = QHBoxLayout()
        title = QLabel("멀티모달 비디오 프리뷰")
        title.setObjectName("PanelTitle")
        self.lbl_timecode = QLabel("00:00.000 / 00:00.000")
        self.lbl_timecode.setStyleSheet(
            "background:#0B101B;color:#38BDF8;border:1px solid #222E44;border-radius:4px;padding:2px 7px;font-family:Consolas;"
        )

        self.cmb_sub_mode = QComboBox()
        self.cmb_sub_mode.addItem("시각 스타일 (.ASS 프리뷰)", "ass")
        self.cmb_sub_mode.addItem("표준 지문 (.SRT 프리뷰)", "srt")
        self.cmb_sub_mode.addItem("기존·원음 자막 비교", "raw")

        self.chk_debug = QCheckBox("디버그 오버레이 (OCR·ASD)")
        self.chk_debug.setChecked(True)

        hdr.addWidget(title)
        hdr.addWidget(self.lbl_timecode)
        hdr.addStretch(1)
        hdr.addWidget(self.cmb_sub_mode)
        hdr.addWidget(self.chk_debug)
        layout.addLayout(hdr)

        # Graphics Video Canvas
        self.canvas = OverlayVideoGraphicsView(self)
        self.player.setVideoOutput(self.canvas.video_item)
        layout.addWidget(self.canvas, stretch=1)

        # Transport Controls Row
        ctrl = QHBoxLayout()
        self.btn_play = QPushButton("▶ 재생")
        self.btn_play.setFixedWidth(78)
        self.sld_seek = QSlider(Qt.Orientation.Horizontal)
        self.sld_seek.setRange(0, 1000)

        lbl_vol = QLabel("음량")
        lbl_vol.setObjectName("MutedLabel")
        self.sld_vol = QSlider(Qt.Orientation.Horizontal)
        self.sld_vol.setRange(0, 100)
        self.sld_vol.setValue(85)
        self.sld_vol.setFixedWidth(76)

        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.sld_seek, stretch=1)
        ctrl.addWidget(lbl_vol)
        ctrl.addWidget(self.sld_vol)
        layout.addLayout(ctrl)

        # Connections
        self.btn_play.clicked.connect(self.toggle_play_pause)
        self.cmb_sub_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.chk_debug.toggled.connect(self.canvas.set_debug_enabled)
        self.sld_vol.valueChanged.connect(lambda v: self.audio_output.setVolume(v / 100.0))

        self.sld_seek.sliderPressed.connect(self._on_slider_pressed)
        self.sld_seek.sliderReleased.connect(self._on_slider_released)
        self.sld_seek.sliderMoved.connect(self._on_slider_moved)

        self.player.positionChanged.connect(self._on_player_position)
        self.player.durationChanged.connect(self._on_player_duration)
        self.player.playbackStateChanged.connect(self._on_state_changed)

    def load_project_media(self, doc: MoSDocument) -> None:
        self.player.stop()
        media_p = Path(doc.input_media_path)
        has_media = media_p.exists()
        self._duration_ms = max(1, doc.duration_ms)
        self.sld_seek.setRange(0, self._duration_ms)
        self.canvas.set_document(doc, has_media=has_media)

        if has_media:
            self.player.setSource(QUrl.fromLocalFile(str(media_p.resolve())))
            self.player.pause()
        else:
            self.player.setSource(QUrl())

        first_ms = doc.segments[0].start_ms + 30 if doc.segments else 0
        self.seek_ms(first_ms)

    def seek_ms(self, ms: int) -> None:
        safe = max(0, min(self._duration_ms, int(ms)))
        if self.player.source().isValid():
            self.player.setPosition(safe)
        self.canvas.set_playback_time(safe)
        if not self._slider_dragging:
            self.sld_seek.setValue(safe)
        self.lbl_timecode.setText(f"{_fmt_ms(safe)} / {_fmt_ms(self._duration_ms)}")
        self.time_updated.emit(safe)

    def refresh_overlay(self) -> None:
        self.canvas.viewport().update()

    def toggle_play_pause(self) -> None:
        if not self.player.source().isValid():
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_mode_changed(self) -> None:
        mode = str(self.cmb_sub_mode.currentData() or "ass")
        self.canvas.set_subtitle_mode(mode)

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        self.seek_ms(self.sld_seek.value())

    def _on_slider_moved(self, val: int) -> None:
        self.canvas.set_playback_time(val)
        self.lbl_timecode.setText(f"{_fmt_ms(val)} / {_fmt_ms(self._duration_ms)}")
        self.time_updated.emit(val)

    def _on_player_position(self, pos_ms: int) -> None:
        val = int(pos_ms)
        self.canvas.set_playback_time(val)
        if not self._slider_dragging:
            self.sld_seek.setValue(val)
        self.lbl_timecode.setText(f"{_fmt_ms(val)} / {_fmt_ms(self._duration_ms)}")
        self.time_updated.emit(val)

    def _on_player_duration(self, dur_ms: int) -> None:
        if dur_ms > 0:
            self._duration_ms = int(dur_ms)
            self.sld_seek.setRange(0, self._duration_ms)

    def _on_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setText("⏸ 일시정지")
        else:
            self.btn_play.setText("▶ 재생")
