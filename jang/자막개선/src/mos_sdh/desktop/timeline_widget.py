"""Middle Band: Custom QPainter 4-Track Synchronized MoS Timeline Scrubber."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from mos_sdh.desktop.theme import EMOTION_HEX_MAP
from mos_sdh.schemas.document import MoSDocument, SegmentMoS


def _fmt_short_ms(ms: int) -> str:
    tot_s, milli = divmod(max(0, int(ms)), 1000)
    m, s = divmod(tot_s, 60)
    return f"{m:02d}:{s:02d}.{milli // 100}"


class MultitrackCanvas(QWidget):
    """High-DPI QPainter 4-lane timeline with click-and-drag playhead scrubbing."""

    seek_requested = Signal(int)
    segment_clicked = Signal(int)

    LABEL_COL_W = 112.0
    RULER_H = 20.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(106)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._doc: MoSDocument | None = None
        self._selected_seg_id: int | None = None
        self._playhead_ms: int = 0

    def set_document(self, doc: MoSDocument | None, selected_seg_id: int | None = None) -> None:
        self._doc = doc
        self._selected_seg_id = selected_seg_id
        self.update()

    def set_selected_segment(self, seg_id: int | None) -> None:
        self._selected_seg_id = seg_id
        self.update()

    def set_playhead_ms(self, ms: int) -> None:
        self._playhead_ms = max(0, int(ms))
        self.update()

    def _total_ms(self) -> int:
        if self._doc is None:
            return 1000
        max_end = max((s.end_ms + 200 for s in self._doc.segments), default=1000)
        return max(1000, self._doc.duration_ms, max_end)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)

        w = float(self.width())
        h = float(self.height())
        track_x = self.LABEL_COL_W
        track_w = max(10.0, w - track_x - 6.0)
        lane_h = max(18.0, (h - self.RULER_H) / 4.0)

        # Background
        p.fillRect(QRectF(0, 0, w, h), QColor("#080C14"))
        p.fillRect(QRectF(0, 0, track_x, h), QColor("#0C121E"))
        p.fillRect(QRectF(track_x, 0, w - track_x, self.RULER_H), QColor("#0D1320"))

        # Track labels & horizontal grid lines
        lane_titles = ["1. 운율·강조", "2. 정서·반어", "3. 한국어 6화계", "4. 자막·OCR충돌"]
        p.setFont(QFont("Malgun Gothic", 8, QFont.Weight.DemiBold))
        p.setPen(QColor("#64748B"))
        p.drawText(QRectF(8, 0, track_x - 10, self.RULER_H), Qt.AlignmentFlag.AlignVCenter, "시간축")

        for idx, title in enumerate(lane_titles):
            ly = self.RULER_H + idx * lane_h
            p.setPen(QPen(QColor("#1E293B"), 1.0))
            p.drawLine(QPointF(0, ly), QPointF(w, ly))
            p.setPen(QColor("#94A3B8"))
            p.drawText(QRectF(8, ly, track_x - 10, lane_h), Qt.AlignmentFlag.AlignVCenter, title)

        p.setPen(QPen(QColor("#222E44"), 1.0))
        p.drawLine(QPointF(track_x, 0), QPointF(track_x, h))

        total_ms = self._total_ms()
        # Time ruler ticks
        p.setFont(QFont("Consolas", 8))
        p.setPen(QColor("#64748B"))
        for i in range(6):
            ratio = i / 5.0
            tx = track_x + ratio * track_w
            tick_ms = int(round(ratio * total_ms))
            p.drawText(
                QRectF(tx - 28, 1, 56, self.RULER_H - 2),
                Qt.AlignmentFlag.AlignCenter,
                _fmt_short_ms(tick_ms),
            )

        if self._doc is None:
            p.end()
            return

        # Segment blocks across all 4 tracks
        p.setFont(QFont("Malgun Gothic", 8, QFont.Weight.Bold))
        for seg in self._doc.segments:
            sx = track_x + (seg.start_ms / total_ms) * track_w
            sw = max(18.0, ((seg.end_ms - seg.start_ms) / total_ms) * track_w)
            is_sel = seg.segment_id == self._selected_seg_id

            # Lane 0: Prosody & <toe_k>
            max_e = max((wd.emphasis_level for wd in seg.words), default=0)
            c0 = QColor("#B45309") if max_e >= 3 else QColor("#0369A1")
            t0 = f"★e{max_e} Z={seg.arousal_zscore:+.1f}" if max_e >= 3 else f"Z={seg.arousal_zscore:+.1f}"
            self._draw_seg_block(p, QRectF(sx, self.RULER_H + 2, sw, lane_h - 4), c0, t0, is_sel)

            # Lane 1: Emotion & Irony (CmGI)
            emo = seg.emotion_irony.dominant_emotion
            c1 = QColor(EMOTION_HEX_MAP.get(emo, "#475569"))
            t1 = f"⚡반어 {emo}" if seg.emotion_irony.irony_detected else emo
            self._draw_seg_block(p, QRectF(sx, self.RULER_H + lane_h + 2, sw, lane_h - 4), c1, t1, is_sel)

            # Lane 2: Korean 6-Speech-Levels
            st = seg.honorific.consistency_status
            if st == "mismatch_error":
                c2, pre = QColor("#E11D48"), "🔴 "
            elif st == "dramatic_shift":
                c2, pre = QColor("#EA580C"), "🟠 "
            elif seg.honorific.is_polite_group:
                c2, pre = QColor("#0284C7"), ""
            else:
                c2, pre = QColor("#4F46E5"), ""
            self._draw_seg_block(
                p,
                QRectF(sx, self.RULER_H + 2 * lane_h + 2, sw, lane_h - 4),
                c2,
                f"{pre}{seg.honorific.speech_level}",
                is_sel,
            )

            # Lane 3: Subtitle & OCR Collision
            c3 = QColor("#059669") if seg.visual.avoid_bottom_region else QColor("#334155")
            t3 = (
                "⬆\\an8 회피"
                if seg.visual.avoid_bottom_region
                else ("화면밖" if seg.visual.on_screen_speaker is False else seg.speaker_id)
            )
            self._draw_seg_block(
                p,
                QRectF(sx, self.RULER_H + 3 * lane_h + 2, sw, lane_h - 4),
                c3,
                t3,
                is_sel,
            )

        # Red vertical Playhead line + top triangle
        ph_ratio = min(1.0, max(0.0, self._playhead_ms / total_ms))
        px = track_x + ph_ratio * track_w
        p.setPen(QPen(QColor("#F43F5E"), 2.0))
        p.drawLine(QPointF(px, 0), QPointF(px, h))
        tri = QPolygonF([QPointF(px - 5, 0), QPointF(px + 5, 0), QPointF(px, 7)])
        p.setBrush(QColor("#F43F5E"))
        p.drawPolygon(tri)

        p.end()

    def _draw_seg_block(
        self,
        p: QPainter,
        r: QRectF,
        color: QColor,
        text: str,
        selected: bool,
    ) -> None:
        fill = QColor(color)
        fill.setAlpha(215 if selected else 175)
        p.setBrush(fill)
        border = QPen(QColor("#FFFFFF"), 1.6) if selected else QPen(QColor(255, 255, 255, 50), 1.0)
        p.setPen(border)
        p.drawRoundedRect(r, 3.5, 3.5)

        p.setPen(QColor("#FFFFFF"))
        text_r = r.adjusted(4, 0, -4, 0)
        elided = p.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, max(0, int(text_r.width())))
        p.drawText(text_r, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, elided)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._handle_mouse_at(event.position(), select_hit=True)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._handle_mouse_at(event.position(), select_hit=False)

    def _handle_mouse_at(self, pos: QPointF, select_hit: bool) -> None:
        if self._doc is None:
            return
        track_w = max(10.0, float(self.width()) - self.LABEL_COL_W - 6.0)
        ratio = min(1.0, max(0.0, (pos.x() - self.LABEL_COL_W) / track_w))
        target_ms = int(round(ratio * self._total_ms()))

        if select_hit:
            hit_seg: SegmentMoS | None = next(
                (s for s in self._doc.segments if s.start_ms <= target_ms <= s.end_ms),
                None,
            )
            if hit_seg is not None:
                self.segment_clicked.emit(hit_seg.segment_id)
                return

        self.seek_requested.emit(target_ms)


class MultitrackTimelinePanel(QFrame):
    """Framed Middle Band wrapping header legend + MultitrackCanvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StudioPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        title = QLabel("멀티트랙 MoS 동기화 타임라인")
        title.setObjectName("PanelTitle")
        legend = QLabel(
            "★ 강조(<toe_3~4>)   ·   ⚡ 반어법(CmGI)   ·   🔴 화계오류(CAPE)   ·   🟠 극적전환   ·   ⬆ OCR상단회피(\\an8)"
        )
        legend.setObjectName("MutedLabel")
        hint = QLabel("타임라인 클릭/드래그 탐색 · Space 재생/정지 · ↑/↓ 세그먼트 이동")
        hint.setObjectName("MutedLabel")

        hdr.addWidget(title)
        hdr.addSpacing(12)
        hdr.addWidget(legend)
        hdr.addStretch(1)
        hdr.addWidget(hint)
        layout.addLayout(hdr)

        self.canvas = MultitrackCanvas(self)
        layout.addWidget(self.canvas, stretch=1)
