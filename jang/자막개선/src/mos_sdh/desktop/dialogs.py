"""Native Windows Qt6 Dialogs: 5-Stage GPU Analysis Runner & Bang et al. (2020) sWER Calculator."""

from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mos_sdh.config import AnalysisConfig
from mos_sdh.desktop.workers import PipelineWorker
from mos_sdh.schemas.document import MoSDocument
from mos_sdh.stage4_mos.swer_evaluator import evaluate_cer_wer_swer


class NewAnalysisDialog(QDialog):
    """Native file-picker modal executing the 5-Stage GPU pipeline on a background QThread."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("새 영상 5-Stage 시청각 화법(MoS) 분석")
        self.setMinimumWidth(560)
        self.result_doc: MoSDocument | None = None
        self.result_out_dir: Path | None = None
        self._worker: PipelineWorker | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        hdr = QLabel("로컬 GPU 5-Stage 배리어프리(SDH/CC) 자막 분석 및 생성")
        hdr.setObjectName("PanelTitle")
        lay.addWidget(hdr)

        # Media File Picker
        lay.addWidget(QLabel("분석할 영상 또는 오디오 파일 (.mp4, .mkv, .mov, .wav) *"))
        r_media = QHBoxLayout()
        self.inp_media = QLineEdit()
        self.inp_media.setPlaceholderText("찾아보기를 눌러 영상 파일을 선택하세요…")
        btn_browse_media = QPushButton("파일 찾아보기…")
        btn_browse_media.clicked.connect(self._browse_media)
        r_media.addWidget(self.inp_media, stretch=1)
        r_media.addWidget(btn_browse_media)
        lay.addLayout(r_media)

        # Optional Subtitle Picker
        lay.addWidget(QLabel("비교·진단할 기존 자막 파일 (선택 · 지정 시 Mode B sWER & 화계 결손 QA 수행)"))
        r_sub = QHBoxLayout()
        self.inp_sub = QLineEdit()
        self.inp_sub.setPlaceholderText("비워두면 영상 단독 End-to-End 생성(Mode A)으로 동작합니다")
        btn_browse_sub = QPushButton("자막 찾아보기…")
        btn_browse_sub.clicked.connect(self._browse_sub)
        r_sub.addWidget(self.inp_sub, stretch=1)
        r_sub.addWidget(btn_browse_sub)
        lay.addLayout(r_sub)

        # Output Folder Picker
        lay.addWidget(QLabel("출력 저장 폴더 (.srt / .vtt / .ass / 리포트)"))
        r_out = QHBoxLayout()
        self.inp_out = QLineEdit(str((Path("./output/studio_run")).resolve()))
        btn_browse_out = QPushButton("폴더 변경…")
        btn_browse_out.clicked.connect(self._browse_out)
        r_out.addWidget(self.inp_out, stretch=1)
        r_out.addWidget(btn_browse_out)
        lay.addLayout(r_out)

        # GPU Stage Options
        opt_box = QFrame()
        opt_box.setObjectName("SubCard")
        ol = QVBoxLayout(opt_box)
        ol.setContentsMargins(10, 8, 10, 8)
        ol.setSpacing(6)
        ol.addWidget(QLabel("로컬 GPU 순차 실행 스테이지 옵션 (종료 즉시 VRAM 자동 회수)"))

        self.chk_demucs = QCheckBox("Stage 1-A 음원 분리 (Demucs — 배경음악·효과음 고조 분리)")
        self.chk_demucs.setChecked(True)
        self.chk_visual = QCheckBox("Stage 2 고밀도 시각 (YuNet 구순 ASD + ViT 표정 + EasyOCR 하단 충돌 회피)")
        self.chk_visual.setChecked(True)
        self.chk_vlm = QCheckBox("Stage 3 로컬 4-bit VLM (Qwen2.5-VL 씬 상황 요약 · 미체크 시 고속 휴리스틱)")
        self.chk_vlm.setChecked(False)
        self.chk_llm = QCheckBox("Stage 5 로컬 4-bit LLM 증강 제로샷 재작성 (미체크 시 위치 태그 3후보 고속 실행)")
        self.chk_llm.setChecked(False)

        for chk in (self.chk_demucs, self.chk_visual, self.chk_vlm, self.chk_llm):
            ol.addWidget(chk)
        lay.addWidget(opt_box)

        # Progress Indicator
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 0)
        self.prog_bar.setVisible(False)
        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("MutedLabel")
        self.lbl_status.setVisible(False)
        lay.addWidget(self.prog_bar)
        lay.addWidget(self.lbl_status)

        # Buttons
        foot = QHBoxLayout()
        foot.addStretch(1)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_start = QPushButton("5-Stage 로컬 GPU 분석 시작")
        self.btn_start.setObjectName("PrimaryBtn")
        self.btn_start.clicked.connect(self._start_pipeline)
        foot.addWidget(self.btn_cancel)
        foot.addWidget(self.btn_start)
        lay.addLayout(foot)

    def _browse_media(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "분석할 영상 또는 오디오 선택",
            "",
            "미디어 파일 (*.mp4 *.mkv *.mov *.avi *.wav *.mp3 *.m4a);;모든 파일 (*.*)",
        )
        if path:
            self.inp_media.setText(path)
            stem = Path(path).stem
            self.inp_out.setText(str((Path("./output") / stem).resolve()))

    def _browse_sub(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "비교할 기존 자막 선택",
            "",
            "자막 파일 (*.srt *.vtt *.ass);;모든 파일 (*.*)",
        )
        if path:
            self.inp_sub.setText(path)

    def _browse_out(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "산출물 저장 폴더 선택", self.inp_out.text())
        if folder:
            self.inp_out.setText(folder)

    def _start_pipeline(self) -> None:
        media_str = self.inp_media.text().strip()
        if not media_str or not Path(media_str).exists():
            QMessageBox.warning(self, "입력 확인", "유효한 영상 또는 오디오 파일을 선택해 주세요.")
            return

        sub_str = self.inp_sub.text().strip()
        sub_path = Path(sub_str) if sub_str else None
        if sub_path is not None and not sub_path.exists():
            QMessageBox.warning(self, "입력 확인", "지정한 기존 자막 파일을 찾을 수 없습니다.")
            return

        out_dir = Path(self.inp_out.text().strip() or "./output/studio_run")
        cfg = AnalysisConfig(
            enable_source_separation=self.chk_demucs.isChecked(),
            enable_visual_stage=self.chk_visual.isChecked(),
            enable_vlm_stage=self.chk_vlm.isChecked(),
            enable_llm_stage=self.chk_llm.isChecked(),
        )

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.prog_bar.setVisible(True)
        self.lbl_status.setVisible(True)

        self._worker = PipelineWorker(
            input_media=Path(media_str),
            out_dir=out_dir,
            config=cfg,
            existing_subtitle=sub_path,
        )
        self._worker.status_changed.connect(self.lbl_status.setText)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, doc: MoSDocument, out_dir: Path) -> None:
        self.result_doc = doc
        self.result_out_dir = out_dir
        self.accept()

    def _on_failed(self, err_msg: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.prog_bar.setVisible(False)
        self.lbl_status.setVisible(False)
        QMessageBox.critical(self, "파이프라인 실행 오류", f"분석 중 오류가 발생했습니다:\n{err_msg}")


class SWERCalculatorDialog(QDialog):
    """Interactive desktop calculator for Korean CER / WER / Space-Normalized sWER (Bang et al., 2020)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("한국어 전사 오류율 계산기 (CER / WER / 공백 정규화 sWER)")
        self.setMinimumSize(540, 400)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)

        desc = QLabel(
            "Bang et al. (2020) Algorithm A1 & A2 기반으로 한국어 띄어쓰기 차이에 의한 WER 과대 왜곡을 제거한 sWER을 즉시 계산합니다."
        )
        desc.setWordWrap(True)
        desc.setObjectName("MutedLabel")
        lay.addWidget(desc)

        lay.addWidget(QLabel("참조 원음 전사 (줄 단위 입력)"))
        self.txt_ref = QPlainTextEdit()
        self.txt_ref.setPlaceholderText("참조 원음 전사 문장을 줄 단위로 입력하세요…")
        self.txt_ref.setFixedHeight(82)
        lay.addWidget(self.txt_ref)

        lay.addWidget(QLabel("비교 가설 자막 (줄 단위 입력 · SDH 대괄호/소괄호 지문은 자동 제외)"))
        self.txt_hyp = QPlainTextEdit()
        self.txt_hyp.setPlaceholderText("비교할 자막 문장을 줄 단위로 입력하세요…")
        self.txt_hyp.setFixedHeight(82)
        lay.addWidget(self.txt_hyp)

        self.lbl_res = QLabel("참조 문장과 가설 문장을 입력한 뒤 [오류율 계산 실행]을 누르세요.")
        self.lbl_res.setStyleSheet(
            "background:#0B101B;border:1px solid #222E44;border-radius:6px;padding:10px;font-family:Consolas;font-size:12px;"
        )
        lay.addWidget(self.lbl_res)

        foot = QHBoxLayout()
        foot.addStretch(1)
        btn_calc = QPushButton("오류율 계산 실행")
        btn_calc.setObjectName("AccentBtn")
        btn_calc.clicked.connect(self.compute)
        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(self.accept)
        foot.addWidget(btn_calc)
        foot.addWidget(btn_close)
        lay.addLayout(foot)

    def compute(self) -> None:
        refs = [ln.strip() for ln in self.txt_ref.toPlainText().splitlines() if ln.strip()]
        hyps = [ln.strip() for ln in self.txt_hyp.toPlainText().splitlines() if ln.strip()]
        if not refs or not hyps:
            self.lbl_res.setText("참조 문장과 가설 문장을 모두 입력해 주세요.")
            return
        if len(refs) != len(hyps):
            refs = [" ".join(refs)]
            hyps = [" ".join(hyps)]

        comp = evaluate_cer_wer_swer(refs, hyps)
        self.lbl_res.setText(
            f"비교 문장 수 : {comp.compared_utterances}문장\n"
            f"음절 오류율 (CER)            : {comp.cer_percent:.2f}%\n"
            f"일반 어절 오류율 (WER)       : {comp.wer_percent:.2f}%\n"
            f"공백 정규화 어절 오류율(sWER): {comp.swer_percent:.2f}%  (띄어쓰기 왜곡 {comp.wer_percent - comp.swer_percent:+.2f}%p 보정)"
        )
        self.lbl_res.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
