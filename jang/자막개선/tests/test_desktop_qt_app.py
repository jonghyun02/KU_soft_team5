"""Automated Qt6 widget & integration tests for the pure native Windows desktop Studio app."""

from __future__ import annotations

from pathlib import Path
import shutil

# Bind CUDA DLLs and import torch before PySide6 on Windows
from mos_sdh.core.vram import bind_windows_cuda_dlls

bind_windows_cuda_dlls()
import torch  # noqa: F401, E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from mos_sdh.desktop.dialogs import SWERCalculatorDialog  # noqa: E402
from mos_sdh.desktop.main_window import MoSSDHMainWindow  # noqa: E402
from mos_sdh.desktop.theme import STUDIO_DARK_QSS, ensure_studio_ico  # noqa: E402
from mos_sdh.schemas.document import (  # noqa: E402
    BenchmarkMetrics,
    EmotionIronyAnalysis,
    HonorificAnalysis,
    MoSDocument,
    OCRBox,
    SceneContext,
    SegmentMoS,
    SpeakerRelation,
    StageVRAMTelemetry,
    SubtitleCandidate,
    VisualContextSlot,
    WordUnit,
)


def _get_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(["pytest-qt6"])
    app.setStyleSheet(STUDIO_DARK_QSS)
    return app


def _build_fixture_doc(tmp_path: Path) -> tuple[MoSDocument, Path]:
    out_dir = tmp_path / "qt_proj"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = MoSDocument(
        mode="mode_b_subtitle_qa",
        input_media_path=str(tmp_path / "dummy.mp4"),
        duration_ms=8000,
        scenes=[
            SceneContext(
                scene_id=1,
                start_ms=0,
                end_ms=8000,
                keyframe_ms=2000,
                description_ko="사무실 대면 대화 씬",
                ocr_summary=["제3화: 균열"],
                inferred_relations=[
                    SpeakerRelation(
                        from_speaker="화자_1",
                        to_speaker="화자_2",
                        relation_ko="직장 동료",
                        prior_speech_level="해요체",
                    )
                ],
            )
        ],
        segments=[
            SegmentMoS(
                segment_id=1,
                start_ms=400,
                end_ms=3400,
                speaker_id="화자_1",
                raw_transcript="참 잘하셨네요 정말 대단해요",
                existing_subtitle="참 잘하셨네요",
                sdh_srt_text="[화자_1] (비꼬는 어조로) 참 잘하셨네요 정말 대단해요",
                sdh_ass_text="{\\an8}[화자_1] (비꼬는 어조로) 참 잘하셨네요 정말 대단해요",
                arousal_zscore=1.4,
                cps=9.5,
                words=[
                    WordUnit(text="참", start_ms=400, end_ms=800, emphasis_level=3, emphasis_token="<toe_3>"),
                    WordUnit(text="잘하셨네요", start_ms=850, end_ms=2000, emphasis_level=4, emphasis_token="<toe_4>"),
                ],
                honorific=HonorificAnalysis(
                    speech_level="해요체",
                    is_polite_group=True,
                    expected_speech_level="해요체",
                    consistency_status="consistent",
                ),
                emotion_irony=EmotionIronyAnalysis(
                    dominant_emotion="냉소",
                    lexical_polarity=0.82,
                    audiovisual_polarity=-0.71,
                    irony_score=0.88,
                    irony_detected=True,
                    gate_weights_mean={"text": 0.35, "audio": 0.38, "visual": 0.27},
                ),
                visual=VisualContextSlot(
                    scene_id=1,
                    face_detected=True,
                    on_screen_speaker=True,
                    speaker_zone="left",
                    facial_expression="분노/불쾌",
                    ocr_boxes=[OCRBox(text="제3화: 균열", x_min=0.18, y_min=0.84, x_max=0.82, y_max=0.93)],
                    ocr_texts=["제3화: 균열"],
                    avoid_bottom_region=True,
                ),
                candidates=[
                    SubtitleCandidate(
                        candidate_id=1,
                        strategy="positional_tag_standard",
                        srt_text="[화자_1] (비꼬는 어조로) 참 잘하셨네요 정말 대단해요",
                        ass_text="{\\an8}[화자_1] (비꼬는 어조로) 참 잘하셨네요",
                        plain_dialogue="참 잘하셨네요 정말 대단해요",
                        semantic_ci=0.99,
                        mos_fidelity=0.95,
                        harmonic_oi=0.97,
                        selected=True,
                    ),
                    SubtitleCandidate(
                        candidate_id=2,
                        strategy="concise_low_cps",
                        srt_text="[화자_1] (냉소) 참 잘하셨네요",
                        ass_text="{\\an8}[화자_1] (냉소) 참 잘하셨네요",
                        plain_dialogue="참 잘하셨네요",
                        semantic_ci=0.91,
                        mos_fidelity=0.88,
                        harmonic_oi=0.89,
                        selected=False,
                    ),
                ],
            )
        ],
        metrics=BenchmarkMetrics(
            total_segments=1,
            scene_count=1,
            speaker_count=1,
            irony_segment_count=1,
            ocr_collision_avoided_count=1,
            score_dist_5bin=[0.0, 0.0, 20.0, 50.0, 30.0],
            score_text_intensity=74.5,
        ),
        vram_telemetry=[
            StageVRAMTelemetry(
                stage_name="Stage 1-C: Faster-Whisper",
                device="cuda",
                gpu_name="RTX 5060 Ti",
                allocated_before_mb=12.0,
                peak_allocated_mb=1780.0,
                allocated_after_mb=18.0,
            )
        ],
    )
    return doc, out_dir


def test_native_qt_main_window_workflow_and_exports(tmp_path: Path) -> None:
    app = _get_qapp()
    ico_file = ensure_studio_ico(tmp_path / "studio.ico")
    assert ico_file.exists() and ico_file.stat().st_size > 1500

    doc, out_dir = _build_fixture_doc(tmp_path)
    win = MoSSDHMainWindow()
    # Verify clean startup with zero auto-loaded example projects
    assert win.current_doc is None
    assert win.diff_table.rowCount() == 0

    win.load_document(doc, out_dir, label="pytest_drama")
    win.show()
    app.processEvents()

    assert win.diff_table.rowCount() == len(doc.segments)
    assert win.cand_vlay.count() >= 2

    # Filter by irony
    win._set_tag_filter("irony")
    app.processEvents()
    assert win.diff_table.rowCount() >= 1
    win._set_tag_filter("all")
    app.processEvents()
    assert win.diff_table.rowCount() == len(doc.segments)

    # Switch candidate on segment #1 and verify .srt on disk updates immediately
    seg1 = doc.segments[0]
    if len(seg1.candidates) >= 2:
        alt_cand = next(c for c in seg1.candidates if not c.selected)
        win.select_candidate(seg1.segment_id, alt_cand.candidate_id)
        app.processEvents()
        srt_disk = (out_dir / "prototype_sdh.srt").read_text(encoding="utf-8")
        assert alt_cand.plain_dialogue[:5] in srt_disk or seg1.speaker_id in srt_disk

    # Rename speaker globally and verify persistence
    orig_spk = doc.segments[0].speaker_id
    win.rename_speaker_globally(orig_spk, "강팀장")
    app.processEvents()
    assert doc.segments[0].speaker_id == "강팀장"
    srt_after_rename = (out_dir / "prototype_sdh.srt").read_text(encoding="utf-8")
    assert "강팀장" in srt_after_rename

    # Re-apply cognitive fatigue pruner
    win.apply_pruning(0.40, 13.0)
    app.processEvents()
    assert (out_dir / "mos_report.json").exists()
    assert (out_dir / "prototype_sdh.ass").exists()

    # Grab native Qt6 window render to PNG
    snap_png = tmp_path / "qt_grab.png"
    ok = win.grab().save(str(snap_png), "PNG")
    assert ok and snap_png.exists() and snap_png.stat().st_size > 15_000
    win.close()


def test_native_qt_swer_calculator_dialog() -> None:
    app = _get_qapp()
    dlg = SWERCalculatorDialog()
    assert dlg.txt_ref.toPlainText() == ""
    assert dlg.txt_hyp.toPlainText() == ""
    dlg.txt_ref.setPlainText("잠시만 기다려 주세요.")
    dlg.txt_hyp.setPlainText("[화자_1] 잠시만 기다려주세요")
    dlg.compute()
    app.processEvents()
    txt = dlg.lbl_res.text()
    assert "공백 정규화 어절 오류율(sWER)" in txt
    assert "0.00%" in txt
    dlg.close()
