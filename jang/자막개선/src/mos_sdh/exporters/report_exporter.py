"""JSON and Markdown quantitative diagnostic report exporters across all 5 MoS stages."""

from __future__ import annotations

from pathlib import Path

from mos_sdh.schemas.document import MoSDocument


def export_reports(doc: MoSDocument, out_dir: Path) -> dict[str, Path]:
    """Write `mos_report.json` and human-readable `mos_report.md`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "mos_report.json"
    json_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")

    md_path = out_dir / "mos_report.md"
    md_path.write_text(_render_markdown_report(doc), encoding="utf-8")

    return {"json": json_path, "markdown": md_path}


def _fmt_ms(ms: int) -> str:
    total_sec, milli = divmod(max(0, ms), 1000)
    minutes, seconds = divmod(total_sec, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milli:03d}"


def _render_markdown_report(doc: MoSDocument) -> str:
    m = doc.metrics
    lines: list[str] = [
        "# MoS-SDH Studio 5-Stage 정량 화법(MoS) 진단 리포트",
        "",
        f"- **버전 / 실행 모드**: `v{doc.version}` / `{doc.mode}`",
        f"- **입력 미디어**: `{doc.input_media_path}`",
        f"- **입력 자막**: `{doc.input_subtitle_path or '없음 (영상 단독 End-to-End 생성)'}`",
        f"- **총 재생 길이**: `{_fmt_ms(doc.duration_ms)}` ({doc.duration_ms} ms)",
        "",
        "## 1. 핵심 MoS 요약 지표 (Stages 1~5)",
        "",
        "| 항목 | 측정치 | 근거 모델 / 논문 |",
        "| :--- | :--- | :--- |",
        f"| 총 씬(Scene) / 대사 세그먼트 / 화자 수 | {m.scene_count}씬 / {m.total_segments}개 / {m.speaker_count}명 | PySceneDetect + Silero VAD + 음향 군집화 |",
        f"| 경어·화계 문맥 일치율 | **{m.honorific_consistency_rate:.2f}%** | *Hwang et al. (2021)* $K=3000\\text{{ms}}$ + VLM 관계 사전확률 |",
        f"| 화계 오인식 의심 (`mismatch_error`) | {m.mismatch_error_count}건 | CAPE 형태소 교정 대상 |",
        f"| 의도적 극적 전환 (`dramatic_shift`) | {m.dramatic_shift_count}건 | 고각성 반말/존댓말 전환 보존 |",
        f"| 반어·비꼼 불일치 탐지 (`irony_detected`) | **{m.irony_segment_count}건** | *Gao et al. (2024)* Trimodal CmGI + $\\mathcal{{I}}_{{\\text{{irony}}}}$ |",
        f"| 화면 밖 발화(`ASD`) / 하단 OCR 충돌 회피 | {m.off_screen_segment_count}건 / {m.ocr_collision_avoided_count}건 | 구순 활성도 추적 + EasyOCR (`{{\\an8}}`) |",
        f"| 음향 강조 단어 (`<toe_1>`~`<toe_4>`) | 총 {m.emphasized_word_count}개 (강한 강조 {m.strong_emphasis_word_count}개) | *Tokuyama et al. (2021)* |",
        f"| 준언어·배경 단서 (`b/, l/, sfx`) | {m.paralinguistic_cue_count}건 | *Bang et al. (2020)* |",
        f"| 의미 보존도 평균 (`C_i`) / 조화 점수 (`O_i`) | `{m.mean_semantic_ci:.4f}` / `{m.mean_harmonic_oi:.4f}` | *de Rivero et al. (2021)* 식 (3), (5) |",
        f"| 평균 읽기 속도 (`CPS`) / 준수율 / 지문 밀도 | {m.mean_cps:.2f}자/초 / {m.cps_compliance_rate:.2f}% / {m.cue_attachment_ratio * 100:.1f}% | 권장 $\\le 14$자/초 & 과잉 지문 억제 |",
        f"| 거시 화법 강도 (`Score_text`) | {m.score_text_intensity:.2f} / 100 | *Sun et al. (2024)* 식 (2) |",
        f"| 5구간 화법 분포 (`Score_dist`) | `{m.score_dist_5bin}` | *Sun et al. (2024)* 식 (3) |",
    ]

    if m.swer_comparison is not None:
        sw = m.swer_comparison
        lines.extend(
            [
                "",
                "## 2. 기존 자막 대 원음 전사 정합성 (`Bang et al., 2020` sWER)",
                "",
                f"- **비교 세그먼트 수**: {sw.compared_utterances}문장",
                f"- **음절 오류율 (`CER`)**: `{sw.cer_percent:.2f}%`",
                f"- **일반 어절 오류율 (`WER`)**: `{sw.wer_percent:.2f}%`",
                f"- **공백 정규화 어절 오류율 (`sWER`)**: **`{sw.swer_percent:.2f}%`**",
            ]
        )

    lines.extend(
        [
            "",
            "## 3. 6화계 및 시청각 정서(CmGI) 분포",
            "",
            "### 3.1 한국어 6화계 분포",
        ]
    )
    for lvl, cnt in sorted(m.speech_level_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- **{lvl}**: {cnt}문장")

    if m.emotion_counts:
        lines.extend(["", "### 3.2 Trimodal CmGI 정서·냉소 분포"])
        for emo, cnt in sorted(m.emotion_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- **{emo}**: {cnt}문장")

    if doc.scenes:
        lines.extend(
            [
                "",
                "## 4. 씬(Scene) 맥락 및 VLM 추론 사회적 관계도 (Stages 2~3)",
                "",
                "| 씬 # | 시간 구간 | 장면 요약 (`description_ko`) | 화면 OCR 텍스트 | 추론된 화자 관계 (`prior_speech_level`) |",
                "| ---: | :--- | :--- | :--- | :--- |",
            ]
        )
        for sc in doc.scenes:
            sc_range = f"{_fmt_ms(sc.start_ms)}~{_fmt_ms(sc.end_ms)}"
            ocr_txt = ", ".join(sc.ocr_summary).replace("|", "\\|") if sc.ocr_summary else "-"
            rel_txt = (
                "; ".join(
                    f"{r.from_speaker}→{r.to_speaker}: {r.relation_ko} ({r.prior_speech_level})"
                    for r in sc.inferred_relations
                ).replace("|", "\\|")
                if sc.inferred_relations
                else "-"
            )
            desc = (sc.description_ko or "-").replace("|", "\\|")
            lines.append(f"| {sc.scene_id} | `{sc_range}` | {desc} | {ocr_txt} | {rel_txt} |")

    if doc.vram_telemetry:
        lines.extend(
            [
                "",
                "## 5. 로컬 GPU 순차 스테이지 VRAM 텔레메트리",
                "",
                "| 스테이지 | 디바이스 | 진입 전 (MB) | 피크 점유 (MB) | 회수 후 (MB) |",
                "| :--- | :--- | ---: | ---: | ---: |",
            ]
        )
        for t in doc.vram_telemetry:
            lines.append(
                f"| {t.stage_name} | {t.gpu_name} (`{t.device}`) | {t.allocated_before_mb:.1f} | "
                f"**{t.peak_allocated_mb:.1f}** | {t.allocated_after_mb:.1f} |"
            )

    lines.extend(
        [
            "",
            "## 6. 타임라인 세그먼트별 5-Stage MoS 상세 분석",
            "",
            "| # | 시간 구간 | 화자/위치 | 6화계 (판정) | 정서 / 반어(`I_irony`) | 선택 후보(`O_i`) | 최종 배리어프리 자막 (`.srt`) |",
            "| ---: | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for seg in doc.segments:
        t_range = f"{_fmt_ms(seg.start_ms)}~{_fmt_ms(seg.end_ms)}"
        loc_note = "화면밖" if seg.visual.on_screen_speaker is False else seg.visual.speaker_zone
        if seg.visual.avoid_bottom_region:
            loc_note += "/상단회피"
        spk_col = f"{seg.speaker_id} (`{loc_note}`)"
        h_label = f"{seg.honorific.speech_level} (`{seg.honorific.consistency_status}`)"
        emo_col = f"{seg.emotion_irony.dominant_emotion} (`{seg.emotion_irony.irony_score:.2f}`)"
        if seg.emotion_irony.irony_detected:
            emo_col = f"**{emo_col} [반어]**"

        sel_cand = next((c for c in seg.candidates if c.selected), None)
        cand_col = (
            f"`{sel_cand.strategy}` (`O={sel_cand.harmonic_oi:.2f}`, `C={sel_cand.semantic_ci:.2f}`)"
            if sel_cand is not None
            else "-"
        )
        sdh_line = seg.sdh_srt_text.replace("|", "\\|")
        lines.append(
            f"| {seg.segment_id} | `{t_range}` | {spk_col} | {h_label} | {emo_col} | {cand_col} | {sdh_line} |"
        )

    lines.append("")
    return "\n".join(lines)
