"""Unit & integration tests for Phase 1-B: Stage 2 Visual/OCR, Stage 3 VLM priors, Stage 4 CmGI Irony, Stage 5 Ranker/Pruner, and FastAPI."""

from __future__ import annotations

from pathlib import Path
import cv2
from fastapi.testclient import TestClient
import numpy as np
import torch

from mos_sdh.api.server import ProjectStore, create_app
from mos_sdh.config import AnalysisConfig
from mos_sdh.exporters.subtitle_exporter import build_ass_line, build_srt_line
from mos_sdh.schemas.document import (
    BenchmarkMetrics,
    EmotionIronyAnalysis,
    HonorificAnalysis,
    MoSDocument,
    ParalinguisticCue,
    SceneContext,
    SegmentMoS,
    SpeakerRelation,
    SubtitleCandidate,
    VisualContextSlot,
    WordUnit,
)
from mos_sdh.stage2_visual.face_lip_fer import analyze_faces_lips_and_expressions
from mos_sdh.stage2_visual.scene_detector import (
    detect_scenes_and_keyframes,
    ensure_ascii_safe_video_path,
)
from mos_sdh.stage2_visual.scene_ocr import run_scene_ocr_and_collision_detection
from mos_sdh.stage3_vlm.scene_vlm import _apply_vlm_json_to_scene
from mos_sdh.stage4_mos.context_consistency import evaluate_discourse_honorific_consistency
from mos_sdh.stage4_mos.trimodal_cmgi import (
    TrimodalCmGIFusion,
    analyze_trimodal_emotion_and_irony,
)
from mos_sdh.stage5_llm.candidate_generator import generate_sdh_candidates
from mos_sdh.stage5_llm.candidate_ranker import score_and_rank_candidates
from mos_sdh.stage5_llm.overstylization_pruner import prune_overstylized_cues


def test_trimodal_cmgi_zero_mask_and_sarcasm_irony_detection() -> None:
    cmgi = TrimodalCmGIFusion(d_model=16)
    xa = torch.randn(6, 16)
    xl = torch.randn(6, 16)
    xv = torch.randn(6, 16)

    pooled_no_face, gates_no_face = cmgi(xa, xl, xv, visual_mask=0.0)
    assert pooled_no_face.shape == (96,)
    assert gates_no_face["visual"] == 0.0
    assert abs(gates_no_face["audio"] + gates_no_face["lexical"] - 1.0) < 1e-3

    # Sarcastic Korean praise: positive lexical praise + flat/cold prosody + sigh + sneer FER
    seg_sarcasm = SegmentMoS(
        segment_id=1,
        start_ms=0,
        end_ms=3200,
        speaker_id="화자_2",
        raw_transcript="참 일 처리 한 번 깔끔하게 잘하시는군요",
        arousal_zscore=-0.95,
        words=[
            WordUnit(text="참", start_ms=0, end_ms=500, duration_per_syllable_ms=310.0),
            WordUnit(text="일", start_ms=520, end_ms=800, duration_per_syllable_ms=190.0),
            WordUnit(text="처리", start_ms=820, end_ms=1300, duration_per_syllable_ms=200.0),
            WordUnit(text="깔끔하게", start_ms=1350, end_ms=2150, duration_per_syllable_ms=200.0),
            WordUnit(text="잘하시는군요", start_ms=2200, end_ms=3200, duration_per_syllable_ms=185.0),
        ],
        cues=[ParalinguisticCue(cue_code="b/", label_ko="한숨", start_ms=3000, end_ms=3300)],
        visual=VisualContextSlot(
            face_detected=True,
            on_screen_speaker=True,
            facial_expression="냉소",
            expression_probs={"냉소": 0.72, "분노": 0.18, "중립": 0.10},
        ),
    )

    fer_seq = np.tile(np.array([[0.18, 0.0, 0.0, 0.0, 0.72]], dtype=np.float32), (5, 1))
    cfg = AnalysisConfig(irony_threshold=0.50)
    analyzed = analyze_trimodal_emotion_and_irony([seg_sarcasm], {1: fer_seq}, cfg)[0]

    assert analyzed.emotion_irony.lexical_polarity > 0.50
    assert analyzed.emotion_irony.audiovisual_polarity < -0.40
    assert analyzed.emotion_irony.irony_score >= 0.50
    assert analyzed.emotion_irony.irony_detected is True
    assert analyzed.emotion_irony.dominant_emotion == "냉소"
    assert "(비꼬는 어조로)" in build_srt_line(analyzed)


def test_vlm_relationship_priors_seed_honorific_consistency() -> None:
    sc = SceneContext(scene_id=1, start_ms=0, end_ms=5000, keyframe_ms=2500)
    raw_vlm_output = (
        '```json\n'
        '{"description_ko": "사무실에서 부하 직원이 팀장에게 보고하는 장면", '
        '"relations": [{"from_speaker": "화자_1", "to_speaker": "상대방", '
        '"relation_ko": "부하->팀장", "prior_speech_level": "합쇼체"}]}\n'
        '```'
    )
    _apply_vlm_json_to_scene(sc, raw_vlm_output)
    assert "사무실" in sc.description_ko
    assert len(sc.inferred_relations) == 1
    assert sc.inferred_relations[0].prior_speech_level == "합쇼체"

    # First utterance in scene has unintended 반말 slip with calm arousal -> caught immediately via VLM prior!
    seg1 = SegmentMoS(
        segment_id=1,
        start_ms=400,
        end_ms=2200,
        speaker_id="화자_1",
        raw_transcript="보고서 여기 있습니다",
        existing_subtitle="보고서 여기 있어",
        arousal_zscore=0.1,
    )
    evaluated = evaluate_discourse_honorific_consistency([seg1], AnalysisConfig(), scenes=[sc])[0]
    assert evaluated.honorific.expected_speech_level == "합쇼체"
    assert evaluated.honorific.consistency_status == "mismatch_error"
    assert evaluated.honorific.suggested_correction is not None
    assert evaluated.honorific.suggested_correction.endswith("습니다") or evaluated.honorific.suggested_correction.endswith("요")


def test_stage5_candidates_hallucination_guard_and_overstylization_pruner() -> None:
    cfg = AnalysisConfig(enable_llm_stage=False, max_cue_density_ratio=0.40)

    seg_irony = SegmentMoS(
        segment_id=1,
        start_ms=0,
        end_ms=3000,
        speaker_id="화자_1",
        raw_transcript="참 잘 하십니다",
        emotion_irony=EmotionIronyAnalysis(
            dominant_emotion="냉소",
            irony_score=0.78,
            irony_detected=True,
        ),
        words=[
            WordUnit(text="참", start_ms=0, end_ms=600),
            WordUnit(text="잘", start_ms=650, end_ms=1400, emphasis_level=3, emphasis_token="<toe_3>"),
            WordUnit(text="하십니다", start_ms=1450, end_ms=3000),
        ],
        honorific=HonorificAnalysis(speech_level="합쇼체", is_polite_group=True, consistency_status="consistent"),
    )

    # 3 extra segments with mild emotion directions so cue density starts at 4/4 = 1.0 > 0.40
    mild_segments = [
        SegmentMoS(
            segment_id=i,
            start_ms=i * 3000,
            end_ms=(i + 1) * 3000,
            speaker_id="화자_2",
            raw_transcript="오늘 회의 일정 확인했습니다",
            arousal_zscore=0.1 * i,
            emotion_irony=EmotionIronyAnalysis(dominant_emotion="기쁨", irony_detected=False),
            honorific=HonorificAnalysis(speech_level="합쇼체", is_polite_group=True, consistency_status="consistent"),
        )
        for i in (2, 3, 4)
    ]

    all_segs = [seg_irony] + mild_segments
    generate_sdh_candidates(all_segs, cfg, build_srt_line, build_ass_line)

    # Inject a completely hallucinated candidate into seg_irony to verify C_i < 0.80 guard rejection
    seg_irony.candidates.append(
        SubtitleCandidate(
            candidate_id=99,
            strategy="hallucinated_llm_test",
            srt_text="[화자_1] (비꼬는 어조로) 바다에서 고래가 수영하고 있습니다",
            ass_text="[화자_1] 바다에서 고래가 수영하고 있습니다",
            plain_dialogue="바다에서 고래가 수영하고 있습니다",
        )
    )

    score_and_rank_candidates(all_segs, cfg)
    hallucinated = next(c for c in seg_irony.candidates if c.candidate_id == 99)
    assert hallucinated.passed_guard is False
    assert hallucinated.selected is False

    pruned_segs, final_ratio = prune_overstylized_cues(all_segs, cfg)
    # Critical irony cue on segment 1 MUST survive pruning, while mild cues are pruned down to <= 0.40
    assert "(비꼬는 어조로)" in pruned_segs[0].sdh_srt_text
    assert final_ratio <= 0.50
    assert any(s.stage_direction_pruned for s in pruned_segs[1:])


def test_stage2_scene_detection_offscreen_asd_and_ocr_collision(tmp_path: Path) -> None:
    video_path = tmp_path / "ocr_bottom_test.mp4"
    width, height, fps = 640, 360, 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for _ in range(20):
        frame = np.full((height, width, 3), 25, dtype=np.uint8)
        # High-contrast white caption bar in bottom region (y = 290..345 => y_max = 0.958 >= 0.72)
        cv2.rectangle(frame, (80, 280), (560, 345), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "EPISODE 03 LIVE",
            (115, 328),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()

    seg = SegmentMoS(
        segment_id=1,
        start_ms=200,
        end_ms=1800,
        speaker_id="화자_1",
        raw_transcript="상황실 연결 완료했습니다",
    )
    cfg = AnalysisConfig(cache_dir=tmp_path / "cache")
    safe_vid = ensure_ascii_safe_video_path(video_path, cfg.cache_dir)
    assert safe_vid is not None

    scenes, kf = detect_scenes_and_keyframes(safe_vid, 2000, [seg], cfg.cache_dir)
    assert len(scenes) >= 1
    assert 1 in kf

    analyze_faces_lips_and_expressions(safe_vid, [seg], cfg)
    assert seg.visual.face_detected is False
    assert seg.visual.on_screen_speaker is False

    run_scene_ocr_and_collision_detection(safe_vid, [seg], scenes, cfg)
    assert seg.visual.avoid_bottom_region is True
    assert any("EPISODE" in t.upper() or "LIVE" in t.upper() for t in seg.visual.ocr_texts)

    srt_out = build_srt_line(seg)
    ass_out = build_ass_line(seg)
    assert "[화자_1(화면 밖)]" in srt_out
    assert r"{\an8}[화자_1(화면 밖)]" in ass_out


def test_fastapi_health_swer_and_human_in_the_loop_candidate_selection(tmp_path: Path) -> None:
    store = ProjectStore(root_dir=tmp_path / "projects")
    api_app = create_app(store=store)
    client = TestClient(api_app)

    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_swer = client.post(
        "/api/v1/eval-swer",
        json={
            "ref_lines": ["잠시만 기다려 주세요"],
            "hyp_lines": ["잠시만 기다려주세요"],
        },
    )
    assert r_swer.status_code == 200
    sw_body = r_swer.json()
    assert sw_body["wer_percent"] > 0.0
    assert sw_body["swer_percent"] == 0.0

    seg = SegmentMoS(
        segment_id=1,
        start_ms=0,
        end_ms=2500,
        speaker_id="화자_1",
        raw_transcript="지금 바로 갑니다",
        sdh_srt_text="[화자_1] 지금 바로 갑니다",
        sdh_ass_text="[화자_1] 지금 바로 갑니다",
        candidates=[
            SubtitleCandidate(
                candidate_id=1,
                strategy="positional_tag_standard",
                srt_text="[화자_1] 지금 바로 갑니다",
                ass_text="[화자_1] 지금 바로 갑니다",
                plain_dialogue="지금 바로 갑니다",
                semantic_ci=1.0,
                mos_fidelity=0.85,
                harmonic_oi=0.9189,
                selected=True,
            ),
            SubtitleCandidate(
                candidate_id=2,
                strategy="tokuyama_intensifier_insertion",
                srt_text="[화자_1] 정말 '지금' 바로 갑니다",
                ass_text=r"[화자_1] 정말 {\b1\fscx116\fscy116}지금{\r} 바로 갑니다",
                plain_dialogue="정말 지금 바로 갑니다",
                semantic_ci=0.94,
                mos_fidelity=0.92,
                harmonic_oi=0.9299,
                selected=False,
            ),
        ],
    )
    doc = MoSDocument(
        mode="mode_a_video_only",
        input_media_path="demo.mp4",
        duration_ms=2500,
        segments=[seg],
        metrics=BenchmarkMetrics(total_segments=1, speaker_count=1),
    )
    proj_out = tmp_path / "proj_out"
    store.register(doc, proj_out, project_id="demo123")

    r_get = client.get("/api/v1/projects/demo123")
    assert r_get.status_code == 200
    assert r_get.json()["segments"][0]["sdh_srt_text"] == "[화자_1] 지금 바로 갑니다"

    r_sel = client.post(
        "/api/v1/projects/demo123/segments/1/select-candidate",
        json={"candidate_id": 2},
    )
    assert r_sel.status_code == 200
    updated_seg = r_sel.json()["segment"]
    assert updated_seg["sdh_srt_text"] == "[화자_1] 정말 '지금' 바로 갑니다"
    assert (proj_out / "prototype_sdh.srt").exists()
    assert "정말 '지금' 바로 갑니다" in (proj_out / "prototype_sdh.srt").read_text(encoding="utf-8")
