"""Tests for Phase 1-C: MoS-SDH Studio v1.0 Web UI static serving, media streaming, exports, manual edits, and live re-pruning."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from mos_sdh.api.server import ProjectStore, create_app
from mos_sdh.schemas.document import (
    BenchmarkMetrics,
    EmotionIronyAnalysis,
    HonorificAnalysis,
    MoSDocument,
    SegmentMoS,
    SubtitleCandidate,
)


def test_studio_static_ui_media_exports_and_live_editing(tmp_path: Path) -> None:
    dummy_mp4 = tmp_path / "clip.mp4"
    dummy_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

    store = ProjectStore(root_dir=tmp_path / "projects", auto_discover=False)
    app = create_app(store=store)
    client = TestClient(app)

    # 1. Verify Studio HTML, CSS, and JS assets serve cleanly
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "MoS-SDH Studio" in r_index.text

    r_css = client.get("/static/styles.css")
    assert r_css.status_code == 200
    assert "--bg-app" in r_css.text

    r_js = client.get("/static/app.js")
    assert r_js.status_code == 200
    assert "initStudio" in r_js.text

    # 2. Register a multi-segment project with mild & irony directions
    seg1 = SegmentMoS(
        segment_id=1,
        start_ms=0,
        end_ms=2500,
        speaker_id="화자_1",
        raw_transcript="참 잘 하십니다",
        sdh_srt_text="[화자_1] (비꼬는 어조로) 참 잘 하십니다",
        sdh_ass_text=r"[화자_1] {\c&HFFCC66&\i1}(비꼬는 어조로){\i0\r} 참 잘 하십니다",
        emotion_irony=EmotionIronyAnalysis(dominant_emotion="냉소", irony_score=0.82, irony_detected=True),
        honorific=HonorificAnalysis(speech_level="합쇼체", is_polite_group=True, consistency_status="consistent"),
        candidates=[
            SubtitleCandidate(
                candidate_id=1,
                strategy="positional_tag_standard",
                srt_text="[화자_1] (비꼬는 어조로) 참 잘 하십니다",
                ass_text=r"[화자_1] {\c&HFFCC66&\i1}(비꼬는 어조로){\i0\r} 참 잘 하십니다",
                plain_dialogue="참 잘 하십니다",
                selected=True,
            )
        ],
    )
    seg2 = SegmentMoS(
        segment_id=2,
        start_ms=2600,
        end_ms=5200,
        speaker_id="화자_1",
        raw_transcript="일정 확인했습니다",
        sdh_srt_text="[화자_1] (밝은 목소리로) 일정 확인했습니다",
        sdh_ass_text="[화자_1] 일정 확인했습니다",
        emotion_irony=EmotionIronyAnalysis(dominant_emotion="기쁨", irony_detected=False),
        honorific=HonorificAnalysis(speech_level="합쇼체", is_polite_group=True, consistency_status="consistent"),
        candidates=[
            SubtitleCandidate(
                candidate_id=1,
                strategy="positional_tag_standard",
                srt_text="[화자_1] (밝은 목소리로) 일정 확인했습니다",
                ass_text="[화자_1] 일정 확인했습니다",
                plain_dialogue="일정 확인했습니다",
                selected=True,
            )
        ],
    )
    doc = MoSDocument(
        mode="mode_a_video_only",
        input_media_path=str(dummy_mp4),
        duration_ms=5500,
        segments=[seg1, seg2],
        metrics=BenchmarkMetrics(total_segments=2, speaker_count=1),
    )
    out_dir = tmp_path / "studio_proj"
    store.register(doc, out_dir, project_id="studio01")

    # 3. Test media streaming & file download endpoints
    r_media = client.get("/api/v1/projects/studio01/media")
    assert r_media.status_code == 200
    assert b"ftypmp42" in r_media.content

    for fmt in ("srt", "vtt", "ass", "md", "json"):
        r_dl = client.get(f"/api/v1/projects/studio01/export/{fmt}")
        assert r_dl.status_code == 200
        assert len(r_dl.text) > 10

    # 4. Test live over-stylization re-pruning (cap=0.45 -> mild direction on seg2 pruned, irony on seg1 kept!)
    r_prune = client.post(
        "/api/v1/projects/studio01/prune",
        json={"max_cue_density_ratio": 0.45, "recommended_max_cps": 14.0},
    )
    assert r_prune.status_code == 200
    pruned_doc = r_prune.json()
    assert "(비꼬는 어조로)" in pruned_doc["segments"][0]["sdh_srt_text"]
    assert "(밝은 목소리로)" not in pruned_doc["segments"][1]["sdh_srt_text"]

    # 5. Test manual segment edit & speaker rename via PATCH
    r_patch = client.patch(
        "/api/v1/projects/studio01/segments/1",
        json={"speaker_id": "박 부장", "sdh_srt_text": "[박 부장] (비꼬는 어조로) 참, 잘도 하십니다."},
    )
    assert r_patch.status_code == 200
    assert r_patch.json()["segment"]["speaker_id"] == "박 부장"
    srt_disk = (out_dir / "prototype_sdh.srt").read_text(encoding="utf-8")
    assert "[박 부장] (비꼬는 어조로) 참, 잘도 하십니다." in srt_disk
