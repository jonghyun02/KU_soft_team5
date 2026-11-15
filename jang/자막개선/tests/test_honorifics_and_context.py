"""Unit tests for Korean 3-axis honorifics, 6-speech-levels, and K=3000ms discourse consistency."""

from __future__ import annotations

import pytest

from mos_sdh.config import AnalysisConfig
from mos_sdh.schemas.document import SegmentMoS, WordUnit
from mos_sdh.stage4_mos.context_consistency import evaluate_discourse_honorific_consistency
from mos_sdh.stage4_mos.korean_honorifics import (
    analyze_utterance_honorifics,
    convert_speech_level_cape,
)


@pytest.mark.parametrize(
    ("sentence", "expected_level", "expected_polite"),
    [
        ("지금 바로 보고드리겠습니다.", "합쇼체", True),
        ("잠시만 여기서 기다려요.", "해요체", True),
        ("내 생각은 그리하오.", "하오체", False),
        ("자네도 이제 그만 돌아가게.", "하게체", False),
        ("난 오늘 먼저 갈게.", "반말체", False),
        ("나는 지금 집으로 간다.", "해라체", False),
    ],
)
def test_six_speech_levels_classification(
    sentence: str,
    expected_level: str,
    expected_polite: bool,
) -> None:
    hon = analyze_utterance_honorifics(sentence)
    assert hon.speech_level == expected_level
    assert hon.is_polite_group is expected_polite


def test_subject_and_object_honorific_axes() -> None:
    subj = analyze_utterance_honorifics("아버지께서 방에 들어가셨어요.")
    assert subj.subject_honorific is True
    assert subj.speech_level == "해요체"

    obj = analyze_utterance_honorifics("선생님께 먼저 여쭈어 보았습니다.")
    assert obj.object_honorific is True
    assert obj.speech_level == "합쇼체"


def test_context_separates_mismatch_error_from_dramatic_shift() -> None:
    cfg = AnalysisConfig(context_window_ms=3000, dramatic_arousal_z_threshold=1.15)

    segments = [
        # Turn 1: Subordinate speaks politely (해요체)
        SegmentMoS(
            segment_id=1,
            start_ms=0,
            end_ms=1500,
            speaker_id="화자_1",
            raw_transcript="부장님 말씀대로 수정했어요.",
            arousal_zscore=0.1,
            words=[WordUnit(text="수정했어요.", start_ms=900, end_ms=1500, confidence=0.95)],
        ),
        # Turn 2 (within K=3000ms): Low-arousal, low-confidence ASR slip into 반말체 -> mismatch_error!
        SegmentMoS(
            segment_id=2,
            start_ms=2100,
            end_ms=3400,
            speaker_id="화자_1",
            raw_transcript="잠시만 기다려",
            arousal_zscore=0.2,
            words=[WordUnit(text="기다려", start_ms=2700, end_ms=3400, confidence=0.58)],
        ),
        # Turn 3: High-arousal emotional outburst in 반말체 -> dramatic_shift!
        SegmentMoS(
            segment_id=3,
            start_ms=4200,
            end_ms=5800,
            speaker_id="화자_1",
            raw_transcript="더 이상은 못 참아!",
            arousal_zscore=1.85,
            words=[WordUnit(text="참아!", start_ms=5000, end_ms=5800, confidence=0.96, emphasis_level=4)],
        ),
    ]

    analyzed = evaluate_discourse_honorific_consistency(segments, cfg)

    assert analyzed[1].honorific.consistency_status == "mismatch_error"
    assert analyzed[1].honorific.suggested_correction is not None
    assert analyzed[1].honorific.suggested_correction.endswith("요")

    assert analyzed[2].honorific.consistency_status == "dramatic_shift"
    assert analyzed[2].honorific.suggested_correction is None


def test_cape_speech_level_conversion() -> None:
    polite = convert_speech_level_cape("지금 바로 확인해", "해요체")
    assert polite.endswith("요")
