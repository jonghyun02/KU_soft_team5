"""파서 시험 자료 만들기: 실제 대본 lines.jsonl 에서 짧은 조각을 뽑아 tests/fixtures/parse/caseNN_*.jsonl 로 쓴다.

명세 4.4.3 까다로운 경우 11가지마다 조각 하나 이상. 머리 행 note 에 출처 doc_id 와 n 범위를 적는다.
줄 행은 원래 n 그대로 둔다(파서는 n 이 0부터가 아니어도, 띄엄띄엄이어도 동작해야 한다).

사용: .venv/Scripts/python -I tools/make_parse_fixtures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sinkprep import paths, util  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "parse"

# (파일 이름, doc_id, [(시작 n, 끝 n 포함)], 설명)
CASES = [
    ("case01_two_speakers_one_line", "fm55542", [(1760, 1785)],
     "#1 한 줄에 두 화자 '민우 : 괜찮습… / 장사장 : (동시에) 자, 펴.' (M, rtf)"),
    ("case01b_slash_speakers", "fm2405876", [(95, 113)],
     "#1 '태우/지웅 ⇥ (동시에) …' 두 화자 한 대사 (티끌모아 로맨스, 글자층 PDF)"),
    ("case01c_dual_columns", "fm2338951", [(3115, 3135)],
     "#1 좌우 두 칸 동시 대사 'ANNA ⇥ DECLAN' (Leap Year, 글자층 PDF)"),
    ("case02_speaker_tags", "fm3659631", [(20, 59)],
     "#2 화자 꼬리표 '표종성 (독일어)', '아심 (영어 소리)', '천세황 (소리)' (베를린, 글자층 PDF)"),
    ("case03_foreign_gloss", "fm416771", [(13, 48)],
     "#3 외국어 대사와 병기 번역 'Sir! Terrible accident! (심각한 문제가 발생했어요!)', 일동, P1 씬 번호 (해운대, doc)"),
    ("case03b_bilingual_slash", "fm36187123", [(614, 645)],
     "#3 원문/번역 병기 '와. / Woah.' (Past Lives, 영어 대본 속 한국어 대사)"),
    ("case04_onscreen_song_narration", "fm416771", [(177, 188), (654, 660), (770, 778)],
     "#4 노래 ♪(승현), 자막 <자막 : …>, 내레이션 '김  휘(Na)' (해운대, 세 구간)"),
    ("case05_paren_directions", "fm21201869", [(4, 38)],
     "#5 대사 안 괄호 지시와 소리 괄호 '(buzzer goes)', '(casual)' (Fleabag, 글자층 PDF)"),
    ("case06_scene_numbers", "fm54114", [(801, 806), (860, 882)],
     "#6 씬 번호 변형 'S# 34-1', 'S# 34-2', 'S#34-4' (행복한 장의사, hwp3)"),
    ("case06b_scene_letters", "fm56555", [(599, 628)],
     "#6 씬 번호 '17A', '17B' 좌우 반복 (Collateral, 글자층 PDF)"),
    ("case07_page_split", "fm21201869", [(39, 58), (91, 95), (131, 135), (176, 180)],
     "#7 쪽 넘김 (MORE) / 머리말 / 'FLEABAG (CONT'D)' (Fleabag, 네 구간: 머리말 되풀이를 보려고 여러 쪽)"),
    ("case08_ocr_quality", "fm5691757", [(0, 10), (34, 42), (67, 74), (99, 110)],
     "#8 OCR 문서(conf), 쪽 머리말 'Page n /81 …'·꼬리말 'ZIP CINEMA …', 대사 빠진 이름 줄 (감시자들, 스캔 PDF OCR, 네 구간)"),
    ("case09_name_variants", "fm54554", [(796, 801), (847, 852), (1883, 1900)],
     "#9 성 생략 '김법민'·'법민', 여러 화자 '법민, 흠순, 품일', 번호 엑스트라 '백제병사1(거시기)' (황산벌, hwp3, 세 구간)"),
    ("case10_front_castlist", "fm54897", [(0, 39)],
     "#10 앞머리 인물표 '수현: 29세의 남자.'(이름 : 대사처럼 보임), 번호 없는 씬 머리 '프롤로그 - … (실외. 낮)' (누구에게나 비밀은 있다, hwp5)"),
    ("case11_version_front", "fm5691168", [(0, 14)],
     "#11 판본 표기 'VER.5.1015'가 있는 앞머리 (터널, 글자층 PDF). 같은 영화 여러 판본은 catalog version_label·dup_group·is_primary"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, doc_id, ranges, desc in CASES:
        rows = util.read_jsonl(paths.DOCS / doc_id / "lines.jsonl")
        head, body = rows[0], rows[1:]
        pick = [r for r in body if any(a <= r["n"] <= b for a, b in ranges)]
        rng = ", ".join(f"{a}-{b}" for a, b in ranges)
        hdr = {"schema": "sink.lines/1", "doc_id": doc_id, "files": head.get("files"), "made": head.get("made"),
               "case": name[4:6], "note": f"{desc}. 출처 {doc_id} lines.jsonl n {rng} (실제 대본 조각, {len(pick)}줄)"}
        util.write_jsonl(OUT / f"{name}.jsonl", [hdr] + pick)
        print(f"{name}: {len(pick)}줄 ({doc_id} n {rng})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
