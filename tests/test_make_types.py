"""tools/make_types.py 의 대사 표기 재기(dialogue_form)와 특수 분기 읽기."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import make_types as M  # noqa: E402


def el(speaker_raw, raw, **kw):
    return dict({"type": "dialogue", "speaker_raw": speaker_raw, "raw": raw}, **kw)


@pytest.mark.parametrize("speaker_raw, raw, want", [
    ("철수", "철수\n어디 가?", ("own_line", "")),
    ("JOHN (V.O.)", "JOHN (V.O.)\nWhere are you going?", ("own_line", "")),
    ("철수", "철수", ("own_line", "")),                                    # 이름표만 있는 대사
    ("철수", "철수\t어디 가?", ("tab", "")),
    ("철수 (웃으며)", "철수\t(웃으며) 어디 가?", ("tab", "")),            # speaker_raw 에 꼬리 괄호가 붙어도 이름 뒤 탭을 본다
    ("조 원", "조 원(V.O)\t부인, 어디 가오?", ("tab", "letterspaced")),
    ("철수", "철수 : 어디 가?", ("colon", "")),
    ("철수", "철수:어디 가?", ("colon", "")),
    ("철수", "철수/ 어디 가?", ("colon", "")),                             # 빗금·쌍반점은 파서가 쌍점 표기로 받는다
    ("철수", "철수; 어디 가?", ("colon", "")),
    ("철 수", "철 수 : 어디 가?", ("colon", "letterspaced")),
    ("철수", "철수 :\t어디 가?", ("tab", "")),                             # 쌍점 뒤 탭: 파서는 탭을 먼저 본다
    ("철수", "철수 어디 가?", ("space1", "")),
    ("철수 (화난 듯)", "철수 (화난 듯) 어디 가?", ("space1", "")),
    ("철수", "철수 「어디 가?」", ("space1", "")),                          # 파서가 따옴표로 받지 않는 글자
    ("철수", "철수   어디 가?", ("space2", "")),
    ("철수", "철수  ''어디 가?''", ("space2", "")),                        # 공백 둘이 따옴표보다 먼저
    ("철수", "철수\"어디 가?\"", ("quote", "")),
    ("철수 (겁에 질려)", "철수 (겁에 질려) “어디 가?”", ("quote", "")),
    ("철수", "[철수] 어디 가?", ("bracket", "")),
    ("철수", "<철수>어디 가?", ("angle", "")),
    ("<철수>", "<철수>\n어디 가?", ("angle", "")),                         # speaker_raw 에 꺾쇠가 남아 있어도 같은 표기
    ("철수", "*철수\t어디 가?", ("tab", "bullet")),
    ("철수", "철수어디 가?", ("joined", "")),
    ("철수", "영희\t어디 가?", ("other", "")),                             # 첫머리에서 이름을 찾지 못함
])
def test_dialogue_form(speaker_raw, raw, want):
    assert M.dialogue_form(el(speaker_raw, raw)) == want


def test_dialogue_form_skips_unnamed_and_inferred():
    assert M.dialogue_form(el(None, "어디 가?")) is None
    assert M.dialogue_form(el("철수", "어디 가?", speaker_inferred=True)) is None


def test_branch_patterns_match_parser_notes():
    notes = {
        "flat_names": "대사 표기: 들여쓰기 없는 이름 줄 356(빈 줄 앞까지 대사)",
        "space1_vocab": "대사 표기: 이름 뒤 공백 하나(이름 12개를 줄 첫 낱말 빈도로 찾음)",
        "letterspaced_names": "대사 표기: 글자 사이를 띄운 이름 251줄(민석, 은비)",
        "collapsed_cues": "대사 표기: 대사가 지문 여백에 놓인 대문자 이름 줄 40개(대사는 문단 간격 전까지)",
        "shifted_pages": "대사 표기: 이름 줄 938; 지문 여백 x=0.2, 단계 [(0.2, 637)], 쪽 밀림 맞춤 12쪽",
        "scene_restart": "씬 번호가 784번 줄에서 1로 다시 시작: 앞 번호 159개는 앞머리(인물표·설정)로 봄",
        "vo_resolved": "(V.O.): 같은 씬에 그 인물의 다른 대사가 있으면 off(8), 없으면 narration(18)",
        "subtitle_file": "자막 파일 꼴(SRT·WebVTT): 화살표 시각 줄과 자막 번호 1592줄을 dropped(timecode·subtitle_cue)",
    }
    rx = dict(M._BRANCH_RE)
    for key, note in notes.items():
        assert rx[key].search(note), key
        others = [k for k, r in M._BRANCH_RE if k != key and r.search(note)]
        assert others == [], (key, others)
    plain = "대사 표기: 이름⇥대사 394, 이름 : 대사 3; 지문 여백 x=0.205, 단계 [(0.205, 637)]"
    assert [k for k, r in M._BRANCH_RE if r.search(plain)] == []
