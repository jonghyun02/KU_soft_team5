"""tools/make_types.py 의 대사 표기 재기(dialogue_form), 특수 분기 읽기, 공백 복원 칸(spacing)."""
import json
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
        # parse 0.3.x 의 새 꼬리표(추격자 fm55664 등 다시 만든 문서의 notes 꼴)
        "scene_colon_heads": "씬 머리: 쌍점이 든 번호 씬 머리 53개를 받음(받아들인 씬 머리와 같은 칸, 빈 번호를 차례대로 메움)",
        "wordless_turns": "대사 표기: 말 없이 점·괄호만 있는 한 줄 차례 56줄은 이름 줄이 아니라 한 줄 대사",
        "paren_names": "대사 표기: 괄호에 든 이름 (이름)⇥대사 24줄(그 인물의 화면 밖 목소리, off; 이름표에 없는 이름 3줄 포함)",
        "tight_cont": "대사 이어짐: 지문 앞에 빈 줄을 두는 문서(대사 다음 지문 120줄 중 붙여 쓴 것 3줄): "
                      "대사 바로 아래 붙여 쓴 들여쓴 줄은 앞 대사의 이어짐",
    }
    rx = dict(M._BRANCH_RE)
    for key, note in notes.items():
        assert rx[key].search(note), key
        others = [k for k, r in M._BRANCH_RE if k != key and r.search(note)]
        assert others == [], (key, others)
    plain = "대사 표기: 이름⇥대사 394, 이름 : 대사 3; 지문 여백 x=0.205, 단계 [(0.205, 637)]"
    assert [k for k, r in M._BRANCH_RE if r.search(plain)] == []


def test_new_branch_notes_are_the_ones_the_parser_writes():
    """parse 0.3.x 꼬리표는 sinkprep/parse.py 가 quality.notes 에 적는 글 머리와 글자까지 같다(파서 글이 바뀌면 여기서 걸린다)."""
    src = (Path(__file__).resolve().parent.parent / "sinkprep" / "parse.py").read_text(encoding="utf-8")
    pats = {k: p for k, p, _, _ in M.BRANCHES}
    assert set(M.PARSE_03_BRANCHES) <= set(pats)
    for k in M.PARSE_03_BRANCHES:
        lit = pats[k].lstrip("^")
        assert lit == M.re.escape(lit).replace("\\", "") and lit in src, k


# ------------------------------------------------------------------ 공백 복원(lines made.spacing)
def _row(doc_id="fm1"):
    return {"doc_id": doc_id, "category": "ko_film", "title_norm": "시험", "real_format": "hwp5", "files_list": [],
            "file_formats_list": [], "script_lang": "ko", "is_translation": "false", "version_label": "unknown",
            "is_primary": "true", "note": ""}


def _put_doc(docs: Path, doc_id: str, spacing=None, notes=None, version="0.4.1"):
    made = {"stage": "lines", "version": version, "sinkprep": "0"}
    if spacing is not None:
        made["spacing"] = spacing
    d = docs / doc_id
    d.mkdir(parents=True)
    head = {"schema": "sink.lines/1", "doc_id": doc_id, "files": [], "made": made}
    (d / "lines.jsonl").write_text(json.dumps(head, ensure_ascii=False) + "\n"
                                   + json.dumps({"n": 1, "text": "가", "src": "hwp", "file": 0}) + "\n", encoding="utf-8")
    if notes is not None:
        s = {"schema": "sink.script/1", "doc_id": doc_id, "parser": {"name": "sink-parse", "version": "0.3.1"},
             "scenes": [], "elements": [], "quality": {"score": 1.0, "notes": notes}}
        (d / "script.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")


def test_spacing_column_reads_lines_head(tmp_path, monkeypatch):
    monkeypatch.setattr(M.paths, "DOCS", tmp_path)
    src = {"file": 0, "from": "source", "loss": 0.3089, "matched": 0.98, "lines": 980, "spaces": 3700, "tabs": 260,
           "removed": 1, "lowered": 26}
    _put_doc(tmp_path, "fm1", [src])
    _put_doc(tmp_path, "fm2", [{"file": 0, "from": None, "skipped": "source_missing"}])
    _put_doc(tmp_path, "fm3")                                         # 기록 없음: 견줬고 문턱 아래
    _put_doc(tmp_path, "fm4", [src, {"file": 1, "from": None, "skipped": "few_bounds:12"}])
    _put_doc(tmp_path, "fm5", version="0.3.0")                        # 공백 손실을 재기 전 판
    got = {d: M.doc_types(_row(d), {}, {}) for d in ("fm1", "fm2", "fm3", "fm4", "fm5", "fm9")}
    assert {d: t["spacing"] for d, t in got.items()} == {"fm1": "source", "fm2": "skipped", "fm3": "", "fm4": "source+skipped",
                                                         "fm5": "", "fm9": ""}
    assert got["fm1"]["_spacing_loss"] == [0.3089] and got["fm2"]["_spacing_loss"] == []
    assert got["fm5"]["_lines_version"] == "0.3.0" and got["fm9"]["_lines_version"] == ""
    assert M.lines_head("fm9") == {}


def test_types_csv_keeps_old_columns_and_adds_spacing_last():
    old = ["doc_id", "category", "title", "real_format", "files", "route", "convert_method", "ocr_why", "two_up",
           "script_lang", "is_translation", "version_label", "content", "is_primary", "parser", "lines", "scenes",
           "dialogues", "dialogue_form", "form_share", "name_deco", "parser_branches", "score", "needs_review"]
    assert M.CSV_FIELDS[:len(old)] == old and M.CSV_FIELDS[len(old):] == ["spacing"]


def test_new_parser_branches_land_in_the_row(tmp_path, monkeypatch):
    monkeypatch.setattr(M.paths, "DOCS", tmp_path)
    notes = ["씬 머리: 쌍점이 든 번호 씬 머리 53개를 받음(받아들인 씬 머리와 같은 칸, 빈 번호를 차례대로 메움)",
             "대사 표기: 말 없이 점·괄호만 있는 한 줄 차례 56줄은 이름 줄이 아니라 한 줄 대사",
             "대사 표기: 괄호에 든 이름 (이름)⇥대사 24줄(그 인물의 화면 밖 목소리, off)",
             "대사 표기: 이름⇥대사 1007, 이름 따옴표 대사 4; 지문 여백 x=0.1667, 단계 [(0.167, 1500)]"]
    _put_doc(tmp_path, "fm1", notes=notes)
    _put_doc(tmp_path, "fm2", notes=["대사 이어짐: 지문 앞에 빈 줄을 두는 문서(대사 다음 지문 120줄 중 붙여 쓴 것 3줄): "
                                     "대사 바로 아래 붙여 쓴 들여쓴 줄은 앞 대사의 이어짐"])
    assert M.doc_types(_row("fm1"), {}, {})["parser_branches"] == "scene_colon_heads|paren_names|wordless_turns"
    assert M.doc_types(_row("fm2"), {}, {})["parser_branches"] == "tight_cont"
