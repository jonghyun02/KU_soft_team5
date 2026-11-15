"""sinkprep.parse 시험 (US-007, US-008).

tests/fixtures/parse/caseNN_*.jsonl 은 실제 대본 lines.jsonl 에서 뽑은 조각이다(머리 행 note 에 출처 doc_id·n 범위).
조각마다 명세 4.4.3 까다로운 경우 11가지가 4.4.2 형식대로 표현되는지 단언한다. 공통으로 모든 조각에 대해
스키마(B1·B3)·줄 덮기(B2)·참조 규칙을 본다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sinkprep import parse as P
from sinkprep import parse_text as T
from sinkprep import validate as V

FX = Path(__file__).resolve().parent / "fixtures" / "parse"
ALL = sorted(FX.glob("case*.jsonl"))


# ---------------------------------------------------------------- 도구

def load(name: str) -> tuple[dict, list[dict]]:
    p = next(FX.glob(f"{name}*.jsonl"))
    rows = [json.loads(s) for s in jsonl_rows(p)]
    return rows[0], rows[1:]


def jsonl_rows(p: Path) -> list[str]:
    # splitlines() 는 글 안의 U+2029(Past Lives 글자층)도 줄로 나눠 버린다: '\n' 으로만 나눈다
    return [s for s in p.read_text(encoding="utf-8").split(chr(10)) if s.strip()]


_CACHE: dict[str, tuple[dict, list[dict]]] = {}


def parsed(name: str) -> dict:
    if name not in _CACHE:
        head, rows = load(name)
        _CACHE[name] = (P.parse_rows(head["doc_id"], rows), rows)
    return _CACHE[name][0]


def el(s: dict, n: int) -> dict:
    for e in s["elements"]:
        if n in e["lines"]:
            return e
    raise KeyError(f"줄 {n} 이 요소에 없음")


def why(s: dict, n: int) -> str | None:
    for d in s["dropped"]:
        if n in d["lines"]:
            return d["why"]
    return None


def names(s: dict, e: dict) -> list[str]:
    by = {p["id"]: p for p in s["people"]}
    return [by[x]["name"] for x in e["speakers"]]


def person(s: dict, pid: str) -> dict:
    return next(p for p in s["people"] if p["id"] == pid)


# ---------------------------------------------------------------- 모든 조각 공통: 형식 규칙

def test_fixtures_exist_for_all_11_cases():
    cases = {p.name[4:6] for p in ALL}
    assert cases >= {f"{k:02d}" for k in range(1, 12)}
    for p in ALL:
        head = json.loads(jsonl_rows(p)[0])
        assert head["schema"] == "sink.lines/1" and head["doc_id"].startswith("fm")
        assert "출처" in head["note"] and "n " in head["note"]
        n_rows = len(jsonl_rows(p)) - 1
        assert 10 <= n_rows <= 40, (p.name, n_rows)


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_fixture_schema_and_line_cover(path: Path, tmp_path: Path):
    head, rows = load(path.stem)
    s = P.parse_rows(head["doc_id"], rows)
    # B2: 모든 줄이 요소 또는 dropped 에 정확히 한 번
    seen: list[int] = [n for e in s["elements"] for n in e["lines"]] + [n for d in s["dropped"] for n in d["lines"]]
    assert sorted(seen) == sorted(r["n"] for r in rows)
    # B1·B3·참조: 스키마와 의미 규칙(validate_script, 줄 덮기는 위에서 직접 봄)
    out = tmp_path / "script.json"
    out.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    assert V.validate_script(out) == []
    assert {e["type"] for e in s["elements"]} <= {"front", "scene", "action", "dialogue", "onscreen", "transition",
                                                    "unknown"}
    # paren 위치는 text 안
    for e in s["elements"]:
        if e["type"] == "dialogue":
            assert all(0 <= p["at"] <= len(e["text"]) for p in e["paren"])
    # 결정적
    assert P.parse_rows(head["doc_id"], rows) == s


# ---------------------------------------------------------------- 4.4.3 #1 한 대사에 화자 여럿

def test_case01_two_speakers_in_one_line():
    s = parsed("case01_two")
    e = el(s, 1771)                     # '민우 : 괜찮습… / 장사장 : (동시에) 자, 펴.'
    assert e["type"] == "dialogue"
    assert sorted(names(s, e)) == ["민우", "장사장"]
    assert e["overlap"] is not None
    assert any(p["text"] == "동시에" for p in e["paren"])
    assert "자, 펴." in e["text"] and "장사장" not in e["text"]


def test_case01_slash_speakers_and_simultaneous():
    s = parsed("case01b")
    e = el(s, 112)                      # '태우/지웅 ⇥ (동시에) 너는 여자 있잖아 새끼야!!!'
    assert sorted(names(s, e)) == ["지웅", "태우"]
    assert e["speaker_raw"] == "태우/지웅"
    assert e["paren"] == [{"text": "동시에", "at": 0}]
    assert e["overlap"] is not None
    assert e["text"] == "너는 여자 있잖아 새끼야!!!"


def test_case01_dual_columns():
    s = parsed("case01c")
    e = el(s, 3127)                     # 'ANNA ⇥ DECLAN' 두 칸 동시 대사
    assert e["type"] == "dialogue" and 3128 in e["lines"]
    assert names(s, e) == ["ANNA", "DECLAN"]
    assert e["overlap"] is not None
    assert "When we were just outside." in e["text"] and "At breakfast." in e["text"]


def test_case01_group_speaker():
    s = parsed("case03_")                # 해운대 '일  동 ⇥ 으아아아아!!!'
    e = el(s, 22)
    assert names(s, e) == ["일동"] and e["speaker_raw"] == "일  동"


# ---------------------------------------------------------------- #2 화자 꼬리표 조합

def test_case02_speaker_tags():
    s = parsed("case02")
    e = el(s, 32)                       # '표종성 (독일어)'
    assert names(s, e) == ["표종성"] and e["speaker_raw"] == "표종성 (독일어)"
    assert e["lang"] == "de" and e["how"] == "on"
    e = el(s, 37)                       # '아심 (영어 소리)'
    assert names(s, e) == ["아심"] and e["lang"] == "en" and e["how"] == "off"
    e = el(s, 43)                       # '표종성 (영어)'
    assert e["lang"] == "en" and e["how"] == "on"
    e = el(s, 52)                       # '천세황 (소리)'
    assert names(s, e) == ["천세황"] and e["how"] == "off" and e["lang"] == "ko"
    # 같은 사람은 꼬리표가 달라도 한 인물
    assert el(s, 32)["speakers"] == el(s, 43)["speakers"]


def test_case02_tag_parsing_units():
    assert T.split_speaker("천세황(무전소리)") == ("천세황", ["무전소리"])
    assert T.how_of(["무전소리"]) == "radio"
    assert T.how_of(["전화"]) == "phone"
    assert T.how_of(["E"]) == "off" and T.how_of(["O.S."]) == "off"
    assert T.how_of(["NA"]) == "narration" and T.how_of(["Na"]) == "narration"
    assert T.how_of(["V.O."]) == "vo"            # 문서 안에서 off/narration 으로 정함
    assert T.lang_of(["영어소리"]) == "en" and T.lang_of(["日"]) == "ja" and T.lang_of(["en anglais"]) == "en"
    assert T.split_speaker("MAN (V.O.)(CONT'D)") == ("MAN", ["V.O.", "CONT'D"])
    assert T.split_speaker("<우주>")[0] == "우주"


# ---------------------------------------------------------------- #3 대본 언어 ≠ 대사 언어, 병기 번역

def test_case03_foreign_line_with_gloss():
    s = parsed("case03_")
    assert s["lang"] == "ko"
    e = el(s, 40)                       # '김 휘 ⇥ (가픈 숨을 몰아쉬며) Sir! Terrible accident! (심각한 문제가 발생했어요! )'
    assert e["lang"] == "en"
    assert e["text"] == "Sir! Terrible accident!"
    assert e["gloss"] == "심각한 문제가 발생했어요!"
    assert e["paren"] == [{"text": "가픈 숨을 몰아쉬며", "at": 0}]
    assert el(s, 24)["lang"] == "ko"


def test_case03_bilingual_slash():
    s = parsed("case03b")
    assert s["lang"] == "en"
    e = el(s, 627)                      # '잘 먹겠습니다. / Thank you for this' + 'meal.'
    assert e["lang"] == "ko" and e["text"] == "잘 먹겠습니다."
    assert e["gloss"] == "Thank you for this meal."
    assert el(s, 633)["gloss"] == "Yeah, kinda."


# ---------------------------------------------------------------- #4 화면 글자, 노래, 내레이션

def test_case04_onscreen_song_narration():
    s = parsed("case04")
    e = el(s, 655)                      # '<자막 : 한국해양연구소>'
    assert e["type"] == "onscreen" and e["kind"] == "subtitle" and e["text"] == "한국해양연구소"
    e = el(s, 183)                      # '(공포에 질려 노래) 노는 게 제일 좋아♩♪♫♬~~'
    assert e["type"] == "dialogue" and e["sung"] is True and 184 in e["lines"]
    assert el(s, 186)["sung"] is True
    assert el(s, 185)["sung"] is False
    e = el(s, 772)                      # '김  휘(Na) ⇥ …'
    assert e["how"] == "narration" and names(s, e) == ["김휘"]
    assert el(s, 778)["how"] == "on" and el(s, 778)["speakers"] == e["speakers"]


# ---------------------------------------------------------------- #5 대사 안 괄호 지시, 소리 괄호

def test_case05_parens_with_positions():
    s = parsed("case05")
    e = el(s, 11)                       # FLEABAG + (Earnest, touch of pain) … (buzzer goes) …
    assert e["type"] == "dialogue" and 25 in e["lines"] and 28 in e["lines"]
    texts = [p["text"] for p in e["paren"]]
    assert texts == ["Earnest, touch of pain", "buzzer goes"]
    assert e["paren"][0]["at"] == 0
    at = e["paren"][1]["at"]
    assert e["text"][:at].rstrip().endswith("until the buzzer goes -")
    assert e["text"][at:].startswith("And then you open the door")
    assert "(" not in e["text"]
    assert "(buzzer goes)" in e["raw"]


def test_case05_assemble_units():
    text, parens = T.assemble(T.split_parens("아.. 그게 그러니까.. (봇물 터지듯) ‘쟤가 잘 보이고"))
    assert text == "아.. 그게 그러니까.. ‘쟤가 잘 보이고"
    assert parens == [{"text": "봇물 터지듯", "at": len("아.. 그게 그러니까.. ")}]
    text, parens = T.assemble(T.split_parens("(버럭)안돼요!"))
    assert text == "안돼요!" and parens == [{"text": "버럭", "at": 0}]


# ---------------------------------------------------------------- #6 씬 번호 변형

def test_case06_scene_number_variants():
    s = parsed("case06_")
    assert [sc["number"] for sc in s["scenes"]] == ["34-1", "34-2", "34-3", "34-4"]
    sc = s["scenes"][0]
    assert sc["heading"] == "S# 34-1 학교 (운동장가)" and sc["place"] == "학교 (운동장가)"
    s = parsed("case06b")
    nums = [sc["number"] for sc in s["scenes"]]
    assert "17A" in nums and "17B" in nums
    sc = next(x for x in s["scenes"] if x["number"] == "17B")
    assert sc["place"] == "CAB - MAX"


def test_case06_number_less_heading():
    s = parsed("case10")                # '프롤로그 - 2차선 도로 (실외. 낮)'
    sc = s["scenes"][0]
    assert sc["number"] is None and "프롤로그" in sc["heading"] and sc["time"] and "낮" in sc["time"]
    assert s["scenes"][1]["number"] == "1"


def test_heading_units():
    for t, num in [("S#1. 빵집. 낮", "1"), ("#100-1  백화점 옥상.", "100-1"), ("C#3   -보트를 고치던", "C#3"),
                   ("17A\tEXT. FEDERAL BUILDING - VINCENT\t17A".replace("\t", " "), "17A"),
                   ("8.A.-이상한 놈-기차내부/삼둥칸~화장실.INT.D", "8A"), ("씬21연결.", "21"), ("P3. 인도양 원양어선(N)", "P3"),
                   ("76-3. 산길", "76-3")]:
        h = T.match_heading(t)
        assert h is not None and h[1] == num, (t, h)


# ---------------------------------------------------------------- #7 쪽 넘김으로 쪼개진 대사

def test_case07_split_dialogue_merged():
    s = parsed("case07")
    e = el(s, 43)                       # FLEABAG … (MORE) / 머리말 / FLEABAG (CONT'D) …
    assert e["type"] == "dialogue" and {44, 46, 50, 52} <= set(e["lines"])
    assert why(s, 47) == "cont_marker"          # (MORE)
    assert why(s, 49) == "cont_marker"          # 다음 쪽 'FLEABAG (CONT'D)'
    assert why(s, 48) == "page_header"          # 'Episode 1 SHOOTING SCRIPT 02.03.15 ⇥ 2.'
    assert "edging towards your arsehole. But you’re drunk" in e["text"]
    assert 95 in el(s, 57)["lines"] and why(s, 94) == "cont_marker"


# ---------------------------------------------------------------- #8 OCR·글자층 품질

def test_case08_ocr_quality_and_decorations():
    s = parsed("case08")
    for n in (0, 39, 71, 108):          # 'Page n /81 ⇥ 감시…/Ver 5. ⇥ 2012년10월24일'(OCR 오탈자로 조금씩 다름)
        assert why(s, n) == "page_header", n
    for n in (36, 37, 68, 69, 105, 106):  # '영화사 집 ZIP CINEMA …'
        assert why(s, n) == "page_footer", n
    for n in (38, 70, 107):             # '1/81'
        assert why(s, n) == "page_number", n
    q = s["quality"]
    assert any("OCR 평균 conf" in x for x in q["notes"])
    assert q["score"] < 1.0                     # OCR 신뢰도가 점수에 반영
    assert [sc["number"] for sc in s["scenes"]] == ["1", "7", "11"]
    assert s["front"]["version_raw"] == "Ver 5." and s["front"]["date"] == "2012년10월24일"


def test_garbage_lines_become_unknown():
    rows = [{"n": 0, "file": 0, "page": 1, "x": 0.1, "y": 0.1, "text": "S#1. 거리 / 낮", "bold": False, "cell": None,
             "src": "ocr", "conf": 0.95},
            {"n": 1, "file": 0, "page": 1, "x": 0.1, "y": 0.12, "text": "ÇÑ±¹¾î ÀÔ´Ï´Ù", "bold": False, "cell": None,
             "src": "ocr", "conf": 0.9},
            {"n": 2, "file": 0, "page": 1, "x": 0.1, "y": 0.14, "text": "흐릿한 글자", "bold": False, "cell": None,
             "src": "ocr", "conf": 0.31}]
    s = P.parse_rows("fm1", rows)
    assert el(s, 1)["type"] == "unknown" and el(s, 2)["type"] == "unknown"
    assert s["quality"]["unknown_ratio"] > 0.5 and s["quality"]["typed_ratio"] < P.TH_TYPED
    assert {p["problem"] for p in P.problems_for(s)} >= {"extract_broken", "parse_low_dialogue"}


# ---------------------------------------------------------------- #9 인물 이름 변형

def test_case09_name_variants():
    s = parsed("case09")
    e = el(s, 1885)                     # '법민, 흠순, 품일'
    assert sorted(names(s, e)) == ["법민", "품일", "흠순"]
    beop = el(s, 796)                   # '김법민' = '법민'(성 생략)
    assert beop["speakers"][0] in e["speakers"] and beop["speaker_raw"] == "김법민"
    p = person(s, beop["speakers"][0])
    assert {p["name"], *p["aliases"]} == {"법민", "김법민"}
    # 번호 엑스트라는 묶지 않는다
    a, b = el(s, 850), el(s, 847)
    assert a["speaker_raw"] == "백제병사1(거시기)" and names(s, a) == ["백제병사1"]
    assert names(s, b) == ["백제병사2"] and a["speakers"] != b["speakers"]


# ---------------------------------------------------------------- #10 앞머리(인물표가 대사·씬 머리처럼 보임)

def test_case10_front_castlist():
    s = parsed("case10")
    for n in range(0, 21):              # '등장인물', '수현: 29세의 남자.' … '그 외의 인물들....'
        assert el(s, n)["type"] == "front", n
    assert all(e["scene"] is None for e in s["elements"] if e["type"] == "front")
    # 인물표의 이름은 화자가 아니다(대사가 있는 미영만)
    assert [p["name"] for p in s["people"]] == ["미영"]
    assert el(s, 36)["how"] == "off"            # '미영 ⇥ (소리) 헤어져.'


# ---------------------------------------------------------------- #11 같은 영화 여러 판본

def test_case11_version_in_front():
    s = parsed("case11")
    assert s["front"]["version_raw"] == "VER.5.1015"
    assert s["front"]["title"] == "터 널" and s["front"]["writer"] == "김성훈"
    e = el(s, 11)                       # '(소리) ⇥ 정수씨, …' 이름 없이 들리는 방식만
    assert e["type"] == "dialogue" and e["speakers"] == [] and e["how"] == "off" and e["speaker_raw"] == "(소리)"


def test_case11_catalog_has_version_fields():
    from sinkprep import catalog, paths
    if not paths.CATALOG.exists():
        pytest.skip("catalog.csv 없음")
    row = next((d for d in catalog.load() if d["doc_id"] == "fm5691168"), None)
    assert row is not None
    for k in ("version_label", "version_raw", "dup_group", "is_primary"):
        assert k in row


# ---------------------------------------------------------------- 실패 대비·품질 기준

def test_exception_falls_back_to_unknown(monkeypatch):
    head, rows = load("case01_two")

    def boom(self):
        raise RuntimeError("일부러")

    monkeypatch.setattr(P.DocParser, "parse", boom)
    s = P.parse_rows(head["doc_id"], rows)
    assert all(e["type"] == "unknown" for e in s["elements"]) and len(s["elements"]) == len(rows)
    assert s["quality"]["unknown_ratio"] == 1.0 and "파서 예외" in s["quality"]["notes"][0]
    assert P.problems_for(s)[0]["problem"] == "other"


def test_empty_document():
    s = P.parse_rows("fm2", [])
    assert s["elements"] == [] and s["dropped"] == [] and s["quality"]["lines"] == 0
    assert P.problems_for(s) == []


def test_problem_thresholds():
    s = parsed("case09")
    assert P.problems_for(s) == []              # 대사 많고 화자 다 있음
    s2 = json.loads(json.dumps(s))
    s2["quality"]["speakerless_dialogue"] = 5
    assert "speaker_missing" in {p["problem"] for p in P.problems_for(s2)}


# ---------------------------------------------------------------- 엑셀 표 대본(녹음 대본)

def _xls_rows(cells: list[tuple[int, str]]) -> list[dict]:
    """(첫 칸 열, 칸들을 탭으로 이은 글) -> lines.lines_from_xls 와 같은 모양의 줄."""
    out = []
    for r, (col, text) in enumerate(cells):
        out.append({"n": r, "file": 0, "page": None, "x": round(col / 3, 4), "y": round(r / len(cells), 4),
                    "text": text, "bold": False, "cell": {"table": 0, "row": r, "col": col},
                    "src": "xls", "conf": None})
    return out


def test_xls_rows_scene_and_shot_notes():
    """교회누나(fm54537) 모양: '번호⇥장소⇥화면' 행은 씬 머리, '이름 cu' 촬영 크기 줄은 대사가 아니라 지문."""
    cells = [(0, "s#\t장소\t내용 / 대사"), (0, "1\t웨딩거리\t영일 cu"),
             (2, "나레이션: 참 오래간만에 그녀를 기다리고 있습니다."), (2, "영일 bs"),
             (2, "나레이션: 내가 사랑하는 여자에요"), (2, "영일 cu"),
             (0, "2\t교회가는 거리\t영일, 주희 fs"), (2, "주희: 살쪘나?"), (2, "영일: 그래?몸무게는 그대론데"),
             (2, "주희 : 아니야.훨씬 건강해보여"), (2, "주희 cu"), (2, "영일 : 감기 걸렸어?"),
             (0, "3\t교회 외경\t교회외경 fs"), (0, "4\t교회 안\t영일 주희 fs"), (2, "영일 : 교회 많이 좋아졌다.")]
    s = P.parse_rows("fmxls", _xls_rows(cells))
    assert [e["type"] for e in s["elements"] if e["type"] == "scene"] == ["scene"] * 4
    sc = {x["id"]: x for x in s["scenes"]}
    assert el(s, 1)["type"] == "scene" and sc[el(s, 1)["scene"]]["number"] == "1"
    assert sc[el(s, 1)["scene"]]["place"] == "웨딩거리"
    for n in (3, 5, 10):
        assert el(s, n)["type"] == "action", n
    assert el(s, 4)["type"] == "dialogue" and 5 not in el(s, 4)["lines"]
    assert names(s, el(s, 7)) == ["주희"] and names(s, el(s, 8)) == ["영일"]
    covered = sorted(n for e in s["elements"] for n in e["lines"]) + sorted(n for d in s["dropped"] for n in d["lines"])
    assert sorted(covered) == list(range(len(cells)))           # B2: 모든 줄을 정확히 한 번


def test_shot_guard_keeps_english_dialogue():
    """'Ms. Smith…' 같은 영어 대사는 촬영 크기 표기로 보지 않는다."""
    assert not P._RX_SHOT.match("Ms. Smith is waiting for you.")
    assert not P._RX_SHOT.match("Ms Jones, please sit.")
    assert P._RX_SHOT.match("cu , 시계보는 영일") and P._RX_SHOT.match("FS")


def _plain_rows(texts: list[tuple[float, str]]) -> list[dict]:
    return [{"n": i, "file": 0, "page": 1 + i // 40, "x": x, "y": (i % 40) / 40, "text": t, "bold": False,
             "cell": None, "src": "hwp", "conf": None} for i, (x, t) in enumerate(texts)]


@pytest.mark.parametrize("sep", ["; ", "/ ", " : "])
def test_semicolon_slash_star_speakers(sep):
    """'감찰반장; 대사'(킬리만자로), '왕/ 대사'·'조내시(o.s)/ 대사'(음란서생), '*김상두 : 대사'(두사부일체)."""
    star = "*" if sep == " : " else ""
    body = []
    for k in range(6):
        body += [(0.143, f"{k + 1}. 궁궐 안 - 낮"), (0.143, "조내시가 허리를 숙이고 들어선다."),
                 (0.143, f"{star}조내시{sep}전하, 김윤서 입시이옵니다."), (0.143, f"{star}왕{sep}아, 그래? 무슨 일이지?"),
                 (0.143, f"{star}조내시(o.s){sep}들라 하라 하셨사옵니다.")]
    s = P.parse_rows("fmsep", _plain_rows(body))
    d = [e for e in s["elements"] if e["type"] == "dialogue"]
    assert len(d) == 18
    assert names(s, el(s, 2)) == ["조내시"] and el(s, 2)["text"].startswith("전하")
    assert names(s, el(s, 3)) == ["왕"]
    assert el(s, 0)["type"] == "scene" and el(s, 1)["type"] == "action"


# ---------------------------------------------------------------- 0.2.0 규칙(합성 줄)

def _pages(pages: list[list[tuple]], src: str = "hwp", pitch: float = 0.014) -> list[dict]:
    """쪽마다 (x, 글[, 앞 빈 줄 수]) 목록 -> 줄. y 는 한 줄 간격 pitch, 빈 줄 수만큼 더 띄운다."""
    out = []
    for p, items in enumerate(pages, 1):
        y = 0.2                                      # 위 띠(머리말 후보 y<=0.16) 밖에서 시작
        for it in items:
            x, t = it[0], it[1]
            y += pitch * (it[2] if len(it) > 2 else 0)
            out.append({"n": len(out), "file": 0, "page": p, "x": x, "y": round(y, 4), "text": t, "bold": False,
                        "cell": None, "src": src, "conf": None})
            y += pitch
    return out


def _covered_once(s: dict, rows: list[dict]) -> bool:
    got = sorted(n for e in s["elements"] for n in e["lines"]) + [n for d in s["dropped"] for n in d["lines"]]
    return sorted(got) == [r["n"] for r in rows]


def test_page_shift_needs_better_fit():
    """대사만 있는 쪽(가장 왼쪽이 대사 칸)은 여백이 밀린 쪽으로 보지 않는다(Natural Born Killers·Annie Hall).
    진짜로 쪽 전체가 밀린 쪽(Alien)은 그대로 되돌린다."""
    def normal(k):
        w = ["red", "blue", "green", "black", "white", "gray", "pink", "gold", "brown", "tan"][k]
        return [(0.119, f"INT. {w.upper()} KITCHEN - DAY"), (0.119, f"Mary stands at the {w} stove, stirring."),
                (0.254, "MARY"), (0.186, f"Where were you, {w} boy?"), (0.119, f"Bob shrugs and sits on a {w} chair."),
                (0.254, "BOB"), (0.186, f"Out. With {w} friends."), (0.186, f"Nothing {w}, really."),
                (0.119, f"She turns away from the {w} wall.")]
    talk = [(0.254, "MARY"), (0.186, "You always say that."), (0.186, "Every single time."), (0.254, "BOB"),
            (0.186, "Because it is true."), (0.254, "MARY"), (0.186, "Then prove it."), (0.186, "Prove it now.")]
    def shifted(k):
        return [(x + 0.03, t) for x, t in normal(k)]
    # 홀짝 쪽 여백이 번갈아 다른 문서(Alien): 밀린 쪽 단계도 문서 단계가 되어 맞춤 수가 같다
    rows = _pages([normal(0), shifted(1), normal(2), shifted(3), normal(4), talk, shifted(9)])
    s = P.parse_rows("fmshift", rows)
    base = 9 * 5
    assert names(s, el(s, base + 0)) == ["MARY"] and el(s, base + 1)["type"] == "dialogue"
    assert names(s, el(s, base + 4)) == ["BOB"] and el(s, base + 7)["type"] == "dialogue"
    for sh in (9, 27, base + 8):
        assert el(s, sh)["type"] == "scene" and el(s, sh + 1)["type"] == "action"
        assert names(s, el(s, sh + 3)) == ["MARY"] and el(s, sh + 3)["lines"] == [sh + 2, sh + 3]
        assert el(s, sh + 8)["type"] == "action"
    assert any("쪽 밀림 맞춤 3쪽" in n for n in s["quality"]["notes"])
    assert _covered_once(s, rows)


def test_bullet_names_and_bare_sharp_heading():
    """'*장달수⇥대사'(색즉시공): 되풀이되는 글머리표는 떼고 이름으로. 'S#⇥병실 (밤)'(오버 더 레인보우)은 번호 없는 씬 머리."""
    pages = []
    for k in range(5):
        pages.append([(0.143, f"S#\t{['병실', '복도', '호프집', '도서관', '기숙사'][k]} (밤)"),
                      (0.143, "툴툴거리며 복도를 지나는 달수, 성국, 경호."),
                      (0.218, "*장달수\t아까 그 사람 누구에요?"), (0.218, "*최성국\t체교과 이윤경이잖아~ 우리학교 킹카…"),
                      (0.218, "*이윤경\t(말을 끊으며, 혀가 꼬인 상태) 너? 야! 대학에서는"),
                      (0.337, "무조건 학번이야! 학번!"), (0.218, "*함상욱(E)\t나야…"), (0.143, "또다시 달수를 바라보는 윤경.")])
    rows = _pages(pages)
    s = P.parse_rows("fmbullet", rows)
    assert el(s, 0)["type"] == "scene"
    sc = s["scenes"][0]
    assert sc["number"] is None and sc["place"] == "병실" and sc["time"] == "밤"
    assert len(s["scenes"]) == 5
    assert names(s, el(s, 2)) == ["장달수"] and el(s, 2)["text"].startswith("아까")
    assert names(s, el(s, 4)) == ["이윤경"] and el(s, 4)["lines"] == [4, 5]
    assert names(s, el(s, 6)) == ["함상욱"] and el(s, 6)["how"] == "off"
    assert not T.name_like("S#") and not T.name_like("C#1") and T.name_like("COP #1")
    assert _covered_once(s, rows)


def test_bullet_needs_repetition():
    """글머리표 이름이 몇 번뿐이면(인물표·메모) 떼지 않는다."""
    body = [(0.143, f"{k + 1}. 도서관 - 낮") for k in range(4)]
    body += [(0.143, "*참고\t이 장면은 밤에 찍는다."), (0.143, "*메모\t소품 확인.")]
    s = P.parse_rows("fmbullet2", _pages([body]))
    assert all(e["type"] != "dialogue" for e in s["elements"])


def test_letterspaced_inline_names():
    """'진 수 (웃으며) 대사', '연 희 아냐.', 'P⇥D⇥대사'(오버 더 레인보우): 붙인 이름이 지문에도 낱말로 나오면 이름.
    매달린 들여쓰기 줄은 대사를 잇고, 지시어로 시작하는 줄('그 날 …')은 이름이 아니다."""
    places = ["병실", "복도", "옥상", "식당"]
    pages = []
    for k, w in enumerate(places):
        pages.append([(0.143, f"S#\t{w} (밤)"), (0.143, f"{w}에 앉아 있는 진수의 모습. 연희가 들어온다."),
                      (0.143, f"진 수 (돌아다보며) 어. {w}에 왔어?"), (0.143, f"연 희 아냐.\t{w}은 지금 딱 좋은데."),
                      (0.143, f"P\tD\t(돈을 내밀며) 자-! {w}값이다."), (0.143, f"진 수 (맥주를 건네며) {w}에서는"),
                      (0.277, f"이런 거 안 주던? {w} 말이야."), (0.143, f"그 날 {w}에는 비가 왔다. 연희, 창밖을 본다."),
                      (0.143, f"쾅 쾅 {w} 문을 두드리는 소리.")])
    rows = _pages(pages)
    s = P.parse_rows("fmlsp", rows)
    assert names(s, el(s, 2)) == ["진수"] and el(s, 2)["text"].startswith("어.") and el(s, 2)["paren"][0]["at"] == 0
    assert names(s, el(s, 3)) == ["연희"] and el(s, 3)["text"].startswith("아냐.")
    assert names(s, el(s, 4)) == ["PD"]
    assert names(s, el(s, 5)) == ["진수"] and el(s, 5)["lines"] == [5, 6]
    assert el(s, 7)["type"] == "action" and el(s, 8)["type"] == "action"
    assert _covered_once(s, rows)



def test_centered_numbered_heading_series():
    """가운데 정렬된 번호 씬 머리 '4.⇥공항 활주로 (낮)'(공동경비구역 JSA)는 들여썼어도 문서 전체 번호 차례로 받는다.
    지문 속 씬 참조('S#84 회상에서 … 장면들이다.')는 씬 머리가 아니다."""
    places = ["공항 활주로", "입국장", "차 안", "판문점", "초소", "다리 위", "회견장", "입원실", "신문실", "의무실",
              "모의전투 훈련장", "북측 초소"]
    pages = [[] for _ in range(4)]
    for k, w in enumerate(places):
        pg = pages[k // 3]
        pg += [(0.45 - 0.01 * len(w), f"{k + 1}.\t{w} (낮)", 2), (0.143, f"{w}에 서 있는 소피. 바람이 분다."),
               (0.47, "소피", 1), (0.33, f"여기가 {w}인가요?"), (0.47, "수혁", 1), (0.36, f"네. {w}입니다.")]
    pages[3].append((0.143, "S#84 회상에서 틀린 부분을 바로잡는 장면들이다.", 2))
    rows = _pages(pages)
    s = P.parse_rows("fmseries", rows)
    assert [sc["number"] for sc in s["scenes"]] == [str(k + 1) for k in range(12)]
    assert s["scenes"][0]["place"] == "공항 활주로" and s["scenes"][0]["time"] == "낮"
    assert el(s, len(rows) - 1)["type"] != "scene"
    assert names(s, el(s, 3)) == ["소피"]
    assert _covered_once(s, rows)


def test_speaker_carry_over_after_action(tmp_path: Path):
    """이름 줄 문서에서 지문 뒤 이름 줄 없이 대사 칸에 이어진 말은 같은 씬 앞 대사의 화자(speaker_inferred=true).
    씬이 바뀌었거나 사이 지문이 3개 이상이면 이어받지 않는다. 옛 파일(칸 없음)도 스키마를 통과한다."""
    pages = []
    words = ["교실", "복도", "운동장", "창고"]
    for k, w in enumerate(words):
        pages.append([(0.143, f"{k + 1}. {w} - 낮", 2), (0.143, f"{w}에 아이들이 모여 있다.", 1),
                      (0.47, "미주", 1), (0.28, f"니 {w}에 혼자 왔나?"),
                      (0.143, f"인호가 {w} 쪽을 보며 망설이는데", 1), (0.28, f"{w}에서 기다리께.", 1),
                      (0.47, "인호", 1), (0.28, "알았다."),
                      (0.143, f"{w}의 종이 울린다.", 1), (0.143, "아이들이 흩어진다.", 1), (0.143, "바람이 분다.", 1),
                      (0.28, f"{w} 문 닫는다!", 1)])
    rows = _pages(pages)
    s = P.parse_rows("fm9000001", rows)
    e = el(s, 5)
    assert e["type"] == "dialogue" and names(s, e) == ["미주"] and e["speaker_inferred"] is True
    assert e["speaker_raw"] is None
    assert "speaker_inferred" not in el(s, 3)                       # 이름 줄이 있는 대사에는 칸이 없다
    last = el(s, 11)
    assert last["type"] == "dialogue" and last["speakers"] == [] and "speaker_inferred" not in last
    assert s["quality"]["speakerless_dialogue"] == 4
    out = tmp_path / "script.json"
    out.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    assert V.validate_script(out) == []
    for x in s["elements"]:
        x.pop("speaker_inferred", None)
    out.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    assert V.validate_script(out) == []
    assert _covered_once(s, rows)


def test_centered_dialogue_by_paragraph_gap():
    """가운데 정렬 대사(파이란·친절한 금자씨): 대사 줄 x 가 길이에 따라 0.16~0.49 로 퍼져 한 묶음으로 이어진다.
    지문 여백은 똑같은 x 가 가장 많은 0.172, 이름 줄 다음 줄은 x 와 상관없이 문단 간격 전까지 대사."""
    names_ = ["희재", "인하", "정란"]
    words = ["교실", "복도", "거리", "카페", "공원", "옥상", "강가", "부엌", "창가", "다리"]
    pages, pg = [], []
    for k in range(40):
        w = ["작은", "넓은", "낡은", "밝은"][k // 10] + " " + words[k % 10]
        nm = names_[k % 3]
        pg += [(0.172, f"{w} 쪽으로 걸음을 옮기는 인하.", 2.2), (0.172, f"{w}에 바람이 분다.", 0),
               (0.483, nm, 2.2), (0.160 + 0.003 * (k % 5), f"(전혀 물러나지 않을 자세로) {w}에서 그러는 거, 굉장히", 0),
               (0.175 + 0.0078 * k, f"비겁한 짓이야. {w}", 0)]
        if len(pg) >= 15 or k == 39:
            pages.append(pg)
            pg = []
    rows = _pages(pages, pitch=0.019)
    s = P.parse_rows("fm9000002", rows)
    assert any(n.startswith("대사 표기: 이름 줄") for n in s["quality"]["notes"])
    for b in range(0, len(rows), 5):
        assert el(s, b)["type"] == "action" and el(s, b)["lines"] == [b, b + 1], b
        d = el(s, b + 2)
        assert d["type"] == "dialogue" and d["lines"] == [b + 2, b + 3, b + 4], b
        assert names(s, d) == [names_[(b // 5) % 3]] and d["paren"][0]["at"] == 0
    assert _covered_once(s, rows)


def test_collapsed_dialogue_column():
    """The Graduate·Apocalypse Now: 대문자 이름 줄은 들여썼지만 대사는 지문 여백(0.186)에 놓인다. 빈 줄 뒤 이름 줄 +
    한 줄 간격의 여백 줄이 되풀이되면 이름 줄로 받고, 대사는 다음 문단 간격 전까지. '-----page 3' 줄은 쪽 장식."""
    cues = ["BEN", "MRS. ROBINSON", "ELAINE"]
    words = ["pool", "car", "hotel", "bar", "house", "church", "bus", "room", "garden", "party", "club", "beach"]
    pages, pg = [], []
    for k in range(24):
        w = words[k % 12] + (" again" if k >= 12 else "")
        pg += [(0.186, f"He walks to the {w} and stops.", 1), (0.186, "Nobody looks at him.", 0),
               (0.254, cues[k % 3], 1), (0.186, f"Look, could you explain the {w} to", 0),
               (0.186, "me, please?", 0)]
        if len(pg) >= 20:
            pages.append(pg)
            pg = []
    pages[-1] += [(0.186, "------------------------------------------\tpage 7", 0)]
    rows = _pages(pages)
    s = P.parse_rows("fm9000003", rows)
    assert any("대사가 지문 여백에 놓인" in n for n in s["quality"]["notes"])
    for b in range(0, 120, 5):
        assert el(s, b)["type"] == "action" and el(s, b)["lines"] == [b, b + 1], b
        d = el(s, b + 2)
        assert d["type"] == "dialogue" and d["lines"] == [b + 2, b + 3, b + 4], b
        assert names(s, d) == [cues[(b // 5) % 3]]
    assert why(s, 120) == "page_number"
    assert _covered_once(s, rows)


def test_bracket_names():
    """'[김반장] 대사'(날 보러 와요): 3번 이상 되풀이되는 대괄호 이름은 화자. '[페이지] 001'·'[제목]' 표지는 아니다."""
    body = [(0.143, "#1 형사계 사무실"), (0.143, "실내등이 켜지며 김형사 등장한다.")]
    words = ["바다", "여관", "논길", "철길", "다방", "지서", "공장", "학교"]
    for w in words:
        body += [(0.143, f"[김반장] (서류를 넘기며) {w}에서 뭐 나왔어?"), (0.143, f"[박형사] {w}엔 아무것도 없습니다."),
                 (0.143, f"[조형사] 제가 {w}에 다시 가 보겠습니다.")]
    body += [(0.143, "[페이지] 001"), (0.143, "[제목] 날 보러 와요"), (0.143, "[범인] 나야, 나.")]
    rows = _pages([body])
    s = P.parse_rows("fm9000004", rows)
    assert names(s, el(s, 2)) == ["김반장"] and el(s, 2)["paren"][0]["text"] == "서류를 넘기며"
    assert names(s, el(s, 3)) == ["박형사"] and el(s, 3)["text"].endswith("없습니다.")
    n = len(body)
    assert el(s, n - 3)["type"] != "dialogue" or not el(s, n - 3)["speakers"]
    assert all(p["name"] not in ("페이지", "제목") for p in s["people"])
    assert el(s, n - 1)["type"] == "action" or names(s, el(s, n - 1)) == ["범인"]
    assert el(s, n - 3)["type"] == "action"                          # 표지 줄은 앞 대사에 붙지 않는다
    assert _covered_once(s, rows)
    # 몇 줄뿐인 대괄호 표지('[자료화면] …', '[日] …')는 대사 표기가 아니다
    few = [(0.143, "#1 거리")] + [(0.143, f"[자료화면] {w}에 줄 선 사람들.") for w in words[:4]] \
        + [(0.143, f"[日] {w}에 가요?") for w in words[:3]]
    s2 = P.parse_rows("fm9000007", _pages([few]))
    assert all(p["name"] not in ("자료화면", "日") for p in s2["people"])


def test_srt_arrow_timecodes_dropped():
    """자막 파일을 옮긴 문서(fm56910): '00:00:20,000 --> 00:00:25,000' 은 timecode, 그 앞 맨 번호는 subtitle_cue."""
    body = []
    for k in range(25):
        body += [(0.143, str(k + 1)), (0.143, f"00:{k:02d}:20,000 --> 00:{k:02d}:25,{k:03d}"),
                 (0.143, f"Line number {['one', 'two', 'three', 'four', 'five'][k % 5]} of the {k}th cue.")]
    rows = _pages([body[i:i + 30] for i in range(0, len(body), 30)])
    s = P.parse_rows("fm9000005", rows)
    assert why(s, 0) == "subtitle_cue" and why(s, 1) == "timecode" and why(s, 4) == "timecode"
    assert el(s, 2)["type"] in ("action", "dialogue")
    assert s["quality"]["typed_ratio"] == 1.0
    assert T.match_transition("00:00:20,000 --> 00:00:25,000") is False
    assert _covered_once(s, rows)


def test_voice_word_speaker_is_speakerless_off():
    """꽃잎(fm54568): '소리 : …' 가 3번 이상이면 화자 없는 대사(how=off), '우리 : …' 는 무리 화자.
    '소리 : 천둥 소리' 같은 소리 묘사는 대사가 아니다."""
    body = [(0.143, f"{k + 1}. 거리 - 낮") for k in range(3)]
    body += [(0.143, "민우 : 어디 가?"), (0.143, "영희 : 집에."), (0.143, "소리 : 그날은 무슨 날이었을까."),
             (0.143, "민우 : 잠깐만."), (0.143, "영희 : 왜?"), (0.143, "소리 : 엄마랑 나는 왜 거기에 있었을까."),
             (0.143, "민우 : 아니야."), (0.143, "소리 : 잔칫집에 가는 것처럼 소풍이라도 가는 것처럼."),
             (0.143, "우리 : 그녀는 장항으로 가지 않았다"), (0.143, "영희 : 그래?"), (0.143, "우리 : 우리는 그녀를 찾았다"),
             (0.143, "민우 : 가자."), (0.143, "우리2 : 그녀는 어디 있을까"), (0.143, "소리 : 천둥 소리")]
    s = P.parse_rows("fm9000006", _pages([body]))
    e = el(s, 5)
    assert e["type"] == "dialogue" and e["speakers"] == [] and e["how"] == "off"
    assert el(s, 8)["speakers"] == [] and el(s, 10)["how"] == "off"
    assert names(s, el(s, 11)) == ["우리"] and names(s, el(s, 15)) == ["우리2"]
    assert not (el(s, 16)["type"] == "dialogue" and el(s, 16)["lines"] == [16])


def test_quality_flags_empty_dialogue():
    """이름 줄만 대사로 잡히고 말은 지문으로 간 문서(내 마음의 풍금 꼴)는 말 없는 대사 비율로 needs_review 에 오른다."""
    els = [{"type": "dialogue", "lines": [i], "text": "", "speakers": ["P01"]} for i in range(24)]
    els += [{"type": "dialogue", "lines": [100 + i], "text": "대사", "speakers": ["P01"]} for i in range(6)]
    script = {"doc_id": "fmq", "elements": els,
              "quality": {"lines": 300, "typed_ratio": 1.0, "dialogue_ratio": 0.5, "unknown_ratio": 0.0,
                          "speakerless_dialogue": 0, "empty_dialogue": 24, "inferred_speaker": 9, "score": 0.2, "notes": []}}
    probs = P.problems_for(script)
    assert [p["problem"] for p in probs] == ["parse_low_dialogue", "speaker_missing"]
    assert "말 없는 대사" in probs[0]["detail"] and probs[0]["score"] == 0.2
    # 대사가 20개 미만인 조각·짧은 문서는 비율로 판정하지 않음(한두 개로 10% 를 넘기 쉬움)
    small = {**script, "elements": els[:2] + els[-6:], "quality": {**script["quality"], "empty_dialogue": 2, "inferred_speaker": 1}}
    assert P.problems_for(small) == []


# ---------------------------------------------------------------- 0.3.0 규칙(합성 줄, 추격자 fm55664 꼴)

_PLACES = ["망원역", "은행 앞", "골목길", "초소 앞", "먹자골목", "보도방", "모텔", "마트", "지구대", "기수대", "병원", "교회"]
_AX, _TX, _CX = 0.1667, 0.2327, 0.3          # 지문 여백, 이름 칸, 이어짐 칸


def _chaser_rows() -> tuple[list[dict], dict[str, list[int]]]:
    """한글 탭 대본: 지문은 여백, '이름⇥대사' 는 이름 칸, 이어지는 줄은 이어짐 칸. 줄 번호를 종류별로 함께 돌려준다."""
    pages, at = [], {}
    n = 0

    def add(pg: list, kind: str, x: float, text: str, gap: int = 0) -> None:
        nonlocal n
        pg.append((x, text, gap))
        at.setdefault(kind, []).append(n)
        n += 1

    for k, w in enumerate(_PLACES):
        pg: list = []
        if k % 3 == 1:
            add(pg, "colon_head", _AX, f"{k + 1}.\t영민의 거처 : {w} / 밤", 2)
        elif k == 8:
            add(pg, "colon_head", _AX, f"{k + 1}.\t‘{w}을 지나는 중호’ 시퀀스 : 망원동 {w} / 밤", 2)
        else:
            add(pg, "head", _AX, f"{k + 1}.\t{w} / 밤", 2)
        add(pg, "action", _AX, f"{w} 쪽으로 걸어오는 지영이 보인다. 지영과 영민이 이야기를 나눈다.", 1)
        add(pg, "turn", _TX, f"중호\t야, {w}에 사팔팔오 너지?", 1)
        add(pg, "dots", _TX, "영민\t.......")
        add(pg, "paren_only", _TX, "미진\t(한숨)")
        add(pg, "turn", _TX, f"중호\t{w}에서 뭐 하는 거야?")
        add(pg, "pname", _TX, f"(영민)\t{w}에 차를 저렇게 대면 어떡해요?")
        add(pg, "turn", _TX, f"대장\t{w}는 내가 책임진다.")
        add(pg, "cont_name", _CX, f"검사 제끼고 영장도 없이 {w}까지 가는 거니까, 알겠냐?")
        add(pg, "turn", _TX, f"검사\t{w} 영장은 안 됩니다.")
        add(pg, "turn", _TX, f"미진\t오빠, {w}에 왔어요.")
        add(pg, "cont_paren", _CX, "(울며)\t정말 못하겠어요... 너무 무섭고...")
        add(pg, "action_name", _AX, f"중호 앞으로 {w} 진술서와 인주를 들이미는 최형사.", 1)
        if k == 0:
            add(pg, "pname_rare", _TX, "(지영)\t괜찮아요. 금방 나올 건데요, 뭐.", 1)
        if k == 5:
            add(pg, "numbered_talk", _AX, "9. 중호 : 야, 거기 서!", 1)      # 번호 붙은 대사 줄(씬 6 과 7 사이, 번호 9)
        pages.append(pg)
    return _pages(pages, pitch=0.019), at


def test_colon_scene_headings_fill_number_gaps():
    """쌍점이 든 번호 씬 머리('2. 영민의 거처 : 은행 앞 / 밤', 따옴표로 시작하는 '9. ‘…’ 시퀀스 : …')는 '이름 : 대사' 와
    모양이 같아도, 받아들인 씬 머리와 같은 칸에서 빈 번호를 차례대로 메우면 씬 머리다. 번호 붙은 대사 줄은 아니다."""
    rows, at = _chaser_rows()
    s = P.parse_rows("fm9200001", rows)
    assert [sc["number"] for sc in s["scenes"]] == [str(k + 1) for k in range(12)]
    for n in at["colon_head"] + at["head"]:
        assert el(s, n)["type"] == "scene", n
    assert s["scenes"][1]["place"] == "영민의 거처 : 은행 앞" and s["scenes"][1]["time"] == "밤"
    assert el(s, at["numbered_talk"][0])["type"] != "scene"
    assert any("쌍점이 든 번호 씬 머리 5개" in x for x in s["quality"]["notes"])
    # 씬 소속: 쌍점 씬 머리 다음 대사는 그 씬에 든다
    sc = {x["id"]: x["number"] for x in s["scenes"]}
    assert sc[el(s, at["colon_head"][0] + 2)["scene"]] == "2"
    assert _covered_once(s, rows)


def test_colon_heading_guard_keeps_numbered_dialogue():
    """번호 붙은 대사 줄이 많은 녹취(반칙왕 fm55450: '3. 태백산: 안 가세요?')는 번호가 씬 머리 사이에 들어맞아도 씬 머리가
    아니다(쌍점 앞이 화자로 나오는 이름이거나 뒤가 말)."""
    body = []
    for k in range(8):
        body += [(0.12, f"{2 * k + 1}. 체육관 {k}호 / 낮", 2), (0.146, "태백산: 놔봐."), (0.146, "대호 : 안 아퍼."),
                 (0.12, f"{2 * k + 2}. 태백산: 안 가세요?", 1), (0.146, "대호: 팔 꺾여서 안 아픈 사람이 어딨어?"),
                 (0.12, f"{2 * k + 2}. 관장실 : 어서 오세요?", 1)]
    s = P.parse_rows("fm9200002", _pages([body[i:i + 24] for i in range(0, len(body), 24)]))
    assert len(s["scenes"]) == 8 and all(int(sc["number"]) % 2 == 1 for sc in s["scenes"])


def test_paren_name_turns_are_offscreen_voice():
    """'(영민)⇥차를 저렇게 대면 어떡해요?': 괄호에 든 이름 칸은 그 인물의 화면 밖 목소리(how=off, speaker_raw 는 적힌 그대로).
    이름표에 없는 이름도 이 표기가 자리 잡은 문서에서 같은 칸에 놓이고 지문에 인물로 나오면 받는다('(지영)').
    이어짐 칸의 '(울며)⇥…' 같은 연기 지시 줄은 화자가 아니라 앞 대사의 이어짐이다."""
    rows, at = _chaser_rows()
    s = P.parse_rows("fm9200001", rows)
    for n in at["pname"]:
        e = el(s, n)
        assert e["type"] == "dialogue" and names(s, e) == ["영민"] and e["how"] == "off", n
        assert e["speaker_raw"] == "(영민)" and e["lines"] == [n] and e["text"].endswith("대면 어떡해요?")
    e = el(s, at["pname_rare"][0])
    assert names(s, e) == ["지영"] and e["how"] == "off" and e["speaker_raw"] == "(지영)"
    for n in at["cont_paren"]:
        e = el(s, n)
        assert names(s, e) == ["미진"] and e["lines"] == [n - 1, n] and "울며" in [p["text"] for p in e["paren"]]
    assert all(p["name"] not in ("울며", "(영민)") for p in s["people"])
    assert any("괄호에 든 이름 (이름)⇥대사 13줄" in x for x in s["quality"]["notes"])


def test_paren_direction_lines_are_not_names():
    """'(웃으며)⇥대사' 줄만 있는 문서: 괄호 안이 인물이 아니므로 대사 시작이 아니다."""
    body = []
    for k, w in enumerate(_PLACES[:8]):
        body += [(_AX, f"{k + 1}.\t{w} / 밤", 2), (_TX, f"중호\t{w}에 있어?"), (_TX, f"(웃으며)\t{w}엔 없어요."),
                 (_TX, f"미진\t{w}로 갈게요."), (_TX, "(당황)\t아니, 오지 마."), (_AX, f"{w}의 불이 꺼진다.", 1)]
    s = P.parse_rows("fm9200003", _pages([body[i:i + 24] for i in range(0, len(body), 24)]))
    assert {p["name"] for p in s["people"]} == {"중호", "미진"}
    assert all(e["how"] == "on" for e in s["elements"] if e["type"] == "dialogue")


def test_wordless_turns_keep_name_in_speaker_raw():
    """말 없는 차례 '영민⇥.......', '미진⇥(한숨)': 다음 줄이 들여쓴 줄이어도 한 줄 표기 문서에서는 이름 줄이 아니다.
    speaker_raw 는 이름만, 점은 text, 괄호는 paren 으로 가고, 다음 차례를 삼키지 않는다. 화자 이어받기도 켜지지 않는다."""
    rows, at = _chaser_rows()
    s = P.parse_rows("fm9200001", rows)
    for n in at["dots"]:
        e = el(s, n)
        assert e["speaker_raw"] == "영민" and e["text"] == "......." and e["paren"] == [] and e["lines"] == [n], n
        assert names(s, e) == ["영민"]
    for n in at["paren_only"]:
        e = el(s, n)
        assert e["speaker_raw"] == "미진" and e["text"] == "" and e["paren"] == [{"text": "한숨", "at": 0}], n
        assert e["lines"] == [n]
    q = s["quality"]
    assert q["empty_dialogue"] == 0 and q["inferred_speaker"] == 0 and q["speakerless_dialogue"] == 0
    assert all("speaker_inferred" not in e for e in s["elements"])
    assert any("말 없이 점·괄호만 있는 한 줄 차례 24줄" in x for x in q["notes"])


def test_wordless_name_lines_stay_in_name_line_documents():
    """이름 줄 문서('미주⇥(웃으며)' 아래 줄에 말)에서는 전처럼 이름 줄이다(말은 아래 줄, speaker_raw 는 줄 그대로)."""
    pages = []
    for k, w in enumerate(["교실", "복도", "운동장", "창고", "옥상", "매점"]):
        pages.append([(0.143, f"{k + 1}. {w} - 낮", 2), (0.143, f"{w}에 아이들이 모여 있다.", 1),
                      (0.47, "미주\t(웃으며)", 1), (0.28, f"니 {w}에 혼자 왔나?"),
                      (0.47, "인호", 1), (0.28, "알았다."), (0.47, "미주", 1), (0.28, f"{w}에서 기다리께."),
                      (0.47, "인호\t.......", 1), (0.28, "가자.")])
    rows = _pages(pages)
    s = P.parse_rows("fm9200004", rows)
    e = el(s, 2)
    assert e["speaker_raw"] == "미주 (웃으며)" and e["lines"] == [2, 3] and e["text"] == "니 교실에 혼자 왔나?"
    assert el(s, 8)["lines"] == [8, 9] and el(s, 8)["text"] == "가자." and names(s, el(s, 8)) == ["인호"]


def test_single_syllable_name_before_tab_is_not_letterspaced():
    """'홍⇥이 선생님은 …'(연애의 목적 fm55036): 한 글자 이름 뒤의 탭 + '이' 로 시작하는 말을 띄운 이름 '홍 이' 로 읽지 않는다
    (그 글자가 '홍⇥무슨 말씀…' 처럼 한 글자 이름으로 3번 이상 쓰이는 문서). 탭으로 띄운 진짜 이름 '영⇥신 (…) 대사' 는 그대로."""
    body = []
    for k, w in enumerate(_PLACES[:8]):
        body += [(_AX, f"{k + 1}.\t{w} / 밤", 2), (_AX, f"{w}에서 홍이 유림을 본다. 홍이 웃는다."),
                 (_TX, f"홍\t이 선생님은 {w}에 왜 오셨어요?"), (_TX, f"유림\t{w}가 좋아서요."), (_TX, f"홍\t이 냄새, {w} 냄새네."),
                 (_TX, f"홍\t무슨 말씀이세요, {w}에서?")]
    s = P.parse_rows("fm9200005", _pages([body[i:i + 24] for i in range(0, len(body), 24)]))
    assert {p["name"] for p in s["people"]} == {"홍", "유림"}
    assert el(s, 2)["speaker_raw"] == "홍" and el(s, 2)["text"].startswith("이 선생님은")
    assert el(s, 4)["text"].startswith("이 냄새")
    body = []
    for k, w in enumerate(_PLACES[:8]):
        body += [(_AX, f"S#\t{w} (밤)", 2), (_AX, f"{w}에 앉아 있는 영신의 모습. 문수가 영신을 본다."),
                 (_AX, f"영\t신 (돌아다보며) {w}에 왔어?"), (_AX, f"문 수 아냐. {w}은 지금 딱 좋은데."),
                 (_AX, f"영 신 (웃으며) {w}에서는 안 그래.")]
    s = P.parse_rows("fm9200006", _pages([body[i:i + 25] for i in range(0, len(body), 25)]))
    assert {p["name"] for p in s["people"]} == {"영신", "문수"}
    assert names(s, el(s, 2)) == ["영신"] and el(s, 2)["text"].endswith("에 왔어?")


def test_space_turn_starts_only_in_turn_column():
    """이름 칸이 뚜렷한 문서(강한 표기 줄의 95% 이상이 한 칸)에서는 지문 여백의 '중호 앞으로 … 최형사.' 나 이어짐 칸의
    '검사 제끼고 …' 처럼 인물 이름 + 공백으로 시작하는 줄이 대사 시작이 아니다(이름 뒤 공백 하나 표기는 이름 칸에서만)."""
    rows, at = _chaser_rows()
    s = P.parse_rows("fm9200001", rows)
    for n in at["action_name"]:
        assert el(s, n)["type"] == "action", n
    for n in at["cont_name"]:
        e = el(s, n)
        assert names(s, e) == ["대장"] and e["lines"] == [n - 1, n] and "검사 제끼고" in e["text"], n
    assert {p["name"] for p in s["people"]} == {"중호", "영민", "미진", "대장", "검사", "지영"}
    assert person(s, next(p["id"] for p in s["people"] if p["name"] == "중호"))["count"] == 24
    assert _covered_once(s, rows)


# ---------------------------------------------------------------- 0.3.1 (0.3.0 규칙의 가드와 회귀, 씬 머리의 장소·시간)

@pytest.mark.parametrize("rest,place,time", [
    ("초소 앞 / 오후 / 비", "초소 앞", "오후"),                                   # 시간 뒤에 덧붙은 날씨는 버린다
    ("망원동 골목길 : 오프닝 시퀀스 / 오후-밤 / 비-비그침", "망원동 골목길 : 오프닝 시퀀스", "오후-밤"),
    ("영민의 거처 : 정원 / 밤 / 비", "영민의 거처 : 정원", "밤"),
    ("시골 예식장 / 낮 / INT", "시골 예식장", "낮"),
    ("극장 안/ 밤/ 과거", "극장 안", "밤"),
    ("거리 / 낮 / 10년 후", "거리", "낮"),
    ("형태사무실 / 밤~낮", "형태사무실", "밤~낮"),
    ("도망남의 집 밖 / 짙은 새벽", "도망남의 집 밖", "짙은 새벽"),               # 꾸밈말 + 시간
    ("부산지검 앞/이른 오후", "부산지검 앞", "이른 오후"),
    ("도로 - 흐린 오후", "도로", "흐린 오후"),
    ("은행 (늦은 저녁)", "은행", "늦은 저녁"),
    ("여수댁 / 다음날 아침.", "여수댁", "다음날 아침"),                           # 시간 낱말이 이어진 것
    ("교무실(다음날 아침)", "교무실", "다음날 아침"),
    ("에필로그 / 아침-밤", "에필로그", "아침-밤"),
    ("권투도장 안 - 낮 / 오후 / 저녁.", "권투도장 안", "낮 / 오후 / 저녁"),
    ("고속도로 (밤-새벽)", "고속도로", "밤-새벽"),
    ("INT. 배두상의 차/ 오후(밤)", "배두상의 차", "오후(밤)"),
    ("기수대 외부 / 밤", "기수대 외부", "밤"),                                    # 장소 낱말에 붙은 내부·외부는 장소
    ("은혜원 내부. 낮.", "은혜원 내부", "낮"),
    ("경마장 내 / 낮", "경마장 내", "낮"),
    ("연립주택 외부 / N", "연립주택 외부", "N"),
    ("웨스틴 호텔 옥상 통로 - 실내/낮", "웨스틴 호텔 옥상 통로", "실내/낮"),      # 구분자로 떨어진 실내외는 전처럼 시간 쪽
    ("공주의 방 (내부, 밤)", "공주의 방", "내부, 밤"),
    ("진원의 방. 내부/밤", "진원의 방", "내부/밤"),
    ("국도 실외/저녁", "국도", "실외/저녁"),
    ("술집 / 외부 밤.", "술집", "외부 밤"),
    ("판문점 북한 밤", "판문점 북한", "밤"),                                      # 장소 낱말은 꾸밈말이 아니다
    ("BAR D / N", "BAR D", "N"),
    ("미진의 집 안 / 도로 (중호의 차 안) 교차 / 밤", "미진의 집 안 / 도로 (중호의 차 안) 교차", "밤"),
    ("집 / 밤 / 중호의 차 안에서 바라본 풍경", "집 / 밤 / 중호의 차 안에서 바라본 풍경", None),   # 뒤 조각이 길면 손대지 않는다
    ("BEDROOM - THE NEXT MORNING", "BEDROOM", "THE NEXT MORNING"),
    ("거실", "거실", None),
])
def test_split_place_time(rest, place, time):
    assert T.split_place_time(rest) == (place, time)


def test_scene_time_with_weather_tail_in_document():
    """추격자(fm55664) 꼴 씬 머리 '93.⇥성미산 / 밤 / 비', '119.⇥집 밖 / 짙은 새벽', '62.⇥기수대 외부 / 밤' 의 장소·시간.
    heading 은 적힌 그대로다."""
    tails = ["밤 / 비", "짙은 새벽", "밤", "오후 / 비", "밤 / 비", "아침", "밤 / 비", "낮"]
    body = []
    for k, w in enumerate(_PLACES[:8]):
        body += [(_AX, f"{k + 1}.\t{w}{' 외부' if k == 2 else ''} / {tails[k]}", 2), (_AX, f"{w}에 비가 내린다.", 1),
                 (_TX, f"중호\t{w}에 있어?", 1), (_TX, f"미진\t{w}로 갈게요.")]
    s = P.parse_rows("fm9210001", _pages([body[i:i + 16] for i in range(0, len(body), 16)], pitch=0.019))
    assert [(sc["place"], sc["time"]) for sc in s["scenes"]] == [
        ("망원역", "밤"), ("은행 앞", "짙은 새벽"), ("골목길 외부", "밤"), ("초소 앞", "오후"), ("먹자골목", "밤"),
        ("보도방", "아침"), ("모텔", "밤"), ("마트", "낮")]
    assert s["scenes"][0]["heading"] == "1. 망원역 / 밤 / 비" and s["scenes"][2]["heading"] == "3. 골목길 외부 / 밤"


def _voice_doc(extra: list[tuple], per_scene: list[tuple] | None = None, scenes: int = 8) -> tuple[dict, list[dict], int]:
    """이름 칸이 뚜렷한 한글 탭 대본(씬 scenes 개)에 extra 줄들을 첫 씬 끝에 더한 문서. (script, 줄, extra 첫 줄 번호).
    본문 줄이 200개를 넘어야 한 번 나온 이름이 이름표에서 빠진다(작은 문서는 모든 이름을 이름표에 넣는다): scenes=34."""
    pages = []
    first = -1
    n = 0
    for k in range(scenes):
        w = _PLACES[k % len(_PLACES)] + ("" if k < len(_PLACES) else f" {k // len(_PLACES) + 1}층")
        pg = [(_AX, f"{k + 1}.\t{w} / 밤", 2), (_AX, f"{w} 쪽으로 걸어오는 영새와 채린이 보인다. 부인이 운다.", 1),
              (_TX, f"영새\t{w}에 다 왔어.", 1), (_TX, f"채린\t{w}는 처음이에요."), (_TX, f"영새\t{w}에서 기다려."),
              (_TX, f"채린\t{w}에서요?"), (_TX, f"영새\t그래, {w}에서."), (_TX, f"채린\t알았어요, {w}.")]
        pg += [(x, t.replace("{w}", w), *g) for x, t, *g in (per_scene or [])]
        if k == 0:
            first = n + len(pg)
            pg += list(extra)
        n += len(pg)
        pages.append(pg)
    rows = _pages(pages, pitch=0.019)
    return P.parse_rows("fm9210002", rows), rows, first


def test_paren_name_with_voice_word_is_that_person():
    """'(영새소리)⇥불 끕니다.', '(채린 목소리)⇥…': 괄호 안이 이름 + 소리·목소리면 그 인물의 화면 밖 목소리다(댄서의 순정
    fm55087). 이름표에 없는 '(발자국소리)⇥…' 와 '(소리)⇥…' 는 전처럼 화자 없는 화면 밖 소리."""
    s, rows, a = _voice_doc([(_TX, "(영새소리)\t불 끕니다.", 1), (_TX, "(채린 목소리)\t아저씨, 어디 있어요?"),
                             (_TX, "(발자국소리)\t저벅, 저벅."), (_TX, "(소리)\t거기 누구야?")])
    e = el(s, a)
    assert names(s, e) == ["영새"] and e["how"] == "off" and e["speaker_raw"] == "(영새소리)" and e["text"] == "불 끕니다."
    e = el(s, a + 1)
    assert names(s, e) == ["채린"] and e["how"] == "off" and e["speaker_raw"] == "(채린 목소리)"
    for n in (a + 2, a + 3):
        e = el(s, n)
        assert e["type"] == "dialogue" and e["speakers"] == [] and e["how"] == "off" and e["lines"] == [n], n
    assert {p["name"] for p in s["people"]} == {"영새", "채린"}
    assert _covered_once(s, rows)


def test_paren_name_with_sound_description_is_action():
    """'(부인)⇥울음소리.', '(영새)⇥개를 어르는 소리.'(곡성 fm4412879): 괄호에 든 이름 줄의 오른쪽이 소리 묘사면 그 인물의
    말이 아니라 지문이고, 앞 대사에 이어 붙지도 않는다. '(영새)⇥그게 무슨 소리.' 는 말이다."""
    per = [(_TX, "(영새)\t{w}에 불 꺼.", 1), (_TX, "(채린)\t{w}엔 아무도 없어요.")]
    s, rows, a = _voice_doc([(_TX, "(채린)\t(비명을 지르며) 뭐예요!", 1), (_TX, "(채린)\t울음소리."),
                             (_TX, "(영새)\t개를 어르는 소리.", 1), (_TX, "(영새)\t그게 무슨 소리.", 1)], per)
    assert names(s, el(s, a)) == ["채린"] and el(s, a)["lines"] == [a]
    assert el(s, a + 1)["type"] == "action" and el(s, a + 2)["type"] == "action"
    e = el(s, a + 3)
    assert e["type"] == "dialogue" and names(s, e) == ["영새"] and e["text"] == "그게 무슨 소리."


def test_voice_marked_turns_outside_turn_column():
    """이름 칸 밖에 놓인 '이름 + 공백' 줄이라도 이름 바로 뒤가 들리는 방식 표시면 대사 시작이다: '채린(off) 따님이 …',
    '영새 (NAR) 항상 …', '채린 (소리)엄마!', '영새 나레이션'(다음 줄부터 말). 연기 지시 괄호 뒤가 말투일 때도
    ('영새 (다리를 주무르며) … 하는디...'). 인물 소개('영새 (유민 갱조직의 왕) 매서운 눈동자를 …')와 이름으로 시작하는
    지문은 지문이다. 이름표에 없는 이름의 탭 줄은 지문 칸에 놓였으면 대사가 아니지만, 꼬리표가 붙었으면 받는다."""
    s, rows, a = _voice_doc([
        (0.36, "채린(off) 따님이 참 예쁘네요.", 1),                                    # a
        (_CX, "영새 (NAR) 항상 웃을 준비만 하는 놈들.. 다 너희 때문이야.", 1),          # a+1
        (0.242, "채린 (소리)엄마!", 1),                                                # a+2
        (0.242, "영새 (다리를 주무르며) 몸에 기온이 빠져나가불기 전에 펴야 하는디...", 1),   # a+3
        (_AX, "영새 (유민 갱조직의 왕) 매서운 눈동자를 번뜩인다.", 1),                  # a+4
        (_AX, "채린 앞으로 진술서와 인주를 들이미는 형사.", 1),                         # a+5
        (_AX, "아나운서(V.O)\t표절 논란으로 물의를 빚었던 작가가 어제 새벽 체포됐습니다.", 1),   # a+6
        (_AX, "서부역 급식소\t실외 아침 풍경이 펼쳐진다.", 1),                          # a+7
        (0.19, "영새 나레이션", 1), (_CX, "난 채린이랑 같이 있어야 됩니다."),            # a+8, a+9
    ], scenes=34)
    assert names(s, el(s, a)) == ["채린"] and el(s, a)["how"] == "off"
    assert names(s, el(s, a + 1)) == ["영새"] and el(s, a + 1)["how"] == "narration"
    assert names(s, el(s, a + 2)) == ["채린"] and el(s, a + 2)["text"] == "엄마!"
    e = el(s, a + 3)
    assert names(s, e) == ["영새"] and e["text"].endswith("하는디...") and e["speaker_raw"] == "영새 (다리를 주무르며)"
    assert el(s, a + 4)["type"] == "action" and el(s, a + 5)["type"] == "action"
    e = el(s, a + 6)
    assert e["type"] == "dialogue" and e["speaker_raw"] == "아나운서(V.O)" and e["text"].startswith("표절 논란")
    assert el(s, a + 7)["type"] == "action"
    e = el(s, a + 8)                                # 꼬리표 낱말은 0.2.5 때처럼 text 앞에 남는다('나레이션 난 …')
    assert names(s, e) == ["영새"] and e["lines"] == [a + 8, a + 9] and e["text"].endswith("같이 있어야 됩니다.")
    assert {p["name"] for p in s["people"]} == {"영새", "채린", "아나운서"}
    assert _covered_once(s, rows)


_NX, _BX = 0.1429, 0.2774                       # 이름 칸(= 지문 여백 없음), 지문·이어짐이 함께 놓이는 칸


def _same_col_rows(tight_action: bool) -> tuple[list[dict], dict[str, list[int]]]:
    """지문과 이어지는 말이 같은 칸에 놓이는 한글 탭 대본(야수 fm55120 꼴). 지문 앞에는 빈 줄을 둔다(tight_action 이면
    대사 바로 아래 붙여 쓴 지문도 섞는다)."""
    pages, at = [], {}
    n = 0

    def add(pg: list, kind: str, x: float, text: str, gap: int = 0) -> None:
        nonlocal n
        pg.append((x, text, gap))
        at.setdefault(kind, []).append(n)
        n += 1

    for k, w in enumerate(_PLACES):
        pg: list = []
        add(pg, "head", _NX, f"{k + 1}. {w} / 밤", 2)
        add(pg, "action", _BX, f"그때, {w} 앞에 차가 멈춘다. 오진우가 서류를 넘긴다.", 1)
        add(pg, "turn", _NX, f"장도영\t{w} 영수증인데... 거기 날짜 보이죠?", 1)
        add(pg, "cont_name", _BX, "조영철 잠적한 담부터 정확히 사흘에 한번씩 들락거렸어요.")
        add(pg, "turn", _NX, f"오진우\t{w}에서 무슨 일이 있었던 거야.")
        add(pg, "cont", _BX, "이 안에서 대체 무슨 짓을 한 거냐고!")
        add(pg, "turn", _NX, f"조영철\t{w}는 모릅니다.")
        add(pg, "turn", _NX, "장도영\t그래서?")
        add(pg, "action", _BX, "장도영이 고개를 들어 창밖을 본다.", 0 if tight_action else 1)
        add(pg, "turn", _NX, f"조영철\t{w} 얘긴 그만하시죠.", 1)
        add(pg, "action", _BX, "조영철이 자리에서 일어난다.", 1)
        pages.append(pg)
    return _pages(pages, pitch=0.019), at


def test_tight_lines_below_dialogue_continue_it_when_action_follows_blank_line():
    """지문과 이어지는 말이 같은 칸에 놓이는 문서: 지문 앞에 늘 빈 줄을 두면, 대사 바로 아래 붙여 쓴 들여쓴 줄은 앞 대사의
    이어짐이다. 인물 이름으로 시작해도('조영철 잠적한 담부터 … 들락거렸어요.') 새 차례도 지문도 아니다. 빈 줄 뒤의 줄은
    지문으로 남는다. 대사 바로 아래 붙여 쓴 지문이 섞인 문서에서는 이 규칙을 켜지 않는다."""
    rows, at = _same_col_rows(tight_action=False)
    s = P.parse_rows("fm9210003", rows)
    for n in at["cont_name"]:
        e = el(s, n)
        assert e["type"] == "dialogue" and names(s, e) == ["장도영"] and e["lines"] == [n - 1, n], n
        assert e["text"].endswith("들락거렸어요.")
    for n in at["cont"]:
        e = el(s, n)
        assert names(s, e) == ["오진우"] and e["lines"] == [n - 1, n], n
    for n in at["action"]:
        assert el(s, n)["type"] == "action", n
    assert {p["name"] for p in s["people"]} == {"장도영", "오진우", "조영철"}
    assert any(x.startswith("대사 이어짐: 지문 앞에 빈 줄을 두는 문서") for x in s["quality"]["notes"])
    assert _covered_once(s, rows)
    rows, at = _same_col_rows(tight_action=True)
    s = P.parse_rows("fm9210004", rows)
    assert not any(x.startswith("대사 이어짐") for x in s["quality"]["notes"])
    assert all(el(s, n)["type"] == "action" for n in at["cont_name"])
    assert all(el(s, n)["type"] == "action" for n in at["action"])


@pytest.mark.parametrize("action_col_known", [True, False])
def test_speech_below_wordless_paren_turn_belongs_to_it(action_col_known):
    """'순임⇥(어찌할 줄 모르는 막막한 표정을 짓는다.)' 처럼 말 없이 괄호 지시만 있는 한 줄 차례 바로 아래 붙여 쓴 들여쓴
    줄은 그 차례의 말이다(박하사탕 fm54127, 잠복근무 fm55051). 괄호로 시작하는 줄(괄호 지문 덩어리, 텔 미 썸딩 fm54281)은
    잇지 않는다. 지문 칸을 아는 문서(씬 머리 다음 줄로 정함)와 모르는 문서 둘 다에서 본다."""
    lead = "그때, " if action_col_known else ""          # '그때,' 로 시작해야 씬 머리 다음 줄이 지문 칸 증거가 된다
    body = []
    for k, w in enumerate(_PLACES[:8]):
        body += [(_NX, f"{k + 1}. {w} / 밤", 2), (_BX, f"{lead}{w} 앞에 순임이 서 있다.", 1),
                 (_BX, f"순임의 어깨 위로 {w}의 눈이 내려앉는다."), (_BX, f"멀리서 {w}행 기차 소리가 들린다."),
                 (_NX, f"영호\t{w}엔 왜 왔어요?", 1), (_NX, f"순임\t{w}가 보고 싶어서요."), (_NX, f"영호\t{w}는 이제 없어요."),
                 (_NX, "순임\t(어찌할 줄 모르는 막막한 표정을 짓는다.)"),
                 (_BX, f"그래도…… 어떻게 안될까요? 잠깐만이라도…… {w}까지 오느라고"), (_BX, "얼마나 고생했다고요."),
                 (_NX, "영호\t(고개를 젓는다.)"), (_BX, f"(순임의 손목을 느닷없이 {w} 밖으로 확 밀치는 M."),
                 (_BX, f"뒤로 휘청 밀려나는 순임, {w} 바닥에 주저앉는다.)")]
    rows = _pages([body[i:i + 26] for i in range(0, len(body), 26)], pitch=0.019)
    dp = P.DocParser("fm9210005", rows)
    s = dp.parse()
    assert (dp.act_x is not None) == action_col_known and not dp.cont_cols and not dp.tight_cont
    for k in range(8):
        e = el(s, 13 * k + 7)
        assert names(s, e) == ["순임"] and e["lines"] == [13 * k + 7, 13 * k + 8, 13 * k + 9], k
        assert e["text"].startswith("그래도……") and e["text"].endswith("고생했다고요.")
        assert e["paren"][0]["text"].startswith("어찌할 줄 모르는")
        e = el(s, 13 * k + 10)
        assert names(s, e) == ["영호"] and e["lines"] == [13 * k + 10] and e["text"] == "", k
        assert el(s, 13 * k + 11)["type"] == "action" and el(s, 13 * k + 12)["type"] == "action"
    assert _covered_once(s, rows)


def test_action_at_page_top_does_not_continue_paren_name_turn():
    """괄호에 든 이름 차례('(쟝)⇥그래도 먹어보실 거라구요?', 앤티크 fm5691751·아저씨 fm765450)가 쪽 끝에 놓이고 다음 쪽이
    지문으로 시작하면, 그 지문은 이어지는 말이 아니다. 같은 쪽에서 이어짐 칸에 놓인 줄은 그대로 이어진다."""
    pages = []
    for k, w in enumerate(_PLACES[:10]):
        pg = [(_AX, f"{w} 유리창에 금이 간다. 쟝이 웃는다.", 0 if k else 2)] if k else []
        pg += [(_AX, f"{k + 1}.\t{w} / 밤", 2), (_AX, f"{w}에 케잌이 든 접시가 놓여 있다. 쟝이 태식을 본다.", 1),
               (_TX, f"태식\t{w}에 누구 있어?", 1), (_TX, f"쟝\t{w}엔 아무도 없어요."), (_TX, f"태식\t{w}에서 기다려."),
               (_TX, f"(쟝)\t{w}에 온 걸 환영합니다."), (_CX, "(웃으며) 천천히 둘러보세요."),
               (_TX, f"(쟝)\t그래도 {w} 케잌을 먹어보실 거라구요? 후후.", 1)]
        pages.append(pg)
    pages.append([(_AX, "순간! 케잌 위로 난데없이 꽂히는 칼날. 비명 소리가 엉킨다.")])
    rows = _pages(pages, pitch=0.019)
    s = P.parse_rows("fm9210006", rows)
    tops = [r["n"] for r in rows if r["n"] and rows[r["n"] - 1]["page"] != r["page"]]       # 쪽마다 첫 줄
    assert len(tops) == 10
    for n in tops:
        assert el(s, n)["type"] == "action", n
        e = el(s, n - 1)
        assert names(s, e) == ["쟝"] and e["how"] == "off" and e["lines"] == [n - 1], n
    cont = [r["n"] for r in rows if r["text"].startswith("(웃으며)")]
    assert all(el(s, n)["lines"] == [n - 1, n] and names(s, el(s, n)) == ["쟝"] for n in cont)
    assert _covered_once(s, rows)


def _numbered_doc(heads: list[tuple[float, str]], extra: dict[int, list[tuple]] | None = None) -> tuple[dict, list[dict]]:
    """번호 씬 머리 heads(x, 글)마다 지문 한 줄과 대사 넷이 따르는 한글 탭 대본. extra[k] 는 k 번째 씬 끝에 더하는 줄."""
    pages = []
    for k, (x, h) in enumerate(heads):
        w = _PLACES[k % len(_PLACES)]
        pg = [(x, h, 2), (_AX, f"그때, {w} 쪽으로 걸어오는 지영이 보인다.", 1), (_TX, f"중호\t야, {w}에 사팔팔오 너지?", 1),
              (_TX, f"영민\t{w}는 처음인데요."), (_TX, f"중호\t{w}에서 뭐 하는 거야?"), (_TX, f"영민\t{w} 구경요.")]
        pg += list((extra or {}).get(k, []))
        pages.append(pg)
    rows = _pages(pages, pitch=0.019)
    return P.parse_rows("fm9210007", rows), rows


def test_colon_heading_needs_order_column_and_series():
    """쌍점이 든 번호 줄을 씬 머리로 받는 조건: 번호가 앞뒤 씬 머리 사이를 차례대로 메울 것(앞 번호보다 1~3 큼), 받아들인
    씬 머리와 같은 칸일 것, 그 문서에 번호 씬 머리가 5개 이상일 것. 쌍점 없이 따옴표로 시작하는 번호 줄은 시간 꼬리가
    있어야 한다."""
    plain = [(_AX, f"{k + 1}.\t{w} / 밤") for k, w in enumerate(_PLACES)]

    def numbers(s: dict) -> list[str]:
        return [sc["number"] for sc in s["scenes"]]

    # 기준: 빈 번호 3 을 같은 칸의 쌍점 줄이 메운다
    heads = plain[:2] + [(_AX, "3.\t영민의 거처 : 창고 / 밤")] + plain[3:]
    s, rows = _numbered_doc(heads)
    assert numbers(s) == [str(k + 1) for k in range(12)] and s["scenes"][2]["place"] == "영민의 거처 : 창고"
    # 번호가 차례에 맞지 않는다(씬 6 과 7 사이의 40)
    s, rows = _numbered_doc(plain, {5: [(_AX, "40.\t영민의 거처 : 뒤편 / 밤", 2), (_AX, "그때, 문이 열린다.", 1)]})
    assert numbers(s) == [str(k + 1) for k in range(12)]
    assert el(s, next(r["n"] for r in rows if r["text"].startswith("40.")))["type"] != "scene"
    # 다른 칸에 놓였다(씬 머리 칸에서 0.03 들여쓴 '3. …')
    heads = plain[:2] + [(_AX + 0.03, "3.\t영민의 거처 : 창고 / 밤")] + plain[3:]
    s, rows = _numbered_doc(heads)
    assert numbers(s) == [str(k + 1) for k in range(12) if k != 2]
    # 번호 씬 머리가 4개뿐인 문서
    heads = plain[:4] + [(_AX, "5.\t영민의 거처 : 창고 / 밤")]
    s, rows = _numbered_doc(heads)
    assert numbers(s) == ["1", "2", "3", "4"]
    # 따옴표로 시작하는 번호 줄: 시간 꼬리가 없으면 씬 머리가 아니다
    heads = plain[:2] + [(_AX, "3.\t‘젓가락 행진곡’으로 넘어간다")] + plain[3:]
    s, rows = _numbered_doc(heads)
    assert numbers(s) == [str(k + 1) for k in range(12) if k != 2]
    heads = plain[:2] + [(_AX, "3.\t‘영민의 집’과 ‘지구대’ 교차 / 낮")] + plain[3:]
    s, rows = _numbered_doc(heads)
    assert numbers(s) == [str(k + 1) for k in range(12)] and s["scenes"][2]["time"] == "낮"


def test_shot_size_with_dots_is_not_dialogue():
    """'영희 B.S', '병권 C.U.' 처럼 점을 찍은 촬영 크기 표기도 대사가 아니라 화면 설명이다(인터뷰 fm54519)."""
    for t in ("B.S", "C.U.", "b.s , 돌아보는 영희", "E.C.U", "M.L.S."):
        assert P._RX_SHOT.match(t), t
    for t in ("B.S.A 총을 든다.", "C.U.T 하고 외친다", "Ms. Smith is waiting for you."):
        assert not P._RX_SHOT.match(t), t
    s, rows, a = _voice_doc([(_TX, "채린 B.S", 1), (_AX, "그때, 문이 열린다.", 1), (_TX, "영새 C.U.", 1),
                             (_AX, "그때, 불이 꺼진다.", 1)], scenes=34)
    assert el(s, a)["type"] == "action" and el(s, a + 2)["type"] == "action"
    assert {p["name"] for p in s["people"]} == {"영새", "채린"}


def test_paren_words_at_name_column_need_evidence():
    """괄호 이름 표기가 자리 잡은 문서(8줄 이상)의 이름 칸에 놓인 '(사이)⇥…', '(행인)⇥…': 이름표에 없고 다른 곳에서 화자로
    나오지도 지문에 인물로 나오지도 않는 낱말은 인물로 받지 않는다. 지문에 인물로 나오는 '(지영)' 은 받는다."""
    rows, at = _chaser_rows()
    base = len(rows)
    more = _pages([[(_AX, "13.\t교회 뒤 / 밤", 2), (_AX, "교회 뒤로 사람들이 지나간다.", 1), (_TX, "중호\t거기 누구야?", 1),
                    (_TX, "(사이)\t아무도 없나...", 1), (_TX, "(행인)\t저요? 지나가는 길인데요.", 1),
                    (_TX, "(지영)\t여기예요, 여기.", 1)]], pitch=0.019)
    for r in more:
        r["n"] += base
        r["page"] += rows[-1]["page"]
    rows = rows + more
    s = P.parse_rows("fm9200001", rows)
    assert {p["name"] for p in s["people"]} == {"중호", "영민", "미진", "대장", "검사", "지영"}
    assert el(s, base + 3)["type"] != "dialogue" or names(s, el(s, base + 3)) == ["중호"]       # 인물 '사이' 는 없다
    assert names(s, el(s, base + 5)) == ["지영"] and el(s, base + 5)["how"] == "off"
    assert _covered_once(s, rows)


def test_no_turn_column_when_strong_turns_are_spread():
    """이름 칸은 강한 표기 줄의 95% 이상이 한 칸에 놓일 때만 정한다. 열에 하나가 다른 칸에 놓인 문서에서는 이름 칸으로
    거르지 않으므로, 이름 + 공백으로 시작하는 줄을 전처럼(0.2.5) 읽는다."""
    def doc(spread: bool) -> tuple[P.DocParser, list[dict], list[int]]:
        pages, marks = [], []
        n = 0
        for k, w in enumerate(_PLACES):
            pg = [(_AX, f"{k + 1}.\t{w} / 밤", 2), (_AX, f"그때, {w} 쪽으로 걸어오는 지영이 보인다.", 1)]
            for j in range(5):
                x = 0.36 if spread and j == 4 and k % 2 == 0 else _TX        # 60줄 중 6줄(10%)이 다른 칸
                pg.append((x, f"{'중호' if j % 2 == 0 else '영민'}\t{w}에서 {j}번 문제는 어떻게 됐어?", 1 if j == 0 else 0))
            marks.append(n + len(pg))
            pg.append((_AX, f"중호 앞으로 {w} 진술서와 인주를 들이미는 최형사.", 1))
            n += len(pg)
            pages.append(pg)
        rows = _pages(pages, pitch=0.019)
        dp = P.DocParser("fm9210008", rows)
        dp.script = dp.parse()
        return dp, rows, marks

    dp, rows, marks = doc(spread=False)
    assert dp.turn_col is not None and all(el(dp.script, n)["type"] == "action" for n in marks)
    dp, rows, marks = doc(spread=True)
    assert dp.turn_col is None
    assert all(el(dp.script, n)["type"] == "dialogue" and names(dp.script, el(dp.script, n)) == ["중호"] for n in marks)


@pytest.mark.parametrize("name_lines,one_line", [(5, True), (7, False)])
def test_wordless_turns_follow_the_dominant_notation(name_lines, one_line):
    """말 없는 차례 '영민⇥.......' 를 한 줄 대사로 둘지 이름 줄로 볼지는 그 문서의 표기 비율로 정한다: 같은 줄에 말이 있는
    한 줄 표기가 이름 줄의 4배 이상(20줄 이상)이면 한 줄 대사, 아니면 전처럼 이름 줄(말은 아래 줄)."""
    pages, dots = [], []
    n = 0
    for k, w in enumerate(_PLACES[:8]):
        pg = [(_AX, f"{k + 1}.\t{w} / 밤", 2), (_AX, f"그때, {w} 쪽으로 걸어오는 지영이 보인다.", 1),
              (_TX, f"중호\t야, {w}에 사팔팔오 너지?", 1), (_TX, f"미진\t{w}는 처음인데요."), (_TX, f"중호\t{w}에서 뭐 하는 거야?")]
        dots.append(n + len(pg))
        pg += [(_TX, "영민\t.......", 1), (_CX, f"{w} 구경하러 왔는데요.")]
        if k < name_lines:
            pg += [(_TX, "지영", 1), (_CX, f"{w}에 아무도 없어요.")]
        n += len(pg)
        pages.append(pg)
    rows = _pages(pages, pitch=0.019)
    dp = P.DocParser("fm9210009", rows)
    s = dp.parse()
    assert any("말 없이 점·괄호만 있는 한 줄 차례" in x for x in s["quality"]["notes"]) == one_line
    for d in dots:
        e = el(s, d)
        if one_line:
            assert e["speaker_raw"] == "영민" and e["text"].startswith(".......")
        else:
            assert e["speaker_raw"] == "영민 ......." and e["lines"] == [d, d + 1] and e["text"].endswith("구경하러 왔는데요.")
        assert names(s, e) == ["영민"]
