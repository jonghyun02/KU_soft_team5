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
